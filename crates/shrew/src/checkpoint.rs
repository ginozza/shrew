// Checkpoint — Save and load model parameters
//
// Binary checkpoint format (.shrew):
//
//   Header:
//     magic:   [u8; 4]  = b"SHRW"
//     version: u32 LE   = 1
//     count:   u32 LE   = number of tensors
//
//   For each tensor:
//     key_len:  u32 LE
//     key:      [u8; key_len]  (UTF-8, format: "graph/param")
//     dtype:    u8             (0=F32, 1=F64, 2=U8, 3=U32, 4=I64)
//     ndim:     u32 LE
//     dims:     [u32 LE; ndim]
//     data_len: u64 LE         (in bytes)
//     data:     [u8; data_len] (raw little-endian typed data)
//
// Usage:
//   // Save
//   checkpoint::save("model.shrew", &executor)?;
//   checkpoint::save_tensors("weights.shrew", &named_tensors)?;
//
//   // Load
//   checkpoint::load("model.shrew", &mut executor)?;
//   let tensors = checkpoint::load_tensors::<CpuBackend>("weights.shrew", &device)?;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use shrew_core::backend::Backend;
use shrew_core::tensor::Tensor;
use shrew_core::DType;
use shrew_optim::OptimizerState;

use crate::exec::Executor;

// Constants

const MAGIC: &[u8; 4] = b"SHRW";
const VERSION: u32 = 1;

// DType <-> u8 encoding

fn dtype_to_u8(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::U8 => 2,
        DType::U32 => 3,
        DType::I64 => 4,
        DType::F16 => 5,
        DType::BF16 => 6,
    }
}

fn u8_to_dtype(v: u8) -> shrew_core::Result<DType> {
    match v {
        0 => Ok(DType::F32),
        1 => Ok(DType::F64),
        2 => Ok(DType::U8),
        3 => Ok(DType::U32),
        4 => Ok(DType::I64),
        5 => Ok(DType::F16),
        6 => Ok(DType::BF16),
        _ => Err(shrew_core::Error::msg(format!("Unknown dtype tag: {v}"))),
    }
}

// Raw bytes extraction from tensor (via f64 roundtrip for portability)

/// Convert a tensor to raw LE bytes, preserving the original dtype.
fn tensor_to_bytes<B: Backend>(tensor: &Tensor<B>) -> shrew_core::Result<Vec<u8>> {
    // Make contiguous first, then extract data as f64 and convert to native bytes
    let t = tensor.contiguous()?;
    let data = t.to_f64_vec()?;
    let dtype = t.dtype();

    Ok(match dtype {
        DType::F16 => data
            .iter()
            .flat_map(|&v| half::f16::from_f64(v).to_le_bytes())
            .collect(),
        DType::BF16 => data
            .iter()
            .flat_map(|&v| half::bf16::from_f64(v).to_le_bytes())
            .collect(),
        DType::F32 => data
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect(),
        DType::F64 => data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        DType::U8 => data.iter().map(|&v| v as u8).collect(),
        DType::U32 => data
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect(),
        DType::I64 => data
            .iter()
            .flat_map(|&v| (v as i64).to_le_bytes())
            .collect(),
    })
}

/// Reconstruct a tensor from raw LE bytes + metadata.
fn tensor_from_bytes<B: Backend>(
    bytes: &[u8],
    shape: Vec<usize>,
    dtype: DType,
    device: &B::Device,
) -> shrew_core::Result<Tensor<B>> {
    let data_f64: Vec<f64> = match dtype {
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect(),
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        DType::F64 => bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
        DType::U8 => bytes.iter().map(|&b| b as f64).collect(),
        DType::U32 => bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        DType::I64 => bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64)
            .collect(),
    };

    Tensor::<B>::from_f64_slice(&data_f64, shape, dtype, device)
}

// Low-level IO helpers

fn write_u8(w: &mut impl Write, v: u8) -> std::io::Result<()> {
    w.write_all(&[v])
}

fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_bytes(w: &mut impl Write, data: &[u8]) -> std::io::Result<()> {
    w.write_all(data)
}

fn read_u8(r: &mut impl Read) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_bytes(r: &mut impl Read, len: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// Write checkpoint

/// Write a set of named tensors to a writer in the Shrew checkpoint format.
pub fn write_checkpoint<B: Backend>(
    writer: &mut impl Write,
    tensors: &[(String, Tensor<B>)],
) -> shrew_core::Result<()> {
    // Header
    write_bytes(writer, MAGIC).map_err(io_err)?;
    write_u32(writer, VERSION).map_err(io_err)?;
    write_u32(writer, tensors.len() as u32).map_err(io_err)?;

    // Each tensor
    for (key, tensor) in tensors {
        let key_bytes = key.as_bytes();
        write_u32(writer, key_bytes.len() as u32).map_err(io_err)?;
        write_bytes(writer, key_bytes).map_err(io_err)?;

        write_u8(writer, dtype_to_u8(tensor.dtype())).map_err(io_err)?;

        let dims = tensor.dims();
        write_u32(writer, dims.len() as u32).map_err(io_err)?;
        for &d in dims {
            write_u32(writer, d as u32).map_err(io_err)?;
        }

        let data = tensor_to_bytes(tensor)?;
        write_u64(writer, data.len() as u64).map_err(io_err)?;
        write_bytes(writer, &data).map_err(io_err)?;
    }

    Ok(())
}

/// Read named tensors from a reader in the Shrew checkpoint format.
pub fn read_checkpoint<B: Backend>(
    reader: &mut impl Read,
    device: &B::Device,
) -> shrew_core::Result<Vec<(String, Tensor<B>)>> {
    // Header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(io_err)?;
    if &magic != MAGIC {
        return Err(shrew_core::Error::msg(format!(
            "Invalid checkpoint: expected magic {:?}, got {:?}",
            MAGIC, magic
        )));
    }

    let version = read_u32(reader).map_err(io_err)?;
    if version != VERSION {
        return Err(shrew_core::Error::msg(format!(
            "Unsupported checkpoint version: {} (expected {})",
            version, VERSION
        )));
    }

    let count = read_u32(reader).map_err(io_err)? as usize;
    let mut tensors = Vec::with_capacity(count);

    for _ in 0..count {
        let key_len = read_u32(reader).map_err(io_err)? as usize;
        let key_bytes = read_bytes(reader, key_len).map_err(io_err)?;
        let key = String::from_utf8(key_bytes)
            .map_err(|e| shrew_core::Error::msg(format!("Invalid UTF-8 key: {e}")))?;

        let dtype = u8_to_dtype(read_u8(reader).map_err(io_err)?)?;

        let ndim = read_u32(reader).map_err(io_err)? as usize;
        let mut dims = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            dims.push(read_u32(reader).map_err(io_err)? as usize);
        }

        let data_len = read_u64(reader).map_err(io_err)? as usize;
        let data = read_bytes(reader, data_len).map_err(io_err)?;

        let tensor = tensor_from_bytes::<B>(&data, dims, dtype, device)?;
        tensors.push((key, tensor));
    }

    Ok(tensors)
}

fn io_err(e: std::io::Error) -> shrew_core::Error {
    shrew_core::Error::msg(format!("IO error: {e}"))
}

// High-level API — save/load named tensors

/// Save a list of named tensors to a file.
///
/// ```rust,no_run
/// use shrew::checkpoint;
/// use shrew::prelude::*;
///
/// let w1 = Tensor::<CpuBackend>::zeros((2, 3), DType::F32, &CpuDevice).unwrap();
/// let b1 = Tensor::<CpuBackend>::zeros((2,), DType::F32, &CpuDevice).unwrap();
/// let tensors = vec![
///     ("w1".to_string(), w1),
///     ("b1".to_string(), b1),
/// ];
/// checkpoint::save_tensors("weights.shrew", &tensors).unwrap();
/// ```
pub fn save_tensors<B: Backend>(
    path: impl AsRef<Path>,
    tensors: &[(String, Tensor<B>)],
) -> shrew_core::Result<()> {
    let file = File::create(path.as_ref()).map_err(io_err)?;
    let mut writer = BufWriter::new(file);
    write_checkpoint(&mut writer, tensors)?;
    writer.flush().map_err(io_err)?;
    Ok(())
}

/// Load named tensors from a file.
///
/// ```rust,no_run
/// use shrew::checkpoint;
/// use shrew::prelude::*;
///
/// let tensors = checkpoint::load_tensors::<CpuBackend>("weights.shrew", &CpuDevice).unwrap();
/// for (name, tensor) in &tensors {
///     println!("{name}: {:?}", tensor.dims());
/// }
/// ```
pub fn load_tensors<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> shrew_core::Result<Vec<(String, Tensor<B>)>> {
    let file = File::open(path.as_ref()).map_err(io_err)?;
    let mut reader = BufReader::new(file);
    read_checkpoint(&mut reader, device)
}

// High-level API — save/load Executor parameters

/// Save all parameters from an Executor to a checkpoint file.
///
/// Parameters are stored with keys in the format `"graph_name/param_name"`.
pub fn save<B: Backend>(path: impl AsRef<Path>, executor: &Executor<B>) -> shrew_core::Result<()> {
    let named = executor.named_params();
    save_tensors(path, &named)
}

/// Load parameters from a checkpoint file into an Executor.
///
/// Only parameters present in the checkpoint will be updated.
/// Parameters not found in the file keep their current values.
///
/// Returns the number of parameters loaded.
pub fn load<B: Backend>(
    path: impl AsRef<Path>,
    executor: &mut Executor<B>,
) -> shrew_core::Result<usize> {
    let tensors = load_tensors::<B>(path, executor.device())?;
    let loaded: HashMap<String, Tensor<B>> = tensors.into_iter().collect();

    let mut count = 0;
    for (key, tensor) in &loaded {
        if executor.set_param_by_key(key, tensor.clone()) {
            count += 1;
        }
    }

    Ok(count)
}

/// Save all parameters from a Trainer to a checkpoint file.
pub fn save_trainer<B: Backend>(
    path: impl AsRef<Path>,
    trainer: &crate::exec::Trainer<B>,
) -> shrew_core::Result<()> {
    save(path, &trainer.executor)
}

/// Load parameters from a checkpoint file into a Trainer.
///
/// Returns the number of parameters loaded.
pub fn load_trainer<B: Backend>(
    path: impl AsRef<Path>,
    trainer: &mut crate::exec::Trainer<B>,
) -> shrew_core::Result<usize> {
    load(path, &mut trainer.executor)
}

// In-memory checkpoint (for testing and transfer)

/// Serialize named tensors to an in-memory byte vector.
pub fn to_bytes<B: Backend>(tensors: &[(String, Tensor<B>)]) -> shrew_core::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_checkpoint(&mut buf, tensors)?;
    Ok(buf)
}

/// Deserialize named tensors from an in-memory byte slice.
pub fn from_bytes<B: Backend>(
    data: &[u8],
    device: &B::Device,
) -> shrew_core::Result<Vec<(String, Tensor<B>)>> {
    let mut cursor = std::io::Cursor::new(data);
    read_checkpoint(&mut cursor, device)
}

// Training Checkpoint — Full training state (model + optimizer + metadata)
//
// Binary training checkpoint format (.shrew v2):
//
//   Header:
//     magic:   [u8; 4]  = b"SHRW"
//     version: u32 LE   = 2
//
//   Section 1: Model Parameters
//     tag: u8 = 0x01
//     count: u32 LE
//     [tensors...]  (same format as v1)
//
//   Section 2: Optimizer State
//     tag: u8 = 0x02
//     type_len: u32 LE
//     type_name: [u8; type_len]       (UTF-8, e.g. "Adam")
//     n_scalars: u32 LE
//     [key_len: u32, key: [u8], value: f64] × n_scalars
//     n_buffers: u32 LE
//     [key_len: u32, key: [u8], buf_len: u64, [f64 LE] × buf_len] × n_buffers
//
//   Section 3: Metadata
//     tag: u8 = 0x03
//     epoch: u64 LE
//     global_step: u64 LE
//     best_loss: f64 LE
//     n_loss_history: u32 LE
//     [f64 LE] × n_loss_history
//
//   EOF marker: u8 = 0xFF

const TRAINING_VERSION: u32 = 2;
const TAG_MODEL: u8 = 0x01;
const TAG_OPTIMIZER: u8 = 0x02;
const TAG_METADATA: u8 = 0x03;
const TAG_EOF: u8 = 0xFF;

/// Complete training checkpoint: model weights + optimizer state + training metadata.
///
/// Enables full training resume — not just the model parameters, but all the
/// internal state that would be lost if training were interrupted.
#[derive(Debug, Clone)]
pub struct TrainingCheckpoint<B: Backend> {
    /// Named model parameters (same as standard checkpoint).
    pub model_params: Vec<(String, Tensor<B>)>,
    /// Optimizer internal state (momentum buffers, step counters, etc.)
    pub optimizer_state: Option<OptimizerState>,
    /// Current training epoch (0-indexed).
    pub epoch: u64,
    /// Global optimization step counter.
    pub global_step: u64,
    /// Best loss value seen during training.
    pub best_loss: f64,
    /// Per-epoch loss history.
    pub loss_history: Vec<f64>,
}

impl<B: Backend> Default for TrainingCheckpoint<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> TrainingCheckpoint<B> {
    /// Create an empty checkpoint (no model params).
    pub fn new() -> Self {
        TrainingCheckpoint {
            model_params: Vec::new(),
            optimizer_state: None,
            epoch: 0,
            global_step: 0,
            best_loss: f64::INFINITY,
            loss_history: Vec::new(),
        }
    }

    /// Create a checkpoint from an executor (model params only).
    pub fn from_executor(executor: &Executor<B>) -> Self {
        TrainingCheckpoint {
            model_params: executor.named_params(),
            optimizer_state: None,
            epoch: 0,
            global_step: 0,
            best_loss: f64::INFINITY,
            loss_history: Vec::new(),
        }
    }

    /// Set the optimizer state in this checkpoint.
    pub fn with_optimizer_state(mut self, state: OptimizerState) -> Self {
        self.optimizer_state = Some(state);
        self
    }

    /// Set epoch counter.
    pub fn with_epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }

    /// Set global step counter.
    pub fn with_global_step(mut self, step: u64) -> Self {
        self.global_step = step;
        self
    }

    /// Set best loss.
    pub fn with_best_loss(mut self, loss: f64) -> Self {
        self.best_loss = loss;
        self
    }

    /// Set loss history.
    pub fn with_loss_history(mut self, history: Vec<f64>) -> Self {
        self.loss_history = history;
        self
    }
}

/// Write an OptimizerState section to the writer.
fn write_optimizer_state(w: &mut impl Write, state: &OptimizerState) -> std::io::Result<()> {
    write_u8(w, TAG_OPTIMIZER)?;

    // Optimizer type name
    let type_bytes = state.optimizer_type.as_bytes();
    write_u32(w, type_bytes.len() as u32)?;
    write_bytes(w, type_bytes)?;

    // Scalars
    let scalars: Vec<_> = state.scalars.iter().collect();
    write_u32(w, scalars.len() as u32)?;
    for (key, &value) in &scalars {
        let key_bytes = key.as_bytes();
        write_u32(w, key_bytes.len() as u32)?;
        write_bytes(w, key_bytes)?;
        write_bytes(w, &value.to_le_bytes())?;
    }

    // Buffers
    let buffers: Vec<_> = state.buffers.iter().collect();
    write_u32(w, buffers.len() as u32)?;
    for (key, data) in &buffers {
        let key_bytes = key.as_bytes();
        write_u32(w, key_bytes.len() as u32)?;
        write_bytes(w, key_bytes)?;
        write_u64(w, data.len() as u64)?;
        for &val in data.iter() {
            write_bytes(w, &val.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Read an OptimizerState section from the reader (tag already consumed).
fn read_optimizer_state(r: &mut impl Read) -> std::io::Result<OptimizerState> {
    // Type name
    let type_len = read_u32(r)? as usize;
    let type_bytes = read_bytes(r, type_len)?;
    let type_name = String::from_utf8(type_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut state = OptimizerState::new(type_name);

    // Scalars
    let n_scalars = read_u32(r)? as usize;
    for _ in 0..n_scalars {
        let key_len = read_u32(r)? as usize;
        let key_bytes = read_bytes(r, key_len)?;
        let key = String::from_utf8(key_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let value_bytes = read_bytes(r, 8)?;
        let value = f64::from_le_bytes([
            value_bytes[0],
            value_bytes[1],
            value_bytes[2],
            value_bytes[3],
            value_bytes[4],
            value_bytes[5],
            value_bytes[6],
            value_bytes[7],
        ]);
        state.set_scalar(key, value);
    }

    // Buffers
    let n_buffers = read_u32(r)? as usize;
    for _ in 0..n_buffers {
        let key_len = read_u32(r)? as usize;
        let key_bytes = read_bytes(r, key_len)?;
        let key = String::from_utf8(key_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let buf_len = read_u64(r)? as usize;
        let mut data = Vec::with_capacity(buf_len);
        for _ in 0..buf_len {
            let val_bytes = read_bytes(r, 8)?;
            data.push(f64::from_le_bytes([
                val_bytes[0],
                val_bytes[1],
                val_bytes[2],
                val_bytes[3],
                val_bytes[4],
                val_bytes[5],
                val_bytes[6],
                val_bytes[7],
            ]));
        }
        state.set_buffer(key, data);
    }

    Ok(state)
}

/// Write a full training checkpoint to a writer.
pub fn write_training_checkpoint<B: Backend>(
    writer: &mut impl Write,
    checkpoint: &TrainingCheckpoint<B>,
) -> shrew_core::Result<()> {
    // Header
    write_bytes(writer, MAGIC).map_err(io_err)?;
    write_u32(writer, TRAINING_VERSION).map_err(io_err)?;

    // Section 1: Model parameters
    write_u8(writer, TAG_MODEL).map_err(io_err)?;
    write_u32(writer, checkpoint.model_params.len() as u32).map_err(io_err)?;
    for (key, tensor) in &checkpoint.model_params {
        let key_bytes = key.as_bytes();
        write_u32(writer, key_bytes.len() as u32).map_err(io_err)?;
        write_bytes(writer, key_bytes).map_err(io_err)?;
        write_u8(writer, dtype_to_u8(tensor.dtype())).map_err(io_err)?;
        let dims = tensor.dims();
        write_u32(writer, dims.len() as u32).map_err(io_err)?;
        for &d in dims {
            write_u32(writer, d as u32).map_err(io_err)?;
        }
        let data = tensor_to_bytes(tensor)?;
        write_u64(writer, data.len() as u64).map_err(io_err)?;
        write_bytes(writer, &data).map_err(io_err)?;
    }

    // Section 2: Optimizer state (optional)
    if let Some(ref opt_state) = checkpoint.optimizer_state {
        write_optimizer_state(writer, opt_state).map_err(io_err)?;
    }

    // Section 3: Metadata
    write_u8(writer, TAG_METADATA).map_err(io_err)?;
    write_u64(writer, checkpoint.epoch).map_err(io_err)?;
    write_u64(writer, checkpoint.global_step).map_err(io_err)?;
    write_bytes(writer, &checkpoint.best_loss.to_le_bytes()).map_err(io_err)?;
    write_u32(writer, checkpoint.loss_history.len() as u32).map_err(io_err)?;
    for &loss in &checkpoint.loss_history {
        write_bytes(writer, &loss.to_le_bytes()).map_err(io_err)?;
    }

    // EOF
    write_u8(writer, TAG_EOF).map_err(io_err)?;

    Ok(())
}

/// Read a full training checkpoint from a reader.
pub fn read_training_checkpoint<B: Backend>(
    reader: &mut impl Read,
    device: &B::Device,
) -> shrew_core::Result<TrainingCheckpoint<B>> {
    // Header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(io_err)?;
    if &magic != MAGIC {
        return Err(shrew_core::Error::msg(format!(
            "Invalid checkpoint: expected magic {:?}, got {:?}",
            MAGIC, magic
        )));
    }

    let version = read_u32(reader).map_err(io_err)?;
    if version != TRAINING_VERSION {
        return Err(shrew_core::Error::msg(format!(
            "Unsupported training checkpoint version: {} (expected {})",
            version, TRAINING_VERSION
        )));
    }

    let mut model_params = Vec::new();
    let mut optimizer_state = None;
    let mut epoch = 0u64;
    let mut global_step = 0u64;
    let mut best_loss = f64::INFINITY;
    let mut loss_history = Vec::new();

    loop {
        let tag = read_u8(reader).map_err(io_err)?;

        match tag {
            TAG_MODEL => {
                let count = read_u32(reader).map_err(io_err)? as usize;
                for _ in 0..count {
                    let key_len = read_u32(reader).map_err(io_err)? as usize;
                    let key_bytes = read_bytes(reader, key_len).map_err(io_err)?;
                    let key = String::from_utf8(key_bytes)
                        .map_err(|e| shrew_core::Error::msg(format!("Invalid UTF-8 key: {e}")))?;

                    let dtype = u8_to_dtype(read_u8(reader).map_err(io_err)?)?;
                    let ndim = read_u32(reader).map_err(io_err)? as usize;
                    let mut dims = Vec::with_capacity(ndim);
                    for _ in 0..ndim {
                        dims.push(read_u32(reader).map_err(io_err)? as usize);
                    }
                    let data_len = read_u64(reader).map_err(io_err)? as usize;
                    let data = read_bytes(reader, data_len).map_err(io_err)?;
                    let tensor = tensor_from_bytes::<B>(&data, dims, dtype, device)?;
                    model_params.push((key, tensor));
                }
            }
            TAG_OPTIMIZER => {
                optimizer_state =
                    Some(read_optimizer_state(reader).map_err(|e| {
                        shrew_core::Error::msg(format!("Optimizer state error: {e}"))
                    })?);
            }
            TAG_METADATA => {
                epoch = read_u64(reader).map_err(io_err)?;
                global_step = read_u64(reader).map_err(io_err)?;
                let bl_bytes = read_bytes(reader, 8).map_err(io_err)?;
                best_loss = f64::from_le_bytes([
                    bl_bytes[0],
                    bl_bytes[1],
                    bl_bytes[2],
                    bl_bytes[3],
                    bl_bytes[4],
                    bl_bytes[5],
                    bl_bytes[6],
                    bl_bytes[7],
                ]);
                let n_losses = read_u32(reader).map_err(io_err)? as usize;
                loss_history = Vec::with_capacity(n_losses);
                for _ in 0..n_losses {
                    let lb = read_bytes(reader, 8).map_err(io_err)?;
                    loss_history.push(f64::from_le_bytes([
                        lb[0], lb[1], lb[2], lb[3], lb[4], lb[5], lb[6], lb[7],
                    ]));
                }
            }
            TAG_EOF => break,
            other => {
                return Err(shrew_core::Error::msg(format!(
                    "Unknown section tag in training checkpoint: 0x{other:02X}"
                )));
            }
        }
    }

    Ok(TrainingCheckpoint {
        model_params,
        optimizer_state,
        epoch,
        global_step,
        best_loss,
        loss_history,
    })
}

// High-level API — save/load training checkpoints

/// Save a complete training checkpoint to a file.
///
/// This saves model parameters, optimizer state, epoch, and loss history,
/// enabling full training resume.
///
/// ```rust,no_run
/// use shrew::checkpoint::{self, TrainingCheckpoint};
/// use shrew::prelude::*;
///
/// // During training:
/// let ckpt = TrainingCheckpoint::<CpuBackend>::new()
///     .with_epoch(10)
///     .with_global_step(5000)
///     .with_best_loss(0.032)
///     .with_loss_history(vec![0.5, 0.2, 0.1, 0.05, 0.032]);
/// checkpoint::save_training("training.shrew", &ckpt).unwrap();
/// ```
pub fn save_training<B: Backend>(
    path: impl AsRef<Path>,
    checkpoint: &TrainingCheckpoint<B>,
) -> shrew_core::Result<()> {
    let file = File::create(path.as_ref()).map_err(io_err)?;
    let mut writer = BufWriter::new(file);
    write_training_checkpoint(&mut writer, checkpoint)?;
    writer.flush().map_err(io_err)?;
    Ok(())
}

/// Load a complete training checkpoint from a file.
///
/// ```rust,no_run
/// use shrew::checkpoint;
/// use shrew::prelude::*;
///
/// let ckpt = checkpoint::load_training::<CpuBackend>("training.shrew", &CpuDevice).unwrap();
/// println!("Resuming from epoch {}, step {}", ckpt.epoch, ckpt.global_step);
/// println!("Best loss so far: {}", ckpt.best_loss);
///
/// // Restore model params into executor...
/// // Restore optimizer state with optimizer.load_state_dict(ckpt.optimizer_state)...
/// ```
pub fn load_training<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> shrew_core::Result<TrainingCheckpoint<B>> {
    let file = File::open(path.as_ref()).map_err(io_err)?;
    let mut reader = BufReader::new(file);
    read_training_checkpoint(&mut reader, device)
}

/// Serialize a training checkpoint to an in-memory byte vector.
pub fn training_to_bytes<B: Backend>(
    checkpoint: &TrainingCheckpoint<B>,
) -> shrew_core::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_training_checkpoint(&mut buf, checkpoint)?;
    Ok(buf)
}

/// Deserialize a training checkpoint from an in-memory byte slice.
pub fn training_from_bytes<B: Backend>(
    data: &[u8],
    device: &B::Device,
) -> shrew_core::Result<TrainingCheckpoint<B>> {
    let mut cursor = std::io::Cursor::new(data);
    read_training_checkpoint(&mut cursor, device)
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use shrew_cpu::{CpuBackend, CpuDevice};

    type CpuTensor = Tensor<CpuBackend>;

    #[test]
    fn test_roundtrip_f32() {
        let dev = CpuDevice;
        let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev).unwrap();

        let tensors = vec![("w".to_string(), t.clone())];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "w");
        assert_eq!(loaded[0].1.dims(), &[2, 2]);
        assert_eq!(loaded[0].1.dtype(), DType::F32);

        let orig = t.to_f64_vec().unwrap();
        let restored = loaded[0].1.to_f64_vec().unwrap();
        assert_eq!(orig, restored);
    }

    #[test]
    fn test_roundtrip_f64() {
        let dev = CpuDevice;
        let vals = vec![std::f64::consts::PI, std::f64::consts::E, 0.0, -1.5];
        let t = CpuTensor::from_f64_slice(&vals, (4,), DType::F64, &dev).unwrap();

        let tensors = vec![("precision_test".to_string(), t.clone())];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        let orig = t.to_f64_vec().unwrap();
        let restored = loaded[0].1.to_f64_vec().unwrap();
        // F64 should be bit-exact
        assert_eq!(orig, restored);
    }

    #[test]
    fn test_roundtrip_u8() {
        let dev = CpuDevice;
        let t = CpuTensor::from_f64_slice(&[0.0, 128.0, 255.0], (3,), DType::U8, &dev).unwrap();

        let tensors = vec![("pixels".to_string(), t.clone())];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        assert_eq!(loaded[0].1.dtype(), DType::U8);
        let orig = t.to_f64_vec().unwrap();
        let restored = loaded[0].1.to_f64_vec().unwrap();
        assert_eq!(orig, restored);
    }

    #[test]
    fn test_roundtrip_multiple_tensors() {
        let dev = CpuDevice;
        let w =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F64, &dev)
                .unwrap();
        let b = CpuTensor::from_f64_slice(&[0.1, 0.2, 0.3], (1, 3), DType::F64, &dev).unwrap();

        let tensors = vec![
            ("Forward/w1".to_string(), w.clone()),
            ("Forward/b1".to_string(), b.clone()),
        ];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "Forward/w1");
        assert_eq!(loaded[0].1.dims(), &[2, 3]);
        assert_eq!(loaded[1].0, "Forward/b1");
        assert_eq!(loaded[1].1.dims(), &[1, 3]);
    }

    #[test]
    fn test_invalid_magic() {
        let data = b"BADXsomejunk";
        let result = from_bytes::<CpuBackend>(data, &CpuDevice);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid checkpoint"));
    }

    #[test]
    fn test_file_roundtrip() {
        let dev = CpuDevice;
        let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let tensors = vec![("test".to_string(), t.clone())];

        let path = std::env::temp_dir().join("shrew_test_checkpoint.shrew");
        save_tensors(&path, &tensors).unwrap();
        let loaded = load_tensors::<CpuBackend>(&path, &dev).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "test");
        let orig = t.to_f64_vec().unwrap();
        let restored = loaded[0].1.to_f64_vec().unwrap();
        assert_eq!(orig, restored);
    }

    #[test]
    fn test_empty_checkpoint() {
        let tensors: Vec<(String, CpuTensor)> = vec![];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &CpuDevice).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_3d_tensor_roundtrip() {
        let dev = CpuDevice;
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let t = CpuTensor::from_f64_slice(&data, (2, 3, 4), DType::F32, &dev).unwrap();

        let tensors = vec![("volume".to_string(), t.clone())];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        assert_eq!(loaded[0].1.dims(), &[2, 3, 4]);
        let orig = t.to_f64_vec().unwrap();
        let restored = loaded[0].1.to_f64_vec().unwrap();
        for (a, b) in orig.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_training_checkpoint_roundtrip() {
        use shrew_optim::OptimizerState;

        let dev = CpuDevice;
        let w = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev).unwrap();
        let b = CpuTensor::from_f64_slice(&[0.1, 0.2], (2,), DType::F32, &dev).unwrap();

        // Build optimizer state
        let mut opt_state = OptimizerState::new("Adam");
        opt_state.set_scalar("t", 100.0);
        opt_state.set_scalar("lr", 0.001);
        opt_state.set_scalar("beta1", 0.9);
        opt_state.set_buffer("m.0", vec![0.01, 0.02, 0.03, 0.04]);
        opt_state.set_buffer("v.0", vec![0.001, 0.002, 0.003, 0.004]);

        let ckpt = TrainingCheckpoint {
            model_params: vec![
                ("layer1/weight".to_string(), w),
                ("layer1/bias".to_string(), b),
            ],
            optimizer_state: Some(opt_state),
            epoch: 10,
            global_step: 5000,
            best_loss: 0.032,
            loss_history: vec![1.5, 0.8, 0.4, 0.1, 0.05, 0.032],
        };

        let bytes = training_to_bytes(&ckpt).unwrap();
        let loaded = training_from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        // Verify model params
        assert_eq!(loaded.model_params.len(), 2);
        assert_eq!(loaded.model_params[0].0, "layer1/weight");
        assert_eq!(loaded.model_params[0].1.dims(), &[2, 2]);
        assert_eq!(loaded.model_params[1].0, "layer1/bias");

        // Verify metadata
        assert_eq!(loaded.epoch, 10);
        assert_eq!(loaded.global_step, 5000);
        assert!((loaded.best_loss - 0.032).abs() < 1e-10);
        assert_eq!(loaded.loss_history.len(), 6);
        assert!((loaded.loss_history[0] - 1.5).abs() < 1e-10);
        assert!((loaded.loss_history[5] - 0.032).abs() < 1e-10);

        // Verify optimizer state
        let opt = loaded.optimizer_state.unwrap();
        assert_eq!(opt.optimizer_type, "Adam");
        assert_eq!(opt.get_scalar("t"), Some(100.0));
        assert_eq!(opt.get_scalar("lr"), Some(0.001));
        assert_eq!(opt.get_scalar("beta1"), Some(0.9));
        let m0 = opt.get_buffer("m.0").unwrap();
        assert_eq!(m0.len(), 4);
        assert!((m0[2] - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_training_checkpoint_no_optimizer() {
        let dev = CpuDevice;
        let t = CpuTensor::from_f64_slice(&[1.0, 2.0], (2,), DType::F64, &dev).unwrap();

        let ckpt = TrainingCheckpoint {
            model_params: vec![("w".to_string(), t)],
            optimizer_state: None,
            epoch: 5,
            global_step: 250,
            best_loss: 0.1,
            loss_history: vec![0.5, 0.3, 0.1],
        };

        let bytes = training_to_bytes(&ckpt).unwrap();
        let loaded = training_from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        assert_eq!(loaded.model_params.len(), 1);
        assert!(loaded.optimizer_state.is_none());
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.global_step, 250);
    }
}
