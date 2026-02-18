// ONNX — Import / Export for interoperability
//
// ONNX (Open Neural Network Exchange) is the industry standard for
// exchanging trained models between frameworks (PyTorch, TensorFlow,
// CoreML, TensorRT, etc.).
//
// This module provides:
//
//   - Export: convert a Shrew module's state_dict to ONNX format
//   - Import: load an ONNX model's weights into Shrew tensors
//
// ONNX files use Protocol Buffers encoding. We implement a minimal
// protobuf encoder/decoder (no external crate needed) that handles
// the subset of the ONNX spec we need: ModelProto, GraphProto,
// TensorProto, and NodeProto.
//
// SUPPORTED ONNX OPS (for graph export):
//   MatMul, Add, Relu, Sigmoid, Tanh, Softmax, Gemm, Reshape, Transpose,
//   Conv, BatchNormalization, Dropout, Concat, Flatten
//
// REFERENCE:
//   https://onnx.ai/onnx/repo-docs/IR.html
//   https://protobuf.dev/programming-guides/encoding/

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use shrew_nn::Module;

// ONNX constants

/// ONNX IR version (we target ONNX IR version 9 / opset 17).
const ONNX_IR_VERSION: i64 = 9;
/// Default opset version.
const ONNX_OPSET_VERSION: i64 = 17;
/// Magic bytes + version for ONNX protobuf.
const ONNX_DOMAIN: &str = "";

// ONNX TensorProto data types
/// See https://onnx.ai/onnx/repo-docs/IR.html#tensor-data-types
const ONNX_FLOAT: i32 = 1;
const ONNX_DOUBLE: i32 = 11;
const ONNX_FLOAT16: i32 = 10;
const ONNX_BFLOAT16: i32 = 16;
const ONNX_INT8: i32 = 3;
const ONNX_UINT8: i32 = 2;
const ONNX_INT32: i32 = 6;
const ONNX_INT64: i32 = 7;
const ONNX_UINT32: i32 = 12;

// Minimal protobuf encoder

/// A minimal protobuf wire-format encoder. Supports:
/// - Varint (field type 0)
/// - Length-delimited (field type 2: bytes, strings, nested messages)
/// - Fixed32/Fixed64 (field types 5 and 1)
struct PbEncoder {
    buf: Vec<u8>,
}

impl PbEncoder {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Write a varint.
    fn write_varint(&mut self, mut val: u64) {
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            if val == 0 {
                self.buf.push(byte);
                break;
            } else {
                self.buf.push(byte | 0x80);
            }
        }
    }

    /// Write a field tag (field_number << 3 | wire_type).
    fn write_tag(&mut self, field: u32, wire_type: u32) {
        self.write_varint(((field as u64) << 3) | wire_type as u64);
    }

    /// Write a varint field.
    fn write_varint_field(&mut self, field: u32, val: u64) {
        self.write_tag(field, 0);
        self.write_varint(val);
    }

    /// Write a signed varint field (zigzag encoding for negative values).
    fn write_sint64_field(&mut self, field: u32, val: i64) {
        self.write_varint_field(field, val as u64);
    }

    /// Write a length-delimited bytes field.
    fn write_bytes_field(&mut self, field: u32, data: &[u8]) {
        self.write_tag(field, 2);
        self.write_varint(data.len() as u64);
        self.buf.extend_from_slice(data);
    }

    /// Write a string field.
    fn write_string_field(&mut self, field: u32, val: &str) {
        self.write_bytes_field(field, val.as_bytes());
    }

    /// Write a nested message field.
    fn write_message_field(&mut self, field: u32, encoder: &PbEncoder) {
        self.write_bytes_field(field, &encoder.buf);
    }

    /// Write raw float data as bytes.
    #[allow(dead_code)]
    fn write_float_data(&mut self, field: u32, data: &[f32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.write_bytes_field(field, &bytes);
    }

    /// Write raw double data as bytes.
    #[allow(dead_code)]
    fn write_double_data(&mut self, field: u32, data: &[f64]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.write_bytes_field(field, &bytes);
    }
}

// Minimal protobuf decoder

/// A minimal protobuf wire-format decoder.
struct PbDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> PbDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            if self.pos >= self.data.len() {
                return Err(shrew_core::Error::msg("protobuf: unexpected end of data"));
            }
            let byte = self.data[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err(shrew_core::Error::msg("protobuf: varint too long"));
            }
        }
        Ok(result)
    }

    fn read_tag(&mut self) -> Result<(u32, u32)> {
        let val = self.read_varint()?;
        let field = (val >> 3) as u32;
        let wire_type = (val & 0x7) as u32;
        Ok((field, wire_type))
    }

    fn read_bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.read_varint()? as usize;
        if self.pos + len > self.data.len() {
            return Err(shrew_core::Error::msg("protobuf: bytes field exceeds data"));
        }
        let result = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(result)
    }

    fn read_string(&mut self) -> Result<String> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| shrew_core::Error::msg("protobuf: invalid UTF-8 string"))
    }

    fn skip_field(&mut self, wire_type: u32) -> Result<()> {
        match wire_type {
            0 => {
                self.read_varint()?;
            }
            1 => {
                self.pos += 8;
            } // fixed64
            2 => {
                self.read_bytes()?;
            }
            5 => {
                self.pos += 4;
            } // fixed32
            _ => {
                return Err(shrew_core::Error::msg(format!(
                    "protobuf: unsupported wire type {wire_type}"
                )))
            }
        }
        Ok(())
    }
}

// ONNX TensorProto

/// Represents an ONNX TensorProto (a named tensor with shape and data).
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// Tensor name.
    pub name: String,
    /// ONNX data type (ONNX_FLOAT, ONNX_DOUBLE, etc.).
    pub data_type: i32,
    /// Shape dimensions.
    pub dims: Vec<i64>,
    /// Raw float data (for FLOAT type).
    pub float_data: Vec<f32>,
    /// Raw double data (for DOUBLE type).
    pub double_data: Vec<f64>,
    /// Raw bytes (for packed formats like FLOAT16).
    pub raw_data: Vec<u8>,
}

impl OnnxTensor {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            data_type: ONNX_FLOAT,
            dims: Vec::new(),
            float_data: Vec::new(),
            double_data: Vec::new(),
            raw_data: Vec::new(),
        }
    }

    /// Convert to protobuf bytes.
    fn encode(&self) -> Vec<u8> {
        let mut enc = PbEncoder::new();
        // field 1: dims (repeated int64)
        for &d in &self.dims {
            enc.write_sint64_field(1, d);
        }
        // field 2: data_type (int32)
        enc.write_varint_field(2, self.data_type as u64);
        // field 8: name (string)
        if !self.name.is_empty() {
            enc.write_string_field(8, &self.name);
        }
        // field 4: float_data (packed repeated float — as raw_data for efficiency)
        if !self.float_data.is_empty() {
            // field 13: raw_data (bytes) — more efficient than repeated float
            let bytes: Vec<u8> = self
                .float_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            enc.write_bytes_field(13, &bytes);
        } else if !self.double_data.is_empty() {
            let bytes: Vec<u8> = self
                .double_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            enc.write_bytes_field(13, &bytes);
        } else if !self.raw_data.is_empty() {
            enc.write_bytes_field(13, &self.raw_data);
        }
        enc.into_bytes()
    }

    /// Decode from protobuf bytes.
    fn decode(data: &[u8]) -> Result<Self> {
        let mut dec = PbDecoder::new(data);
        let mut tensor = OnnxTensor::new("");
        while dec.remaining() > 0 {
            let (field, wire_type) = dec.read_tag()?;
            match (field, wire_type) {
                (1, 0) => {
                    // dims (varint)
                    let v = dec.read_varint()? as i64;
                    tensor.dims.push(v);
                }
                (1, 2) => {
                    // dims (packed)
                    let bytes = dec.read_bytes()?;
                    let mut sub = PbDecoder::new(bytes);
                    while sub.remaining() > 0 {
                        tensor.dims.push(sub.read_varint()? as i64);
                    }
                }
                (2, 0) => {
                    // data_type
                    tensor.data_type = dec.read_varint()? as i32;
                }
                (8, 2) => {
                    // name
                    tensor.name = dec.read_string()?;
                }
                (13, 2) => {
                    // raw_data
                    tensor.raw_data = dec.read_bytes()?.to_vec();
                }
                (4, 2) => {
                    // float_data (packed)
                    let bytes = dec.read_bytes()?;
                    for chunk in bytes.chunks_exact(4) {
                        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        tensor.float_data.push(val);
                    }
                }
                (4, 5) => {
                    // float_data (repeated fixed32)
                    let bytes = &dec.data[dec.pos..dec.pos + 4];
                    dec.pos += 4;
                    let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    tensor.float_data.push(val);
                }
                (5, 2) => {
                    // double_data (packed)
                    let bytes = dec.read_bytes()?;
                    for chunk in bytes.chunks_exact(8) {
                        let val = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        tensor.double_data.push(val);
                    }
                }
                _ => {
                    dec.skip_field(wire_type)?;
                }
            }
        }
        Ok(tensor)
    }

    /// Get the float data (converting from raw_data if needed).
    fn to_f64_vec(&self) -> Vec<f64> {
        if !self.double_data.is_empty() {
            return self.double_data.clone();
        }
        if !self.float_data.is_empty() {
            return self.float_data.iter().map(|&v| v as f64).collect();
        }
        if !self.raw_data.is_empty() {
            match self.data_type {
                ONNX_FLOAT => self
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                    .collect(),
                ONNX_DOUBLE => self
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
                ONNX_FLOAT16 => self
                    .raw_data
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        half::f16::from_bits(bits).to_f64()
                    })
                    .collect(),
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        }
    }
}

/// Convert a Shrew DType to ONNX data type integer.
fn dtype_to_onnx(dtype: DType) -> i32 {
    match dtype {
        DType::F32 => ONNX_FLOAT,
        DType::F64 => ONNX_DOUBLE,
        DType::F16 => ONNX_FLOAT16,
        DType::BF16 => ONNX_BFLOAT16,
        DType::U8 => ONNX_UINT8,
        DType::U32 => ONNX_UINT32,
        DType::I64 => ONNX_INT64,
    }
}

/// Convert an ONNX data type integer to Shrew DType.
fn onnx_to_dtype(onnx_type: i32) -> Result<DType> {
    match onnx_type {
        ONNX_FLOAT => Ok(DType::F32),
        ONNX_DOUBLE => Ok(DType::F64),
        ONNX_FLOAT16 => Ok(DType::F16),
        ONNX_BFLOAT16 => Ok(DType::BF16),
        ONNX_UINT8 => Ok(DType::U8),
        ONNX_UINT32 => Ok(DType::U32),
        ONNX_INT64 => Ok(DType::I64),
        ONNX_INT8 => Ok(DType::U8),   // map to u8
        ONNX_INT32 => Ok(DType::I64), // upcast
        _ => Err(shrew_core::Error::msg(format!(
            "unsupported ONNX data type: {onnx_type}"
        ))),
    }
}

// ONNX NodeProto (graph operation)

/// An ONNX graph node (operation).
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Operation type (e.g., "MatMul", "Relu", "Add").
    pub op_type: String,
    /// Node name (for debugging).
    pub name: String,
    /// String attributes (key → value).
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// An ONNX attribute value.
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

impl OnnxNode {
    fn encode(&self) -> Vec<u8> {
        let mut enc = PbEncoder::new();
        // field 1: inputs (repeated string)
        for input in &self.inputs {
            enc.write_string_field(1, input);
        }
        // field 2: outputs (repeated string)
        for output in &self.outputs {
            enc.write_string_field(2, output);
        }
        // field 3: name (string)
        if !self.name.is_empty() {
            enc.write_string_field(3, &self.name);
        }
        // field 4: op_type (string)
        enc.write_string_field(4, &self.op_type);
        // field 5: attributes (repeated AttributeProto)
        for (key, val) in &self.attributes {
            let attr = encode_attribute(key, val);
            enc.write_message_field(5, &attr);
        }
        enc.into_bytes()
    }
}

fn encode_attribute(name: &str, val: &OnnxAttribute) -> PbEncoder {
    let mut enc = PbEncoder::new();
    enc.write_string_field(1, name); // field 1: name
    match val {
        OnnxAttribute::Int(i) => {
            enc.write_varint_field(2, 2); // type = INT
            enc.write_sint64_field(3, *i); // field 3: i
        }
        OnnxAttribute::Float(f) => {
            enc.write_varint_field(2, 1); // type = FLOAT
                                          // field 4: f (float, fixed32)
            enc.write_tag(4, 5);
            enc.buf.extend_from_slice(&f.to_le_bytes());
        }
        OnnxAttribute::String(s) => {
            enc.write_varint_field(2, 3); // type = STRING
            enc.write_bytes_field(5, s.as_bytes()); // field 5: s
        }
        OnnxAttribute::Ints(ints) => {
            enc.write_varint_field(2, 7); // type = INTS
            for &i in ints {
                enc.write_sint64_field(8, i); // field 8: ints
            }
        }
        OnnxAttribute::Floats(floats) => {
            enc.write_varint_field(2, 6); // type = FLOATS
            for &f in floats {
                enc.write_tag(7, 5); // field 7: floats (fixed32)
                enc.buf.extend_from_slice(&f.to_le_bytes());
            }
        }
    }
    enc
}

// ONNX ModelProto — top-level export

/// An ONNX model with graph, metadata, and opset information.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    /// Model producer name.
    pub producer_name: String,
    /// Model producer version.
    pub producer_version: String,
    /// Graph name.
    pub graph_name: String,
    /// Graph nodes (operations).
    pub nodes: Vec<OnnxNode>,
    /// Initializer tensors (weights).
    pub initializers: Vec<OnnxTensor>,
    /// Graph inputs (names and shapes).
    pub inputs: Vec<(String, Vec<i64>, i32)>,
    /// Graph outputs (names and shapes).
    pub outputs: Vec<(String, Vec<i64>, i32)>,
}

impl OnnxModel {
    /// Create a new empty ONNX model.
    pub fn new(graph_name: &str) -> Self {
        Self {
            producer_name: "Shrew".to_string(),
            producer_version: "0.1.0".to_string(),
            graph_name: graph_name.to_string(),
            nodes: Vec::new(),
            initializers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Encode to ONNX protobuf binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Build GraphProto
        let mut graph = PbEncoder::new();

        // field 1: nodes (repeated NodeProto)
        for node in &self.nodes {
            let node_bytes = node.encode();
            graph.write_bytes_field(1, &node_bytes);
        }

        // field 2: name
        graph.write_string_field(2, &self.graph_name);

        // field 5: initializers (repeated TensorProto — the weights)
        for init in &self.initializers {
            let tensor_bytes = init.encode();
            graph.write_bytes_field(5, &tensor_bytes);
        }

        // field 11: inputs (repeated ValueInfoProto)
        for (name, dims, dtype) in &self.inputs {
            let vi = encode_value_info(name, dims, *dtype);
            graph.write_message_field(11, &vi);
        }

        // field 12: outputs (repeated ValueInfoProto)
        for (name, dims, dtype) in &self.outputs {
            let vi = encode_value_info(name, dims, *dtype);
            graph.write_message_field(12, &vi);
        }

        // Build ModelProto
        let mut model = PbEncoder::new();
        // field 1: ir_version (int64)
        model.write_varint_field(1, ONNX_IR_VERSION as u64);
        // field 2: producer_name
        model.write_string_field(2, &self.producer_name);
        // field 3: producer_version
        model.write_string_field(3, &self.producer_version);
        // field 7: graph (GraphProto)
        model.write_message_field(7, &graph);
        // field 8: opset_import (OperatorSetIdProto)
        let mut opset = PbEncoder::new();
        opset.write_string_field(1, ONNX_DOMAIN); // domain
        opset.write_varint_field(2, ONNX_OPSET_VERSION as u64); // version
        model.write_message_field(8, &opset);

        model.into_bytes()
    }

    /// Save ONNX model to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_bytes();
        fs::write(path.as_ref(), &bytes)
            .map_err(|e| shrew_core::Error::msg(format!("failed to write ONNX file: {e}")))
    }
}

/// Encode a ValueInfoProto (input/output description).
fn encode_value_info(name: &str, dims: &[i64], data_type: i32) -> PbEncoder {
    let mut vi = PbEncoder::new();
    vi.write_string_field(1, name); // field 1: name

    // field 2: type (TypeProto)
    let mut type_proto = PbEncoder::new();
    // field 1: tensor_type (Tensor_TypeProto)
    let mut tensor_type = PbEncoder::new();
    tensor_type.write_varint_field(1, data_type as u64); // elem_type
                                                         // field 2: shape (TensorShapeProto)
    let mut shape = PbEncoder::new();
    for &d in dims {
        let mut dim = PbEncoder::new();
        if d >= 0 {
            dim.write_sint64_field(1, d); // dim_value
        } else {
            dim.write_string_field(2, "dynamic"); // dim_param (symbolic)
        }
        shape.write_message_field(1, &dim);
    }
    tensor_type.write_message_field(2, &shape);
    type_proto.write_message_field(1, &tensor_type);
    vi.write_message_field(2, &type_proto);

    vi
}

// Export API

/// Export a module's weights as an ONNX model file.
///
/// This creates a "weight-only" ONNX model: the initializer tensors contain
/// the model's learned parameters, and the graph describes a simple
/// sequential pass from input to output.
///
/// # Arguments
/// - `path`: output file path (typically `.onnx`)
/// - `module`: the trained module to export
/// - `model_name`: name for the ONNX graph
/// - `input_shape`: shape of the model's input tensor
///
/// # Example
/// ```ignore
/// let model = Linear::new(784, 10, true, DType::F32, &dev)?;
/// export_weights("model.onnx", &model, "classifier", &[1, 784])?;
/// ```
pub fn export_weights<P, B, M>(
    path: P,
    module: &M,
    model_name: &str,
    input_shape: &[i64],
) -> Result<()>
where
    P: AsRef<Path>,
    B: Backend,
    M: Module<B>,
{
    let named = module.named_parameters();

    let mut model = OnnxModel::new(model_name);

    // Add input
    model
        .inputs
        .push(("input".to_string(), input_shape.to_vec(), ONNX_FLOAT));

    // Add each parameter as an initializer
    for (name, tensor) in &named {
        let data = tensor.to_f64_vec()?;
        let dims: Vec<i64> = tensor.dims().iter().map(|&d| d as i64).collect();

        let mut onnx_tensor = OnnxTensor::new(name);
        onnx_tensor.data_type = dtype_to_onnx(tensor.dtype());
        onnx_tensor.dims = dims;

        match tensor.dtype() {
            DType::F32 => {
                onnx_tensor.float_data = data.iter().map(|&v| v as f32).collect();
            }
            DType::F64 => {
                onnx_tensor.double_data = data;
            }
            _ => {
                // Store as F32 for compatibility
                onnx_tensor.data_type = ONNX_FLOAT;
                onnx_tensor.float_data = data.iter().map(|&v| v as f32).collect();
            }
        }

        model.initializers.push(onnx_tensor);
    }

    // Add a simple identity graph: input → output through the params
    // (The actual computation graph would require op-level tracking)
    model.outputs.push((
        "output".to_string(),
        vec![-1], // dynamic output shape
        ONNX_FLOAT,
    ));

    model.save(path)
}

/// Export named tensors directly to ONNX format.
///
/// Lower-level API: saves a set of named tensors as ONNX initializers.
pub fn export_tensors<P, B>(
    path: P,
    tensors: &[(String, Tensor<B>)],
    model_name: &str,
) -> Result<()>
where
    P: AsRef<Path>,
    B: Backend,
{
    let mut model = OnnxModel::new(model_name);

    for (name, tensor) in tensors {
        let data = tensor.to_f64_vec()?;
        let dims: Vec<i64> = tensor.dims().iter().map(|&d| d as i64).collect();

        let mut onnx_tensor = OnnxTensor::new(name);
        onnx_tensor.data_type = dtype_to_onnx(tensor.dtype());
        onnx_tensor.dims = dims;

        match tensor.dtype() {
            DType::F32 => {
                onnx_tensor.float_data = data.iter().map(|&v| v as f32).collect();
            }
            DType::F64 => {
                onnx_tensor.double_data = data;
            }
            _ => {
                onnx_tensor.data_type = ONNX_FLOAT;
                onnx_tensor.float_data = data.iter().map(|&v| v as f32).collect();
            }
        }

        model.initializers.push(onnx_tensor);
    }

    model.save(path)
}

// Import API

/// Load tensor weights from an ONNX model file.
///
/// Returns a map of tensor name → Tensor for all initializers found.
///
/// # Example
/// ```ignore
/// let weights = load_onnx_weights::<CpuBackend>("model.onnx", &CpuDevice)?;
/// for (name, tensor) in &weights {
///     println!("{}: {:?}", name, tensor.dims());
/// }
/// ```
pub fn load_onnx_weights<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<HashMap<String, Tensor<B>>> {
    let bytes = fs::read(path.as_ref())
        .map_err(|e| shrew_core::Error::msg(format!("failed to read ONNX file: {e}")))?;

    load_onnx_weights_from_bytes::<B>(&bytes, device)
}

/// Load tensor weights from ONNX bytes (in-memory).
pub fn load_onnx_weights_from_bytes<B: Backend>(
    data: &[u8],
    device: &B::Device,
) -> Result<HashMap<String, Tensor<B>>> {
    let mut dec = PbDecoder::new(data);
    let mut result = HashMap::new();

    // Parse ModelProto
    while dec.remaining() > 0 {
        let (field, wire_type) = dec.read_tag()?;
        match (field, wire_type) {
            (7, 2) => {
                // GraphProto
                let graph_bytes = dec.read_bytes()?;
                let tensors = parse_graph_initializers::<B>(graph_bytes, device)?;
                result.extend(tensors);
            }
            _ => {
                dec.skip_field(wire_type)?;
            }
        }
    }

    Ok(result)
}

/// Parse graph initializer tensors from a GraphProto.
fn parse_graph_initializers<B: Backend>(
    data: &[u8],
    device: &B::Device,
) -> Result<HashMap<String, Tensor<B>>> {
    let mut dec = PbDecoder::new(data);
    let mut result = HashMap::new();

    while dec.remaining() > 0 {
        let (field, wire_type) = dec.read_tag()?;
        match (field, wire_type) {
            (5, 2) => {
                // Initializer (TensorProto)
                let tensor_bytes = dec.read_bytes()?;
                let onnx_tensor = OnnxTensor::decode(tensor_bytes)?;

                if !onnx_tensor.name.is_empty() {
                    let dtype = onnx_to_dtype(onnx_tensor.data_type)?;
                    let shape: Vec<usize> = onnx_tensor.dims.iter().map(|&d| d as usize).collect();
                    let f64_data = onnx_tensor.to_f64_vec();

                    if !f64_data.is_empty() {
                        let tensor = Tensor::<B>::from_f64_slice(&f64_data, shape, dtype, device)?;
                        result.insert(onnx_tensor.name.clone(), tensor);
                    }
                }
            }
            _ => {
                dec.skip_field(wire_type)?;
            }
        }
    }

    Ok(result)
}

// Graph Import — Full ONNX graph parsing

/// A fully parsed ONNX graph: nodes + initializers + I/O metadata.
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// Computation nodes in topological order.
    pub nodes: Vec<OnnxNode>,
    /// Initializer tensors (weights / constants).
    pub initializer_protos: Vec<OnnxTensor>,
    /// Graph input names (including initializer names).
    pub input_names: Vec<String>,
    /// Graph output names.
    pub output_names: Vec<String>,
    /// Graph name.
    pub name: String,
}

/// Decode an OnnxNode from protobuf bytes.
fn decode_node(data: &[u8]) -> Result<OnnxNode> {
    let mut dec = PbDecoder::new(data);
    let mut node = OnnxNode {
        inputs: Vec::new(),
        outputs: Vec::new(),
        op_type: String::new(),
        name: String::new(),
        attributes: HashMap::new(),
    };
    while dec.remaining() > 0 {
        let (field, wire_type) = dec.read_tag()?;
        match (field, wire_type) {
            (1, 2) => node.inputs.push(dec.read_string()?),
            (2, 2) => node.outputs.push(dec.read_string()?),
            (3, 2) => node.name = dec.read_string()?,
            (4, 2) => node.op_type = dec.read_string()?,
            (5, 2) => {
                let attr_bytes = dec.read_bytes()?;
                let (key, val) = decode_attribute(attr_bytes)?;
                node.attributes.insert(key, val);
            }
            _ => dec.skip_field(wire_type)?,
        }
    }
    Ok(node)
}

/// Decode an OnnxAttribute from protobuf bytes.
fn decode_attribute(data: &[u8]) -> Result<(String, OnnxAttribute)> {
    let mut dec = PbDecoder::new(data);
    let mut name = String::new();
    let mut attr_type: u64 = 0;
    let mut int_val: i64 = 0;
    let mut float_val: f32 = 0.0;
    let mut string_val = Vec::new();
    let mut ints_val: Vec<i64> = Vec::new();
    let mut floats_val: Vec<f32> = Vec::new();
    while dec.remaining() > 0 {
        let (field, wire_type) = dec.read_tag()?;
        match (field, wire_type) {
            (1, 2) => name = dec.read_string()?,           // name
            (2, 0) => attr_type = dec.read_varint()?,      // type
            (3, 0) => int_val = dec.read_varint()? as i64, // i
            (4, 5) => {
                // f (fixed32)
                if dec.pos + 4 > dec.data.len() {
                    return Err(shrew_core::Error::msg("attribute: unexpected end"));
                }
                let b = &dec.data[dec.pos..dec.pos + 4];
                float_val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                dec.pos += 4;
            }
            (5, 2) => string_val = dec.read_bytes()?.to_vec(), // s
            (7, 5) => {
                // floats (repeated fixed32)
                if dec.pos + 4 > dec.data.len() {
                    return Err(shrew_core::Error::msg("attribute: unexpected end"));
                }
                let b = &dec.data[dec.pos..dec.pos + 4];
                floats_val.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                dec.pos += 4;
            }
            (7, 2) => {
                // floats (packed)
                let bytes = dec.read_bytes()?;
                for c in bytes.chunks_exact(4) {
                    floats_val.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            (8, 0) => ints_val.push(dec.read_varint()? as i64), // ints (repeated varint)
            (8, 2) => {
                // ints (packed)
                let bytes = dec.read_bytes()?;
                let mut sub = PbDecoder::new(bytes);
                while sub.remaining() > 0 {
                    ints_val.push(sub.read_varint()? as i64);
                }
            }
            _ => dec.skip_field(wire_type)?,
        }
    }
    let val = match attr_type {
        1 => OnnxAttribute::Float(float_val),
        2 => OnnxAttribute::Int(int_val),
        3 => OnnxAttribute::String(String::from_utf8(string_val).unwrap_or_default()),
        6 => OnnxAttribute::Floats(floats_val),
        7 => OnnxAttribute::Ints(ints_val),
        _ => OnnxAttribute::Int(int_val), // fallback
    };
    Ok((name, val))
}

/// Parse a full GraphProto: nodes, initializers, inputs, outputs.
fn parse_graph_proto(data: &[u8]) -> Result<OnnxGraph> {
    let mut dec = PbDecoder::new(data);
    let mut graph = OnnxGraph {
        nodes: Vec::new(),
        initializer_protos: Vec::new(),
        input_names: Vec::new(),
        output_names: Vec::new(),
        name: String::new(),
    };
    while dec.remaining() > 0 {
        let (field, wire_type) = dec.read_tag()?;
        match (field, wire_type) {
            (1, 2) => {
                let node_bytes = dec.read_bytes()?;
                graph.nodes.push(decode_node(node_bytes)?);
            }
            (2, 2) => graph.name = dec.read_string()?,
            (5, 2) => {
                let tensor_bytes = dec.read_bytes()?;
                graph
                    .initializer_protos
                    .push(OnnxTensor::decode(tensor_bytes)?);
            }
            (11, 2) => {
                // input ValueInfoProto — extract name (field 1)
                let vi_bytes = dec.read_bytes()?;
                let name = extract_value_info_name(vi_bytes)?;
                graph.input_names.push(name);
            }
            (12, 2) => {
                // output ValueInfoProto
                let vi_bytes = dec.read_bytes()?;
                let name = extract_value_info_name(vi_bytes)?;
                graph.output_names.push(name);
            }
            _ => dec.skip_field(wire_type)?,
        }
    }
    Ok(graph)
}

/// Extract just the name from a ValueInfoProto.
fn extract_value_info_name(data: &[u8]) -> Result<String> {
    let mut dec = PbDecoder::new(data);
    while dec.remaining() > 0 {
        let (field, wire_type) = dec.read_tag()?;
        if field == 1 && wire_type == 2 {
            return dec.read_string();
        }
        dec.skip_field(wire_type)?;
    }
    Ok(String::new())
}

/// Load a full ONNX graph (nodes + initializers) from a file.
pub fn load_onnx_graph(path: impl AsRef<Path>) -> Result<OnnxGraph> {
    let bytes = fs::read(path.as_ref())
        .map_err(|e| shrew_core::Error::msg(format!("failed to read ONNX file: {e}")))?;
    load_onnx_graph_from_bytes(&bytes)
}

/// Load a full ONNX graph from in-memory bytes.
pub fn load_onnx_graph_from_bytes(data: &[u8]) -> Result<OnnxGraph> {
    let mut dec = PbDecoder::new(data);
    while dec.remaining() > 0 {
        let (field, wire_type) = dec.read_tag()?;
        if field == 7 && wire_type == 2 {
            let graph_bytes = dec.read_bytes()?;
            return parse_graph_proto(graph_bytes);
        }
        dec.skip_field(wire_type)?;
    }
    Err(shrew_core::Error::msg("ONNX file contains no graph"))
}

// Graph Execution — Run an ONNX graph with Shrew tensors

/// Execute an ONNX graph on the given backend.
///
/// Takes a parsed `OnnxGraph` and a map of input tensors. Initializer tensors
/// from the graph are materialised on the given device. Each node is executed
/// in order (the ONNX spec requires nodes in topological order).
///
/// Returns a map of output-name → Tensor for all graph outputs.
///
/// # Supported ops
///
/// `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Gemm`, `Relu`, `Sigmoid`, `Tanh`,
/// `Softmax`, `LogSoftmax`, `Reshape`, `Transpose`, `Flatten`, `Squeeze`,
/// `Unsqueeze`, `Concat`, `Identity`, `Neg`, `Sqrt`, `Exp`, `Log`, `Abs`,
/// `Clip`, `ReduceMean`, `ReduceSum`, `ReduceMax`, `ReduceMin`, `Gather`,
/// `BatchNormalization`, `Dropout`, `Shape`, `Cast`, `Pow`.
///
/// Unsupported ops produce an error.
pub fn run_onnx_graph<B: Backend>(
    graph: &OnnxGraph,
    inputs: &HashMap<String, Tensor<B>>,
    device: &B::Device,
) -> Result<HashMap<String, Tensor<B>>> {
    let mut env: HashMap<String, Tensor<B>> = HashMap::new();

    // 1. Load initializers
    for init in &graph.initializer_protos {
        if init.name.is_empty() {
            continue;
        }
        let dtype = onnx_to_dtype(init.data_type)?;
        let shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
        let f64_data = init.to_f64_vec();
        if !f64_data.is_empty() {
            let tensor = Tensor::<B>::from_f64_slice(&f64_data, shape, dtype, device)?;
            env.insert(init.name.clone(), tensor);
        }
    }

    // 2. Insert user-provided inputs (overrides initializers if names clash)
    for (name, tensor) in inputs {
        env.insert(name.clone(), tensor.clone());
    }

    // 3. Execute nodes in order
    for node in &graph.nodes {
        execute_node(node, &mut env, device)?;
    }

    // 4. Collect outputs
    let mut outputs = HashMap::new();
    for name in &graph.output_names {
        if let Some(t) = env.get(name) {
            outputs.insert(name.clone(), t.clone());
        }
    }
    Ok(outputs)
}

/// Helper: get a tensor from the environment by name.
fn get_tensor<'a, B: Backend>(
    env: &'a HashMap<String, Tensor<B>>,
    name: &str,
) -> Result<&'a Tensor<B>> {
    env.get(name)
        .ok_or_else(|| shrew_core::Error::msg(format!("ONNX runtime: tensor '{name}' not found")))
}

/// Helper: get an integer attribute with a default.
fn attr_i(node: &OnnxNode, key: &str, default: i64) -> i64 {
    match node.attributes.get(key) {
        Some(OnnxAttribute::Int(v)) => *v,
        _ => default,
    }
}

/// Helper: get an integer-list attribute.
fn attr_ints(node: &OnnxNode, key: &str) -> Vec<i64> {
    match node.attributes.get(key) {
        Some(OnnxAttribute::Ints(v)) => v.clone(),
        _ => Vec::new(),
    }
}

/// Helper: get a float attribute with default.
fn attr_f(node: &OnnxNode, key: &str, default: f32) -> f32 {
    match node.attributes.get(key) {
        Some(OnnxAttribute::Float(v)) => *v,
        _ => default,
    }
}

/// Execute a single ONNX node, inserting results into the environment.
fn execute_node<B: Backend>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<B>>,
    device: &B::Device,
) -> Result<()> {
    match node.op_type.as_str() {
        // ── Element-wise binary ──────────────────────────────────────────
        "Add" => {
            let a = get_tensor(env, &node.inputs[0])?;
            let b = get_tensor(env, &node.inputs[1])?;
            let out = a.add(b)?;
            env.insert(node.outputs[0].clone(), out);
        }
        "Sub" => {
            let a = get_tensor(env, &node.inputs[0])?;
            let b = get_tensor(env, &node.inputs[1])?;
            let out = a.sub(b)?;
            env.insert(node.outputs[0].clone(), out);
        }
        "Mul" => {
            let a = get_tensor(env, &node.inputs[0])?;
            let b = get_tensor(env, &node.inputs[1])?;
            let out = a.mul(b)?;
            env.insert(node.outputs[0].clone(), out);
        }
        "Div" => {
            let a = get_tensor(env, &node.inputs[0])?;
            let b = get_tensor(env, &node.inputs[1])?;
            let out = a.div(b)?;
            env.insert(node.outputs[0].clone(), out);
        }
        "Pow" => {
            let a = get_tensor(env, &node.inputs[0])?;
            // ONNX Pow has two inputs; exponent is second input
            let b = get_tensor(env, &node.inputs[1])?;
            let exp_val = b.to_f64_vec()?;
            if exp_val.len() == 1 {
                let out = a.powf(exp_val[0])?;
                env.insert(node.outputs[0].clone(), out);
            } else {
                return Err(shrew_core::Error::msg(
                    "ONNX Pow: only scalar exponent supported",
                ));
            }
        }

        // ── MatMul / Gemm ────────────────────────────────────────────────
        "MatMul" => {
            let a = get_tensor(env, &node.inputs[0])?;
            let b = get_tensor(env, &node.inputs[1])?;
            let out = a.matmul(b)?;
            env.insert(node.outputs[0].clone(), out);
        }
        "Gemm" => {
            // Y = alpha * A' * B' + beta * C
            let alpha = attr_f(node, "alpha", 1.0) as f64;
            let beta = attr_f(node, "beta", 1.0) as f64;
            let trans_a = attr_i(node, "transA", 0) != 0;
            let trans_b = attr_i(node, "transB", 0) != 0;

            let mut a = get_tensor(env, &node.inputs[0])?.clone();
            let mut b = get_tensor(env, &node.inputs[1])?.clone();

            if trans_a {
                a = a.t()?;
            }
            if trans_b {
                b = b.t()?;
            }

            let mut out = a.matmul(&b)?;
            if (alpha - 1.0).abs() > 1e-7 {
                out = out.affine(alpha, 0.0)?;
            }
            if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                let c = get_tensor(env, &node.inputs[2])?;
                if (beta - 1.0).abs() > 1e-7 {
                    let bc = c.affine(beta, 0.0)?;
                    out = out.add(&bc)?;
                } else {
                    out = out.add(c)?;
                }
            }
            env.insert(node.outputs[0].clone(), out);
        }

        // ── Unary activations ────────────────────────────────────────────
        "Relu" => {
            let x = get_tensor(env, &node.inputs[0])?;
            env.insert(node.outputs[0].clone(), x.relu()?);
        }
        "Sigmoid" => {
            let x = get_tensor(env, &node.inputs[0])?;
            env.insert(node.outputs[0].clone(), x.sigmoid()?);
        }
        "Tanh" => {
            let x = get_tensor(env, &node.inputs[0])?;
            env.insert(node.outputs[0].clone(), x.tanh()?);
        }
        "Neg" => {
            let x = get_tensor(env, &node.inputs[0])?;
            env.insert(node.outputs[0].clone(), x.neg()?);
        }
        "Sqrt" => {
            let x = get_tensor(env, &node.inputs[0])?;
            env.insert(node.outputs[0].clone(), x.sqrt()?);
        }
        "Exp" => {
            let x = get_tensor(env, &node.inputs[0])?;
            env.insert(node.outputs[0].clone(), x.exp()?);
        }
        "Log" => {
            let x = get_tensor(env, &node.inputs[0])?;
            env.insert(node.outputs[0].clone(), x.log()?);
        }
        "Abs" => {
            let x = get_tensor(env, &node.inputs[0])?;
            env.insert(node.outputs[0].clone(), x.abs()?);
        }

        // ── Softmax / LogSoftmax ─────────────────────────────────────────
        "Softmax" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let axis = attr_i(node, "axis", -1);
            let dim = if axis < 0 {
                (x.rank() as i64 + axis) as usize
            } else {
                axis as usize
            };
            env.insert(node.outputs[0].clone(), x.softmax(dim)?);
        }
        "LogSoftmax" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let axis = attr_i(node, "axis", -1);
            let dim = if axis < 0 {
                (x.rank() as i64 + axis) as usize
            } else {
                axis as usize
            };
            env.insert(node.outputs[0].clone(), x.log_softmax(dim)?);
        }

        // ── Clip (clamp) ─────────────────────────────────────────────────
        "Clip" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let min_val = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                get_tensor(env, &node.inputs[1])?.to_f64_vec()?[0]
            } else {
                f64::NEG_INFINITY
            };
            let max_val = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                get_tensor(env, &node.inputs[2])?.to_f64_vec()?[0]
            } else {
                f64::INFINITY
            };
            env.insert(node.outputs[0].clone(), x.clamp(min_val, max_val)?);
        }

        // ── Shape manipulation ───────────────────────────────────────────
        "Reshape" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let shape_tensor = get_tensor(env, &node.inputs[1])?;
            let shape_vals = shape_tensor.to_f64_vec()?;

            // Resolve -1 dims
            let total = x.elem_count();
            let mut new_shape: Vec<usize> = shape_vals.iter().map(|&v| v as i64 as usize).collect();
            let neg_idx = new_shape.iter().position(|&s| s == usize::MAX); // -1 as usize wraps
            if let Some(idx) = neg_idx {
                let known: usize = new_shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != idx)
                    .map(|(_, &s)| s)
                    .product();
                if known > 0 {
                    new_shape[idx] = total / known;
                }
            }
            env.insert(node.outputs[0].clone(), x.reshape(new_shape)?);
        }
        "Transpose" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let perm = attr_ints(node, "perm");
            if perm.is_empty() {
                // Default: reverse all dims
                let rank = x.rank();
                let rev: Vec<usize> = (0..rank).rev().collect();
                env.insert(node.outputs[0].clone(), x.permute(&rev)?);
            } else {
                let perm_usize: Vec<usize> = perm.iter().map(|&p| p as usize).collect();
                env.insert(node.outputs[0].clone(), x.permute(&perm_usize)?);
            }
        }
        "Flatten" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let axis = attr_i(node, "axis", 1) as usize;
            env.insert(node.outputs[0].clone(), x.flatten(axis, x.rank() - 1)?);
        }
        "Squeeze" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let axes = attr_ints(node, "axes");
            if axes.is_empty() {
                env.insert(node.outputs[0].clone(), x.squeeze_all());
            } else {
                let mut result = x.clone();
                // Squeeze from highest axis to lowest to avoid index shifting
                let mut sorted_axes: Vec<usize> = axes.iter().map(|&a| a as usize).collect();
                sorted_axes.sort_unstable();
                sorted_axes.reverse();
                for ax in sorted_axes {
                    result = result.squeeze(ax)?;
                }
                env.insert(node.outputs[0].clone(), result);
            }
        }
        "Unsqueeze" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let axes = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                // ONNX opset >= 13: axes is a tensor input
                let axes_t = get_tensor(env, &node.inputs[1])?;
                axes_t
                    .to_f64_vec()?
                    .iter()
                    .map(|&v| v as i64)
                    .collect::<Vec<_>>()
            } else {
                attr_ints(node, "axes")
            };
            let mut result = x.clone();
            let mut sorted_axes: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (result.rank() as i64 + a + 1) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            sorted_axes.sort_unstable();
            for ax in sorted_axes {
                result = result.unsqueeze(ax)?;
            }
            env.insert(node.outputs[0].clone(), result);
        }
        "Concat" => {
            let axis = attr_i(node, "axis", 0) as usize;
            let tensors: Vec<Tensor<B>> = node
                .inputs
                .iter()
                .map(|n| get_tensor(env, n).cloned())
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<Tensor<B>> = tensors;
            let out = Tensor::<B>::cat(&refs, axis)?;
            env.insert(node.outputs[0].clone(), out);
        }

        // ── Reductions ───────────────────────────────────────────────────
        "ReduceSum" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let axes = attr_ints(node, "axes");
            let keepdims = attr_i(node, "keepdims", 1) != 0;
            let mut result = x.clone();
            if axes.is_empty() {
                result = result.sum_all()?;
            } else {
                let mut sorted: Vec<usize> = axes.iter().map(|&a| a as usize).collect();
                sorted.sort_unstable();
                sorted.reverse();
                for ax in sorted {
                    result = result.sum(ax, keepdims)?;
                }
            }
            env.insert(node.outputs[0].clone(), result);
        }
        "ReduceMean" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let axes = attr_ints(node, "axes");
            let keepdims = attr_i(node, "keepdims", 1) != 0;
            let mut result = x.clone();
            if axes.is_empty() {
                result = result.mean_all()?;
            } else {
                let mut sorted: Vec<usize> = axes.iter().map(|&a| a as usize).collect();
                sorted.sort_unstable();
                sorted.reverse();
                for ax in sorted {
                    result = result.mean(ax, keepdims)?;
                }
            }
            env.insert(node.outputs[0].clone(), result);
        }
        "ReduceMax" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let axes = attr_ints(node, "axes");
            let keepdims = attr_i(node, "keepdims", 1) != 0;
            let mut result = x.clone();
            let mut sorted: Vec<usize> = axes.iter().map(|&a| a as usize).collect();
            sorted.sort_unstable();
            sorted.reverse();
            for ax in sorted {
                result = result.max(ax, keepdims)?;
            }
            env.insert(node.outputs[0].clone(), result);
        }
        "ReduceMin" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let axes = attr_ints(node, "axes");
            let keepdims = attr_i(node, "keepdims", 1) != 0;
            let mut result = x.clone();
            let mut sorted: Vec<usize> = axes.iter().map(|&a| a as usize).collect();
            sorted.sort_unstable();
            sorted.reverse();
            for ax in sorted {
                result = result.min(ax, keepdims)?;
            }
            env.insert(node.outputs[0].clone(), result);
        }

        // ── Gather ───────────────────────────────────────────────────────
        "Gather" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let indices = get_tensor(env, &node.inputs[1])?;
            let axis = attr_i(node, "axis", 0) as usize;
            env.insert(node.outputs[0].clone(), x.gather(axis, indices)?);
        }

        // ── BatchNormalization ───────────────────────────────────────────
        "BatchNormalization" => {
            // inputs: X, scale, B, mean, var
            let x = get_tensor(env, &node.inputs[0])?;
            let scale = get_tensor(env, &node.inputs[1])?;
            let bias = get_tensor(env, &node.inputs[2])?;
            let mean = get_tensor(env, &node.inputs[3])?;
            let var = get_tensor(env, &node.inputs[4])?;
            let eps = attr_f(node, "epsilon", 1e-5) as f64;

            // y = scale * (x - mean) / sqrt(var + eps) + bias
            // Broadcast: mean/var/scale/bias are 1-D [C], x is [N, C, ...]
            let x_sub = x.sub(mean)?;
            let std_inv = var.affine(1.0, eps)?.sqrt()?.reciprocal()?;
            let normed = x_sub.mul(&std_inv)?;
            let scaled = normed.mul(scale)?;
            let out = scaled.add(bias)?;
            env.insert(node.outputs[0].clone(), out);
        }

        // ── Dropout (inference) ──────────────────────────────────────────
        "Dropout" => {
            // In inference mode, Dropout is identity
            let x = get_tensor(env, &node.inputs[0])?.clone();
            env.insert(node.outputs[0].clone(), x.clone());
            // Optional second output (mask) — insert copy
            if node.outputs.len() > 1 && !node.outputs[1].is_empty() {
                env.insert(node.outputs[1].clone(), x);
            }
        }

        // ── Identity ─────────────────────────────────────────────────────
        "Identity" => {
            let x = get_tensor(env, &node.inputs[0])?;
            env.insert(node.outputs[0].clone(), x.clone());
        }

        // ── Shape ────────────────────────────────────────────────────────
        "Shape" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let shape: Vec<f64> = x.dims().iter().map(|&d| d as f64).collect();
            let n = shape.len();
            let out = Tensor::<B>::from_f64_slice(&shape, vec![n], DType::I64, device)?;
            env.insert(node.outputs[0].clone(), out);
        }

        // ── Cast ─────────────────────────────────────────────────────────
        "Cast" => {
            let x = get_tensor(env, &node.inputs[0])?;
            let to = attr_i(node, "to", ONNX_FLOAT as i64) as i32;
            let target_dtype = onnx_to_dtype(to)?;
            env.insert(node.outputs[0].clone(), x.to_dtype(target_dtype)?);
        }

        // ── Constant ─────────────────────────────────────────────────────
        "Constant" => {
            // Try to get value from attributes
            if let Some(OnnxAttribute::Float(v)) = node.attributes.get("value_float") {
                let out = Tensor::<B>::from_f64_slice(&[*v as f64], vec![1], DType::F32, device)?;
                env.insert(node.outputs[0].clone(), out);
            } else if let Some(OnnxAttribute::Int(v)) = node.attributes.get("value_int") {
                let out = Tensor::<B>::from_f64_slice(&[*v as f64], vec![1], DType::I64, device)?;
                env.insert(node.outputs[0].clone(), out);
            } else if let Some(OnnxAttribute::Floats(v)) = node.attributes.get("value_floats") {
                let data: Vec<f64> = v.iter().map(|f| *f as f64).collect();
                let n = data.len();
                let out = Tensor::<B>::from_f64_slice(&data, vec![n], DType::F32, device)?;
                env.insert(node.outputs[0].clone(), out);
            } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("value_ints") {
                let data: Vec<f64> = v.iter().map(|i| *i as f64).collect();
                let n = data.len();
                let out = Tensor::<B>::from_f64_slice(&data, vec![n], DType::I64, device)?;
                env.insert(node.outputs[0].clone(), out);
            } else {
                return Err(shrew_core::Error::msg(format!(
                    "ONNX Constant: unsupported value attribute in node '{}'",
                    node.name
                )));
            }
        }

        other => {
            return Err(shrew_core::Error::msg(format!(
                "ONNX runtime: unsupported op '{other}' (node '{}')",
                node.name
            )));
        }
    }
    Ok(())
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use shrew_cpu::{CpuBackend, CpuDevice};

    type B = CpuBackend;
    type T = Tensor<B>;
    const DEV: CpuDevice = CpuDevice;

    #[test]
    fn test_protobuf_varint_roundtrip() {
        let mut enc = PbEncoder::new();
        enc.write_varint(0);
        enc.write_varint(1);
        enc.write_varint(127);
        enc.write_varint(128);
        enc.write_varint(300);
        enc.write_varint(16384);

        let mut dec = PbDecoder::new(&enc.buf);
        assert_eq!(dec.read_varint().unwrap(), 0);
        assert_eq!(dec.read_varint().unwrap(), 1);
        assert_eq!(dec.read_varint().unwrap(), 127);
        assert_eq!(dec.read_varint().unwrap(), 128);
        assert_eq!(dec.read_varint().unwrap(), 300);
        assert_eq!(dec.read_varint().unwrap(), 16384);
    }

    #[test]
    fn test_onnx_tensor_encode_decode() {
        let mut tensor = OnnxTensor::new("test_weight");
        tensor.data_type = ONNX_FLOAT;
        tensor.dims = vec![2, 3];
        tensor.float_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let encoded = tensor.encode();
        let decoded = OnnxTensor::decode(&encoded).unwrap();

        assert_eq!(decoded.name, "test_weight");
        assert_eq!(decoded.data_type, ONNX_FLOAT);
        assert_eq!(decoded.dims, vec![2, 3]);
        // Data stored in raw_data (packed float), so check via to_f64_vec
        let data = decoded.to_f64_vec();
        assert_eq!(data.len(), 6);
        for (a, b) in tensor.float_data.iter().zip(data.iter()) {
            assert!((*a as f64 - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_export_import_roundtrip() {
        let linear = shrew_nn::Linear::<B>::new(4, 3, true, DType::F32, &DEV).unwrap();

        // Export
        let path = std::env::temp_dir().join("shrew_test_onnx.onnx");
        export_weights(&path, &linear, "test_model", &[1, 4]).unwrap();

        // Import
        let weights = load_onnx_weights::<B>(&path, &DEV).unwrap();

        // Should have weight and bias
        assert_eq!(weights.len(), 2);

        // Verify weight shape
        let named = linear.named_parameters();
        for (name, original) in &named {
            let loaded = weights.get(name).expect(&format!("missing: {name}"));
            assert_eq!(original.dims(), loaded.dims());

            // Values should match
            let orig_data = original.to_f64_vec().unwrap();
            let load_data = loaded.to_f64_vec().unwrap();
            for (a, b) in orig_data.iter().zip(load_data.iter()) {
                assert!((a - b).abs() < 1e-5, "mismatch for {name}: {a} vs {b}");
            }
        }

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_export_tensors() {
        let t1 = T::randn(vec![2, 3], DType::F32, &DEV).unwrap();
        let t2 = T::ones(vec![5], DType::F32, &DEV).unwrap();

        let tensors = vec![
            ("weight".to_string(), t1.clone()),
            ("bias".to_string(), t2.clone()),
        ];

        let path = std::env::temp_dir().join("shrew_test_tensors.onnx");
        export_tensors(&path, &tensors, "tensors_model").unwrap();

        let loaded = load_onnx_weights::<B>(&path, &DEV).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("weight").unwrap().dims(), &[2, 3]);
        assert_eq!(loaded.get("bias").unwrap().dims(), &[5]);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_onnx_model_builder() {
        let mut model = OnnxModel::new("test_graph");
        model
            .inputs
            .push(("X".to_string(), vec![1, 784], ONNX_FLOAT));
        model
            .outputs
            .push(("Y".to_string(), vec![1, 10], ONNX_FLOAT));

        model.nodes.push(OnnxNode {
            inputs: vec!["X".to_string(), "weight".to_string()],
            outputs: vec!["matmul_out".to_string()],
            op_type: "MatMul".to_string(),
            name: "matmul_0".to_string(),
            attributes: HashMap::new(),
        });

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), OnnxAttribute::Int(1));
        model.nodes.push(OnnxNode {
            inputs: vec!["matmul_out".to_string()],
            outputs: vec!["Y".to_string()],
            op_type: "Softmax".to_string(),
            name: "softmax_0".to_string(),
            attributes: attrs,
        });

        let bytes = model.to_bytes();
        assert!(!bytes.is_empty());
        assert!(bytes.len() > 20); // non-trivial size
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(dtype_to_onnx(DType::F32), ONNX_FLOAT);
        assert_eq!(dtype_to_onnx(DType::F64), ONNX_DOUBLE);
        assert_eq!(dtype_to_onnx(DType::F16), ONNX_FLOAT16);
        assert_eq!(onnx_to_dtype(ONNX_FLOAT).unwrap(), DType::F32);
        assert_eq!(onnx_to_dtype(ONNX_DOUBLE).unwrap(), DType::F64);
        assert_eq!(onnx_to_dtype(ONNX_FLOAT16).unwrap(), DType::F16);
    }

    #[test]
    fn test_double_data_roundtrip() {
        let t = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], DType::F64, &DEV).unwrap();

        let tensors = vec![("w".to_string(), t.clone())];
        let path = std::env::temp_dir().join("shrew_test_f64.onnx");
        export_tensors(&path, &tensors, "f64_model").unwrap();

        let loaded = load_onnx_weights::<B>(&path, &DEV).unwrap();
        let w = loaded.get("w").unwrap();
        assert_eq!(w.dims(), &[2, 2]);

        let orig = t.to_f64_vec().unwrap();
        let load = w.to_f64_vec().unwrap();
        for (a, b) in orig.iter().zip(load.iter()) {
            assert!((a - b).abs() < 1e-10);
        }

        let _ = fs::remove_file(&path);
    }

    // ────────────────────────────────────────────────────────────────────
    //  Graph Import + Execution tests
    // ────────────────────────────────────────────────────────────────────

    /// Helper: build a minimal ONNX model bytes from an OnnxModel.
    fn build_and_reload_graph(model: &OnnxModel) -> OnnxGraph {
        let bytes = model.to_bytes();
        load_onnx_graph_from_bytes(&bytes).unwrap()
    }

    #[test]
    fn test_graph_add_two_inputs() {
        // Graph:  Y = A + B
        let mut model = OnnxModel::new("add_graph");
        model.inputs.push(("A".into(), vec![2, 2], ONNX_FLOAT));
        model.inputs.push(("B".into(), vec![2, 2], ONNX_FLOAT));
        model.outputs.push(("Y".into(), vec![2, 2], ONNX_FLOAT));
        model.nodes.push(OnnxNode {
            inputs: vec!["A".into(), "B".into()],
            outputs: vec!["Y".into()],
            op_type: "Add".into(),
            name: "add_0".into(),
            attributes: HashMap::new(),
        });

        let graph = build_and_reload_graph(&model);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "Add");
        assert_eq!(graph.output_names, vec!["Y"]);

        let a = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], DType::F32, &DEV).unwrap();
        let b = T::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], vec![2, 2], DType::F32, &DEV).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("A".into(), a);
        inputs.insert("B".into(), b);

        let outputs = run_onnx_graph::<B>(&graph, &inputs, &DEV).unwrap();
        let y = outputs.get("Y").unwrap();
        let data = y.to_f64_vec().unwrap();
        assert_eq!(data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_graph_linear_relu() {
        // Graph:  Z = Relu(X * W + B)
        //   matmul_out = MatMul(X, W)
        //   add_out    = Add(matmul_out, B)
        //   Z          = Relu(add_out)
        let mut model = OnnxModel::new("linear_relu");
        model.inputs.push(("X".into(), vec![1, 2], ONNX_FLOAT));
        model.outputs.push(("Z".into(), vec![1, 3], ONNX_FLOAT));

        // W and B as initializers
        let mut w = OnnxTensor::new("W");
        w.data_type = ONNX_FLOAT;
        w.dims = vec![2, 3];
        w.float_data = vec![1.0, -1.0, 0.5, 0.0, 2.0, -0.5];
        model.initializers.push(w);

        let mut b = OnnxTensor::new("B");
        b.data_type = ONNX_FLOAT;
        b.dims = vec![3];
        b.float_data = vec![0.0, 0.0, 0.0];
        model.initializers.push(b);

        model.nodes.push(OnnxNode {
            inputs: vec!["X".into(), "W".into()],
            outputs: vec!["matmul_out".into()],
            op_type: "MatMul".into(),
            name: "matmul_0".into(),
            attributes: HashMap::new(),
        });
        model.nodes.push(OnnxNode {
            inputs: vec!["matmul_out".into(), "B".into()],
            outputs: vec!["add_out".into()],
            op_type: "Add".into(),
            name: "add_0".into(),
            attributes: HashMap::new(),
        });
        model.nodes.push(OnnxNode {
            inputs: vec!["add_out".into()],
            outputs: vec!["Z".into()],
            op_type: "Relu".into(),
            name: "relu_0".into(),
            attributes: HashMap::new(),
        });

        let graph = build_and_reload_graph(&model);
        assert_eq!(graph.nodes.len(), 3);

        // X = [[1.0, -1.0]]
        // X*W = [[1*1+(-1)*0, 1*(-1)+(-1)*2, 1*0.5+(-1)*(-0.5)]]
        //     = [[1.0, -3.0, 1.0]]
        // Relu => [[1.0, 0.0, 1.0]]
        let x = T::from_f64_slice(&[1.0, -1.0], vec![1, 2], DType::F32, &DEV).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("X".into(), x);

        let outputs = run_onnx_graph::<B>(&graph, &inputs, &DEV).unwrap();
        let z = outputs.get("Z").unwrap();
        let data = z.to_f64_vec().unwrap();
        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_graph_identity_and_dropout() {
        let mut model = OnnxModel::new("id_drop");
        model.inputs.push(("X".into(), vec![3], ONNX_FLOAT));
        model.outputs.push(("Y".into(), vec![3], ONNX_FLOAT));

        model.nodes.push(OnnxNode {
            inputs: vec!["X".into()],
            outputs: vec!["id_out".into()],
            op_type: "Identity".into(),
            name: "id_0".into(),
            attributes: HashMap::new(),
        });
        model.nodes.push(OnnxNode {
            inputs: vec!["id_out".into()],
            outputs: vec!["Y".into()],
            op_type: "Dropout".into(),
            name: "drop_0".into(),
            attributes: HashMap::new(),
        });

        let graph = build_and_reload_graph(&model);
        let x = T::from_f64_slice(&[5.0, -3.0, 7.0], vec![3], DType::F32, &DEV).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("X".into(), x.clone());

        let outputs = run_onnx_graph::<B>(&graph, &inputs, &DEV).unwrap();
        let y = outputs.get("Y").unwrap();
        assert_eq!(y.to_f64_vec().unwrap(), x.to_f64_vec().unwrap());
    }

    #[test]
    fn test_graph_file_roundtrip() {
        // Build, save, load from file, execute
        let mut model = OnnxModel::new("file_rt");
        model.inputs.push(("X".into(), vec![2], ONNX_FLOAT));
        model.outputs.push(("Y".into(), vec![2], ONNX_FLOAT));

        model.nodes.push(OnnxNode {
            inputs: vec!["X".into()],
            outputs: vec!["Y".into()],
            op_type: "Sigmoid".into(),
            name: "sig_0".into(),
            attributes: HashMap::new(),
        });

        let path = std::env::temp_dir().join("shrew_test_graph_rt.onnx");
        model.save(&path).unwrap();

        let graph = load_onnx_graph(&path).unwrap();
        assert_eq!(graph.nodes.len(), 1);

        let x = T::from_f64_slice(&[0.0, 1000.0], vec![2], DType::F32, &DEV).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("X".into(), x);

        let outputs = run_onnx_graph::<B>(&graph, &inputs, &DEV).unwrap();
        let data = outputs.get("Y").unwrap().to_f64_vec().unwrap();
        assert!((data[0] - 0.5).abs() < 1e-5); // sigmoid(0) = 0.5
        assert!((data[1] - 1.0).abs() < 1e-3); // sigmoid(1000) ≈ 1.0

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_decode_attribute_roundtrip() {
        // Encode an Int attribute, decode it back
        let attr = OnnxAttribute::Int(42);
        let encoded = encode_attribute("axis", &attr);
        let (name, decoded) = decode_attribute(&encoded.buf).unwrap();
        assert_eq!(name, "axis");
        match decoded {
            OnnxAttribute::Int(v) => assert_eq!(v, 42),
            _ => panic!("expected Int"),
        }
    }
}
