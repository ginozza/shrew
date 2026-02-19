// =============================================================================
// Safetensors — Interoperable tensor serialization (HuggingFace format)
// =============================================================================
//
// The safetensors format stores tensors in a single flat file:
//
//   ┌──────────────┬──────────────────────┬───────────────────────┐
//   │ 8 bytes      │ N bytes              │ raw data bytes        │
//   │ header size  │ JSON header (UTF-8)  │ (contiguous, LE)      │
//   │ (u64 LE)     │                      │                       │
//   └──────────────┴──────────────────────┴───────────────────────┘
//
// JSON header example:
//   {
//     "__metadata__": { "format": "shrew" },
//     "layer.weight": {
//       "dtype": "F32",
//       "shape": [64, 128],
//       "data_offsets": [0, 32768]
//     }
//   }
//
// Supported dtypes: F32, F64, U8, I64 (U32 stored as I64 for interop).
//
// This implementation intentionally avoids external safetensors crates —
// the format is simple enough to implement from scratch, and this keeps
// the dependency tree lean.
//
// Usage:
//   safetensors::save("model.safetensors", &named_tensors)?;
//   let tensors = safetensors::load::<CpuBackend>("model.safetensors", &device)?;
//
//   // Module-level convenience:
//   safetensors::save_module("model.safetensors", &my_module)?;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use shrew_core::backend::Backend;
use shrew_core::tensor::Tensor;
use shrew_core::DType;

// ─────────────────────────────────────────────────────────────────────────────
// DType ↔ safetensors string
// ─────────────────────────────────────────────────────────────────────────────

fn dtype_to_st(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::U8 => "U8",
        DType::U32 => "U32",
        DType::I64 => "I64",
    }
}

fn st_to_dtype(s: &str) -> shrew_core::Result<DType> {
    match s {
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "F32" => Ok(DType::F32),
        "F64" => Ok(DType::F64),
        "U8" | "BOOL" => Ok(DType::U8),
        "U32" | "U16" | "I32" | "I16" | "I8" => Ok(DType::U32),
        "I64" => Ok(DType::I64),
        _ => Err(shrew_core::Error::msg(format!(
            "Unsupported safetensors dtype: {s}"
        ))),
    }
}

fn dtype_elem_size(dtype: DType) -> usize {
    match dtype {
        DType::F16 => 2,
        DType::BF16 => 2,
        DType::F32 => 4,
        DType::F64 => 8,
        DType::U8 => 1,
        DType::U32 => 4,
        DType::I64 => 8,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Raw bytes extraction / reconstruction
// ─────────────────────────────────────────────────────────────────────────────

fn tensor_to_bytes<B: Backend>(tensor: &Tensor<B>) -> shrew_core::Result<Vec<u8>> {
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

fn tensor_from_bytes<B: Backend>(
    raw: &[u8],
    dims: Vec<usize>,
    dtype: DType,
    device: &B::Device,
) -> shrew_core::Result<Tensor<B>> {
    let elem_size = dtype_elem_size(dtype);
    let num_elems: usize = dims.iter().product();
    let expected = num_elems * elem_size;
    if raw.len() != expected {
        return Err(shrew_core::Error::msg(format!(
            "safetensors: expected {expected} bytes for {num_elems} elements of {:?}, got {}",
            dtype,
            raw.len()
        )));
    }

    let data_f64: Vec<f64> = match dtype {
        DType::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect(),
        DType::BF16 => raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect(),
        DType::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        DType::F64 => raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
        DType::U8 => raw.iter().map(|&v| v as f64).collect(),
        DType::U32 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        DType::I64 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64)
            .collect(),
    };

    let shape = shrew_core::Shape::new(dims);
    Tensor::from_f64_slice(&data_f64, shape, dtype, device)
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON header builder (no serde dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Escape a string for JSON (handles \, ", and control characters).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

struct TensorMeta {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    data_offset_start: usize,
    data_offset_end: usize,
}

fn build_header_json(metas: &[TensorMeta], metadata: Option<&HashMap<String, String>>) -> String {
    let mut json = String::from("{");

    // __metadata__ (optional)
    if let Some(md) = metadata {
        json.push_str("\"__metadata__\":{");
        for (i, (k, v)) in md.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&json_escape(k));
            json.push(':');
            json.push_str(&json_escape(v));
        }
        json.push('}');
        if !metas.is_empty() {
            json.push(',');
        }
    }

    // Tensor entries
    for (i, meta) in metas.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&json_escape(&meta.name));
        json.push_str(":{\"dtype\":\"");
        json.push_str(dtype_to_st(meta.dtype));
        json.push_str("\",\"shape\":[");
        for (j, &d) in meta.shape.iter().enumerate() {
            if j > 0 {
                json.push(',');
            }
            json.push_str(&d.to_string());
        }
        json.push_str("],\"data_offsets\":[");
        json.push_str(&meta.data_offset_start.to_string());
        json.push(',');
        json.push_str(&meta.data_offset_end.to_string());
        json.push_str("]}");
    }

    json.push('}');
    json
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON header parser (minimal, no serde dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Parsed tensor entry from safetensors header.
struct ParsedEntry {
    name: String,
    dtype_str: String,
    shape: Vec<usize>,
    data_offset_start: usize,
    data_offset_end: usize,
}

/// Parse the safetensors JSON header using serde_json.
fn parse_header(json_str: &str) -> shrew_core::Result<Vec<ParsedEntry>> {
    let value: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| shrew_core::Error::msg(format!("safetensors: invalid JSON header: {e}")))?;

    let obj = value
        .as_object()
        .ok_or_else(|| shrew_core::Error::msg("safetensors: header is not a JSON object"))?;

    let mut entries = Vec::new();

    for (key, val) in obj {
        // Skip __metadata__
        if key == "__metadata__" {
            continue;
        }

        let tensor_obj = val.as_object().ok_or_else(|| {
            shrew_core::Error::msg(format!("safetensors: entry '{key}' is not an object"))
        })?;

        let dtype_str = tensor_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| shrew_core::Error::msg(format!("safetensors: '{key}' missing dtype")))?
            .to_string();

        let shape_arr = tensor_obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| shrew_core::Error::msg(format!("safetensors: '{key}' missing shape")))?;

        let shape: Vec<usize> = shape_arr
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let offsets = tensor_obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                shrew_core::Error::msg(format!("safetensors: '{key}' missing data_offsets"))
            })?;

        if offsets.len() != 2 {
            return Err(shrew_core::Error::msg(format!(
                "safetensors: '{key}' data_offsets must have exactly 2 elements"
            )));
        }

        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;

        entries.push(ParsedEntry {
            name: key.clone(),
            dtype_str,
            shape,
            data_offset_start: start,
            data_offset_end: end,
        });
    }

    Ok(entries)
}

// ─────────────────────────────────────────────────────────────────────────────
// Write safetensors
// ─────────────────────────────────────────────────────────────────────────────

/// Write named tensors in safetensors format to a writer.
pub fn write_safetensors<B: Backend>(
    writer: &mut impl Write,
    tensors: &[(String, Tensor<B>)],
) -> shrew_core::Result<()> {
    // Step 1: Serialize all tensor data and compute offsets
    let mut all_data: Vec<u8> = Vec::new();
    let mut metas: Vec<TensorMeta> = Vec::with_capacity(tensors.len());

    for (name, tensor) in tensors {
        let bytes = tensor_to_bytes(tensor)?;
        let start = all_data.len();
        let end = start + bytes.len();
        all_data.extend_from_slice(&bytes);

        metas.push(TensorMeta {
            name: name.clone(),
            dtype: tensor.dtype(),
            shape: tensor.dims().to_vec(),
            data_offset_start: start,
            data_offset_end: end,
        });
    }

    // Step 2: Build JSON header
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "shrew".to_string());
    let header_json = build_header_json(&metas, Some(&metadata));
    let header_bytes = header_json.as_bytes();

    // Step 3: Write header size (u64 LE)
    let header_size = header_bytes.len() as u64;
    writer
        .write_all(&header_size.to_le_bytes())
        .map_err(io_err)?;

    // Step 4: Write JSON header
    writer.write_all(header_bytes).map_err(io_err)?;

    // Step 5: Write raw tensor data
    writer.write_all(&all_data).map_err(io_err)?;

    Ok(())
}

/// Read named tensors from safetensors format.
pub fn read_safetensors<B: Backend>(
    reader: &mut impl Read,
    device: &B::Device,
) -> shrew_core::Result<Vec<(String, Tensor<B>)>> {
    // Step 1: Read header size
    let mut size_buf = [0u8; 8];
    reader.read_exact(&mut size_buf).map_err(io_err)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;

    // Sanity check: header shouldn't be unreasonably large
    if header_size > 100_000_000 {
        return Err(shrew_core::Error::msg(format!(
            "safetensors: header size {header_size} bytes is unreasonably large"
        )));
    }

    // Step 2: Read JSON header
    let mut header_bytes = vec![0u8; header_size];
    reader.read_exact(&mut header_bytes).map_err(io_err)?;
    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| shrew_core::Error::msg(format!("safetensors: invalid UTF-8 header: {e}")))?;

    // Step 3: Parse header
    let entries = parse_header(header_str)?;

    // Step 4: Read all raw data
    let max_offset = entries.iter().map(|e| e.data_offset_end).max().unwrap_or(0);
    let mut all_data = vec![0u8; max_offset];
    if max_offset > 0 {
        reader.read_exact(&mut all_data).map_err(io_err)?;
    }

    // Step 5: Reconstruct tensors
    let mut tensors = Vec::with_capacity(entries.len());
    for entry in &entries {
        let dtype = st_to_dtype(&entry.dtype_str)?;
        let raw = &all_data[entry.data_offset_start..entry.data_offset_end];
        let tensor = tensor_from_bytes::<B>(raw, entry.shape.clone(), dtype, device)?;
        tensors.push((entry.name.clone(), tensor));
    }

    Ok(tensors)
}

fn io_err(e: std::io::Error) -> shrew_core::Error {
    shrew_core::Error::msg(format!("IO error: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// High-level file API
// ─────────────────────────────────────────────────────────────────────────────

/// Save named tensors to a `.safetensors` file.
///
/// ```rust,no_run
/// use shrew::safetensors;
/// use shrew::prelude::*;
///
/// let w = Tensor::<CpuBackend>::zeros((2, 3), DType::F32, &CpuDevice).unwrap();
/// let tensors = vec![("weight".to_string(), w)];
/// safetensors::save("model.safetensors", &tensors).unwrap();
/// ```
pub fn save<B: Backend>(
    path: impl AsRef<Path>,
    tensors: &[(String, Tensor<B>)],
) -> shrew_core::Result<()> {
    let file = File::create(path.as_ref()).map_err(io_err)?;
    let mut writer = BufWriter::new(file);
    write_safetensors(&mut writer, tensors)?;
    writer.flush().map_err(io_err)?;
    Ok(())
}

/// Load named tensors from a `.safetensors` file.
///
/// ```rust,no_run
/// use shrew::safetensors;
/// use shrew::prelude::*;
///
/// let tensors = safetensors::load::<CpuBackend>("model.safetensors", &CpuDevice).unwrap();
/// for (name, tensor) in &tensors {
///     println!("{name}: {:?}", tensor.dims());
/// }
/// ```
pub fn load<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> shrew_core::Result<Vec<(String, Tensor<B>)>> {
    let file = File::open(path.as_ref()).map_err(io_err)?;
    let mut reader = BufReader::new(file);
    read_safetensors(&mut reader, device)
}

// ─────────────────────────────────────────────────────────────────────────────
// In-memory API (for testing)
// ─────────────────────────────────────────────────────────────────────────────

/// Serialize named tensors to an in-memory byte vector in safetensors format.
pub fn to_bytes<B: Backend>(tensors: &[(String, Tensor<B>)]) -> shrew_core::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_safetensors(&mut buf, tensors)?;
    Ok(buf)
}

/// Deserialize named tensors from an in-memory safetensors byte slice.
pub fn from_bytes<B: Backend>(
    data: &[u8],
    device: &B::Device,
) -> shrew_core::Result<Vec<(String, Tensor<B>)>> {
    let mut cursor = std::io::Cursor::new(data);
    read_safetensors(&mut cursor, device)
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-level convenience
// ─────────────────────────────────────────────────────────────────────────────

/// Save a module's parameters to a `.safetensors` file using its
/// `named_parameters()`.
///
/// ```rust,no_run
/// use shrew::safetensors;
/// use shrew::prelude::*;
///
/// let linear = Linear::<CpuBackend>::new(3, 2, true, DType::F32, &CpuDevice).unwrap();
/// safetensors::save_module("linear.safetensors", &linear).unwrap();
/// ```
pub fn save_module<B: Backend>(
    path: impl AsRef<Path>,
    module: &dyn shrew_nn::Module<B>,
) -> shrew_core::Result<()> {
    let named = module.named_parameters();
    save(path, &named)
}

/// Load parameters from a `.safetensors` file into a state-dict map.
///
/// Returns a `HashMap<String, Tensor<B>>` for flexible parameter loading.
pub fn load_state_dict<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> shrew_core::Result<HashMap<String, Tensor<B>>> {
    let tensors = load(path, device)?;
    Ok(tensors.into_iter().collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shrew_cpu::{CpuBackend, CpuDevice};

    type CpuTensor = Tensor<CpuBackend>;

    #[test]
    fn test_roundtrip_f32() {
        let dev = CpuDevice;
        let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev).unwrap();

        let tensors = vec![("weight".to_string(), t.clone())];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "weight");
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

        let tensors = vec![("precision".to_string(), t.clone())];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        let orig = t.to_f64_vec().unwrap();
        let restored = loaded[0].1.to_f64_vec().unwrap();
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
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev)
                .unwrap();
        let b = CpuTensor::from_f64_slice(&[0.1, 0.2, 0.3], (3,), DType::F32, &dev).unwrap();

        let tensors = vec![
            ("layer.weight".to_string(), w.clone()),
            ("layer.bias".to_string(), b.clone()),
        ];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &dev).unwrap();

        assert_eq!(loaded.len(), 2);

        // Find by name (JSON object order may vary)
        let map: HashMap<String, CpuTensor> = loaded.into_iter().collect();
        assert!(map.contains_key("layer.weight"));
        assert!(map.contains_key("layer.bias"));
        assert_eq!(map["layer.weight"].dims(), &[2, 3]);
        assert_eq!(map["layer.bias"].dims(), &[3]);
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
    fn test_empty() {
        let tensors: Vec<(String, CpuTensor)> = vec![];
        let bytes = to_bytes(&tensors).unwrap();
        let loaded = from_bytes::<CpuBackend>(&bytes, &CpuDevice).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_file_roundtrip() {
        let dev = CpuDevice;
        let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let tensors = vec![("test_param".to_string(), t.clone())];

        let path = std::env::temp_dir().join("shrew_test_safetensors.safetensors");
        save(&path, &tensors).unwrap();
        let loaded = load::<CpuBackend>(&path, &dev).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.len(), 1);
        let orig = t.to_f64_vec().unwrap();
        let restored = loaded[0].1.to_f64_vec().unwrap();
        assert_eq!(orig, restored);
    }

    #[test]
    fn test_header_format() {
        // Verify the header can be parsed independently
        let metas = vec![TensorMeta {
            name: "layer.weight".to_string(),
            dtype: DType::F32,
            shape: vec![3, 4],
            data_offset_start: 0,
            data_offset_end: 48,
        }];
        let json = build_header_json(&metas, None);

        // Should be valid JSON parsable by serde_json
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let entry = parsed.get("layer.weight").unwrap();
        assert_eq!(entry["dtype"].as_str().unwrap(), "F32");
        assert_eq!(entry["shape"][0].as_u64().unwrap(), 3);
        assert_eq!(entry["shape"][1].as_u64().unwrap(), 4);
        assert_eq!(entry["data_offsets"][0].as_u64().unwrap(), 0);
        assert_eq!(entry["data_offsets"][1].as_u64().unwrap(), 48);
    }

    #[test]
    fn test_json_escape() {
        assert_eq!(json_escape("hello"), "\"hello\"");
        assert_eq!(json_escape("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_escape("a\\b"), "\"a\\\\b\"");
        assert_eq!(json_escape("a\nb"), "\"a\\nb\"");
    }

    #[test]
    fn test_state_dict_roundtrip() {
        let dev = CpuDevice;
        let w = CpuTensor::from_f64_slice(&[1.0, 2.0], (1, 2), DType::F32, &dev).unwrap();
        let b = CpuTensor::from_f64_slice(&[0.5], (1,), DType::F32, &dev).unwrap();

        let tensors = vec![("fc.weight".to_string(), w), ("fc.bias".to_string(), b)];

        let path = std::env::temp_dir().join("shrew_test_state_dict.safetensors");
        save(&path, &tensors).unwrap();
        let sd = load_state_dict::<CpuBackend>(&path, &dev).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(sd.contains_key("fc.weight"));
        assert!(sd.contains_key("fc.bias"));
        assert_eq!(sd["fc.weight"].dims(), &[1, 2]);
        assert_eq!(sd["fc.bias"].dims(), &[1]);
    }

    #[test]
    fn test_save_module_linear() {
        use shrew_nn::Linear;

        let dev = CpuDevice;
        let linear = Linear::<CpuBackend>::new(3, 2, true, DType::F32, &dev).unwrap();

        let path = std::env::temp_dir().join("shrew_test_module_save.safetensors");
        save_module(&path, &linear).unwrap();
        let loaded = load::<CpuBackend>(&path, &dev).unwrap();
        std::fs::remove_file(&path).ok();

        let map: HashMap<String, CpuTensor> = loaded.into_iter().collect();
        assert!(map.contains_key("weight"), "missing 'weight' key");
        assert!(map.contains_key("bias"), "missing 'bias' key");
        assert_eq!(map["weight"].dims(), &[2, 3]);
        assert_eq!(map["bias"].dims(), &[1, 2]);
    }
}
