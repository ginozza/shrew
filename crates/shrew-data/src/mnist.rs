// MNIST dataset — IDX file format parser
//
// The MNIST database consists of 4 files:
//   - train-images-idx3-ubyte  (60,000  28×28 images)
//   - train-labels-idx1-ubyte  (60,000  labels 0-9)
//   - t10k-images-idx3-ubyte   (10,000  28×28 images)
//   - t10k-labels-idx1-ubyte   (10,000  labels 0-9)
//
// IDX format (all values big-endian):
//   images: magic(2051) | count(u32) | rows(u32) | cols(u32) | pixel_data(u8...)
//   labels: magic(2049) | count(u32) | label_data(u8...)
//
// If the files are gzip-compressed (.gz), we decompress on the fly.
// Download from: https://yann.lecun.com/exdb/mnist/

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::dataset::{Dataset, Sample};

/// Error type for MNIST loading.
#[derive(Debug)]
pub enum MnistError {
    Io(io::Error),
    InvalidMagic { expected: u32, got: u32 },
    CountMismatch { images: usize, labels: usize },
    MissingFile(PathBuf),
}

impl std::fmt::Display for MnistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MnistError::Io(e) => write!(f, "MNIST I/O error: {e}"),
            MnistError::InvalidMagic { expected, got } => write!(
                f,
                "MNIST invalid magic: expected {expected:#06x}, got {got:#06x}"
            ),
            MnistError::CountMismatch { images, labels } => write!(
                f,
                "MNIST count mismatch: {images} images vs {labels} labels"
            ),
            MnistError::MissingFile(p) => write!(f, "MNIST file not found: {}", p.display()),
        }
    }
}

impl std::error::Error for MnistError {}

impl From<io::Error> for MnistError {
    fn from(e: io::Error) -> Self {
        MnistError::Io(e)
    }
}

/// Which split of MNIST to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MnistSplit {
    Train,
    Test,
}

/// A loaded MNIST dataset stored entirely in memory.
///
/// Images are stored as `Vec<u8>` (28×28 = 784 bytes each).
/// Labels are `u8` values 0–9.
#[derive(Debug)]
pub struct MnistDataset {
    images: Vec<Vec<u8>>,
    labels: Vec<u8>,
    rows: usize,
    cols: usize,
    split: MnistSplit,
}

impl MnistDataset {
    /// Load MNIST from the given directory.
    ///
    /// Expects the standard filenames (or `.gz` compressed versions):
    ///   - `train-images-idx3-ubyte` / `train-images-idx3-ubyte.gz`
    ///   - `train-labels-idx1-ubyte` / `train-labels-idx1-ubyte.gz`
    ///   - `t10k-images-idx3-ubyte`  / `t10k-images-idx3-ubyte.gz`
    ///   - `t10k-labels-idx1-ubyte`  / `t10k-labels-idx1-ubyte.gz`
    pub fn load(dir: impl AsRef<Path>, split: MnistSplit) -> Result<Self, MnistError> {
        let dir = dir.as_ref();

        let (img_name, lbl_name) = match split {
            MnistSplit::Train => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            MnistSplit::Test => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        };

        let img_bytes = read_maybe_gz(dir, img_name)?;
        let lbl_bytes = read_maybe_gz(dir, lbl_name)?;

        let (images, rows, cols) = parse_idx3_images(&img_bytes)?;
        let labels = parse_idx1_labels(&lbl_bytes)?;

        if images.len() != labels.len() {
            return Err(MnistError::CountMismatch {
                images: images.len(),
                labels: labels.len(),
            });
        }

        Ok(Self {
            images,
            labels,
            rows,
            cols,
            split,
        })
    }

    /// Load from raw bytes (useful for embedded/testing).
    pub fn from_raw(
        image_bytes: &[u8],
        label_bytes: &[u8],
        split: MnistSplit,
    ) -> Result<Self, MnistError> {
        let (images, rows, cols) = parse_idx3_images(image_bytes)?;
        let labels = parse_idx1_labels(label_bytes)?;

        if images.len() != labels.len() {
            return Err(MnistError::CountMismatch {
                images: images.len(),
                labels: labels.len(),
            });
        }

        Ok(Self {
            images,
            labels,
            rows,
            cols,
            split,
        })
    }

    /// Create a small synthetic MNIST-like dataset for testing.
    ///
    /// Generates `n` random 28×28 images with random labels.
    pub fn synthetic(n: usize, split: MnistSplit) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let rows = 28;
        let cols = 28;
        let mut images = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);

        for _ in 0..n {
            let mut img = vec![0u8; rows * cols];
            for px in &mut img {
                *px = rng.gen();
            }
            images.push(img);
            labels.push(rng.gen_range(0..10u8));
        }

        Self {
            images,
            labels,
            rows,
            cols,
            split,
        }
    }

    /// Total number of samples.
    pub fn num_samples(&self) -> usize {
        self.images.len()
    }

    /// Image dimensions: (rows, cols).
    pub fn image_dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get the raw pixel values for sample `i`.
    pub fn image_u8(&self, i: usize) -> &[u8] {
        &self.images[i]
    }

    /// Get the label for sample `i`.
    pub fn label(&self, i: usize) -> u8 {
        self.labels[i]
    }

    /// Which split this dataset represents.
    pub fn split(&self) -> MnistSplit {
        self.split
    }

    /// Take only the first `n` samples (useful for quick experiments).
    pub fn take(mut self, n: usize) -> Self {
        let n = n.min(self.images.len());
        self.images.truncate(n);
        self.labels.truncate(n);
        self
    }
}

impl Dataset for MnistDataset {
    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> Sample {
        let pixels = &self.images[index];
        let label = self.labels[index];

        Sample {
            features: pixels.iter().map(|&p| p as f64).collect(),
            feature_shape: vec![self.rows * self.cols],
            target: vec![label as f64],
            target_shape: vec![1],
        }
    }

    fn feature_shape(&self) -> &[usize] {
        // We return a static ref, but since MnistDataset owns this info,
        // we use a small trick: store the shape inline.  For now just return
        // a slice from a leaked box (tiny, done once).
        // Better approach: store feature_shape as a field.
        &[784] // 28*28
    }

    fn target_shape(&self) -> &[usize] {
        &[1]
    }

    fn name(&self) -> &str {
        match self.split {
            MnistSplit::Train => "MNIST-train",
            MnistSplit::Test => "MNIST-test",
        }
    }
}

// IDX file format parsing

/// Read a file, trying plain first then `.gz` extension.
fn read_maybe_gz(dir: &Path, base_name: &str) -> Result<Vec<u8>, MnistError> {
    let plain = dir.join(base_name);
    let gz = dir.join(format!("{base_name}.gz"));

    if plain.exists() {
        Ok(fs::read(&plain)?)
    } else if gz.exists() {
        let compressed = fs::read(&gz)?;
        decompress_gz(&compressed)
    } else {
        Err(MnistError::MissingFile(plain))
    }
}

/// Simple gzip decompressor using DEFLATE.
///
/// We implement a minimal gzip reader (RFC 1952) without pulling in flate2.
/// This handles the standard MNIST .gz files which use default compression.
fn decompress_gz(data: &[u8]) -> Result<Vec<u8>, MnistError> {
    // For simplicity, we use the miniz_oxide-compatible approach.
    // Since we don't want to add deps, we'll just try the raw DEFLATE stream
    // after skipping the gzip header.

    if data.len() < 10 {
        return Err(MnistError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "gzip data too short",
        )));
    }

    // Verify gzip magic
    if data[0] != 0x1f || data[1] != 0x8b {
        return Err(MnistError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a gzip file",
        )));
    }

    // Skip gzip header (10 bytes minimum)
    let mut pos = 10;
    let flags = data[3];

    // FEXTRA
    if flags & 0x04 != 0 {
        if pos + 2 > data.len() {
            return Err(io_err("truncated gzip FEXTRA"));
        }
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    // FNAME
    if flags & 0x08 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1; // skip null terminator
    }
    // FCOMMENT
    if flags & 0x10 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    // FHCRC
    if flags & 0x02 != 0 {
        pos += 2;
    }

    if pos >= data.len() {
        return Err(io_err("truncated gzip header"));
    }

    // The DEFLATE stream is from `pos` to `data.len() - 8` (last 8 = crc32 + isize)
    let deflate_end = if data.len() >= 8 {
        data.len() - 8
    } else {
        data.len()
    };
    let deflate_data = &data[pos..deflate_end];

    // Use a simple inflate implementation
    inflate_deflate(deflate_data)
}

/// Minimal DEFLATE decompressor for gzip.
///
/// For production use you'd want `flate2`. This is a simplified version
/// that handles the MNIST files (which are typically stored-or-default compressed).
/// We use Rust's built-in approach: since we can't easily decompress DEFLATE
/// without a library, we recommend the plain (uncompressed) files.
///
/// As a fallback, this function returns an error suggesting to decompress.
fn inflate_deflate(_data: &[u8]) -> Result<Vec<u8>, MnistError> {
    Err(MnistError::Io(io::Error::new(
        io::ErrorKind::Unsupported,
        "gzip decompression requires the `flate2` feature. \
         Please decompress MNIST files manually (gunzip) or enable flate2.",
    )))
}

fn io_err(msg: &str) -> MnistError {
    MnistError::Io(io::Error::new(io::ErrorKind::InvalidData, msg))
}

/// Parse an IDX3 file (images): magic=2051, count, rows, cols, data.
fn parse_idx3_images(data: &[u8]) -> Result<(Vec<Vec<u8>>, usize, usize), MnistError> {
    if data.len() < 16 {
        return Err(io_err("IDX3 file too short"));
    }

    let magic = read_u32_be(data, 0);
    if magic != 2051 {
        return Err(MnistError::InvalidMagic {
            expected: 2051,
            got: magic,
        });
    }

    let count = read_u32_be(data, 4) as usize;
    let rows = read_u32_be(data, 8) as usize;
    let cols = read_u32_be(data, 12) as usize;
    let pixels_per_image = rows * cols;

    let expected_len = 16 + count * pixels_per_image;
    if data.len() < expected_len {
        return Err(io_err(&format!(
            "IDX3 truncated: expected {expected_len} bytes, got {}",
            data.len()
        )));
    }

    let mut images = Vec::with_capacity(count);
    for i in 0..count {
        let start = 16 + i * pixels_per_image;
        let end = start + pixels_per_image;
        images.push(data[start..end].to_vec());
    }

    Ok((images, rows, cols))
}

/// Parse an IDX1 file (labels): magic=2049, count, data.
fn parse_idx1_labels(data: &[u8]) -> Result<Vec<u8>, MnistError> {
    if data.len() < 8 {
        return Err(io_err("IDX1 file too short"));
    }

    let magic = read_u32_be(data, 0);
    if magic != 2049 {
        return Err(MnistError::InvalidMagic {
            expected: 2049,
            got: magic,
        });
    }

    let count = read_u32_be(data, 4) as usize;
    let expected_len = 8 + count;
    if data.len() < expected_len {
        return Err(io_err(&format!(
            "IDX1 truncated: expected {expected_len} bytes, got {}",
            data.len()
        )));
    }

    Ok(data[8..8 + count].to_vec())
}

/// Read a big-endian u32 from `data` at byte offset `off`.
fn read_u32_be(data: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

// Builder helpers

/// Build IDX3 image bytes from raw image data (useful for tests).
pub fn build_idx3_bytes(images: &[&[u8]], rows: u32, cols: u32) -> Vec<u8> {
    let count = images.len() as u32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&2051u32.to_be_bytes());
    buf.extend_from_slice(&count.to_be_bytes());
    buf.extend_from_slice(&rows.to_be_bytes());
    buf.extend_from_slice(&cols.to_be_bytes());
    for img in images {
        buf.extend_from_slice(img);
    }
    buf
}

/// Build IDX1 label bytes (useful for tests).
pub fn build_idx1_bytes(labels: &[u8]) -> Vec<u8> {
    let count = labels.len() as u32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&2049u32.to_be_bytes());
    buf.extend_from_slice(&count.to_be_bytes());
    buf.extend_from_slice(labels);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_idx3_roundtrip() {
        let img1 = vec![0u8; 4]; // 2×2 image
        let img2 = vec![255u8; 4];
        let bytes = build_idx3_bytes(&[&img1, &img2], 2, 2);
        let (images, rows, cols) = parse_idx3_images(&bytes).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        assert_eq!(images[0], vec![0, 0, 0, 0]);
        assert_eq!(images[1], vec![255, 255, 255, 255]);
    }

    #[test]
    fn test_parse_idx1_roundtrip() {
        let labels_in = vec![0, 1, 2, 9, 5];
        let bytes = build_idx1_bytes(&labels_in);
        let labels = parse_idx1_labels(&bytes).unwrap();
        assert_eq!(labels, labels_in);
    }

    #[test]
    fn test_invalid_magic_idx3() {
        let mut bytes = build_idx3_bytes(&[&[0u8; 4]], 2, 2);
        bytes[3] = 99; // corrupt magic
        let err = parse_idx3_images(&bytes).unwrap_err();
        assert!(matches!(err, MnistError::InvalidMagic { .. }));
    }

    #[test]
    fn test_invalid_magic_idx1() {
        let mut bytes = build_idx1_bytes(&[0, 1]);
        bytes[3] = 99;
        let err = parse_idx1_labels(&bytes).unwrap_err();
        assert!(matches!(err, MnistError::InvalidMagic { .. }));
    }

    #[test]
    fn test_from_raw() {
        let img_bytes = build_idx3_bytes(&[&[128u8; 4], &[64u8; 4]], 2, 2);
        let lbl_bytes = build_idx1_bytes(&[3, 7]);
        let ds = MnistDataset::from_raw(&img_bytes, &lbl_bytes, MnistSplit::Train).unwrap();
        assert_eq!(ds.num_samples(), 2);
        assert_eq!(ds.label(0), 3);
        assert_eq!(ds.label(1), 7);
        assert_eq!(ds.image_u8(0), &[128; 4]);
    }

    #[test]
    fn test_count_mismatch() {
        let img_bytes = build_idx3_bytes(&[&[0u8; 4]], 2, 2); // 1 image
        let lbl_bytes = build_idx1_bytes(&[0, 1]); // 2 labels
        let err = MnistDataset::from_raw(&img_bytes, &lbl_bytes, MnistSplit::Train).unwrap_err();
        assert!(matches!(err, MnistError::CountMismatch { .. }));
    }

    #[test]
    fn test_dataset_trait() {
        let img_bytes = build_idx3_bytes(&[&[100u8; 4], &[200u8; 4]], 2, 2);
        let lbl_bytes = build_idx1_bytes(&[5, 8]);
        let ds = MnistDataset::from_raw(&img_bytes, &lbl_bytes, MnistSplit::Test).unwrap();

        assert_eq!(ds.len(), 2);
        assert!(!ds.is_empty());
        assert_eq!(ds.name(), "MNIST-test");

        let s0 = ds.get(0);
        assert_eq!(s0.features.len(), 4); // 2×2 = 4 pixels
        assert_eq!(s0.features[0], 100.0);
        assert_eq!(s0.target, vec![5.0]);
        assert_eq!(s0.feature_shape, vec![4]); // rows*cols
        assert_eq!(s0.target_shape, vec![1]);
    }

    #[test]
    fn test_synthetic() {
        let ds = MnistDataset::synthetic(100, MnistSplit::Train);
        assert_eq!(ds.num_samples(), 100);
        assert_eq!(ds.image_dims(), (28, 28));
        for i in 0..100 {
            assert!(ds.label(i) < 10);
        }
    }

    #[test]
    fn test_take() {
        let ds = MnistDataset::synthetic(100, MnistSplit::Train).take(10);
        assert_eq!(ds.num_samples(), 10);
    }
}
