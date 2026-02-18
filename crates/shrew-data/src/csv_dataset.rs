// CsvDataset — load tabular data from CSV files
//
// A lightweight CSV parser that doesn't require an external CSV crate.
// Supports headerless or header-row CSVs. The caller specifies which columns
// are features and which are targets.

use std::fs;
use std::path::Path;

use crate::dataset::{Dataset, Sample};

/// A dataset loaded from a CSV file.
///
/// All values are parsed as `f64`.  Non-numeric fields will cause a panic.
///
/// # Example
/// ```ignore
/// // Load iris.csv: 4 feature columns, 1 target column (last)
/// let ds = CsvDataset::load("data/iris.csv", CsvConfig {
///     has_header: true,
///     feature_cols: vec![0, 1, 2, 3],
///     target_cols: vec![4],
///     delimiter: b',',
/// }).unwrap();
/// ```
#[derive(Debug)]
pub struct CsvDataset {
    samples: Vec<Sample>,
    feature_shape: Vec<usize>,
    target_shape: Vec<usize>,
}

/// Configuration for loading a CSV file.
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// Whether the first row is a header (to be skipped).
    pub has_header: bool,
    /// Column indices to use as features.
    pub feature_cols: Vec<usize>,
    /// Column indices to use as targets.
    pub target_cols: Vec<usize>,
    /// Delimiter character (default: `,`).
    pub delimiter: u8,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            has_header: true,
            feature_cols: Vec::new(),
            target_cols: Vec::new(),
            delimiter: b',',
        }
    }
}

impl CsvConfig {
    pub fn has_header(mut self, h: bool) -> Self {
        self.has_header = h;
        self
    }
    pub fn feature_cols(mut self, cols: Vec<usize>) -> Self {
        self.feature_cols = cols;
        self
    }
    pub fn target_cols(mut self, cols: Vec<usize>) -> Self {
        self.target_cols = cols;
        self
    }
    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }
}

impl CsvDataset {
    /// Load a CSV file from disk.
    pub fn load<P: AsRef<Path>>(path: P, config: CsvConfig) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("CsvDataset: failed to read {:?}: {}", path.as_ref(), e))?;
        Self::from_string(&content, config)
    }

    /// Parse CSV from an in-memory string.
    pub fn from_string(content: &str, config: CsvConfig) -> Result<Self, String> {
        let delim = config.delimiter as char;
        let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();

        if lines.is_empty() {
            return Err("CsvDataset: empty CSV".to_string());
        }

        let start = if config.has_header { 1 } else { 0 };
        if start >= lines.len() {
            return Err("CsvDataset: CSV has only a header, no data".to_string());
        }

        // Auto-detect columns if not specified
        let first_row: Vec<&str> = lines[start].split(delim).collect();
        let num_cols = first_row.len();

        let feat_cols = if config.feature_cols.is_empty() {
            // All columns except the last
            (0..num_cols.saturating_sub(1)).collect::<Vec<_>>()
        } else {
            config.feature_cols
        };

        let tgt_cols = if config.target_cols.is_empty() {
            // Last column only
            vec![num_cols - 1]
        } else {
            config.target_cols
        };

        let mut samples = Vec::with_capacity(lines.len() - start);

        for (line_no, &line) in lines[start..].iter().enumerate() {
            let cols: Vec<&str> = line.split(delim).collect();
            if cols.len() != num_cols {
                return Err(format!(
                    "CsvDataset: line {} has {} columns, expected {}",
                    line_no + start + 1,
                    cols.len(),
                    num_cols
                ));
            }

            let mut features = Vec::with_capacity(feat_cols.len());
            for &c in &feat_cols {
                let val: f64 = cols[c].trim().parse().map_err(|e| {
                    format!(
                        "CsvDataset: line {}, col {}: parse error: {}",
                        line_no + start + 1,
                        c,
                        e
                    )
                })?;
                features.push(val);
            }

            let mut target = Vec::with_capacity(tgt_cols.len());
            for &c in &tgt_cols {
                let val: f64 = cols[c].trim().parse().map_err(|e| {
                    format!(
                        "CsvDataset: line {}, col {}: parse error: {}",
                        line_no + start + 1,
                        c,
                        e
                    )
                })?;
                target.push(val);
            }

            samples.push(Sample {
                features,
                feature_shape: vec![feat_cols.len()],
                target,
                target_shape: vec![tgt_cols.len()],
            });
        }

        let feature_shape = vec![feat_cols.len()];
        let target_shape = vec![tgt_cols.len()];

        Ok(Self {
            samples,
            feature_shape,
            target_shape,
        })
    }
}

impl Dataset for CsvDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Sample {
        self.samples[index].clone()
    }

    fn feature_shape(&self) -> &[usize] {
        &self.feature_shape
    }

    fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    fn name(&self) -> &str {
        "csv"
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_with_header() {
        let csv = "a,b,c\n1.0,2.0,0.0\n3.0,4.0,1.0\n5.0,6.0,0.0\n";
        let config = CsvConfig::default();
        let ds = CsvDataset::from_string(csv, config).unwrap();
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.feature_shape(), &[2]);
        assert_eq!(ds.target_shape(), &[1]);
        assert_eq!(ds.get(0).features, vec![1.0, 2.0]);
        assert_eq!(ds.get(0).target, vec![0.0]);
        assert_eq!(ds.get(2).features, vec![5.0, 6.0]);
    }

    #[test]
    fn csv_no_header() {
        let csv = "1.0,2.0,3.0\n4.0,5.0,6.0\n";
        let config = CsvConfig::default().has_header(false);
        let ds = CsvDataset::from_string(csv, config).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.get(0).features, vec![1.0, 2.0]);
        assert_eq!(ds.get(0).target, vec![3.0]);
    }

    #[test]
    fn csv_custom_columns() {
        let csv = "a,b,c,d\n1,2,3,4\n5,6,7,8\n";
        let config = CsvConfig::default()
            .feature_cols(vec![0, 2])
            .target_cols(vec![1, 3]);
        let ds = CsvDataset::from_string(csv, config).unwrap();
        assert_eq!(ds.feature_shape(), &[2]);
        assert_eq!(ds.target_shape(), &[2]);
        assert_eq!(ds.get(0).features, vec![1.0, 3.0]);
        assert_eq!(ds.get(0).target, vec![2.0, 4.0]);
    }

    #[test]
    fn csv_tab_delimiter() {
        let csv = "a\tb\tc\n1.0\t2.0\t0.0\n3.0\t4.0\t1.0\n";
        let config = CsvConfig::default().delimiter(b'\t');
        let ds = CsvDataset::from_string(csv, config).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.get(0).features, vec![1.0, 2.0]);
    }

    #[test]
    fn csv_parse_error() {
        let csv = "a,b,c\n1.0,hello,0.0\n";
        let config = CsvConfig::default();
        let result = CsvDataset::from_string(csv, config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("parse error"));
    }

    #[test]
    fn csv_empty() {
        let csv = "";
        let result = CsvDataset::from_string(csv, CsvConfig::default());
        assert!(result.is_err());
    }
}
