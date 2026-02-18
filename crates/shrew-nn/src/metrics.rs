// Evaluation Metrics
//
// Classification metrics:  accuracy, precision, recall, f1_score, confusion_matrix
// Regression metrics:      r2_score, mae, rmse, mape
// Language model metrics:  perplexity
// Ranking metrics:         top_k_accuracy
//
// All functions operate on f64 slices or Tensor<B> for maximum flexibility.
// For classification, we follow sklearn conventions:
//   - predictions = class indices (argmax of logits)
//   - targets = true class indices
//
// Multi-class averaging: macro (default), micro, weighted, per-class.

use shrew_core::backend::Backend;
use shrew_core::tensor::Tensor;
use shrew_core::Result;

// Averaging strategy

/// How to average per-class metrics for multi-class problems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Average {
    /// Compute metric per class, then take unweighted mean.
    Macro,
    /// Compute globally: total TP / (total TP + total FP/FN).
    Micro,
    /// Per-class metric weighted by class support (number of true instances).
    Weighted,
}

// Confusion matrix

/// NxN confusion matrix. Entry [i][j] = count of samples with true class i
/// predicted as class j.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    pub matrix: Vec<Vec<u64>>,
    pub n_classes: usize,
}

impl ConfusionMatrix {
    /// Build a confusion matrix from predicted and true class indices.
    pub fn from_predictions(predictions: &[usize], targets: &[usize], n_classes: usize) -> Self {
        let mut matrix = vec![vec![0u64; n_classes]; n_classes];
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            if target < n_classes && pred < n_classes {
                matrix[target][pred] += 1;
            }
        }
        ConfusionMatrix { matrix, n_classes }
    }

    /// True positives for class c.
    pub fn tp(&self, c: usize) -> u64 {
        self.matrix[c][c]
    }

    /// False positives for class c (predicted c but was not c).
    pub fn fp(&self, c: usize) -> u64 {
        (0..self.n_classes)
            .map(|r| if r != c { self.matrix[r][c] } else { 0 })
            .sum()
    }

    /// False negatives for class c (was c but predicted other).
    pub fn fn_(&self, c: usize) -> u64 {
        (0..self.n_classes)
            .map(|col| if col != c { self.matrix[c][col] } else { 0 })
            .sum()
    }

    /// True negatives for class c.
    pub fn tn(&self, c: usize) -> u64 {
        let total: u64 = self.matrix.iter().flat_map(|r| r.iter()).sum();
        total - self.tp(c) - self.fp(c) - self.fn_(c)
    }

    /// Support (number of true instances) for class c.
    pub fn support(&self, c: usize) -> u64 {
        self.matrix[c].iter().sum()
    }

    /// Total number of samples.
    pub fn total(&self) -> u64 {
        self.matrix.iter().flat_map(|r| r.iter()).sum()
    }

    /// Pretty-print the confusion matrix.
    pub fn to_string_table(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("{:>8}", ""));
        for c in 0..self.n_classes {
            s.push_str(&format!("{:>8}", format!("Pred {c}")));
        }
        s.push('\n');
        for r in 0..self.n_classes {
            s.push_str(&format!("{:>8}", format!("True {r}")));
            for c in 0..self.n_classes {
                s.push_str(&format!("{:>8}", self.matrix[r][c]));
            }
            s.push('\n');
        }
        s
    }
}

// Classification metrics (from class indices)

/// Classification accuracy: fraction of correct predictions.
pub fn accuracy(predictions: &[usize], targets: &[usize]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();
    correct as f64 / predictions.len() as f64
}

/// Precision for multi-class classification.
///
/// Precision = TP / (TP + FP) — how many selected items are relevant.
pub fn precision(predictions: &[usize], targets: &[usize], n_classes: usize, avg: Average) -> f64 {
    let cm = ConfusionMatrix::from_predictions(predictions, targets, n_classes);
    match avg {
        Average::Micro => {
            let total_tp: u64 = (0..n_classes).map(|c| cm.tp(c)).sum();
            let total_tp_fp: u64 = (0..n_classes).map(|c| cm.tp(c) + cm.fp(c)).sum();
            if total_tp_fp == 0 {
                0.0
            } else {
                total_tp as f64 / total_tp_fp as f64
            }
        }
        Average::Macro => {
            let precs: Vec<f64> = (0..n_classes)
                .map(|c| {
                    let denom = cm.tp(c) + cm.fp(c);
                    if denom == 0 {
                        0.0
                    } else {
                        cm.tp(c) as f64 / denom as f64
                    }
                })
                .collect();
            precs.iter().sum::<f64>() / n_classes as f64
        }
        Average::Weighted => {
            let total = cm.total() as f64;
            if total == 0.0 {
                return 0.0;
            }
            (0..n_classes)
                .map(|c| {
                    let denom = cm.tp(c) + cm.fp(c);
                    let p = if denom == 0 {
                        0.0
                    } else {
                        cm.tp(c) as f64 / denom as f64
                    };
                    p * cm.support(c) as f64 / total
                })
                .sum()
        }
    }
}

/// Recall for multi-class classification.
///
/// Recall = TP / (TP + FN) — how many relevant items are selected.
pub fn recall(predictions: &[usize], targets: &[usize], n_classes: usize, avg: Average) -> f64 {
    let cm = ConfusionMatrix::from_predictions(predictions, targets, n_classes);
    match avg {
        Average::Micro => {
            let total_tp: u64 = (0..n_classes).map(|c| cm.tp(c)).sum();
            let total_tp_fn: u64 = (0..n_classes).map(|c| cm.tp(c) + cm.fn_(c)).sum();
            if total_tp_fn == 0 {
                0.0
            } else {
                total_tp as f64 / total_tp_fn as f64
            }
        }
        Average::Macro => {
            let recs: Vec<f64> = (0..n_classes)
                .map(|c| {
                    let denom = cm.tp(c) + cm.fn_(c);
                    if denom == 0 {
                        0.0
                    } else {
                        cm.tp(c) as f64 / denom as f64
                    }
                })
                .collect();
            recs.iter().sum::<f64>() / n_classes as f64
        }
        Average::Weighted => {
            let total = cm.total() as f64;
            if total == 0.0 {
                return 0.0;
            }
            (0..n_classes)
                .map(|c| {
                    let denom = cm.tp(c) + cm.fn_(c);
                    let r = if denom == 0 {
                        0.0
                    } else {
                        cm.tp(c) as f64 / denom as f64
                    };
                    r * cm.support(c) as f64 / total
                })
                .sum()
        }
    }
}

/// F1 Score — harmonic mean of precision and recall.
///
/// F1 = 2 * (precision * recall) / (precision + recall)
pub fn f1_score(predictions: &[usize], targets: &[usize], n_classes: usize, avg: Average) -> f64 {
    let p = precision(predictions, targets, n_classes, avg);
    let r = recall(predictions, targets, n_classes, avg);
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

/// Per-class precision, recall, F1, and support — like sklearn's classification_report.
pub fn classification_report(
    predictions: &[usize],
    targets: &[usize],
    n_classes: usize,
) -> Vec<ClassMetrics> {
    let cm = ConfusionMatrix::from_predictions(predictions, targets, n_classes);
    (0..n_classes)
        .map(|c| {
            let tp = cm.tp(c) as f64;
            let fp = cm.fp(c) as f64;
            let fn_ = cm.fn_(c) as f64;
            let prec = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
            let rec = if tp + fn_ == 0.0 {
                0.0
            } else {
                tp / (tp + fn_)
            };
            let f1 = if prec + rec == 0.0 {
                0.0
            } else {
                2.0 * prec * rec / (prec + rec)
            };
            ClassMetrics {
                class: c,
                precision: prec,
                recall: rec,
                f1,
                support: cm.support(c),
            }
        })
        .collect()
}

/// Per-class metric report entry.
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    pub class: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: u64,
}

/// Top-K accuracy: fraction of samples where the true class is in the top-K predictions.
///
/// `scores` is a flat array of shape [n_samples, n_classes] with raw logits/probs.
pub fn top_k_accuracy(scores: &[f64], targets: &[usize], n_classes: usize, k: usize) -> f64 {
    let n_samples = targets.len();
    if n_samples == 0 {
        return 0.0;
    }
    let mut correct = 0usize;
    for i in 0..n_samples {
        let row = &scores[i * n_classes..(i + 1) * n_classes];
        // Get top-k indices
        let mut indexed: Vec<(usize, f64)> = row.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<usize> = indexed.iter().take(k).map(|(idx, _)| *idx).collect();
        if top_k.contains(&targets[i]) {
            correct += 1;
        }
    }
    correct as f64 / n_samples as f64
}

// Regression metrics

/// R² (coefficient of determination).
///
/// R² = 1 - SS_res / SS_tot, where:
///   SS_res = sum((y_true - y_pred)²)
///   SS_tot = sum((y_true - mean(y_true))²)
pub fn r2_score(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = targets.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean_y: f64 = targets.iter().sum::<f64>() / n;
    let ss_tot: f64 = targets.iter().map(|y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = targets
        .iter()
        .zip(predictions.iter())
        .map(|(y, p)| (y - p).powi(2))
        .sum();
    if ss_tot == 0.0 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Mean Absolute Error: mean(|y_true - y_pred|).
pub fn mae(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = targets.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    targets
        .iter()
        .zip(predictions.iter())
        .map(|(y, p)| (y - p).abs())
        .sum::<f64>()
        / n
}

/// Root Mean Squared Error: sqrt(mean((y_true - y_pred)²)).
pub fn rmse(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = targets.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mse: f64 = targets
        .iter()
        .zip(predictions.iter())
        .map(|(y, p)| (y - p).powi(2))
        .sum::<f64>()
        / n;
    mse.sqrt()
}

/// Mean Absolute Percentage Error: mean(|y_true - y_pred| / |y_true|) * 100.
pub fn mape(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = targets.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    targets
        .iter()
        .zip(predictions.iter())
        .filter(|(y, _)| **y != 0.0)
        .map(|(y, p)| ((y - p) / y).abs())
        .sum::<f64>()
        / n
        * 100.0
}

// Language model metrics

/// Perplexity from cross-entropy loss: exp(loss).
///
/// Lower perplexity = better language model.
pub fn perplexity(cross_entropy_loss: f64) -> f64 {
    cross_entropy_loss.exp()
}

/// Perplexity from a flat array of per-token log-probabilities.
///
/// PPL = exp(-1/N * sum(log_probs))
pub fn perplexity_from_log_probs(log_probs: &[f64]) -> f64 {
    let n = log_probs.len() as f64;
    if n == 0.0 {
        return f64::INFINITY;
    }
    let avg_neg_log_prob = -log_probs.iter().sum::<f64>() / n;
    avg_neg_log_prob.exp()
}

// Tensor-level helpers

/// Compute argmax along the last axis, returning class indices.
///
/// Input: [batch, n_classes] logits/probabilities.
/// Output: Vec of length `batch` with predicted class indices.
pub fn argmax_classes<B: Backend>(logits: &Tensor<B>) -> Result<Vec<usize>> {
    let data = logits.to_f64_vec()?;
    let dims = logits.dims();
    let n_classes = *dims.last().unwrap_or(&1);
    let batch = data.len() / n_classes;

    let mut classes = Vec::with_capacity(batch);
    for i in 0..batch {
        let row = &data[i * n_classes..(i + 1) * n_classes];
        let (max_idx, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        classes.push(max_idx);
    }
    Ok(classes)
}

/// Compute accuracy directly from logit tensors and one-hot/class-index targets.
///
/// - If `targets` has shape [batch, n_classes] (one-hot), takes argmax of both.
/// - If `targets` has shape [batch] or [batch, 1], treats as class indices.
pub fn tensor_accuracy<B: Backend>(logits: &Tensor<B>, targets: &Tensor<B>) -> Result<f64> {
    let pred_classes = argmax_classes(logits)?;
    let target_data = targets.to_f64_vec()?;
    let target_dims = targets.dims();
    let logit_dims = logits.dims();

    let target_classes: Vec<usize> =
        if target_dims.len() >= 2 && target_dims.last() == logit_dims.last() {
            // One-hot encoded targets — take argmax
            let n_classes = *target_dims.last().unwrap_or(&1);
            let batch = target_data.len() / n_classes;
            (0..batch)
                .map(|i| {
                    let row = &target_data[i * n_classes..(i + 1) * n_classes];
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                })
                .collect()
        } else {
            // Class indices
            target_data.iter().map(|v| *v as usize).collect()
        };

    Ok(accuracy(&pred_classes, &target_classes))
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        assert_eq!(accuracy(&[0, 1, 2, 0], &[0, 1, 2, 0]), 1.0);
    }

    #[test]
    fn test_accuracy_50_percent() {
        assert_eq!(accuracy(&[0, 1, 0, 1], &[0, 0, 1, 1]), 0.5);
    }

    #[test]
    fn test_confusion_matrix_binary() {
        // TP=2, FP=1, FN=1, TN=2
        let preds = [1, 1, 1, 0, 0, 0];
        let targets = [1, 1, 0, 0, 0, 1];
        let cm = ConfusionMatrix::from_predictions(&preds, &targets, 2);
        assert_eq!(cm.tp(1), 2); // true positive for class 1
        assert_eq!(cm.fp(1), 1); // false positive for class 1
        assert_eq!(cm.fn_(1), 1); // false negative for class 1
        assert_eq!(cm.tn(1), 2); // true negative for class 1
    }

    #[test]
    fn test_precision_recall_f1_binary() {
        let preds = [1, 1, 1, 0, 0, 0];
        let targets = [1, 1, 0, 0, 0, 1];
        let p = precision(&preds, &targets, 2, Average::Macro);
        let r = recall(&preds, &targets, 2, Average::Macro);
        let f = f1_score(&preds, &targets, 2, Average::Macro);
        assert!((p - 0.6667).abs() < 0.01);
        assert!((r - 0.6667).abs() < 0.01);
        assert!((f - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_precision_micro() {
        let preds = [0, 1, 2, 0, 1, 2];
        let targets = [0, 1, 2, 1, 0, 2];
        // micro precision = accuracy for multi-class
        let p = precision(&preds, &targets, 3, Average::Micro);
        let a = accuracy(&preds, &targets);
        assert!((p - a).abs() < 1e-10);
    }

    #[test]
    fn test_classification_report() {
        let preds = [0, 0, 1, 1, 2, 2];
        let targets = [0, 1, 1, 2, 2, 0];
        let report = classification_report(&preds, &targets, 3);
        assert_eq!(report.len(), 3);
        assert_eq!(report[0].support, 2); // class 0 has 2 true samples
    }

    #[test]
    fn test_r2_perfect() {
        let preds = [1.0, 2.0, 3.0, 4.0];
        let targets = [1.0, 2.0, 3.0, 4.0];
        assert!((r2_score(&preds, &targets) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mae_rmse() {
        let preds = [1.0, 2.0, 3.0];
        let targets = [1.0, 3.0, 5.0];
        assert!((mae(&preds, &targets) - 1.0).abs() < 1e-10);
        assert!(
            (rmse(&preds, &targets) - (2.0f64 / 1.0).sqrt() * (3.0f64 / 3.0).sqrt()).abs() < 0.2
        );
    }

    #[test]
    fn test_perplexity() {
        assert!((perplexity(0.0) - 1.0).abs() < 1e-10);
        assert!((perplexity(1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_top_k_accuracy() {
        // 2 samples, 3 classes
        let scores = [
            0.1, 0.7, 0.2, // sample 0: pred class 1
            0.8, 0.05, 0.15,
        ]; // sample 1: pred class 0, 2nd is class 2
        let targets = [1, 2];
        // top-1: only sample 0 correct
        assert!((top_k_accuracy(&scores, &targets, 3, 1) - 0.5).abs() < 1e-10);
        // top-2: both correct (class 2 is 2nd highest for sample 1)
        assert!((top_k_accuracy(&scores, &targets, 3, 2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_argmax_classes() {
        use shrew_cpu::{CpuBackend, CpuDevice};
        let t = Tensor::<CpuBackend>::from_f64_slice(
            &[
                0.1, 0.9, 0.3, // class 1
                0.8, 0.1, 0.1,
            ], // class 0
            vec![2, 3],
            shrew_core::DType::F64,
            &CpuDevice,
        )
        .unwrap();
        let classes = argmax_classes(&t).unwrap();
        assert_eq!(classes, vec![1, 0]);
    }
}
