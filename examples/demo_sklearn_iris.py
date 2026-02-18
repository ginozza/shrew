#!/usr/bin/env python3
"""
=============================================================================
 Shrew + Scikit-Learn Demo — Iris Classification with .sw Model
=============================================================================

Demonstrates:
  1. Loading a real dataset from scikit-learn (Iris: 150 samples, 4 features, 3 classes)
  2. Preprocessing: train/test split + feature standardization
  3. Loading a .sw model definition (iris_classifier.sw)
  4. Training the model using Shrew's Python API (Adam optimizer)
  5. Evaluating with Shrew's built-in metrics: accuracy, precision, recall,
     F1 score, confusion matrix, and per-class classification report

Usage:
  python examples/demo_sklearn_iris.py
"""

import time
import math
import shrew_python as shrew
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & preprocess the Iris dataset
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  SHREW + SCIKIT-LEARN DEMO — Iris Classification")
print("=" * 70)

iris = load_iris()
X, y = iris.data, iris.target      # (150, 4), (150,)  3 classes
class_names = iris.target_names     # ['setosa', 'versicolor', 'virginica']

# Train/test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (zero-mean, unit-variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nDataset: Iris ({len(iris.data)} samples, {X.shape[1]} features, {len(class_names)} classes)")
print(f"Classes: {list(class_names)}")
print(f"Train:   {len(X_train)} samples")
print(f"Test:    {len(X_test)} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load the .sw model
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-" * 70)
print("  Loading .sw model...")
print("-" * 70)

executor = shrew.Executor.load("examples/iris_classifier.sw")
print(f"Model:       {executor}")
print(f"Parameters:  {executor.param_count('forward')}")
print(f"Inputs:      {executor.input_names('forward')}")
print(f"Outputs:     {executor.output_names('forward')}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Convert data to Shrew tensors
# ─────────────────────────────────────────────────────────────────────────────

n_train = len(X_train)
n_test  = len(X_test)
n_classes = 3

# Flatten numpy arrays to lists for Shrew
X_train_t = shrew.Tensor.from_list(X_train.flatten().tolist(), [n_train, 4])
X_test_t  = shrew.Tensor.from_list(X_test.flatten().tolist(),  [n_test, 4])

# One-hot encode targets for cross-entropy
def one_hot(labels, n_classes):
    """Convert integer labels to one-hot encoding."""
    result = []
    for l in labels:
        row = [0.0] * n_classes
        row[int(l)] = 1.0
        result.extend(row)
    return result

y_train_onehot = shrew.Tensor.from_list(one_hot(y_train, n_classes), [n_train, n_classes])
y_test_onehot  = shrew.Tensor.from_list(one_hot(y_test, n_classes),  [n_test, n_classes])

print(f"\nTensor shapes:")
print(f"  X_train: {X_train_t.shape}   y_train: {y_train_onehot.shape}")
print(f"  X_test:  {X_test_t.shape}   y_test:  {y_test_onehot.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train the model
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-" * 70)
print("  Training with Adam optimizer...")
print("-" * 70)

# Collect trainable parameters
params_dict = executor.named_params()
param_names = list(params_dict.keys())
param_tensors = list(params_dict.values())

# Adam optimizer
lr = 0.01
adam = shrew.Adam(param_tensors, lr=lr)

epochs = 300
log_every = 50

t0 = time.perf_counter()

for epoch in range(1, epochs + 1):
    # Forward pass
    outputs = executor.run("forward", {"x": X_train_t})
    logits = outputs["out"]

    # Cross-entropy loss
    loss = shrew.cross_entropy_loss(logits, y_train_onehot)

    # Backward pass
    grads = loss.backward()

    # Optimizer step
    adam.step(grads)

    # Update executor params from the optimized tensors
    for name, new_val in zip(param_names, param_tensors):
        executor.set_param(name, new_val)

    if epoch % log_every == 0 or epoch == 1:
        loss_val = loss.to_list()[0]
        # Quick training accuracy
        pred_classes = shrew.argmax_classes(logits)
        train_acc = shrew.accuracy(pred_classes, y_train.tolist())
        print(f"  Epoch {epoch:>4d}/{epochs}  |  loss: {loss_val:.6f}  |  train_acc: {train_acc:.4f}")

elapsed = time.perf_counter() - t0
print(f"\nTraining completed in {elapsed:.2f}s ({epochs} epochs)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-" * 70)
print("  Evaluation on test set")
print("-" * 70)

# Forward pass on test data
test_outputs = executor.run("forward", {"x": X_test_t})
test_logits = test_outputs["out"]

# Get predicted classes
pred_classes = shrew.argmax_classes(test_logits)
true_classes = y_test.tolist()

# --- Metrics ---

acc = shrew.accuracy(pred_classes, true_classes)
print(f"\n  Accuracy:           {acc:.4f}  ({int(acc * len(true_classes))}/{len(true_classes)} correct)")

# Tensor-level accuracy (alternative API)
tensor_acc = shrew.tensor_accuracy(test_logits, y_test_onehot)
print(f"  Tensor Accuracy:    {tensor_acc:.4f}")

# Precision, Recall, F1 (macro average)
prec_macro = shrew.precision(pred_classes, true_classes, n_classes, "macro")
rec_macro  = shrew.recall(pred_classes, true_classes, n_classes, "macro")
f1_macro   = shrew.f1_score(pred_classes, true_classes, n_classes, "macro")
print(f"\n  Precision (macro):  {prec_macro:.4f}")
print(f"  Recall    (macro):  {rec_macro:.4f}")
print(f"  F1 Score  (macro):  {f1_macro:.4f}")

# Weighted average
prec_w = shrew.precision(pred_classes, true_classes, n_classes, "weighted")
rec_w  = shrew.recall(pred_classes, true_classes, n_classes, "weighted")
f1_w   = shrew.f1_score(pred_classes, true_classes, n_classes, "weighted")
print(f"\n  Precision (weighted): {prec_w:.4f}")
print(f"  Recall    (weighted): {rec_w:.4f}")
print(f"  F1 Score  (weighted): {f1_w:.4f}")

# Per-class report
print(f"\n  {'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
print(f"  {'-'*60}")
report = shrew.classification_report(pred_classes, true_classes, n_classes)
for cls_id, prec, rec, f1, support in report:
    name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
    print(f"  {name:<20s} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10d}")

# Confusion matrix
print(f"\n  Confusion Matrix:")
cm = shrew.confusion_matrix(pred_classes, true_classes, n_classes)
header = "  " + f"{'':>12s}" + "".join(f"{'Pred ' + class_names[c]:>12s}" for c in range(n_classes))
print(header)
for r in range(n_classes):
    row_str = "  " + f"{'True ' + class_names[r]:>12s}"
    for c in range(n_classes):
        row_str += f"{cm[r][c]:>12d}"
    print(row_str)

# Perplexity (from final training loss)
final_loss = loss.to_list()[0]
ppl = shrew.perplexity(final_loss)
print(f"\n  Perplexity (from final loss): {ppl:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print(f"  SUMMARY")
print(f"    Model:      iris_classifier.sw (4→16→16→3)")
print(f"    Dataset:    Iris ({n_train} train / {n_test} test)")
print(f"    Optimizer:  Adam (lr={lr})")
print(f"    Epochs:     {epochs}")
print(f"    Test Acc:   {acc:.2%}")
print(f"    F1 (macro): {f1_macro:.4f}")
print(f"    Time:       {elapsed:.2f}s")
print("=" * 70)
