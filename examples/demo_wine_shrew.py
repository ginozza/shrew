#!/usr/bin/env python3
"""
Shrew — Wine Classification with Maximum Shrew, Minimum Python

Python only loads data (sklearn) and converts it to Shrew tensors.
ALL neural network computation happens inside Shrew:
    model definition (.sw), forward pass, loss, backward, optimizer, metrics

Dataset: sklearn Wine (178 samples, 13 features, 3 classes)

Usage:
    python examples/demo_wine_shrew.py
"""

import time
import shrew_python as shrew
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data loading
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Convert numpy → Shrew tensors
n_train, n_test, n_classes = len(X_train), len(X_test), 3

X_train_t = shrew.Tensor.from_list(X_train.flatten().tolist(), [n_train, 13])
X_test_t  = shrew.Tensor.from_list(X_test.flatten().tolist(),  [n_test, 13])


def one_hot(labels, c):
    out = []
    for l in labels:
        row = [0.0] * c
        row[int(l)] = 1.0
        out.extend(row)
    return out


y_train_oh = shrew.Tensor.from_list(one_hot(y_train, n_classes), [n_train, n_classes])
y_test_oh  = shrew.Tensor.from_list(one_hot(y_test, n_classes),  [n_test, n_classes])

# Everything below is Shrew-powered
print("=" * 65)
print("  Shrew — Wine Classification")
print("=" * 65)
print(f"\n  Dataset:  Wine ({n_train} train / {n_test} test, {n_classes} classes)")
print(f"  Classes:  {list(wine.target_names)}")

# 1. Load the .sw model
executor = shrew.Executor.load("examples/wine_classifier.sw")
print(f"\n  Model:    {executor}")
print(f"  Params:   {executor.param_count('forward')}")

# 2. Train
params_dict   = executor.named_params()
param_names   = list(params_dict.keys())
param_tensors = list(params_dict.values())
adam = shrew.Adam(param_tensors, lr=0.005, weight_decay=0.01)

epochs = 400
print(f"\n  Training ({epochs} epochs, Adam lr=0.005)")

t0 = time.perf_counter()

for epoch in range(1, epochs + 1):
    outputs = executor.run("forward", {"x": X_train_t})
    logits  = outputs["out"]

    loss  = shrew.cross_entropy_loss(logits, y_train_oh)
    grads = loss.backward()
    adam.step(grads)

    for name, val in zip(param_names, param_tensors):
        executor.set_param(name, val)

    if epoch % 100 == 0 or epoch == 1:
        lv = loss.to_list()[0]
        pc = shrew.argmax_classes(logits)
        acc = shrew.accuracy(pc, y_train.tolist())
        print(f"  Epoch {epoch:>4d}/{epochs}  |  loss: {lv:.6f}  |  train_acc: {acc:.4f}")

elapsed = time.perf_counter() - t0
print(f"\n  Trained in {elapsed:.2f}s")

# 3. Evaluate
print(f"\n  Test Evaluation")

test_out    = executor.run("forward", {"x": X_test_t})
test_logits = test_out["out"]

pred = shrew.argmax_classes(test_logits)
true = y_test.tolist()

acc   = shrew.accuracy(pred, true)
prec  = shrew.precision(pred, true, n_classes, "macro")
rec   = shrew.recall(pred, true, n_classes, "macro")
f1    = shrew.f1_score(pred, true, n_classes, "macro")

print(f"\n  Accuracy:          {acc:.4f}")
print(f"  Precision (macro): {prec:.4f}")
print(f"  Recall    (macro): {rec:.4f}")
print(f"  F1 Score  (macro): {f1:.4f}")

# Per-class report
names = wine.target_names
print(f"\n  {'Class':<15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
report = shrew.classification_report(pred, true, n_classes)
for cls_id, p, r, f, s in report:
    name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
    print(f"  {name:<15s} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10d}")

# Confusion matrix
cm = shrew.confusion_matrix(pred, true, n_classes)
print(f"\n  Confusion Matrix:")
hdr = "  " + f"{'':>12s}" + "".join(f"{'P-' + names[c]:>12s}" for c in range(n_classes))
print(hdr)
for r in range(n_classes):
    row = "  " + f"{'T-' + names[r]:>12s}"
    for c in range(n_classes):
        row += f"{cm[r][c]:>12d}"
    print(row)

print(f"\n{'=' * 65}")
print(f"  Python:  data loading + conversion only")
print(f"  Shrew:   model (.sw), forward, loss, backward, optimizer, metrics")
print(f"  Result:  {acc:.1%} accuracy in {elapsed:.2f}s")
print(f"{'=' * 65}")
