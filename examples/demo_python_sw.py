"""
Shrew v1.0 — Python + .sw Integration Demo
=============================================

This demo shows the killer feature of Shrew: defining a model architecture
in Shrew's declarative .sw language, then loading it from Python with full
tensor interoperability — run forward passes, inspect params, feed real data,
and even train the .sw model using Shrew's Python optimizers.

Architecture (defined in demo_mlp.sw):
    input  x: [4, 2]
    hidden:   matmul(x, w1) + b1 → relu    → [4, 8]
    output:   matmul(h, w2) + b2 → sigmoid → [4, 1]
"""
import shrew_python as shrew
import time

print("=" * 65)
print("  SHREW v1.0 — Python + .sw Model Integration Demo")
print("=" * 65)

# ═════════════════════════════════════════════════════════════════
# PART 1: Load the .sw model from Python
# ═════════════════════════════════════════════════════════════════
print("\n╔══ PART 1: Loading .sw model ══╗")

executor = shrew.Executor.load("examples/demo_mlp.sw", dtype="f64")
print(f"  {executor}")
print(f"  Graphs:  {executor.graph_names()}")
print(f"  Inputs:  {executor.input_names('Forward')}")
print(f"  Outputs: {executor.output_names('Forward')}")
print(f"  Params:  {executor.param_count('Forward')}")

# ═════════════════════════════════════════════════════════════════
# PART 2: Inspect model parameters (initialized from .sw specs)
# ═════════════════════════════════════════════════════════════════
print("\n╔══ PART 2: Model parameters ══╗")

params = executor.named_params()
for name, tensor in sorted(params.items()):
    print(f"  {name:20s}  shape={tensor.shape}  dtype={tensor.dtype}")

total_params = sum(
    1
    for p in params.values()
    for _ in range(1)  # count tensors
)
total_elements = 0
for p in params.values():
    n = 1
    for d in p.shape:
        n *= d
    total_elements += n
print(f"  Total: {total_params} parameter tensors, {total_elements} elements")

# ═════════════════════════════════════════════════════════════════
# PART 3: Forward pass with Python-created data
# ═════════════════════════════════════════════════════════════════
print("\n╔══ PART 3: Forward pass ══╗")

# XOR dataset created in Python
xor_data = shrew.Tensor.from_list(
    [0.0, 0.0,   # [0,0]
     0.0, 1.0,   # [0,1]
     1.0, 0.0,   # [1,0]
     1.0, 1.0],  # [1,1]
    [4, 2], dtype="f64"
)
xor_targets = shrew.Tensor.from_list([0.0, 1.0, 1.0, 0.0], [4, 1], dtype="f64")

print(f"  Input x: shape={xor_data.shape}")
print(f"  Targets: {xor_targets.to_list()}")

# Run the .sw graph
result = executor.run("Forward", {"x": xor_data})
output = result["out"]
print(f"  Output:  {[round(v, 4) for v in output.to_list()]}")
print(f"  (Random weights — not trained yet)")

# ═════════════════════════════════════════════════════════════════
# PART 4: Train the .sw model using Python optimizer
# ═════════════════════════════════════════════════════════════════
print("\n╔══ PART 4: Training .sw model from Python ══╗")
print("  Using Adam optimizer to train XOR...")
print()

# Build a trainable executor (training=True)
exec_train = shrew.Executor.load("examples/demo_mlp.sw", dtype="f64", training=True)

# Collect all model parameters from the .sw model
model_params = list(exec_train.named_params().values())
optimizer = shrew.Adam(model_params, lr=0.05)

t0 = time.time()
epochs = 500
for epoch in range(epochs):
    # Forward: run the .sw graph
    result = exec_train.run("Forward", {"x": xor_data})
    pred = result["out"]

    # Loss: MSE computed in Python
    loss = shrew.mse_loss(pred, xor_targets)

    # Backward: autograd through the entire .sw computation graph
    grads = loss.backward()

    # Step: update .sw model params via Python optimizer
    optimizer.step(grads)

    if epoch % 100 == 0 or epoch == epochs - 1:
        loss_val = loss.item()
        preds = [round(v, 3) for v in pred.to_list()]
        print(f"  Epoch {epoch:4d}  Loss: {loss_val:.6f}  Preds: {preds}")

elapsed = time.time() - t0

# ═════════════════════════════════════════════════════════════════
# PART 5: Final evaluation
# ═════════════════════════════════════════════════════════════════
print(f"\n╔══ PART 5: Results ({elapsed:.2f}s) ══╗")

result = exec_train.run("Forward", {"x": xor_data})
pred = result["out"]
pred_list = pred.to_list()

print(f"  {'Input':>10s}  {'Predicted':>10s}  {'Target':>8s}  {'Correct':>8s}")
print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")
labels = [[0,0], [0,1], [1,0], [1,1]]
targets = [0, 1, 1, 0]
all_correct = True
for inp, p, t in zip(labels, pred_list, targets):
    correct = (round(p) == t)
    all_correct = all_correct and correct
    mark = "✓" if correct else "✗"
    print(f"  {str(inp):>10s}  {p:>10.4f}  {t:>8d}  {mark:>8s}")

print()
if all_correct:
    print("  ★ XOR learned successfully!")
else:
    print("  Training may need more epochs.")

# ═════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  Summary:")
print("  • Model architecture defined in .sw (declarative)")
print("  • Data created in Python")
print("  • Forward pass through .sw computation graph")
print("  • Autograd computes gradients through the entire .sw graph")
print("  • Python optimizer updates .sw model parameters")
print("  • Full training loop: Python ←→ .sw interop")
print("=" * 65)
