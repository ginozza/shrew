# Python Usage

The Python API provides an interface similar to the Rust API but in a more Pythonic way.

## Example

```python
import shrew
import numpy as np

# 1. Load the model
model = shrew.load("models/linear_regression.sw")

# 2. Prepare input (using NumPy)
x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

# 3. Run inference
# Inputs are passed as a dictionary mapping naming to NumPy arrays
outputs = model.forward({"x": x})

# 4. Get result
y = outputs["y"]
print("Output shape:", y.shape)
print("Output data:", y)
```
