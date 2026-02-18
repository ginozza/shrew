# shrew-python

Python bindings for Shrew via PyO3.

Allows using Shrew models directly from Python.

## Usage

```python
import shrew

model = shrew.load("model.sw")
output = model.forward(input_tensor)
```

## License

Apache-2.0
