# shrew-cli

Command-line interface for inspecting, validating, and benchmarking `.sw` deep learning programs.

## Usage

```bash
# Print the lowered IR graph
shrew dump model.sw

# Validate a .sw program
shrew validate model.sw

# Benchmark forward pass
shrew bench model.sw --batch 32 --dtype f32

# Show model summary
shrew info model.sw
```

## License

Apache-2.0
