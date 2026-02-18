# Rust API Setup

To use Shrew in your Rust project, add the dependencies to your `Cargo.toml`.

## Dependencies

```toml
[dependencies]
shrew = "0.1"
anyhow = "1.0" # Recommended for error handling
```

## Feature Flags

- **`cuda`**: Enable CUDA backend support.
- **`python`**: Enable Python bindings.

```toml
[dependencies]
shrew = { version = "0.1", features = ["cuda"] }
```
