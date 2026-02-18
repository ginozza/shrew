# shrew-nn

Neural Network primitives and layers for Shrew.

Contains high-level modules like:
- Linear / Dense
- Conv2d
- LSTM / GRU
- MultiHeadAttention
- LayerNorm / BatchNorm

Usage example:
```rust
let layer = Linear::new(768, 256);
let output = layer.forward(&input);
```

## License

Apache-2.0
