# Tensors & Types

Shrew is a strongly-typed language designed for tensor operations. The core type is the `Tensor`.

## Tensor Type

The syntax for a tensor is:
`Tensor<[Dimensions], DataType>`

### Dimensions
Dimensions can be:
- **Named Symbolic**: `Batch`, `Channels` (inferred at runtime)
- **Fixed Integer**: `224`, `1024`
- **Inferred**: `?` (unknown rank or dimension)

### Data Types (dtype)
Supported dtypes:
- Floating point: `f16`, `bf16`, `f32`, `f64`
- Integer: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- Boolean: `bool`
- Complex: `complex64`, `complex128`

## Examples

```sw
// A 2D matrix of shape [32, 128] with float32 elements
Tensor<[32, 128], f32>

// A generic batch of images
Tensor<[Batch, 3, Height, Width], f32>

// A vector of 10 integers
Tensor<[10], i32>
```

## Literals

You can define constant tensors using double brackets `[[ ... ]]`:

```sw
// 1D tensor
[[1, 2, 3]]

// 2D tensor
[[
    [1, 0],
    [0, 1]
]]
```
