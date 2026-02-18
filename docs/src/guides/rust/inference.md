# Running Inference

This guide shows how to load a model and run a forward pass using the Rust API.

## Full Example

```rust
use shrew::prelude::*;
use anyhow::Result;

fn main() -> Result<()> {
    // 1. Load the model from a .sw file
    let model = Model::load("models/linear_regression.sw")?;

    // 2. Prepare input tensor
    // Shape: [Batch=1, Input=3]
    let input_data = vec![1.0, 2.0, 3.0];
    let input = Tensor::new(&[1, 3], input_data)
        .to_device(Device::Cpu)?;

    // 3. Run inference
    // The Input ID "x" matches the `input x:` declaration in the .sw file
    let outputs = model.forward(hashmap! {
        "x" => input
    })?;

    // 4. Get output
    let result = outputs.get("y").expect("Output 'y' not found");
    println!("Result: {:?}", result);

    Ok(())
}
```

## Key Components

- **`Model::load`**: Parses and compiles the `.sw` file.
- **`Tensor::new`**: Creates a tensor from a shape and a flat data vector.
- **`Device`**: specifices where the tensor lives (`Cpu`, `Cuda(0)`, etc.).
