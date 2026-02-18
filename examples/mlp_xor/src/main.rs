// MLP XOR Example — Training a Multi-Layer Perceptron from scratch
//
// XOR is the classic problem that demonstrates why we NEED hidden layers.
// A single linear layer cannot learn XOR because it's not linearly separable.
// A 2-layer MLP (with a nonlinear activation like ReLU) CAN learn it.
//
// Architecture: Input(2) → Linear(2,16) → ReLU → Linear(16,1) → Output
//
// This example demonstrates:
//   1. Creating layers with Module trait
//   2. Manual forward pass through the network
//   3. Computing MSE loss
//   4. Backpropagation with loss.backward()
//   5. Parameter update with Adam optimizer
//   6. Full training loop with loss tracking

use shrew::nn::Module;
use shrew::prelude::*;

fn main() -> shrew::Result<()> {
    let dev = CpuDevice;

    println!("=== Shrew — MLP XOR Example ===");
    println!();

    // 1. Define the training data
    // XOR truth table:
    //   (0,0) → 0
    //   (0,1) → 1
    //   (1,0) → 1
    //   (1,1) → 0

    let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y_data = vec![0.0, 1.0, 1.0, 0.0];

    let x = CpuTensor::from_f64_slice(&x_data, (4, 2), DType::F64, &dev)?;
    let y = CpuTensor::from_f64_slice(&y_data, (4, 1), DType::F64, &dev)?;

    println!("Training data (XOR):");
    println!("  (0,0) → 0");
    println!("  (0,1) → 1");
    println!("  (1,0) → 1");
    println!("  (1,1) → 0");
    println!();

    // 2. Create the network layers
    // We use two linear layers with a ReLU activation in between.
    // The hidden layer has 16 neurons — more than enough for XOR.

    let l1 = Linear::<CpuBackend>::new(2, 16, true, DType::F64, &dev)?;
    let l2 = Linear::<CpuBackend>::new(16, 1, true, DType::F64, &dev)?;

    println!("Network: Linear(2→16) → ReLU → Linear(16→1)");
    println!(
        "  Layer 1: weight [{},{}] + bias [{},{}]",
        l1.weight().dims()[0],
        l1.weight().dims()[1],
        l1.bias().unwrap().dims()[0],
        l1.bias().unwrap().dims()[1]
    );
    println!(
        "  Layer 2: weight [{},{}] + bias [{},{}]",
        l2.weight().dims()[0],
        l2.weight().dims()[1],
        l2.bias().unwrap().dims()[0],
        l2.bias().unwrap().dims()[1]
    );

    let total_params: usize = l1
        .parameters()
        .iter()
        .chain(l2.parameters().iter())
        .map(|t| t.elem_count())
        .sum();
    println!("  Total parameters: {}", total_params);
    println!();

    // 3. Set up the optimizer
    let mut all_params: Vec<CpuTensor> = Vec::new();
    all_params.extend(l1.parameters());
    all_params.extend(l2.parameters());

    let mut optimizer = Adam::<CpuBackend>::new(all_params, 0.01);

    println!("Optimizer: Adam (lr=0.01, β1=0.9, β2=0.999)");
    println!();

    // 4. Training loop
    println!("Training for 500 epochs...");
    println!("{:-<50}", "");

    let epochs = 500;

    for epoch in 0..epochs {
        //  Forward pass
        // Layer 1: h = ReLU(x @ W1^T + b1)
        let h = {
            let w1 = &optimizer.params()[0]; // weight [16, 2]
            let b1 = &optimizer.params()[1]; // bias   [1, 16]
            let wt1 = w1.t()?.contiguous()?; // [2, 16]
            let out = x.matmul(&wt1)?; // [4, 16]

            // Expand bias to batch dimension
            let b1_data = b1.to_f64_vec()?;
            let expanded: Vec<f64> = (0..4).flat_map(|_| b1_data.iter().copied()).collect();
            let b1_exp = CpuTensor::from_f64_slice(&expanded, (4, 16), DType::F64, &dev)?;
            out.add(&b1_exp)?.relu()?
        };

        // Layer 2: y_pred = h @ W2^T + b2
        let y_pred = {
            let w2 = &optimizer.params()[2]; // weight [1, 16]
            let b2 = &optimizer.params()[3]; // bias   [1, 1]
            let wt2 = w2.t()?.contiguous()?; // [16, 1]
            let out = h.matmul(&wt2)?; // [4, 1]

            let b2_data = b2.to_f64_vec()?;
            let expanded: Vec<f64> = (0..4).flat_map(|_| b2_data.iter().copied()).collect();
            let b2_exp = CpuTensor::from_f64_slice(&expanded, (4, 1), DType::F64, &dev)?;
            out.add(&b2_exp)?
        };

        //  Compute loss
        let loss = mse_loss(&y_pred, &y)?;
        let loss_val = loss.to_scalar_f64()?;

        //  Backward pass
        let grads = loss.backward()?;

        //  Optimizer step
        optimizer.step(&grads)?;

        // Print progress every 50 epochs
        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!("  Epoch {:>4} | Loss: {:.6}", epoch, loss_val);
        }
    }

    println!("{:-<50}", "");
    println!();

    // 5. Evaluate the trained model
    println!("Predictions after training:");

    // Forward pass with final parameters
    let h = {
        let w1 = &optimizer.params()[0];
        let b1 = &optimizer.params()[1];
        let wt1 = w1.t()?.contiguous()?;
        let out = x.matmul(&wt1)?;
        let b1_data = b1.to_f64_vec()?;
        let expanded: Vec<f64> = (0..4).flat_map(|_| b1_data.iter().copied()).collect();
        let b1_exp = CpuTensor::from_f64_slice(&expanded, (4, 16), DType::F64, &dev)?;
        out.add(&b1_exp)?.relu()?
    };

    let y_pred = {
        let w2 = &optimizer.params()[2];
        let b2 = &optimizer.params()[3];
        let wt2 = w2.t()?.contiguous()?;
        let out = h.matmul(&wt2)?;
        let b2_data = b2.to_f64_vec()?;
        let expanded: Vec<f64> = (0..4).flat_map(|_| b2_data.iter().copied()).collect();
        let b2_exp = CpuTensor::from_f64_slice(&expanded, (4, 1), DType::F64, &dev)?;
        out.add(&b2_exp)?
    };

    let preds = y_pred.to_f64_vec()?;
    let inputs = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let targets = [0.0, 1.0, 1.0, 0.0];

    for (i, ((a, b), &target)) in inputs.iter().zip(targets.iter()).enumerate() {
        let pred = preds[i];
        let rounded = if pred > 0.5 { 1 } else { 0 };
        let correct = if rounded as f64 == target {
            "✓"
        } else {
            "✗"
        };
        println!(
            "  ({},{}) → {:.4}  (rounded: {})  target: {}  {}",
            a, b, pred, rounded, target as i32, correct
        );
    }

    println!();
    println!("=== Done! ===");

    Ok(())
}
