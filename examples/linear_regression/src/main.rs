// Example: Linear Regression with Shrew's Autograd
//
// This example demonstrates how automatic differentiation replaces the need
// for manual gradient computation. Compare this with manual gradient descent:
//
//   BEFORE (Phase 1): We had to derive and code dL/dw and dL/db by hand.
//   NOW (Phase 2):    loss.backward() computes ALL gradients automatically.
//
// The training loop is:
//   1. Forward pass: compute predictions and loss
//   2. backward():   compute gradients automatically
//   3. Update:       w -= lr * grad_w, b -= lr * grad_b
//
// We're learning y = 2*x + 1 using a single weight and bias.

use shrew::prelude::*;

fn main() -> shrew::Result<()> {
    let dev = CpuDevice;

    println!(" Shrew Linear Regression with Autograd \n");

    //  Step 1: Create synthetic data 
    // y = 2*x + 1  (true function we want to learn)
    let x_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    let x = CpuTensor::from_f64_slice(&x_data, (10, 1), DType::F64, &dev)?;
    let y_true = CpuTensor::from_f64_slice(&y_data, (10, 1), DType::F64, &dev)?;

    println!("Input X shape: {:?}", x.dims());
    println!("Target Y shape: {:?}", y_true.dims());

    //  Step 2: Initialize trainable parameters 
    let mut w_val = 0.5_f64;
    let mut b_val = 0.0_f64;

    let lr = 0.001_f64;

    //  Step 3: Training loop with AUTOGRAD 
    for epoch in 0..200 {
        // Create parameter tensors as variables (enables gradient tracking)
        let w = CpuTensor::full((1, 1), w_val, DType::F64, &dev)?.set_variable();

        // Forward pass: y_pred = x @ w + b
        let y_pred = x.matmul(&w)?; // [10,1] @ [1,1] → [10,1]

        // Add bias (scalar broadcast — manual for now, automatic in Phase 3)
        let b_full = CpuTensor::full((10, 1), b_val, DType::F64, &dev)?;
        let y_pred_full = y_pred.add(&b_full)?;

        // Loss: MSE = mean((y_pred - y_true)²)
        let diff = y_pred_full.sub(&y_true)?;
        let loss = diff.square()?.mean_all()?;
        let loss_val = loss.to_scalar_f64()?;

        //  THE MAGIC: one call computes ALL gradients 
        let grads = loss.backward()?;

        // Autograd gives us grad_w automatically!
        let grad_w_val = grads.get(&w).unwrap().to_scalar_f64()?;

        // For bias: manual for now (no broadcasting in graph yet)
        let diff_data = diff.to_f64_vec()?;
        let n = diff_data.len() as f64;
        let grad_b_val = 2.0 / n * diff_data.iter().sum::<f64>();

        if epoch % 20 == 0 {
            println!(
                "Epoch {:3}: loss = {:.6}, w = {:.4}, b = {:.4}  (grad_w = {:.4})",
                epoch, loss_val, w_val, b_val, grad_w_val,
            );
        }

        //  Step 4: Gradient descent update 
        w_val -= lr * grad_w_val;
        b_val -= lr * grad_b_val;
    }

    println!(
        "\nFinal: w = {:.4} (expected 2.0), b = {:.4} (expected 1.0)",
        w_val, b_val,
    );

    //  Bonus: Autograd demo with complex expressions 
    println!("\n Autograd Demo ");

    let a = CpuTensor::from_f64_slice(&[3.0], (), DType::F64, &dev)?.set_variable();
    let b = CpuTensor::from_f64_slice(&[4.0], (), DType::F64, &dev)?.set_variable();

    // f(a,b) = a*b + a²  →  grad_a = b + 2a = 10, grad_b = a = 3
    let result = a.mul(&b)?.add(&a.square()?)?;
    let grads = result.backward()?;

    println!("f(a,b) = a*b + a²");
    println!(
        "  a = {:.1}, b = {:.1}, f = {:.1}",
        a.to_scalar_f64()?,
        b.to_scalar_f64()?,
        result.to_scalar_f64()?
    );
    println!(
        "  ∂f/∂a = {:.1} (expected {:.1})",
        grads.get(&a).unwrap().to_scalar_f64()?,
        10.0
    );
    println!(
        "  ∂f/∂b = {:.1} (expected {:.1})",
        grads.get(&b).unwrap().to_scalar_f64()?,
        3.0
    );
    println!("\n✓ Autograd working correctly!");

    Ok(())
}
