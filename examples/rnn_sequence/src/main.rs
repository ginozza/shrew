// RNN Sequence Prediction — Sine Wave Forecasting
//
// This example trains a vanilla RNN to predict the next value in a sine wave
// sequence. It's a classic demonstration of recurrent networks learning
// temporal patterns.
//
// Architecture: RNN(input=1, hidden=32) → Linear(32, 1)
//
// The network sees a window of `SEQ_LEN` past values and predicts the next one.
// After training, the loss should decrease as the RNN learns the sine pattern.
//
// This example demonstrates:
//   1. Creating synthetic time-series data
//   2. Using the RNN module for sequence processing
//   3. Combining RNN hidden output with a Linear readout layer
//   4. Training with Adam optimizer
//   5. Using nn::init for custom initialization

use shrew::nn::init;
use shrew::prelude::*;

const SEQ_LEN: usize = 20; // Input sequence length
const HIDDEN_SIZE: usize = 32; // RNN hidden dimension
const NUM_SAMPLES: usize = 200; // Number of training sequences
const EPOCHS: usize = 100; // Training epochs
const LR: f64 = 0.005; // Learning rate

fn main() -> shrew::Result<()> {
    let dev = CpuDevice;
    let dtype = DType::F64;

    println!("=== Shrew — RNN Sine Wave Prediction ===");
    println!();
    println!(
        "Architecture: RNN(1→{}) → Linear({}→1)",
        HIDDEN_SIZE, HIDDEN_SIZE
    );
    println!("Sequence length: {}, Samples: {}", SEQ_LEN, NUM_SAMPLES);
    println!();

    // =========================================================================
    // 1. Generate synthetic sine-wave data
    // =========================================================================
    //
    // We sample points from sin(x) at regular intervals. Each training sample
    // is a window of SEQ_LEN consecutive values → next value prediction.

    let total_points = NUM_SAMPLES + SEQ_LEN + 1;
    let sine_data: Vec<f64> = (0..total_points).map(|i| (i as f64 * 0.1).sin()).collect();

    // Build input sequences [NUM_SAMPLES, SEQ_LEN, 1] and targets [NUM_SAMPLES, 1]
    let mut x_data = Vec::with_capacity(NUM_SAMPLES * SEQ_LEN);
    let mut y_data = Vec::with_capacity(NUM_SAMPLES);

    for i in 0..NUM_SAMPLES {
        for j in 0..SEQ_LEN {
            x_data.push(sine_data[i + j]);
        }
        y_data.push(sine_data[i + SEQ_LEN]); // target = next value
    }

    let x = CpuTensor::from_f64_slice(&x_data, (NUM_SAMPLES, SEQ_LEN, 1), dtype, &dev)?;
    let y = CpuTensor::from_f64_slice(&y_data, (NUM_SAMPLES, 1), dtype, &dev)?;

    println!("Input shape:  {:?}", x.dims());
    println!("Target shape: {:?}", y.dims());
    println!();

    // =========================================================================
    // 2. Create the model: RNN + Linear readout
    // =========================================================================

    let rnn = RNN::new(1, HIDDEN_SIZE, true, dtype, &dev)?;

    // Initialize readout layer with Xavier for better convergence
    let w_out = init::xavier_uniform::<CpuBackend>((1, HIDDEN_SIZE), 1.0, dtype, &dev)?;
    let b_out = init::zeros::<CpuBackend>((1, 1), dtype, &dev)?;

    // Collect all parameters
    let mut params = rnn.parameters();
    params.push(w_out.clone());
    params.push(b_out.clone());

    println!("Model parameters:");
    let total_params: usize = params.iter().map(|p| p.elem_count()).sum();
    println!("  Total: {} parameters", total_params);
    println!();

    // =========================================================================
    // 3. Create optimizer
    // =========================================================================

    let mut optimizer: Adam<CpuBackend> = Adam::new(params, LR);

    // =========================================================================
    // 4. Training loop
    // =========================================================================

    println!("Training for {} epochs...", EPOCHS);
    println!("{:>5}  {:>12}", "Epoch", "MSE Loss");
    println!("{:>5}  {:>12}", "-----", "--------");

    for epoch in 1..=EPOCHS {
        // Forward pass
        let (output, _h_n) = rnn.forward(&x, None)?;

        // Take the last hidden state from the sequence: [batch, hidden_size]
        let h_last = output
            .narrow(1, SEQ_LEN - 1, 1)?
            .reshape((NUM_SAMPLES, HIDDEN_SIZE))?;

        // Readout: y_pred = h_last @ w_out^T + b_out → [batch, 1]
        let w_out_t = w_out.t()?.contiguous()?;
        let y_pred = h_last.matmul(&w_out_t)?.add(&b_out)?;

        // MSE loss
        let diff = y_pred.sub(&y)?;
        let loss = diff.mul(&diff)?.mean_all()?;

        // Backward + update
        let grads = loss.backward()?;
        optimizer.step(&grads)?;

        // Print progress
        if epoch == 1 || epoch % 10 == 0 || epoch == EPOCHS {
            let loss_val = loss.to_scalar_f64()?;
            println!("{:>5}  {:>12.6}", epoch, loss_val);
        }
    }

    // =========================================================================
    // 5. Evaluation — show some predictions
    // =========================================================================

    println!();
    println!("Sample predictions (last 5 training samples):");
    println!("{:>8}  {:>8}  {:>8}", "True", "Pred", "Error");
    println!("{:>8}  {:>8}  {:>8}", "----", "----", "-----");

    // Re-run forward to get final predictions
    let (output, _) = rnn.forward(&x, None)?;
    let h_last = output
        .narrow(1, SEQ_LEN - 1, 1)?
        .reshape((NUM_SAMPLES, HIDDEN_SIZE))?;
    let w_out_t = w_out.t()?.contiguous()?;
    let y_pred = h_last.matmul(&w_out_t)?.add(&b_out)?;
    let pred_vals = y_pred.to_f64_vec()?;
    let true_vals = y.to_f64_vec()?;

    for i in (NUM_SAMPLES - 5)..NUM_SAMPLES {
        let err = (true_vals[i] - pred_vals[i]).abs();
        println!("{:>8.4}  {:>8.4}  {:>8.4}", true_vals[i], pred_vals[i], err);
    }

    println!();
    println!("Done! The RNN has learned to predict the next sine wave value.");

    Ok(())
}
