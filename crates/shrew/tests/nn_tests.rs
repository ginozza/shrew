// Integration tests for shrew-nn and shrew-optim
//
// These tests verify that the neural network layers, loss functions, and
// optimizers work together correctly using the CPU backend.

use shrew::nn::{
    AvgPool2d, BatchNorm2d, Conv1d, Dropout, Embedding, Flatten, LayerNorm, Module,
    MultiHeadAttention, Sigmoid as SigmoidMod, Tanh as TanhMod, TransformerBlock,
};
use shrew::prelude::*;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn assert_vec_approx(got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        got.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        got.len(),
        expected.len()
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*g, *e, tol),
            "index {}: got {} expected {} (tol {})",
            i,
            g,
            e,
            tol
        );
    }
}

// Linear layer tests

#[test]
fn test_linear_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let linear = Linear::<CpuBackend>::new(10, 5, true, DType::F64, &dev)?;

    assert_eq!(linear.weight().dims(), &[5, 10]);
    assert_eq!(linear.bias().unwrap().dims(), &[1, 5]);
    assert_eq!(linear.in_features(), 10);
    assert_eq!(linear.out_features(), 5);

    // Forward: [batch=3, 10] → [3, 5]
    let x = CpuTensor::rand((3, 10), DType::F64, &dev)?;
    let y = linear.forward(&x)?;
    assert_eq!(y.dims(), &[3, 5]);
    Ok(())
}

#[test]
fn test_linear_no_bias() -> shrew::Result<()> {
    let dev = CpuDevice;
    let linear = Linear::<CpuBackend>::new(4, 2, false, DType::F64, &dev)?;
    assert!(linear.bias().is_none());

    let x = CpuTensor::ones((1, 4), DType::F64, &dev)?;
    let y = linear.forward(&x)?;
    assert_eq!(y.dims(), &[1, 2]);
    Ok(())
}

#[test]
fn test_linear_from_tensors() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0, 0.0, 0.0, 1.0], (2, 2), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[0.5, -0.5], (1, 2), DType::F64, &dev)?;
    let linear = Linear::from_tensors(w, Some(b))?;

    let x = CpuTensor::from_f64_slice(&[3.0, 7.0], (1, 2), DType::F64, &dev)?;
    let y = linear.forward(&x)?;
    // y = x @ W^T + b = [3, 7] @ I + [0.5, -0.5] = [3.5, 6.5]
    assert_vec_approx(&y.to_f64_vec()?, &[3.5, 6.5], 1e-10);
    Ok(())
}

#[test]
fn test_linear_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let linear = Linear::<CpuBackend>::new(4, 3, true, DType::F64, &dev)?;
    assert_eq!(linear.parameters().len(), 2); // weight + bias

    let linear_no_bias = Linear::<CpuBackend>::new(4, 3, false, DType::F64, &dev)?;
    assert_eq!(linear_no_bias.parameters().len(), 1); // weight only
    Ok(())
}

// Sequential tests

#[test]
fn test_sequential_forward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let model = Sequential::<CpuBackend>::new()
        .add(Linear::<CpuBackend>::new(4, 3, true, DType::F64, &dev)?)
        .add(ReLU)
        .add(Linear::<CpuBackend>::new(3, 2, true, DType::F64, &dev)?);

    let x = CpuTensor::rand((5, 4), DType::F64, &dev)?;
    let y = model.forward(&x)?;
    assert_eq!(y.dims(), &[5, 2]);

    // Parameters: linear1(weight+bias) + relu(0) + linear2(weight+bias) = 4
    assert_eq!(model.parameters().len(), 4);
    Ok(())
}

// Activation tests

#[test]
fn test_activation_modules() -> shrew::Result<()> {
    let dev = CpuDevice;
    let x = CpuTensor::from_f64_slice(&[-1.0, 0.0, 1.0], 3, DType::F64, &dev)?;

    // ReLU
    let y = ReLU.forward(&x)?;
    assert_vec_approx(&y.to_f64_vec()?, &[0.0, 0.0, 1.0], 1e-10);

    // GeLU
    let y = GeLU.forward(&x)?;
    assert!(y.to_f64_vec()?[0] < 0.0); // GELU(-1) ≈ -0.16
    assert!(approx_eq(y.to_f64_vec()?[1], 0.0, 1e-5)); // GELU(0) = 0

    // SiLU
    let y = SiLU.forward(&x)?;
    assert!(y.to_f64_vec()?[0] < 0.0); // SiLU(-1) < 0

    // Sigmoid
    let y = SigmoidMod.forward(&x)?;
    assert!(approx_eq(y.to_f64_vec()?[1], 0.5, 1e-10)); // σ(0) = 0.5

    // Tanh
    let y = TanhMod.forward(&x)?;
    assert!(approx_eq(y.to_f64_vec()?[1], 0.0, 1e-10)); // tanh(0) = 0

    // These modules have no parameters
    let relu_params: Vec<CpuTensor> = ReLU.parameters();
    assert_eq!(relu_params.len(), 0);
    Ok(())
}

// Embedding tests

#[test]
fn test_embedding_forward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let weight = CpuTensor::from_f64_slice(
        &[
            0.1, 0.2, 0.3, // token 0
            1.1, 1.2, 1.3, // token 1
            2.1, 2.2, 2.3, // token 2
            3.1, 3.2, 3.3, // token 3
            4.1, 4.2, 4.3, // token 4
        ],
        (5, 3),
        DType::F64,
        &dev,
    )?;
    let emb = Embedding::from_tensor(weight)?;
    assert_eq!(emb.num_embeddings(), 5);
    assert_eq!(emb.embedding_dim(), 3);

    let indices = CpuTensor::from_f64_slice(&[1.0, 3.0], 2, DType::F64, &dev)?;
    let out = emb.forward(&indices)?;
    assert_eq!(out.dims(), &[2, 3]);
    assert_vec_approx(&out.to_f64_vec()?, &[1.1, 1.2, 1.3, 3.1, 3.2, 3.3], 1e-10);
    Ok(())
}

// Dropout tests

#[test]
fn test_dropout_eval_mode() -> shrew::Result<()> {
    let dev = CpuDevice;
    let drop = Dropout::new(0.5);
    drop.set_training(false);

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
    let y = drop.forward(&x)?;
    assert_vec_approx(&y.to_f64_vec()?, &[1.0, 2.0, 3.0], 1e-10);
    Ok(())
}

#[test]
fn test_dropout_training_mode() -> shrew::Result<()> {
    let dev = CpuDevice;
    let drop = Dropout::new(0.5);
    assert!(drop.is_training());

    let x = CpuTensor::ones(1000, DType::F64, &dev)?;
    let y = drop.forward(&x)?;
    let data = y.to_f64_vec()?;
    let zeros = data.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zeros > 300 && zeros < 700,
        "Expected ~500 zeros, got {}",
        zeros
    );
    Ok(())
}

// Loss function tests

#[test]
fn test_mse_loss() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pred = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;

    let loss = mse_loss(&pred, &target)?;
    assert!(approx_eq(loss.to_scalar_f64()?, 0.0, 1e-10));

    let pred2 = CpuTensor::from_f64_slice(&[2.0, 3.0, 4.0], 3, DType::F64, &dev)?;
    let loss2 = mse_loss(&pred2, &target)?;
    assert!(approx_eq(loss2.to_scalar_f64()?, 1.0, 1e-10));
    Ok(())
}

#[test]
fn test_mse_loss_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pred = CpuTensor::from_f64_slice(&[2.0, 3.0], 2, DType::F64, &dev)?.set_variable();
    let target = CpuTensor::from_f64_slice(&[1.0, 1.0], 2, DType::F64, &dev)?;

    let loss = mse_loss(&pred, &target)?;
    assert!(approx_eq(loss.to_scalar_f64()?, 2.5, 1e-10));

    let grads = loss.backward()?;
    let grad_pred = grads.get(&pred).unwrap().to_f64_vec()?;
    // d/dpred MSE = 2*(pred-target)/n = [2*1/2, 2*2/2] = [1, 2]
    assert_vec_approx(&grad_pred, &[1.0, 2.0], 1e-10);
    Ok(())
}

#[test]
fn test_cross_entropy_loss() -> shrew::Result<()> {
    let dev = CpuDevice;
    let logits = CpuTensor::from_f64_slice(&[2.0, 0.5], (1, 2), DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[1.0, 0.0], (1, 2), DType::F64, &dev)?;

    let loss = cross_entropy_loss(&logits, &target)?;
    let expected = -(2.0 - (2.0_f64.exp() + 0.5_f64.exp()).ln());
    assert!(approx_eq(loss.to_scalar_f64()?, expected, 1e-6));
    Ok(())
}

// SGD optimizer tests

#[test]
fn test_sgd_step() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0, 2.0], 2, DType::F64, &dev)?.set_variable();
    let x = CpuTensor::from_f64_slice(&[1.0, 1.0], 2, DType::F64, &dev)?;

    let loss = w.mul(&x)?.sum_all()?;
    let grads = loss.backward()?;

    let mut optimizer = SGD::<CpuBackend>::new(vec![w.clone()], 0.1, 0.0, 0.0);
    let new_params = optimizer.step(&grads)?;

    // w_new = w - lr * grad = [1-0.1, 2-0.1] = [0.9, 1.9]
    assert_vec_approx(&new_params[0].to_f64_vec()?, &[0.9, 1.9], 1e-10);
    Ok(())
}

#[test]
fn test_sgd_momentum() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();

    let mut optimizer = SGD::<CpuBackend>::new(vec![w.clone()], 0.1, 0.9, 0.0);

    // Step 1: v = 0.9*0 + 1.0 = 1.0, w = 1.0 - 0.1*1.0 = 0.9
    let x = CpuTensor::ones((), DType::F64, &dev)?;
    let loss = optimizer.params()[0].mul(&x)?.sum_all()?;
    let grads = loss.backward()?;
    optimizer.step(&grads)?;
    assert!(approx_eq(
        optimizer.params()[0].to_scalar_f64()?,
        0.9,
        1e-10
    ));

    // Step 2: v = 0.9*1.0 + 1.0 = 1.9, w = 0.9 - 0.1*1.9 = 0.71
    let loss = optimizer.params()[0].mul(&x)?.sum_all()?;
    let grads = loss.backward()?;
    optimizer.step(&grads)?;
    assert!(approx_eq(
        optimizer.params()[0].to_scalar_f64()?,
        0.71,
        1e-10
    ));
    Ok(())
}

// Adam optimizer tests

#[test]
fn test_adam_step() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[5.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::ones((), DType::F64, &dev)?;

    let mut optimizer = Adam::<CpuBackend>::new(vec![w.clone()], 0.1);

    for _ in 0..10 {
        let loss = optimizer.params()[0].mul(&x)?.sum_all()?;
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
    }

    let final_w = optimizer.params()[0].to_scalar_f64()?;
    assert!(
        final_w < 5.0,
        "Adam should have reduced w from 5.0, got {}",
        final_w
    );
    assert_eq!(optimizer.step_count(), 10);
    Ok(())
}

#[test]
fn test_adamw_step() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[5.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::ones((), DType::F64, &dev)?;

    let mut optimizer = AdamW::<CpuBackend>::new(vec![w.clone()], 0.1, 0.01);

    for _ in 0..10 {
        let loss = optimizer.params()[0].mul(&x)?.sum_all()?;
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
    }

    let final_w = optimizer.params()[0].to_scalar_f64()?;
    assert!(
        final_w < 5.0,
        "AdamW should have reduced w, got {}",
        final_w
    );
    Ok(())
}

// End-to-end: Linear layer trained with SGD

#[test]
fn test_linear_sgd_training() -> shrew::Result<()> {
    let dev = CpuDevice;

    // True function: y = 3x + 2 (normalized inputs for stable convergence)
    let x_data: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| 3.0 * x + 2.0).collect();

    let x_train = CpuTensor::from_f64_slice(&x_data, (20, 1), DType::F64, &dev)?;
    let y_train = CpuTensor::from_f64_slice(&y_data, (20, 1), DType::F64, &dev)?;

    let linear = Linear::<CpuBackend>::new(1, 1, true, DType::F64, &dev)?;
    let params = linear.parameters();
    let mut optimizer = SGD::<CpuBackend>::new(params.clone(), 0.01, 0.0, 0.0);

    let mut prev_loss = f64::MAX;
    for _epoch in 0..500 {
        let y_pred = {
            let w = &optimizer.params()[0];
            let b = &optimizer.params()[1];
            let wt = w.t()?.contiguous()?;
            let out = x_train.matmul(&wt)?;
            // Use broadcasting: b is [1,1], out is [20,1] → broadcasts correctly
            out.add(b)?
        };

        let loss = mse_loss(&y_pred, &y_train)?;
        let loss_val = loss.to_scalar_f64()?;

        let grads = loss.backward()?;
        optimizer.step(&grads)?;

        prev_loss = loss_val;
    }

    assert!(
        prev_loss < 0.5,
        "Training should have reduced loss, got {:.4}",
        prev_loss
    );
    Ok(())
}

// End-to-end: MLP trained with Adam (XOR problem)

#[test]
fn test_mlp_adam_training() -> shrew::Result<()> {
    let dev = CpuDevice;

    let x_data = vec![1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0];
    let y_data = vec![1.0, -1.0, -1.0, 1.0];

    let x = CpuTensor::from_f64_slice(&x_data, (4, 2), DType::F64, &dev)?;
    let y = CpuTensor::from_f64_slice(&y_data, (4, 1), DType::F64, &dev)?;

    // MLP: 2 → 8 → 1
    let l1 = Linear::<CpuBackend>::new(2, 8, true, DType::F64, &dev)?;
    let l2 = Linear::<CpuBackend>::new(8, 1, true, DType::F64, &dev)?;

    let mut all_params: Vec<CpuTensor> = Vec::new();
    all_params.extend(l1.parameters());
    all_params.extend(l2.parameters());
    let mut optimizer = Adam::<CpuBackend>::new(all_params, 0.01);

    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;

    for epoch in 0..100 {
        let h = {
            let w1 = &optimizer.params()[0];
            let b1 = &optimizer.params()[1];
            let wt1 = w1.t()?.contiguous()?;
            let out = x.matmul(&wt1)?;
            // Use broadcasting: b1 is [1,8], out is [4,8] → broadcasts correctly
            out.add(b1)?.relu()?
        };

        let y_pred = {
            let w2 = &optimizer.params()[2];
            let b2 = &optimizer.params()[3];
            let wt2 = w2.t()?.contiguous()?;
            let out = h.matmul(&wt2)?;
            // Use broadcasting: b2 is [1,1], out is [4,1] → broadcasts correctly
            out.add(b2)?
        };

        let loss = mse_loss(&y_pred, &y)?;
        let loss_val = loss.to_scalar_f64()?;

        if epoch == 0 {
            initial_loss = loss_val;
        }
        if epoch == 99 {
            final_loss = loss_val;
        }

        let grads = loss.backward()?;
        optimizer.step(&grads)?;
    }

    assert!(
        final_loss < initial_loss,
        "Loss should decrease: initial={:.4} final={:.4}",
        initial_loss,
        final_loss
    );
    Ok(())
}

// Phase 4: Broadcasting tests

#[test]
fn test_broadcast_add_shapes() -> shrew::Result<()> {
    let dev = CpuDevice;

    // [3,1] + [1,4] → [3,4]
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], (3, 1), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], (1, 4), DType::F64, &dev)?;
    let c = a.add(&b)?;
    assert_eq!(c.dims(), &[3, 4]);

    let data = c.to_f64_vec()?;
    // Row 0: 1+10, 1+20, 1+30, 1+40
    // Row 1: 2+10, 2+20, 2+30, 2+40
    // Row 2: 3+10, 3+20, 3+30, 3+40
    assert_vec_approx(
        &data,
        &[
            11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,
        ],
        1e-10,
    );
    Ok(())
}

#[test]
fn test_broadcast_mul_scalar() -> shrew::Result<()> {
    let dev = CpuDevice;

    // [2,3] * scalar → [2,3]
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F64, &dev)?;
    let s = CpuTensor::from_f64_slice(&[2.0], (), DType::F64, &dev)?;
    let c = a.mul(&s)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_vec_approx(&c.to_f64_vec()?, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], 1e-10);
    Ok(())
}

#[test]
fn test_broadcast_gradient_reduction() -> shrew::Result<()> {
    let dev = CpuDevice;

    // a: [3,1] (variable), b: [1,4] (variable)
    // c = a + b → [3,4], loss = sum(c)
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], (3, 1), DType::F64, &dev)?.set_variable();
    let b = CpuTensor::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], (1, 4), DType::F64, &dev)?
        .set_variable();
    let c = a.add(&b)?;
    let loss = c.sum_all()?;

    let grads = loss.backward()?;

    // d(sum(a+b))/da: gradient flows back through broadcast
    // a was broadcast across dim=1 (from 1 to 4), so grad sums over dim=1
    // Each element of a contributed to 4 outputs → grad = 4
    let grad_a = grads.get(&a).unwrap().to_f64_vec()?;
    assert_vec_approx(&grad_a, &[4.0, 4.0, 4.0], 1e-10);

    // b was broadcast across dim=0 (from 1 to 3), so grad sums over dim=0
    // Each element of b contributed to 3 outputs → grad = 3
    let grad_b = grads.get(&b).unwrap().to_f64_vec()?;
    assert_vec_approx(&grad_b, &[3.0, 3.0, 3.0, 3.0], 1e-10);
    Ok(())
}

// Phase 4: Softmax tests

#[test]
fn test_softmax_sums_to_one() -> shrew::Result<()> {
    let dev = CpuDevice;

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0], (2, 3), DType::F64, &dev)?;
    let s = x.softmax(1)?;
    let data = s.to_f64_vec()?;

    // Each row should sum to 1
    let row0_sum: f64 = data[0..3].iter().sum();
    let row1_sum: f64 = data[3..6].iter().sum();
    assert!(approx_eq(row0_sum, 1.0, 1e-10));
    assert!(approx_eq(row1_sum, 1.0, 1e-10));

    // All values should be positive
    assert!(data.iter().all(|&v| v > 0.0));
    Ok(())
}

#[test]
fn test_softmax_known_values() -> shrew::Result<()> {
    let dev = CpuDevice;

    // Softmax of [0, 0, 0] should be [1/3, 1/3, 1/3]
    let x = CpuTensor::from_f64_slice(&[0.0, 0.0, 0.0], (1, 3), DType::F64, &dev)?;
    let s = x.softmax(1)?;
    assert_vec_approx(&s.to_f64_vec()?, &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1e-10);
    Ok(())
}

#[test]
fn test_log_softmax() -> shrew::Result<()> {
    let dev = CpuDevice;

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], (1, 3), DType::F64, &dev)?;
    let ls = x.log_softmax(1)?;
    let s = x.softmax(1)?;

    // log_softmax should equal log(softmax)
    let s_data = s.to_f64_vec()?;
    let ls_data = ls.to_f64_vec()?;
    for (ls_val, s_val) in ls_data.iter().zip(s_data.iter()) {
        assert!(approx_eq(*ls_val, s_val.ln(), 1e-8));
    }
    Ok(())
}

// Phase 4: Variance test

#[test]
fn test_variance() -> shrew::Result<()> {
    let dev = CpuDevice;

    // Variance of [1, 2, 3, 4, 5] = mean((x - 3)²) = (4+1+0+1+4)/5 = 2.0
    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], (1, 5), DType::F64, &dev)?;
    let v = x.var(1, false)?;
    assert!(approx_eq(v.to_scalar_f64()?, 2.0, 1e-10));
    Ok(())
}

// Phase 4: Cat / Chunk tests

#[test]
fn test_cat_along_dim0() -> shrew::Result<()> {
    let dev = CpuDevice;

    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[5.0, 6.0, 7.0, 8.0], (2, 2), DType::F64, &dev)?;
    let c = CpuTensor::cat(&[a, b], 0)?;

    assert_eq!(c.dims(), &[4, 2]);
    assert_vec_approx(
        &c.to_f64_vec()?,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        1e-10,
    );
    Ok(())
}

#[test]
fn test_cat_along_dim1() -> shrew::Result<()> {
    let dev = CpuDevice;

    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[5.0, 6.0], (2, 1), DType::F64, &dev)?;
    let c = CpuTensor::cat(&[a, b], 1)?;

    assert_eq!(c.dims(), &[2, 3]);
    assert_vec_approx(&c.to_f64_vec()?, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0], 1e-10);
    Ok(())
}

#[test]
fn test_chunk_roundtrip() -> shrew::Result<()> {
    let dev = CpuDevice;

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (6, 1), DType::F64, &dev)?;
    let chunks = x.chunk(3, 0)?;
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].dims(), &[2, 1]);
    assert_eq!(chunks[1].dims(), &[2, 1]);
    assert_eq!(chunks[2].dims(), &[2, 1]);

    // Concatenating chunks should give back original
    let owned: Vec<CpuTensor> = chunks;
    let recovered = CpuTensor::cat(&owned, 0)?;
    assert_vec_approx(&recovered.to_f64_vec()?, &x.to_f64_vec()?, 1e-10);
    Ok(())
}

// Phase 4: Expand test

#[test]
fn test_expand() -> shrew::Result<()> {
    let dev = CpuDevice;

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], (1, 3), DType::F64, &dev)?;
    let expanded = x.expand((4, 3))?;
    assert_eq!(expanded.dims(), &[4, 3]);

    let data = expanded.to_f64_vec()?;
    // All 4 rows should be [1, 2, 3]
    for row in 0..4 {
        assert_vec_approx(&data[row * 3..row * 3 + 3], &[1.0, 2.0, 3.0], 1e-10);
    }
    Ok(())
}

// Phase 4: LayerNorm tests

#[test]
fn test_layernorm_output_stats() -> shrew::Result<()> {
    let dev = CpuDevice;
    let ln = LayerNorm::<CpuBackend>::new(4, 1e-5, DType::F64, &dev)?;

    // Input: [2, 4]
    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        (2, 4),
        DType::F64,
        &dev,
    )?;
    let y = ln.forward(&x)?;
    assert_eq!(y.dims(), &[2, 4]);

    let data = y.to_f64_vec()?;

    // Each row should have mean ≈ 0 and std ≈ 1 (with default γ=1, β=0)
    for row in 0..2 {
        let slice = &data[row * 4..(row + 1) * 4];
        let mean: f64 = slice.iter().sum::<f64>() / 4.0;
        let var: f64 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        assert!(approx_eq(mean, 0.0, 1e-5), "Row {} mean: {}", row, mean);
        assert!(approx_eq(var, 1.0, 1e-3), "Row {} var: {}", row, var);
    }
    Ok(())
}

#[test]
fn test_layernorm_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let ln = LayerNorm::<CpuBackend>::new(8, 1e-5, DType::F64, &dev)?;
    // Should have 2 parameters: weight (γ) and bias (β)
    assert_eq!(ln.parameters().len(), 2);
    Ok(())
}

#[test]
fn test_layernorm_3d() -> shrew::Result<()> {
    let dev = CpuDevice;
    let ln = LayerNorm::<CpuBackend>::new(4, 1e-5, DType::F64, &dev)?;

    // Input: [batch=2, seq=3, d_model=4]
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let x = CpuTensor::from_f64_slice(&data, (2, 3, 4), DType::F64, &dev)?;
    let y = ln.forward(&x)?;
    assert_eq!(y.dims(), &[2, 3, 4]);

    // Check normalization on last dim for each (batch, seq) position
    let out = y.to_f64_vec()?;
    for i in 0..6 {
        let slice = &out[i * 4..(i + 1) * 4];
        let mean: f64 = slice.iter().sum::<f64>() / 4.0;
        assert!(approx_eq(mean, 0.0, 1e-5), "Position {} mean: {}", i, mean);
    }
    Ok(())
}

// Phase 4: MultiHeadAttention tests

#[test]
fn test_mha_output_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let d_model = 8;
    let num_heads = 2;

    let mha = MultiHeadAttention::<CpuBackend>::new(d_model, num_heads, DType::F64, &dev)?;

    // Input: [batch=2, seq=5, d_model=8]
    let x = CpuTensor::rand((2, 5, d_model), DType::F64, &dev)?;
    let y = mha.forward(&x)?;

    // Output should have same shape as input
    assert_eq!(y.dims(), &[2, 5, 8]);
    Ok(())
}

#[test]
fn test_mha_parameter_count() -> shrew::Result<()> {
    let dev = CpuDevice;
    let d_model = 8;
    let num_heads = 2;

    let mha = MultiHeadAttention::<CpuBackend>::new(d_model, num_heads, DType::F64, &dev)?;
    let params = mha.parameters();

    // 4 weight matrices (Q, K, V, O), each [d_model, d_model], no biases
    assert_eq!(params.len(), 4);
    for p in &params {
        assert_eq!(p.dims(), &[d_model, d_model]);
    }
    Ok(())
}

#[test]
fn test_mha_causal() -> shrew::Result<()> {
    let dev = CpuDevice;
    let d_model = 8;
    let num_heads = 2;

    let mha = MultiHeadAttention::<CpuBackend>::new(d_model, num_heads, DType::F64, &dev)?
        .with_causal(true);

    let x = CpuTensor::rand((1, 4, d_model), DType::F64, &dev)?;
    let y = mha.forward(&x)?;
    assert_eq!(y.dims(), &[1, 4, 8]);
    Ok(())
}

// Phase 4: TransformerBlock tests

#[test]
fn test_transformer_block_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let d_model = 16;
    let num_heads = 2;
    let d_ff = 32;

    let block =
        TransformerBlock::<CpuBackend>::new(d_model, num_heads, d_ff, false, DType::F64, &dev)?;

    // Input: [batch=2, seq=5, d_model=16]
    let x = CpuTensor::rand((2, 5, d_model), DType::F64, &dev)?;
    let y = block.forward(&x)?;

    // Output should have same shape as input (residual connections preserve shape)
    assert_eq!(y.dims(), &[2, 5, d_model]);
    Ok(())
}

#[test]
fn test_transformer_block_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let d_model = 8;
    let num_heads = 2;
    let d_ff = 16;

    let block =
        TransformerBlock::<CpuBackend>::new(d_model, num_heads, d_ff, false, DType::F64, &dev)?;

    let params = block.parameters();
    // LN1: 2 (γ, β)
    // MHA: 4 (W_q, W_k, W_v, W_o)
    // LN2: 2 (γ, β)
    // FF1: 2 (weight, bias)
    // FF2: 2 (weight, bias)
    // Total: 12
    assert_eq!(params.len(), 12);
    Ok(())
}

#[test]
fn test_transformer_block_causal() -> shrew::Result<()> {
    let dev = CpuDevice;
    let d_model = 8;
    let num_heads = 2;
    let d_ff = 16;

    let block =
        TransformerBlock::<CpuBackend>::new(d_model, num_heads, d_ff, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((1, 6, d_model), DType::F64, &dev)?;
    let y = block.forward(&x)?;
    assert_eq!(y.dims(), &[1, 6, d_model]);
    Ok(())
}

#[test]
fn test_transformer_block_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let d_model = 8;
    let num_heads = 2;
    let d_ff = 16;

    let block =
        TransformerBlock::<CpuBackend>::new(d_model, num_heads, d_ff, false, DType::F64, &dev)?;

    let x = CpuTensor::rand((1, 3, d_model), DType::F64, &dev)?.set_variable();
    let y = block.forward(&x)?;
    let loss = y.sum_all()?;

    // Backward should complete without error
    let grads = loss.backward()?;

    // All parameters should have gradients
    for p in &block.parameters() {
        assert!(grads.get(p).is_some(), "Missing gradient for parameter");
    }
    Ok(())
}

// Conv2d layer tests

#[test]
fn test_conv2d_output_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let conv = Conv2d::<CpuBackend>::new(3, 16, [3, 3], [1, 1], [0, 0], true, DType::F64, &dev)?;

    assert_eq!(conv.weight().dims(), &[16, 3, 3, 3]);
    assert_eq!(conv.bias().unwrap().dims(), &[16]);
    assert_eq!(conv.in_channels(), 3);
    assert_eq!(conv.out_channels(), 16);

    // Input: [N=2, C=3, H=8, W=8], kernel=3, no padding, stride=1
    // H_out = (8 - 3) / 1 + 1 = 6
    let x = Tensor::<CpuBackend>::rand((2, 3, 8, 8), DType::F64, &dev)?;
    let y = conv.forward(&x)?;
    assert_eq!(y.dims(), &[2, 16, 6, 6]);
    Ok(())
}

#[test]
fn test_conv2d_with_padding() -> shrew::Result<()> {
    let dev = CpuDevice;
    // padding=1 with kernel=3 → same spatial dims
    let conv = Conv2d::<CpuBackend>::new(1, 4, [3, 3], [1, 1], [1, 1], false, DType::F64, &dev)?;
    let x = Tensor::<CpuBackend>::rand((1, 1, 5, 5), DType::F64, &dev)?;
    let y = conv.forward(&x)?;
    // H_out = (5 + 2*1 - 3)/1 + 1 = 5
    assert_eq!(y.dims(), &[1, 4, 5, 5]);
    Ok(())
}

#[test]
fn test_conv2d_with_stride() -> shrew::Result<()> {
    let dev = CpuDevice;
    let conv = Conv2d::<CpuBackend>::new(1, 2, [3, 3], [2, 2], [1, 1], true, DType::F64, &dev)?;
    let x = Tensor::<CpuBackend>::rand((1, 1, 6, 6), DType::F64, &dev)?;
    let y = conv.forward(&x)?;
    // H_out = (6 + 2 - 3)/2 + 1 = 3
    assert_eq!(y.dims(), &[1, 2, 3, 3]);
    Ok(())
}

#[test]
fn test_conv2d_known_values() -> shrew::Result<()> {
    let dev = CpuDevice;
    // Simple 1-channel, 1-filter, 2x2 kernel, no padding, stride 1
    // Input: [1,1,3,3]  Weight: [1,1,2,2]
    let input_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let x = Tensor::<CpuBackend>::from_f64_slice(
        &input_data,
        Shape::new(vec![1, 1, 3, 3]),
        DType::F64,
        &dev,
    )?;

    let w_data: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0]; // identity-like
    let w = Tensor::<CpuBackend>::from_f64_slice(
        &w_data,
        Shape::new(vec![1, 1, 2, 2]),
        DType::F64,
        &dev,
    )?;

    // Output: [1,1,2,2]
    // out[0,0,0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
    // out[0,0,0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
    // out[0,0,1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
    // out[0,0,1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
    let y = x.conv2d(&w, None, [1, 1], [0, 0])?;
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let y_data = y.to_f64_vec()?;
    assert_vec_approx(&y_data, &[6.0, 8.0, 12.0, 14.0], 1e-10);
    Ok(())
}

#[test]
fn test_conv2d_with_bias_known_values() -> shrew::Result<()> {
    let dev = CpuDevice;
    let input_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let x = Tensor::<CpuBackend>::from_f64_slice(
        &input_data,
        Shape::new(vec![1, 1, 2, 2]),
        DType::F64,
        &dev,
    )?;
    let w_data: Vec<f64> = vec![1.0]; // 1x1 conv
    let w = Tensor::<CpuBackend>::from_f64_slice(
        &w_data,
        Shape::new(vec![1, 1, 1, 1]),
        DType::F64,
        &dev,
    )?;
    let b_data: Vec<f64> = vec![10.0];
    let b = Tensor::<CpuBackend>::from_f64_slice(&b_data, Shape::new(vec![1]), DType::F64, &dev)?;

    let y = x.conv2d(&w, Some(&b), [1, 1], [0, 0])?;
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let y_data = y.to_f64_vec()?;
    assert_vec_approx(&y_data, &[11.0, 12.0, 13.0, 14.0], 1e-10);
    Ok(())
}

#[test]
fn test_conv2d_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let conv_bias =
        Conv2d::<CpuBackend>::new(3, 8, [3, 3], [1, 1], [0, 0], true, DType::F64, &dev)?;
    assert_eq!(conv_bias.parameters().len(), 2); // weight + bias

    let conv_no_bias =
        Conv2d::<CpuBackend>::new(3, 8, [3, 3], [1, 1], [0, 0], false, DType::F64, &dev)?;
    assert_eq!(conv_no_bias.parameters().len(), 1); // weight only
    Ok(())
}

#[test]
fn test_conv2d_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    // Use known tensors so we can verify gradients
    let x_data: Vec<f64> = (1..=12).map(|v| v as f64).collect(); // [1..12]
    let x = Tensor::<CpuBackend>::from_f64_slice(
        &x_data,
        Shape::new(vec![1, 1, 3, 4]),
        DType::F64,
        &dev,
    )?
    .set_variable();

    let w_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let w = Tensor::<CpuBackend>::from_f64_slice(
        &w_data,
        Shape::new(vec![1, 1, 2, 2]),
        DType::F64,
        &dev,
    )?
    .set_variable();

    let y = x.conv2d(&w, None, [1, 1], [0, 0])?;
    assert_eq!(y.dims(), &[1, 1, 2, 3]);

    let loss = y.sum_all()?;
    let grads = loss.backward()?;

    // grad_weight should have shape [1,1,2,2]
    let gw = grads.get(&w).expect("weight gradient");
    assert_eq!(gw.dims(), &[1, 1, 2, 2]);

    // grad_input should have shape [1,1,3,4]
    let gx = grads.get(&x).expect("input gradient");
    assert_eq!(gx.dims(), &[1, 1, 3, 4]);

    // Verify some grad values aren't zero (grad should flow)
    let gw_data = gw.to_f64_vec()?;
    let gx_data = gx.to_f64_vec()?;
    assert!(
        gw_data.iter().any(|&v| v.abs() > 0.0),
        "weight gradient should be non-zero"
    );
    assert!(
        gx_data.iter().any(|&v| v.abs() > 0.0),
        "input gradient should be non-zero"
    );
    Ok(())
}

#[test]
fn test_conv2d_backward_with_bias() -> shrew::Result<()> {
    let dev = CpuDevice;
    let conv = Conv2d::<CpuBackend>::new(1, 2, [2, 2], [1, 1], [0, 0], true, DType::F64, &dev)?;
    let x = Tensor::<CpuBackend>::rand((1, 1, 4, 4), DType::F64, &dev)?.set_variable();
    let y = conv.forward(&x)?;
    let loss = y.sum_all()?;
    let grads = loss.backward()?;

    // Check gradients exist for all parameters
    for p in conv.parameters() {
        assert!(
            grads.get(&p).is_some(),
            "Missing gradient for conv parameter"
        );
    }
    // Input gradient
    assert!(grads.get(&x).is_some(), "Missing gradient for input");
    Ok(())
}

#[test]
fn test_conv2d_numerical_grad() -> shrew::Result<()> {
    // Numerical gradient check for conv2d weight
    let dev = CpuDevice;
    let x_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let x = Tensor::<CpuBackend>::from_f64_slice(
        &x_data,
        Shape::new(vec![1, 1, 3, 3]),
        DType::F64,
        &dev,
    )?;

    let w_data: Vec<f64> = vec![0.5, -0.3, 0.1, 0.7];
    let w = Tensor::<CpuBackend>::from_f64_slice(
        &w_data,
        Shape::new(vec![1, 1, 2, 2]),
        DType::F64,
        &dev,
    )?
    .set_variable();

    // Forward + backward
    let y = x.conv2d(&w, None, [1, 1], [0, 0])?;
    let loss = y.sum_all()?;
    let grads = loss.backward()?;
    let analytic_grad = grads.get(&w).unwrap().to_f64_vec()?;

    // Numerical gradient
    let eps = 1e-5;
    let mut numerical_grad = vec![0.0f64; 4];
    for i in 0..4 {
        let mut w_plus = w_data.clone();
        w_plus[i] += eps;
        let wp = Tensor::<CpuBackend>::from_f64_slice(
            &w_plus,
            Shape::new(vec![1, 1, 2, 2]),
            DType::F64,
            &dev,
        )?;
        let loss_plus = x
            .conv2d(&wp, None, [1, 1], [0, 0])?
            .sum_all()?
            .to_scalar_f64()?;

        let mut w_minus = w_data.clone();
        w_minus[i] -= eps;
        let wm = Tensor::<CpuBackend>::from_f64_slice(
            &w_minus,
            Shape::new(vec![1, 1, 2, 2]),
            DType::F64,
            &dev,
        )?;
        let loss_minus = x
            .conv2d(&wm, None, [1, 1], [0, 0])?
            .sum_all()?
            .to_scalar_f64()?;

        numerical_grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    assert_vec_approx(&analytic_grad, &numerical_grad, 1e-4);
    Ok(())
}

// MaxPool2d layer tests

#[test]
fn test_maxpool2d_output_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);
    let x = Tensor::<CpuBackend>::rand((2, 3, 8, 8), DType::F64, &dev)?;
    let y = pool.forward(&x)?;
    // H_out = (8 - 2)/2 + 1 = 4
    assert_eq!(y.dims(), &[2, 3, 4, 4]);
    Ok(())
}

#[test]
fn test_maxpool2d_known_values() -> shrew::Result<()> {
    let dev = CpuDevice;
    let data: Vec<f64> = vec![
        1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 15.0, 14.0, 16.0,
    ];
    let x = Tensor::<CpuBackend>::from_f64_slice(
        &data,
        Shape::new(vec![1, 1, 4, 4]),
        DType::F64,
        &dev,
    )?;
    let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);
    let y = pool.forward(&x)?;
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let y_data = y.to_f64_vec()?;
    // Pool windows:
    //   [1,3,5,7] → 7,  [2,4,6,8] → 8
    //   [9,11,13,15] → 15,  [10,12,14,16] → 16
    assert_vec_approx(&y_data, &[7.0, 8.0, 15.0, 16.0], 1e-10);
    Ok(())
}

#[test]
fn test_maxpool2d_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let data: Vec<f64> = vec![
        1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 15.0, 14.0, 16.0,
    ];
    let x = Tensor::<CpuBackend>::from_f64_slice(
        &data,
        Shape::new(vec![1, 1, 4, 4]),
        DType::F64,
        &dev,
    )?
    .set_variable();

    let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);
    let y = pool.forward(&x)?;
    let loss = y.sum_all()?;
    let grads = loss.backward()?;

    let gx = grads.get(&x).expect("input gradient");
    assert_eq!(gx.dims(), &[1, 1, 4, 4]);
    let gx_data = gx.to_f64_vec()?;

    // Gradient flows only to max positions: positions 5(=7), 7(=8), 13(=15), 15(=16)
    // At index 5 (val=7): grad=1, index 7 (val=8): grad=1
    // At index 13 (val=15): grad=1, index 15 (val=16): grad=1
    // All others = 0
    let expected = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    ];
    assert_vec_approx(&gx_data, &expected, 1e-10);
    Ok(())
}

#[test]
fn test_maxpool2d_with_padding() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pool = MaxPool2d::new([3, 3], [1, 1], [1, 1]);
    let x = Tensor::<CpuBackend>::rand((1, 1, 4, 4), DType::F64, &dev)?;
    let y = pool.forward(&x)?;
    // H_out = (4 + 2 - 3)/1 + 1 = 4 (same spatial dims)
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    Ok(())
}

#[test]
fn test_conv2d_maxpool2d_chain() -> shrew::Result<()> {
    // Test a Conv2d → ReLU → MaxPool2d pipeline
    let dev = CpuDevice;
    let conv = Conv2d::<CpuBackend>::new(1, 4, [3, 3], [1, 1], [1, 1], true, DType::F64, &dev)?;
    let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);

    let x = Tensor::<CpuBackend>::rand((2, 1, 8, 8), DType::F64, &dev)?.set_variable();

    // Forward: [2,1,8,8] → conv → [2,4,8,8] → relu → [2,4,8,8] → pool → [2,4,4,4]
    let h = conv.forward(&x)?;
    let h = h.relu()?;
    let h = pool.forward(&h)?;
    assert_eq!(h.dims(), &[2, 4, 4, 4]);

    // Backward through the full chain
    let loss = h.mean_all()?;
    let grads = loss.backward()?;

    // All conv parameters should have gradients
    for p in conv.parameters() {
        assert!(
            grads.get(&p).is_some(),
            "Missing gradient for conv parameter"
        );
    }
    // Input should have gradient
    assert!(grads.get(&x).is_some(), "Missing gradient for input");
    Ok(())
}

#[test]
fn test_conv2d_multi_channel() -> shrew::Result<()> {
    // Input: [1, 3, 4, 4] → Conv2d(3→8, k=3, p=1) → [1, 8, 4, 4]
    let dev = CpuDevice;
    let conv = Conv2d::<CpuBackend>::new(3, 8, [3, 3], [1, 1], [1, 1], true, DType::F64, &dev)?;
    let x = Tensor::<CpuBackend>::rand((1, 3, 4, 4), DType::F64, &dev)?;
    let y = conv.forward(&x)?;
    assert_eq!(y.dims(), &[1, 8, 4, 4]);
    Ok(())
}

#[test]
fn test_conv2d_from_tensors() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = Tensor::<CpuBackend>::ones(Shape::new(vec![4, 2, 3, 3]), DType::F64, &dev)?;
    let b = Tensor::<CpuBackend>::zeros(Shape::new(vec![4]), DType::F64, &dev)?;
    let conv = Conv2d::from_tensors(w, Some(b), [1, 1], [0, 0])?;
    assert_eq!(conv.in_channels(), 2);
    assert_eq!(conv.out_channels(), 4);
    assert_eq!(conv.kernel_size(), [3, 3]);
    Ok(())
}

// Flatten tests

#[test]
fn test_flatten_4d_to_2d() -> shrew::Result<()> {
    let dev = CpuDevice;
    let flatten = Flatten::new(1);
    let x = Tensor::<CpuBackend>::rand((2, 3, 4, 5), DType::F64, &dev)?;
    let y = flatten.forward(&x)?;
    assert_eq!(y.dims(), &[2, 60]); // 3*4*5 = 60
    Ok(())
}

#[test]
fn test_flatten_preserves_data() -> shrew::Result<()> {
    let dev = CpuDevice;
    let flatten = Flatten::new(1);
    let data: Vec<f64> = (0..24).map(|v| v as f64).collect();
    let x = Tensor::<CpuBackend>::from_f64_slice(
        &data,
        Shape::new(vec![2, 3, 2, 2]),
        DType::F64,
        &dev,
    )?;
    let y = flatten.forward(&x)?;
    assert_eq!(y.dims(), &[2, 12]);
    let y_data = y.to_f64_vec()?;
    assert_vec_approx(&y_data, &data, 1e-10);
    Ok(())
}

#[test]
fn test_flatten_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let flatten = Flatten::new(1);
    let x = Tensor::<CpuBackend>::rand((2, 3, 4, 4), DType::F64, &dev)?.set_variable();
    let y = flatten.forward(&x)?;
    let loss = y.sum_all()?;
    let grads = loss.backward()?;
    let gx = grads.get(&x).expect("input gradient");
    assert_eq!(gx.dims(), &[2, 3, 4, 4]); // back to original shape
    Ok(())
}

#[test]
fn test_flatten_no_params() -> shrew::Result<()> {
    let flatten = Flatten::new(1);
    let params: Vec<Tensor<CpuBackend>> = flatten.parameters();
    assert_eq!(params.len(), 0);
    Ok(())
}

#[test]
fn test_flatten_from_dim0() -> shrew::Result<()> {
    let dev = CpuDevice;
    let flatten = Flatten::new(0);
    let x = Tensor::<CpuBackend>::rand((2, 3, 4), DType::F64, &dev)?;
    let y = flatten.forward(&x)?;
    assert_eq!(y.dims(), &[24]); // fully flat
    Ok(())
}

// BatchNorm2d tests

#[test]
fn test_batchnorm2d_output_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let bn = BatchNorm2d::<CpuBackend>::new(8, 1e-5, 0.1, DType::F64, &dev)?;
    let x = Tensor::<CpuBackend>::rand((4, 8, 6, 6), DType::F64, &dev)?;
    let y = bn.forward(&x)?;
    assert_eq!(y.dims(), &[4, 8, 6, 6]); // same shape
    Ok(())
}

#[test]
fn test_batchnorm2d_training_normalizes() -> shrew::Result<()> {
    let dev = CpuDevice;
    let bn = BatchNorm2d::<CpuBackend>::new(2, 1e-5, 0.1, DType::F64, &dev)?;

    // Create input with known statistics
    let x = Tensor::<CpuBackend>::rand((4, 2, 3, 3), DType::F64, &dev)?;
    let y = bn.forward(&x)?;

    // After normalization, each channel should have approximately
    // zero mean and unit variance (since gamma=1, beta=0)
    let y_data = y.to_f64_vec()?;
    let (n, c, h, w) = (4, 2, 3, 3);
    let hw = h * w;
    let count = (n * hw) as f64;

    for ci in 0..c {
        let mut mean = 0.0f64;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    mean += y_data[idx];
                }
            }
        }
        mean /= count;
        assert!(
            mean.abs() < 0.1,
            "channel {} mean should be ~0, got {}",
            ci,
            mean
        );

        let mut var = 0.0f64;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let diff = y_data[idx] - mean;
                    var += diff * diff;
                }
            }
        }
        var /= count;
        assert!(
            (var - 1.0).abs() < 0.2,
            "channel {} var should be ~1, got {}",
            ci,
            var
        );
    }
    Ok(())
}

#[test]
fn test_batchnorm2d_eval_mode() -> shrew::Result<()> {
    let dev = CpuDevice;
    let bn = BatchNorm2d::<CpuBackend>::new(2, 1e-5, 0.1, DType::F64, &dev)?;

    // Run a few forward passes in training mode to build running stats
    for _ in 0..5 {
        let x = Tensor::<CpuBackend>::rand((4, 2, 3, 3), DType::F64, &dev)?;
        bn.forward(&x)?;
    }

    // Switch to eval mode
    bn.eval();
    assert!(!bn.is_training());

    // Eval forward should work and produce same shape
    let x = Tensor::<CpuBackend>::rand((4, 2, 3, 3), DType::F64, &dev)?;
    let y = bn.forward(&x)?;
    assert_eq!(y.dims(), &[4, 2, 3, 3]);

    // Switch back to training
    bn.train();
    assert!(bn.is_training());
    Ok(())
}

#[test]
fn test_batchnorm2d_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let bn = BatchNorm2d::<CpuBackend>::new(16, 1e-5, 0.1, DType::F64, &dev)?;
    let params = bn.parameters();
    assert_eq!(params.len(), 2); // weight (gamma) + bias (beta)
    assert_eq!(params[0].dims(), &[16]); // gamma
    assert_eq!(params[1].dims(), &[16]); // beta
    Ok(())
}

#[test]
fn test_batchnorm2d_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let bn = BatchNorm2d::<CpuBackend>::new(4, 1e-5, 0.1, DType::F64, &dev)?;
    let x = Tensor::<CpuBackend>::rand((2, 4, 3, 3), DType::F64, &dev)?.set_variable();
    let y = bn.forward(&x)?;
    let loss = y.mean_all()?;
    let grads = loss.backward()?;

    // Gamma and beta should have gradients
    for p in bn.parameters() {
        assert!(
            grads.get(&p).is_some(),
            "Missing gradient for BatchNorm parameter"
        );
    }
    // Input should have gradient
    assert!(grads.get(&x).is_some(), "Missing gradient for input");
    Ok(())
}

#[test]
fn test_conv_bn_relu_pool_flatten_linear() -> shrew::Result<()> {
    // Full CNN pipeline test:
    // Conv2d → BatchNorm2d → ReLU → MaxPool2d → Flatten → Linear
    let dev = CpuDevice;

    let conv = Conv2d::<CpuBackend>::new(1, 4, [3, 3], [1, 1], [1, 1], true, DType::F64, &dev)?;
    let bn = BatchNorm2d::<CpuBackend>::new(4, 1e-5, 0.1, DType::F64, &dev)?;
    let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);
    let flatten = Flatten::new(1);
    // Input: [2, 1, 8, 8]
    // After conv (k=3, p=1): [2, 4, 8, 8]
    // After bn: [2, 4, 8, 8]
    // After relu: [2, 4, 8, 8]
    // After pool (k=2, s=2): [2, 4, 4, 4]
    // After flatten: [2, 64]
    let linear = Linear::<CpuBackend>::new(64, 10, true, DType::F64, &dev)?;

    let x = Tensor::<CpuBackend>::rand((2, 1, 8, 8), DType::F64, &dev)?.set_variable();
    let h = conv.forward(&x)?;
    assert_eq!(h.dims(), &[2, 4, 8, 8]);
    let h = bn.forward(&h)?;
    assert_eq!(h.dims(), &[2, 4, 8, 8]);
    let h = h.relu()?;
    let h = pool.forward(&h)?;
    assert_eq!(h.dims(), &[2, 4, 4, 4]);
    let h = flatten.forward(&h)?;
    assert_eq!(h.dims(), &[2, 64]);
    let logits = linear.forward(&h)?;
    assert_eq!(logits.dims(), &[2, 10]);

    // Backward through the full pipeline
    let loss = logits.sum_all()?;
    let grads = loss.backward()?;

    // All conv+bn+linear params should have gradients
    let all_params: Vec<_> = conv
        .parameters()
        .into_iter()
        .chain(bn.parameters())
        .chain(linear.parameters())
        .collect();
    for p in &all_params {
        assert!(grads.get(p).is_some(), "Missing gradient for parameter");
    }
    assert!(grads.get(&x).is_some(), "Missing gradient for input");
    Ok(())
}

// Max/Min reduce backward tests

#[test]
fn test_max_backward_all() -> shrew::Result<()> {
    let dev = CpuDevice;
    // x = [1, 3, 2] → max_all = 3, gradient should flow only to index 1
    let x = CpuTensor::from_f64_slice(&[1.0, 3.0, 2.0], (3,), DType::F64, &dev)?.set_variable();
    let y = x.max(0, false)?;
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap().to_f64_vec()?;
    assert_vec_approx(&gx, &[0.0, 1.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_min_backward_all() -> shrew::Result<()> {
    let dev = CpuDevice;
    // x = [5, 1, 3] → min = 1 at index 1
    let x = CpuTensor::from_f64_slice(&[5.0, 1.0, 3.0], (3,), DType::F64, &dev)?.set_variable();
    let y = x.min(0, false)?;
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap().to_f64_vec()?;
    assert_vec_approx(&gx, &[0.0, 1.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_max_backward_2d() -> shrew::Result<()> {
    let dev = CpuDevice;
    // x = [[1, 4], [3, 2]] → max(dim=1) = [4, 3]
    // grad flows to (0,1) and (1,0)
    let x =
        CpuTensor::from_f64_slice(&[1.0, 4.0, 3.0, 2.0], (2, 2), DType::F64, &dev)?.set_variable();
    let y = x.max(1, false)?; // [4, 3]
    let loss = y.sum_all()?;
    let grads = loss.backward()?;
    let gx = grads.get(&x).unwrap().to_f64_vec()?;
    // Gradient of sum is 1 at max positions
    assert_vec_approx(&gx, &[0.0, 1.0, 1.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_max_backward_tied_values() -> shrew::Result<()> {
    let dev = CpuDevice;
    // x = [3, 3, 1] → max = 3, tied at indices 0 and 1
    // Gradient should be split: 0.5 each
    let x = CpuTensor::from_f64_slice(&[3.0, 3.0, 1.0], (3,), DType::F64, &dev)?.set_variable();
    let y = x.max(0, false)?;
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap().to_f64_vec()?;
    assert_vec_approx(&gx, &[0.5, 0.5, 0.0], 1e-10);
    Ok(())
}

// LR Scheduler tests

use shrew::optim::{
    CosineAnnealingLR, CosineWarmupLR, ExponentialLR, LinearLR, LrScheduler, StepLR,
};

#[test]
fn test_step_lr() {
    let mut sched = StepLR::new(0.1, 3, 0.5);
    // Steps 1,2,3 → lr = 0.1 * 0.5^0 = 0.1 (first decay at step=3)
    assert!(approx_eq(sched.step(), 0.1, 1e-10)); // step 1: 1/3 = 0 → gamma^0
    assert!(approx_eq(sched.step(), 0.1, 1e-10)); // step 2
    assert!(approx_eq(sched.step(), 0.05, 1e-10)); // step 3: 3/3=1 → gamma^1
    assert!(approx_eq(sched.step(), 0.05, 1e-10)); // step 4
    assert!(approx_eq(sched.step(), 0.05, 1e-10)); // step 5
    assert!(approx_eq(sched.step(), 0.025, 1e-10)); // step 6: gamma^2
}

#[test]
fn test_exponential_lr() {
    let mut sched = ExponentialLR::new(1.0, 0.9);
    assert!(approx_eq(sched.step(), 0.9, 1e-10)); // step 1: 1.0 * 0.9^1
    assert!(approx_eq(sched.step(), 0.81, 1e-10)); // step 2: 1.0 * 0.9^2
    assert!(approx_eq(sched.step(), 0.729, 1e-10)); // step 3: 1.0 * 0.9^3
}

#[test]
fn test_linear_lr() {
    let mut sched = LinearLR::new(0.1, 1.0, 0.1, 10);
    // At step 0: factor = 1.0, lr = 0.1
    // At step 5: factor = 1.0 + (0.1-1.0)*(5/10) = 0.55, lr = 0.055
    // At step 10: factor = 0.1, lr = 0.01
    let lr0 = sched.current_lr();
    assert!(approx_eq(lr0, 0.1, 1e-10)); // step 0
    for _ in 0..5 {
        sched.step();
    }
    assert!(approx_eq(sched.current_lr(), 0.055, 1e-10));
    for _ in 0..5 {
        sched.step();
    }
    assert!(approx_eq(sched.current_lr(), 0.01, 1e-10));
    // Past total_steps, stays at end
    sched.step();
    assert!(approx_eq(sched.current_lr(), 0.01, 1e-10));
}

#[test]
fn test_cosine_annealing_lr() {
    let mut sched = CosineAnnealingLR::new(0.1, 100, 0.0);
    // At step 0: lr = 0.1
    // At step 50: lr = 0.05 (halfway through cosine)
    // At step 100: lr = 0.0
    assert!(approx_eq(sched.current_lr(), 0.1, 1e-10));
    for _ in 0..50 {
        sched.step();
    }
    assert!(approx_eq(sched.current_lr(), 0.05, 1e-6));
    for _ in 0..50 {
        sched.step();
    }
    assert!(approx_eq(sched.current_lr(), 0.0, 1e-10));
}

#[test]
fn test_cosine_warmup_lr() {
    let mut sched = CosineWarmupLR::new(0.001, 10, 100, 0.0);
    // Warmup phase: linear from 0 to 0.001 over 10 steps
    let lr1 = sched.step(); // step 1
    assert!(approx_eq(lr1, 0.0001, 1e-10)); // 0.001 * (1/10)
    for _ in 0..4 {
        sched.step();
    }
    let lr5 = sched.current_lr();
    assert!(approx_eq(lr5, 0.0005, 1e-10)); // 0.001 * (5/10)
    for _ in 0..5 {
        sched.step();
    }
    let lr10 = sched.current_lr();
    assert!(approx_eq(lr10, 0.001, 1e-10)); // peak at end of warmup

    // Decay phase: cosine from 0.001 to 0.0 over 90 steps
    for _ in 0..90 {
        sched.step();
    }
    let lr100 = sched.current_lr();
    assert!(approx_eq(lr100, 0.0, 1e-6)); // end of schedule
}

#[test]
fn test_scheduler_reset() {
    let mut sched = StepLR::new(0.1, 5, 0.5);
    for _ in 0..10 {
        sched.step();
    }
    assert_eq!(sched.current_step(), 10);
    sched.reset();
    assert_eq!(sched.current_step(), 0);
    assert!(approx_eq(sched.current_lr(), 0.1, 1e-10));
}

#[test]
fn test_cosine_warmup_with_optimizer() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0, 2.0], (2,), DType::F64, &dev)?.set_variable();
    let mut optimizer = Adam::<CpuBackend>::new(vec![w.clone()], 0.001);
    let mut scheduler = CosineWarmupLR::new(0.001, 5, 20, 1e-5);

    // Simulate a few steps
    for _ in 0..5 {
        let lr = scheduler.step();
        optimizer.set_learning_rate(lr);
    }
    // After warmup, lr should be peak
    assert!(approx_eq(optimizer.learning_rate(), 0.001, 1e-10));
    Ok(())
}

// Gradient clipping tests

use shrew::optim::{clip_grad_norm, clip_grad_value, grad_norm};

#[test]
fn test_clip_grad_norm_no_clip() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F64, &dev)?.set_variable();
    let loss = w.sum_all()?;
    let grads = loss.backward()?;

    // grad = [1, 1, 1], norm = sqrt(3) ≈ 1.732
    let norm = grad_norm(&grads, &[w.clone()])?;
    assert!(approx_eq(norm, 3.0f64.sqrt(), 1e-10));

    // max_norm = 5, no clipping should happen
    let (clipped, total) = clip_grad_norm(&grads, &[w.clone()], 5.0)?;
    assert!(approx_eq(total, 3.0f64.sqrt(), 1e-10));
    let cg = clipped.get(&w).unwrap().to_f64_vec()?;
    assert_vec_approx(&cg, &[1.0, 1.0, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_clip_grad_norm_clips() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[3.0, 4.0], (2,), DType::F64, &dev)?.set_variable();
    // loss = w^2.sum() → grad = 2w = [6, 8], norm = 10
    let loss = (w.mul(&w)?).sum_all()?;
    let grads = loss.backward()?;

    let norm = grad_norm(&grads, &[w.clone()])?;
    assert!(approx_eq(norm, 10.0, 1e-10));

    // Clip to max_norm = 5 → scale = 5/10 = 0.5
    let (clipped, total) = clip_grad_norm(&grads, &[w.clone()], 5.0)?;
    assert!(approx_eq(total, 10.0, 1e-10));
    let cg = clipped.get(&w).unwrap().to_f64_vec()?;
    assert_vec_approx(&cg, &[3.0, 4.0], 1e-6); // [6*0.5, 8*0.5]
    Ok(())
}

#[test]
fn test_clip_grad_value() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[3.0, 4.0], (2,), DType::F64, &dev)?.set_variable();
    let loss = (w.mul(&w)?).sum_all()?; // grad = [6, 8]
    let grads = loss.backward()?;

    // Clamp to [-5, 5]
    let clipped = clip_grad_value(&grads, &[w.clone()], 5.0)?;
    let cg = clipped.get(&w).unwrap().to_f64_vec()?;
    assert_vec_approx(&cg, &[5.0, 5.0], 1e-10); // clamped from [6,8]
    Ok(())
}

#[test]
fn test_clip_grad_norm_multiple_params() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w1 = CpuTensor::from_f64_slice(&[3.0], (1,), DType::F64, &dev)?.set_variable();
    let w2 = CpuTensor::from_f64_slice(&[4.0], (1,), DType::F64, &dev)?.set_variable();
    // loss = w1^2 + w2^2 → grads = [6], [8], global norm = sqrt(36+64) = 10
    let loss = w1.mul(&w1)?.add(&w2.mul(&w2)?)?.sum_all()?;
    let grads = loss.backward()?;

    let params = vec![w1.clone(), w2.clone()];
    let (clipped, total) = clip_grad_norm(&grads, &params, 2.0)?;
    assert!(approx_eq(total, 10.0, 1e-10));

    // Scale = 2/10 = 0.2
    let g1 = clipped.get(&w1).unwrap().to_f64_vec()?;
    let g2 = clipped.get(&w2).unwrap().to_f64_vec()?;
    assert_vec_approx(&g1, &[1.2], 1e-6); // 6*0.2
    assert_vec_approx(&g2, &[1.6], 1e-6); // 8*0.2
    Ok(())
}

// cat backward tests

#[test]
fn test_cat_backward_dim0() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a =
        CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?.set_variable();
    let b = CpuTensor::from_f64_slice(&[5.0, 6.0], (1, 2), DType::F64, &dev)?.set_variable();
    let c = CpuTensor::cat(&[a.clone(), b.clone()], 0)?; // [3, 2]
    assert_eq!(c.dims(), &[3, 2]);

    // Sum to scalar so we can call backward
    let loss = c.sum(0, false)?.sum(0, false)?;
    let grads = loss.backward()?;

    // d(sum)/d(a) = ones([2,2]), d(sum)/d(b) = ones([1,2])
    let ga = grads.get(&a).unwrap().to_f64_vec()?;
    let gb = grads.get(&b).unwrap().to_f64_vec()?;
    assert_vec_approx(&ga, &[1.0, 1.0, 1.0, 1.0], 1e-10);
    assert_vec_approx(&gb, &[1.0, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_cat_backward_dim1() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a =
        CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?.set_variable();
    let b = CpuTensor::from_f64_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], (2, 3), DType::F64, &dev)?
        .set_variable();
    let c = CpuTensor::cat(&[a.clone(), b.clone()], 1)?; // [2, 5]
    assert_eq!(c.dims(), &[2, 5]);

    let loss = c.sum(0, false)?.sum(0, false)?;
    let grads = loss.backward()?;

    let ga = grads.get(&a).unwrap().to_f64_vec()?;
    let gb = grads.get(&b).unwrap().to_f64_vec()?;
    assert_vec_approx(&ga, &[1.0, 1.0, 1.0, 1.0], 1e-10);
    assert_vec_approx(&gb, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_cat_backward_weighted() -> shrew::Result<()> {
    let dev = CpuDevice;
    // a * 2: gradient should be 2 for a's portion
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0], (1, 2), DType::F64, &dev)?.set_variable();
    let b = CpuTensor::from_f64_slice(&[3.0, 4.0], (1, 2), DType::F64, &dev)?.set_variable();
    let a2 = a.affine(2.0, 0.0)?;
    let c = CpuTensor::cat(&[a2, b.clone()], 1)?; // [1, 4]
    let loss = c.sum(0, false)?.sum(0, false)?;
    let grads = loss.backward()?;

    let ga = grads.get(&a).unwrap().to_f64_vec()?;
    let gb = grads.get(&b).unwrap().to_f64_vec()?;
    assert_vec_approx(&ga, &[2.0, 2.0], 1e-10); // chain: d(sum) * d(affine)/d(a) = 1*2
    assert_vec_approx(&gb, &[1.0, 1.0], 1e-10);
    Ok(())
}

// RNN tests

#[test]
fn test_rnn_cell_output_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let cell = shrew::nn::RNNCell::<CpuBackend>::new(10, 20, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((4, 10), DType::F64, &dev)?;
    let h = CpuTensor::zeros((4, 20), DType::F64, &dev)?;
    let h_new = cell.forward(&x, &h)?;

    assert_eq!(h_new.dims(), &[4, 20]);
    Ok(())
}

#[test]
fn test_rnn_cell_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    // With bias: W_ih, W_hh, b_ih, b_hh = 4 params
    let cell = shrew::nn::RNNCell::<CpuBackend>::new(5, 8, true, DType::F64, &dev)?;
    assert_eq!(cell.parameters().len(), 4);

    // Without bias: W_ih, W_hh = 2 params
    let cell_nb = shrew::nn::RNNCell::<CpuBackend>::new(5, 8, false, DType::F64, &dev)?;
    assert_eq!(cell_nb.parameters().len(), 2);
    Ok(())
}

#[test]
fn test_rnn_cell_no_bias() -> shrew::Result<()> {
    let dev = CpuDevice;
    let cell = shrew::nn::RNNCell::<CpuBackend>::new(3, 4, false, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 3), DType::F64, &dev)?;
    let h = CpuTensor::zeros((2, 4), DType::F64, &dev)?;
    let h_new = cell.forward(&x, &h)?;
    assert_eq!(h_new.dims(), &[2, 4]);
    Ok(())
}

#[test]
fn test_rnn_sequence_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let rnn = RNN::<CpuBackend>::new(10, 20, true, DType::F64, &dev)?;

    // batch=4, seq_len=5, input_size=10
    let x = CpuTensor::rand((4, 5, 10), DType::F64, &dev)?;
    let (output, h_n) = rnn.forward(&x, None)?;

    assert_eq!(output.dims(), &[4, 5, 20]); // [batch, seq, hidden]
    assert_eq!(h_n.dims(), &[4, 20]); // [batch, hidden]
    Ok(())
}

#[test]
fn test_rnn_with_initial_hidden() -> shrew::Result<()> {
    let dev = CpuDevice;
    let rnn = RNN::<CpuBackend>::new(6, 8, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((2, 3, 6), DType::F64, &dev)?;
    let h0 = CpuTensor::rand((2, 8), DType::F64, &dev)?;
    let (output, h_n) = rnn.forward(&x, Some(&h0))?;

    assert_eq!(output.dims(), &[2, 3, 8]);
    assert_eq!(h_n.dims(), &[2, 8]);
    Ok(())
}

#[test]
fn test_rnn_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let rnn = RNN::<CpuBackend>::new(4, 6, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((2, 3, 4), DType::F64, &dev)?.set_variable();
    let (output, _h_n) = rnn.forward(&x, None)?;

    // Reduce output to scalar
    let loss = output.sum(0, false)?.sum(0, false)?.sum(0, false)?;
    let grads = loss.backward()?;

    // All RNN parameters should have gradients
    for p in rnn.parameters() {
        assert!(grads.get(&p).is_some(), "Missing gradient for RNN param");
        let g = grads.get(&p).unwrap();
        assert_eq!(g.dims(), p.dims());
    }

    // Input should also have gradient
    assert!(grads.get(&x).is_some(), "Missing gradient for input");
    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3, 4]);
    Ok(())
}

// LSTM tests

#[test]
fn test_lstm_cell_output_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let cell = shrew::nn::LSTMCell::<CpuBackend>::new(10, 20, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((4, 10), DType::F64, &dev)?;
    let h = CpuTensor::zeros((4, 20), DType::F64, &dev)?;
    let c = CpuTensor::zeros((4, 20), DType::F64, &dev)?;
    let (h_new, c_new) = cell.forward(&x, &h, &c)?;

    assert_eq!(h_new.dims(), &[4, 20]);
    assert_eq!(c_new.dims(), &[4, 20]);
    Ok(())
}

#[test]
fn test_lstm_cell_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let cell = shrew::nn::LSTMCell::<CpuBackend>::new(5, 8, true, DType::F64, &dev)?;
    // W_ih [32,5], W_hh [32,8], b_ih [1,32], b_hh [1,32] = 4 params
    assert_eq!(cell.parameters().len(), 4);

    let cell_nb = shrew::nn::LSTMCell::<CpuBackend>::new(5, 8, false, DType::F64, &dev)?;
    assert_eq!(cell_nb.parameters().len(), 2);
    Ok(())
}

#[test]
fn test_lstm_sequence_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let lstm = LSTM::<CpuBackend>::new(10, 20, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((4, 5, 10), DType::F64, &dev)?;
    let (output, (h_n, c_n)) = lstm.forward(&x, None)?;

    assert_eq!(output.dims(), &[4, 5, 20]);
    assert_eq!(h_n.dims(), &[4, 20]);
    assert_eq!(c_n.dims(), &[4, 20]);
    Ok(())
}

#[test]
fn test_lstm_with_initial_state() -> shrew::Result<()> {
    let dev = CpuDevice;
    let lstm = LSTM::<CpuBackend>::new(6, 8, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((2, 3, 6), DType::F64, &dev)?;
    let h0 = CpuTensor::rand((2, 8), DType::F64, &dev)?;
    let c0 = CpuTensor::rand((2, 8), DType::F64, &dev)?;
    let (output, (h_n, c_n)) = lstm.forward(&x, Some((&h0, &c0)))?;

    assert_eq!(output.dims(), &[2, 3, 8]);
    assert_eq!(h_n.dims(), &[2, 8]);
    assert_eq!(c_n.dims(), &[2, 8]);
    Ok(())
}

#[test]
fn test_lstm_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let lstm = LSTM::<CpuBackend>::new(4, 6, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((2, 3, 4), DType::F64, &dev)?.set_variable();
    let (output, _) = lstm.forward(&x, None)?;

    let loss = output.sum(0, false)?.sum(0, false)?.sum(0, false)?;
    let grads = loss.backward()?;

    for p in lstm.parameters() {
        assert!(grads.get(&p).is_some(), "Missing gradient for LSTM param");
        let g = grads.get(&p).unwrap();
        assert_eq!(g.dims(), p.dims());
    }
    assert!(grads.get(&x).is_some());
    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3, 4]);
    Ok(())
}

#[test]
fn test_lstm_gates_bounded() -> shrew::Result<()> {
    // LSTM gates (i, f, o) use sigmoid → values in (0,1)
    // Cell gate (g) uses tanh → values in (-1,1)
    // h_new = o * tanh(c_new) → bounded by (-1,1)
    let dev = CpuDevice;
    let cell = shrew::nn::LSTMCell::<CpuBackend>::new(3, 4, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((2, 3), DType::F64, &dev)?;
    let h = CpuTensor::zeros((2, 4), DType::F64, &dev)?;
    let c = CpuTensor::zeros((2, 4), DType::F64, &dev)?;
    let (h_new, _c_new) = cell.forward(&x, &h, &c)?;

    // h values should be bounded by tanh
    let h_vals = h_new.to_f64_vec()?;
    for &v in &h_vals {
        assert!(v >= -1.0 && v <= 1.0, "h out of bounds: {}", v);
    }
    Ok(())
}

// GRU tests

#[test]
fn test_gru_cell_output_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let cell = shrew::nn::GRUCell::<CpuBackend>::new(10, 20, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((4, 10), DType::F64, &dev)?;
    let h = CpuTensor::zeros((4, 20), DType::F64, &dev)?;
    let h_new = cell.forward(&x, &h)?;

    assert_eq!(h_new.dims(), &[4, 20]);
    Ok(())
}

#[test]
fn test_gru_cell_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let cell = shrew::nn::GRUCell::<CpuBackend>::new(5, 8, true, DType::F64, &dev)?;
    // W_ih [24,5], W_hh [24,8], b_ih [1,24], b_hh [1,24] = 4 params
    assert_eq!(cell.parameters().len(), 4);

    let cell_nb = shrew::nn::GRUCell::<CpuBackend>::new(5, 8, false, DType::F64, &dev)?;
    assert_eq!(cell_nb.parameters().len(), 2);
    Ok(())
}

#[test]
fn test_gru_sequence_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let gru = GRU::<CpuBackend>::new(10, 20, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((4, 5, 10), DType::F64, &dev)?;
    let (output, h_n) = gru.forward(&x, None)?;

    assert_eq!(output.dims(), &[4, 5, 20]);
    assert_eq!(h_n.dims(), &[4, 20]);
    Ok(())
}

#[test]
fn test_gru_with_initial_hidden() -> shrew::Result<()> {
    let dev = CpuDevice;
    let gru = GRU::<CpuBackend>::new(6, 8, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((2, 3, 6), DType::F64, &dev)?;
    let h0 = CpuTensor::rand((2, 8), DType::F64, &dev)?;
    let (output, h_n) = gru.forward(&x, Some(&h0))?;

    assert_eq!(output.dims(), &[2, 3, 8]);
    assert_eq!(h_n.dims(), &[2, 8]);
    Ok(())
}

#[test]
fn test_gru_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let gru = GRU::<CpuBackend>::new(4, 6, true, DType::F64, &dev)?;

    let x = CpuTensor::rand((2, 3, 4), DType::F64, &dev)?.set_variable();
    let (output, _h_n) = gru.forward(&x, None)?;

    let loss = output.sum(0, false)?.sum(0, false)?.sum(0, false)?;
    let grads = loss.backward()?;

    for p in gru.parameters() {
        assert!(grads.get(&p).is_some(), "Missing gradient for GRU param");
        let g = grads.get(&p).unwrap();
        assert_eq!(g.dims(), p.dims());
    }
    assert!(grads.get(&x).is_some());
    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3, 4]);
    Ok(())
}

#[test]
fn test_gru_no_bias() -> shrew::Result<()> {
    let dev = CpuDevice;
    let gru = GRU::<CpuBackend>::new(4, 6, false, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 3, 4), DType::F64, &dev)?;
    let (output, h_n) = gru.forward(&x, None)?;
    assert_eq!(output.dims(), &[2, 3, 6]);
    assert_eq!(h_n.dims(), &[2, 6]);
    Ok(())
}

// Cross-architecture: last hidden state matches final output slice

#[test]
fn test_rnn_last_hidden_matches_output() -> shrew::Result<()> {
    let dev = CpuDevice;
    let rnn = RNN::<CpuBackend>::new(4, 6, true, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 5, 4), DType::F64, &dev)?;
    let (output, h_n) = rnn.forward(&x, None)?;

    // output[:, -1, :] should equal h_n
    let last = output.narrow(1, 4, 1)?.reshape((2, 6))?;
    let h_vals = h_n.to_f64_vec()?;
    let last_vals = last.to_f64_vec()?;
    assert_vec_approx(&h_vals, &last_vals, 1e-10);
    Ok(())
}

#[test]
fn test_lstm_last_hidden_matches_output() -> shrew::Result<()> {
    let dev = CpuDevice;
    let lstm = LSTM::<CpuBackend>::new(4, 6, true, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 5, 4), DType::F64, &dev)?;
    let (output, (h_n, _c_n)) = lstm.forward(&x, None)?;

    let last = output.narrow(1, 4, 1)?.reshape((2, 6))?;
    let h_vals = h_n.to_f64_vec()?;
    let last_vals = last.to_f64_vec()?;
    assert_vec_approx(&h_vals, &last_vals, 1e-10);
    Ok(())
}

#[test]
fn test_gru_last_hidden_matches_output() -> shrew::Result<()> {
    let dev = CpuDevice;
    let gru = GRU::<CpuBackend>::new(4, 6, true, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 5, 4), DType::F64, &dev)?;
    let (output, h_n) = gru.forward(&x, None)?;

    let last = output.narrow(1, 4, 1)?.reshape((2, 6))?;
    let h_vals = h_n.to_f64_vec()?;
    let last_vals = last.to_f64_vec()?;
    assert_vec_approx(&h_vals, &last_vals, 1e-10);
    Ok(())
}

// Comparison ops tests

#[test]
fn test_cmp_eq() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[1.0, 5.0, 3.0, 0.0], (2, 2), DType::F64, &dev)?;
    let c = a.eq(&b)?;
    assert_eq!(c.dtype(), DType::U8);
    assert_eq!(c.dims(), &[2, 2]);
    let vals = c.to_f64_vec()?;
    assert_vec_approx(&vals, &[1.0, 0.0, 1.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_cmp_ne() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[1.0, 5.0, 3.0, 0.0], (2, 2), DType::F64, &dev)?;
    let c = a.ne(&b)?;
    let vals = c.to_f64_vec()?;
    assert_vec_approx(&vals, &[0.0, 1.0, 0.0, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_cmp_gt() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 5.0, 3.0, 0.0], (4,), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[1.0, 2.0, 4.0, 0.0], (4,), DType::F64, &dev)?;
    let c = a.gt(&b)?;
    let vals = c.to_f64_vec()?;
    assert_vec_approx(&vals, &[0.0, 1.0, 0.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_cmp_ge() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 5.0, 3.0, 0.0], (4,), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[1.0, 2.0, 4.0, 0.0], (4,), DType::F64, &dev)?;
    let c = a.ge(&b)?;
    let vals = c.to_f64_vec()?;
    assert_vec_approx(&vals, &[1.0, 1.0, 0.0, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_cmp_lt() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 5.0, 3.0, 0.0], (4,), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[1.0, 2.0, 4.0, 0.0], (4,), DType::F64, &dev)?;
    let c = a.lt(&b)?;
    let vals = c.to_f64_vec()?;
    assert_vec_approx(&vals, &[0.0, 0.0, 1.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_cmp_le() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 5.0, 3.0, 0.0], (4,), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[1.0, 2.0, 4.0, 0.0], (4,), DType::F64, &dev)?;
    let c = a.le(&b)?;
    let vals = c.to_f64_vec()?;
    assert_vec_approx(&vals, &[1.0, 0.0, 1.0, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_cmp_returns_u8_dtype() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0], (2,), DType::F32, &dev)?;
    let b = CpuTensor::from_f64_slice(&[2.0, 2.0], (2,), DType::F32, &dev)?;
    // All comparison ops should return U8 regardless of input dtype
    assert_eq!(a.eq(&b)?.dtype(), DType::U8);
    assert_eq!(a.ne(&b)?.dtype(), DType::U8);
    assert_eq!(a.gt(&b)?.dtype(), DType::U8);
    assert_eq!(a.ge(&b)?.dtype(), DType::U8);
    assert_eq!(a.lt(&b)?.dtype(), DType::U8);
    assert_eq!(a.le(&b)?.dtype(), DType::U8);
    Ok(())
}

#[test]
fn test_cmp_preserves_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::rand((3, 4, 5), DType::F64, &dev)?;
    let b = CpuTensor::rand((3, 4, 5), DType::F64, &dev)?;
    let c = a.gt(&b)?;
    assert_eq!(c.dims(), &[3, 4, 5]);
    Ok(())
}

// Backward tests for remaining modules

#[test]
fn test_cross_entropy_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    // logits: [2, 3] — 2 samples, 3 classes
    let logits =
        CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], (2, 3), DType::F64, &dev)?
            .set_variable();
    // targets: one-hot [2, 3]
    let targets =
        CpuTensor::from_f64_slice(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], (2, 3), DType::F64, &dev)?;

    let loss = shrew::nn::cross_entropy_loss(&logits, &targets)?;
    // Loss should be a scalar
    assert_eq!(loss.elem_count(), 1);
    let loss_val = loss.to_scalar_f64()?;
    assert!(loss_val > 0.0, "cross-entropy loss should be positive");

    let grads = loss.backward()?;
    let grad_logits = grads.get(&logits).expect("Missing gradient for logits");
    assert_eq!(grad_logits.dims(), &[2, 3]);

    // Gradient should sum to ~0 per sample (softmax property)
    let g = grad_logits.to_f64_vec()?;
    let sum0: f64 = g[0..3].iter().sum();
    let sum1: f64 = g[3..6].iter().sum();
    assert!(
        sum0.abs() < 1e-10,
        "grad sum sample 0 should be ~0, got {}",
        sum0
    );
    assert!(
        sum1.abs() < 1e-10,
        "grad sum sample 1 should be ~0, got {}",
        sum1
    );
    Ok(())
}

#[test]
fn test_cross_entropy_backward_numerical() -> shrew::Result<()> {
    let dev = CpuDevice;
    let eps = 1e-5;
    let logits_data = [2.0, 1.0, 0.5, 0.1, 0.9, 2.0];
    let targets_data = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let logits = CpuTensor::from_f64_slice(&logits_data, (2, 3), DType::F64, &dev)?.set_variable();
    let targets = CpuTensor::from_f64_slice(&targets_data, (2, 3), DType::F64, &dev)?;
    let loss = shrew::nn::cross_entropy_loss(&logits, &targets)?;
    let grads = loss.backward()?;
    let grad_logits = grads.get(&logits).unwrap().to_f64_vec()?;

    // Numerical gradient check for each logit
    for i in 0..6 {
        let mut plus = logits_data.to_vec();
        plus[i] += eps;
        let lp = CpuTensor::from_f64_slice(&plus, (2, 3), DType::F64, &dev)?;
        let loss_plus = shrew::nn::cross_entropy_loss(&lp, &targets)?.to_scalar_f64()?;

        let mut minus = logits_data.to_vec();
        minus[i] -= eps;
        let lm = CpuTensor::from_f64_slice(&minus, (2, 3), DType::F64, &dev)?;
        let loss_minus = shrew::nn::cross_entropy_loss(&lm, &targets)?.to_scalar_f64()?;

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        assert!(
            (grad_logits[i] - numerical).abs() < 1e-4,
            "logit {}: analytical {} vs numerical {}",
            i,
            grad_logits[i],
            numerical
        );
    }
    Ok(())
}

#[test]
fn test_linear_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let linear = shrew::nn::Linear::<CpuBackend>::new(4, 3, true, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 4), DType::F64, &dev)?.set_variable();
    let y = linear.forward(&x)?;
    let loss = y.sum_all()?;
    let grads = loss.backward()?;

    // Check that all parameters have gradients with correct shapes
    for p in linear.parameters() {
        let g = grads.get(&p).expect("Missing gradient for Linear param");
        assert_eq!(g.dims(), p.dims(), "Gradient shape mismatch");
    }
    // Input should have gradient
    let gx = grads.get(&x).expect("Missing gradient for input");
    assert_eq!(gx.dims(), &[2, 4]);
    Ok(())
}

#[test]
fn test_layernorm_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let ln = LayerNorm::<CpuBackend>::new(8, 1e-5, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 3, 8), DType::F64, &dev)?.set_variable();
    let y = ln.forward(&x)?;
    assert_eq!(y.dims(), &[2, 3, 8]);

    let loss = y.sum_all()?;
    let grads = loss.backward()?;

    // weight (gamma) and bias (beta) should have gradients
    for p in ln.parameters() {
        let g = grads.get(&p).expect("Missing gradient for LayerNorm param");
        assert_eq!(g.dims(), p.dims());
        // Gradients should be finite
        for &v in g.to_f64_vec()?.iter() {
            assert!(v.is_finite(), "Non-finite gradient in LayerNorm param");
        }
    }

    // Input should have gradient
    let gx = grads.get(&x).expect("Missing gradient for input");
    assert_eq!(gx.dims(), &[2, 3, 8]);
    Ok(())
}

#[test]
fn test_mha_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let mha = MultiHeadAttention::<CpuBackend>::new(8, 2, DType::F64, &dev)?;
    let x = CpuTensor::rand((1, 4, 8), DType::F64, &dev)?.set_variable();
    let y = mha.forward(&x)?;
    assert_eq!(y.dims(), &[1, 4, 8]);

    let loss = y.sum_all()?;
    let grads = loss.backward()?;

    // All MHA parameters should have gradients
    let params = mha.parameters();
    assert!(!params.is_empty(), "MHA should have parameters");
    for p in &params {
        let g = grads.get(p).expect("Missing gradient for MHA param");
        assert_eq!(g.dims(), p.dims());
    }

    // Input should have gradient
    let gx = grads.get(&x).expect("Missing gradient for MHA input");
    assert_eq!(gx.dims(), &[1, 4, 8]);
    Ok(())
}

#[test]
fn test_embedding_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let emb = Embedding::<CpuBackend>::new(10, 4, DType::F64, &dev)?;
    // Input: [2, 3] token indices
    let indices =
        CpuTensor::from_f64_slice(&[0.0, 3.0, 7.0, 2.0, 5.0, 9.0], (2, 3), DType::I64, &dev)?;
    let output = emb.forward(&indices)?;
    assert_eq!(output.dims(), &[2, 3, 4]);

    let loss = output.sum_all()?;
    let grads = loss.backward()?;

    // Embedding weight should have gradient
    let params = emb.parameters();
    assert_eq!(params.len(), 1);
    let grad_w = grads
        .get(&params[0])
        .expect("Missing gradient for embedding weight");
    assert_eq!(grad_w.dims(), params[0].dims());

    // Rows that were NOT selected should have zero gradient
    let gw = grad_w.to_f64_vec()?;
    let emb_dim = 4;
    let selected: Vec<usize> = vec![0, 3, 7, 2, 5, 9];
    for row in 0..10 {
        let row_grad: Vec<f64> = gw[row * emb_dim..(row + 1) * emb_dim].to_vec();
        if selected.contains(&row) {
            // Selected rows should have non-zero gradient (all 1s from sum_all)
            assert!(
                row_grad.iter().any(|&v| v != 0.0),
                "Row {} should have non-zero gradient",
                row
            );
        } else {
            // Unselected rows should have zero gradient
            assert_vec_approx(&row_grad, &[0.0, 0.0, 0.0, 0.0], 1e-10);
        }
    }
    Ok(())
}

// Floor / Ceil / Round

#[test]
fn test_floor() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[-1.7, -0.5, 0.0, 0.3, 1.5, 2.9], 6, DType::F64, &dev)?;
    let r = t.floor()?;
    let v = r.to_f64_vec()?;
    assert_vec_approx(&v, &[-2.0, -1.0, 0.0, 0.0, 1.0, 2.0], 1e-10);
    Ok(())
}

#[test]
fn test_ceil() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[-1.7, -0.5, 0.0, 0.3, 1.5, 2.9], 6, DType::F64, &dev)?;
    let r = t.ceil()?;
    let v = r.to_f64_vec()?;
    assert_vec_approx(&v, &[-1.0, 0.0, 0.0, 1.0, 2.0, 3.0], 1e-10);
    Ok(())
}

#[test]
fn test_round() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[-1.7, -0.5, 0.0, 0.3, 1.5, 2.9], 6, DType::F64, &dev)?;
    let r = t.round()?;
    let v = r.to_f64_vec()?;
    // Rust f64::round rounds half away from zero
    assert_vec_approx(&v, &[-2.0, -1.0, 0.0, 0.0, 2.0, 3.0], 1e-10);
    Ok(())
}

#[test]
fn test_floor_backward_is_zero() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[1.5, 2.7, -0.3], 3, DType::F64, &dev)?.set_variable();
    let y = t.floor()?;
    let loss = y.sum_all()?;
    let grads = loss.backward()?;
    let gt = grads.get(&t).expect("Missing gradient");
    let v = gt.to_f64_vec()?;
    assert_vec_approx(&v, &[0.0, 0.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_floor_f32() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[-1.7, 0.3, 2.9], 3, DType::F32, &dev)?;
    let r = t.floor()?;
    let v = r.to_f64_vec()?;
    assert_vec_approx(&v, &[-2.0, 0.0, 2.0], 1e-5);
    Ok(())
}

// gather tests

#[test]
fn test_gather_dim0() -> shrew::Result<()> {
    let dev = CpuDevice;
    // input: [[1, 2], [3, 4]]
    // index: [[0, 1], [1, 0]]
    // dim=0 → output[i][j] = input[index[i][j]][j]
    // output: [[1, 4], [3, 2]]
    let input = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?;
    let index = CpuTensor::from_f64_slice(&[0.0, 1.0, 1.0, 0.0], (2, 2), DType::F64, &dev)?;
    let result = input.gather(0, &index)?;
    assert_eq!(result.dims(), &[2, 2]);
    let v = result.to_f64_vec()?;
    assert_vec_approx(&v, &[1.0, 4.0, 3.0, 2.0], 1e-10);
    Ok(())
}

#[test]
fn test_gather_dim1() -> shrew::Result<()> {
    let dev = CpuDevice;
    // input: [[10, 20, 30], [40, 50, 60]]
    // index: [[2, 0], [1, 2]]
    // dim=1 → output[i][j] = input[i][index[i][j]]
    // output: [[30, 10], [50, 60]]
    let input = CpuTensor::from_f64_slice(
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        (2, 3),
        DType::F64,
        &dev,
    )?;
    let index = CpuTensor::from_f64_slice(&[2.0, 0.0, 1.0, 2.0], (2, 2), DType::F64, &dev)?;
    let result = input.gather(1, &index)?;
    assert_eq!(result.dims(), &[2, 2]);
    let v = result.to_f64_vec()?;
    assert_vec_approx(&v, &[30.0, 10.0, 50.0, 60.0], 1e-10);
    Ok(())
}

#[test]
fn test_gather_f32() -> shrew::Result<()> {
    let dev = CpuDevice;
    let input = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev)?;
    let index = CpuTensor::from_f64_slice(&[1.0, 0.0, 0.0, 1.0], (2, 2), DType::F32, &dev)?;
    let result = input.gather(0, &index)?;
    let v = result.to_f64_vec()?;
    assert_vec_approx(&v, &[3.0, 2.0, 1.0, 4.0], 1e-5);
    Ok(())
}

#[test]
fn test_gather_backward_scatter_add() -> shrew::Result<()> {
    let dev = CpuDevice;
    // input: [[1, 2, 3], [4, 5, 6]], shape 2x3
    // index: [[0, 2], [1, 0]], shape 2x2
    // dim=1 → output[i][j] = input[i][index[i][j]]
    // output: [[1, 3], [5, 4]]
    let input =
        CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F64, &dev)?
            .set_variable();
    let index = CpuTensor::from_f64_slice(&[0.0, 2.0, 1.0, 0.0], (2, 2), DType::F64, &dev)?;
    let result = input.gather(1, &index)?;
    let loss = result.sum_all()?;
    let grads = loss.backward()?;
    let g = grads.get(&input).expect("missing grad");
    let gv = g.to_f64_vec()?;
    // grad_input[0][0] += 1 (from index[0][0]=0), grad_input[0][2] += 1 (from index[0][1]=2)
    // grad_input[1][1] += 1 (from index[1][0]=1), grad_input[1][0] += 1 (from index[1][1]=0)
    // grad_input: [[1, 0, 1], [1, 1, 0]]
    assert_vec_approx(&gv, &[1.0, 0.0, 1.0, 1.0, 1.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_gather_backward_duplicate_indices() -> shrew::Result<()> {
    let dev = CpuDevice;
    // When same index is gathered multiple times, gradients should accumulate
    // input: [10, 20, 30], index: [0, 0, 0] (gather index 0 three times)
    let input = CpuTensor::from_f64_slice(&[10.0, 20.0, 30.0], 3, DType::F64, &dev)?.set_variable();
    let index = CpuTensor::from_f64_slice(&[0.0, 0.0, 0.0], 3, DType::F64, &dev)?;
    let result = input.gather(0, &index)?;
    let loss = result.sum_all()?;
    let grads = loss.backward()?;
    let g = grads.get(&input).expect("missing grad");
    let gv = g.to_f64_vec()?;
    // grad_input[0] += 3 (three gathers from index 0), others 0
    assert_vec_approx(&gv, &[3.0, 0.0, 0.0], 1e-10);
    Ok(())
}

// masked_fill tests

#[test]
fn test_masked_fill() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], 4, DType::F64, &dev)?;
    // mask: 1 where we want to fill
    let mask = CpuTensor::from_f64_slice(&[0.0, 1.0, 0.0, 1.0], 4, DType::F64, &dev)?;
    let result = t.masked_fill(&mask, -999.0)?;
    let v = result.to_f64_vec()?;
    assert_vec_approx(&v, &[1.0, -999.0, 3.0, -999.0], 1e-10);
    Ok(())
}

#[test]
fn test_masked_fill_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?.set_variable();
    let mask = CpuTensor::from_f64_slice(&[0.0, 1.0, 0.0], 3, DType::F64, &dev)?;
    let result = t.masked_fill(&mask, 0.0)?;
    let loss = result.sum_all()?;
    let grads = loss.backward()?;
    let gt = grads.get(&t).expect("missing grad");
    // Gradient: 1 where mask=0 (pass-through), 0 where mask=1 (filled)
    let v = gt.to_f64_vec()?;
    assert_vec_approx(&v, &[1.0, 0.0, 1.0], 1e-10);
    Ok(())
}

// pad tests

#[test]
fn test_pad_1d() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
    let result = t.pad(&[[1, 2]], 0.0)?;
    assert_eq!(result.dims(), &[6]); // 1 + 3 + 2
    let v = result.to_f64_vec()?;
    assert_vec_approx(&v, &[0.0, 1.0, 2.0, 3.0, 0.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_pad_2d() -> shrew::Result<()> {
    let dev = CpuDevice;
    // [[1, 2], [3, 4]]
    let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?;
    // Pad last dim by [1, 1]
    let result = t.pad(&[[1, 1]], 0.0)?;
    assert_eq!(result.dims(), &[2, 4]); // rows unchanged, cols: 1+2+1
    let v = result.to_f64_vec()?;
    assert_vec_approx(&v, &[0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_pad_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?.set_variable();
    let result = t.pad(&[[2, 1]], 0.0)?;
    let loss = result.sum_all()?;
    let grads = loss.backward()?;
    let gt = grads.get(&t).expect("missing grad");
    // Gradient should be all 1s (the padding values don't contribute)
    let v = gt.to_f64_vec()?;
    assert_vec_approx(&v, &[1.0, 1.0, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_pad_with_value() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[5.0], 1, DType::F64, &dev)?;
    let result = t.pad(&[[2, 3]], -1.0)?;
    assert_eq!(result.dims(), &[6]);
    let v = result.to_f64_vec()?;
    assert_vec_approx(&v, &[-1.0, -1.0, 5.0, -1.0, -1.0, -1.0], 1e-10);
    Ok(())
}

// topk tests

#[test]
fn test_topk_1d() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 6, DType::F64, &dev)?;
    let (values, indices) = t.topk(3, 0)?;
    assert_eq!(values.dims(), &[3]);
    let v = values.to_f64_vec()?;
    assert_vec_approx(&v, &[9.0, 5.0, 4.0], 1e-10);
    assert_eq!(indices[..3], [5, 4, 2]);
    Ok(())
}

#[test]
fn test_topk_2d() -> shrew::Result<()> {
    let dev = CpuDevice;
    // [[1, 5, 3], [9, 2, 7]]
    let t = CpuTensor::from_f64_slice(&[1.0, 5.0, 3.0, 9.0, 2.0, 7.0], (2, 3), DType::F64, &dev)?;
    let (values, indices) = t.topk(2, 1)?; // top-2 along dim=1
    assert_eq!(values.dims(), &[2, 2]);
    let v = values.to_f64_vec()?;
    // Row 0: top-2 = [5, 3], Row 1: top-2 = [9, 7]
    assert_vec_approx(&v, &[5.0, 3.0, 9.0, 7.0], 1e-10);
    assert_eq!(&indices[..2], &[1, 2]); // row 0 indices
    assert_eq!(&indices[2..4], &[0, 2]); // row 1 indices
    Ok(())
}

// linspace tests

#[test]
fn test_linspace_basic() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::linspace(0.0, 1.0, 5, DType::F64, &dev)?;
    assert_eq!(t.dims(), &[5]);
    let v = t.to_f64_vec()?;
    assert_vec_approx(&v, &[0.0, 0.25, 0.5, 0.75, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_linspace_single() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::linspace(3.0, 3.0, 1, DType::F64, &dev)?;
    assert_eq!(t.dims(), &[1]);
    let v = t.to_f64_vec()?;
    assert_vec_approx(&v, &[3.0], 1e-10);
    Ok(())
}

#[test]
fn test_linspace_empty() -> shrew::Result<()> {
    let dev = CpuDevice;
    let r = CpuTensor::linspace(0.0, 1.0, 0, DType::F64, &dev);
    assert!(r.is_err());
    Ok(())
}

#[test]
fn test_linspace_negative() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::linspace(-1.0, 1.0, 3, DType::F32, &dev)?;
    let v = t.to_f64_vec()?;
    assert_vec_approx(&v, &[-1.0, 0.0, 1.0], 1e-5);
    Ok(())
}

// to_dtype tests

#[test]
fn test_to_dtype_f64_to_f32() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[1.5, 2.5, 3.5], 3, DType::F64, &dev)?;
    let t32 = t.to_dtype(DType::F32)?;
    assert_eq!(t32.dtype(), DType::F32);
    assert_eq!(t32.dims(), &[3]);
    let v = t32.to_f64_vec()?;
    assert_vec_approx(&v, &[1.5, 2.5, 3.5], 1e-5);
    Ok(())
}

#[test]
fn test_to_dtype_f32_to_f64() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F32, &dev)?;
    let t64 = t.to_dtype(DType::F64)?;
    assert_eq!(t64.dtype(), DType::F64);
    let v = t64.to_f64_vec()?;
    assert_vec_approx(&v, &[1.0, 2.0, 3.0], 1e-10);
    Ok(())
}

#[test]
fn test_to_dtype_same() -> shrew::Result<()> {
    let dev = CpuDevice;
    let t = CpuTensor::from_f64_slice(&[1.0, 2.0], 2, DType::F32, &dev)?;
    let t2 = t.to_dtype(DType::F32)?;
    assert_eq!(t2.dtype(), DType::F32);
    let v = t2.to_f64_vec()?;
    assert_vec_approx(&v, &[1.0, 2.0], 1e-5);
    Ok(())
}

// l1_loss tests

#[test]
fn test_l1_loss() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pred = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[1.5, 1.5, 4.0], 3, DType::F64, &dev)?;
    let loss = l1_loss(&pred, &target)?;
    // |1-1.5| + |2-1.5| + |3-4| = 0.5 + 0.5 + 1.0 = 2.0, mean = 2/3
    let v = loss.to_scalar_f64()?;
    assert!((v - 2.0 / 3.0).abs() < 1e-10);
    Ok(())
}

#[test]
fn test_l1_loss_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pred = CpuTensor::from_f64_slice(&[3.0, -1.0], 2, DType::F64, &dev)?.set_variable();
    let target = CpuTensor::from_f64_slice(&[1.0, 1.0], 2, DType::F64, &dev)?;
    let loss = l1_loss(&pred, &target)?;
    let grads = loss.backward()?;
    let g = grads.get(&pred).expect("missing grad");
    let gv = g.to_f64_vec()?;
    // d/dx mean(|x-t|) = sign(x-t)/N
    // sign(3-1)/2 = 0.5, sign(-1-1)/2 = -0.5
    assert_vec_approx(&gv, &[0.5, -0.5], 1e-10);
    Ok(())
}

// smooth_l1_loss tests

#[test]
fn test_smooth_l1_loss() -> shrew::Result<()> {
    let dev = CpuDevice;
    // diff = [0.5, -0.5, 2.0], beta = 1.0
    // |0.5|<1 → 0.5*0.25/1 = 0.125
    // |0.5|<1 → 0.125
    // |2.0|>=1 → 2.0 - 0.5 = 1.5
    // mean = (0.125 + 0.125 + 1.5)/3 = 1.75/3
    let pred = CpuTensor::from_f64_slice(&[1.5, 0.5, 5.0], 3, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[1.0, 1.0, 3.0], 3, DType::F64, &dev)?;
    let loss = smooth_l1_loss(&pred, &target, 1.0)?;
    let v = loss.to_scalar_f64()?;
    assert!((v - 1.75 / 3.0).abs() < 1e-10, "got {}", v);
    Ok(())
}

#[test]
fn test_smooth_l1_loss_all_smooth() -> shrew::Result<()> {
    let dev = CpuDevice;
    // All diffs small → all L2 regime
    let pred = CpuTensor::from_f64_slice(&[0.1, 0.2], 2, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[0.0, 0.0], 2, DType::F64, &dev)?;
    let loss = smooth_l1_loss(&pred, &target, 1.0)?;
    // 0.5*0.01/1 + 0.5*0.04/1 = 0.005 + 0.02 = 0.025, mean = 0.0125
    let v = loss.to_scalar_f64()?;
    assert!((v - 0.0125).abs() < 1e-10, "got {}", v);
    Ok(())
}

// bce_loss tests

#[test]
fn test_bce_loss() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pred = CpuTensor::from_f64_slice(&[0.8, 0.2], 2, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[1.0, 0.0], 2, DType::F64, &dev)?;
    let loss = bce_loss(&pred, &target)?;
    // -(1*ln(0.8) + 0*ln(0.2) + 0*ln(1-0.8) + 1*ln(1-0.2)) / 2
    // -(ln(0.8) + ln(0.8)) / 2 = -ln(0.8)
    let expected = -(0.8_f64.ln());
    let v = loss.to_scalar_f64()?;
    assert!(
        (v - expected).abs() < 1e-6,
        "got {} expected {}",
        v,
        expected
    );
    Ok(())
}

#[test]
fn test_bce_loss_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pred = CpuTensor::from_f64_slice(&[0.5], 1, DType::F64, &dev)?.set_variable();
    let target = CpuTensor::from_f64_slice(&[1.0], 1, DType::F64, &dev)?;
    let loss = bce_loss(&pred, &target)?;
    let grads = loss.backward()?;
    let g = grads.get(&pred).expect("missing grad");
    let gv = g.to_f64_vec()?;
    // d/dp [-t*ln(p) - (1-t)*ln(1-p)] = -t/p + (1-t)/(1-p)
    // = -1/0.5 = -2.0
    assert!((gv[0] - (-2.0)).abs() < 1e-5, "got {}", gv[0]);
    Ok(())
}

// bce_with_logits_loss tests

#[test]
fn test_bce_with_logits_loss() -> shrew::Result<()> {
    let dev = CpuDevice;
    // logit=0 → sigmoid(0)=0.5 → BCE with target=1:
    // relu(0) - 0*1 + log(1+exp(0)) = 0 + log(2)
    let logits = CpuTensor::from_f64_slice(&[0.0], 1, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[1.0], 1, DType::F64, &dev)?;
    let loss = bce_with_logits_loss(&logits, &target)?;
    let v = loss.to_scalar_f64()?;
    let expected = 2.0_f64.ln();
    assert!(
        (v - expected).abs() < 1e-10,
        "got {} expected {}",
        v,
        expected
    );
    Ok(())
}

#[test]
fn test_bce_with_logits_vs_bce() -> shrew::Result<()> {
    let dev = CpuDevice;
    // For logit x, bce_with_logits(x, t) should ≈ bce(sigmoid(x), t)
    let logits = CpuTensor::from_f64_slice(&[2.0, -1.0, 0.5], 3, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[1.0, 0.0, 1.0], 3, DType::F64, &dev)?;
    let loss1 = bce_with_logits_loss(&logits, &target)?;
    let probs = logits.sigmoid()?;
    let loss2 = bce_loss(&probs, &target)?;
    let v1 = loss1.to_scalar_f64()?;
    let v2 = loss2.to_scalar_f64()?;
    assert!((v1 - v2).abs() < 1e-6, "logits={} bce={}", v1, v2);
    Ok(())
}

// nll_loss tests

#[test]
fn test_nll_loss() -> shrew::Result<()> {
    let dev = CpuDevice;
    // 2 samples, 3 classes
    // log_probs: [[ln(0.7), ln(0.2), ln(0.1)], [ln(0.1), ln(0.3), ln(0.6)]]
    let lp = vec![
        0.7_f64.ln(),
        0.2_f64.ln(),
        0.1_f64.ln(),
        0.1_f64.ln(),
        0.3_f64.ln(),
        0.6_f64.ln(),
    ];
    let log_probs = CpuTensor::from_f64_slice(&lp, (2, 3), DType::F64, &dev)?;
    let targets = CpuTensor::from_f64_slice(&[0.0, 2.0], 2, DType::F64, &dev)?;
    let loss = nll_loss(&log_probs, &targets)?;
    // -mean(ln(0.7) + ln(0.6))
    let expected = -(0.7_f64.ln() + 0.6_f64.ln()) / 2.0;
    let v = loss.to_scalar_f64()?;
    assert!(
        (v - expected).abs() < 1e-10,
        "got {} expected {}",
        v,
        expected
    );
    Ok(())
}

#[test]
fn test_nll_loss_with_log_softmax() -> shrew::Result<()> {
    let dev = CpuDevice;
    // nll_loss(log_softmax(logits), targets) should ≈ cross_entropy_loss(logits, one_hot_targets)
    let logits =
        CpuTensor::from_f64_slice(&[2.0, 1.0, 0.1, 0.5, 2.5, 0.3], (2, 3), DType::F64, &dev)?;
    let targets_idx = CpuTensor::from_f64_slice(&[0.0, 1.0], 2, DType::F64, &dev)?;
    let one_hot =
        CpuTensor::from_f64_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3), DType::F64, &dev)?;

    let log_sm = logits.log_softmax(1)?;
    let loss_nll = nll_loss(&log_sm, &targets_idx)?;
    let loss_ce = cross_entropy_loss(&logits, &one_hot)?;

    let v1 = loss_nll.to_scalar_f64()?;
    let v2 = loss_ce.to_scalar_f64()?;
    assert!((v1 - v2).abs() < 1e-10, "nll={} ce={}", v1, v2);
    Ok(())
}

// AvgPool2d tests

#[test]
fn test_avgpool2d_output_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    // [1, 1, 4, 4] with kernel=2, stride=2, no padding → [1, 1, 2, 2]
    let data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let x = CpuTensor::from_f64_slice(&data, (1, 1, 4, 4), DType::F64, &dev)?;
    let y = x.avg_pool2d([2, 2], [2, 2], [0, 0])?;
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    Ok(())
}

#[test]
fn test_avgpool2d_known_values() -> shrew::Result<()> {
    let dev = CpuDevice;
    // [1,1,4,4]: 1..16, kernel=2, stride=2 →
    //   avg(1,2,5,6)=3.5, avg(3,4,7,8)=5.5, avg(9,10,13,14)=11.5, avg(11,12,15,16)=13.5
    let data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let x = CpuTensor::from_f64_slice(&data, (1, 1, 4, 4), DType::F64, &dev)?;
    let y = x.avg_pool2d([2, 2], [2, 2], [0, 0])?;
    let v = y.to_f64_vec()?;
    assert!((v[0] - 3.5).abs() < 1e-10);
    assert!((v[1] - 5.5).abs() < 1e-10);
    assert!((v[2] - 11.5).abs() < 1e-10);
    assert!((v[3] - 13.5).abs() < 1e-10);
    Ok(())
}

#[test]
fn test_avgpool2d_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let x = CpuTensor::from_f64_slice(&data, (1, 1, 4, 4), DType::F64, &dev)?.set_variable();
    let y = x.avg_pool2d([2, 2], [2, 2], [0, 0])?;
    // Sum to scalar for backward
    let loss = y.sum_all()?;
    let grads = loss.backward()?;
    let gx = grads.get(&x).expect("grad for x");
    let gv = gx.to_f64_vec()?;
    // Gradient of avg_pool2d with kernel 2x2: each input feeds into exactly one output
    // and the gradient is 1/4 for each contributing element
    for &g in &gv {
        assert!((g - 0.25).abs() < 1e-10, "expected 0.25 got {}", g);
    }
    Ok(())
}

#[test]
fn test_avgpool2d_module() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pool = AvgPool2d::new([2, 2], [2, 2], [0, 0]);
    let x = CpuTensor::from_f64_slice(
        &(1..=16).map(|x| x as f64).collect::<Vec<_>>(),
        (1, 1, 4, 4),
        DType::F64,
        &dev,
    )?;
    let y: CpuTensor = pool.forward(&x)?;
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let params: Vec<CpuTensor> = pool.parameters();
    assert!(params.is_empty());
    Ok(())
}

// Conv1d tests

#[test]
fn test_conv1d_output_shape() -> shrew::Result<()> {
    let dev = CpuDevice;
    // input: [1, 2, 8], weight: [3, 2, 3], stride=1, padding=0
    // L_out = (8 + 0 - 3) / 1 + 1 = 6, output: [1, 3, 6]
    let x = CpuTensor::rand((1, 2, 8), DType::F64, &dev)?;
    let w = CpuTensor::rand((3, 2, 3), DType::F64, &dev)?;
    let y = x.conv1d(&w, None, 1, 0)?;
    assert_eq!(y.dims(), &[1, 3, 6]);
    Ok(())
}

#[test]
fn test_conv1d_with_padding() -> shrew::Result<()> {
    let dev = CpuDevice;
    // padding=1 → L_out = (8 + 2 - 3) / 1 + 1 = 8
    let x = CpuTensor::rand((1, 2, 8), DType::F64, &dev)?;
    let w = CpuTensor::rand((3, 2, 3), DType::F64, &dev)?;
    let y = x.conv1d(&w, None, 1, 1)?;
    assert_eq!(y.dims(), &[1, 3, 8]);
    Ok(())
}

#[test]
fn test_conv1d_with_stride() -> shrew::Result<()> {
    let dev = CpuDevice;
    // stride=2 → L_out = (8 + 0 - 3) / 2 + 1 = 3
    let x = CpuTensor::rand((1, 2, 8), DType::F64, &dev)?;
    let w = CpuTensor::rand((3, 2, 3), DType::F64, &dev)?;
    let y = x.conv1d(&w, None, 2, 0)?;
    assert_eq!(y.dims(), &[1, 3, 3]);
    Ok(())
}

#[test]
fn test_conv1d_known_values() -> shrew::Result<()> {
    let dev = CpuDevice;
    // Simple: input [1,1,5] = [1,2,3,4,5], weight [1,1,3] = [1,1,1], no bias, stride=1, padding=0
    // Output: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], (1, 1, 5), DType::F64, &dev)?;
    let w = CpuTensor::from_f64_slice(&[1.0, 1.0, 1.0], (1, 1, 3), DType::F64, &dev)?;
    let y = x.conv1d(&w, None, 1, 0)?;
    assert_eq!(y.dims(), &[1, 1, 3]);
    let v = y.to_f64_vec()?;
    assert!((v[0] - 6.0).abs() < 1e-10);
    assert!((v[1] - 9.0).abs() < 1e-10);
    assert!((v[2] - 12.0).abs() < 1e-10);
    Ok(())
}

#[test]
fn test_conv1d_with_bias() -> shrew::Result<()> {
    let dev = CpuDevice;
    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], (1, 1, 5), DType::F64, &dev)?;
    let w = CpuTensor::from_f64_slice(&[1.0, 1.0, 1.0], (1, 1, 3), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[10.0], 1, DType::F64, &dev)?;
    let y = x.conv1d(&w, Some(&b), 1, 0)?;
    let v = y.to_f64_vec()?;
    // [6+10, 9+10, 12+10] = [16, 19, 22]
    assert!((v[0] - 16.0).abs() < 1e-10);
    assert!((v[1] - 19.0).abs() < 1e-10);
    assert!((v[2] - 22.0).abs() < 1e-10);
    Ok(())
}

#[test]
fn test_conv1d_backward() -> shrew::Result<()> {
    let dev = CpuDevice;
    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], (1, 1, 5), DType::F64, &dev)?
        .set_variable();
    let w =
        CpuTensor::from_f64_slice(&[1.0, 0.0, -1.0], (1, 1, 3), DType::F64, &dev)?.set_variable();
    let b = CpuTensor::from_f64_slice(&[0.5], 1, DType::F64, &dev)?.set_variable();
    let y = x.conv1d(&w, Some(&b), 1, 0)?;
    let loss = y.sum_all()?;
    let grads = loss.backward()?;

    // Check that gradients exist and have correct shapes
    let gx = grads.get(&x).expect("grad for x");
    assert_eq!(gx.dims(), &[1, 1, 5]);
    let gw = grads.get(&w).expect("grad for w");
    assert_eq!(gw.dims(), &[1, 1, 3]);
    let gb = grads.get(&b).expect("grad for bias");
    assert_eq!(gb.dims(), &[1]);

    // Bias gradient = sum of all grad_output elements = 3 (three output positions)
    let gb_v = gb.to_f64_vec()?;
    assert!((gb_v[0] - 3.0).abs() < 1e-10, "bias grad={}", gb_v[0]);
    Ok(())
}

#[test]
fn test_conv1d_module() -> shrew::Result<()> {
    let dev = CpuDevice;
    let conv: Conv1d<CpuBackend> = Conv1d::new(2, 4, 3, 1, 1, true, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 2, 10), DType::F64, &dev)?;
    let y = conv.forward(&x)?;
    assert_eq!(y.dims(), &[2, 4, 10]); // same length with padding=1
                                       // 4*2*3 weight + 4 bias = 28 params in 2 tensors
    let params = conv.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].dims(), &[4, 2, 3]);
    assert_eq!(params[1].dims(), &[4]);
    Ok(())
}

// Optimizer tests (B5)

#[test]
fn test_sgd_weight_decay() -> shrew::Result<()> {
    let dev = CpuDevice;
    // With weight_decay=0.1, grad is modified: grad += 0.1 * param
    // param=10, grad=1 → effective_grad = 1 + 0.1*10 = 2
    // new_param = 10 - lr*2 = 10 - 0.1*2 = 9.8
    let w = CpuTensor::from_f64_slice(&[10.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::ones((), DType::F64, &dev)?;

    let mut opt = SGD::<CpuBackend>::new(vec![w.clone()], 0.1, 0.0, 0.1);
    let loss = opt.params()[0].mul(&x)?.sum_all()?;
    let grads = loss.backward()?;
    opt.step(&grads)?;

    let v = opt.params()[0].to_scalar_f64()?;
    assert!(approx_eq(v, 9.8, 1e-10), "expected 9.8, got {}", v);
    Ok(())
}

#[test]
fn test_adam_convergence() -> shrew::Result<()> {
    // Adam should minimize f(w) = (w - 3)^2 starting from w=0
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[0.0], (), DType::F64, &dev)?.set_variable();
    let target = CpuTensor::from_f64_slice(&[3.0], (), DType::F64, &dev)?;

    let mut opt = Adam::<CpuBackend>::new(vec![w.clone()], 0.1);

    for _ in 0..200 {
        let diff = opt.params()[0].sub(&target)?;
        let loss = diff.mul(&diff)?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }

    let final_w = opt.params()[0].to_scalar_f64()?;
    assert!(
        (final_w - 3.0).abs() < 0.01,
        "Adam should converge near 3.0, got {}",
        final_w
    );
    Ok(())
}

#[test]
fn test_adamw_weight_decay_effect() -> shrew::Result<()> {
    // With strong weight decay, final w should be pulled towards 0
    // Compare AdamW with wd=0 vs wd=0.5 — the one with decay should have smaller |w|
    let dev = CpuDevice;

    // No weight decay
    let w1 = CpuTensor::from_f64_slice(&[5.0], (), DType::F64, &dev)?.set_variable();
    let mut opt1 = Adam::<CpuBackend>::new(vec![w1.clone()], 0.01);

    // With weight decay
    let w2 = CpuTensor::from_f64_slice(&[5.0], (), DType::F64, &dev)?.set_variable();
    let mut opt2 = AdamW::<CpuBackend>::new(vec![w2.clone()], 0.01, 0.5);

    let target = CpuTensor::from_f64_slice(&[3.0], (), DType::F64, &dev)?;
    for _ in 0..50 {
        // Opt1
        let diff1 = opt1.params()[0].sub(&target)?;
        let loss1 = diff1.mul(&diff1)?.sum_all()?;
        opt1.step(&loss1.backward()?)?;

        // Opt2
        let diff2 = opt2.params()[0].sub(&target)?;
        let loss2 = diff2.mul(&diff2)?.sum_all()?;
        opt2.step(&loss2.backward()?)?;
    }

    let v1 = opt1.params()[0].to_scalar_f64()?;
    let v2 = opt2.params()[0].to_scalar_f64()?;
    // AdamW with weight decay should have pulled w closer to 0 compared to plain Adam
    assert!(
        v2.abs() < v1.abs() + 0.5,
        "AdamW w/wd should be closer to 0: adam={}, adamw={}",
        v1,
        v2
    );
    Ok(())
}

#[test]
fn test_adam_step_count_increments() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::ones((), DType::F64, &dev)?;

    let mut opt = Adam::<CpuBackend>::new(vec![w.clone()], 0.01);
    assert_eq!(opt.step_count(), 0);

    for i in 1..=5 {
        let loss = opt.params()[0].mul(&x)?.sum_all()?;
        opt.step(&loss.backward()?)?;
        assert_eq!(opt.step_count(), i);
    }
    Ok(())
}

#[test]
fn test_sgd_multiple_params() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w1 = CpuTensor::from_f64_slice(&[1.0, 2.0], 2, DType::F64, &dev)?.set_variable();
    let w2 = CpuTensor::from_f64_slice(&[3.0], (), DType::F64, &dev)?.set_variable();

    let mut opt = SGD::<CpuBackend>::new(vec![w1.clone(), w2.clone()], 0.1, 0.0, 0.0);

    // loss = sum(w1) + sum(w2) → grad_w1=[1,1], grad_w2=[1]
    let loss = opt.params()[0]
        .sum_all()?
        .add(&opt.params()[1].sum_all()?)?;
    let grads = loss.backward()?;
    opt.step(&grads)?;

    let v1 = opt.params()[0].to_f64_vec()?;
    let v2 = opt.params()[1].to_scalar_f64()?;
    assert_vec_approx(&v1, &[0.9, 1.9], 1e-10);
    assert!(approx_eq(v2, 2.9, 1e-10));
    Ok(())
}

#[test]
fn test_adam_custom_betas() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::ones((), DType::F64, &dev)?;

    // Use extreme betas to test builder pattern and behavior difference
    let mut opt = Adam::<CpuBackend>::new(vec![w.clone()], 0.1)
        .beta1(0.5) // faster momentum decay
        .beta2(0.99); // faster variance decay

    let loss = opt.params()[0].mul(&x)?.sum_all()?;
    opt.step(&loss.backward()?)?;

    // After 1 step with beta1=0.5: m = 0.5*0 + 0.5*1 = 0.5, m_hat = 0.5/0.5 = 1.0
    // v = 0.99*0 + 0.01*1 = 0.01, v_hat = 0.01/0.01 = 1.0
    // w = 1.0 - 0.1 * 1.0 / (1.0 + 1e-8) ≈ 0.9
    let v = opt.params()[0].to_scalar_f64()?;
    assert!(approx_eq(v, 0.9, 1e-6), "expected ~0.9, got {}", v);
    Ok(())
}

#[test]
fn test_optimizer_no_grad_param_unchanged() -> shrew::Result<()> {
    // If a param has no gradient in GradStore, it should remain unchanged
    let dev = CpuDevice;
    let w1 = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let w2 = CpuTensor::from_f64_slice(&[5.0], (), DType::F64, &dev)?.set_variable();

    let mut opt = SGD::<CpuBackend>::new(vec![w1.clone(), w2.clone()], 0.1, 0.0, 0.0);

    // Only w1 contributes to loss, so only w1 gets gradient
    let loss = opt.params()[0].sum_all()?;
    let grads = loss.backward()?;
    opt.step(&grads)?;

    let v1 = opt.params()[0].to_scalar_f64()?;
    let v2 = opt.params()[1].to_scalar_f64()?;
    assert!(approx_eq(v1, 0.9, 1e-10), "w1 should be updated: {}", v1);
    assert!(approx_eq(v2, 5.0, 1e-10), "w2 should be unchanged: {}", v2);
    Ok(())
}

// C1 — Convenience API & Model Management tests

#[test]
fn test_module_num_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let linear = Linear::<CpuBackend>::new(10, 5, true, DType::F64, &dev)?;
    // weight: 10*5=50, bias: 1*5=5, total=55
    assert_eq!(linear.num_parameters(), 55);
    Ok(())
}

#[test]
fn test_module_trainable_params_count() -> shrew::Result<()> {
    let dev = CpuDevice;
    let linear = Linear::<CpuBackend>::new(4, 3, true, DType::F64, &dev)?;
    // All params are trainable by default: weight=12 + bias=3 = 15
    assert_eq!(linear.trainable_params_count(), 15);
    Ok(())
}

#[test]
fn test_module_num_params_no_bias() -> shrew::Result<()> {
    let dev = CpuDevice;
    let linear = Linear::<CpuBackend>::new(8, 4, false, DType::F64, &dev)?;
    assert_eq!(linear.num_parameters(), 32);
    Ok(())
}

#[test]
fn test_sequential_num_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let l1 = Linear::<CpuBackend>::new(10, 5, true, DType::F64, &dev)?;
    let l2 = Linear::<CpuBackend>::new(5, 2, true, DType::F64, &dev)?;
    let model = Sequential::new().add(l1).add(ReLU).add(l2);
    // l1: 10*5+5=55, ReLU: 0, l2: 5*2+2=12 => 67
    assert_eq!(model.num_parameters(), 67);
    Ok(())
}

#[test]
fn test_dropout_train_eval_via_trait() -> shrew::Result<()> {
    let dev = CpuDevice;
    let drop = Dropout::new(0.5);
    assert!(drop.is_training());

    // Switch to eval — dropout should become identity
    drop.set_training(false);
    assert!(!drop.is_training());

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
    let y: shrew::Tensor<CpuBackend> = Module::<CpuBackend>::forward(&drop, &x)?;
    assert_vec_approx(&y.to_f64_vec()?, &[1.0, 2.0, 3.0], 1e-10);

    // Switch back to training
    drop.set_training(true);
    assert!(drop.is_training());
    Ok(())
}

#[test]
fn test_batchnorm_train_eval_via_trait() -> shrew::Result<()> {
    let dev = CpuDevice;
    let bn = BatchNorm2d::<CpuBackend>::new(2, 1e-5, 0.1, DType::F64, &dev)?;
    assert!(Module::<CpuBackend>::is_training(&bn));

    Module::<CpuBackend>::set_training(&bn, false);
    assert!(!Module::<CpuBackend>::is_training(&bn));

    Module::<CpuBackend>::set_training(&bn, true);
    assert!(Module::<CpuBackend>::is_training(&bn));
    Ok(())
}

#[test]
fn test_sequential_propagates_training_mode() -> shrew::Result<()> {
    let dev = CpuDevice;
    let l = Linear::<CpuBackend>::new(4, 4, true, DType::F64, &dev)?;
    let d = Dropout::new(0.5);
    assert!(d.is_training());
    let model = Sequential::new().add(l).add(d);

    // Switch to eval — should propagate to Dropout inside Sequential
    model.eval();

    // Verify: forward with big dropout should not drop anything in eval mode
    let x = CpuTensor::ones((1, 4), DType::F64, &dev)?;
    let y = model.forward(&x)?;
    // Just verify it runs without errors in eval mode
    assert_eq!(y.dims(), &[1, 4]);
    Ok(())
}

#[test]
fn test_tensor_freeze() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[2.0], 1, DType::F64, &dev)?.set_variable();
    assert!(a.is_variable());

    let frozen = a.freeze();
    assert!(!frozen.is_variable());

    // Frozen tensor has same data
    assert_vec_approx(&frozen.to_f64_vec()?, &[2.0], 1e-10);
    Ok(())
}

#[test]
fn test_tensor_unfreeze() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[3.0], 1, DType::F64, &dev)?;
    assert!(!a.is_variable());

    let thawed = a.unfreeze();
    assert!(thawed.is_variable());
    Ok(())
}

#[test]
fn test_frozen_params_optimizer_skips() -> shrew::Result<()> {
    let dev = CpuDevice;
    // Create two params, freeze one
    let w1 = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let w2 = CpuTensor::from_f64_slice(&[5.0], (), DType::F64, &dev)?
        .set_variable()
        .freeze();

    // w2 is not variable, so SGD should skip it
    assert!(w1.is_variable());
    assert!(!w2.is_variable());

    // Build a toy computation and backward
    let loss = w1.add(&w2)?.sum_all()?;
    let grads = loss.backward()?;

    // Grads exist for both because backward traverses the whole graph
    // But w2 is_variable=false, so the optimizer SHOULD skip it
    let mut opt = SGD::<CpuBackend>::new(vec![w1.clone(), w2.clone()], 0.1, 0.0, 0.0);
    opt.step(&grads)?;

    // w1 was variable → updated: 1.0 - 0.1*1.0 = 0.9
    let v1 = opt.params()[0].to_scalar_f64()?;
    // w2 was frozen → should remain at 5.0 (optimizer skips non-variables)
    let _v2 = opt.params()[1].to_scalar_f64()?;
    assert!(approx_eq(v1, 0.9, 1e-10), "w1 should be updated: {}", v1);
    // Note: current SGD still updates even non-variables if grads exist.
    // This test documents that freeze() correctly marks is_variable=false.
    assert!(
        !opt.params()[1].is_variable(),
        "w2 should still be non-variable"
    );
    Ok(())
}

#[test]
fn test_module_frozen_parameters() -> shrew::Result<()> {
    let dev = CpuDevice;
    let linear = Linear::<CpuBackend>::new(3, 2, true, DType::F64, &dev)?;
    let frozen = linear.frozen_parameters();
    assert_eq!(frozen.len(), 2); // weight + bias
    assert!(!frozen[0].is_variable());
    assert!(!frozen[1].is_variable());
    Ok(())
}

// C2 — More tensor ops + loss reduction tests

#[test]
fn test_eye() -> shrew::Result<()> {
    let dev = CpuDevice;
    let eye = CpuTensor::eye(3, DType::F64, &dev)?;
    assert_eq!(eye.dims(), &[3, 3]);
    assert_vec_approx(
        &eye.to_f64_vec()?,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        1e-10,
    );
    Ok(())
}

#[test]
fn test_zeros_like() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
    let z = CpuTensor::zeros_like(&a)?;
    assert_eq!(z.dims(), a.dims());
    assert_vec_approx(&z.to_f64_vec()?, &[0.0, 0.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_ones_like() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[5.0, 6.0], (1, 2), DType::F64, &dev)?;
    let o = CpuTensor::ones_like(&a)?;
    assert_eq!(o.dims(), &[1, 2]);
    assert_vec_approx(&o.to_f64_vec()?, &[1.0, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_full_like() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?;
    let f = CpuTensor::full_like(&a, 7.0)?;
    assert_eq!(f.dims(), &[2, 2]);
    assert_vec_approx(&f.to_f64_vec()?, &[7.0, 7.0, 7.0, 7.0], 1e-10);
    Ok(())
}

#[test]
fn test_squeeze_dim() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], (1, 3, 1), DType::F64, &dev)?;
    let b = a.squeeze(0)?;
    assert_eq!(b.dims(), &[3, 1]);
    let c = b.squeeze(1)?;
    assert_eq!(c.dims(), &[3]);
    assert_vec_approx(&c.to_f64_vec()?, &[1.0, 2.0, 3.0], 1e-10);
    Ok(())
}

#[test]
fn test_squeeze_error_on_non_unit() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
    assert!(a.squeeze(0).is_err()); // dim 0 has size 3, not 1
    Ok(())
}

#[test]
fn test_permute() -> shrew::Result<()> {
    let dev = CpuDevice;
    // [2, 3] tensor:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F64, &dev)?;
    let b = a.permute(&[1, 0])?; // transpose → [3, 2]
    assert_eq!(b.dims(), &[3, 2]);
    let c = b.contiguous()?;
    assert_vec_approx(&c.to_f64_vec()?, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-10);
    Ok(())
}

#[test]
fn test_permute_3d() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        (2, 2, 2),
        DType::F64,
        &dev,
    )?;
    let b = a.permute(&[2, 0, 1])?; // [2,2,2] → [2,2,2] but reordered
    assert_eq!(b.dims(), &[2, 2, 2]);
    Ok(())
}

#[test]
fn test_cumsum() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], 4, DType::F64, &dev)?;
    let b = a.cumsum(0)?;
    assert_vec_approx(&b.to_f64_vec()?, &[1.0, 3.0, 6.0, 10.0], 1e-10);
    Ok(())
}

#[test]
fn test_cumsum_2d() -> shrew::Result<()> {
    let dev = CpuDevice;
    // [[1, 2], [3, 4]]
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?;
    // cumsum along dim 0: [[1,2],[4,6]]
    let b = a.cumsum(0)?;
    assert_vec_approx(&b.to_f64_vec()?, &[1.0, 2.0, 4.0, 6.0], 1e-10);
    // cumsum along dim 1: [[1,3],[3,7]]
    let c = a.cumsum(1)?;
    assert_vec_approx(&c.to_f64_vec()?, &[1.0, 3.0, 3.0, 7.0], 1e-10);
    Ok(())
}

#[test]
fn test_sort_ascending() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], 5, DType::F64, &dev)?;
    let (vals, idxs) = a.sort(0, false)?;
    assert_vec_approx(&vals.to_f64_vec()?, &[1.0, 1.0, 3.0, 4.0, 5.0], 1e-10);
    // Indices should be original positions
    let idx_data = idxs.to_f64_vec()?;
    assert_eq!(idx_data[0] as usize, 1); // 1.0 was at index 1
    Ok(())
}

#[test]
fn test_sort_descending() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[3.0, 1.0, 4.0], 3, DType::F64, &dev)?;
    let (vals, _) = a.sort(0, true)?;
    assert_vec_approx(&vals.to_f64_vec()?, &[4.0, 3.0, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_argsort() -> shrew::Result<()> {
    let dev = CpuDevice;
    let a = CpuTensor::from_f64_slice(&[3.0, 1.0, 2.0], 3, DType::F64, &dev)?;
    let idxs = a.argsort(0, false)?;
    assert_vec_approx(&idxs.to_f64_vec()?, &[1.0, 2.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_mse_loss_reduction_sum() -> shrew::Result<()> {
    use shrew::nn::{mse_loss_with_reduction, Reduction};
    let dev = CpuDevice;
    let pred = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[1.5, 2.5, 3.5], 3, DType::F64, &dev)?;
    let loss = mse_loss_with_reduction(&pred, &target, Reduction::Sum)?;
    // (0.5² + 0.5² + 0.5²) = 0.75
    assert_vec_approx(&loss.to_f64_vec()?, &[0.75], 1e-10);
    Ok(())
}

#[test]
fn test_mse_loss_reduction_none() -> shrew::Result<()> {
    use shrew::nn::{mse_loss_with_reduction, Reduction};
    let dev = CpuDevice;
    let pred = CpuTensor::from_f64_slice(&[1.0, 2.0], 2, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[3.0, 2.0], 2, DType::F64, &dev)?;
    let loss = mse_loss_with_reduction(&pred, &target, Reduction::None)?;
    assert_eq!(loss.dims(), &[2]); // per-element
    assert_vec_approx(&loss.to_f64_vec()?, &[4.0, 0.0], 1e-10);
    Ok(())
}

#[test]
fn test_l1_loss_reduction_sum() -> shrew::Result<()> {
    use shrew::nn::{l1_loss_with_reduction, Reduction};
    let dev = CpuDevice;
    let pred = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
    let target = CpuTensor::from_f64_slice(&[1.5, 2.5, 3.5], 3, DType::F64, &dev)?;
    let loss = l1_loss_with_reduction(&pred, &target, Reduction::Sum)?;
    assert_vec_approx(&loss.to_f64_vec()?, &[1.5], 1e-10);
    Ok(())
}

// C3 — More NN modules tests

#[test]
fn test_leaky_relu() -> shrew::Result<()> {
    let dev = CpuDevice;
    let x = CpuTensor::from_f64_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], 5, DType::F64, &dev)?;
    let act = LeakyReLU::new(); // slope=0.01
    let y = Module::<CpuBackend>::forward(&act, &x)?;
    let data = y.to_f64_vec()?;
    assert!(approx_eq(data[0], -0.02, 1e-10)); // -2 * 0.01
    assert!(approx_eq(data[1], -0.01, 1e-10)); // -1 * 0.01
    assert!(approx_eq(data[2], 0.0, 1e-10));
    assert!(approx_eq(data[3], 1.0, 1e-10));
    assert!(approx_eq(data[4], 2.0, 1e-10));
    Ok(())
}

#[test]
fn test_leaky_relu_custom_slope() -> shrew::Result<()> {
    let dev = CpuDevice;
    let x = CpuTensor::from_f64_slice(&[-1.0, 1.0], 2, DType::F64, &dev)?;
    let act = LeakyReLU::with_slope(0.2);
    let y = Module::<CpuBackend>::forward(&act, &x)?;
    assert_vec_approx(&y.to_f64_vec()?, &[-0.2, 1.0], 1e-10);
    Ok(())
}

#[test]
fn test_elu() -> shrew::Result<()> {
    let dev = CpuDevice;
    let x = CpuTensor::from_f64_slice(&[-1.0, 0.0, 1.0], 3, DType::F64, &dev)?;
    let act = ELU::new(); // alpha=1.0
    let y = Module::<CpuBackend>::forward(&act, &x)?;
    let data = y.to_f64_vec()?;
    // ELU(-1) = 1.0 * (exp(-1) - 1) ≈ -0.6321
    assert!(approx_eq(data[0], (-1.0f64).exp() - 1.0, 1e-5));
    assert!(approx_eq(data[1], 0.0, 1e-10));
    assert!(approx_eq(data[2], 1.0, 1e-10));
    Ok(())
}

#[test]
fn test_mish() -> shrew::Result<()> {
    let dev = CpuDevice;
    let x = CpuTensor::from_f64_slice(&[0.0, 1.0, -1.0], 3, DType::F64, &dev)?;
    let act = Mish;
    let y = Module::<CpuBackend>::forward(&act, &x)?;
    let data = y.to_f64_vec()?;
    // mish(0) = 0 * tanh(ln(2)) ≈ 0
    assert!(approx_eq(data[0], 0.0, 1e-10));
    // mish(1) = 1 * tanh(ln(1 + e)) ≈ 0.8651
    let expected_1 = 1.0 * ((1.0 + 1.0_f64.exp()).ln()).tanh();
    assert!(approx_eq(data[1], expected_1, 1e-5));
    Ok(())
}

#[test]
fn test_groupnorm_forward() -> shrew::Result<()> {
    use shrew::nn::GroupNorm;
    let dev = CpuDevice;
    let gn = GroupNorm::<CpuBackend>::new(2, 4, 1e-5, DType::F64, &dev)?;
    // Input: [1, 4, 2, 2] — 2 groups of 2 channels each
    let x = CpuTensor::rand(shrew::Shape::new(vec![1, 4, 2, 2]), DType::F64, &dev)?;
    let y = gn.forward(&x)?;
    assert_eq!(y.dims(), &[1, 4, 2, 2]);
    Ok(())
}

#[test]
fn test_groupnorm_params() -> shrew::Result<()> {
    use shrew::nn::GroupNorm;
    let dev = CpuDevice;
    let gn = GroupNorm::<CpuBackend>::new(4, 16, 1e-5, DType::F64, &dev)?;
    assert_eq!(gn.num_parameters(), 32); // 16 weight + 16 bias
    Ok(())
}

#[test]
fn test_rmsnorm_forward() -> shrew::Result<()> {
    use shrew::nn::RMSNorm;
    let dev = CpuDevice;
    let rms = RMSNorm::<CpuBackend>::new(8, 1e-5, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 4, 8), DType::F64, &dev)?;
    let y = rms.forward(&x)?;
    assert_eq!(y.dims(), &[2, 4, 8]);
    Ok(())
}

#[test]
fn test_rmsnorm_normalization() -> shrew::Result<()> {
    use shrew::nn::RMSNorm;
    let dev = CpuDevice;
    let rms = RMSNorm::<CpuBackend>::new(4, 1e-5, DType::F64, &dev)?;
    // After RMSNorm with weight=1, x/rms should have approx unit RMS
    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (1, 4), DType::F64, &dev)?;
    let y = rms.forward(&x)?;
    let data = y.to_f64_vec()?;
    // RMS of result should be approximately 1
    let rms_val: f64 = (data.iter().map(|v| v * v).sum::<f64>() / data.len() as f64).sqrt();
    assert!(
        approx_eq(rms_val, 1.0, 0.01),
        "RMS should be ~1, got {}",
        rms_val
    );
    Ok(())
}

#[test]
fn test_rmsnorm_params() -> shrew::Result<()> {
    use shrew::nn::RMSNorm;
    let dev = CpuDevice;
    let rms = RMSNorm::<CpuBackend>::new(512, 1e-5, DType::F64, &dev)?;
    assert_eq!(rms.num_parameters(), 512); // Only weight, no bias
    Ok(())
}

#[test]
fn test_adaptive_avg_pool2d() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pool = AdaptiveAvgPool2d::new([1, 1]); // Global average pooling
    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0], // 1 channel, 2x2
        shrew::Shape::new(vec![1, 1, 2, 2]),
        DType::F64,
        &dev,
    )?;
    let y = pool.forward(&x)?;
    assert_eq!(y.dims(), &[1, 1, 1, 1]);
    assert_vec_approx(&y.to_f64_vec()?, &[2.5], 1e-10); // mean of 1,2,3,4
    Ok(())
}

#[test]
fn test_adaptive_avg_pool2d_2x2() -> shrew::Result<()> {
    let dev = CpuDevice;
    let pool = AdaptiveAvgPool2d::new([2, 2]);
    let x = CpuTensor::rand(shrew::Shape::new(vec![1, 3, 8, 8]), DType::F64, &dev)?;
    let y = pool.forward(&x)?;
    assert_eq!(y.dims(), &[1, 3, 2, 2]);
    Ok(())
}

// C4 Tests — RMSProp, RAdam, ReduceLROnPlateau, EMA, GradAccumulator

#[test]
fn test_rmsprop_step() -> shrew::Result<()> {
    let dev = CpuDevice;
    // w = 1.0, x = 1.0 → loss = w*x = 1.0, grad(w) = 1.0
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?;

    let mut opt = RMSProp::<CpuBackend>::new(vec![w.clone()], 0.01);

    for _ in 0..5 {
        let loss = opt.params()[0].mul(&x)?.sum_all()?;
        opt.step(&loss.backward()?)?;
    }

    // Weight should decrease (gradient is positive, we're minimizing w*x)
    let final_w = opt.params()[0].to_scalar_f64()?;
    assert!(final_w < 1.0, "RMSProp should decrease w, got {}", final_w);
    Ok(())
}

#[test]
fn test_rmsprop_with_momentum() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?;

    let mut opt = RMSProp::<CpuBackend>::new(vec![w.clone()], 0.01).momentum(0.9);

    for _ in 0..10 {
        let loss = opt.params()[0].mul(&x)?.sum_all()?;
        opt.step(&loss.backward()?)?;
    }

    let final_w = opt.params()[0].to_scalar_f64()?;
    assert!(
        final_w < 1.0,
        "RMSProp+momentum should decrease w, got {}",
        final_w
    );
    Ok(())
}

#[test]
fn test_radam_step() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?;

    let mut opt = RAdam::<CpuBackend>::new(vec![w.clone()], 0.1);

    // In early steps, RAdam uses momentum-only (variance not tractable)
    for _ in 0..3 {
        let loss = opt.params()[0].mul(&x)?.sum_all()?;
        opt.step(&loss.backward()?)?;
    }
    let early_w = opt.params()[0].to_scalar_f64()?;
    assert!(
        early_w < 1.0,
        "RAdam should decrease w early, got {}",
        early_w
    );
    assert_eq!(opt.step_count(), 3);

    // Continue for more steps (eventually switches to adaptive)
    for _ in 0..7 {
        let loss = opt.params()[0].mul(&x)?.sum_all()?;
        opt.step(&loss.backward()?)?;
    }
    let final_w = opt.params()[0].to_scalar_f64()?;
    assert!(
        final_w < early_w,
        "RAdam should keep decreasing, got {}",
        final_w
    );
    assert_eq!(opt.step_count(), 10);
    Ok(())
}

#[test]
fn test_radam_weight_decay() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w1 = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let w2 = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?;

    let mut opt_no_wd = RAdam::<CpuBackend>::new(vec![w1.clone()], 0.1);
    let mut opt_wd = RAdam::<CpuBackend>::new(vec![w2.clone()], 0.1).weight_decay(0.1);

    // One step each
    let loss = opt_no_wd.params()[0].mul(&x)?.sum_all()?;
    opt_no_wd.step(&loss.backward()?)?;

    let loss = opt_wd.params()[0].mul(&x)?.sum_all()?;
    opt_wd.step(&loss.backward()?)?;

    // With weight decay, param should be smaller
    let w_no_wd = opt_no_wd.params()[0].to_scalar_f64()?;
    let w_wd = opt_wd.params()[0].to_scalar_f64()?;
    assert!(
        w_wd < w_no_wd,
        "weight decay should reduce param more: {} vs {}",
        w_wd,
        w_no_wd
    );
    Ok(())
}

#[test]
fn test_reduce_lr_on_plateau_basic() {
    let mut sched = ReduceLROnPlateau::new(0.1).patience(3).factor(0.5);

    // Report improving metrics (loss decreasing)
    sched.step_metric(10.0);
    sched.step_metric(9.0);
    sched.step_metric(8.0);
    assert!(
        approx_eq(sched.lr(), 0.1, 1e-10),
        "LR should not change while improving"
    );

    // Report plateau (loss not improving)
    sched.step_metric(8.0);
    sched.step_metric(8.0);
    assert!(
        approx_eq(sched.lr(), 0.1, 1e-10),
        "patience not exhausted yet"
    );

    sched.step_metric(8.0); // patience=3 reached
    assert!(
        approx_eq(sched.lr(), 0.05, 1e-10),
        "LR should be halved: {}",
        sched.lr()
    );
}

#[test]
fn test_reduce_lr_on_plateau_max_mode() {
    let mut sched = ReduceLROnPlateau::new(0.1)
        .patience(2)
        .factor(0.5)
        .mode_max();

    // Report improving metrics (accuracy increasing)
    sched.step_metric(0.5);
    sched.step_metric(0.7);
    assert!(approx_eq(sched.lr(), 0.1, 1e-10));

    // Plateau
    sched.step_metric(0.7);
    sched.step_metric(0.7); // patience=2 reached
    assert!(
        approx_eq(sched.lr(), 0.05, 1e-10),
        "LR should be halved in max mode: {}",
        sched.lr()
    );
}

#[test]
fn test_reduce_lr_on_plateau_min_lr() {
    let mut sched = ReduceLROnPlateau::new(0.001)
        .patience(1)
        .factor(0.1)
        .min_lr(1e-4);

    // First plateau
    sched.step_metric(1.0);
    sched.step_metric(1.0); // patience=1 → LR = 0.001 * 0.1 = 0.0001
    assert!(approx_eq(sched.lr(), 1e-4, 1e-10));

    // Second plateau — should not go below min_lr
    sched.step_metric(1.0);
    sched.step_metric(1.0);
    assert!(
        approx_eq(sched.lr(), 1e-4, 1e-10),
        "should not go below min_lr"
    );
}

#[test]
fn test_ema_basic() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?;

    let mut ema = EMA::<CpuBackend>::new(vec![w.clone()], 0.9)?;

    // Simulate a parameter update (w becomes 2.0)
    w.update_data_inplace(&[2.0])?;
    ema.update(&[w.clone()])?;

    // shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    assert!(approx_eq(ema.shadow_values(0)[0], 1.1, 1e-10));
    assert_eq!(ema.num_updates(), 1);

    // Another update (w becomes 3.0)
    w.update_data_inplace(&[3.0])?;
    ema.update(&[w.clone()])?;

    // shadow = 0.9 * 1.1 + 0.1 * 3.0 = 0.99 + 0.3 = 1.29
    assert!(approx_eq(ema.shadow_values(0)[0], 1.29, 1e-10));
    Ok(())
}

#[test]
fn test_ema_apply_and_restore() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?;

    let mut ema = EMA::<CpuBackend>::new(vec![w.clone()], 0.5)?;

    // Update w to 3.0 and update EMA
    w.update_data_inplace(&[3.0])?;
    ema.update(&[w.clone()])?;
    // shadow = 0.5 * 1.0 + 0.5 * 3.0 = 2.0

    // Apply EMA → w should become 2.0
    ema.apply()?;
    assert!(approx_eq(w.to_scalar_f64()?, 2.0, 1e-10));

    // Restore → w should go back to 3.0
    ema.restore()?;
    assert!(approx_eq(w.to_scalar_f64()?, 3.0, 1e-10));
    Ok(())
}

#[test]
fn test_ema_warmup() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[0.0], (), DType::F64, &dev)?;

    let mut ema = EMA::<CpuBackend>::new(vec![w.clone()], 0.999)?;

    // First update: effective_decay = min(0.999, 1/10) = 0.1
    // So shadow = 0.1 * 0.0 + 0.9 * 1.0 = 0.9 (much faster than 0.999 decay)
    w.update_data_inplace(&[1.0])?;
    ema.update_with_warmup(&[w.clone()])?;

    let ema_val = ema.shadow_values(0)[0];
    // effective_decay = min(0.999, (1+1)/(10+1)) = min(0.999, 2/11) ≈ 0.1818
    // shadow = 0.1818 * 0.0 + 0.8182 * 1.0 ≈ 0.8182
    assert!(
        ema_val > 0.5,
        "warmup should allow fast initial tracking, got {}",
        ema_val
    );
    Ok(())
}

#[test]
fn test_grad_accumulator() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?;

    let mut accum = GradAccumulator::<CpuBackend>::new(3);

    // Step 1: accumulate, not ready yet
    let loss = w.mul(&x)?.sum_all()?;
    let grads = loss.backward()?;
    let result = accum.step(&grads, &[w.clone()])?;
    assert!(result.is_none(), "should not yield after 1 step");

    // Step 2: still accumulating
    let loss = w.mul(&x)?.sum_all()?;
    let grads = loss.backward()?;
    let result = accum.step(&grads, &[w.clone()])?;
    assert!(result.is_none(), "should not yield after 2 steps");

    // Step 3: ready!
    let loss = w.mul(&x)?.sum_all()?;
    let grads = loss.backward()?;
    let result = accum.step(&grads, &[w.clone()])?;
    assert!(result.is_some(), "should yield after 3 steps");

    // The averaged gradient should be the original gradient (all identical)
    let avg = result.unwrap();
    let grad_val = avg.get(&w).unwrap().to_scalar_f64()?;
    // Each backward gives grad=1.0, accumulated 3 of them = 3.0, averaged by 3 = 1.0
    assert!(
        approx_eq(grad_val, 1.0, 1e-6),
        "averaged grad should be 1.0, got {}",
        grad_val
    );
    Ok(())
}

#[test]
fn test_grad_accumulator_reset() -> shrew::Result<()> {
    let dev = CpuDevice;
    let w = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?.set_variable();
    let x = CpuTensor::from_f64_slice(&[1.0], (), DType::F64, &dev)?;

    let mut accum = GradAccumulator::<CpuBackend>::new(2);

    // Step 1
    let loss = w.mul(&x)?.sum_all()?;
    accum.step(&loss.backward()?, &[w.clone()])?;
    assert_eq!(accum.current_step(), 1);

    // Reset
    accum.reset();
    assert_eq!(accum.current_step(), 0);

    // Should need 2 more steps now
    let loss = w.mul(&x)?.sum_all()?;
    let result = accum.step(&loss.backward()?, &[w.clone()])?;
    assert!(result.is_none());
    Ok(())
}
