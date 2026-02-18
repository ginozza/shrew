// GPU NN Layer Tests — Verify neural network layers work on CudaBackend

#[cfg(test)]
mod tests {
    use shrew_core::dtype::DType;
    use shrew_cuda::{CudaBackend, CudaDevice, CudaTensor};
    use shrew_nn::module::Module;

    type T = CudaTensor;
    type B = CudaBackend;

    fn gpu() -> CudaDevice {
        CudaDevice::new(0).expect("CUDA device 0 not available")
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ─────────────────────────────────────────────────────────────────────
    // Linear
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_linear_forward() {
        let dev = gpu();
        let linear = shrew_nn::linear::Linear::<B>::new(4, 3, true, DType::F32, &dev).unwrap();
        let x = T::randn((2, 4), DType::F32, &dev).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_linear_no_bias() {
        let dev = gpu();
        let linear = shrew_nn::linear::Linear::<B>::new(8, 4, false, DType::F32, &dev).unwrap();
        let x = T::randn((5, 8), DType::F32, &dev).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[5, 4]);
    }

    #[test]
    fn test_linear_parameters() {
        let dev = gpu();
        let linear = shrew_nn::linear::Linear::<B>::new(10, 5, true, DType::F32, &dev).unwrap();
        let params = linear.parameters();
        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].elem_count(), 50); // 10*5
        assert_eq!(params[1].elem_count(), 5); // bias
    }

    // ─────────────────────────────────────────────────────────────────────
    // Activations (GPU — all operate element-wise)
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_relu_layer() {
        let dev = gpu();
        let relu = shrew_nn::activation::ReLU;
        let x = T::from_f64_slice(&[-1.0, 0.0, 1.0, 2.0], (4,), DType::F32, &dev).unwrap();
        let y = relu.forward(&x).unwrap();
        let data = y.to_f64_vec().unwrap();
        assert!(approx(data[0], 0.0, 1e-5));
        assert!(approx(data[2], 1.0, 1e-5));
        assert!(approx(data[3], 2.0, 1e-5));
    }

    #[test]
    fn test_gelu_layer() {
        let dev = gpu();
        let gelu = shrew_nn::activation::GeLU;
        let x = T::from_f64_slice(&[0.0, 1.0], (2,), DType::F32, &dev).unwrap();
        let y = gelu.forward(&x).unwrap();
        let data = y.to_f64_vec().unwrap();
        assert!(approx(data[0], 0.0, 1e-4));
        assert!(approx(data[1], 0.841, 5e-2));
    }

    #[test]
    fn test_sigmoid_layer() {
        let dev = gpu();
        let sig = shrew_nn::activation::Sigmoid;
        let x = T::from_f64_slice(&[0.0], (1,), DType::F32, &dev).unwrap();
        let y = sig.forward(&x).unwrap();
        assert!(approx(y.to_scalar_f64().unwrap(), 0.5, 1e-5));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Dropout
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dropout_training() {
        let dev = gpu();
        let dropout = shrew_nn::dropout::Dropout::new(0.5);
        let x = T::ones((100,), DType::F32, &dev).unwrap();
        let y = dropout.forward(&x).unwrap();
        let data = y.to_f64_vec().unwrap();
        // In training mode with p=0.5, roughly half should be 0 and half ~2.0
        let nonzero: Vec<_> = data.iter().filter(|v| **v > 0.1).collect();
        assert!(
            nonzero.len() > 20 && nonzero.len() < 80,
            "Expected ~50% nonzero, got {}",
            nonzero.len()
        );
    }

    #[test]
    fn test_dropout_eval() {
        let dev = gpu();
        let dropout = shrew_nn::dropout::Dropout::new(0.5);
        dropout.set_training(false);
        let x = T::ones((10,), DType::F32, &dev).unwrap();
        let y = dropout.forward(&x).unwrap();
        let data = y.to_f64_vec().unwrap();
        // In eval mode, dropout is identity
        for &v in &data {
            assert!(approx(v, 1.0, 1e-5));
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // LayerNorm
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_layernorm_forward() {
        let dev = gpu();
        let ln = shrew_nn::layernorm::LayerNorm::<B>::new(4, 1e-5, DType::F32, &dev).unwrap();
        let x = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (1, 4), DType::F32, &dev).unwrap();
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[1, 4]);
        // Normalized output should have mean ≈ 0
        let data = y.to_f64_vec().unwrap();
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        assert!(
            mean.abs() < 0.2,
            "LayerNorm output mean should be near 0, got {mean}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Embedding
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_embedding_forward() {
        let dev = gpu();
        let emb = shrew_nn::embedding::Embedding::<B>::new(100, 16, DType::F32, &dev).unwrap();
        // Token indices as I64
        let indices = T::from_f64_slice(&[0.0, 5.0, 99.0], (3,), DType::I64, &dev).unwrap();
        let y = emb.forward(&indices).unwrap();
        assert_eq!(y.shape().dims(), &[3, 16]);
    }

    #[test]
    fn test_embedding_2d() {
        let dev = gpu();
        let emb = shrew_nn::embedding::Embedding::<B>::new(50, 8, DType::F32, &dev).unwrap();
        // [batch=2, seq=3]
        let indices =
            T::from_f64_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], (2, 3), DType::I64, &dev).unwrap();
        let y = emb.forward(&indices).unwrap();
        assert_eq!(y.shape().dims(), &[2, 3, 8]);
    }

    // ─────────────────────────────────────────────────────────────────────
    // BatchNorm2d
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_batchnorm_forward() {
        let dev = gpu();
        let bn =
            shrew_nn::batchnorm::BatchNorm2d::<B>::new(4, 1e-5, 0.1, DType::F32, &dev).unwrap();
        // [N=2, C=4, H=3, W=3]
        let x = T::randn((2, 4, 3, 3), DType::F32, &dev).unwrap();
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[2, 4, 3, 3]);
    }

    #[test]
    fn test_batchnorm_eval_mode() {
        let dev = gpu();
        let bn =
            shrew_nn::batchnorm::BatchNorm2d::<B>::new(2, 1e-5, 0.1, DType::F32, &dev).unwrap();
        // Run one forward in training to update running stats
        let x = T::randn((4, 2, 4, 4), DType::F32, &dev).unwrap();
        let _ = bn.forward(&x).unwrap();
        // Switch to eval
        bn.eval();
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[4, 2, 4, 4]);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Sequential
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_sequential_mlp() {
        let dev = gpu();
        // Build a 2-layer MLP: Linear(4,8) → ReLU → Linear(8,2)
        let l1 = shrew_nn::linear::Linear::<B>::new(4, 8, true, DType::F32, &dev).unwrap();
        let l2 = shrew_nn::linear::Linear::<B>::new(8, 2, true, DType::F32, &dev).unwrap();

        let seq = shrew_nn::sequential::Sequential::<B>::new()
            .add(l1)
            .add(shrew_nn::activation::ReLU)
            .add(l2);

        let x = T::randn((3, 4), DType::F32, &dev).unwrap();
        let y = seq.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[3, 2]);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Loss functions on GPU
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mse_loss_gpu() {
        let dev = gpu();
        let pred = T::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let target = T::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let loss = shrew_nn::loss::mse_loss(&pred, &target).unwrap();
        let loss_val = loss.to_scalar_f64().unwrap();
        assert!(
            approx(loss_val, 0.0, 1e-5),
            "MSE of identical tensors should be 0, got {loss_val}"
        );
    }

    #[test]
    fn test_cross_entropy_gpu() {
        let dev = gpu();
        // logits: [batch=2, classes=3]
        let logits =
            T::from_f64_slice(&[2.0, 1.0, 0.1, 0.1, 1.0, 2.0], (2, 3), DType::F32, &dev).unwrap();
        // targets: one-hot [2, 3]
        let targets =
            T::from_f64_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], (2, 3), DType::F32, &dev).unwrap();
        let loss = shrew_nn::loss::cross_entropy_loss(&logits, &targets).unwrap();
        let loss_val = loss.to_scalar_f64().unwrap();
        // Both predictions are correct → loss should be relatively small
        assert!(loss_val < 2.0, "Cross entropy loss too high: {loss_val}");
        assert!(loss_val > 0.0, "Cross entropy shouldn't be 0");
    }

    // ─────────────────────────────────────────────────────────────────────
    // End-to-end: GPU forward+backward on MLP
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_gpu_mlp_forward_backward() {
        let dev = gpu();
        // 2-layer MLP: 4 → 8 → 2
        let l1 = shrew_nn::linear::Linear::<B>::new(4, 8, true, DType::F32, &dev).unwrap();
        let l2 = shrew_nn::linear::Linear::<B>::new(8, 2, true, DType::F32, &dev).unwrap();

        let x = T::randn((3, 4), DType::F32, &dev).unwrap();
        let h = l1.forward(&x).unwrap().relu().unwrap();
        let y = l2.forward(&h).unwrap();

        // Compute MSE loss against zeros
        let target = T::zeros((3, 2), DType::F32, &dev).unwrap();
        let loss = shrew_nn::loss::mse_loss(&y, &target).unwrap();
        let loss_val = loss.to_scalar_f64().unwrap();
        assert!(loss_val >= 0.0);

        // Backward pass
        let grads = loss.backward().unwrap();

        // Verify gradients exist for all parameters
        let params = l1.parameters();
        for p in &params {
            let g = grads.get(p);
            assert!(g.is_some(), "Missing gradient for Linear l1 parameter");
        }
        let params2 = l2.parameters();
        for p in &params2 {
            let g = grads.get(p);
            assert!(g.is_some(), "Missing gradient for Linear l2 parameter");
        }
    }
}
