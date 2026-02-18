// Mixed Precision CUDA Tests — F16/BF16 dtype casting and mixed precision

#[cfg(test)]
mod tests {
    use shrew_core::dtype::DType;
    use shrew_cuda::{CudaBackend, CudaDevice, CudaTensor};

    type T = CudaTensor;

    fn gpu() -> CudaDevice {
        CudaDevice::new(0).expect("CUDA device 0 not available")
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ─────────────────────────────────────────────────────────────────────
    // On-device dtype casting (CUDA cast kernels)
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_cast_f32_to_f16() {
        let dev = gpu();
        let t = T::from_f64_slice(&[1.0, 2.5, -3.0, 0.0], 4, DType::F32, &dev).unwrap();
        let f16 = t.to_dtype(DType::F16).unwrap();
        assert_eq!(f16.dtype(), DType::F16);
        assert_eq!(f16.dims(), &[4]);
        let data = f16.to_f64_vec().unwrap();
        assert!(approx(data[0], 1.0, 1e-3));
        assert!(approx(data[1], 2.5, 1e-3));
        assert!(approx(data[2], -3.0, 1e-3));
        assert!(approx(data[3], 0.0, 1e-3));
    }

    #[test]
    fn test_cast_f16_to_f32() {
        let dev = gpu();
        let t = T::from_f64_slice(&[1.0, -0.5, 100.0], 3, DType::F16, &dev).unwrap();
        let f32 = t.to_dtype(DType::F32).unwrap();
        assert_eq!(f32.dtype(), DType::F32);
        let data = f32.to_f64_vec().unwrap();
        assert!(approx(data[0], 1.0, 1e-3));
        assert!(approx(data[1], -0.5, 1e-3));
        assert!(approx(data[2], 100.0, 1e-1));
    }

    #[test]
    fn test_cast_f32_to_bf16() {
        let dev = gpu();
        let t = T::from_f64_slice(&[1.0, 2.5, -3.0], 3, DType::F32, &dev).unwrap();
        let bf16 = t.to_dtype(DType::BF16).unwrap();
        assert_eq!(bf16.dtype(), DType::BF16);
        let data = bf16.to_f64_vec().unwrap();
        assert!(approx(data[0], 1.0, 1e-2));
        assert!(approx(data[1], 2.5, 1e-2));
        assert!(approx(data[2], -3.0, 1e-2));
    }

    #[test]
    fn test_cast_bf16_to_f32() {
        let dev = gpu();
        let t = T::from_f64_slice(&[1.0, -0.5, 42.0], 3, DType::BF16, &dev).unwrap();
        let f32 = t.to_dtype(DType::F32).unwrap();
        assert_eq!(f32.dtype(), DType::F32);
        let data = f32.to_f64_vec().unwrap();
        assert!(approx(data[0], 1.0, 1e-2));
        assert!(approx(data[1], -0.5, 1e-2));
        assert!(approx(data[2], 42.0, 1e-1));
    }

    #[test]
    fn test_cast_f16_to_f64() {
        let dev = gpu();
        let t = T::from_f64_slice(&[1.0, -2.0], 2, DType::F16, &dev).unwrap();
        let f64 = t.to_dtype(DType::F64).unwrap();
        assert_eq!(f64.dtype(), DType::F64);
        let data = f64.to_f64_vec().unwrap();
        assert!(approx(data[0], 1.0, 1e-3));
        assert!(approx(data[1], -2.0, 1e-3));
    }

    #[test]
    fn test_cast_f16_to_bf16() {
        let dev = gpu();
        let t = T::from_f64_slice(&[1.0, -3.0, 0.5], 3, DType::F16, &dev).unwrap();
        let bf = t.to_dtype(DType::BF16).unwrap();
        assert_eq!(bf.dtype(), DType::BF16);
        let data = bf.to_f64_vec().unwrap();
        assert!(approx(data[0], 1.0, 1e-2));
        assert!(approx(data[1], -3.0, 1e-2));
        assert!(approx(data[2], 0.5, 1e-2));
    }

    #[test]
    fn test_cast_roundtrip_f32_f16_f32() {
        let dev = gpu();
        let vals = [1.0, 2.5, -3.0, 0.0, 100.0, -0.125];
        let t = T::from_f64_slice(&vals, 6, DType::F32, &dev).unwrap();
        let f16 = t.to_dtype(DType::F16).unwrap();
        let back = f16.to_dtype(DType::F32).unwrap();
        assert_eq!(back.dtype(), DType::F32);
        let data = back.to_f64_vec().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(approx(data[i], v, 0.1), "index {}: {} != {}", i, data[i], v);
        }
    }

    #[test]
    fn test_cast_noop_same_dtype() {
        let dev = gpu();
        let t = T::from_f64_slice(&[1.0, 2.0], 2, DType::F32, &dev).unwrap();
        let same = t.to_dtype(DType::F32).unwrap();
        assert_eq!(same.dtype(), DType::F32);
        let data = same.to_f64_vec().unwrap();
        assert!(approx(data[0], 1.0, 1e-6));
    }

    // ─────────────────────────────────────────────────────────────────────
    // F16 computation on GPU (ops work with half-precision tensors)
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_f16_binary_ops() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F16, &dev).unwrap();
        let b = T::from_f64_slice(&[4.0, 5.0, 6.0], 3, DType::F16, &dev).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.dtype(), DType::F16);
        let data = c.to_f64_vec().unwrap();
        assert!(approx(data[0], 5.0, 1e-2));
        assert!(approx(data[1], 7.0, 1e-2));
        assert!(approx(data[2], 9.0, 1e-2));
    }

    #[test]
    fn test_f16_matmul() {
        let dev = gpu();
        // 2x2 identity @ 2x2 values
        let a = T::from_f64_slice(&[1.0, 0.0, 0.0, 1.0], (2, 2), DType::F16, &dev).unwrap();
        let b = T::from_f64_slice(&[5.0, 6.0, 7.0, 8.0], (2, 2), DType::F16, &dev).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dtype(), DType::F16);
        let data = c.to_f64_vec().unwrap();
        assert!(approx(data[0], 5.0, 1e-1));
        assert!(approx(data[1], 6.0, 1e-1));
        assert!(approx(data[2], 7.0, 1e-1));
        assert!(approx(data[3], 8.0, 1e-1));
    }

    #[test]
    fn test_f16_unary_ops() {
        let dev = gpu();
        let t = T::from_f64_slice(&[0.0, 1.0, -1.0, 2.0], 4, DType::F16, &dev).unwrap();
        let r = t.relu().unwrap();
        assert_eq!(r.dtype(), DType::F16);
        let data = r.to_f64_vec().unwrap();
        assert!(approx(data[0], 0.0, 1e-3));
        assert!(approx(data[1], 1.0, 1e-3));
        assert!(approx(data[2], 0.0, 1e-3));
        assert!(approx(data[3], 2.0, 1e-3));
    }

    #[test]
    fn test_f16_reduction() {
        let dev = gpu();
        let t = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], 4, DType::F16, &dev).unwrap();
        let s = t.sum_all().unwrap();
        let v = s.to_scalar_f64().unwrap();
        assert!(approx(v, 10.0, 1e-1));
    }

    #[test]
    fn test_bf16_binary_ops() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::BF16, &dev).unwrap();
        let b = T::from_f64_slice(&[10.0, 20.0, 30.0], 3, DType::BF16, &dev).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.dtype(), DType::BF16);
        let data = c.to_f64_vec().unwrap();
        assert!(approx(data[0], 10.0, 1.0));
        assert!(approx(data[1], 40.0, 1.0));
        assert!(approx(data[2], 90.0, 1.0));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Mixed precision forward+backward (F16 activations, F32 weights)
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_f16_forward_backward() {
        // Simulate mixed precision: F16 values through a computation graph
        let dev = gpu();
        let w = T::from_f64_slice(&[0.5, -0.3, 0.8, 0.1], (2, 2), DType::F16, &dev)
            .unwrap()
            .set_variable();
        let x = T::from_f64_slice(&[1.0, 2.0], (1, 2), DType::F16, &dev).unwrap();
        let y = x.matmul(&w.transpose(0, 1).unwrap()).unwrap();
        let loss = y.sum_all().unwrap();
        let v = loss.to_scalar_f64().unwrap();
        // y = [1*0.5 + 2*(-0.3), 1*0.8 + 2*0.1] = [-0.1, 1.0], sum = 0.9
        assert!(approx(v, 0.9, 0.2)); // F16 tolerance

        // Backward should work and produce gradients
        let grads = loss.backward().unwrap();
        let gw = grads.get(&w).expect("weight gradient missing");
        assert_eq!(gw.dims(), &[2, 2]);
        assert_eq!(gw.dtype(), DType::F16);
    }

    #[test]
    fn test_cast_in_autograd_graph() {
        // Verify that to_dtype preserves gradient flow
        let dev = gpu();
        let w = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev)
            .unwrap()
            .set_variable();

        // Cast to F16 (should record Op::ToDtype)
        let w_f16 = w.to_dtype(DType::F16).unwrap();
        assert_eq!(w_f16.dtype(), DType::F16);

        // Do some F16 computation
        let x = T::from_f64_slice(&[1.0, 1.0], (1, 2), DType::F16, &dev).unwrap();
        let y = x.matmul(&w_f16.transpose(0, 1).unwrap()).unwrap();
        let loss = y.sum_all().unwrap();

        // Backward should flow through the dtype cast
        let grads = loss.backward().unwrap();
        let gw = grads
            .get(&w)
            .expect("gradient should flow through to_dtype");
        assert_eq!(gw.dtype(), DType::F32); // gradient is in original dtype
        assert_eq!(gw.dims(), &[2, 2]);
    }

    #[test]
    fn test_large_f16_cast_performance() {
        // Ensure large tensors can be cast efficiently on GPU
        let dev = gpu();
        let n = 1_000_000;
        let t = T::ones(n, DType::F32, &dev).unwrap();
        let f16 = t.to_dtype(DType::F16).unwrap();
        assert_eq!(f16.dtype(), DType::F16);
        assert_eq!(f16.dims(), &[n]);
        let back = f16.to_dtype(DType::F32).unwrap();
        assert_eq!(back.dtype(), DType::F32);
        // Spot check: first element should still be 1.0
        let data = back.to_f64_vec().unwrap();
        assert!(approx(data[0], 1.0, 1e-3));
        assert!(approx(data[n - 1], 1.0, 1e-3));
    }

    #[test]
    fn test_mixed_precision_linear_simulation() {
        // Simulate what MixedPrecisionTrainer does:
        // FP32 master weights → cast to F16 → forward → backward → cast grads to FP32
        let dev = gpu();

        // FP32 master weights
        let master_w = T::from_f64_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (2, 3), DType::F32, &dev)
            .unwrap()
            .set_variable();

        // Cast to F16 for compute
        let w_f16 = master_w.to_dtype(DType::F16).unwrap();

        // F16 input
        let x = T::from_f64_slice(&[1.0, 2.0, 3.0], (1, 3), DType::F16, &dev).unwrap();

        // Forward in F16
        let y = x.matmul(&w_f16.transpose(0, 1).unwrap()).unwrap();
        assert_eq!(y.dtype(), DType::F16);

        // Loss
        let target = T::from_f64_slice(&[1.0, 1.0], (1, 2), DType::F16, &dev).unwrap();
        let diff = y.sub(&target).unwrap();
        let loss = diff.mul(&diff).unwrap().sum_all().unwrap();

        // Backward — gradients flow through to_dtype back to master_w
        let grads = loss.backward().unwrap();
        let gw = grads
            .get(&master_w)
            .expect("master weight gradient should exist");
        assert_eq!(gw.dtype(), DType::F32); // gradient auto-cast to FP32
        assert_eq!(gw.dims(), &[2, 3]);

        // Simulate weight update: w_new = w - lr * grad
        let lr = 0.01;
        let w_updated = master_w.affine(1.0, 0.0).unwrap(); // clone-ish
        let step = gw.affine(-lr, 0.0).unwrap();
        let _w_new = w_updated.add(&step).unwrap();
    }

    #[test]
    fn test_mixed_precision_trainer_gpu() {
        use shrew_nn::Module;

        type B = CudaBackend;
        let dev = gpu();

        // Create a simple linear model with F16 weights on GPU
        let linear = shrew_nn::Linear::<B>::new(4, 2, true, DType::F16, &dev).unwrap();
        let optimizer = shrew_optim::SGD::new(linear.parameters(), 0.01, 0.0, 0.0);

        let config = shrew::distributed::LossScaleConfig {
            init_scale: 1.0, // Small scale for F16 to avoid overflow
            ..Default::default()
        };
        let mut trainer =
            shrew::distributed::MixedPrecisionTrainer::new(linear, optimizer, DType::F16, config);

        // F16 input and target
        let input = T::randn((2, 4), DType::F16, &dev).unwrap();
        let target = T::zeros((2, 2), DType::F16, &dev).unwrap();

        let metrics = trainer
            .train_step(&input, &target, |pred, tgt| shrew_nn::mse_loss(pred, tgt))
            .unwrap();

        assert!(!metrics.skipped);
        assert!(metrics.loss >= 0.0);
        assert_eq!(metrics.compute_dtype, DType::F16);
    }
}
