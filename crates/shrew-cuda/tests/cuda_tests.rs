// CUDA Backend Tests — Comprehensive tests for GPU tensor operations
//
// Run with: `cargo test -p shrew-cuda --features cuda-tests`
//          (or just `cargo test -p shrew-cuda` if CUDA is available)
//
// All tests use `#[cfg(test)]` and create a CudaDevice(0).

#[cfg(test)]
mod tests {
    use shrew_core::dtype::DType;
    use shrew_cuda::{CudaDevice, CudaTensor};

    type T = CudaTensor;

    fn gpu() -> CudaDevice {
        CudaDevice::new(0).expect("CUDA device 0 not available — skip CUDA tests")
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn assert_approx_vec(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(approx(*a, *e, tol), "index {i}: {a} != {e} (tol={tol})");
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tensor creation
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_zeros() {
        let dev = gpu();
        let t = T::zeros((2, 3), DType::F32, &dev).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        let data = t.to_f64_vec().unwrap();
        assert_eq!(data, vec![0.0; 6]);
    }

    #[test]
    fn test_ones() {
        let dev = gpu();
        let t = T::ones((2, 3), DType::F32, &dev).unwrap();
        let data = t.to_f64_vec().unwrap();
        assert_eq!(data, vec![1.0; 6]);
    }

    #[test]
    fn test_full() {
        let dev = gpu();
        let t = T::full((3, 2), 42.0, DType::F32, &dev).unwrap();
        let data = t.to_f64_vec().unwrap();
        assert_eq!(data, vec![42.0; 6]);
    }

    #[test]
    fn test_from_f64_slice() {
        let dev = gpu();
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = T::from_f64_slice(&vals, (2, 3), DType::F32, &dev).unwrap();
        let data = t.to_f64_vec().unwrap();
        assert_approx_vec(&data, &vals, 1e-5);
    }

    #[test]
    fn test_randn() {
        let dev = gpu();
        let t = T::randn((1000,), DType::F32, &dev).unwrap();
        assert_eq!(t.elem_count(), 1000);
        let data = t.to_f64_vec().unwrap();
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        // Mean of randn should be near 0
        assert!(mean.abs() < 0.2, "randn mean too far from 0: {mean}");
    }

    #[test]
    fn test_rand_uniform() {
        let dev = gpu();
        let t = T::rand((1000,), DType::F32, &dev).unwrap();
        let data = t.to_f64_vec().unwrap();
        for &v in &data {
            assert!(v >= 0.0 && v <= 1.0, "uniform sample out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_zeros_f64() {
        let dev = gpu();
        let t = T::zeros((4,), DType::F64, &dev).unwrap();
        let data = t.to_f64_vec().unwrap();
        assert_eq!(data, vec![0.0; 4]);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Binary operations
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_add() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[4.0, 5.0, 6.0], (3,), DType::F32, &dev).unwrap();
        let c = a.add(&b).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[5.0, 7.0, 9.0], 1e-5);
    }

    #[test]
    fn test_sub() {
        let dev = gpu();
        let a = T::from_f64_slice(&[10.0, 20.0, 30.0], (3,), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let c = a.sub(&b).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[9.0, 18.0, 27.0], 1e-5);
    }

    #[test]
    fn test_mul() {
        let dev = gpu();
        let a = T::from_f64_slice(&[2.0, 3.0, 4.0], (3,), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[5.0, 6.0, 7.0], (3,), DType::F32, &dev).unwrap();
        let c = a.mul(&b).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[10.0, 18.0, 28.0], 1e-5);
    }

    #[test]
    fn test_div() {
        let dev = gpu();
        let a = T::from_f64_slice(&[10.0, 20.0, 30.0], (3,), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[2.0, 5.0, 10.0], (3,), DType::F32, &dev).unwrap();
        let c = a.div(&b).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[5.0, 4.0, 3.0], 1e-5);
    }

    #[test]
    fn test_binary_2d() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], (2, 2), DType::F32, &dev).unwrap();
        let c = a.add(&b).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[11.0, 22.0, 33.0, 44.0], 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Unary operations
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_neg() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, -2.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let c = a.neg().unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[-1.0, 2.0, -3.0], 1e-5);
    }

    #[test]
    fn test_exp() {
        let dev = gpu();
        let a = T::from_f64_slice(&[0.0, 1.0, 2.0], (3,), DType::F32, &dev).unwrap();
        let c = a.exp().unwrap();
        let data = c.to_f64_vec().unwrap();
        assert!(approx(data[0], 1.0, 1e-4));
        assert!(approx(data[1], std::f64::consts::E, 1e-4));
        assert!(approx(
            data[2],
            std::f64::consts::E * std::f64::consts::E,
            1e-3
        ));
    }

    #[test]
    fn test_log() {
        let dev = gpu();
        let a =
            T::from_f64_slice(&[1.0, std::f64::consts::E, 10.0], (3,), DType::F32, &dev).unwrap();
        let c = a.log().unwrap();
        let data = c.to_f64_vec().unwrap();
        assert!(approx(data[0], 0.0, 1e-5));
        assert!(approx(data[1], 1.0, 1e-4));
    }

    #[test]
    fn test_sqrt() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 4.0, 9.0, 16.0], (4,), DType::F32, &dev).unwrap();
        let c = a.sqrt().unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn test_relu() {
        let dev = gpu();
        let a = T::from_f64_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], (5,), DType::F32, &dev).unwrap();
        let c = a.relu().unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[0.0, 0.0, 0.0, 1.0, 2.0], 1e-5);
    }

    #[test]
    fn test_sigmoid() {
        let dev = gpu();
        let a = T::from_f64_slice(&[0.0], (1,), DType::F32, &dev).unwrap();
        let c = a.sigmoid().unwrap();
        assert!(approx(c.to_scalar_f64().unwrap(), 0.5, 1e-5));
    }

    #[test]
    fn test_tanh() {
        let dev = gpu();
        let a = T::from_f64_slice(&[0.0], (1,), DType::F32, &dev).unwrap();
        let c = a.tanh().unwrap();
        assert!(approx(c.to_scalar_f64().unwrap(), 0.0, 1e-5));
    }

    #[test]
    fn test_abs() {
        let dev = gpu();
        let a = T::from_f64_slice(&[-3.0, -1.0, 0.0, 2.0], (4,), DType::F32, &dev).unwrap();
        let c = a.abs().unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[3.0, 1.0, 0.0, 2.0], 1e-5);
    }

    #[test]
    fn test_gelu() {
        let dev = gpu();
        let a = T::from_f64_slice(&[0.0, 1.0, -1.0], (3,), DType::F32, &dev).unwrap();
        let c = a.gelu().unwrap();
        let data = c.to_f64_vec().unwrap();
        // GeLU(0) = 0, GeLU(1) ≈ 0.8413, GeLU(-1) ≈ -0.1587
        assert!(approx(data[0], 0.0, 1e-4));
        assert!(approx(data[1], 0.8413, 5e-3));
        assert!(approx(data[2], -0.1587, 5e-3));
    }

    #[test]
    fn test_silu() {
        let dev = gpu();
        let a = T::from_f64_slice(&[0.0, 1.0, -1.0], (3,), DType::F32, &dev).unwrap();
        let c = a.silu().unwrap();
        let data = c.to_f64_vec().unwrap();
        // SiLU(0) = 0, SiLU(1) = 1*sigmoid(1) ≈ 0.7311
        assert!(approx(data[0], 0.0, 1e-5));
        assert!(approx(data[1], 0.7311, 5e-3));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Matrix multiplication (cuBLAS)
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_matmul_2x2() {
        let dev = gpu();
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[5.0, 6.0, 7.0, 8.0], (2, 2), DType::F32, &dev).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-3);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let dev = gpu();
        // [2,3] @ [3,2]
        let a =
            T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2), DType::F32, &dev)
            .unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);
        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // [[58, 64], [139, 154]]
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[58.0, 64.0, 139.0, 154.0], 1e-3);
    }

    #[test]
    fn test_matmul_f64() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev).unwrap();
        let b = T::from_f64_slice(&[5.0, 6.0, 7.0, 8.0], (2, 2), DType::F64, &dev).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-10);
    }

    #[test]
    fn test_matmul_identity() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev).unwrap();
        let eye = T::from_f64_slice(&[1.0, 0.0, 0.0, 1.0], (2, 2), DType::F32, &dev).unwrap();
        let c = a.matmul(&eye).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn test_matmul_batched() {
        let dev = gpu();
        // [2, 2, 2] batched matmul — 2 batches of 2x2 matrices
        let a = T::from_f64_slice(
            &[
                1.0, 0.0, 0.0, 1.0, // batch 0: identity
                2.0, 0.0, 0.0, 2.0,
            ], // batch 1: 2*identity
            (2, 2, 2),
            DType::F32,
            &dev,
        )
        .unwrap();
        let b = T::from_f64_slice(
            &[
                5.0, 6.0, 7.0, 8.0, // batch 0
                5.0, 6.0, 7.0, 8.0,
            ], // batch 1
            (2, 2, 2),
            DType::F32,
            &dev,
        )
        .unwrap();
        let c = a.matmul(&b).unwrap();
        let data = c.to_f64_vec().unwrap();
        // batch 0: identity @ [[5,6],[7,8]] = [[5,6],[7,8]]
        assert_approx_vec(&data[0..4], &[5.0, 6.0, 7.0, 8.0], 1e-3);
        // batch 1: 2*identity @ [[5,6],[7,8]] = [[10,12],[14,16]]
        assert_approx_vec(&data[4..8], &[10.0, 12.0, 14.0, 16.0], 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Reductions
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_sum_all() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (4,), DType::F32, &dev).unwrap();
        let s = a.sum_all().unwrap().to_scalar_f64().unwrap();
        assert!(approx(s, 10.0, 1e-5));
    }

    #[test]
    fn test_mean_all() {
        let dev = gpu();
        let a = T::from_f64_slice(&[2.0, 4.0, 6.0, 8.0], (4,), DType::F32, &dev).unwrap();
        let m = a.mean_all().unwrap().to_scalar_f64().unwrap();
        assert!(approx(m, 5.0, 1e-5));
    }

    #[test]
    fn test_sum_dim() {
        let dev = gpu();
        // [[1,2,3],[4,5,6]] → sum(dim=1) → [6, 15]
        let a =
            T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev).unwrap();
        let s = a.sum(1, false).unwrap();
        assert_eq!(s.shape().dims(), &[2]);
        assert_approx_vec(&s.to_f64_vec().unwrap(), &[6.0, 15.0], 1e-5);
    }

    #[test]
    fn test_sum_dim_keepdim() {
        let dev = gpu();
        let a =
            T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev).unwrap();
        let s = a.sum(1, true).unwrap();
        assert_eq!(s.shape().dims(), &[2, 1]);
    }

    #[test]
    fn test_max_dim() {
        let dev = gpu();
        let a =
            T::from_f64_slice(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], (2, 3), DType::F32, &dev).unwrap();
        let m = a.max(1, false).unwrap();
        assert_approx_vec(&m.to_f64_vec().unwrap(), &[5.0, 6.0], 1e-5);
    }

    #[test]
    fn test_min_dim() {
        let dev = gpu();
        let a =
            T::from_f64_slice(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], (2, 3), DType::F32, &dev).unwrap();
        let m = a.min(1, false).unwrap();
        assert_approx_vec(&m.to_f64_vec().unwrap(), &[1.0, 2.0], 1e-5);
    }

    #[test]
    fn test_argmax() {
        let dev = gpu();
        let a =
            T::from_f64_slice(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], (2, 3), DType::F32, &dev).unwrap();
        let idx = a.argmax(1, false).unwrap();
        assert_approx_vec(&idx.to_f64_vec().unwrap(), &[1.0, 2.0], 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Affine, powf, clamp
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_affine() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F32, &dev).unwrap();
        // affine: x * mul + add → x * 2 + 10
        let c = a.affine(2.0, 10.0).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[12.0, 14.0, 16.0], 1e-5);
    }

    #[test]
    fn test_powf() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (4,), DType::F32, &dev).unwrap();
        let c = a.powf(2.0).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[1.0, 4.0, 9.0, 16.0], 1e-4);
    }

    #[test]
    fn test_clamp() {
        let dev = gpu();
        let a = T::from_f64_slice(&[-5.0, 0.0, 3.0, 10.0], (4,), DType::F32, &dev).unwrap();
        let c = a.clamp(0.0, 5.0).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[0.0, 0.0, 3.0, 5.0], 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Comparison ops
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_cmp_lt() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 5.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[3.0, 3.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let c = a.lt(&b).unwrap();
        let data = c.to_f64_vec().unwrap();
        // 1<3=1, 5<3=0, 3<3=0
        assert_approx_vec(&data, &[1.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    fn test_cmp_eq() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 3.0, 5.0], (3,), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[3.0, 3.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let c = a.eq(&b).unwrap();
        let data = c.to_f64_vec().unwrap();
        assert_approx_vec(&data, &[0.0, 1.0, 0.0], 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Cat
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_cat_dim0() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0], (1, 3), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[4.0, 5.0, 6.0], (1, 3), DType::F32, &dev).unwrap();
        let c = T::cat(&[a, b], 0).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        assert_approx_vec(
            &c.to_f64_vec().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            1e-5,
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Index select
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_index_select() {
        let dev = gpu();
        let a = T::from_f64_slice(&[10.0, 20.0, 30.0, 40.0, 50.0], (5,), DType::F32, &dev).unwrap();
        let idx = T::from_f64_slice(&[0.0, 2.0, 4.0], (3,), DType::U32, &dev).unwrap();
        let c = a.index_select(0, &idx).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[10.0, 30.0, 50.0], 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Gather
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_gather() {
        let dev = gpu();
        // [[1,2],[3,4]] gather dim=1, index=[[0,0],[1,0]]
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev).unwrap();
        let idx = T::from_f64_slice(&[0.0, 0.0, 1.0, 0.0], (2, 2), DType::U32, &dev).unwrap();
        let c = a.gather(1, &idx).unwrap();
        // row0: [a[0,0], a[0,0]] = [1,1]; row1: [a[1,1], a[1,0]] = [4,3]
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[1.0, 1.0, 4.0, 3.0], 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Where / conditional
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_where_cond() {
        let dev = gpu();
        let cond = T::from_f64_slice(&[1.0, 0.0, 1.0, 0.0], (4,), DType::U8, &dev).unwrap();
        let on_true = T::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], (4,), DType::F32, &dev).unwrap();
        let on_false =
            T::from_f64_slice(&[100.0, 200.0, 300.0, 400.0], (4,), DType::F32, &dev).unwrap();
        let c = T::where_cond(&cond, &on_true, &on_false).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[10.0, 200.0, 30.0, 400.0], 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Shape operations
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_reshape() {
        let dev = gpu();
        let a =
            T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev).unwrap();
        let b = a.reshape((3, 2)).unwrap();
        assert_eq!(b.shape().dims(), &[3, 2]);
        assert_approx_vec(
            &b.to_f64_vec().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            1e-5,
        );
    }

    #[test]
    fn test_transpose() {
        let dev = gpu();
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        let a =
            T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev).unwrap();
        let b = a.transpose(0, 1).unwrap();
        assert_eq!(b.shape().dims(), &[3, 2]);
        let data = b.to_f64_vec().unwrap();
        assert_approx_vec(&data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Memory pool
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_pool_basic() {
        let dev = gpu();
        let stats = dev.pool_stats();
        assert_eq!(stats.cached_bytes, 0);
        // Create and drop a tensor — the pool reclaim logic is opt-in
        let _t = T::ones((100,), DType::F32, &dev).unwrap();
    }

    // ─────────────────────────────────────────────────────────────────────
    // Large-scale stress tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_large_add() {
        let dev = gpu();
        let n = 1_000_000;
        let a = T::ones(vec![n], DType::F32, &dev).unwrap();
        let b = T::ones(vec![n], DType::F32, &dev).unwrap();
        let c = a.add(&b).unwrap();
        let sum = c.sum_all().unwrap().to_scalar_f64().unwrap();
        assert!(approx(sum, 2.0 * n as f64, 1e-1));
    }

    #[test]
    fn test_large_matmul() {
        let dev = gpu();
        // 128x128 matmul
        let a = T::ones((128, 128), DType::F32, &dev).unwrap();
        let b = T::ones((128, 128), DType::F32, &dev).unwrap();
        let c = a.matmul(&b).unwrap();
        // Every element should be 128 (sum of 128 ones)
        let data = c.to_f64_vec().unwrap();
        assert!(approx(data[0], 128.0, 1e-2));
        assert!(approx(data[127 * 128 + 127], 128.0, 1e-2));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Chained operations
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_chain_ops() {
        let dev = gpu();
        // (a + b) * c → relu → sum
        let a = T::from_f64_slice(&[-1.0, 2.0, -3.0, 4.0], (4,), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[2.0, -1.0, 4.0, -3.0], (4,), DType::F32, &dev).unwrap();
        // a+b = [1, 1, 1, 1]
        let ab = a.add(&b).unwrap();
        // *2 = [2, 2, 2, 2]
        let c = T::full((4,), 2.0, DType::F32, &dev).unwrap();
        let abc = ab.mul(&c).unwrap();
        // relu (all positive so unchanged) = [2, 2, 2, 2]
        let r = abc.relu().unwrap();
        // sum = 8
        let s = r.sum_all().unwrap().to_scalar_f64().unwrap();
        assert!(approx(s, 8.0, 1e-4));
    }

    #[test]
    fn test_softmax_equiv() {
        let dev = gpu();
        // Manual softmax: exp(x) / sum(exp(x)) — now works with broadcast!
        let x = T::from_f64_slice(&[1.0, 2.0, 3.0], (1, 3), DType::F32, &dev).unwrap();
        let ex = x.exp().unwrap();
        let s = ex.sum(1, true).unwrap();
        let softmax = ex.div(&s).unwrap();
        let data = softmax.to_f64_vec().unwrap();
        let total: f64 = data.iter().sum();
        assert!(
            approx(total, 1.0, 1e-4),
            "softmax should sum to 1, got {total}"
        );
        // Check monotonicity: data[2] > data[1] > data[0]
        assert!(data[2] > data[1] && data[1] > data[0]);
    }

    // ─────────────────────────────────────────────────────────────────────
    // DType: F64
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_f64_binary_ops() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F64, &dev).unwrap();
        let b = T::from_f64_slice(&[4.0, 5.0, 6.0], (3,), DType::F64, &dev).unwrap();
        let c = a.add(&b).unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[5.0, 7.0, 9.0], 1e-10);
    }

    #[test]
    fn test_f64_unary_ops() {
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 4.0, 9.0], (3,), DType::F64, &dev).unwrap();
        let c = a.sqrt().unwrap();
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-10);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Broadcasting
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_add_scalar() {
        // [3] + [1] → [3]
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0], (3,), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[10.0], (1,), DType::F32, &dev).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape().dims(), &[3]);
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[11.0, 12.0, 13.0], 1e-5);
    }

    #[test]
    fn test_broadcast_2d() {
        // [2,3] + [1,3] → [2,3]
        let dev = gpu();
        let a =
            T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[10.0, 20.0, 30.0], (1, 3), DType::F32, &dev).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        assert_approx_vec(
            &c.to_f64_vec().unwrap(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
            1e-5,
        );
    }

    #[test]
    fn test_broadcast_div_keepdim() {
        // [1,3] / [1,1] → [1,3]  — the pattern used in softmax/layernorm
        let dev = gpu();
        let a = T::from_f64_slice(&[6.0, 12.0, 18.0], (1, 3), DType::F32, &dev).unwrap();
        let b = T::from_f64_slice(&[3.0], (1, 1), DType::F32, &dev).unwrap();
        let c = a.div(&b).unwrap();
        assert_eq!(c.shape().dims(), &[1, 3]);
        assert_approx_vec(&c.to_f64_vec().unwrap(), &[2.0, 4.0, 6.0], 1e-5);
    }

    #[test]
    fn test_broadcast_4d() {
        // [2,2,1,1] * [1,1,2,2] → [2,2,2,2]
        let dev = gpu();
        let a = T::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2, 1, 1), DType::F32, &dev).unwrap();
        let b =
            T::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], (1, 1, 2, 2), DType::F32, &dev).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2, 2, 2]);
        let data = c.to_f64_vec().unwrap();
        // a[0,0]=1, a[0,1]=2, a[1,0]=3, a[1,1]=4
        // b = [[10,20],[30,40]] broadcast across first 2 dims
        // c[0,0,:,:] = 1 * [[10,20],[30,40]] = [10,20,30,40]
        assert_approx_vec(&data[0..4], &[10.0, 20.0, 30.0, 40.0], 1e-5);
        // c[0,1,:,:] = 2 * [[10,20],[30,40]] = [20,40,60,80]
        assert_approx_vec(&data[4..8], &[20.0, 40.0, 60.0, 80.0], 1e-5);
    }

    #[test]
    fn test_broadcast_sub_mul() {
        // [2,3] - [1,1] → [2,3]
        let dev = gpu();
        let a = T::from_f64_slice(
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            (2, 3),
            DType::F32,
            &dev,
        )
        .unwrap();
        let b = T::from_f64_slice(&[5.0], (1, 1), DType::F32, &dev).unwrap();
        let c = a.sub(&b).unwrap();
        assert_approx_vec(
            &c.to_f64_vec().unwrap(),
            &[5.0, 15.0, 25.0, 35.0, 45.0, 55.0],
            1e-5,
        );
    }
}
