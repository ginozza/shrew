// CUDA Backend — GPU-accelerated compute backend using cudarc
//
// This module provides a full CUDA implementation of the Shrew Backend trait.
// All tensor operations run on NVIDIA GPUs via custom CUDA kernels (compiled
// at device creation via NVRTC) and cuBLAS for matrix multiplication.
//
// ARCHITECTURE:
// - CudaDevice wraps cudarc's device handle + cuBLAS handle
// - CudaStorage is an enum over CudaSlice<T> for each supported dtype
// - All kernels operate on contiguous data; non-contiguous inputs are
//   first copied to contiguous layout using a strided-copy kernel
// - Random number generation happens on host and is transferred to device
// - F16 and BF16 are stored as CudaSlice<u16> and computed via promote-to-F32
//   CUDA kernels (portable across all GPU architectures)
//
// USAGE:
//   let device = CudaDevice::new(0)?;  // GPU ordinal 0
//   let tensor = Tensor::<CudaBackend>::zeros(&[2, 3], DType::F32, &device)?;

mod kernels;
pub mod pool;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use half::{bf16, f16};
use pool::CudaMemPool;
use std::fmt;
use std::sync::Arc;

use shrew_core::backend::{
    Backend, BackendDevice, BackendStorage, BinaryOp, CmpOp, ReduceOp, UnaryOp,
};
use shrew_core::dtype::DType;
use shrew_core::error::{Error, Result};
use shrew_core::layout::Layout;
use shrew_core::shape::Shape;

// CudaDevice — Wraps a cudarc CUDA device + cuBLAS handle

/// A CUDA device handle. Contains the cudarc device and a cuBLAS handle
/// for matrix multiplication. Clonable (uses Arc internally).
pub struct CudaDevice {
    dev: Arc<cudarc::driver::CudaDevice>,
    blas: Arc<CudaBlas>,
    pool: Arc<CudaMemPool>,
    ordinal: usize,
}

impl CudaDevice {
    /// Create a new CUDA device for the given GPU ordinal (0, 1, ...).
    /// Compiles all Shrew CUDA kernels on first creation.
    pub fn new(ordinal: usize) -> Result<Self> {
        let dev = cudarc::driver::CudaDevice::new(ordinal)
            .map_err(|e| Error::msg(format!("CUDA device creation failed: {e}")))?;

        let blas = CudaBlas::new(dev.clone())
            .map_err(|e| Error::msg(format!("cuBLAS init failed: {e}")))?;

        // Compile and load all kernels
        // Query the device compute capability and target it with NVRTC.
        // Use sm_XX (native SASS) instead of compute_XX (PTX) to avoid
        // PTX version mismatches between toolkit and driver versions.
        let major = dev.attribute(cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(8);
        let minor = dev.attribute(cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(9);
        let arch_str: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        let opts = CompileOptions {
            arch: Some(arch_str),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(kernels::KERNEL_SOURCE, opts)
            .map_err(|e| Error::msg(format!("NVRTC compilation failed: {e}")))?;
        dev.load_ptx(ptx, kernels::MODULE_NAME, kernels::KERNEL_NAMES)
            .map_err(|e| Error::msg(format!("PTX load failed: {e}")))?;

        Ok(CudaDevice {
            dev,
            blas: Arc::new(blas),
            pool: Arc::new(CudaMemPool::new()),
            ordinal,
        })
    }

    /// Get the underlying cudarc device handle.
    pub fn device(&self) -> &Arc<cudarc::driver::CudaDevice> {
        &self.dev
    }

    /// Get the cuBLAS handle.
    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// Get a compiled kernel function by name.
    fn get_func(&self, name: &str) -> Result<cudarc::driver::CudaFunction> {
        self.dev
            .get_func(kernels::MODULE_NAME, name)
            .ok_or_else(|| Error::msg(format!("CUDA kernel '{name}' not found")))
    }

    // ── Memory pool helpers ──────────────────────────────────────────────

    /// Get the memory pool.
    pub fn pool(&self) -> &CudaMemPool {
        &self.pool
    }

    /// Release all cached GPU memory back to the CUDA driver.
    pub fn empty_cache(&self) {
        self.pool.empty_cache();
    }

    /// Return pool statistics (cached bytes, hits, misses, etc.).
    pub fn pool_stats(&self) -> pool::PoolStats {
        self.pool.stats()
    }

    /// Reclaim a CudaStorage buffer into the pool for future reuse.
    pub fn reclaim(&self, storage: CudaStorage) {
        self.pool.reclaim_storage(storage);
    }

    // ── Pool-aware allocation helpers ────────────────────────────────────

    /// Allocate `n` elements from the pool (content undefined).
    pub fn pool_alloc_f32(&self, n: usize) -> Result<CudaSlice<f32>> {
        self.pool
            .alloc_f32(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc f32: {e}")))
    }
    pub fn pool_alloc_f64(&self, n: usize) -> Result<CudaSlice<f64>> {
        self.pool
            .alloc_f64(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc f64: {e}")))
    }
    pub fn pool_alloc_u16(&self, n: usize) -> Result<CudaSlice<u16>> {
        self.pool
            .alloc_u16(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc u16: {e}")))
    }
    pub fn pool_alloc_u8(&self, n: usize) -> Result<CudaSlice<u8>> {
        self.pool
            .alloc_u8(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc u8: {e}")))
    }
    pub fn pool_alloc_u32(&self, n: usize) -> Result<CudaSlice<u32>> {
        self.pool
            .alloc_u32(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc u32: {e}")))
    }
    pub fn pool_alloc_i64(&self, n: usize) -> Result<CudaSlice<i64>> {
        self.pool
            .alloc_i64(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc i64: {e}")))
    }

    /// Allocate `n` elements from the pool, zeroed.
    pub fn pool_alloc_zeros_f32(&self, n: usize) -> Result<CudaSlice<f32>> {
        self.pool
            .alloc_zeros_f32(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc zeros f32: {e}")))
    }
    pub fn pool_alloc_zeros_f64(&self, n: usize) -> Result<CudaSlice<f64>> {
        self.pool
            .alloc_zeros_f64(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc zeros f64: {e}")))
    }
    pub fn pool_alloc_zeros_u16(&self, n: usize) -> Result<CudaSlice<u16>> {
        self.pool
            .alloc_zeros_u16(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc zeros u16: {e}")))
    }
    pub fn pool_alloc_zeros_u8(&self, n: usize) -> Result<CudaSlice<u8>> {
        self.pool
            .alloc_zeros_u8(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc zeros u8: {e}")))
    }
    pub fn pool_alloc_zeros_u32(&self, n: usize) -> Result<CudaSlice<u32>> {
        self.pool
            .alloc_zeros_u32(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc zeros u32: {e}")))
    }
    pub fn pool_alloc_zeros_i64(&self, n: usize) -> Result<CudaSlice<i64>> {
        self.pool
            .alloc_zeros_i64(&self.dev, n)
            .map_err(|e| Error::msg(format!("pool alloc zeros i64: {e}")))
    }
}

impl Clone for CudaDevice {
    fn clone(&self) -> Self {
        CudaDevice {
            dev: self.dev.clone(),
            blas: self.blas.clone(),
            pool: self.pool.clone(),
            ordinal: self.ordinal,
        }
    }
}

impl fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CudaDevice(cuda:{})", self.ordinal)
    }
}

// Safety: cudarc's device is thread-safe (CUDA runtime is thread-safe)
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl BackendDevice for CudaDevice {
    fn name(&self) -> String {
        format!("cuda:{}", self.ordinal)
    }
}

// CudaStorage — Device memory for each supported dtype

/// GPU-side storage. Each variant wraps a cudarc CudaSlice for the corresponding dtype.
/// F16 and BF16 are stored as CudaSlice<u16> (bit-level representation).
pub enum CudaStorage {
    F16(CudaSlice<u16>),
    BF16(CudaSlice<u16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
    U8(CudaSlice<u8>),
    U32(CudaSlice<u32>),
    I64(CudaSlice<i64>),
}

impl Clone for CudaStorage {
    fn clone(&self) -> Self {
        match self {
            CudaStorage::F16(s) => CudaStorage::F16(s.clone()),
            CudaStorage::BF16(s) => CudaStorage::BF16(s.clone()),
            CudaStorage::F32(s) => CudaStorage::F32(s.clone()),
            CudaStorage::F64(s) => CudaStorage::F64(s.clone()),
            CudaStorage::U8(s) => CudaStorage::U8(s.clone()),
            CudaStorage::U32(s) => CudaStorage::U32(s.clone()),
            CudaStorage::I64(s) => CudaStorage::I64(s.clone()),
        }
    }
}

impl fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaStorage::F16(s) => write!(f, "CudaStorage::F16(len={})", s.len()),
            CudaStorage::BF16(s) => write!(f, "CudaStorage::BF16(len={})", s.len()),
            CudaStorage::F32(s) => write!(f, "CudaStorage::F32(len={})", s.len()),
            CudaStorage::F64(s) => write!(f, "CudaStorage::F64(len={})", s.len()),
            CudaStorage::U8(s) => write!(f, "CudaStorage::U8(len={})", s.len()),
            CudaStorage::U32(s) => write!(f, "CudaStorage::U32(len={})", s.len()),
            CudaStorage::I64(s) => write!(f, "CudaStorage::I64(len={})", s.len()),
        }
    }
}

unsafe impl Send for CudaStorage {}
unsafe impl Sync for CudaStorage {}

impl BackendStorage for CudaStorage {
    fn dtype(&self) -> DType {
        match self {
            CudaStorage::F16(_) => DType::F16,
            CudaStorage::BF16(_) => DType::BF16,
            CudaStorage::F32(_) => DType::F32,
            CudaStorage::F64(_) => DType::F64,
            CudaStorage::U8(_) => DType::U8,
            CudaStorage::U32(_) => DType::U32,
            CudaStorage::I64(_) => DType::I64,
        }
    }

    fn len(&self) -> usize {
        match self {
            CudaStorage::F16(s) => s.len(),
            CudaStorage::BF16(s) => s.len(),
            CudaStorage::F32(s) => s.len(),
            CudaStorage::F64(s) => s.len(),
            CudaStorage::U8(s) => s.len(),
            CudaStorage::U32(s) => s.len(),
            CudaStorage::I64(s) => s.len(),
        }
    }
}

// Helpers

/// Standard CUDA launch configuration for N elements.
fn launch_cfg(n: usize) -> LaunchConfig {
    const BLOCK: u32 = 256;
    let grid = (n as u32).div_ceil(BLOCK);
    LaunchConfig {
        block_dim: (BLOCK, 1, 1),
        grid_dim: (grid.max(1), 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Make a CudaStorage contiguous according to the given layout.
/// If already contiguous with offset 0, returns a clone (cheap Arc bump).
/// Otherwise launches a strided-copy kernel.
fn ensure_contiguous(
    storage: &CudaStorage,
    layout: &Layout,
    device: &CudaDevice,
) -> Result<CudaStorage> {
    if layout.is_contiguous() && layout.offset() == 0 {
        return Ok(storage.clone());
    }

    let n = layout.elem_count();
    let cfg = launch_cfg(n);
    let ndim = layout.rank() as i32;
    let offset = layout.offset() as i32;

    // Upload shape and strides to device
    let shape_i32: Vec<i32> = layout.dims().iter().map(|&d| d as i32).collect();
    let strides_i32: Vec<i32> = layout.strides().iter().map(|&s| s as i32).collect();
    let shape_dev = device
        .dev
        .htod_copy(shape_i32)
        .map_err(|e| Error::msg(format!("htod shape: {e}")))?;
    let strides_dev = device
        .dev
        .htod_copy(strides_i32)
        .map_err(|e| Error::msg(format!("htod strides: {e}")))?;

    match storage {
        CudaStorage::F16(src) | CudaStorage::BF16(src) => {
            let mut dst: CudaSlice<u16> = device
                .dev
                .alloc_zeros(n)
                .map_err(|e| Error::msg(format!("alloc: {e}")))?;
            let func = device.get_func("to_contiguous_u16")?;
            unsafe {
                func.launch(
                    cfg,
                    (
                        src,
                        &mut dst,
                        &shape_dev,
                        &strides_dev,
                        offset,
                        ndim,
                        n as u32,
                    ),
                )
            }
            .map_err(|e| Error::msg(format!("launch to_contiguous_u16: {e}")))?;
            match storage {
                CudaStorage::F16(_) => Ok(CudaStorage::F16(dst)),
                _ => Ok(CudaStorage::BF16(dst)),
            }
        }
        CudaStorage::F32(src) => {
            let mut dst: CudaSlice<f32> = device
                .dev
                .alloc_zeros(n)
                .map_err(|e| Error::msg(format!("alloc: {e}")))?;
            let func = device.get_func("to_contiguous_f32")?;
            unsafe {
                func.launch(
                    cfg,
                    (
                        src,
                        &mut dst,
                        &shape_dev,
                        &strides_dev,
                        offset,
                        ndim,
                        n as u32,
                    ),
                )
            }
            .map_err(|e| Error::msg(format!("launch to_contiguous_f32: {e}")))?;
            Ok(CudaStorage::F32(dst))
        }
        CudaStorage::F64(src) => {
            let mut dst: CudaSlice<f64> = device
                .dev
                .alloc_zeros(n)
                .map_err(|e| Error::msg(format!("alloc: {e}")))?;
            let func = device.get_func("to_contiguous_f64")?;
            unsafe {
                func.launch(
                    cfg,
                    (
                        src,
                        &mut dst,
                        &shape_dev,
                        &strides_dev,
                        offset,
                        ndim,
                        n as u32,
                    ),
                )
            }
            .map_err(|e| Error::msg(format!("launch to_contiguous_f64: {e}")))?;
            Ok(CudaStorage::F64(dst))
        }
        CudaStorage::U8(src) => {
            let mut dst: CudaSlice<u8> = device
                .dev
                .alloc_zeros(n)
                .map_err(|e| Error::msg(format!("alloc: {e}")))?;
            let func = device.get_func("to_contiguous_u8")?;
            unsafe {
                func.launch(
                    cfg,
                    (
                        src,
                        &mut dst,
                        &shape_dev,
                        &strides_dev,
                        offset,
                        ndim,
                        n as u32,
                    ),
                )
            }
            .map_err(|e| Error::msg(format!("launch to_contiguous_u8: {e}")))?;
            Ok(CudaStorage::U8(dst))
        }
        _ => Err(Error::msg(
            "to_contiguous not implemented for this dtype on CUDA",
        )),
    }
}

/// Get the CudaDevice from a CudaSlice (needed to find the device for operations).
fn device_from_storage(storage: &CudaStorage) -> Arc<cudarc::driver::CudaDevice> {
    match storage {
        CudaStorage::F16(s) => s.device(),
        CudaStorage::BF16(s) => s.device(),
        CudaStorage::F32(s) => s.device(),
        CudaStorage::F64(s) => s.device(),
        CudaStorage::U8(s) => s.device(),
        CudaStorage::U32(s) => s.device(),
        CudaStorage::I64(s) => s.device(),
    }
}

/// Reconstruct a CudaDevice from a storage reference (for Backend trait methods
/// that don't receive the device explicitly).
fn dev_from_storage(storage: &CudaStorage) -> Result<CudaDevice> {
    let raw_dev = device_from_storage(storage);
    let blas = CudaBlas::new(raw_dev.clone()).map_err(|e| Error::msg(format!("blas: {e}")))?;
    Ok(CudaDevice {
        dev: raw_dev,
        blas: Arc::new(blas),
        pool: Arc::new(CudaMemPool::new()),
        ordinal: 0,
    })
}

// CudaBackend — The Backend trait implementation

/// The CUDA GPU backend. This is a zero-sized marker type.
#[derive(Clone, Debug)]
pub struct CudaBackend;

impl Backend for CudaBackend {
    type Device = CudaDevice;
    type Storage = CudaStorage;

    // ---- Creation ----

    fn zeros(shape: &Shape, dtype: DType, device: &CudaDevice) -> Result<CudaStorage> {
        let n = shape.elem_count();
        match dtype {
            DType::F16 => {
                let s: CudaSlice<u16> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc zeros f16: {e}")))?;
                Ok(CudaStorage::F16(s))
            }
            DType::BF16 => {
                let s: CudaSlice<u16> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc zeros bf16: {e}")))?;
                Ok(CudaStorage::BF16(s))
            }
            DType::F32 => {
                let s: CudaSlice<f32> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc zeros f32: {e}")))?;
                Ok(CudaStorage::F32(s))
            }
            DType::F64 => {
                let s: CudaSlice<f64> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc zeros f64: {e}")))?;
                Ok(CudaStorage::F64(s))
            }
            DType::U8 => {
                let s: CudaSlice<u8> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc zeros u8: {e}")))?;
                Ok(CudaStorage::U8(s))
            }
            DType::U32 => {
                let s: CudaSlice<u32> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc zeros u32: {e}")))?;
                Ok(CudaStorage::U32(s))
            }
            DType::I64 => {
                let s: CudaSlice<i64> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc zeros i64: {e}")))?;
                Ok(CudaStorage::I64(s))
            }
        }
    }

    fn ones(shape: &Shape, dtype: DType, device: &CudaDevice) -> Result<CudaStorage> {
        Self::full(shape, 1.0, dtype, device)
    }

    fn full(shape: &Shape, val: f64, dtype: DType, device: &CudaDevice) -> Result<CudaStorage> {
        let n = shape.elem_count();
        let cfg = launch_cfg(n);
        match dtype {
            DType::F16 => {
                let mut s: CudaSlice<u16> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = device.get_func("fill_f16")?;
                unsafe { func.launch(cfg, (&mut s, val as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("fill_f16: {e}")))?;
                Ok(CudaStorage::F16(s))
            }
            DType::BF16 => {
                let mut s: CudaSlice<u16> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = device.get_func("fill_bf16")?;
                unsafe { func.launch(cfg, (&mut s, val as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("fill_bf16: {e}")))?;
                Ok(CudaStorage::BF16(s))
            }
            DType::F32 => {
                let mut s: CudaSlice<f32> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = device.get_func("fill_f32")?;
                unsafe { func.launch(cfg, (&mut s, val as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("fill_f32: {e}")))?;
                Ok(CudaStorage::F32(s))
            }
            DType::F64 => {
                let mut s: CudaSlice<f64> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = device.get_func("fill_f64")?;
                unsafe { func.launch(cfg, (&mut s, val, n as u32)) }
                    .map_err(|e| Error::msg(format!("fill_f64: {e}")))?;
                Ok(CudaStorage::F64(s))
            }
            DType::U8 => {
                let mut s: CudaSlice<u8> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = device.get_func("fill_u8")?;
                unsafe { func.launch(cfg, (&mut s, val as u8, n as u32)) }
                    .map_err(|e| Error::msg(format!("fill_u8: {e}")))?;
                Ok(CudaStorage::U8(s))
            }
            DType::U32 => {
                let mut s: CudaSlice<u32> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = device.get_func("fill_u32")?;
                unsafe { func.launch(cfg, (&mut s, val as u32, n as u32)) }
                    .map_err(|e| Error::msg(format!("fill_u32: {e}")))?;
                Ok(CudaStorage::U32(s))
            }
            DType::I64 => {
                let mut s: CudaSlice<i64> = device
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = device.get_func("fill_i64")?;
                unsafe { func.launch(cfg, (&mut s, val as i64, n as u32)) }
                    .map_err(|e| Error::msg(format!("fill_i64: {e}")))?;
                Ok(CudaStorage::I64(s))
            }
        }
    }

    fn from_f64_slice(data: &[f64], dtype: DType, device: &CudaDevice) -> Result<CudaStorage> {
        match dtype {
            DType::F16 => {
                let host: Vec<u16> = data.iter().map(|&v| f16::from_f64(v).to_bits()).collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod f16: {e}")))?;
                Ok(CudaStorage::F16(s))
            }
            DType::BF16 => {
                let host: Vec<u16> = data.iter().map(|&v| bf16::from_f64(v).to_bits()).collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod bf16: {e}")))?;
                Ok(CudaStorage::BF16(s))
            }
            DType::F32 => {
                let host: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod f32: {e}")))?;
                Ok(CudaStorage::F32(s))
            }
            DType::F64 => {
                let s = device
                    .dev
                    .htod_copy(data.to_vec())
                    .map_err(|e| Error::msg(format!("htod f64: {e}")))?;
                Ok(CudaStorage::F64(s))
            }
            DType::U8 => {
                let host: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod u8: {e}")))?;
                Ok(CudaStorage::U8(s))
            }
            DType::U32 => {
                let host: Vec<u32> = data.iter().map(|&v| v as u32).collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod u32: {e}")))?;
                Ok(CudaStorage::U32(s))
            }
            DType::I64 => {
                let host: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod i64: {e}")))?;
                Ok(CudaStorage::I64(s))
            }
        }
    }

    fn rand_uniform(shape: &Shape, dtype: DType, device: &CudaDevice) -> Result<CudaStorage> {
        // Generate on host, transfer to device
        use rand::Rng;
        let n = shape.elem_count();
        let mut rng = rand::thread_rng();
        match dtype {
            DType::F16 => {
                let host: Vec<u16> = (0..n)
                    .map(|_| f16::from_f32(rng.gen::<f32>()).to_bits())
                    .collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod rand_uniform f16: {e}")))?;
                Ok(CudaStorage::F16(s))
            }
            DType::BF16 => {
                let host: Vec<u16> = (0..n)
                    .map(|_| bf16::from_f32(rng.gen::<f32>()).to_bits())
                    .collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod rand_uniform bf16: {e}")))?;
                Ok(CudaStorage::BF16(s))
            }
            DType::F32 => {
                let host: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod rand_uniform f32: {e}")))?;
                Ok(CudaStorage::F32(s))
            }
            DType::F64 => {
                let host: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod rand_uniform f64: {e}")))?;
                Ok(CudaStorage::F64(s))
            }
            _ => Err(Error::msg(format!(
                "rand_uniform not supported for {:?}",
                dtype
            ))),
        }
    }

    fn rand_normal(shape: &Shape, dtype: DType, device: &CudaDevice) -> Result<CudaStorage> {
        use rand::Rng;
        use rand_distr::StandardNormal;
        let n = shape.elem_count();
        let mut rng = rand::thread_rng();
        match dtype {
            DType::F16 => {
                let host: Vec<u16> = (0..n)
                    .map(|_| f16::from_f32(rng.sample::<f32, _>(StandardNormal)).to_bits())
                    .collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod rand_normal f16: {e}")))?;
                Ok(CudaStorage::F16(s))
            }
            DType::BF16 => {
                let host: Vec<u16> = (0..n)
                    .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal)).to_bits())
                    .collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod rand_normal bf16: {e}")))?;
                Ok(CudaStorage::BF16(s))
            }
            DType::F32 => {
                let host: Vec<f32> = (0..n)
                    .map(|_| rng.sample::<f32, _>(StandardNormal))
                    .collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod rand_normal f32: {e}")))?;
                Ok(CudaStorage::F32(s))
            }
            DType::F64 => {
                let host: Vec<f64> = (0..n)
                    .map(|_| rng.sample::<f64, _>(StandardNormal))
                    .collect();
                let s = device
                    .dev
                    .htod_copy(host)
                    .map_err(|e| Error::msg(format!("htod rand_normal f64: {e}")))?;
                Ok(CudaStorage::F64(s))
            }
            _ => Err(Error::msg(format!(
                "rand_normal not supported for {:?}",
                dtype
            ))),
        }
    }

    // ---- Binary ops ----

    fn binary_op(
        op: BinaryOp,
        lhs: &CudaStorage,
        lhs_layout: &Layout,
        rhs: &CudaStorage,
        rhs_layout: &Layout,
    ) -> Result<CudaStorage> {
        let dev = dev_from_storage(lhs)?;

        // Make contiguous
        let lhs_c = ensure_contiguous(lhs, lhs_layout, &dev)?;
        let rhs_c = ensure_contiguous(rhs, rhs_layout, &dev)?;

        let lhs_shape = lhs_layout.shape();
        let rhs_shape = rhs_layout.shape();

        let op_name = match op {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
        };

        // Check if broadcasting is needed
        let needs_broadcast = lhs_shape.dims() != rhs_shape.dims();

        if needs_broadcast {
            // Broadcast path: compute output shape and per-operand strides
            let out_shape = Shape::broadcast_shape(lhs_shape, rhs_shape)?;
            let a_strides = lhs_shape.broadcast_strides(&out_shape);
            let b_strides = rhs_shape.broadcast_strides(&out_shape);
            let n = out_shape.elem_count();
            let cfg = launch_cfg(n);

            // Upload shape & strides to GPU
            let out_dims_u32: Vec<u32> = out_shape.dims().iter().map(|&d| d as u32).collect();
            let a_strides_u32: Vec<u32> = a_strides.iter().map(|&s| s as u32).collect();
            let b_strides_u32: Vec<u32> = b_strides.iter().map(|&s| s as u32).collect();
            let rank = out_shape.dims().len() as u32;

            let dims_gpu = dev
                .dev
                .htod_copy(out_dims_u32)
                .map_err(|e| Error::msg(format!("alloc dims: {e}")))?;
            let a_strides_gpu = dev
                .dev
                .htod_copy(a_strides_u32)
                .map_err(|e| Error::msg(format!("alloc a_strides: {e}")))?;
            let b_strides_gpu = dev
                .dev
                .htod_copy(b_strides_u32)
                .map_err(|e| Error::msg(format!("alloc b_strides: {e}")))?;

            match (&lhs_c, &rhs_c) {
                (CudaStorage::F32(a), CudaStorage::F32(b)) => {
                    let mut out: CudaSlice<f32> = dev
                        .dev
                        .alloc_zeros(n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("bcast_binary_{op_name}_f32"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                a,
                                b,
                                &mut out,
                                &dims_gpu,
                                &a_strides_gpu,
                                &b_strides_gpu,
                                rank,
                                n as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("bcast binary op: {e}")))?;
                    Ok(CudaStorage::F32(out))
                }
                (CudaStorage::F64(a), CudaStorage::F64(b)) => {
                    let mut out: CudaSlice<f64> = dev
                        .dev
                        .alloc_zeros(n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("bcast_binary_{op_name}_f64"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                a,
                                b,
                                &mut out,
                                &dims_gpu,
                                &a_strides_gpu,
                                &b_strides_gpu,
                                rank,
                                n as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("bcast binary op: {e}")))?;
                    Ok(CudaStorage::F64(out))
                }
                (CudaStorage::F16(a), CudaStorage::F16(b)) => {
                    let mut out: CudaSlice<u16> = dev
                        .dev
                        .alloc_zeros(n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("bcast_binary_{op_name}_f16"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                a,
                                b,
                                &mut out,
                                &dims_gpu,
                                &a_strides_gpu,
                                &b_strides_gpu,
                                rank,
                                n as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("bcast binary op: {e}")))?;
                    Ok(CudaStorage::F16(out))
                }
                (CudaStorage::BF16(a), CudaStorage::BF16(b)) => {
                    let mut out: CudaSlice<u16> = dev
                        .dev
                        .alloc_zeros(n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("bcast_binary_{op_name}_bf16"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                a,
                                b,
                                &mut out,
                                &dims_gpu,
                                &a_strides_gpu,
                                &b_strides_gpu,
                                rank,
                                n as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("bcast binary op: {e}")))?;
                    Ok(CudaStorage::BF16(out))
                }
                _ => Err(Error::msg(
                    "bcast binary_op: dtype mismatch or unsupported dtype",
                )),
            }
        } else {
            // Fast path: same shape, element-wise
            let n = lhs_layout.elem_count();
            let cfg = launch_cfg(n);

            match (&lhs_c, &rhs_c) {
                (CudaStorage::F16(a), CudaStorage::F16(b)) => {
                    let mut out: CudaSlice<u16> = dev
                        .dev
                        .alloc_zeros(n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("binary_{op_name}_f16"))?;
                    unsafe { func.launch(cfg, (a, b, &mut out, n as u32)) }
                        .map_err(|e| Error::msg(format!("binary op: {e}")))?;
                    Ok(CudaStorage::F16(out))
                }
                (CudaStorage::BF16(a), CudaStorage::BF16(b)) => {
                    let mut out: CudaSlice<u16> = dev
                        .dev
                        .alloc_zeros(n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("binary_{op_name}_bf16"))?;
                    unsafe { func.launch(cfg, (a, b, &mut out, n as u32)) }
                        .map_err(|e| Error::msg(format!("binary op: {e}")))?;
                    Ok(CudaStorage::BF16(out))
                }
                (CudaStorage::F32(a), CudaStorage::F32(b)) => {
                    let mut out: CudaSlice<f32> = dev
                        .dev
                        .alloc_zeros(n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("binary_{op_name}_f32"))?;
                    unsafe { func.launch(cfg, (a, b, &mut out, n as u32)) }
                        .map_err(|e| Error::msg(format!("binary op: {e}")))?;
                    Ok(CudaStorage::F32(out))
                }
                (CudaStorage::F64(a), CudaStorage::F64(b)) => {
                    let mut out: CudaSlice<f64> = dev
                        .dev
                        .alloc_zeros(n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("binary_{op_name}_f64"))?;
                    unsafe { func.launch(cfg, (a, b, &mut out, n as u32)) }
                        .map_err(|e| Error::msg(format!("binary op: {e}")))?;
                    Ok(CudaStorage::F64(out))
                }
                _ => Err(Error::msg("binary_op: dtype mismatch or unsupported dtype")),
            }
        }
    }

    // ---- Unary ops ----

    fn unary_op(op: UnaryOp, input: &CudaStorage, layout: &Layout) -> Result<CudaStorage> {
        let dev = dev_from_storage(input)?;

        let input_c = ensure_contiguous(input, layout, &dev)?;
        let n = layout.elem_count();
        let cfg = launch_cfg(n);

        let op_name = match op {
            UnaryOp::Neg => "neg",
            UnaryOp::Abs => "abs",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Relu => "relu",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Gelu => "gelu",
            UnaryOp::Silu => "silu",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Square => "square",
            UnaryOp::Floor => "floor",
            UnaryOp::Ceil => "ceil",
            UnaryOp::Round => "round",
        };

        match &input_c {
            CudaStorage::F16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func(&format!("unary_{op_name}_f16"))?;
                unsafe { func.launch(cfg, (inp, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("unary op: {e}")))?;
                Ok(CudaStorage::F16(out))
            }
            CudaStorage::BF16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func(&format!("unary_{op_name}_bf16"))?;
                unsafe { func.launch(cfg, (inp, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("unary op: {e}")))?;
                Ok(CudaStorage::BF16(out))
            }
            CudaStorage::F32(inp) => {
                let mut out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func(&format!("unary_{op_name}_f32"))?;
                unsafe { func.launch(cfg, (inp, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("unary op: {e}")))?;
                Ok(CudaStorage::F32(out))
            }
            CudaStorage::F64(inp) => {
                let mut out: CudaSlice<f64> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func(&format!("unary_{op_name}_f64"))?;
                unsafe { func.launch(cfg, (inp, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("unary op: {e}")))?;
                Ok(CudaStorage::F64(out))
            }
            _ => Err(Error::msg("unary_op: only float types supported")),
        }
    }

    // ---- Reduction ----

    fn reduce_op(
        op: ReduceOp,
        input: &CudaStorage,
        layout: &Layout,
        dims: &[usize],
        _keep_dim: bool,
    ) -> Result<CudaStorage> {
        let dev = dev_from_storage(input)?;

        let input_c = ensure_contiguous(input, layout, &dev)?;
        let shape_dims = layout.dims();
        let rank = shape_dims.len();

        // Determine reduction dimension (or all)
        let reduce_dim = if dims.is_empty() {
            None
        } else if dims.len() == 1 {
            Some(dims[0])
        } else {
            return Err(Error::msg(
                "CUDA reduce_op: multi-dim reduction not yet supported",
            ));
        };

        let (outer_size, reduce_size, inner_size) = if let Some(dim) = reduce_dim {
            if dim >= rank {
                return Err(Error::msg(format!(
                    "dim {dim} out of range for rank {rank}"
                )));
            }
            let outer: usize = shape_dims[..dim].iter().product::<usize>().max(1);
            let red = shape_dims[dim];
            let inner: usize = shape_dims[dim + 1..].iter().product::<usize>().max(1);
            (outer, red, inner)
        } else {
            let total: usize = shape_dims.iter().product();
            (1usize, total, 1usize)
        };

        let out_n = outer_size * inner_size;
        let cfg = launch_cfg(out_n);

        let is_arg = matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin);

        let op_name = match op {
            ReduceOp::Sum => "sum",
            ReduceOp::Mean => "mean",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::ArgMax => "argmax",
            ReduceOp::ArgMin => "argmin",
        };

        if is_arg {
            // ArgMax/ArgMin → output I64
            let mut out: CudaSlice<i64> = dev
                .dev
                .alloc_zeros(out_n)
                .map_err(|e| Error::msg(format!("alloc: {e}")))?;
            match &input_c {
                CudaStorage::F16(inp) => {
                    let func = dev.get_func(&format!("reduce_{op_name}_f16"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer_size as u32,
                                reduce_size as u32,
                                inner_size as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("reduce: {e}")))?;
                }
                CudaStorage::BF16(inp) => {
                    let func = dev.get_func(&format!("reduce_{op_name}_bf16"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer_size as u32,
                                reduce_size as u32,
                                inner_size as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("reduce: {e}")))?;
                }
                CudaStorage::F32(inp) => {
                    let func = dev.get_func(&format!("reduce_{op_name}_f32"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer_size as u32,
                                reduce_size as u32,
                                inner_size as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("reduce: {e}")))?;
                }
                CudaStorage::F64(inp) => {
                    let func = dev.get_func(&format!("reduce_{op_name}_f64"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer_size as u32,
                                reduce_size as u32,
                                inner_size as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("reduce: {e}")))?;
                }
                _ => return Err(Error::msg("reduce: only float types supported")),
            }
            Ok(CudaStorage::I64(out))
        } else {
            // Sum/Mean/Max/Min → output same type
            match &input_c {
                CudaStorage::F16(inp) => {
                    let mut out: CudaSlice<u16> = dev
                        .dev
                        .alloc_zeros(out_n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("reduce_{op_name}_f16"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer_size as u32,
                                reduce_size as u32,
                                inner_size as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("reduce: {e}")))?;
                    Ok(CudaStorage::F16(out))
                }
                CudaStorage::BF16(inp) => {
                    let mut out: CudaSlice<u16> = dev
                        .dev
                        .alloc_zeros(out_n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("reduce_{op_name}_bf16"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer_size as u32,
                                reduce_size as u32,
                                inner_size as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("reduce: {e}")))?;
                    Ok(CudaStorage::BF16(out))
                }
                CudaStorage::F32(inp) => {
                    let mut out: CudaSlice<f32> = dev
                        .dev
                        .alloc_zeros(out_n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("reduce_{op_name}_f32"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer_size as u32,
                                reduce_size as u32,
                                inner_size as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("reduce: {e}")))?;
                    Ok(CudaStorage::F32(out))
                }
                CudaStorage::F64(inp) => {
                    let mut out: CudaSlice<f64> = dev
                        .dev
                        .alloc_zeros(out_n)
                        .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                    let func = dev.get_func(&format!("reduce_{op_name}_f64"))?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer_size as u32,
                                reduce_size as u32,
                                inner_size as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("reduce: {e}")))?;
                    Ok(CudaStorage::F64(out))
                }
                _ => Err(Error::msg("reduce: only float types supported")),
            }
        }
    }

    // ---- Matmul (cuBLAS) ----
    // F16/BF16: promote to F32 → sgemm → demote back

    fn matmul(
        lhs: &CudaStorage,
        lhs_layout: &Layout,
        rhs: &CudaStorage,
        rhs_layout: &Layout,
    ) -> Result<CudaStorage> {
        let dev = dev_from_storage(lhs)?;

        // Make contiguous
        let lhs_c = ensure_contiguous(lhs, lhs_layout, &dev)?;
        let rhs_c = ensure_contiguous(rhs, rhs_layout, &dev)?;

        let lhs_dims = lhs_layout.dims();
        let rhs_dims = rhs_layout.dims();
        let rank = lhs_dims.len();
        let m = lhs_dims[rank - 2];
        let k = lhs_dims[rank - 1];
        let n = rhs_dims[rhs_dims.len() - 1];
        let batch_size: usize = lhs_dims[..rank - 2].iter().product::<usize>().max(1);

        match (&lhs_c, &rhs_c) {
            (CudaStorage::F16(a), CudaStorage::F16(b)) => {
                // Promote F16 → F32, matmul, demote back
                let a_n = a.len();
                let b_n = b.len();
                let mn = m * n;
                let total = batch_size * mn;

                // Cast A to F32
                let mut a_f32: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(a_n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let cast_a = dev.get_func("cast_f16_to_f32")?;
                let cfg_a = launch_cfg(a_n);
                unsafe { cast_a.launch(cfg_a, (a, &mut a_f32, a_n as u32)) }
                    .map_err(|e| Error::msg(format!("cast: {e}")))?;

                // Cast B to F32
                let mut b_f32: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(b_n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let cast_b = dev.get_func("cast_f16_to_f32")?;
                let cfg_b = launch_cfg(b_n);
                unsafe { cast_b.launch(cfg_b, (b, &mut b_f32, b_n as u32)) }
                    .map_err(|e| Error::msg(format!("cast: {e}")))?;

                // sgemm
                let out_f32: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(total)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;

                use cudarc::cublas::sys::cublasOperation_t;
                for batch in 0..batch_size {
                    let a_offset = batch * m * k;
                    let b_offset = batch * k * n;
                    let c_offset = batch * mn;
                    let a_slice = a_f32.slice(a_offset..a_offset + m * k);
                    let b_slice = b_f32.slice(b_offset..b_offset + k * n);
                    let c_slice = out_f32.slice(c_offset..c_offset + mn);
                    unsafe {
                        cudarc::cublas::result::sgemm(
                            *dev.blas.handle(),
                            cublasOperation_t::CUBLAS_OP_N,
                            cublasOperation_t::CUBLAS_OP_N,
                            n as i32,
                            m as i32,
                            k as i32,
                            (&1.0f32) as *const f32,
                            *b_slice.device_ptr() as *const f32,
                            n as i32,
                            *a_slice.device_ptr() as *const f32,
                            k as i32,
                            (&0.0f32) as *const f32,
                            *c_slice.device_ptr() as *mut f32,
                            n as i32,
                        )
                    }
                    .map_err(|e| Error::msg(format!("cuBLAS sgemm: {e}")))?;
                }

                // Demote F32 → F16
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(total)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let cast_out = dev.get_func("cast_f32_to_f16")?;
                let cfg_out = launch_cfg(total);
                unsafe { cast_out.launch(cfg_out, (&out_f32, &mut out, total as u32)) }
                    .map_err(|e| Error::msg(format!("cast: {e}")))?;

                Ok(CudaStorage::F16(out))
            }
            (CudaStorage::BF16(a), CudaStorage::BF16(b)) => {
                // Promote BF16 → F32, matmul, demote back
                let a_n = a.len();
                let b_n = b.len();
                let mn = m * n;
                let total = batch_size * mn;

                let mut a_f32: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(a_n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let cast_a = dev.get_func("cast_bf16_to_f32")?;
                let cfg_a = launch_cfg(a_n);
                unsafe { cast_a.launch(cfg_a, (a, &mut a_f32, a_n as u32)) }
                    .map_err(|e| Error::msg(format!("cast: {e}")))?;

                let mut b_f32: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(b_n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let cast_b = dev.get_func("cast_bf16_to_f32")?;
                let cfg_b = launch_cfg(b_n);
                unsafe { cast_b.launch(cfg_b, (b, &mut b_f32, b_n as u32)) }
                    .map_err(|e| Error::msg(format!("cast: {e}")))?;

                let out_f32: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(total)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;

                use cudarc::cublas::sys::cublasOperation_t;
                for batch in 0..batch_size {
                    let a_offset = batch * m * k;
                    let b_offset = batch * k * n;
                    let c_offset = batch * mn;
                    let a_slice = a_f32.slice(a_offset..a_offset + m * k);
                    let b_slice = b_f32.slice(b_offset..b_offset + k * n);
                    let c_slice = out_f32.slice(c_offset..c_offset + mn);
                    unsafe {
                        cudarc::cublas::result::sgemm(
                            *dev.blas.handle(),
                            cublasOperation_t::CUBLAS_OP_N,
                            cublasOperation_t::CUBLAS_OP_N,
                            n as i32,
                            m as i32,
                            k as i32,
                            (&1.0f32) as *const f32,
                            *b_slice.device_ptr() as *const f32,
                            n as i32,
                            *a_slice.device_ptr() as *const f32,
                            k as i32,
                            (&0.0f32) as *const f32,
                            *c_slice.device_ptr() as *mut f32,
                            n as i32,
                        )
                    }
                    .map_err(|e| Error::msg(format!("cuBLAS sgemm: {e}")))?;
                }

                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(total)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let cast_out = dev.get_func("cast_f32_to_bf16")?;
                let cfg_out = launch_cfg(total);
                unsafe { cast_out.launch(cfg_out, (&out_f32, &mut out, total as u32)) }
                    .map_err(|e| Error::msg(format!("cast: {e}")))?;

                Ok(CudaStorage::BF16(out))
            }
            (CudaStorage::F32(a), CudaStorage::F32(b)) => {
                let mn = m * n;
                let total = batch_size * mn;
                let out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(total)
                    .map_err(|e| Error::msg(format!("alloc matmul: {e}")))?;

                use cudarc::cublas::sys::cublasOperation_t;

                for batch in 0..batch_size {
                    let a_offset = batch * m * k;
                    let b_offset = batch * k * n;
                    let c_offset = batch * mn;

                    let a_slice = a.slice(a_offset..a_offset + m * k);
                    let b_slice = b.slice(b_offset..b_offset + k * n);
                    let c_slice = out.slice(c_offset..c_offset + mn);

                    unsafe {
                        cudarc::cublas::result::sgemm(
                            *dev.blas.handle(),
                            cublasOperation_t::CUBLAS_OP_N,
                            cublasOperation_t::CUBLAS_OP_N,
                            n as i32,
                            m as i32,
                            k as i32,
                            (&1.0f32) as *const f32,
                            *b_slice.device_ptr() as *const f32,
                            n as i32,
                            *a_slice.device_ptr() as *const f32,
                            k as i32,
                            (&0.0f32) as *const f32,
                            *c_slice.device_ptr() as *mut f32,
                            n as i32,
                        )
                    }
                    .map_err(|e| Error::msg(format!("cuBLAS sgemm: {e}")))?;
                }
                Ok(CudaStorage::F32(out))
            }
            (CudaStorage::F64(a), CudaStorage::F64(b)) => {
                let mn = m * n;
                let total = batch_size * mn;
                let out: CudaSlice<f64> = dev
                    .dev
                    .alloc_zeros(total)
                    .map_err(|e| Error::msg(format!("alloc matmul: {e}")))?;

                use cudarc::cublas::sys::cublasOperation_t;

                for batch in 0..batch_size {
                    let a_offset = batch * m * k;
                    let b_offset = batch * k * n;
                    let c_offset = batch * mn;

                    let a_slice = a.slice(a_offset..a_offset + m * k);
                    let b_slice = b.slice(b_offset..b_offset + k * n);
                    let c_slice = out.slice(c_offset..c_offset + mn);

                    unsafe {
                        cudarc::cublas::result::dgemm(
                            *dev.blas.handle(),
                            cublasOperation_t::CUBLAS_OP_N,
                            cublasOperation_t::CUBLAS_OP_N,
                            n as i32,
                            m as i32,
                            k as i32,
                            (&1.0f64) as *const f64,
                            *b_slice.device_ptr() as *const f64,
                            n as i32,
                            *a_slice.device_ptr() as *const f64,
                            k as i32,
                            (&0.0f64) as *const f64,
                            *c_slice.device_ptr() as *mut f64,
                            n as i32,
                        )
                    }
                    .map_err(|e| Error::msg(format!("cuBLAS dgemm: {e}")))?;
                }
                Ok(CudaStorage::F64(out))
            }
            _ => Err(Error::msg("matmul: only f16/bf16/f32/f64 supported")),
        }
    }

    // ---- to_contiguous ----

    fn to_contiguous(input: &CudaStorage, layout: &Layout) -> Result<CudaStorage> {
        let dev = dev_from_storage(input)?;
        ensure_contiguous(input, layout, &dev)
    }

    // ---- to_f64_vec (device → host) ----

    fn to_f64_vec(input: &CudaStorage, layout: &Layout) -> Result<Vec<f64>> {
        let dev = dev_from_storage(input)?;

        // Make contiguous first
        let input_c = ensure_contiguous(input, layout, &dev)?;

        match &input_c {
            CudaStorage::F16(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh f16: {e}")))?;
                Ok(host
                    .iter()
                    .map(|&bits| f16::from_bits(bits).to_f64())
                    .collect())
            }
            CudaStorage::BF16(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh bf16: {e}")))?;
                Ok(host
                    .iter()
                    .map(|&bits| bf16::from_bits(bits).to_f64())
                    .collect())
            }
            CudaStorage::F32(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh f32: {e}")))?;
                Ok(host.iter().map(|&v| v as f64).collect())
            }
            CudaStorage::F64(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh f64: {e}")))?;
                Ok(host)
            }
            CudaStorage::U8(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh u8: {e}")))?;
                Ok(host.iter().map(|&v| v as f64).collect())
            }
            CudaStorage::U32(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh u32: {e}")))?;
                Ok(host.iter().map(|&v| v as f64).collect())
            }
            CudaStorage::I64(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh i64: {e}")))?;
                Ok(host.iter().map(|&v| v as f64).collect())
            }
        }
    }

    // ---- Comparison ops ----

    fn cmp_op(
        op: CmpOp,
        lhs: &CudaStorage,
        lhs_layout: &Layout,
        rhs: &CudaStorage,
        rhs_layout: &Layout,
    ) -> Result<CudaStorage> {
        let dev = dev_from_storage(lhs)?;

        let lhs_c = ensure_contiguous(lhs, lhs_layout, &dev)?;
        let rhs_c = ensure_contiguous(rhs, rhs_layout, &dev)?;
        let n = lhs_layout.elem_count();
        let cfg = launch_cfg(n);

        let op_name = match op {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
        };

        let mut out: CudaSlice<u8> = dev
            .dev
            .alloc_zeros(n)
            .map_err(|e| Error::msg(format!("alloc: {e}")))?;

        match (&lhs_c, &rhs_c) {
            (CudaStorage::F16(a), CudaStorage::F16(b)) => {
                let func = dev.get_func(&format!("cmp_{op_name}_f16"))?;
                unsafe { func.launch(cfg, (a, b, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("cmp: {e}")))?;
            }
            (CudaStorage::BF16(a), CudaStorage::BF16(b)) => {
                let func = dev.get_func(&format!("cmp_{op_name}_bf16"))?;
                unsafe { func.launch(cfg, (a, b, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("cmp: {e}")))?;
            }
            (CudaStorage::F32(a), CudaStorage::F32(b)) => {
                let func = dev.get_func(&format!("cmp_{op_name}_f32"))?;
                unsafe { func.launch(cfg, (a, b, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("cmp: {e}")))?;
            }
            (CudaStorage::F64(a), CudaStorage::F64(b)) => {
                let func = dev.get_func(&format!("cmp_{op_name}_f64"))?;
                unsafe { func.launch(cfg, (a, b, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("cmp: {e}")))?;
            }
            _ => return Err(Error::msg("cmp_op: dtype mismatch or unsupported")),
        }

        Ok(CudaStorage::U8(out))
    }

    // ---- Affine ----

    fn affine(input: &CudaStorage, layout: &Layout, mul: f64, add: f64) -> Result<CudaStorage> {
        let dev = dev_from_storage(input)?;

        let input_c = ensure_contiguous(input, layout, &dev)?;
        let n = layout.elem_count();
        let cfg = launch_cfg(n);

        match &input_c {
            CudaStorage::F16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("affine_f16")?;
                unsafe { func.launch(cfg, (inp, &mut out, mul as f32, add as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("affine: {e}")))?;
                Ok(CudaStorage::F16(out))
            }
            CudaStorage::BF16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("affine_bf16")?;
                unsafe { func.launch(cfg, (inp, &mut out, mul as f32, add as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("affine: {e}")))?;
                Ok(CudaStorage::BF16(out))
            }
            CudaStorage::F32(inp) => {
                let mut out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("affine_f32")?;
                unsafe { func.launch(cfg, (inp, &mut out, mul as f32, add as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("affine: {e}")))?;
                Ok(CudaStorage::F32(out))
            }
            CudaStorage::F64(inp) => {
                let mut out: CudaSlice<f64> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("affine_f64")?;
                unsafe { func.launch(cfg, (inp, &mut out, mul, add, n as u32)) }
                    .map_err(|e| Error::msg(format!("affine: {e}")))?;
                Ok(CudaStorage::F64(out))
            }
            _ => Err(Error::msg("affine: only float types supported")),
        }
    }

    // ---- Index select ----

    fn index_select(
        input: &CudaStorage,
        input_layout: &Layout,
        indices: &CudaStorage,
        indices_layout: &Layout,
        dim: usize,
    ) -> Result<CudaStorage> {
        let dev = dev_from_storage(input)?;

        let input_c = ensure_contiguous(input, input_layout, &dev)?;
        let indices_c = ensure_contiguous(indices, indices_layout, &dev)?;

        let input_dims = input_layout.dims();

        let pre_dim: usize = input_dims[..dim].iter().product::<usize>().max(1);
        let src_dim = input_dims[dim];
        let post_dim: usize = input_dims[dim + 1..].iter().product::<usize>().max(1);
        let idx_len = indices_layout.elem_count();
        let out_n = pre_dim * idx_len * post_dim;
        let cfg = launch_cfg(out_n);

        // We need indices as i64 on device
        let idx_i64 = match &indices_c {
            CudaStorage::I64(s) => s.clone(),
            CudaStorage::U32(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                let host_i64: Vec<i64> = host.iter().map(|&v| v as i64).collect();
                dev.dev
                    .htod_copy(host_i64)
                    .map_err(|e| Error::msg(format!("htod: {e}")))?
            }
            _ => return Err(Error::msg("index_select: indices must be integer type")),
        };

        match &input_c {
            CudaStorage::F16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(out_n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("index_select_f16")?;
                unsafe {
                    func.launch(
                        cfg,
                        (
                            inp,
                            &idx_i64,
                            &mut out,
                            pre_dim as u32,
                            src_dim as u32,
                            post_dim as u32,
                            idx_len as u32,
                            out_n as u32,
                        ),
                    )
                }
                .map_err(|e| Error::msg(format!("index_select: {e}")))?;
                Ok(CudaStorage::F16(out))
            }
            CudaStorage::BF16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(out_n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("index_select_bf16")?;
                unsafe {
                    func.launch(
                        cfg,
                        (
                            inp,
                            &idx_i64,
                            &mut out,
                            pre_dim as u32,
                            src_dim as u32,
                            post_dim as u32,
                            idx_len as u32,
                            out_n as u32,
                        ),
                    )
                }
                .map_err(|e| Error::msg(format!("index_select: {e}")))?;
                Ok(CudaStorage::BF16(out))
            }
            CudaStorage::F32(inp) => {
                let mut out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(out_n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("index_select_f32")?;
                unsafe {
                    func.launch(
                        cfg,
                        (
                            inp,
                            &idx_i64,
                            &mut out,
                            pre_dim as u32,
                            src_dim as u32,
                            post_dim as u32,
                            idx_len as u32,
                            out_n as u32,
                        ),
                    )
                }
                .map_err(|e| Error::msg(format!("index_select: {e}")))?;
                Ok(CudaStorage::F32(out))
            }
            CudaStorage::F64(inp) => {
                let mut out: CudaSlice<f64> = dev
                    .dev
                    .alloc_zeros(out_n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("index_select_f64")?;
                unsafe {
                    func.launch(
                        cfg,
                        (
                            inp,
                            &idx_i64,
                            &mut out,
                            pre_dim as u32,
                            src_dim as u32,
                            post_dim as u32,
                            idx_len as u32,
                            out_n as u32,
                        ),
                    )
                }
                .map_err(|e| Error::msg(format!("index_select: {e}")))?;
                Ok(CudaStorage::F64(out))
            }
            _ => Err(Error::msg("index_select: only float types supported")),
        }
    }

    // ---- Powf ----

    fn powf(input: &CudaStorage, layout: &Layout, exponent: f64) -> Result<CudaStorage> {
        let dev = dev_from_storage(input)?;

        let input_c = ensure_contiguous(input, layout, &dev)?;
        let n = layout.elem_count();
        let cfg = launch_cfg(n);

        match &input_c {
            CudaStorage::F16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("powf_f16")?;
                unsafe { func.launch(cfg, (inp, &mut out, exponent as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("powf: {e}")))?;
                Ok(CudaStorage::F16(out))
            }
            CudaStorage::BF16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("powf_bf16")?;
                unsafe { func.launch(cfg, (inp, &mut out, exponent as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("powf: {e}")))?;
                Ok(CudaStorage::BF16(out))
            }
            CudaStorage::F32(inp) => {
                let mut out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("powf_f32")?;
                unsafe { func.launch(cfg, (inp, &mut out, exponent as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("powf: {e}")))?;
                Ok(CudaStorage::F32(out))
            }
            CudaStorage::F64(inp) => {
                let mut out: CudaSlice<f64> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("powf_f64")?;
                unsafe { func.launch(cfg, (inp, &mut out, exponent, n as u32)) }
                    .map_err(|e| Error::msg(format!("powf: {e}")))?;
                Ok(CudaStorage::F64(out))
            }
            _ => Err(Error::msg("powf: only float types supported")),
        }
    }

    // ---- Clamp ----

    fn clamp(input: &CudaStorage, layout: &Layout, min: f64, max: f64) -> Result<CudaStorage> {
        let dev = dev_from_storage(input)?;

        let input_c = ensure_contiguous(input, layout, &dev)?;
        let n = layout.elem_count();
        let cfg = launch_cfg(n);

        match &input_c {
            CudaStorage::F16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("clamp_f16")?;
                unsafe { func.launch(cfg, (inp, &mut out, min as f32, max as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("clamp: {e}")))?;
                Ok(CudaStorage::F16(out))
            }
            CudaStorage::BF16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("clamp_bf16")?;
                unsafe { func.launch(cfg, (inp, &mut out, min as f32, max as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("clamp: {e}")))?;
                Ok(CudaStorage::BF16(out))
            }
            CudaStorage::F32(inp) => {
                let mut out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("clamp_f32")?;
                unsafe { func.launch(cfg, (inp, &mut out, min as f32, max as f32, n as u32)) }
                    .map_err(|e| Error::msg(format!("clamp: {e}")))?;
                Ok(CudaStorage::F32(out))
            }
            CudaStorage::F64(inp) => {
                let mut out: CudaSlice<f64> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("clamp_f64")?;
                unsafe { func.launch(cfg, (inp, &mut out, min, max, n as u32)) }
                    .map_err(|e| Error::msg(format!("clamp: {e}")))?;
                Ok(CudaStorage::F64(out))
            }
            _ => Err(Error::msg("clamp: only float types supported")),
        }
    }

    // ---- Where / conditional select ----

    fn where_cond(
        mask: &CudaStorage,
        mask_layout: &Layout,
        on_true: &CudaStorage,
        on_true_layout: &Layout,
        on_false: &CudaStorage,
        on_false_layout: &Layout,
    ) -> Result<CudaStorage> {
        let dev = dev_from_storage(mask)?;

        let mask_c = ensure_contiguous(mask, mask_layout, &dev)?;
        let true_c = ensure_contiguous(on_true, on_true_layout, &dev)?;
        let false_c = ensure_contiguous(on_false, on_false_layout, &dev)?;
        let n = mask_layout.elem_count();
        let cfg = launch_cfg(n);

        let mask_u8 = match &mask_c {
            CudaStorage::U8(s) => s,
            _ => return Err(Error::msg("where_cond: mask must be u8")),
        };

        match (&true_c, &false_c) {
            (CudaStorage::F16(t), CudaStorage::F16(f_vals)) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("where_cond_f16")?;
                unsafe { func.launch(cfg, (mask_u8, t, f_vals, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("where_cond: {e}")))?;
                Ok(CudaStorage::F16(out))
            }
            (CudaStorage::BF16(t), CudaStorage::BF16(f_vals)) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("where_cond_bf16")?;
                unsafe { func.launch(cfg, (mask_u8, t, f_vals, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("where_cond: {e}")))?;
                Ok(CudaStorage::BF16(out))
            }
            (CudaStorage::F32(t), CudaStorage::F32(f_vals)) => {
                let mut out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("where_cond_f32")?;
                unsafe { func.launch(cfg, (mask_u8, t, f_vals, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("where_cond: {e}")))?;
                Ok(CudaStorage::F32(out))
            }
            (CudaStorage::F64(t), CudaStorage::F64(f_vals)) => {
                let mut out: CudaSlice<f64> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("where_cond_f64")?;
                unsafe { func.launch(cfg, (mask_u8, t, f_vals, &mut out, n as u32)) }
                    .map_err(|e| Error::msg(format!("where_cond: {e}")))?;
                Ok(CudaStorage::F64(out))
            }
            _ => Err(Error::msg("where_cond: dtype mismatch")),
        }
    }

    // ---- Gather ----

    fn gather(
        input: &CudaStorage,
        input_layout: &Layout,
        index: &CudaStorage,
        index_layout: &Layout,
        dim: usize,
    ) -> Result<CudaStorage> {
        let dev = dev_from_storage(input)?;

        let input_c = ensure_contiguous(input, input_layout, &dev)?;
        let index_c = ensure_contiguous(index, index_layout, &dev)?;

        let input_dims = input_layout.dims();
        let index_dims = index_layout.dims();

        let pre: usize = input_dims[..dim].iter().product::<usize>().max(1);
        let inp_dim = input_dims[dim];
        let idx_dim = index_dims[dim];
        let post: usize = input_dims[dim + 1..].iter().product::<usize>().max(1);
        let n = index_layout.elem_count();
        let cfg = launch_cfg(n);

        // Convert index to i64
        let idx_i64 = match &index_c {
            CudaStorage::I64(s) => s.clone(),
            CudaStorage::U32(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                let host_i64: Vec<i64> = host.iter().map(|&v| v as i64).collect();
                dev.dev
                    .htod_copy(host_i64)
                    .map_err(|e| Error::msg(format!("htod: {e}")))?
            }
            CudaStorage::F32(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                let host_i64: Vec<i64> = host.iter().map(|&v| v as i64).collect();
                dev.dev
                    .htod_copy(host_i64)
                    .map_err(|e| Error::msg(format!("htod: {e}")))?
            }
            CudaStorage::F64(s) => {
                let host = dev
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                let host_i64: Vec<i64> = host.iter().map(|&v| v as i64).collect();
                dev.dev
                    .htod_copy(host_i64)
                    .map_err(|e| Error::msg(format!("htod: {e}")))?
            }
            _ => return Err(Error::msg("gather: unsupported index dtype")),
        };

        match &input_c {
            CudaStorage::F16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("gather_f16")?;
                unsafe {
                    func.launch(
                        cfg,
                        (
                            inp,
                            &idx_i64,
                            &mut out,
                            pre as u32,
                            inp_dim as u32,
                            idx_dim as u32,
                            post as u32,
                            n as u32,
                        ),
                    )
                }
                .map_err(|e| Error::msg(format!("gather: {e}")))?;
                Ok(CudaStorage::F16(out))
            }
            CudaStorage::BF16(inp) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("gather_bf16")?;
                unsafe {
                    func.launch(
                        cfg,
                        (
                            inp,
                            &idx_i64,
                            &mut out,
                            pre as u32,
                            inp_dim as u32,
                            idx_dim as u32,
                            post as u32,
                            n as u32,
                        ),
                    )
                }
                .map_err(|e| Error::msg(format!("gather: {e}")))?;
                Ok(CudaStorage::BF16(out))
            }
            CudaStorage::F32(inp) => {
                let mut out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("gather_f32")?;
                unsafe {
                    func.launch(
                        cfg,
                        (
                            inp,
                            &idx_i64,
                            &mut out,
                            pre as u32,
                            inp_dim as u32,
                            idx_dim as u32,
                            post as u32,
                            n as u32,
                        ),
                    )
                }
                .map_err(|e| Error::msg(format!("gather: {e}")))?;
                Ok(CudaStorage::F32(out))
            }
            CudaStorage::F64(inp) => {
                let mut out: CudaSlice<f64> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("gather_f64")?;
                unsafe {
                    func.launch(
                        cfg,
                        (
                            inp,
                            &idx_i64,
                            &mut out,
                            pre as u32,
                            inp_dim as u32,
                            idx_dim as u32,
                            post as u32,
                            n as u32,
                        ),
                    )
                }
                .map_err(|e| Error::msg(format!("gather: {e}")))?;
                Ok(CudaStorage::F64(out))
            }
            _ => Err(Error::msg("gather: only float types supported")),
        }
    }

    // ---- Concatenation ----

    fn cat(
        inputs: &[(&CudaStorage, &Layout)],
        out_shape: &Shape,
        dim: usize,
    ) -> Result<CudaStorage> {
        if inputs.is_empty() {
            return Err(Error::msg("cat: empty input list"));
        }

        let dev = dev_from_storage(inputs[0].0)?;

        let out_dims = out_shape.dims();
        let out_n = out_shape.elem_count();

        let outer: usize = out_dims[..dim].iter().product::<usize>().max(1);
        let total_dim = out_dims[dim];
        let inner: usize = out_dims[dim + 1..].iter().product::<usize>().max(1);

        match inputs[0].0 {
            CudaStorage::F16(_) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(out_n)
                    .map_err(|e| Error::msg(format!("alloc cat: {e}")))?;
                let mut dim_offset = 0u32;

                for &(storage, layout) in inputs {
                    let storage_c = ensure_contiguous(storage, layout, &dev)?;
                    let inp = match &storage_c {
                        CudaStorage::F16(s) => s,
                        _ => return Err(Error::msg("cat: dtype mismatch")),
                    };
                    let t_dims = layout.dims();
                    let this_dim = t_dims[dim];
                    let src_n = layout.elem_count();
                    let cfg = launch_cfg(src_n);

                    let func = dev.get_func("cat_copy_f16")?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer as u32,
                                this_dim as u32,
                                inner as u32,
                                total_dim as u32,
                                dim_offset,
                                src_n as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("cat_copy: {e}")))?;

                    dim_offset += this_dim as u32;
                }
                Ok(CudaStorage::F16(out))
            }
            CudaStorage::BF16(_) => {
                let mut out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(out_n)
                    .map_err(|e| Error::msg(format!("alloc cat: {e}")))?;
                let mut dim_offset = 0u32;

                for &(storage, layout) in inputs {
                    let storage_c = ensure_contiguous(storage, layout, &dev)?;
                    let inp = match &storage_c {
                        CudaStorage::BF16(s) => s,
                        _ => return Err(Error::msg("cat: dtype mismatch")),
                    };
                    let t_dims = layout.dims();
                    let this_dim = t_dims[dim];
                    let src_n = layout.elem_count();
                    let cfg = launch_cfg(src_n);

                    let func = dev.get_func("cat_copy_bf16")?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer as u32,
                                this_dim as u32,
                                inner as u32,
                                total_dim as u32,
                                dim_offset,
                                src_n as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("cat_copy: {e}")))?;

                    dim_offset += this_dim as u32;
                }
                Ok(CudaStorage::BF16(out))
            }
            CudaStorage::F32(_) => {
                let mut out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(out_n)
                    .map_err(|e| Error::msg(format!("alloc cat: {e}")))?;
                let mut dim_offset = 0u32;

                for &(storage, layout) in inputs {
                    let storage_c = ensure_contiguous(storage, layout, &dev)?;
                    let inp = match &storage_c {
                        CudaStorage::F32(s) => s,
                        _ => return Err(Error::msg("cat: dtype mismatch")),
                    };
                    let t_dims = layout.dims();
                    let this_dim = t_dims[dim];
                    let src_n = layout.elem_count();
                    let cfg = launch_cfg(src_n);

                    let func = dev.get_func("cat_copy_f32")?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer as u32,
                                this_dim as u32,
                                inner as u32,
                                total_dim as u32,
                                dim_offset,
                                src_n as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("cat_copy: {e}")))?;

                    dim_offset += this_dim as u32;
                }
                Ok(CudaStorage::F32(out))
            }
            CudaStorage::F64(_) => {
                let mut out: CudaSlice<f64> = dev
                    .dev
                    .alloc_zeros(out_n)
                    .map_err(|e| Error::msg(format!("alloc cat: {e}")))?;
                let mut dim_offset = 0u32;

                for &(storage, layout) in inputs {
                    let storage_c = ensure_contiguous(storage, layout, &dev)?;
                    let inp = match &storage_c {
                        CudaStorage::F64(s) => s,
                        _ => return Err(Error::msg("cat: dtype mismatch")),
                    };
                    let t_dims = layout.dims();
                    let this_dim = t_dims[dim];
                    let src_n = layout.elem_count();
                    let cfg = launch_cfg(src_n);

                    let func = dev.get_func("cat_copy_f64")?;
                    unsafe {
                        func.launch(
                            cfg,
                            (
                                inp,
                                &mut out,
                                outer as u32,
                                this_dim as u32,
                                inner as u32,
                                total_dim as u32,
                                dim_offset,
                                src_n as u32,
                            ),
                        )
                    }
                    .map_err(|e| Error::msg(format!("cat_copy: {e}")))?;

                    dim_offset += this_dim as u32;
                }
                Ok(CudaStorage::F64(out))
            }
            _ => Err(Error::msg("cat: only float types supported")),
        }
    }

    fn cast(
        input: &CudaStorage,
        layout: &Layout,
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<CudaStorage> {
        let src_dtype = input.dtype();
        if src_dtype == dtype {
            return Ok(input.clone());
        }

        // For F16↔F32 and BF16↔F32 we have dedicated CUDA kernels.
        let dev = dev_from_storage(input)?;
        let contig = ensure_contiguous(input, layout, &dev)?;
        let n = layout.shape().elem_count();
        let cfg = launch_cfg(n);

        match (src_dtype, dtype) {
            (DType::F16, DType::F32) => {
                let src_slice = contig.as_cuda_slice_u16()?;
                let out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("cast_f16_to_f32")?;
                unsafe { func.launch(cfg, (src_slice, &out, n as u32)) }
                    .map_err(|e| Error::msg(format!("launch cast_f16_to_f32: {e}")))?;
                Ok(CudaStorage::F32(out))
            }
            (DType::F32, DType::F16) => {
                let src_slice = contig.as_cuda_slice_f32()?;
                let out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("cast_f32_to_f16")?;
                unsafe { func.launch(cfg, (src_slice, &out, n as u32)) }
                    .map_err(|e| Error::msg(format!("launch cast_f32_to_f16: {e}")))?;
                Ok(CudaStorage::F16(out))
            }
            (DType::BF16, DType::F32) => {
                let src_slice = contig.as_cuda_slice_u16()?;
                let out: CudaSlice<f32> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("cast_bf16_to_f32")?;
                unsafe { func.launch(cfg, (src_slice, &out, n as u32)) }
                    .map_err(|e| Error::msg(format!("launch cast_bf16_to_f32: {e}")))?;
                Ok(CudaStorage::F32(out))
            }
            (DType::F32, DType::BF16) => {
                let src_slice = contig.as_cuda_slice_f32()?;
                let out: CudaSlice<u16> = dev
                    .dev
                    .alloc_zeros(n)
                    .map_err(|e| Error::msg(format!("alloc: {e}")))?;
                let func = dev.get_func("cast_f32_to_bf16")?;
                unsafe { func.launch(cfg, (src_slice, &out, n as u32)) }
                    .map_err(|e| Error::msg(format!("launch cast_f32_to_bf16: {e}")))?;
                Ok(CudaStorage::BF16(out))
            }
            // For F16↔F64, BF16↔F64, F16↔BF16, and integer casts:
            // go through F32 intermediate or fall back to host round-trip
            (DType::F16, DType::F64) | (DType::BF16, DType::F64) => {
                let f32_storage = Self::cast(
                    &contig,
                    &Layout::contiguous(layout.shape().clone()),
                    DType::F32,
                    device,
                )?;
                let f32_layout = Layout::contiguous(layout.shape().clone());
                Self::cast(&f32_storage, &f32_layout, DType::F64, device)
            }
            (DType::F64, DType::F16) | (DType::F64, DType::BF16) => {
                let f32_storage = Self::cast(
                    &contig,
                    &Layout::contiguous(layout.shape().clone()),
                    DType::F32,
                    device,
                )?;
                let f32_layout = Layout::contiguous(layout.shape().clone());
                Self::cast(&f32_storage, &f32_layout, dtype, device)
            }
            (DType::F16, DType::BF16) | (DType::BF16, DType::F16) => {
                let f32_storage = Self::cast(
                    &contig,
                    &Layout::contiguous(layout.shape().clone()),
                    DType::F32,
                    device,
                )?;
                let f32_layout = Layout::contiguous(layout.shape().clone());
                Self::cast(&f32_storage, &f32_layout, dtype, device)
            }
            _ => {
                // Fallback: host round-trip for integer ↔ float, F32↔F64, etc.
                let data = Self::to_f64_vec(&contig, &Layout::contiguous(layout.shape().clone()))?;
                Self::from_f64_slice(&data, dtype, device)
            }
        }
    }
}

// Host ↔ Device transfer helpers

impl CudaStorage {
    /// Get the underlying CudaSlice<f32> (returns error if dtype doesn't match).
    pub fn as_cuda_slice_f32(&self) -> Result<&CudaSlice<f32>> {
        match self {
            CudaStorage::F32(s) => Ok(s),
            _ => Err(Error::msg(format!(
                "expected F32 storage, got {:?}",
                self.dtype()
            ))),
        }
    }

    /// Get the underlying CudaSlice<u16> for F16 or BF16 storage.
    pub fn as_cuda_slice_u16(&self) -> Result<&CudaSlice<u16>> {
        match self {
            CudaStorage::F16(s) | CudaStorage::BF16(s) => Ok(s),
            _ => Err(Error::msg(format!(
                "expected F16/BF16 storage, got {:?}",
                self.dtype()
            ))),
        }
    }

    /// Transfer data from host Vec<f32> to a new CudaStorage on the given device.
    pub fn from_f32_vec(data: Vec<f32>, device: &CudaDevice) -> Result<Self> {
        let s = device
            .dev
            .htod_copy(data)
            .map_err(|e| Error::msg(format!("htod f32: {e}")))?;
        Ok(CudaStorage::F32(s))
    }

    /// Transfer data from host Vec<f64> to a new CudaStorage on the given device.
    pub fn from_f64_vec(data: Vec<f64>, device: &CudaDevice) -> Result<Self> {
        let s = device
            .dev
            .htod_copy(data)
            .map_err(|e| Error::msg(format!("htod f64: {e}")))?;
        Ok(CudaStorage::F64(s))
    }

    /// Transfer data from host Vec<f16> to a new F16 CudaStorage on the given device.
    pub fn from_f16_vec(data: Vec<f16>, device: &CudaDevice) -> Result<Self> {
        let bits: Vec<u16> = data.iter().map(|v| v.to_bits()).collect();
        let s = device
            .dev
            .htod_copy(bits)
            .map_err(|e| Error::msg(format!("htod f16: {e}")))?;
        Ok(CudaStorage::F16(s))
    }

    /// Transfer data from host Vec<bf16> to a new BF16 CudaStorage on the given device.
    pub fn from_bf16_vec(data: Vec<bf16>, device: &CudaDevice) -> Result<Self> {
        let bits: Vec<u16> = data.iter().map(|v| v.to_bits()).collect();
        let s = device
            .dev
            .htod_copy(bits)
            .map_err(|e| Error::msg(format!("htod bf16: {e}")))?;
        Ok(CudaStorage::BF16(s))
    }

    /// Copy all data to host as Vec<f64>.
    pub fn to_host_f64(&self, device: &CudaDevice) -> Result<Vec<f64>> {
        match self {
            CudaStorage::F16(s) => {
                let host = device
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                Ok(host
                    .iter()
                    .map(|&bits| f16::from_bits(bits).to_f64())
                    .collect())
            }
            CudaStorage::BF16(s) => {
                let host = device
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                Ok(host
                    .iter()
                    .map(|&bits| bf16::from_bits(bits).to_f64())
                    .collect())
            }
            CudaStorage::F32(s) => {
                let host = device
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                Ok(host.iter().map(|&v| v as f64).collect())
            }
            CudaStorage::F64(s) => device
                .dev
                .dtoh_sync_copy(s)
                .map_err(|e| Error::msg(format!("dtoh: {e}"))),
            CudaStorage::U8(s) => {
                let host = device
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                Ok(host.iter().map(|&v| v as f64).collect())
            }
            CudaStorage::U32(s) => {
                let host = device
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                Ok(host.iter().map(|&v| v as f64).collect())
            }
            CudaStorage::I64(s) => {
                let host = device
                    .dev
                    .dtoh_sync_copy(s)
                    .map_err(|e| Error::msg(format!("dtoh: {e}")))?;
                Ok(host.iter().map(|&v| v as f64).collect())
            }
        }
    }
}

/// Convenience type alias for CUDA tensors.
pub type CudaTensor = shrew_core::Tensor<CudaBackend>;
