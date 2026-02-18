// CUDA Memory Pool — Caching allocator for GPU buffer reuse
//
// Avoids repeated cudaMalloc/cudaFree round-trips by maintaining per-type,
// per-size free lists of previously allocated CudaSlice buffers.
//
// When a buffer is "returned" to the pool it is not freed to the CUDA driver;
// instead it is cached.  Future allocations of the same element type and count
// will reuse these cached buffers, eliminating the allocation overhead.
//
// This is conceptually similar to PyTorch's CUDA caching allocator.
//
// Usage (through CudaDevice helpers):
//
//   let buf: CudaSlice<f32> = device.pool_alloc::<f32>(1024)?;   // from pool
//   device.pool_reclaim_f32(buf);                                  // return
//   let stats = device.pool_stats();                               // query
//   device.empty_cache();                                          // release

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use cudarc::driver::{CudaSlice, DeviceSlice};

// Pool statistics

/// Snapshot of the pool's allocation statistics.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    /// Total bytes currently held in the cache (not in use by tensors).
    pub cached_bytes: usize,
    /// Number of individual buffers currently in the cache.
    pub cached_buffers: usize,
    /// Cumulative cache hits (allocations served from the cache).
    pub hits: u64,
    /// Cumulative cache misses (allocations that fell through to cudaMalloc).
    pub misses: u64,
}

// Typed free-list bucket

/// A per-type free-list: maps element count → stack of free CudaSlice<T>.
struct TypedPool<T> {
    buckets: Mutex<HashMap<usize, Vec<CudaSlice<T>>>>,
}

impl<T> TypedPool<T> {
    fn new() -> Self {
        TypedPool {
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Try to pop a cached buffer of exactly `n` elements.
    fn try_pop(&self, n: usize) -> Option<CudaSlice<T>> {
        let mut map = self.buckets.lock().unwrap();
        if let Some(stack) = map.get_mut(&n) {
            stack.pop()
        } else {
            None
        }
    }

    /// Push a buffer back into the cache.
    fn push(&self, slice: CudaSlice<T>)
    where
        CudaSlice<T>: DeviceSlice<T>,
    {
        let n = slice.len();
        let mut map = self.buckets.lock().unwrap();
        map.entry(n).or_default().push(slice);
    }

    /// Drain all cached buffers, returning the count and total elements freed.
    fn drain(&self) -> (usize, usize) {
        let mut map = self.buckets.lock().unwrap();
        let mut count = 0usize;
        let mut elems = 0usize;
        for (n, stack) in map.drain() {
            count += stack.len();
            elems += n * stack.len();
        }
        (count, elems)
    }

    /// Count of cached buffers and total cached elements.
    fn stats(&self) -> (usize, usize) {
        let map = self.buckets.lock().unwrap();
        let mut count = 0usize;
        let mut elems = 0usize;
        for (n, stack) in map.iter() {
            count += stack.len();
            elems += *n * stack.len();
        }
        (count, elems)
    }
}

// CudaMemPool

/// A CUDA memory caching allocator.
///
/// Maintains per-dtype free lists keyed by element count. Reuses buffers
/// when possible, falling back to `cudaMalloc` on cache miss.
pub struct CudaMemPool {
    pool_u8: TypedPool<u8>,
    pool_u16: TypedPool<u16>,
    pool_u32: TypedPool<u32>,
    pool_f32: TypedPool<f32>,
    pool_f64: TypedPool<f64>,
    pool_i64: TypedPool<i64>,

    // Atomic counters — no lock contention on the hot path
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CudaMemPool {
    /// Create a new empty memory pool.
    pub fn new() -> Self {
        CudaMemPool {
            pool_u8: TypedPool::new(),
            pool_u16: TypedPool::new(),
            pool_u32: TypedPool::new(),
            pool_f32: TypedPool::new(),
            pool_f64: TypedPool::new(),
            pool_i64: TypedPool::new(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    // Allocation helpers (per type)

    /// Allocate `n` elements of type `f32`, reusing a cached buffer if available.
    /// The returned buffer content is **undefined** (not zeroed).
    pub fn alloc_f32(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<f32>, cudarc::driver::DriverError> {
        if let Some(buf) = self.pool_f32.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            unsafe { dev.alloc::<f32>(n) }
        }
    }

    /// Allocate `n` elements of `f32` and zero them.
    pub fn alloc_zeros_f32(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<f32>, cudarc::driver::DriverError> {
        if let Some(mut buf) = self.pool_f32.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            dev.memset_zeros(&mut buf)?;
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            dev.alloc_zeros::<f32>(n)
        }
    }

    /// Allocate `n` elements of type `f64`.
    pub fn alloc_f64(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<f64>, cudarc::driver::DriverError> {
        if let Some(buf) = self.pool_f64.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            unsafe { dev.alloc::<f64>(n) }
        }
    }

    pub fn alloc_zeros_f64(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<f64>, cudarc::driver::DriverError> {
        if let Some(mut buf) = self.pool_f64.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            dev.memset_zeros(&mut buf)?;
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            dev.alloc_zeros::<f64>(n)
        }
    }

    /// Allocate `n` elements of type `u16` (used for F16/BF16 storage).
    pub fn alloc_u16(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<u16>, cudarc::driver::DriverError> {
        if let Some(buf) = self.pool_u16.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            unsafe { dev.alloc::<u16>(n) }
        }
    }

    pub fn alloc_zeros_u16(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<u16>, cudarc::driver::DriverError> {
        if let Some(mut buf) = self.pool_u16.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            dev.memset_zeros(&mut buf)?;
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            dev.alloc_zeros::<u16>(n)
        }
    }

    /// Allocate `n` elements of type `u8`.
    pub fn alloc_u8(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<u8>, cudarc::driver::DriverError> {
        if let Some(buf) = self.pool_u8.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            unsafe { dev.alloc::<u8>(n) }
        }
    }

    pub fn alloc_zeros_u8(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<u8>, cudarc::driver::DriverError> {
        if let Some(mut buf) = self.pool_u8.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            dev.memset_zeros(&mut buf)?;
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            dev.alloc_zeros::<u8>(n)
        }
    }

    /// Allocate `n` elements of type `u32`.
    pub fn alloc_u32(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<u32>, cudarc::driver::DriverError> {
        if let Some(buf) = self.pool_u32.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            unsafe { dev.alloc::<u32>(n) }
        }
    }

    pub fn alloc_zeros_u32(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<u32>, cudarc::driver::DriverError> {
        if let Some(mut buf) = self.pool_u32.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            dev.memset_zeros(&mut buf)?;
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            dev.alloc_zeros::<u32>(n)
        }
    }

    /// Allocate `n` elements of type `i64`.
    pub fn alloc_i64(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<i64>, cudarc::driver::DriverError> {
        if let Some(buf) = self.pool_i64.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            unsafe { dev.alloc::<i64>(n) }
        }
    }

    pub fn alloc_zeros_i64(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        n: usize,
    ) -> std::result::Result<CudaSlice<i64>, cudarc::driver::DriverError> {
        if let Some(mut buf) = self.pool_i64.try_pop(n) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            dev.memset_zeros(&mut buf)?;
            Ok(buf)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            dev.alloc_zeros::<i64>(n)
        }
    }

    // Reclaim (return buffer to pool)

    pub fn reclaim_f32(&self, s: CudaSlice<f32>) {
        self.pool_f32.push(s);
    }
    pub fn reclaim_f64(&self, s: CudaSlice<f64>) {
        self.pool_f64.push(s);
    }
    pub fn reclaim_u16(&self, s: CudaSlice<u16>) {
        self.pool_u16.push(s);
    }
    pub fn reclaim_u8(&self, s: CudaSlice<u8>) {
        self.pool_u8.push(s);
    }
    pub fn reclaim_u32(&self, s: CudaSlice<u32>) {
        self.pool_u32.push(s);
    }
    pub fn reclaim_i64(&self, s: CudaSlice<i64>) {
        self.pool_i64.push(s);
    }

    /// Reclaim all buffers inside a `CudaStorage`, returning them to the pool.
    pub fn reclaim_storage(&self, storage: super::CudaStorage) {
        match storage {
            super::CudaStorage::F16(s) => self.pool_u16.push(s),
            super::CudaStorage::BF16(s) => self.pool_u16.push(s),
            super::CudaStorage::F32(s) => self.pool_f32.push(s),
            super::CudaStorage::F64(s) => self.pool_f64.push(s),
            super::CudaStorage::U8(s) => self.pool_u8.push(s),
            super::CudaStorage::U32(s) => self.pool_u32.push(s),
            super::CudaStorage::I64(s) => self.pool_i64.push(s),
        }
    }

    // Cache management

    /// Release all cached buffers back to the CUDA driver.
    /// This actually frees GPU memory.
    pub fn empty_cache(&self) {
        self.pool_u8.drain();
        self.pool_u16.drain();
        self.pool_u32.drain();
        self.pool_f32.drain();
        self.pool_f64.drain();
        self.pool_i64.drain();
    }

    /// Return a snapshot of pool statistics.
    pub fn stats(&self) -> PoolStats {
        let (c_u8, e_u8) = self.pool_u8.stats();
        let (c_u16, e_u16) = self.pool_u16.stats();
        let (c_u32, e_u32) = self.pool_u32.stats();
        let (c_f32, e_f32) = self.pool_f32.stats();
        let (c_f64, e_f64) = self.pool_f64.stats();
        let (c_i64, e_i64) = self.pool_i64.stats();

        let cached_buffers = c_u8 + c_u16 + c_u32 + c_f32 + c_f64 + c_i64;
        let cached_bytes = e_u8 * std::mem::size_of::<u8>()
            + e_u16 * std::mem::size_of::<u16>()
            + e_u32 * std::mem::size_of::<u32>()
            + e_f32 * std::mem::size_of::<f32>()
            + e_f64 * std::mem::size_of::<f64>()
            + e_i64 * std::mem::size_of::<i64>();

        PoolStats {
            cached_bytes,
            cached_buffers,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }

    /// Reset hit/miss counters.
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

impl Default for CudaMemPool {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: All interior mutability is through Mutex + Atomics.
unsafe impl Send for CudaMemPool {}
unsafe impl Sync for CudaMemPool {}
