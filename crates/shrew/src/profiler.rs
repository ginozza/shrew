// =============================================================================
// Profiling & Benchmarking — Op-level timing, memory tracking, model benchmarks
// =============================================================================

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use shrew_core::{Backend, Result, Tensor};
use shrew_nn::Module;

// ---------------------------------------------------------------------------
// ProfileEvent — a single recorded timing event
// ---------------------------------------------------------------------------

/// A single profiling event with a name, duration, and optional metadata.
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    /// Name / label for this event.
    pub name: String,
    /// Category (e.g. "forward", "backward", "data", "optimizer").
    pub category: String,
    /// Wall-clock duration.
    pub duration: Duration,
}

// ---------------------------------------------------------------------------
// Profiler — collects named timing events
// ---------------------------------------------------------------------------

/// A lightweight profiler that collects named timing events.
///
/// # Example
/// ```
/// use shrew::profiler::Profiler;
///
/// let mut prof = Profiler::new();
/// let t = prof.start_event("forward", "compute");
/// // ... do work ...
/// prof.end_event(t, "forward", "compute");
/// let report = prof.report();
/// println!("{}", report);
/// ```
pub struct Profiler {
    events: Vec<ProfileEvent>,
    pending: HashMap<String, Instant>,
}

impl Profiler {
    /// Create a new empty profiler.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            pending: HashMap::new(),
        }
    }

    /// Mark the start of a named event. Returns the [`Instant`].
    pub fn start_event(&mut self, name: &str, _category: &str) -> Instant {
        let now = Instant::now();
        self.pending.insert(name.to_string(), now);
        now
    }

    /// End an event started with [`start_event`]. Records elapsed time.
    pub fn end_event(&mut self, start: Instant, name: &str, category: &str) {
        let elapsed = start.elapsed();
        self.pending.remove(name);
        self.events.push(ProfileEvent {
            name: name.to_string(),
            category: category.to_string(),
            duration: elapsed,
        });
    }

    /// Measure a closure and record it as a named event.
    pub fn measure<F, R>(&mut self, name: &str, category: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        self.events.push(ProfileEvent {
            name: name.to_string(),
            category: category.to_string(),
            duration: elapsed,
        });
        result
    }

    /// Return all recorded events.
    pub fn events(&self) -> &[ProfileEvent] {
        &self.events
    }

    /// Reset / clear all recorded events.
    pub fn clear(&mut self) {
        self.events.clear();
        self.pending.clear();
    }

    /// Total wall-clock time across all events.
    pub fn total_time(&self) -> Duration {
        self.events.iter().map(|e| e.duration).sum()
    }

    /// Generate a human-readable [`ProfileReport`].
    pub fn report(&self) -> ProfileReport {
        // Aggregate by name
        let mut by_name: HashMap<String, Vec<Duration>> = HashMap::new();
        for ev in &self.events {
            by_name
                .entry(ev.name.clone())
                .or_default()
                .push(ev.duration);
        }

        let mut entries: Vec<ProfileEntry> = by_name
            .into_iter()
            .map(|(name, durations)| {
                let count = durations.len();
                let total: Duration = durations.iter().sum();
                let min = durations.iter().min().copied().unwrap_or_default();
                let max = durations.iter().max().copied().unwrap_or_default();
                let avg = total / count as u32;
                ProfileEntry {
                    name,
                    count,
                    total,
                    min,
                    max,
                    avg,
                }
            })
            .collect();

        // Sort by total time descending
        entries.sort_by(|a, b| b.total.cmp(&a.total));

        let total = self.total_time();

        ProfileReport { entries, total }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ProfileEntry — aggregated stats for one event name
// ---------------------------------------------------------------------------

/// Aggregated statistics for a single event name.
#[derive(Debug, Clone)]
pub struct ProfileEntry {
    pub name: String,
    pub count: usize,
    pub total: Duration,
    pub min: Duration,
    pub max: Duration,
    pub avg: Duration,
}

// ---------------------------------------------------------------------------
// ProfileReport — pretty-printable summary
// ---------------------------------------------------------------------------

/// A formatted profiling report, printed with `Display`.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    pub entries: Vec<ProfileEntry>,
    pub total: Duration,
}

impl fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            f,
            "║                          Shrew Profile Report                               ║"
        )?;
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ {:<20} {:>6} {:>12} {:>12} {:>12} {:>8} ║",
            "Event", "Count", "Total", "Avg", "Min/Max", "%"
        )?;
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        )?;
        for entry in &self.entries {
            let pct = if self.total.as_nanos() > 0 {
                (entry.total.as_nanos() as f64 / self.total.as_nanos() as f64) * 100.0
            } else {
                0.0
            };
            let min_max = format!("{:.2?}/{:.2?}", entry.min, entry.max);
            writeln!(
                f,
                "║ {:<20} {:>6} {:>12.2?} {:>12.2?} {:>12} {:>7.1}% ║",
                truncate_str(&entry.name, 20),
                entry.count,
                entry.total,
                entry.avg,
                min_max,
                pct
            )?;
        }
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ Total: {:>12.2?}                                                        ║",
            self.total
        )?;
        writeln!(
            f,
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        )?;
        Ok(())
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}…", &s[..max - 1])
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// MemoryTracker — track tensor allocations and peak memory
// ---------------------------------------------------------------------------

/// Tracks tensor memory allocations for profiling.
///
/// Call [`alloc`] when a tensor is created and [`dealloc`] when freed.
///
/// # Example
/// ```
/// use shrew::profiler::MemoryTracker;
///
/// let mut mem = MemoryTracker::new();
/// mem.alloc("weight", 1024 * 4);
/// mem.alloc("bias", 128 * 4);
/// assert_eq!(mem.current_bytes(), 1024 * 4 + 128 * 4);
/// mem.dealloc("bias");
/// assert_eq!(mem.current_bytes(), 1024 * 4);
/// ```
pub struct MemoryTracker {
    allocations: HashMap<String, usize>,
    current_bytes: usize,
    peak_bytes: usize,
    total_allocated: usize,
    alloc_count: usize,
    dealloc_count: usize,
}

impl MemoryTracker {
    /// Create a new tracker with zero allocations.
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            current_bytes: 0,
            peak_bytes: 0,
            total_allocated: 0,
            alloc_count: 0,
            dealloc_count: 0,
        }
    }

    /// Record a tensor allocation.
    pub fn alloc(&mut self, name: &str, bytes: usize) {
        // If re-allocating same name, free old first
        if let Some(old) = self.allocations.remove(name) {
            self.current_bytes = self.current_bytes.saturating_sub(old);
        }
        self.allocations.insert(name.to_string(), bytes);
        self.current_bytes += bytes;
        self.total_allocated += bytes;
        self.alloc_count += 1;
        if self.current_bytes > self.peak_bytes {
            self.peak_bytes = self.current_bytes;
        }
    }

    /// Record a tensor deallocation.
    pub fn dealloc(&mut self, name: &str) {
        if let Some(bytes) = self.allocations.remove(name) {
            self.current_bytes = self.current_bytes.saturating_sub(bytes);
            self.dealloc_count += 1;
        }
    }

    /// Current live memory in bytes.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Peak memory usage in bytes.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Total bytes allocated over the tracker's lifetime.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Number of allocations recorded.
    pub fn alloc_count(&self) -> usize {
        self.alloc_count
    }

    /// Number of deallocations recorded.
    pub fn dealloc_count(&self) -> usize {
        self.dealloc_count
    }

    /// Number of currently live allocations.
    pub fn live_count(&self) -> usize {
        self.allocations.len()
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.current_bytes = 0;
        self.peak_bytes = 0;
        self.total_allocated = 0;
        self.alloc_count = 0;
        self.dealloc_count = 0;
    }

    /// Format a human-readable memory summary.
    pub fn summary(&self) -> String {
        format!(
            "Memory: current={}, peak={}, total_allocated={}, allocs={}, deallocs={}, live={}",
            format_bytes(self.current_bytes),
            format_bytes(self.peak_bytes),
            format_bytes(self.total_allocated),
            self.alloc_count,
            self.dealloc_count,
            self.live_count(),
        )
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Format bytes into a human-readable string (B, KB, MB, GB).
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ---------------------------------------------------------------------------
// ModelSummary — compute model stats (params, layers, etc.)
// ---------------------------------------------------------------------------

/// Summary statistics for a model.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// Total number of trainable parameters.
    pub total_params: usize,
    /// Number of named parameter tensors.
    pub num_tensors: usize,
    /// Parameter count per named parameter.
    pub param_details: Vec<(String, Vec<usize>, usize)>,
    /// Estimated memory in bytes (assuming F32).
    pub estimated_bytes: usize,
}

impl ModelSummary {
    /// Compute a summary for the given module.
    pub fn from_module<B: Backend>(module: &dyn Module<B>) -> Self {
        let named = module.named_parameters();
        let mut total_params = 0usize;
        let mut param_details = Vec::new();

        for (name, tensor) in &named {
            let numel = tensor.shape().elem_count();
            total_params += numel;
            param_details.push((name.clone(), tensor.shape().dims().to_vec(), numel));
        }

        let estimated_bytes = total_params * 4; // assume F32

        ModelSummary {
            total_params,
            num_tensors: named.len(),
            param_details,
            estimated_bytes,
        }
    }
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "┌──────────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│                   Model Summary                      │"
        )?;
        writeln!(
            f,
            "├──────────────────────────────────────────────────────┤"
        )?;
        for (name, shape, numel) in &self.param_details {
            writeln!(
                f,
                "│ {:<30} {:>10?} {:>8} │",
                truncate_str(name, 30),
                shape,
                numel
            )?;
        }
        writeln!(
            f,
            "├──────────────────────────────────────────────────────┤"
        )?;
        writeln!(
            f,
            "│ Total params: {:<12} Tensors: {:<6} Mem: {:<8} │",
            format_params(self.total_params),
            self.num_tensors,
            format_bytes(self.estimated_bytes),
        )?;
        writeln!(
            f,
            "└──────────────────────────────────────────────────────┘"
        )?;
        Ok(())
    }
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

// ---------------------------------------------------------------------------
// Benchmark — time a model's forward pass over multiple iterations
// ---------------------------------------------------------------------------

/// Result of benchmarking a model's forward pass.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Number of iterations run.
    pub iterations: usize,
    /// Batch size used.
    pub batch_size: usize,
    /// Total wall time for all iterations.
    pub total_time: Duration,
    /// Average time per iteration.
    pub avg_time: Duration,
    /// Minimum time across iterations.
    pub min_time: Duration,
    /// Maximum time across iterations.
    pub max_time: Duration,
    /// Throughput in samples per second.
    pub throughput: f64,
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Benchmark: {} iters, batch_size={}",
            self.iterations, self.batch_size
        )?;
        writeln!(f, "  Total:      {:.2?}", self.total_time)?;
        writeln!(f, "  Avg/iter:   {:.2?}", self.avg_time)?;
        writeln!(
            f,
            "  Min/Max:    {:.2?} / {:.2?}",
            self.min_time, self.max_time
        )?;
        writeln!(f, "  Throughput: {:.1} samples/sec", self.throughput)?;
        Ok(())
    }
}

/// Benchmark a model's forward pass.
///
/// Runs `warmup` untimed iterations, then `iterations` timed iterations,
/// calling `input_fn` on each iteration to produce the input tensor.
///
/// # Example
/// ```no_run
/// use shrew::prelude::*;
/// use shrew::profiler::benchmark_forward;
///
/// let model = Linear::<CpuBackend>::new(16, 8, true, DType::F32, &CpuDevice).unwrap();
/// let result = benchmark_forward(
///     &model,
///     || Tensor::<CpuBackend>::rand((4, 16), DType::F32, &CpuDevice).unwrap(),
///     4, // batch_size (for throughput calc)
///     3, // warmup
///     10, // iterations
/// ).unwrap();
/// println!("{}", result);
/// ```
pub fn benchmark_forward<B, M, F>(
    model: &M,
    input_fn: F,
    batch_size: usize,
    warmup: usize,
    iterations: usize,
) -> Result<BenchmarkResult>
where
    B: Backend,
    M: Module<B>,
    F: Fn() -> Tensor<B>,
{
    // Warmup
    for _ in 0..warmup {
        let input = input_fn();
        let _ = model.forward(&input)?;
    }

    let mut times = Vec::with_capacity(iterations);
    let total_start = Instant::now();

    for _ in 0..iterations {
        let input = input_fn();
        let start = Instant::now();
        let _ = model.forward(&input)?;
        times.push(start.elapsed());
    }

    let total_time = total_start.elapsed();
    let min_time = times.iter().min().copied().unwrap_or_default();
    let max_time = times.iter().max().copied().unwrap_or_default();
    let avg_time = if iterations > 0 {
        total_time / iterations as u32
    } else {
        Duration::ZERO
    };
    let throughput = if total_time.as_secs_f64() > 0.0 {
        (iterations * batch_size) as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        iterations,
        batch_size,
        total_time,
        avg_time,
        min_time,
        max_time,
        throughput,
    })
}

/// Benchmark forward + backward pass.
///
/// Same as [`benchmark_forward`] but also runs backpropagation on each iteration.
pub fn benchmark_forward_backward<B, M, F, L>(
    model: &M,
    input_fn: F,
    loss_fn: L,
    batch_size: usize,
    warmup: usize,
    iterations: usize,
) -> Result<BenchmarkResult>
where
    B: Backend,
    M: Module<B>,
    F: Fn() -> Tensor<B>,
    L: Fn(&Tensor<B>) -> Result<Tensor<B>>,
{
    // Warmup
    for _ in 0..warmup {
        let input = input_fn();
        let out = model.forward(&input)?;
        let loss = loss_fn(&out)?;
        let _ = loss.backward()?;
    }

    let mut times = Vec::with_capacity(iterations);
    let total_start = Instant::now();

    for _ in 0..iterations {
        let input = input_fn();
        let start = Instant::now();
        let out = model.forward(&input)?;
        let loss = loss_fn(&out)?;
        let _ = loss.backward()?;
        times.push(start.elapsed());
    }

    let total_time = total_start.elapsed();
    let min_time = times.iter().min().copied().unwrap_or_default();
    let max_time = times.iter().max().copied().unwrap_or_default();
    let avg_time = if iterations > 0 {
        total_time / iterations as u32
    } else {
        Duration::ZERO
    };
    let throughput = if total_time.as_secs_f64() > 0.0 {
        (iterations * batch_size) as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        iterations,
        batch_size,
        total_time,
        avg_time,
        min_time,
        max_time,
        throughput,
    })
}

// ---------------------------------------------------------------------------
// ScopedTimer — RAII timer that records to a profiler on drop
// ---------------------------------------------------------------------------

/// An RAII timer guard. Drops into a profiler automatically.
///
/// # Example
/// ```
/// use shrew::profiler::{Profiler, ScopedTimer};
/// use std::sync::{Arc, Mutex};
///
/// let profiler = Arc::new(Mutex::new(Profiler::new()));
/// {
///     let _t = ScopedTimer::new(profiler.clone(), "my_op", "compute");
///     // ... do work ...
/// } // timer records elapsed time on drop
/// let prof = profiler.lock().unwrap();
/// assert_eq!(prof.events().len(), 1);
/// ```
pub struct ScopedTimer {
    profiler: std::sync::Arc<std::sync::Mutex<Profiler>>,
    name: String,
    category: String,
    start: Instant,
}

impl ScopedTimer {
    pub fn new(
        profiler: std::sync::Arc<std::sync::Mutex<Profiler>>,
        name: &str,
        category: &str,
    ) -> Self {
        Self {
            profiler,
            name: name.to_string(),
            category: category.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if let Ok(mut prof) = self.profiler.lock() {
            prof.events.push(ProfileEvent {
                name: self.name.clone(),
                category: self.category.clone(),
                duration: elapsed,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// estimate_model_memory — compute memory usage of a module's parameters
// ---------------------------------------------------------------------------

/// Estimate the memory usage of a module's parameters in bytes.
///
/// Accounts for the actual dtype of each parameter tensor.
pub fn estimate_model_memory<B: Backend>(module: &dyn Module<B>) -> usize {
    module
        .named_parameters()
        .iter()
        .map(|(_, t)| {
            let numel = t.shape().elem_count();
            let bytes_per_elem = match t.dtype() {
                shrew_core::DType::F16 | shrew_core::DType::BF16 => 2,
                shrew_core::DType::F32 | shrew_core::DType::U32 => 4,
                shrew_core::DType::F64 | shrew_core::DType::I64 => 8,
                shrew_core::DType::U8 => 1,
            };
            numel * bytes_per_elem
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Stopwatch — simple reusable timer
// ---------------------------------------------------------------------------

/// A simple stopwatch for manual timing.
///
/// # Example
/// ```
/// use shrew::profiler::Stopwatch;
///
/// let mut sw = Stopwatch::new();
/// sw.start();
/// // ... do work ...
/// let elapsed = sw.stop();
/// assert!(elapsed.as_nanos() > 0);
/// sw.start();
/// let lap = sw.lap();
/// let elapsed2 = sw.stop();
/// ```
pub struct Stopwatch {
    start: Option<Instant>,
    laps: Vec<Duration>,
}

impl Stopwatch {
    pub fn new() -> Self {
        Self {
            start: None,
            laps: Vec::new(),
        }
    }

    /// Start (or restart) the stopwatch.
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
        self.laps.clear();
    }

    /// Record a lap split without stopping.
    pub fn lap(&mut self) -> Duration {
        let elapsed = self.start.map(|s| s.elapsed()).unwrap_or_default();
        self.laps.push(elapsed);
        elapsed
    }

    /// Stop and return total elapsed time.
    pub fn stop(&mut self) -> Duration {
        let elapsed = self.start.map(|s| s.elapsed()).unwrap_or_default();
        self.start = None;
        elapsed
    }

    /// Get all recorded laps.
    pub fn laps(&self) -> &[Duration] {
        &self.laps
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use shrew_cpu::{CpuBackend, CpuDevice};
    use shrew_nn::{Linear, Module};
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn test_profiler_measure() {
        let mut prof = Profiler::new();
        let result = prof.measure("op_a", "compute", || {
            thread::sleep(Duration::from_millis(5));
            42
        });
        assert_eq!(result, 42);
        assert_eq!(prof.events().len(), 1);
        assert_eq!(prof.events()[0].name, "op_a");
        assert!(prof.events()[0].duration >= Duration::from_millis(4));
    }

    #[test]
    fn test_profiler_start_end() {
        let mut prof = Profiler::new();
        let start = prof.start_event("forward", "model");
        thread::sleep(Duration::from_millis(5));
        prof.end_event(start, "forward", "model");

        assert_eq!(prof.events().len(), 1);
        assert!(prof.total_time() >= Duration::from_millis(4));
    }

    #[test]
    fn test_profiler_report() {
        let mut prof = Profiler::new();
        for _ in 0..3 {
            prof.measure("matmul", "compute", || {
                thread::sleep(Duration::from_millis(2));
            });
        }
        prof.measure("relu", "compute", || {
            thread::sleep(Duration::from_millis(1));
        });

        let report = prof.report();
        assert_eq!(report.entries.len(), 2);
        // matmul should be first (more total time)
        assert_eq!(report.entries[0].name, "matmul");
        assert_eq!(report.entries[0].count, 3);
        assert_eq!(report.entries[1].name, "relu");
        assert_eq!(report.entries[1].count, 1);

        // Test Display works
        let s = format!("{}", report);
        assert!(s.contains("matmul"));
        assert!(s.contains("relu"));
    }

    #[test]
    fn test_profiler_clear() {
        let mut prof = Profiler::new();
        prof.measure("x", "y", || {});
        assert_eq!(prof.events().len(), 1);
        prof.clear();
        assert_eq!(prof.events().len(), 0);
    }

    #[test]
    fn test_memory_tracker() {
        let mut mem = MemoryTracker::new();
        mem.alloc("weight", 4096);
        mem.alloc("bias", 128);
        assert_eq!(mem.current_bytes(), 4224);
        assert_eq!(mem.peak_bytes(), 4224);
        assert_eq!(mem.alloc_count(), 2);
        assert_eq!(mem.live_count(), 2);

        mem.dealloc("bias");
        assert_eq!(mem.current_bytes(), 4096);
        assert_eq!(mem.peak_bytes(), 4224); // peak unchanged
        assert_eq!(mem.dealloc_count(), 1);
        assert_eq!(mem.live_count(), 1);

        // Re-alloc same name replaces
        mem.alloc("weight", 8192);
        assert_eq!(mem.current_bytes(), 8192);
        assert!(mem.peak_bytes() >= 8192);
    }

    #[test]
    fn test_memory_tracker_summary() {
        let mut mem = MemoryTracker::new();
        mem.alloc("big", 1024 * 1024);
        let s = mem.summary();
        assert!(s.contains("1.00 MB"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_model_summary() {
        let model = Linear::new(16, 8, true, shrew_core::DType::F32, &CpuDevice).unwrap();
        let summary = ModelSummary::from_module::<CpuBackend>(&model);
        // weight: 16*8=128, bias: 1*8=8
        assert_eq!(summary.total_params, 128 + 8);
        assert_eq!(summary.num_tensors, 2);
        let s = format!("{}", summary);
        assert!(s.contains("Model Summary"));
    }

    #[test]
    fn test_benchmark_forward() {
        let model =
            Linear::<CpuBackend>::new(8, 4, true, shrew_core::DType::F32, &CpuDevice).unwrap();
        let result = benchmark_forward(
            &model,
            || Tensor::rand((2, 8), shrew_core::DType::F32, &CpuDevice).unwrap(),
            2,
            1,
            5,
        )
        .unwrap();
        assert_eq!(result.iterations, 5);
        assert_eq!(result.batch_size, 2);
        assert!(result.throughput > 0.0);
        assert!(result.min_time <= result.max_time);
        let s = format!("{}", result);
        assert!(s.contains("Benchmark"));
    }

    #[test]
    fn test_benchmark_forward_backward() {
        let model =
            Linear::<CpuBackend>::new(8, 4, true, shrew_core::DType::F32, &CpuDevice).unwrap();
        let result = benchmark_forward_backward(
            &model,
            || Tensor::rand((2, 8), shrew_core::DType::F32, &CpuDevice).unwrap(),
            |out| out.mean_all(),
            2,
            1,
            3,
        )
        .unwrap();
        assert_eq!(result.iterations, 3);
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_scoped_timer() {
        let profiler = Arc::new(Mutex::new(Profiler::new()));
        {
            let _t = ScopedTimer::new(profiler.clone(), "scoped_op", "test");
            thread::sleep(Duration::from_millis(3));
        }
        let prof = profiler.lock().unwrap();
        assert_eq!(prof.events().len(), 1);
        assert_eq!(prof.events()[0].name, "scoped_op");
        assert!(prof.events()[0].duration >= Duration::from_millis(2));
    }

    #[test]
    fn test_estimate_model_memory() {
        let model = Linear::new(16, 8, true, shrew_core::DType::F32, &CpuDevice).unwrap();
        let bytes = estimate_model_memory::<CpuBackend>(&model);
        // weight: 16*8*4=512 bytes, bias: 1*8*4=32 bytes
        assert_eq!(bytes, 544);
    }

    #[test]
    fn test_stopwatch() {
        let mut sw = Stopwatch::new();
        sw.start();
        thread::sleep(Duration::from_millis(5));
        let lap = sw.lap();
        assert!(lap >= Duration::from_millis(4));
        let total = sw.stop();
        assert!(total >= lap);
        assert_eq!(sw.laps().len(), 1);
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1_500), "1.50K");
        assert_eq!(format_params(1_500_000), "1.50M");
        assert_eq!(format_params(2_500_000_000), "2.50B");
    }
}
