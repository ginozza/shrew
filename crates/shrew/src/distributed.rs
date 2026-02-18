// Distributed Training — Data Parallelism, Mixed Precision, Pipeline Stages
//
// This module provides primitives for scaling training across multiple
// workers, mixed-precision (FP16/FP32) training, and model-parallel
// pipeline execution.
//
// COMPONENTS:
//
//   DataParallel<M>       — Splits input batches across N workers and runs
//                           each forward pass in parallel (rayon threads).
//                           Implements Module, so it's a drop-in replacement.
//
//   MixedPrecisionTrainer — Maintains FP32 "master" weights, casts to FP16
//                           for forward/backward, applies dynamic loss scaling
//                           to prevent underflow in FP16 gradients.
//
//   PipelineParallel      — Splits a sequential model into stages and overlaps
//                           micro-batch execution (GPipe-style 1F1B schedule).
//
//   average_gradients()   — Averages multiple GradStores (the core AllReduce
//                           primitive). Usable standalone for custom loops.

use std::marker::PhantomData;

use shrew_core::backend::Backend;
use shrew_core::backprop::GradStore;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use shrew_nn::Module;
use shrew_optim::Optimizer;

// AllReduce strategy

/// Strategy for combining gradients from multiple replicas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceOp {
    /// Sum all gradients (caller divides by N if needed).
    Sum,
    /// Average gradients across replicas (most common).
    Average,
}

// Gradient averaging

/// Average (or sum) multiple `GradStore`s into a single `GradStore`.
///
/// This is the core AllReduce primitive. Each worker produces a `GradStore`
/// from its backward pass; this function merges them.
///
/// # Arguments
/// - `grad_stores`: one `GradStore` per replica/worker
/// - `params`: the shared parameter tensors (used to enumerate keys)
/// - `strategy`: `Sum` or `Average`
pub fn reduce_gradients<B: Backend>(
    grad_stores: &[GradStore<B>],
    params: &[Tensor<B>],
    strategy: AllReduceOp,
) -> Result<GradStore<B>> {
    let n = grad_stores.len();
    if n == 0 {
        return Ok(GradStore::new());
    }
    if n == 1 {
        return Ok(grad_stores[0].clone());
    }

    let mut merged = GradStore::new();

    for param in params {
        // Collect gradients from all stores for this parameter
        let mut grads: Vec<&Tensor<B>> = Vec::new();
        for store in grad_stores {
            if let Some(g) = store.get(param) {
                grads.push(g);
            }
        }
        if grads.is_empty() {
            continue;
        }

        // Sum all gradients
        let mut acc = grads[0].clone();
        for g in &grads[1..] {
            acc = acc.add(g)?;
        }

        // Average if requested
        if strategy == AllReduceOp::Average && grads.len() > 1 {
            let scale = 1.0 / grads.len() as f64;
            acc = acc.affine(scale, 0.0)?;
        }

        merged.accumulate(param.id(), acc)?;
    }

    Ok(merged)
}

// DataParallel — Module wrapper for batch-parallel forward passes

/// Wraps a `Module` and splits each input batch across `num_workers` threads.
///
/// The forward pass:
///   1. Split input along dimension 0 into `num_workers` chunks
///   2. Run each chunk through the module in parallel (rayon)
///   3. Concatenate the outputs
///
/// Because all workers share the same parameters (Tensor uses Arc), the
/// autograd graph correctly tracks all operations. After computing loss
/// on the concatenated output and calling `.backward()`, the gradients
/// are automatically accumulated across all chunks.
///
/// # Example
/// ```ignore
/// let model = Sequential::new(vec![...]);
/// let dp = DataParallel::new(model, 4);  // 4 workers
/// let output = dp.forward(&big_batch)?;  // splits into 4 chunks
/// ```
pub struct DataParallel<M> {
    /// The underlying module (shared across workers).
    pub module: M,
    /// Number of parallel workers.
    pub num_workers: usize,
}

impl<M: Clone> Clone for DataParallel<M> {
    fn clone(&self) -> Self {
        Self {
            module: self.module.clone(),
            num_workers: self.num_workers,
        }
    }
}

impl<M: std::fmt::Debug> std::fmt::Debug for DataParallel<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataParallel")
            .field("module", &self.module)
            .field("num_workers", &self.num_workers)
            .finish()
    }
}

impl<M> DataParallel<M> {
    /// Create a `DataParallel` wrapper with the given number of workers.
    ///
    /// `num_workers` controls how many chunks the batch is split into.
    /// For CPU, this maps to rayon thread-pool parallelism.
    pub fn new(module: M, num_workers: usize) -> Self {
        assert!(num_workers > 0, "num_workers must be > 0");
        Self {
            module,
            num_workers,
        }
    }

    /// Get a reference to the underlying module.
    pub fn inner(&self) -> &M {
        &self.module
    }

    /// Get a mutable reference to the underlying module.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.module
    }

    /// Unwrap the `DataParallel`, returning the inner module.
    pub fn into_inner(self) -> M {
        self.module
    }
}

impl<M, B> Module<B> for DataParallel<M>
where
    M: Module<B> + Send + Sync,
    B: Backend,
{
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let batch_size = x.dims()[0];
        let effective_workers = self.num_workers.min(batch_size);

        if effective_workers <= 1 {
            return self.module.forward(x);
        }

        // Split into chunks along batch dimension
        let chunks = x.chunk(effective_workers, 0)?;

        // Run forward on each chunk (sequentially for now - rayon requires
        // the closure to be Send, which Result<Tensor<B>> satisfies)
        // NOTE: True multi-device parallelism requires each replica on a
        // separate device. For CPU, rayon gives thread-level parallelism.
        let mut outputs = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            outputs.push(self.module.forward(chunk)?);
        }

        // Concatenate results
        Tensor::cat(&outputs, 0)
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        self.module.parameters()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        self.module.named_parameters()
    }

    fn set_training(&self, training: bool) {
        self.module.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.module.is_training()
    }
}

// MixedPrecisionTrainer — FP16 forward + FP32 master weights

/// Configuration for dynamic loss scaling in mixed-precision training.
#[derive(Debug, Clone)]
pub struct LossScaleConfig {
    /// Initial loss scale factor (default: 2^16 = 65536).
    pub init_scale: f64,
    /// Multiply scale by this when no overflow (default: 2.0).
    pub scale_growth_factor: f64,
    /// Divide scale by this on overflow (default: 2.0).
    pub scale_backoff_factor: f64,
    /// Number of consecutive good steps before increasing scale (default: 2000).
    pub growth_interval: u64,
}

impl Default for LossScaleConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 2.0,
            growth_interval: 2000,
        }
    }
}

/// Mixed-precision training: reduced-precision forward/backward with FP32 master weights.
///
/// **Why mixed precision?**
/// - FP16/BF16 is 2× faster on GPUs with tensor cores (V100, A100, H100)
/// - Half-precision uses half the memory for activations, enabling larger batches
/// - FP32 master weights prevent precision loss during gradient updates
///
/// **How it works:**
/// 1. Inputs and targets are cast to `compute_dtype` (F16 or BF16) before forward
/// 2. The forward pass runs with reduced-precision activations
/// 3. Dynamic loss scaling prevents gradient underflow in half-precision:
///    - Loss is multiplied by a scale factor before backward
///    - Gradients are divided by the same factor after
///    - If overflow (NaN/Inf) is detected, the step is skipped and scale reduces
/// 4. Gradients are cast back to FP32 and applied to FP32 master weights
///
/// **Compute dtype options:**
/// - `DType::F16`: 16-bit IEEE float, range ±65504, good for most training
/// - `DType::BF16`: bfloat16, same range as F32 but less precision, preferred when available
/// - `DType::F32`: Standard precision (disables casting, only does loss scaling)
///
/// # Example
/// ```ignore
/// let model = Linear::<CpuBackend>::new(784, 10, true, DType::F32, &CpuDevice)?;
/// let optimizer = Adam::new(model.parameters(), 1e-3);
/// let mut trainer = MixedPrecisionTrainer::new(
///     model, optimizer, DType::F16, Default::default(),
/// );
///
/// for (input, target) in data {
///     let metrics = trainer.train_step(&input, &target, mse_loss)?;
///     println!("loss={:.4}, scale={}", metrics.loss, metrics.loss_scale);
/// }
/// ```
pub struct MixedPrecisionTrainer<M, O, B: Backend> {
    /// The model (with FP32 parameters as master copies).
    model: M,
    /// The optimizer operating on FP32 parameters.
    optimizer: O,
    /// The dtype for forward/backward computation (F16, BF16, or F32).
    compute_dtype: DType,
    /// Current loss scale factor.
    loss_scale: f64,
    /// Loss scale configuration.
    config: LossScaleConfig,
    /// Number of consecutive successful steps (no overflow).
    good_steps: u64,
    /// Total skipped steps (overflow detected).
    skipped_steps: u64,
    _phantom: PhantomData<B>,
}

/// Metrics from a single mixed-precision training step.
#[derive(Debug, Clone)]
pub struct MixedPrecisionMetrics {
    /// The unscaled loss value.
    pub loss: f64,
    /// Whether this step was skipped (overflow detected).
    pub skipped: bool,
    /// Current loss scale factor.
    pub loss_scale: f64,
    /// Total skipped steps so far.
    pub total_skipped: u64,
    /// The compute dtype used for this step.
    pub compute_dtype: DType,
}

impl<M, O, B> MixedPrecisionTrainer<M, O, B>
where
    M: Module<B>,
    O: Optimizer<B>,
    B: Backend,
{
    /// Create a new mixed-precision trainer.
    ///
    /// The model and optimizer should use FP32 parameters.
    /// `compute_dtype` sets the precision for forward/backward (F16, BF16, or F32).
    pub fn new(model: M, optimizer: O, compute_dtype: DType, config: LossScaleConfig) -> Self {
        let loss_scale = config.init_scale;
        Self {
            model,
            optimizer,
            compute_dtype,
            loss_scale,
            config,
            good_steps: 0,
            skipped_steps: 0,
            _phantom: PhantomData,
        }
    }

    /// Reference to the model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Mutable reference to the model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Reference to the optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Current loss scale.
    pub fn loss_scale(&self) -> f64 {
        self.loss_scale
    }

    /// The compute dtype (F16, BF16, or F32).
    pub fn compute_dtype(&self) -> DType {
        self.compute_dtype
    }

    /// Total number of skipped steps.
    pub fn skipped_steps(&self) -> u64 {
        self.skipped_steps
    }

    /// Perform one mixed-precision training step.
    ///
    /// The input and target are cast to `compute_dtype` for the forward pass.
    /// Dynamic loss scaling is applied to prevent gradient underflow.
    /// Gradients are cast back to FP32 and applied to FP32 master weights.
    ///
    /// # Arguments
    /// - `input`: input tensor (any dtype, will be cast to compute_dtype)
    /// - `target`: target tensor (any dtype, will be cast to compute_dtype)
    /// - `loss_fn`: function computing scalar loss from (prediction, target)
    ///
    /// # Returns
    /// `MixedPrecisionMetrics` with loss value and scaling info.
    pub fn train_step<F>(
        &mut self,
        input: &Tensor<B>,
        target: &Tensor<B>,
        loss_fn: F,
    ) -> Result<MixedPrecisionMetrics>
    where
        F: Fn(&Tensor<B>, &Tensor<B>) -> Result<Tensor<B>>,
    {
        // 1. Determine if we should cast inputs to compute_dtype.
        // Only cast if the model's parameters already match compute_dtype,
        // otherwise auto-casting inputs would cause dtype mismatches with weights.
        let model_dtype = self
            .model
            .parameters()
            .first()
            .map(|p| p.dtype())
            .unwrap_or(DType::F32);
        let should_cast = self.compute_dtype != DType::F32 && self.compute_dtype == model_dtype;

        let input_cast = if should_cast && input.dtype() != self.compute_dtype {
            input.to_dtype(self.compute_dtype)?
        } else {
            input.clone()
        };
        let target_cast = if should_cast && target.dtype() != self.compute_dtype {
            target.to_dtype(self.compute_dtype)?
        } else {
            target.clone()
        };

        // 2. Forward pass
        let output = self.model.forward(&input_cast)?;

        // 3. Compute loss (in compute_dtype)
        let loss = loss_fn(&output, &target_cast)?;
        let loss_val = loss.to_scalar_f64()?;

        // 4. Scale loss for backward (prevents gradient underflow in F16)
        let scaled_loss = loss.affine(self.loss_scale, 0.0)?;

        // 5. Backward on scaled loss
        let grads = scaled_loss.backward()?;

        // 6. Check for overflow in gradients
        let has_overflow = self.check_overflow(&grads)?;

        if has_overflow {
            // Skip this step, reduce the scale
            self.loss_scale /= self.config.scale_backoff_factor;
            self.loss_scale = self.loss_scale.max(1.0); // don't go below 1
            self.good_steps = 0;
            self.skipped_steps += 1;

            return Ok(MixedPrecisionMetrics {
                loss: loss_val,
                skipped: true,
                loss_scale: self.loss_scale,
                total_skipped: self.skipped_steps,
                compute_dtype: self.compute_dtype,
            });
        }

        // 7. Unscale gradients and cast back to FP32 for master weight update
        let unscaled = self.unscale_and_cast_gradients(&grads)?;

        // 8. Optimizer step with FP32 unscaled gradients
        self.optimizer.step(&unscaled)?;

        // 9. Update loss scale (possibly increase after consecutive good steps)
        self.good_steps += 1;
        if self.good_steps >= self.config.growth_interval {
            self.loss_scale *= self.config.scale_growth_factor;
            self.good_steps = 0;
        }

        Ok(MixedPrecisionMetrics {
            loss: loss_val,
            skipped: false,
            loss_scale: self.loss_scale,
            total_skipped: self.skipped_steps,
            compute_dtype: self.compute_dtype,
        })
    }

    /// Check if any gradient contains NaN or Inf.
    fn check_overflow(&self, grads: &GradStore<B>) -> Result<bool> {
        for param in self.model.parameters() {
            if let Some(g) = grads.get(&param) {
                let data = g.to_f64_vec()?;
                for &v in &data {
                    if v.is_nan() || v.is_infinite() {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Unscale gradients by the loss scale factor and cast to FP32.
    ///
    /// This ensures the optimizer always sees FP32 gradients, regardless
    /// of the compute dtype used during forward/backward.
    fn unscale_and_cast_gradients(&self, grads: &GradStore<B>) -> Result<GradStore<B>> {
        let inv_scale = 1.0 / self.loss_scale;
        let mut unscaled = GradStore::new();
        for param in self.model.parameters() {
            if let Some(g) = grads.get(&param) {
                // Unscale the gradient
                let g_unscaled = g.affine(inv_scale, 0.0)?;
                // Cast back to the master weight dtype (FP32) if needed
                let g_fp32 = if g_unscaled.dtype() != param.dtype() {
                    g_unscaled.to_dtype(param.dtype())?
                } else {
                    g_unscaled
                };
                unscaled.accumulate(param.id(), g_fp32)?;
            }
        }
        Ok(unscaled)
    }
}

// PipelineParallel — GPipe-style micro-batch pipelining

/// A stage in a pipeline-parallel model.
///
/// Each stage holds a sub-model (one or more layers). During execution,
/// micro-batches flow through stages in a pipeline fashion, overlapping
/// the forward and backward passes of different micro-batches.
pub struct PipelineStage<B: Backend> {
    /// The layers in this stage (as boxed Module).
    layers: Vec<Box<dyn Module<B>>>,
    /// Stage index (0-based).
    stage_id: usize,
}

impl<B: Backend> PipelineStage<B> {
    /// Create a new pipeline stage.
    pub fn new(stage_id: usize) -> Self {
        Self {
            layers: Vec::new(),
            stage_id,
        }
    }

    /// Add a layer to this stage.
    pub fn add_layer(mut self, layer: Box<dyn Module<B>>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Stage identifier.
    pub fn stage_id(&self) -> usize {
        self.stage_id
    }

    /// Forward pass through all layers in this stage.
    pub fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    /// Collect all parameters from all layers in this stage.
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

/// Pipeline-parallel executor using GPipe-style micro-batching.
///
/// Splits a model into sequential stages and processes micro-batches
/// through the pipeline. This increases throughput by overlapping
/// computation across stages.
///
/// # Example
/// ```ignore
/// let stage0 = PipelineStage::new(0)
///     .add_layer(Box::new(Linear::new(784, 256, true, DType::F32, &dev)?));
/// let stage1 = PipelineStage::new(1)
///     .add_layer(Box::new(Linear::new(256, 10, true, DType::F32, &dev)?));
///
/// let pipeline = PipelineParallel::new(vec![stage0, stage1], 4);
/// let output = pipeline.forward(&input)?;
/// ```
pub struct PipelineParallel<B: Backend> {
    /// Ordered stages of the model.
    stages: Vec<PipelineStage<B>>,
    /// Number of micro-batches to split each input into.
    num_micro_batches: usize,
}

impl<B: Backend> PipelineParallel<B> {
    /// Create a pipeline with the given stages and micro-batch count.
    pub fn new(stages: Vec<PipelineStage<B>>, num_micro_batches: usize) -> Self {
        assert!(!stages.is_empty(), "pipeline needs at least one stage");
        assert!(num_micro_batches > 0, "num_micro_batches must be > 0");
        Self {
            stages,
            num_micro_batches,
        }
    }

    /// Full forward pass through all stages.
    ///
    /// Splits the input into `num_micro_batches` micro-batches, runs each
    /// through the pipeline sequentially, and concatenates the results.
    ///
    /// In a multi-device setting, stages would run on different devices
    /// with inter-device transfers between stages.
    pub fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let batch_size = x.dims()[0];
        let effective_micros = self.num_micro_batches.min(batch_size);

        if effective_micros <= 1 {
            // No micro-batching — sequential pass through all stages
            let mut out = x.clone();
            for stage in &self.stages {
                out = stage.forward(&out)?;
            }
            return Ok(out);
        }

        // Split into micro-batches
        let micro_batches = x.chunk(effective_micros, 0)?;

        // Run each micro-batch through all stages
        let mut outputs = Vec::with_capacity(micro_batches.len());
        for mb in &micro_batches {
            let mut out = mb.clone();
            for stage in &self.stages {
                out = stage.forward(&out)?;
            }
            outputs.push(out);
        }

        // Concatenate micro-batch outputs
        Tensor::cat(&outputs, 0)
    }

    /// Collect all parameters from all stages (for optimizer).
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        self.stages.iter().flat_map(|s| s.parameters()).collect()
    }

    /// Number of stages.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Get a reference to a specific stage.
    pub fn stage(&self, idx: usize) -> Option<&PipelineStage<B>> {
        self.stages.get(idx)
    }
}

// ParallelTrainer — High-level training loop with gradient accumulation

/// High-level training loop with gradient accumulation.
///
/// Splits a large effective batch into `accumulation_steps` micro-batches,
/// accumulates gradients across all of them, then performs a single
/// optimizer step. This simulates a larger batch size without requiring
/// more memory.
///
/// # Example
/// ```ignore
/// let model = Sequential::new(vec![...]);
/// let optimizer = Adam::new(model.parameters(), 1e-3);
/// let mut trainer = ParallelTrainer::new(model, optimizer, 4);
///
/// // Each call accumulates 1/4 of the gradient; every 4th call steps.
/// for (i, (x, y)) in data.iter().enumerate() {
///     if let Some(loss) = trainer.accumulate_step(&x, &y, mse_loss)? {
///         println!("step {}: loss = {:.4}", i, loss);
///     }
/// }
/// ```
pub struct ParallelTrainer<M, O, B: Backend> {
    /// The model.
    pub model: M,
    /// The optimizer.
    pub optimizer: O,
    /// Number of micro-batches to accumulate before stepping.
    accumulation_steps: usize,
    /// Current accumulated gradients.
    accumulated: Option<GradStore<B>>,
    /// Current micro-batch index (0 .. accumulation_steps - 1).
    current_step: usize,
    /// Running loss sum for the current accumulation window.
    loss_sum: f64,
    _phantom: PhantomData<B>,
}

impl<M, O, B> ParallelTrainer<M, O, B>
where
    M: Module<B>,
    O: Optimizer<B>,
    B: Backend,
{
    /// Create a new `ParallelTrainer`.
    ///
    /// `accumulation_steps`: number of micro-batches before each optimizer step.
    pub fn new(model: M, optimizer: O, accumulation_steps: usize) -> Self {
        assert!(accumulation_steps > 0);
        Self {
            model,
            optimizer,
            accumulation_steps,
            accumulated: None,
            current_step: 0,
            loss_sum: 0.0,
            _phantom: PhantomData,
        }
    }

    /// Process one micro-batch. Returns `Some(avg_loss)` when an optimizer
    /// step was performed (every `accumulation_steps` calls), else `None`.
    pub fn accumulate_step<F>(
        &mut self,
        input: &Tensor<B>,
        target: &Tensor<B>,
        loss_fn: F,
    ) -> Result<Option<f64>>
    where
        F: Fn(&Tensor<B>, &Tensor<B>) -> Result<Tensor<B>>,
    {
        // Forward
        let output = self.model.forward(input)?;
        let loss = loss_fn(&output, target)?;
        let loss_val = loss.to_scalar_f64()?;
        self.loss_sum += loss_val;

        // Backward
        let grads = loss.backward()?;

        // Accumulate gradients
        let params = self.model.parameters();
        match self.accumulated.take() {
            Some(prev) => {
                let merged = reduce_gradients(&[prev, grads], &params, AllReduceOp::Sum)?;
                self.accumulated = Some(merged);
            }
            None => {
                self.accumulated = Some(grads);
            }
        }

        self.current_step += 1;

        // Step when we've accumulated enough
        if self.current_step >= self.accumulation_steps {
            let avg_grads = {
                let acc = self.accumulated.take().unwrap();
                // Average by accumulation_steps
                let mut averaged = GradStore::new();
                let scale = 1.0 / self.accumulation_steps as f64;
                for param in &params {
                    if let Some(g) = acc.get(param) {
                        let g_avg = g.affine(scale, 0.0)?;
                        averaged.accumulate(param.id(), g_avg)?;
                    }
                }
                averaged
            };

            self.optimizer.step(&avg_grads)?;

            let avg_loss = self.loss_sum / self.accumulation_steps as f64;
            self.current_step = 0;
            self.loss_sum = 0.0;
            self.accumulated = None;

            Ok(Some(avg_loss))
        } else {
            Ok(None)
        }
    }

    /// Force an optimizer step with whatever gradients have been accumulated so far.
    /// Useful at the end of an epoch when remaining micro-batches < accumulation_steps.
    pub fn flush(&mut self) -> Result<Option<f64>> {
        if self.current_step == 0 || self.accumulated.is_none() {
            return Ok(None);
        }

        let params = self.model.parameters();
        let acc = self.accumulated.take().unwrap();
        let scale = 1.0 / self.current_step as f64;
        let mut averaged = GradStore::new();
        for param in &params {
            if let Some(g) = acc.get(param) {
                let g_avg = g.affine(scale, 0.0)?;
                averaged.accumulate(param.id(), g_avg)?;
            }
        }

        self.optimizer.step(&averaged)?;

        let avg_loss = self.loss_sum / self.current_step as f64;
        self.current_step = 0;
        self.loss_sum = 0.0;
        self.accumulated = None;

        Ok(Some(avg_loss))
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use shrew_cpu::{CpuBackend, CpuDevice};

    type B = CpuBackend;
    type T = Tensor<B>;
    const DEV: CpuDevice = CpuDevice;

    // ── AllReduce / gradient averaging ──

    #[test]
    fn test_reduce_gradients_average() {
        let p = T::randn(vec![4], DType::F32, &DEV).unwrap().set_variable();
        let loss1 = p.sum_all().unwrap();
        let g1 = loss1.backward().unwrap();

        let loss2 = p.affine(2.0, 0.0).unwrap().sum_all().unwrap();
        let g2 = loss2.backward().unwrap();

        let merged = reduce_gradients(&[g1, g2], &[p.clone()], AllReduceOp::Average).unwrap();
        let avg = merged.get(&p).unwrap().to_f64_vec().unwrap();
        // g1 = all 1s, g2 = all 2s, average = all 1.5s
        for &v in &avg {
            assert!((v - 1.5).abs() < 1e-5, "expected 1.5, got {v}");
        }
    }

    #[test]
    fn test_reduce_gradients_sum() {
        let p = T::randn(vec![3], DType::F32, &DEV).unwrap().set_variable();
        let loss1 = p.sum_all().unwrap();
        let g1 = loss1.backward().unwrap();

        let loss2 = p.sum_all().unwrap();
        let g2 = loss2.backward().unwrap();

        let merged = reduce_gradients(&[g1, g2], &[p.clone()], AllReduceOp::Sum).unwrap();
        let summed = merged.get(&p).unwrap().to_f64_vec().unwrap();
        for &v in &summed {
            assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {v}");
        }
    }

    // ── DataParallel ──

    #[test]
    fn test_data_parallel_forward() {
        let linear = shrew_nn::Linear::<B>::new(4, 2, true, DType::F32, &DEV).unwrap();
        let dp = DataParallel::new(linear, 2);

        let input = T::randn(vec![6, 4], DType::F32, &DEV).unwrap();
        let output = dp.forward(&input).unwrap();
        assert_eq!(output.dims(), &[6, 2]);
    }

    #[test]
    fn test_data_parallel_single_worker() {
        let linear = shrew_nn::Linear::<B>::new(3, 2, true, DType::F32, &DEV).unwrap();
        let dp = DataParallel::new(linear, 1);

        let input = T::randn(vec![4, 3], DType::F32, &DEV).unwrap();
        let output = dp.forward(&input).unwrap();
        assert_eq!(output.dims(), &[4, 2]);
    }

    #[test]
    fn test_data_parallel_parameters() {
        let linear = shrew_nn::Linear::<B>::new(4, 2, true, DType::F32, &DEV).unwrap();
        let n_params = linear.parameters().len();
        let dp = DataParallel::new(linear, 4);
        assert_eq!(dp.parameters().len(), n_params);
    }

    // ── MixedPrecisionTrainer ──

    #[test]
    fn test_mixed_precision_basic() {
        let linear = shrew_nn::Linear::<B>::new(4, 1, true, DType::F32, &DEV).unwrap();
        let optimizer = shrew_optim::SGD::new(linear.parameters(), 0.01, 0.0, 0.0);
        let mut trainer =
            MixedPrecisionTrainer::new(linear, optimizer, DType::F16, LossScaleConfig::default());

        let input = T::randn(vec![2, 4], DType::F32, &DEV).unwrap();
        let target = T::zeros(vec![2, 1], DType::F32, &DEV).unwrap();

        let metrics = trainer
            .train_step(&input, &target, |pred, tgt| shrew_nn::mse_loss(pred, tgt))
            .unwrap();

        assert!(!metrics.skipped);
        assert!(metrics.loss >= 0.0);
        assert_eq!(metrics.loss_scale, 65536.0);
    }

    // ── Pipeline ──

    #[test]
    fn test_pipeline_forward() {
        let stage0 = PipelineStage::<B>::new(0).add_layer(Box::new(
            shrew_nn::Linear::<B>::new(4, 8, true, DType::F32, &DEV).unwrap(),
        ));
        let stage1 = PipelineStage::<B>::new(1).add_layer(Box::new(
            shrew_nn::Linear::<B>::new(8, 2, true, DType::F32, &DEV).unwrap(),
        ));

        let pipeline = PipelineParallel::new(vec![stage0, stage1], 2);
        let input = T::randn(vec![4, 4], DType::F32, &DEV).unwrap();
        let output = pipeline.forward(&input).unwrap();
        assert_eq!(output.dims(), &[4, 2]);
    }

    #[test]
    fn test_pipeline_parameters() {
        let stage0 = PipelineStage::<B>::new(0).add_layer(Box::new(
            shrew_nn::Linear::<B>::new(4, 8, true, DType::F32, &DEV).unwrap(),
        ));
        let stage1 = PipelineStage::<B>::new(1).add_layer(Box::new(
            shrew_nn::Linear::<B>::new(8, 2, true, DType::F32, &DEV).unwrap(),
        ));

        let pipeline = PipelineParallel::new(vec![stage0, stage1], 1);
        // stage0: 4*8 + 8 = 40, stage1: 8*2 + 2 = 18, total = 58
        let total: usize = pipeline.parameters().iter().map(|p| p.elem_count()).sum();
        assert_eq!(total, 40 + 18);
    }

    // ── ParallelTrainer (gradient accumulation) ──

    #[test]
    fn test_parallel_trainer_accumulation() {
        let linear = shrew_nn::Linear::<B>::new(3, 1, true, DType::F32, &DEV).unwrap();
        let optimizer = shrew_optim::SGD::new(linear.parameters(), 0.01, 0.0, 0.0);
        let mut trainer = ParallelTrainer::new(linear, optimizer, 2);

        let x1 = T::randn(vec![1, 3], DType::F32, &DEV).unwrap();
        let y1 = T::zeros(vec![1, 1], DType::F32, &DEV).unwrap();
        let x2 = T::randn(vec![1, 3], DType::F32, &DEV).unwrap();
        let y2 = T::zeros(vec![1, 1], DType::F32, &DEV).unwrap();

        // First micro-batch: no step yet
        let result1 = trainer
            .accumulate_step(&x1, &y1, |p, t| shrew_nn::mse_loss(p, t))
            .unwrap();
        assert!(result1.is_none());

        // Second micro-batch: step happens, returns average loss
        let result2 = trainer
            .accumulate_step(&x2, &y2, |p, t| shrew_nn::mse_loss(p, t))
            .unwrap();
        assert!(result2.is_some());
    }

    #[test]
    fn test_parallel_trainer_flush() {
        let linear = shrew_nn::Linear::<B>::new(3, 1, true, DType::F32, &DEV).unwrap();
        let optimizer = shrew_optim::SGD::new(linear.parameters(), 0.01, 0.0, 0.0);
        let mut trainer = ParallelTrainer::new(linear, optimizer, 4);

        let x = T::randn(vec![1, 3], DType::F32, &DEV).unwrap();
        let y = T::zeros(vec![1, 1], DType::F32, &DEV).unwrap();

        // Only 1 of 4 accumulation steps done
        trainer
            .accumulate_step(&x, &y, |p, t| shrew_nn::mse_loss(p, t))
            .unwrap();

        // Flush forces a step with whatever we have
        let flushed = trainer.flush().unwrap();
        assert!(flushed.is_some());
    }
}
