// Learning Rate Schedulers — Adjust the learning rate during training
//
// LR schedulers implement a strategy for changing the learning rate across
// training steps. They are designed to work with any Optimizer via
// `set_learning_rate()`.
//
// IMPLEMENTED:
//   - StepLR: Decay by gamma every `step_size` epochs
//   - CosineAnnealingLR: Cosine decay from initial LR to min LR
//   - CosineWarmupLR: Linear warmup → cosine decay (standard for Transformers)
//   - LinearLR: Linear interpolation from start_factor to end_factor
//   - ExponentialLR: Multiply LR by gamma every epoch
//
// USAGE:
//   let mut scheduler = CosineWarmupLR::new(initial_lr, warmup_steps, total_steps, min_lr);
//   for epoch in 0..epochs {
//       for batch in batches {
//           let lr = scheduler.step();
//           optimizer.set_learning_rate(lr);
//           // ... training step ...
//       }
//   }

use std::f64::consts::PI;

// Scheduler Trait

/// Trait for learning rate schedulers.
///
/// Each call to `step()` advances the internal counter and returns the new LR.
pub trait LrScheduler {
    /// Advance by one step and return the new learning rate.
    fn step(&mut self) -> f64;

    /// Get the current learning rate without advancing.
    fn current_lr(&self) -> f64;

    /// Get the current step count.
    fn current_step(&self) -> u64;

    /// Reset the scheduler to step 0.
    fn reset(&mut self);

    /// Set the internal step counter to a specific value (for checkpoint restore).
    fn set_step(&mut self, step: u64);
}

// StepLR — Decay by gamma every N epochs

/// Multiply the learning rate by `gamma` every `step_size` steps.
///
/// ```text
/// lr = initial_lr * gamma^(current_step / step_size)
/// ```
///
/// # Example
/// ```ignore
/// let mut sched = StepLR::new(0.1, 30, 0.1); // decay by 10x every 30 steps
/// ```
pub struct StepLR {
    initial_lr: f64,
    step_size: u64,
    gamma: f64,
    current: u64,
}

impl StepLR {
    pub fn new(initial_lr: f64, step_size: u64, gamma: f64) -> Self {
        StepLR {
            initial_lr,
            step_size,
            gamma,
            current: 0,
        }
    }
}

impl LrScheduler for StepLR {
    fn step(&mut self) -> f64 {
        self.current += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        let n = self.current / self.step_size;
        self.initial_lr * self.gamma.powi(n as i32)
    }

    fn current_step(&self) -> u64 {
        self.current
    }
    fn reset(&mut self) {
        self.current = 0;
    }
    fn set_step(&mut self, step: u64) {
        self.current = step;
    }
}

// ExponentialLR — Multiply LR by gamma every step

/// Multiply the learning rate by `gamma` every step.
///
/// ```text
/// lr = initial_lr * gamma^step
/// ```
pub struct ExponentialLR {
    initial_lr: f64,
    gamma: f64,
    current: u64,
}

impl ExponentialLR {
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        ExponentialLR {
            initial_lr,
            gamma,
            current: 0,
        }
    }
}

impl LrScheduler for ExponentialLR {
    fn step(&mut self) -> f64 {
        self.current += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        self.initial_lr * self.gamma.powi(self.current as i32)
    }

    fn current_step(&self) -> u64 {
        self.current
    }
    fn reset(&mut self) {
        self.current = 0;
    }
    fn set_step(&mut self, step: u64) {
        self.current = step;
    }
}

// LinearLR — Linear interpolation between two factors

/// Linearly interpolate the learning rate from `start_factor * initial_lr`
/// to `end_factor * initial_lr` over `total_steps` steps.
///
/// After `total_steps`, the LR stays at `end_factor * initial_lr`.
pub struct LinearLR {
    initial_lr: f64,
    start_factor: f64,
    end_factor: f64,
    total_steps: u64,
    current: u64,
}

impl LinearLR {
    pub fn new(initial_lr: f64, start_factor: f64, end_factor: f64, total_steps: u64) -> Self {
        LinearLR {
            initial_lr,
            start_factor,
            end_factor,
            total_steps,
            current: 0,
        }
    }
}

impl LrScheduler for LinearLR {
    fn step(&mut self) -> f64 {
        self.current += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        if self.total_steps == 0 {
            return self.initial_lr * self.end_factor;
        }
        let t = (self.current as f64 / self.total_steps as f64).min(1.0);
        let factor = self.start_factor + (self.end_factor - self.start_factor) * t;
        self.initial_lr * factor
    }

    fn current_step(&self) -> u64 {
        self.current
    }
    fn reset(&mut self) {
        self.current = 0;
    }
    fn set_step(&mut self, step: u64) {
        self.current = step;
    }
}

// CosineAnnealingLR — Cosine decay from initial to minimum LR

/// Cosine annealing from `initial_lr` to `min_lr` over `total_steps`.
///
/// ```text
/// lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * step / total_steps))
/// ```
///
/// After `total_steps`, the LR stays at `min_lr`.
pub struct CosineAnnealingLR {
    initial_lr: f64,
    min_lr: f64,
    total_steps: u64,
    current: u64,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f64, total_steps: u64, min_lr: f64) -> Self {
        CosineAnnealingLR {
            initial_lr,
            min_lr,
            total_steps,
            current: 0,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    fn step(&mut self) -> f64 {
        self.current += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        if self.current >= self.total_steps {
            return self.min_lr;
        }
        let progress = self.current as f64 / self.total_steps as f64;
        self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + (PI * progress).cos())
    }

    fn current_step(&self) -> u64 {
        self.current
    }
    fn reset(&mut self) {
        self.current = 0;
    }
    fn set_step(&mut self, step: u64) {
        self.current = step;
    }
}

// CosineWarmupLR — Linear warmup → cosine decay (THE transformer scheduler)

/// Linear warmup from 0 to `initial_lr` over `warmup_steps`, then cosine
/// decay from `initial_lr` to `min_lr` over the remaining steps.
///
/// This is the standard scheduler used for training transformers (GPT, BERT, etc.).
///
/// ```text
/// warmup phase (step < warmup_steps):
///   lr = initial_lr * step / warmup_steps
///
/// decay phase (step >= warmup_steps):
///   progress = (step - warmup_steps) / (total_steps - warmup_steps)
///   lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * progress))
/// ```
pub struct CosineWarmupLR {
    initial_lr: f64,
    min_lr: f64,
    warmup_steps: u64,
    total_steps: u64,
    current: u64,
}

impl CosineWarmupLR {
    /// Create a cosine warmup scheduler.
    ///
    /// # Arguments
    /// - `initial_lr`: Peak learning rate (reached at end of warmup)
    /// - `warmup_steps`: Number of linear warmup steps
    /// - `total_steps`: Total training steps (warmup + decay)
    /// - `min_lr`: Minimum learning rate at end of training
    pub fn new(initial_lr: f64, warmup_steps: u64, total_steps: u64, min_lr: f64) -> Self {
        assert!(
            warmup_steps <= total_steps,
            "warmup_steps ({warmup_steps}) must be <= total_steps ({total_steps})"
        );
        CosineWarmupLR {
            initial_lr,
            min_lr,
            warmup_steps,
            total_steps,
            current: 0,
        }
    }
}

impl LrScheduler for CosineWarmupLR {
    fn step(&mut self) -> f64 {
        self.current += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        if self.current <= self.warmup_steps {
            // Linear warmup: 0 → initial_lr
            if self.warmup_steps == 0 {
                return self.initial_lr;
            }
            self.initial_lr * (self.current as f64 / self.warmup_steps as f64)
        } else if self.current >= self.total_steps {
            // Past end of schedule
            self.min_lr
        } else {
            // Cosine decay phase
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_current = self.current - self.warmup_steps;
            let progress = decay_current as f64 / decay_steps as f64;
            self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + (PI * progress).cos())
        }
    }

    fn current_step(&self) -> u64 {
        self.current
    }
    fn reset(&mut self) {
        self.current = 0;
    }
    fn set_step(&mut self, step: u64) {
        self.current = step;
    }
}

// ReduceLROnPlateau — Reduce LR when a metric stops improving

/// Reduce the learning rate when a monitored metric plateaus.
///
/// Unlike the other schedulers which step automatically, this scheduler
/// requires you to report the metric value (e.g., validation loss) and
/// it decides whether to reduce the LR.
///
/// # Arguments (builder pattern)
/// - `factor`: Factor to multiply LR by when reducing (default: 0.1)
/// - `patience`: Number of steps with no improvement before reducing (default: 10)
/// - `min_lr`: Lower bound on the learning rate (default: 1e-6)
/// - `threshold`: Minimum improvement to qualify as improvement (default: 1e-4)
///
/// # Example
/// ```ignore
/// let mut sched = ReduceLROnPlateau::new(0.01);
/// // After each epoch:
/// let new_lr = sched.step_metric(val_loss);
/// optimizer.set_learning_rate(new_lr);
/// ```
pub struct ReduceLROnPlateau {
    lr: f64,
    factor: f64,
    patience: u64,
    min_lr: f64,
    threshold: f64,
    /// Whether lower metric is better (true = min mode, false = max mode)
    mode_min: bool,
    best: f64,
    num_bad_steps: u64,
    current_step_count: u64,
}

impl ReduceLROnPlateau {
    /// Create a new ReduceLROnPlateau with sensible defaults.
    ///
    /// Default: factor=0.1, patience=10, min_lr=1e-6, threshold=1e-4, mode=min
    pub fn new(initial_lr: f64) -> Self {
        ReduceLROnPlateau {
            lr: initial_lr,
            factor: 0.1,
            patience: 10,
            min_lr: 1e-6,
            threshold: 1e-4,
            mode_min: true,
            best: f64::INFINITY,
            num_bad_steps: 0,
            current_step_count: 0,
        }
    }

    /// Set the factor by which to reduce LR (default: 0.1).
    pub fn factor(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }

    /// Set patience (steps without improvement before reducing, default: 10).
    pub fn patience(mut self, patience: u64) -> Self {
        self.patience = patience;
        self
    }

    /// Set the minimum learning rate (default: 1e-6).
    pub fn min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Set the improvement threshold (default: 1e-4).
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set mode to maximize (higher metric = better).
    /// Default is minimize (lower metric = better).
    pub fn mode_max(mut self) -> Self {
        self.mode_min = false;
        self.best = f64::NEG_INFINITY;
        self
    }

    /// Report a metric value and return the (possibly updated) learning rate.
    ///
    /// Call this once per epoch/evaluation with the metric value (e.g., val loss).
    pub fn step_metric(&mut self, metric: f64) -> f64 {
        self.current_step_count += 1;

        let improved = if self.mode_min {
            metric < self.best - self.threshold
        } else {
            metric > self.best + self.threshold
        };

        if improved {
            self.best = metric;
            self.num_bad_steps = 0;
        } else {
            self.num_bad_steps += 1;
            if self.num_bad_steps >= self.patience {
                let new_lr = (self.lr * self.factor).max(self.min_lr);
                self.lr = new_lr;
                self.num_bad_steps = 0;
            }
        }

        self.lr
    }

    /// Get the current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Get the best metric value seen so far.
    pub fn best_metric(&self) -> f64 {
        self.best
    }

    /// Get number of steps without improvement.
    pub fn bad_steps(&self) -> u64 {
        self.num_bad_steps
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.num_bad_steps = 0;
        self.current_step_count = 0;
        if self.mode_min {
            self.best = f64::INFINITY;
        } else {
            self.best = f64::NEG_INFINITY;
        }
    }
}
