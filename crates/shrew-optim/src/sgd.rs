// SGD — Stochastic Gradient Descent
//
// The simplest optimizer: θ_new = θ - lr * gradient
//
// With momentum (the default in practice):
//   v = momentum * v_prev + gradient
//   θ_new = θ - lr * v
//
// Momentum accelerates convergence by accumulating a velocity vector.
// Think of a ball rolling down a hill — momentum helps it push through
// flat regions and small bumps.
//
// SGD with momentum is surprisingly competitive. Many state-of-the-art
// models (especially in computer vision) are trained with SGD + momentum.

use shrew_core::backend::Backend;
use shrew_core::backprop::GradStore;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::optimizer::{Optimizer, OptimizerState, Stateful};

/// Stochastic Gradient Descent optimizer with optional momentum.
///
/// # Parameters
/// - `lr`: learning rate (typical: 0.01 - 0.1)
/// - `momentum`: momentum factor (typical: 0.9, 0 = no momentum)
/// - `weight_decay`: L2 regularization coefficient (typical: 1e-4)
pub struct SGD<B: Backend> {
    params: Vec<Tensor<B>>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    /// Velocity buffers for momentum (one per parameter)
    velocities: Vec<Option<Vec<f64>>>,
}

impl<B: Backend> SGD<B> {
    /// Create a new SGD optimizer.
    ///
    /// # Arguments
    /// - `params`: the model parameters to optimize
    /// - `lr`: learning rate
    /// - `momentum`: momentum factor (0 for vanilla SGD)
    /// - `weight_decay`: L2 regularization strength
    pub fn new(params: Vec<Tensor<B>>, lr: f64, momentum: f64, weight_decay: f64) -> Self {
        let n = params.len();
        SGD {
            params,
            lr,
            momentum,
            weight_decay,
            velocities: vec![None; n],
        }
    }

    /// Update the parameter references (needed after step() returns new tensors).
    pub fn update_params(&mut self, new_params: Vec<Tensor<B>>) {
        self.params = new_params;
    }

    /// Access current parameters.
    pub fn params(&self) -> &[Tensor<B>] {
        &self.params
    }

    /// Mutable access to current parameters (for checkpoint loading).
    pub fn params_mut(&mut self) -> &mut Vec<Tensor<B>> {
        &mut self.params
    }
}

impl<B: Backend> Optimizer<B> for SGD<B> {
    fn step(&mut self, grads: &GradStore<B>) -> Result<Vec<Tensor<B>>> {
        let mut new_params = Vec::with_capacity(self.params.len());

        for (i, param) in self.params.iter().enumerate() {
            let grad = match grads.get(param) {
                Some(g) => g,
                None => {
                    // No gradient for this parameter — keep unchanged
                    new_params.push(param.clone());
                    continue;
                }
            };

            let mut grad_data = grad.to_f64_vec()?;
            let param_data = param.to_f64_vec()?;

            // Apply weight decay: grad = grad + weight_decay * param
            if self.weight_decay != 0.0 {
                for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Apply momentum
            if self.momentum != 0.0 {
                let velocity = self.velocities[i].get_or_insert_with(|| vec![0.0; grad_data.len()]);
                for (v, &g) in velocity.iter_mut().zip(grad_data.iter()) {
                    *v = self.momentum * *v + g;
                }
                grad_data = velocity.clone();
            }

            // Update: param = param - lr * grad_with_momentum
            let updated: Vec<f64> = param_data
                .iter()
                .zip(grad_data.iter())
                .map(|(&p, &g)| p - self.lr * g)
                .collect();

            // Update storage in-place so model layers sharing this tensor see new values
            param.update_data_inplace(&updated)?;

            new_params.push(param.clone());
        }

        Ok(new_params)
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// Stateful — Save/restore optimizer internal state

impl<B: Backend> Stateful for SGD<B> {
    fn state_dict(&self) -> OptimizerState {
        let mut state = OptimizerState::new("SGD");

        state.set_scalar("lr", self.lr);
        state.set_scalar("momentum", self.momentum);
        state.set_scalar("weight_decay", self.weight_decay);
        state.set_scalar("n_params", self.velocities.len() as f64);

        for (i, vel) in self.velocities.iter().enumerate() {
            if let Some(v) = vel {
                state.set_buffer(format!("velocity.{i}"), v.clone());
            }
        }

        state
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        if state.optimizer_type != "SGD" {
            return Err(shrew_core::Error::msg(format!(
                "Cannot load {} state into SGD optimizer",
                state.optimizer_type
            )));
        }

        if let Some(lr) = state.get_scalar("lr") {
            self.lr = lr;
        }
        if let Some(m) = state.get_scalar("momentum") {
            self.momentum = m;
        }
        if let Some(wd) = state.get_scalar("weight_decay") {
            self.weight_decay = wd;
        }

        let n = self.velocities.len();
        for i in 0..n {
            if let Some(buf) = state.get_buffer(&format!("velocity.{i}")) {
                self.velocities[i] = Some(buf.clone());
            }
        }

        Ok(())
    }
}
