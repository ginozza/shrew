// Recurrent Neural Network layers — RNN, LSTM, GRU
//
// This module implements the three fundamental recurrent architectures:
//
//   1. RNNCell / RNN   — Vanilla Elman RNN
//   2. LSTMCell / LSTM — Long Short-Term Memory
//   3. GRUCell / GRU   — Gated Recurrent Unit
//
// Each *Cell operates on a single timestep. The full RNN/LSTM/GRU wraps the
// cell and unrolls it over the sequence dimension, collecting all hidden
// states into a single output tensor via differentiable `cat`.
//
// SHAPES (batch_first convention):
//   input:  [batch, seq_len, input_size]
//   output: [batch, seq_len, hidden_size]
//   h_n:    [batch, hidden_size]       (RNN, GRU)
//   (h_n, c_n): ([batch, hidden_size], [batch, hidden_size])  (LSTM)
//
// WEIGHT INITIALIZATION:
//   All weights use Kaiming uniform U(-k, k) where k = sqrt(1/hidden_size),
//   following PyTorch's default initialization for recurrent layers.

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

// RNNCell — Single-step vanilla RNN
//
// h_t = tanh(x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh)
//
// This is the simplest recurrent unit. It suffers from vanishing gradients
// for long sequences, which LSTM and GRU were designed to address.

/// A single-step vanilla RNN cell.
///
/// Computes: `h' = tanh(x @ W_ih^T + b_ih + h @ W_hh^T + b_hh)`
///
/// # Shapes
/// - input x: `[batch, input_size]`
/// - hidden h: `[batch, hidden_size]`
/// - output h': `[batch, hidden_size]`
pub struct RNNCell<B: Backend> {
    w_ih: Tensor<B>,         // [hidden_size, input_size]
    w_hh: Tensor<B>,         // [hidden_size, hidden_size]
    b_ih: Option<Tensor<B>>, // [1, hidden_size]
    b_hh: Option<Tensor<B>>, // [1, hidden_size]
    pub input_size: usize,
    pub hidden_size: usize,
}

impl<B: Backend> RNNCell<B> {
    /// Create a new RNNCell with Kaiming uniform initialization.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        use_bias: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let k = (1.0 / hidden_size as f64).sqrt();

        let w_ih = Tensor::<B>::rand((hidden_size, input_size), dtype, device)?
            .affine(2.0 * k, -k)?
            .set_variable();
        let w_hh = Tensor::<B>::rand((hidden_size, hidden_size), dtype, device)?
            .affine(2.0 * k, -k)?
            .set_variable();

        let (b_ih, b_hh) = if use_bias {
            let bi = Tensor::<B>::rand((1, hidden_size), dtype, device)?
                .affine(2.0 * k, -k)?
                .set_variable();
            let bh = Tensor::<B>::rand((1, hidden_size), dtype, device)?
                .affine(2.0 * k, -k)?
                .set_variable();
            (Some(bi), Some(bh))
        } else {
            (None, None)
        };

        Ok(RNNCell {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            input_size,
            hidden_size,
        })
    }

    /// Forward: h' = tanh(x @ W_ih^T + b_ih + h @ W_hh^T + b_hh)
    ///
    /// - `x`: `[batch, input_size]`
    /// - `h`: `[batch, hidden_size]`
    /// - returns h': `[batch, hidden_size]`
    pub fn forward(&self, x: &Tensor<B>, h: &Tensor<B>) -> Result<Tensor<B>> {
        // x @ W_ih^T → [batch, hidden_size]
        let wih_t = self.w_ih.t()?.contiguous()?;
        let mut gates = x.matmul(&wih_t)?;
        if let Some(ref b) = self.b_ih {
            gates = gates.add(b)?;
        }

        // h @ W_hh^T → [batch, hidden_size]
        let whh_t = self.w_hh.t()?.contiguous()?;
        let mut h_part = h.matmul(&whh_t)?;
        if let Some(ref b) = self.b_hh {
            h_part = h_part.add(b)?;
        }

        gates.add(&h_part)?.tanh()
    }

    /// Return all trainable parameters.
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        let mut params = vec![self.w_ih.clone(), self.w_hh.clone()];
        if let Some(ref b) = self.b_ih {
            params.push(b.clone());
        }
        if let Some(ref b) = self.b_hh {
            params.push(b.clone());
        }
        params
    }

    /// Return all trainable parameters with names.
    pub fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        let mut named = vec![
            ("w_ih".to_string(), self.w_ih.clone()),
            ("w_hh".to_string(), self.w_hh.clone()),
        ];
        if let Some(ref b) = self.b_ih {
            named.push(("b_ih".to_string(), b.clone()));
        }
        if let Some(ref b) = self.b_hh {
            named.push(("b_hh".to_string(), b.clone()));
        }
        named
    }
}

// RNN — Unrolled vanilla RNN over a sequence

/// A multi-step vanilla RNN that unrolls an RNNCell over the sequence dimension.
///
/// # Shapes
/// - input:  `[batch, seq_len, input_size]`
/// - output: `[batch, seq_len, hidden_size]` — all hidden states
/// - h_n:    `[batch, hidden_size]` — final hidden state
pub struct RNN<B: Backend> {
    cell: RNNCell<B>,
}

impl<B: Backend> RNN<B> {
    /// Create a new RNN layer.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        use_bias: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let cell = RNNCell::new(input_size, hidden_size, use_bias, dtype, device)?;
        Ok(RNN { cell })
    }

    /// Forward pass over the full sequence.
    ///
    /// - `x`: `[batch, seq_len, input_size]`
    /// - `h0`: optional initial hidden state `[batch, hidden_size]`.
    ///   If None, zeros are used.
    ///
    /// Returns `(output, h_n)` where:
    /// - `output`: `[batch, seq_len, hidden_size]`
    /// - `h_n`: `[batch, hidden_size]`
    pub fn forward(&self, x: &Tensor<B>, h0: Option<&Tensor<B>>) -> Result<(Tensor<B>, Tensor<B>)> {
        let dims = x.dims();
        let batch = dims[0];
        let seq_len = dims[1];

        // Initialize hidden state
        let mut h = match h0 {
            Some(h) => h.clone(),
            None => Tensor::<B>::zeros((batch, self.cell.hidden_size), x.dtype(), x.device())?,
        };

        // Unroll over timesteps
        let mut outputs: Vec<Tensor<B>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // x_t: [batch, 1, input_size] → [batch, input_size]
            let x_t = x.narrow(1, t, 1)?.reshape((batch, self.cell.input_size))?;
            h = self.cell.forward(&x_t, &h)?;
            // h: [batch, hidden_size] → [batch, 1, hidden_size] for stacking
            outputs.push(h.reshape((batch, 1, self.cell.hidden_size))?);
        }

        // Stack: [batch, seq_len, hidden_size]
        let output = Tensor::cat(&outputs, 1)?;
        Ok((output, h))
    }

    /// Return all trainable parameters.
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        self.cell.parameters()
    }

    /// Return all trainable parameters with names.
    pub fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        self.cell
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("cell.{k}"), v))
            .collect()
    }

    /// Access the underlying cell.
    pub fn cell(&self) -> &RNNCell<B> {
        &self.cell
    }
}

// LSTMCell — Single-step LSTM
//
// The LSTM uses four gates (input, forget, cell, output) to control
// information flow, solving the vanishing gradient problem of vanilla RNNs.
//
// gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh    # [batch, 4*hidden]
// i, f, g, o = chunk(gates, 4)
// i = sigmoid(i)   — input gate:  how much new info to let in
// f = sigmoid(f)   — forget gate: how much old info to keep
// g = tanh(g)      — cell gate:   candidate values to add
// o = sigmoid(o)   — output gate: how much state to expose
// c' = f * c + i * g
// h' = o * tanh(c')

/// A single-step LSTM cell.
///
/// # Shapes
/// - input x: `[batch, input_size]`
/// - hidden h: `[batch, hidden_size]`
/// - cell c: `[batch, hidden_size]`
/// - output (h', c'): `([batch, hidden_size], [batch, hidden_size])`
pub struct LSTMCell<B: Backend> {
    w_ih: Tensor<B>,         // [4*hidden_size, input_size]
    w_hh: Tensor<B>,         // [4*hidden_size, hidden_size]
    b_ih: Option<Tensor<B>>, // [1, 4*hidden_size]
    b_hh: Option<Tensor<B>>, // [1, 4*hidden_size]
    pub input_size: usize,
    pub hidden_size: usize,
}

impl<B: Backend> LSTMCell<B> {
    /// Create a new LSTMCell with Kaiming uniform initialization.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        use_bias: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let gate_size = 4 * hidden_size;
        let k = (1.0 / hidden_size as f64).sqrt();

        let w_ih = Tensor::<B>::rand((gate_size, input_size), dtype, device)?
            .affine(2.0 * k, -k)?
            .set_variable();
        let w_hh = Tensor::<B>::rand((gate_size, hidden_size), dtype, device)?
            .affine(2.0 * k, -k)?
            .set_variable();

        let (b_ih, b_hh) = if use_bias {
            let bi = Tensor::<B>::rand((1, gate_size), dtype, device)?
                .affine(2.0 * k, -k)?
                .set_variable();
            let bh = Tensor::<B>::rand((1, gate_size), dtype, device)?
                .affine(2.0 * k, -k)?
                .set_variable();
            (Some(bi), Some(bh))
        } else {
            (None, None)
        };

        Ok(LSTMCell {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            input_size,
            hidden_size,
        })
    }

    /// Forward: compute (h', c') from (x, h, c)
    ///
    /// - `x`: `[batch, input_size]`
    /// - `h`: `[batch, hidden_size]`
    /// - `c`: `[batch, hidden_size]`
    /// - returns `(h', c')`: `([batch, hidden_size], [batch, hidden_size])`
    pub fn forward(
        &self,
        x: &Tensor<B>,
        h: &Tensor<B>,
        c: &Tensor<B>,
    ) -> Result<(Tensor<B>, Tensor<B>)> {
        // Compute all 4 gates at once: [batch, 4*hidden_size]
        let wih_t = self.w_ih.t()?.contiguous()?;
        let mut gates = x.matmul(&wih_t)?;
        if let Some(ref b) = self.b_ih {
            gates = gates.add(b)?;
        }

        let whh_t = self.w_hh.t()?.contiguous()?;
        let mut h_part = h.matmul(&whh_t)?;
        if let Some(ref b) = self.b_hh {
            h_part = h_part.add(b)?;
        }

        gates = gates.add(&h_part)?;

        // Split into 4 gates: each [batch, hidden_size]
        let chunks = gates.chunk(4, 1)?;
        let i_gate = chunks[0].sigmoid()?; // input gate
        let f_gate = chunks[1].sigmoid()?; // forget gate
        let g_gate = chunks[2].tanh()?; // cell gate (candidate)
        let o_gate = chunks[3].sigmoid()?; // output gate

        // c' = f * c + i * g
        let c_new = f_gate.mul(c)?.add(&i_gate.mul(&g_gate)?)?;

        // h' = o * tanh(c')
        let h_new = o_gate.mul(&c_new.tanh()?)?;

        Ok((h_new, c_new))
    }

    /// Return all trainable parameters.
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        let mut params = vec![self.w_ih.clone(), self.w_hh.clone()];
        if let Some(ref b) = self.b_ih {
            params.push(b.clone());
        }
        if let Some(ref b) = self.b_hh {
            params.push(b.clone());
        }
        params
    }

    /// Return all trainable parameters with names.
    pub fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        let mut named = vec![
            ("w_ih".to_string(), self.w_ih.clone()),
            ("w_hh".to_string(), self.w_hh.clone()),
        ];
        if let Some(ref b) = self.b_ih {
            named.push(("b_ih".to_string(), b.clone()));
        }
        if let Some(ref b) = self.b_hh {
            named.push(("b_hh".to_string(), b.clone()));
        }
        named
    }
}

// LSTM — Unrolled LSTM over a sequence

/// A multi-step LSTM that unrolls an LSTMCell over the sequence dimension.
///
/// # Shapes
/// - input:  `[batch, seq_len, input_size]`
/// - output: `[batch, seq_len, hidden_size]`
/// - h_n:    `[batch, hidden_size]`
/// - c_n:    `[batch, hidden_size]`
pub struct LSTM<B: Backend> {
    cell: LSTMCell<B>,
}

impl<B: Backend> LSTM<B> {
    /// Create a new LSTM layer.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        use_bias: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let cell = LSTMCell::new(input_size, hidden_size, use_bias, dtype, device)?;
        Ok(LSTM { cell })
    }

    /// Forward pass over the full sequence.
    ///
    /// - `x`: `[batch, seq_len, input_size]`
    /// - `hc0`: optional initial `(h0, c0)`, each `[batch, hidden_size]`.
    ///   If None, zeros are used.
    ///
    /// Returns `(output, (h_n, c_n))` where:
    /// - `output`: `[batch, seq_len, hidden_size]`
    /// - `h_n`, `c_n`: `[batch, hidden_size]`
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        x: &Tensor<B>,
        hc0: Option<(&Tensor<B>, &Tensor<B>)>,
    ) -> Result<(Tensor<B>, (Tensor<B>, Tensor<B>))> {
        let dims = x.dims();
        let batch = dims[0];
        let seq_len = dims[1];
        let hs = self.cell.hidden_size;

        // Initialize hidden and cell states
        let (mut h, mut c) = match hc0 {
            Some((h0, c0)) => (h0.clone(), c0.clone()),
            None => (
                Tensor::<B>::zeros((batch, hs), x.dtype(), x.device())?,
                Tensor::<B>::zeros((batch, hs), x.dtype(), x.device())?,
            ),
        };

        // Unroll over timesteps
        let mut outputs: Vec<Tensor<B>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = x.narrow(1, t, 1)?.reshape((batch, self.cell.input_size))?;
            let (h_new, c_new) = self.cell.forward(&x_t, &h, &c)?;
            h = h_new;
            c = c_new;
            outputs.push(h.reshape((batch, 1, hs))?);
        }

        let output = Tensor::cat(&outputs, 1)?;
        Ok((output, (h, c)))
    }

    /// Return all trainable parameters.
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        self.cell.parameters()
    }

    /// Return all trainable parameters with names.
    pub fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        self.cell
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("cell.{k}"), v))
            .collect()
    }

    /// Access the underlying cell.
    pub fn cell(&self) -> &LSTMCell<B> {
        &self.cell
    }
}

// GRUCell — Single-step GRU
//
// The GRU simplifies the LSTM by merging the forget and input gates into a
// single "update" gate, and using a "reset" gate to control how much of the
// previous hidden state to expose.
//
// gates_ih = x @ W_ih^T + b_ih          [batch, 3*hidden]
// gates_hh = h @ W_hh^T + b_hh          [batch, 3*hidden]
// r_ih, z_ih, n_ih = chunk(gates_ih, 3)
// r_hh, z_hh, n_hh = chunk(gates_hh, 3)
//
// r = sigmoid(r_ih + r_hh)    — reset gate
// z = sigmoid(z_ih + z_hh)    — update gate
// n = tanh(n_ih + r * n_hh)   — new gate (candidate)
//
// h' = (1 - z) * n + z * h

/// A single-step GRU cell.
///
/// # Shapes
/// - input x: `[batch, input_size]`
/// - hidden h: `[batch, hidden_size]`
/// - output h': `[batch, hidden_size]`
pub struct GRUCell<B: Backend> {
    w_ih: Tensor<B>,         // [3*hidden_size, input_size]
    w_hh: Tensor<B>,         // [3*hidden_size, hidden_size]
    b_ih: Option<Tensor<B>>, // [1, 3*hidden_size]
    b_hh: Option<Tensor<B>>, // [1, 3*hidden_size]
    pub input_size: usize,
    pub hidden_size: usize,
}

impl<B: Backend> GRUCell<B> {
    /// Create a new GRUCell with Kaiming uniform initialization.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        use_bias: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let gate_size = 3 * hidden_size;
        let k = (1.0 / hidden_size as f64).sqrt();

        let w_ih = Tensor::<B>::rand((gate_size, input_size), dtype, device)?
            .affine(2.0 * k, -k)?
            .set_variable();
        let w_hh = Tensor::<B>::rand((gate_size, hidden_size), dtype, device)?
            .affine(2.0 * k, -k)?
            .set_variable();

        let (b_ih, b_hh) = if use_bias {
            let bi = Tensor::<B>::rand((1, gate_size), dtype, device)?
                .affine(2.0 * k, -k)?
                .set_variable();
            let bh = Tensor::<B>::rand((1, gate_size), dtype, device)?
                .affine(2.0 * k, -k)?
                .set_variable();
            (Some(bi), Some(bh))
        } else {
            (None, None)
        };

        Ok(GRUCell {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            input_size,
            hidden_size,
        })
    }

    /// Forward: compute h' from (x, h)
    ///
    /// - `x`: `[batch, input_size]`
    /// - `h`: `[batch, hidden_size]`
    /// - returns h': `[batch, hidden_size]`
    pub fn forward(&self, x: &Tensor<B>, h: &Tensor<B>) -> Result<Tensor<B>> {
        // Compute input-side and hidden-side gates
        let wih_t = self.w_ih.t()?.contiguous()?;
        let mut gates_ih = x.matmul(&wih_t)?;
        if let Some(ref b) = self.b_ih {
            gates_ih = gates_ih.add(b)?;
        }

        let whh_t = self.w_hh.t()?.contiguous()?;
        let mut gates_hh = h.matmul(&whh_t)?;
        if let Some(ref b) = self.b_hh {
            gates_hh = gates_hh.add(b)?;
        }

        // Split each into 3 parts: reset, update, new
        let ih_chunks = gates_ih.chunk(3, 1)?;
        let hh_chunks = gates_hh.chunk(3, 1)?;

        // r = sigmoid(r_ih + r_hh)  — reset gate
        let r = ih_chunks[0].add(&hh_chunks[0])?.sigmoid()?;

        // z = sigmoid(z_ih + z_hh)  — update gate
        let z = ih_chunks[1].add(&hh_chunks[1])?.sigmoid()?;

        // n = tanh(n_ih + r * n_hh)  — new gate (candidate hidden state)
        let n = ih_chunks[2].add(&r.mul(&hh_chunks[2])?)?.tanh()?;

        // h' = (1 - z) * n + z * h
        // (1 - z) = z.affine(-1.0, 1.0)
        let one_minus_z = z.affine(-1.0, 1.0)?;
        one_minus_z.mul(&n)?.add(&z.mul(h)?)
    }

    /// Return all trainable parameters.
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        let mut params = vec![self.w_ih.clone(), self.w_hh.clone()];
        if let Some(ref b) = self.b_ih {
            params.push(b.clone());
        }
        if let Some(ref b) = self.b_hh {
            params.push(b.clone());
        }
        params
    }

    /// Return all trainable parameters with names.
    pub fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        let mut named = vec![
            ("w_ih".to_string(), self.w_ih.clone()),
            ("w_hh".to_string(), self.w_hh.clone()),
        ];
        if let Some(ref b) = self.b_ih {
            named.push(("b_ih".to_string(), b.clone()));
        }
        if let Some(ref b) = self.b_hh {
            named.push(("b_hh".to_string(), b.clone()));
        }
        named
    }
}

// GRU — Unrolled GRU over a sequence

/// A multi-step GRU that unrolls a GRUCell over the sequence dimension.
///
/// # Shapes
/// - input:  `[batch, seq_len, input_size]`
/// - output: `[batch, seq_len, hidden_size]`
/// - h_n:    `[batch, hidden_size]`
pub struct GRU<B: Backend> {
    cell: GRUCell<B>,
}

impl<B: Backend> GRU<B> {
    /// Create a new GRU layer.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        use_bias: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let cell = GRUCell::new(input_size, hidden_size, use_bias, dtype, device)?;
        Ok(GRU { cell })
    }

    /// Forward pass over the full sequence.
    ///
    /// - `x`: `[batch, seq_len, input_size]`
    /// - `h0`: optional initial hidden state `[batch, hidden_size]`.
    ///   If None, zeros are used.
    ///
    /// Returns `(output, h_n)` where:
    /// - `output`: `[batch, seq_len, hidden_size]`
    /// - `h_n`: `[batch, hidden_size]`
    pub fn forward(&self, x: &Tensor<B>, h0: Option<&Tensor<B>>) -> Result<(Tensor<B>, Tensor<B>)> {
        let dims = x.dims();
        let batch = dims[0];
        let seq_len = dims[1];
        let hs = self.cell.hidden_size;

        let mut h = match h0 {
            Some(h) => h.clone(),
            None => Tensor::<B>::zeros((batch, hs), x.dtype(), x.device())?,
        };

        let mut outputs: Vec<Tensor<B>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = x.narrow(1, t, 1)?.reshape((batch, self.cell.input_size))?;
            h = self.cell.forward(&x_t, &h)?;
            outputs.push(h.reshape((batch, 1, hs))?);
        }

        let output = Tensor::cat(&outputs, 1)?;
        Ok((output, h))
    }

    /// Return all trainable parameters.
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        self.cell.parameters()
    }

    /// Return all trainable parameters with names.
    pub fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        self.cell
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("cell.{k}"), v))
            .collect()
    }

    /// Access the underlying cell.
    pub fn cell(&self) -> &GRUCell<B> {
        &self.cell
    }
}
