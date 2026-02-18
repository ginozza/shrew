// =============================================================================
// MNIST MLP — A simple handwritten-digit classifier in Shrew IR (.sw)
// =============================================================================
//
// Architecture: Input(784) → Linear(784,128) → ReLU → Linear(128,10) → Output
//
// This model can be loaded with:
//   let trainer = shrew::exec::load_trainer::<CpuBackend>(source, device, config)?;
//   let result  = trainer.train(&batches, "target")?;

@model {
    name: "MNIST-MLP";
    version: "0.1.0";
    author: "Shrew Contributors";
}

@config {
    input_dim: 784;
    hidden_dim: 128;
    num_classes: 10;
}

// ─────────────────────────────────────────────────────────────────────────────
// Computation graph
// ─────────────────────────────────────────────────────────────────────────────

@graph Forward(x: Tensor<[Batch, 784], f32>) -> Tensor<[Batch, 10], f32> {

    // ── Input ───────────────────────────────────────────────────────────────
    input x: Tensor<[Batch, 784], f32>;

    // ── Layer 1: Linear(784, 128) + ReLU ────────────────────────────────────
    param w1: Tensor<[128, 784], f32> {
        init: "normal(0, 0.01)";
        frozen: false;
    };

    param b1: Tensor<[1, 128], f32> {
        init: "zeros";
        frozen: false;
    };

    node h1 {
        op: matmul(x, transpose(w1));
    };

    node h1_bias {
        op: add(h1, b1);
    };

    node h1_act {
        op: relu(h1_bias);
    };

    // ── Layer 2: Linear(128, 10) ────────────────────────────────────────────
    param w2: Tensor<[10, 128], f32> {
        init: "normal(0, 0.01)";
        frozen: false;
    };

    param b2: Tensor<[1, 10], f32> {
        init: "zeros";
        frozen: false;
    };

    node logits {
        op: matmul(h1_act, transpose(w2));
    };

    node logits_bias {
        op: add(logits, b2);
    };

    // ── Output ──────────────────────────────────────────────────────────────
    output logits_bias;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration
// ─────────────────────────────────────────────────────────────────────────────

@training {
    model: Forward;
    loss: cross_entropy;

    optimizer: {
        type: "Adam";
        lr: 0.001;
    }

    epochs: 5;
    batch_size: 64;
}
