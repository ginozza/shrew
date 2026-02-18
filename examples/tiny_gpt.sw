// TinyGPT — A small GPT-style language model defined in Shrew IR (.sw)
//
// This file demonstrates the full .sw format: metadata, config, types,
// graph definition, training, inference, metrics, logging, and visualizations.

@import "layers/transformer.sw" as tfm;

// Model metadata

@model {
    name: "TinyGPT";
    version: "0.1.0";
    author: "Shrew Contributors";
}

// Hyperparameters and constants

@config {
    vocab_size: 50257;
    d_model: 256;
    n_heads: 4;
    n_layers: 4;
    d_ff: 256 * 4;
    max_seq_len: 512;
    dropout: 0.1;
    eps: 1e-5;
}

// Named tensor types for documentation & validation

@types {
    type TokenIds   = Tensor<[Batch, SeqLen], i64>;
    type Embeddings = Tensor<[Batch, SeqLen, 256], f32>;
    type Hidden     = Tensor<[Batch, SeqLen, 256], f32>;
    type Logits     = Tensor<[Batch, 50257], f32>;
}

// Main computation graph

@graph Forward(tokens: Tensor<[Batch, SeqLen], i64>) -> Tensor<[Batch, 50257], f32> {

    //  Inputs 
    input tokens: Tensor<[Batch, SeqLen], i64>;

    //  Learnable parameters 
    param wte: Tensor<[50257, 256], f32> {
        init: "normal(0, 0.02)";
        frozen: false;
    };

    param wpe: Tensor<[512, 256], f32> {
        init: "normal(0, 0.02)";
        frozen: false;
    };

    param ln_f_weight: Tensor<[256], f32> {
        init: "ones";
        frozen: false;
    };

    param ln_f_bias: Tensor<[256], f32> {
        init: "zeros";
        frozen: false;
    };

    //  Token + positional embedding 
    node tok_emb {
        op: embedding(tokens, wte);
    };

    node positions {
        op: range(0, SeqLen);
    };

    node pos_emb {
        op: embedding(positions, wpe);
    };

    node h {
        op: tok_emb + pos_emb;
    };

    //  Transformer blocks (repeated n_layers times) 
    node transformer_out {
        op: repeat(4) { transformer_block(h, n_heads: 4) };
        @hint recompute_in_backward;
    };

    //  Final layernorm 
    node ln_out {
        op: layer_norm(transformer_out, ln_f_weight, ln_f_bias, eps: 1e-5);
    };

    //  Language model head (weight tying with wte) 
    node logits {
        op: matmul(ln_out, transpose(wte));
    };

    //  Dimension assertions 
    @assert Batch > 0, "batch size must be positive";
    @assert SeqLen <= 512, "sequence length must not exceed max_seq_len";

    //  Output 
    output logits;
}

// Transformer block as a custom op

@custom_op transformer_block {
    signature: (x: Tensor<[B, S, D], f32>, n_heads: i32) -> Tensor<[B, S, D], f32>;

    impl cpu {
        kernel: "transformer_block_cpu";
    }

    gradient backward {
        impl cpu {
            kernel: "transformer_block_grad_cpu";
        }
    }
}

// Training configuration

@training {
    model: Forward;
    loss: cross_entropy;

    optimizer: {
        type: "AdamW";
        lr: 3e-4;
        weight_decay: 0.1;
        beta1: 0.9;
        beta2: 0.95;
    }

    lr_schedule: {
        type: "cosine";
        warmup_steps: 500;
        min_lr: 1e-5;
    }

    grad_clip: {
        type: "norm";
        max_norm: 1.0;
    }

    precision: "bf16";
    epochs: 20;
    batch_size: 64;
    accumulation_steps: 4;
}


// Inference configuration


@inference {
    model: Forward;

    quantization: {
        mode: "int8";
        calibration: "dynamic";
    }

    generation: {
        strategy: "top_p";
        temperature: 0.9;
        top_p: 0.95;
        max_tokens: 256;
    }
}

// Metrics tracking

@metrics TrainingMetrics {
    track train_loss {
        source: loss;
        aggregate: "ema";
        log_every: 10;
    }

    track grad_norm {
        source: gradients;
        compute: norm;
        aggregate: "mean";
        log_every: 10;
    }

    track learning_rate {
        source: lr;
        aggregate: "last";
        log_every: 50;
    }
}

// Logging configuration

@logging {
    backend: "tensorboard";
    log_dir: "./runs/tiny_gpt";
    flush_secs: 30;
}

// Visualizations

@visualizations {
    plot loss_curve {
        x: "step";
        y: "train_loss";
        title: "Training Loss";
    }

    plot lr_schedule {
        x: "step";
        y: "learning_rate";
        title: "Learning Rate Schedule";
    }
}
