// =============================================================================
// Char-Level GPT — Character-Level Language Model (Shrew)
// =============================================================================
//
// This example trains a small GPT-style transformer on character-level text.
// Given a sequence of characters, the model learns to predict the next one.
//
// Architecture:
//   Token Embedding(vocab_size, d_model)
//   + Positional Embedding(max_seq_len, d_model)
//   → N × TransformerBlock(d_model, num_heads, d_ff, causal=true)
//   → LayerNorm(d_model)
//   → Linear(d_model, vocab_size)  (language modelling head)
//
// Features demonstrated:
//   1. Embedding + positional encoding (differentiable via one-hot matmul)
//   2. Causal (autoregressive) multi-head self-attention
//   3. Stacked TransformerBlocks with pre-norm residual connections
//   4. AdamW optimizer with CosineWarmupLR scheduler
//   5. Gradient clipping (clip_grad_norm)
//   6. Autoregressive text generation with temperature sampling
//   7. Checkpoint save/load
//
// Usage:
//   cargo run --release -p char-gpt-example                       # built-in text
//   cargo run --release -p char-gpt-example -- --text-file t.txt  # custom text
//   cargo run --release -p char-gpt-example -- --generate 200     # generate text
//   cargo run --release -p char-gpt-example -- --epochs 50 --lr 0.001
//
// For fast CPU training, default is a tiny model: d_model=64, 4 heads, 2 layers.
// With the built-in Shakespeare excerpt, you should see coherent patterns after
// ~20-30 epochs.

use shrew::nn::{Embedding, Module};
use shrew::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

struct Config {
    /// Path to a text file for training data (None = built-in text)
    text_file: Option<String>,
    /// Number of training epochs
    epochs: usize,
    /// Learning rate
    lr: f64,
    /// Model dimension
    d_model: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Number of transformer layers
    n_layers: usize,
    /// Feed-forward inner dimension
    d_ff: usize,
    /// Context window (sequence length for training)
    seq_len: usize,
    /// Batch size (number of sequences per update)
    batch_size: usize,
    /// Max gradient norm for clipping
    max_grad_norm: f64,
    /// Number of warmup steps for cosine scheduler
    warmup_steps: usize,
    /// Characters to generate after training (0 = skip generation)
    generate: usize,
    /// Temperature for sampling (higher = more random)
    temperature: f64,
    /// Prompt for text generation
    prompt: String,
    /// Path to save checkpoint after training
    save_path: Option<String>,
    /// Path to load checkpoint before training/generation
    load_path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            text_file: None,
            epochs: 30,
            lr: 1e-3,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
            seq_len: 64,
            batch_size: 16,
            max_grad_norm: 1.0,
            warmup_steps: 50,
            generate: 200,
            temperature: 0.8,
            prompt: String::new(),
            save_path: None,
            load_path: None,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--text-file" => {
                i += 1;
                cfg.text_file = Some(args[i].clone());
            }
            "--epochs" => {
                i += 1;
                cfg.epochs = args[i].parse().expect("invalid --epochs");
            }
            "--lr" => {
                i += 1;
                cfg.lr = args[i].parse().expect("invalid --lr");
            }
            "--d-model" => {
                i += 1;
                cfg.d_model = args[i].parse().expect("invalid --d-model");
            }
            "--n-heads" => {
                i += 1;
                cfg.n_heads = args[i].parse().expect("invalid --n-heads");
            }
            "--n-layers" => {
                i += 1;
                cfg.n_layers = args[i].parse().expect("invalid --n-layers");
            }
            "--d-ff" => {
                i += 1;
                cfg.d_ff = args[i].parse().expect("invalid --d-ff");
            }
            "--seq-len" => {
                i += 1;
                cfg.seq_len = args[i].parse().expect("invalid --seq-len");
            }
            "--batch-size" => {
                i += 1;
                cfg.batch_size = args[i].parse().expect("invalid --batch-size");
            }
            "--max-grad-norm" => {
                i += 1;
                cfg.max_grad_norm = args[i].parse().expect("invalid --max-grad-norm");
            }
            "--warmup" => {
                i += 1;
                cfg.warmup_steps = args[i].parse().expect("invalid --warmup");
            }
            "--generate" => {
                i += 1;
                cfg.generate = args[i].parse().expect("invalid --generate");
            }
            "--temperature" => {
                i += 1;
                cfg.temperature = args[i].parse().expect("invalid --temperature");
            }
            "--prompt" => {
                i += 1;
                cfg.prompt = args[i].clone();
            }
            "--save" => {
                i += 1;
                cfg.save_path = Some(args[i].clone());
            }
            "--load" => {
                i += 1;
                cfg.load_path = Some(args[i].clone());
            }
            "--help" | "-h" => {
                println!("Char-Level GPT training example for Shrew");
                println!();
                println!("Options:");
                println!(
                    "  --text-file <path>    Training text file (default: built-in Shakespeare)"
                );
                println!("  --epochs <n>          Training epochs (default: 30)");
                println!("  --lr <f>              Learning rate (default: 0.001)");
                println!("  --d-model <n>         Model dimension (default: 64)");
                println!("  --n-heads <n>         Attention heads (default: 4)");
                println!("  --n-layers <n>        Transformer layers (default: 2)");
                println!("  --d-ff <n>            FFN inner dim (default: 256)");
                println!("  --seq-len <n>         Context window (default: 64)");
                println!("  --batch-size <n>      Batch size (default: 16)");
                println!("  --max-grad-norm <f>   Gradient clipping norm (default: 1.0)");
                println!("  --warmup <n>          LR warmup steps (default: 50)");
                println!("  --generate <n>        Chars to generate after training (default: 200)");
                println!("  --temperature <f>     Sampling temperature (default: 0.8)");
                println!(
                    "  --prompt <text>       Generation prompt (default: first chars of data)"
                );
                println!("  --save <path>         Save checkpoint after training");
                println!("  --load <path>         Load checkpoint before training/generation");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    cfg
}

// ─────────────────────────────────────────────────────────────────────────────
// Character Tokenizer
// ─────────────────────────────────────────────────────────────────────────────

/// A simple character-level tokenizer.
/// Maps each unique character in the training text to an integer index.
struct CharTokenizer {
    /// char → index
    char_to_idx: std::collections::HashMap<char, usize>,
    /// index → char
    idx_to_char: Vec<char>,
}

impl CharTokenizer {
    /// Build a tokenizer from a text corpus.
    fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text
            .chars()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort(); // deterministic ordering

        let char_to_idx: std::collections::HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        CharTokenizer {
            char_to_idx,
            idx_to_char: chars,
        }
    }

    fn vocab_size(&self) -> usize {
        self.idx_to_char.len()
    }

    /// Encode a string as a vector of token indices.
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    /// Decode a vector of token indices back to a string.
    fn decode(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .filter_map(|&i| self.idx_to_char.get(i))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPT Model
// ─────────────────────────────────────────────────────────────────────────────

/// A small GPT-style language model.
///
/// Architecture:
///   token_emb(vocab_size, d_model) + pos_emb(max_seq_len, d_model)
///   → N × TransformerBlock(d_model, n_heads, d_ff, causal=true)
///   → LayerNorm(d_model)
///   → Linear(d_model, vocab_size)
struct CharGPT {
    /// Token embedding table
    token_emb: Embedding<CpuBackend>,
    /// Positional embedding table
    pos_emb: Embedding<CpuBackend>,
    /// Transformer blocks
    blocks: Vec<TransformerBlock<CpuBackend>>,
    /// Final layer norm
    ln_f: LayerNorm<CpuBackend>,
    /// Language modelling head: d_model → vocab_size
    lm_head: Linear<CpuBackend>,
    /// Model dimension
    d_model: usize,
}

impl CharGPT {
    fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        max_seq_len: usize,
        dev: &CpuDevice,
    ) -> shrew::Result<Self> {
        let dtype = DType::F64;

        let token_emb = Embedding::new(vocab_size, d_model, dtype, dev)?;
        let pos_emb = Embedding::new(max_seq_len, d_model, dtype, dev)?;

        let mut blocks = Vec::new();
        for _ in 0..n_layers {
            blocks.push(TransformerBlock::new(
                d_model, n_heads, d_ff, true, dtype, dev,
            )?);
        }

        let ln_f = LayerNorm::new(d_model, 1e-5, dtype, dev)?;
        let lm_head = Linear::new(d_model, vocab_size, true, dtype, dev)?;

        Ok(CharGPT {
            token_emb,
            pos_emb,
            blocks,
            ln_f,
            lm_head,
            d_model,
        })
    }

    /// Forward pass: token indices [batch, seq] → logits [batch, seq, vocab]
    fn forward(&self, token_ids: &CpuTensor) -> shrew::Result<CpuTensor> {
        let dims = token_ids.dims();
        let batch = dims[0];
        let seq = dims[1];

        // Token embeddings: [batch, seq] → [batch, seq, d_model]
        let tok_emb = self.token_emb.forward(token_ids)?;

        // Positional embeddings: create position indices [1, seq]
        let positions: Vec<f64> = (0..seq).map(|i| i as f64).collect();
        let pos_ids = CpuTensor::from_f64_slice(&positions, (1, seq), DType::F64, &CpuDevice)?;
        let pos_emb = self.pos_emb.forward(&pos_ids)?; // [1, seq, d_model]

        // Combine: token + position embeddings
        let mut x = tok_emb.add(&pos_emb)?; // broadcasts [1,seq,d] to [batch,seq,d]

        // Pass through transformer blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Final layer norm
        x = self.ln_f.forward(&x)?;

        // Language modelling head: [batch, seq, d_model] → [batch*seq, d_model] → [batch*seq, vocab]
        let x_2d = x.reshape((batch * seq, self.d_model))?;
        let logits_2d = self.lm_head.forward(&x_2d)?;

        // Reshape back: [batch, seq, vocab_size]
        let vocab = self.lm_head.out_features();
        logits_2d.reshape((batch, seq, vocab))
    }

    /// Collect all learnable parameters.
    fn parameters(&self) -> Vec<CpuTensor> {
        let mut params = Vec::new();
        params.extend(self.token_emb.parameters());
        params.extend(self.pos_emb.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_f.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    /// Named parameters for checkpoint save/load.
    fn named_parameters(&self) -> Vec<(String, CpuTensor)> {
        let mut named = Vec::new();
        for (i, p) in self.token_emb.parameters().iter().enumerate() {
            named.push((format!("token_emb.{i}"), p.clone()));
        }
        for (i, p) in self.pos_emb.parameters().iter().enumerate() {
            named.push((format!("pos_emb.{i}"), p.clone()));
        }
        for (bi, block) in self.blocks.iter().enumerate() {
            for (i, p) in block.parameters().iter().enumerate() {
                named.push((format!("block{bi}.{i}"), p.clone()));
            }
        }
        for (i, p) in self.ln_f.parameters().iter().enumerate() {
            named.push((format!("ln_f.{i}"), p.clone()));
        }
        for (i, p) in self.lm_head.parameters().iter().enumerate() {
            named.push((format!("lm_head.{i}"), p.clone()));
        }
        named
    }

    fn total_params(&self) -> usize {
        self.parameters().iter().map(|t| t.elem_count()).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Preparation
// ─────────────────────────────────────────────────────────────────────────────

/// Create training batches from encoded text.
///
/// Returns (inputs, targets) where each is [batch_size, seq_len] as f64.
/// For each position, target[i] = input[i+1] (next-character prediction).
fn create_batches(
    data: &[usize],
    seq_len: usize,
    batch_size: usize,
    vocab_size: usize,
) -> shrew::Result<Vec<(CpuTensor, CpuTensor)>> {
    // Total usable positions: we need seq_len+1 chars for each sequence
    let stride = seq_len + 1;
    if data.len() < stride {
        return Err(shrew_core::Error::msg(format!(
            "Text too short: {} chars, need at least {} (seq_len+1)",
            data.len(),
            stride
        )));
    }

    // Collect all possible sequences
    let num_sequences = data.len() - seq_len;

    // Batch them up
    let mut batches = Vec::new();
    let mut offset = 0;

    while offset + batch_size <= num_sequences {
        let mut input_data = Vec::with_capacity(batch_size * seq_len);
        let mut target_data = Vec::with_capacity(batch_size * seq_len * vocab_size);

        for b in 0..batch_size {
            let start = offset + b;
            // Input: chars at positions [start, start+seq_len)
            for t in 0..seq_len {
                input_data.push(data[start + t] as f64);
            }
            // Target: chars at positions [start+1, start+seq_len+1) — one-hot encoded
            for t in 0..seq_len {
                let target_idx = data[start + t + 1];
                for c in 0..vocab_size {
                    target_data.push(if c == target_idx { 1.0 } else { 0.0 });
                }
            }
        }

        let input_tensor =
            CpuTensor::from_f64_slice(&input_data, (batch_size, seq_len), DType::F64, &CpuDevice)?;

        // Target: [batch_size * seq_len, vocab_size] for cross_entropy_loss
        let target_tensor = CpuTensor::from_f64_slice(
            &target_data,
            (batch_size * seq_len, vocab_size),
            DType::F64,
            &CpuDevice,
        )?;

        batches.push((input_tensor, target_tensor));
        offset += batch_size; // non-overlapping batches for diversity
    }

    Ok(batches)
}

// ─────────────────────────────────────────────────────────────────────────────
// Text Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate text autoregressively from a prompt.
///
/// At each step:
/// 1. Feed current context through the model → logits for next token
/// 2. Apply temperature scaling
/// 3. Sample from the probability distribution
/// 4. Append sampled token and repeat
fn generate(
    model: &CharGPT,
    tokenizer: &CharTokenizer,
    prompt: &str,
    num_chars: usize,
    temperature: f64,
    max_seq_len: usize,
) -> shrew::Result<String> {
    let mut context: Vec<usize> = tokenizer.encode(prompt);
    if context.is_empty() {
        // Start with the first token in vocabulary
        context.push(0);
    }

    let mut rng_state: u64 = 42; // simple xorshift PRNG

    for _ in 0..num_chars {
        // Truncate context to max_seq_len
        let start = if context.len() > max_seq_len {
            context.len() - max_seq_len
        } else {
            0
        };
        let ctx = &context[start..];
        let ctx_len = ctx.len();

        // Create input tensor [1, ctx_len]
        let input_data: Vec<f64> = ctx.iter().map(|&i| i as f64).collect();
        let input = CpuTensor::from_f64_slice(&input_data, (1, ctx_len), DType::F64, &CpuDevice)?;

        // Forward pass → [1, ctx_len, vocab_size]
        let logits = model.forward(&input)?;

        // Take logits for the last position: [1, 1, vocab] via narrow
        let last_logits = logits.narrow(1, ctx_len - 1, 1)?; // [1, 1, vocab]
        let vocab_size = tokenizer.vocab_size();
        let last_logits_2d = last_logits.reshape((1, vocab_size))?; // [1, vocab]

        // Apply temperature and get probabilities
        let logits_data = last_logits_2d.to_f64_vec()?;
        let scaled: Vec<f64> = logits_data.iter().map(|&v| v / temperature).collect();

        // Softmax manually (for sampling — no autograd needed)
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f64 = exp_vals.iter().sum();
        let probs: Vec<f64> = exp_vals.iter().map(|&v| v / sum_exp).collect();

        // Sample from the distribution using xorshift64
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let rand_val = (rng_state as f64) / (u64::MAX as f64);

        let mut cumulative = 0.0;
        let mut sampled_idx = vocab_size - 1;
        for (idx, &p) in probs.iter().enumerate() {
            cumulative += p;
            if rand_val < cumulative {
                sampled_idx = idx;
                break;
            }
        }

        context.push(sampled_idx);
    }

    Ok(tokenizer.decode(&context))
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in Training Text
// ─────────────────────────────────────────────────────────────────────────────

/// A short Shakespeare excerpt for quick demo training.
const DEFAULT_TEXT: &str = "\
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even to the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city is risen:
why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?
";

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> shrew::Result<()> {
    let cfg = parse_args();
    let dev = CpuDevice;

    println!("=== Shrew — Char-Level GPT ===");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 1. Load text data
    // ─────────────────────────────────────────────────────────────────────
    let text = match &cfg.text_file {
        Some(path) => {
            println!("Loading text from: {path}");
            std::fs::read_to_string(path)
                .map_err(|e| shrew_core::Error::msg(format!("Failed to read {path}: {e}")))?
        }
        None => {
            println!(
                "Using built-in Shakespeare excerpt ({} chars)",
                DEFAULT_TEXT.len()
            );
            println!("  Tip: use --text-file <path> for custom training text");
            DEFAULT_TEXT.to_string()
        }
    };

    // ─────────────────────────────────────────────────────────────────────
    // 2. Build tokenizer and encode text
    // ─────────────────────────────────────────────────────────────────────
    let tokenizer = CharTokenizer::from_text(&text);
    let encoded = tokenizer.encode(&text);
    let vocab_size = tokenizer.vocab_size();

    println!("  Vocabulary: {} unique characters", vocab_size);
    println!("  Encoded length: {} tokens", encoded.len());
    println!(
        "  Characters: {:?}",
        tokenizer.idx_to_char.iter().collect::<String>()
    );
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 3. Create training batches
    // ─────────────────────────────────────────────────────────────────────
    let batches = create_batches(&encoded, cfg.seq_len, cfg.batch_size, vocab_size)?;
    let num_batches = batches.len();
    let total_steps = cfg.epochs * num_batches;

    println!("Training data:");
    println!("  Sequence length: {}", cfg.seq_len);
    println!("  Batch size: {}", cfg.batch_size);
    println!("  Batches per epoch: {}", num_batches);
    println!("  Total training steps: {}", total_steps);
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 4. Build model
    // ─────────────────────────────────────────────────────────────────────
    let model = CharGPT::new(
        vocab_size,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers,
        cfg.d_ff,
        cfg.seq_len,
        &dev,
    )?;

    println!("Model:");
    println!(
        "  d_model={}, n_heads={}, n_layers={}, d_ff={}",
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff
    );
    println!("  Max sequence length: {}", cfg.seq_len);
    println!("  Vocab size: {}", vocab_size);
    println!("  Total parameters: {}", model.total_params());
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 5. Set up optimizer and scheduler
    // ─────────────────────────────────────────────────────────────────────
    let params = model.parameters();
    let mut optimizer = AdamW::<CpuBackend>::new(params, cfg.lr, 0.01);
    let mut scheduler =
        CosineWarmupLR::new(cfg.lr, cfg.warmup_steps as u64, total_steps as u64, 1e-5);

    // Load pre-trained weights if requested
    if let Some(ref load_path) = cfg.load_path {
        println!("Loading weights from: {load_path}");
        let loaded = shrew::checkpoint::load_tensors::<CpuBackend>(load_path, &dev)?;
        let param_names = model.named_parameters();
        let params = optimizer.params_mut();
        for (name, tensor) in &loaded {
            if let Some(idx) = param_names.iter().position(|(n, _)| n == name) {
                params[idx] = tensor.clone().set_variable();
            }
        }
        println!("  Loaded {} parameters", loaded.len());
        println!();
    }

    println!("Optimizer: AdamW (lr={}, weight_decay=0.01)", cfg.lr);
    println!(
        "Scheduler: CosineWarmup (warmup={}, total={})",
        cfg.warmup_steps, total_steps
    );
    println!("Gradient clipping: max_norm={}", cfg.max_grad_norm);
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 6. Training loop
    // ─────────────────────────────────────────────────────────────────────
    println!("Training for {} epochs...", cfg.epochs);
    println!("{:-<70}", "");

    for epoch in 0..cfg.epochs {
        let mut epoch_loss = 0.0;

        for (batch_idx, (input, target)) in batches.iter().enumerate() {
            // Forward: [batch, seq_len] → [batch, seq_len, vocab]
            let logits = model.forward(input)?;

            // Reshape logits to [batch*seq_len, vocab] for cross_entropy_loss
            let batch_size = input.dims()[0];
            let logits_2d = logits.reshape((batch_size * cfg.seq_len, vocab_size))?;

            // Loss: cross_entropy(logits_2d, target_onehot)
            let loss = cross_entropy_loss(&logits_2d, target)?;
            let loss_val = loss.to_scalar_f64()?;
            epoch_loss += loss_val;

            // Backward
            let grads = loss.backward()?;

            // Gradient clipping
            let (clipped_grads, _grad_norm) =
                clip_grad_norm(&grads, optimizer.params(), cfg.max_grad_norm)?;

            // Optimizer step with clipped gradients
            optimizer.step(&clipped_grads)?;

            // LR scheduler step
            let new_lr = scheduler.step();
            optimizer.set_lr(new_lr);

            // Progress
            if (batch_idx + 1) % 5 == 0 || batch_idx + 1 == num_batches {
                print!(
                    "\r  Epoch {}/{} | Batch {}/{} | Loss: {:.4} | LR: {:.6}",
                    epoch + 1,
                    cfg.epochs,
                    batch_idx + 1,
                    num_batches,
                    epoch_loss / (batch_idx + 1) as f64,
                    scheduler.current_lr()
                );
            }
        }

        let avg_loss = epoch_loss / num_batches as f64;
        println!(
            "\r  Epoch {}/{} | Loss: {:.4} | LR: {:.6}              ",
            epoch + 1,
            cfg.epochs,
            avg_loss,
            scheduler.current_lr()
        );

        // Sample generation every 10 epochs for progress check
        if (epoch + 1) % 10 == 0 && cfg.generate > 0 {
            let sample_prompt = if cfg.prompt.is_empty() {
                &text[..text.len().min(10)]
            } else {
                &cfg.prompt
            };
            let sample = generate(
                &model,
                &tokenizer,
                sample_prompt,
                50,
                cfg.temperature,
                cfg.seq_len,
            )?;
            println!(
                "    Sample: \"{}\"",
                sample.chars().take(80).collect::<String>()
            );
        }
    }

    println!("{:-<70}", "");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 7. Save checkpoint
    // ─────────────────────────────────────────────────────────────────────
    if let Some(ref save_path) = cfg.save_path {
        let named = model.named_parameters();
        shrew::checkpoint::save_tensors(save_path, &named)?;
        println!("Saved {} parameters to: {save_path}", named.len());
        println!();
    }

    // ─────────────────────────────────────────────────────────────────────
    // 8. Generate text
    // ─────────────────────────────────────────────────────────────────────
    if cfg.generate > 0 {
        let prompt = if cfg.prompt.is_empty() {
            text[..text.len().min(10)].to_string()
        } else {
            cfg.prompt.clone()
        };

        println!(
            "Generating {} characters (temperature={})...",
            cfg.generate, cfg.temperature
        );
        println!("Prompt: \"{}\"", prompt);
        println!("{:-<70}", "");

        let generated = generate(
            &model,
            &tokenizer,
            &prompt,
            cfg.generate,
            cfg.temperature,
            cfg.seq_len,
        )?;

        println!("{}", generated);
        println!("{:-<70}", "");
    }

    println!();
    println!("=== Done! ===");

    Ok(())
}
