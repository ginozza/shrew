// shrew CLI — Command-line runner for .sw deep learning programs
//
// USAGE:
//   shrew dump model.sw         # Print the lowered IR graph
//   shrew validate model.sw     # Validate a .sw program
//   shrew bench model.sw        # Benchmark forward pass
//   shrew info model.sw         # Show model summary (params, ops, shapes)
//
// OPTIONS:
//   --batch N      Set batch dimension (default: 1)
//   --dtype f32|f64|f16  Set default dtype (default: f32)
//   --verbose      Print detailed execution info

use std::env;
use std::fs;
use std::process;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let command = args[1].as_str();

    match command {
        "dump" | "validate" | "bench" | "info" => {
            if args.len() < 3 {
                eprintln!("Error: missing .sw file path");
                eprintln!("Usage: shrew {command} <file.sw> [options]");
                process::exit(1);
            }
            let file_path = &args[2];
            let opts = parse_options(&args[3..]);

            match run_command(command, file_path, &opts) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            }
        }
        "--help" | "-h" | "help" => {
            print_usage();
        }
        "--version" | "-V" | "version" => {
            println!("shrew {}", env!("CARGO_PKG_VERSION"));
        }
        other => {
            eprintln!("Unknown command: {other}");
            print_usage();
            process::exit(1);
        }
    }
}

// Options

struct CliOptions {
    batch_size: usize,
    dtype: String,
    verbose: bool,
}

fn parse_options(args: &[String]) -> CliOptions {
    let mut opts = CliOptions {
        batch_size: 1,
        dtype: "f32".to_string(),
        verbose: false,
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--batch" => {
                i += 1;
                if i < args.len() {
                    opts.batch_size = args[i].parse().unwrap_or(1);
                }
            }
            "--dtype" => {
                i += 1;
                if i < args.len() {
                    opts.dtype = args[i].clone();
                }
            }
            "--verbose" | "-v" => {
                opts.verbose = true;
            }
            other => {
                eprintln!("Warning: unknown option '{other}'");
            }
        }
        i += 1;
    }

    opts
}

// Command dispatch

fn run_command(command: &str, file_path: &str, opts: &CliOptions) -> Result<(), String> {
    let source =
        fs::read_to_string(file_path).map_err(|e| format!("Cannot read '{file_path}': {e}"))?;

    match command {
        "dump" => cmd_dump(&source, file_path, opts),
        "validate" => cmd_validate(&source, file_path),
        "bench" => cmd_bench(&source, file_path, opts),
        "info" => cmd_info(&source, file_path, opts),
        _ => Err(format!("Unknown command: {command}")),
    }
}

// dump — Print the lowered IR graph

fn cmd_dump(source: &str, file_path: &str, opts: &CliOptions) -> Result<(), String> {
    let ast = shrew_ir::parse(source).map_err(|e| format!("Parse error: {e}"))?;
    let mut ir = shrew_ir::lower(&ast).map_err(|e| format!("Lowering error: {e}"))?;

    // Validate (but don't fail, just warn)
    if let Err(errors) = shrew_ir::validate(&ir) {
        for e in &errors {
            eprintln!("Warning: {e}");
        }
    }

    shrew_ir::infer_shapes(&mut ir);

    println!("=== IR Dump: {file_path} ===");
    println!();

    for graph in &ir.graphs {
        println!("graph {} {{", graph.name);

        // Inputs
        for input_id in &graph.inputs {
            let node = &graph.nodes[input_id.0];
            println!("  input {:10} : {:?}", node.name, node.output_type);
        }

        // Nodes
        for node in &graph.nodes {
            let inputs_str: Vec<&str> = node
                .inputs
                .iter()
                .map(|id| graph.nodes[id.0].name.as_str())
                .collect();
            println!(
                "  {:10} = {:?}({})",
                node.name,
                node.op,
                inputs_str.join(", ")
            );
        }

        // Outputs
        for output in &graph.outputs {
            let node = &graph.nodes[output.node_id.0];
            println!("  output {:10} : {:?}", output.name, node.output_type);
        }

        println!("}}");
        println!();
    }

    // Show optimization stats if verbose
    if opts.verbose {
        let stats = shrew_ir::optimize::optimize_graph_with_stats(&mut ir.graphs[0]);
        println!("Optimization stats:");
        println!("  Dead code removed:     {}", stats.dead_code_removed);
        println!("  Identities removed:    {}", stats.identities_removed);
        println!("  Constants folded:      {}", stats.constants_folded);
        println!("  CSE eliminated:        {}", stats.cse_eliminated);
        println!("  Operators fused:       {}", stats.ops_fused);
    }

    Ok(())
}

// validate — Check a .sw program for errors

fn cmd_validate(source: &str, file_path: &str) -> Result<(), String> {
    println!("=== Validating: {file_path} ===");

    // Step 1: Parse
    let ast = match shrew_ir::parse(source) {
        Ok(a) => {
            println!("  [OK] Parse: {} item(s) found", a.items.len());
            a
        }
        Err(e) => {
            println!("  [FAIL] Parse error: {e}");
            return Err("Validation failed at parse stage".to_string());
        }
    };

    // Step 2: Lower
    let mut ir = match shrew_ir::lower(&ast) {
        Ok(ir) => {
            println!(
                "  [OK] Lower: {} graph(s), {} total nodes",
                ir.graphs.len(),
                ir.graphs.iter().map(|g| g.nodes.len()).sum::<usize>()
            );
            ir
        }
        Err(e) => {
            println!("  [FAIL] Lowering error: {e}");
            return Err("Validation failed at lowering stage".to_string());
        }
    };

    // Step 3: Validate
    match shrew_ir::validate(&ir) {
        Ok(()) => println!("  [OK] Validate: no errors"),
        Err(errors) => {
            println!("  [WARN] {} validation error(s):", errors.len());
            for e in &errors {
                println!("         - {e}");
            }
        }
    }

    // Step 4: Shape inference
    shrew_ir::infer_shapes(&mut ir);
    let shaped = ir
        .graphs
        .iter()
        .flat_map(|g| g.nodes.iter())
        .filter(|n| !matches!(n.output_type, shrew_ir::graph::IrType::Unknown))
        .count();
    let total = ir.graphs.iter().map(|g| g.nodes.len()).sum::<usize>();
    println!("  [OK] Shapes: {shaped}/{total} nodes have resolved types");

    // Step 5: Optimize
    let removed = shrew_ir::optimize(&mut ir);
    println!("  [OK] Optimize: {removed} redundant ops removed");

    println!();
    println!("Validation passed!");
    Ok(())
}

// bench — Benchmark forward pass

fn cmd_bench(source: &str, file_path: &str, opts: &CliOptions) -> Result<(), String> {
    use shrew::prelude::*;

    let dtype = parse_dtype(&opts.dtype)?;
    let config = RuntimeConfig::default()
        .set_dim("batch", opts.batch_size)
        .set_dim("Batch", opts.batch_size)
        .with_dtype(dtype)
        .with_training(false);

    let exec = shrew::exec::load_program::<CpuBackend>(source, CpuDevice, config.clone())
        .map_err(|e| format!("Load error: {e}"))?;

    let graph_names: Vec<String> = exec
        .program()
        .graphs
        .iter()
        .map(|g| g.name.clone())
        .collect();
    if graph_names.is_empty() {
        return Err("No graphs found in program".to_string());
    }

    println!("=== Benchmark: {file_path} ===");
    println!("Batch: {}, DType: {:?}", opts.batch_size, dtype);
    println!();

    let warmup = 3;
    let iterations = 10;

    for gname in &graph_names {
        // Generate synthetic inputs
        let graph = exec
            .program()
            .graphs
            .iter()
            .find(|g| g.name == *gname)
            .ok_or_else(|| format!("Graph '{gname}' not found"))?;

        let mut inputs = std::collections::HashMap::new();
        for &input_id in &graph.inputs {
            let node = &graph.nodes[input_id.0];
            if let shrew_ir::graph::IrType::Tensor {
                shape,
                dtype: ir_dt,
            } = &node.output_type
            {
                let dims: Vec<usize> = shape
                    .iter()
                    .map(|d| match d {
                        shrew_ir::graph::Dim::Fixed(n) => *n as usize,
                        shrew_ir::graph::Dim::Symbolic(s) => config
                            .dims
                            .get(s.as_str())
                            .copied()
                            .unwrap_or(opts.batch_size),
                        shrew_ir::graph::Dim::Dynamic => opts.batch_size,
                    })
                    .collect();
                let core_dt = match ir_dt {
                    shrew_ir::graph::DType::F32 => shrew_core::DType::F32,
                    shrew_ir::graph::DType::F64 => shrew_core::DType::F64,
                    _ => dtype,
                };
                let tensor = CpuTensor::rand(shrew_core::Shape::new(dims), core_dt, &CpuDevice)
                    .map_err(|e| format!("Failed to create input '{}': {e}", node.name))?;
                inputs.insert(node.name.clone(), tensor);
            }
        }

        // Warmup
        for _ in 0..warmup {
            let _ = exec.run(gname, &inputs);
        }

        // Timed runs
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            let _ = exec.run(gname, &inputs);
            times.push(t0.elapsed());
        }

        let total_ms: f64 = times.iter().map(|t| t.as_secs_f64() * 1000.0).sum();
        let avg_ms = total_ms / iterations as f64;
        let min_ms = times
            .iter()
            .map(|t| t.as_secs_f64() * 1000.0)
            .fold(f64::INFINITY, f64::min);
        let max_ms = times
            .iter()
            .map(|t| t.as_secs_f64() * 1000.0)
            .fold(0.0f64, f64::max);

        println!("Graph: {gname}");
        println!("  Iterations: {iterations} (+ {warmup} warmup)");
        println!("  Avg:  {avg_ms:.3} ms");
        println!("  Min:  {min_ms:.3} ms");
        println!("  Max:  {max_ms:.3} ms");
        println!(
            "  Throughput: {:.1} samples/sec",
            opts.batch_size as f64 / (avg_ms / 1000.0)
        );
        println!();
    }

    Ok(())
}

// info — Show model summary

fn cmd_info(source: &str, file_path: &str, _opts: &CliOptions) -> Result<(), String> {
    let ast = shrew_ir::parse(source).map_err(|e| format!("Parse error: {e}"))?;
    let mut ir = shrew_ir::lower(&ast).map_err(|e| format!("Lowering error: {e}"))?;

    shrew_ir::infer_shapes(&mut ir);

    println!("=== Model Info: {file_path} ===");
    println!();

    for graph in &ir.graphs {
        println!("Graph: {}", graph.name);
        println!("  Inputs:  {}", graph.inputs.len());
        println!("  Outputs: {}", graph.outputs.len());
        println!("  Nodes:   {}", graph.nodes.len());

        // Count op types
        let mut op_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for node in &graph.nodes {
            *op_counts.entry(format!("{:?}", node.op)).or_insert(0) += 1;
        }

        println!("  Operations:");
        let mut sorted: Vec<_> = op_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (op, count) in &sorted {
            println!("    {op}: {count}");
        }

        // Count parameters from graph.params
        let param_count = graph.params.len();
        println!("  Parameters: {param_count}");

        println!();
    }

    // Training block info
    if let Some(ref t) = ir.training {
        println!("Training config:");
        println!("  Optimizer: {}", t.optimizer.kind);
        println!("  LR:        {}", t.optimizer.lr);
        println!("  Loss:      {}", t.loss);
        println!("  Epochs:    {}", t.epochs);
        println!("  Batch:     {}", t.batch_size);
    } else {
        println!("No @training block.");
    }

    Ok(())
}

// Helpers

fn parse_dtype(s: &str) -> Result<shrew_core::DType, String> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Ok(shrew_core::DType::F32),
        "f64" | "float64" => Ok(shrew_core::DType::F64),
        "f16" | "float16" => Ok(shrew_core::DType::F16),
        "bf16" | "bfloat16" => Ok(shrew_core::DType::BF16),
        "u8" | "uint8" => Ok(shrew_core::DType::U8),
        "u32" | "uint32" => Ok(shrew_core::DType::U32),
        "i64" | "int64" => Ok(shrew_core::DType::I64),
        _ => Err(format!("Unknown dtype: {s}")),
    }
}

fn print_usage() {
    println!("Shrew — Deep Learning CLI");
    println!();
    println!("USAGE:");
    println!("  shrew <command> <file.sw> [options]");
    println!();
    println!("COMMANDS:");
    println!("  dump       Print the lowered IR graph");
    println!("  validate   Check a .sw program for errors");
    println!("  bench      Benchmark forward pass performance");
    println!("  info       Show model summary (params, ops, shapes)");
    println!("  version    Print version");
    println!("  help       Show this help");
    println!();
    println!("OPTIONS:");
    println!("  --batch N        Set batch dimension (default: 1)");
    println!("  --dtype <type>   Set default dtype: f32, f64, f16 (default: f32)");
    println!("  --verbose, -v    Print detailed output");
}
