// =============================================================================
// Executor — Runs .sw programs on the Shrew tensor runtime
// =============================================================================
//
// This module bridges the IR pipeline and the tensor runtime. It takes a
// lowered, validated, optimized IrProgram and executes it:
//
//   .sw file → parse → lower → validate → optimize → **execute**
//
// The executor walks the computation graph in topological order, dispatching
// each node's OpKind to the corresponding tensor operation, and manages
// parameter initialization, forward passes, and training loops.
//
// USAGE:
//   let program = shrew_ir::parse(source)?;
//   let mut ir = shrew_ir::lower(&program)?;
//   shrew_ir::validate(&ir)?;
//   shrew_ir::infer_shapes(&mut ir);
//   shrew_ir::optimize(&mut ir);
//
//   let mut exec = Executor::<CpuBackend>::new(ir, CpuDevice, RuntimeConfig::default())?;
//   let outputs = exec.run("Forward", &inputs)?;

mod engine;
pub mod jit;
mod train;

pub use engine::{ExecResult, Executor, RuntimeConfig};
pub use jit::{
    compile_graph, load_jit, CompileStats, CompiledGraph, Instruction, JitExecutor, JitResult,
    MemoryPlan,
};
pub use train::{load_program, load_trainer, EpochLog, TrainResult, Trainer};
