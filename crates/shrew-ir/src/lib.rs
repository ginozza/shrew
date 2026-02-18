//! # shrew-ir
//!
//! Parser, AST, and IR for the `.sw` (Shrew) declarative model format.
//!
//! This crate handles the full `.sw` pipeline:
//
//   .sw source text ──► Lexer ──► Tokens ──► Parser ──► AST ──► Lowering ──► Graph IR
//
// The AST is a faithful, unvalidated representation of the .sw file.
// The Graph IR is a validated DAG representation ready for optimization
// and execution scheduling.
//
// USAGE:
//   // Parse and lower a .sw file:
//   let ast = shrew_ir::parse(source_text)?;
//   let ir = shrew_ir::lower(&ast)?;
//
//   // Inspect the IR:
//   for graph in &ir.graphs {
//       println!("{}", graph.dump());
//   }

pub mod ast;
pub mod error;
pub mod graph;
pub mod lexer;
pub mod lower;
pub mod optimize;
pub mod parser;
pub mod shapes;
pub mod token;
pub mod validate;

pub use ast::Program;
pub use error::{Error, Result};
pub use graph::IrProgram;
pub use lower::lower;
pub use optimize::optimize;
pub use parser::parse;
pub use shapes::infer_shapes;
pub use validate::validate;
