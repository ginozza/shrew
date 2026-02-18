# IR & Lowering

The **Intermediate Representation (IR)** is a directed acyclic graph (DAG) where nodes represent operations and edges represent data dependencies (tensors).

## Lowering Process

The lowering phase converts the AST (which mirrors the syntax tree) into the Graph IR. Key steps include:

1. **Symbol Resolution**: Linking identifiers to their definitions.
2. **Type Checking**: Verifying tensor shapes and data types.
3. **Graph Construction**: Building the node connectivity.

The IR is defined in `crates/shrew-ir/src/graph.rs`.
