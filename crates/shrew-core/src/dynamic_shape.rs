// Dynamic Shapes — Symbolic shape tracking and runtime shape resolution
//
// Deep learning models often have dimensions that vary at runtime:
//   - Batch size: different in training vs inference, or varies each step
//   - Sequence length: varies per input in NLP models
//   - Image size: may differ with data augmentation
//
// This module provides a bridge between:
//   - The IR's symbolic shape system (Dim::Symbolic, Dim::Dynamic)
//   - The runtime's concrete shape system (Shape = Vec<usize>)
//
// COMPONENTS:
//
//   SymDim            — A dimension that can be fixed, symbolic, or dynamic
//   SymbolicShape     — A shape pattern with mixed fixed/symbolic/dynamic dims
//   ShapeEnv          — Environment mapping symbolic names → concrete values
//   ShapeGuard        — Validates concrete tensors against shape patterns
//
// WORKFLOW:
//
//   1. Define model shapes with symbolic dims: [Batch, SeqLen, 768]
//   2. At runtime, bind symbolic dims: { Batch → 32, SeqLen → 128 }
//   3. Resolve symbolic shapes to concrete: [32, 128, 768]
//   4. Or validate: does this concrete tensor match the expected pattern?
//
// EXAMPLES:
//
//   let pattern = SymbolicShape::new(vec![
//       SymDim::Symbolic("Batch".into()),
//       SymDim::Fixed(784),
//   ]);
//   let mut env = ShapeEnv::new();
//   env.bind("Batch", 32);
//   let concrete = pattern.resolve(&env)?; // Shape([32, 784])

use std::collections::HashMap;
use std::fmt;

use crate::error::{Error, Result};
use crate::shape::Shape;

// SymDim — A single dimension that can be fixed, named, or dynamic

/// A dimension that can be concrete, symbolic, or fully dynamic.
///
/// This is the runtime-level counterpart to the IR's `Dim` enum, but
/// designed for use with actual tensor operations and shape validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymDim {
    /// Known at compile time: 768, 50257, etc.
    Fixed(usize),
    /// Named symbolic dimension: "Batch", "SeqLen", "HiddenDim"
    /// Resolved to a concrete value at runtime via ShapeEnv.
    Symbolic(String),
    /// Fully dynamic — matches any concrete value.
    /// Used when a dimension is truly unknown until runtime.
    Dynamic,
}

impl SymDim {
    /// Create a fixed dimension.
    pub fn fixed(n: usize) -> Self {
        SymDim::Fixed(n)
    }

    /// Create a named symbolic dimension.
    pub fn symbolic(name: impl Into<String>) -> Self {
        SymDim::Symbolic(name.into())
    }

    /// Create a dynamic (wildcard) dimension.
    pub fn dynamic() -> Self {
        SymDim::Dynamic
    }

    /// Is this a concrete (fixed) dimension?
    pub fn is_fixed(&self) -> bool {
        matches!(self, SymDim::Fixed(_))
    }

    /// Is this a symbolic (named) dimension?
    pub fn is_symbolic(&self) -> bool {
        matches!(self, SymDim::Symbolic(_))
    }

    /// Is this fully dynamic?
    pub fn is_dynamic(&self) -> bool {
        matches!(self, SymDim::Dynamic)
    }

    /// Try to resolve this dimension to a concrete value.
    ///
    /// - Fixed: returns the value directly
    /// - Symbolic: looks up the name in the environment
    /// - Dynamic: returns None (cannot resolve without a concrete value)
    pub fn resolve(&self, env: &ShapeEnv) -> Option<usize> {
        match self {
            SymDim::Fixed(n) => Some(*n),
            SymDim::Symbolic(name) => env.get(name),
            SymDim::Dynamic => None,
        }
    }

    /// Check if a concrete value matches this dimension pattern.
    ///
    /// - Fixed(n): value must equal n
    /// - Symbolic: checks env if bound, otherwise matches any value
    /// - Dynamic: matches any value
    pub fn matches(&self, value: usize, env: &ShapeEnv) -> bool {
        match self {
            SymDim::Fixed(n) => value == *n,
            SymDim::Symbolic(name) => {
                if let Some(bound) = env.get(name) {
                    value == bound
                } else {
                    true // unbound symbolic matches anything
                }
            }
            SymDim::Dynamic => true,
        }
    }

    /// Try to unify this dimension with a concrete value, potentially
    /// binding a symbolic name in the environment.
    ///
    /// Returns true if unification succeeds.
    pub fn unify(&self, value: usize, env: &mut ShapeEnv) -> bool {
        match self {
            SymDim::Fixed(n) => value == *n,
            SymDim::Symbolic(name) => {
                if let Some(bound) = env.get(name) {
                    value == bound
                } else {
                    env.bind(name, value);
                    true
                }
            }
            SymDim::Dynamic => true,
        }
    }
}

impl fmt::Display for SymDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymDim::Fixed(n) => write!(f, "{n}"),
            SymDim::Symbolic(s) => write!(f, "{s}"),
            SymDim::Dynamic => write!(f, "?"),
        }
    }
}

impl From<usize> for SymDim {
    fn from(n: usize) -> Self {
        SymDim::Fixed(n)
    }
}

impl From<&str> for SymDim {
    fn from(s: &str) -> Self {
        SymDim::Symbolic(s.to_string())
    }
}

// SymbolicShape — A shape pattern with mixed fixed/symbolic/dynamic dims

/// A shape pattern that can contain fixed, symbolic, and dynamic dimensions.
///
/// Think of it as a "shape template" that can be resolved to a concrete
/// `Shape` once all symbolic dimensions are bound.
///
/// # Examples
/// ```ignore
/// // Define a shape pattern for transformer input
/// let pattern = SymbolicShape::from(vec![
///     SymDim::symbolic("Batch"),
///     SymDim::symbolic("SeqLen"),
///     SymDim::fixed(768),
/// ]);
///
/// // Resolve with concrete bindings
/// let mut env = ShapeEnv::new();
/// env.bind("Batch", 32);
/// env.bind("SeqLen", 128);
/// let concrete = pattern.resolve(&env)?; // Shape([32, 128, 768])
///
/// // Or validate a concrete tensor shape
/// let guard = ShapeGuard::new(pattern);
/// assert!(guard.validate_shape(&Shape::new(vec![32, 128, 768]), &env));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicShape {
    dims: Vec<SymDim>,
}

impl SymbolicShape {
    /// Create a new symbolic shape from a vector of SymDim.
    pub fn new(dims: Vec<SymDim>) -> Self {
        Self { dims }
    }

    /// Create a fully-fixed symbolic shape from a concrete shape.
    pub fn from_shape(shape: &Shape) -> Self {
        Self {
            dims: shape.dims().iter().map(|&d| SymDim::Fixed(d)).collect(),
        }
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Get the dimension patterns.
    pub fn dims(&self) -> &[SymDim] {
        &self.dims
    }

    /// Check if all dimensions are fixed (fully concrete).
    pub fn is_concrete(&self) -> bool {
        self.dims.iter().all(|d| d.is_fixed())
    }

    /// Check if any dimension is symbolic or dynamic.
    pub fn has_symbolic(&self) -> bool {
        self.dims.iter().any(|d| !d.is_fixed())
    }

    /// Get all symbolic dimension names used in this shape.
    pub fn symbolic_names(&self) -> Vec<&str> {
        self.dims
            .iter()
            .filter_map(|d| match d {
                SymDim::Symbolic(name) => Some(name.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Try to resolve this symbolic shape to a concrete Shape.
    ///
    /// Returns an error if any symbolic dimension is not bound in the env,
    /// or if any dimension is Dynamic (cannot resolve without a value).
    pub fn resolve(&self, env: &ShapeEnv) -> Result<Shape> {
        let mut concrete = Vec::with_capacity(self.dims.len());
        for (i, dim) in self.dims.iter().enumerate() {
            match dim.resolve(env) {
                Some(n) => concrete.push(n),
                None => {
                    return Err(Error::msg(format!(
                        "cannot resolve dimension {} ({}) — not bound in environment",
                        i, dim
                    )));
                }
            }
        }
        Ok(Shape::new(concrete))
    }

    /// Try to resolve, falling back to a default for unresolved dims.
    /// Dynamic/unbound symbolic dims use the provided default value.
    pub fn resolve_with_default(&self, env: &ShapeEnv, default: usize) -> Shape {
        let concrete: Vec<usize> = self
            .dims
            .iter()
            .map(|d| d.resolve(env).unwrap_or(default))
            .collect();
        Shape::new(concrete)
    }

    /// Check if a concrete shape matches this pattern.
    ///
    /// Returns true if the shapes have the same rank and each dimension
    /// matches (fixed dims must be equal, symbolic dims must match their
    /// binding if bound, dynamic dims match anything).
    pub fn matches(&self, shape: &Shape, env: &ShapeEnv) -> bool {
        if self.rank() != shape.rank() {
            return false;
        }
        self.dims
            .iter()
            .zip(shape.dims().iter())
            .all(|(pattern, &value)| pattern.matches(value, env))
    }

    /// Unify this pattern with a concrete shape, binding symbolic dims.
    ///
    /// Returns true if unification succeeded (all dims compatible and
    /// symbolic bindings are consistent). On success, newly discovered
    /// bindings are added to `env`.
    pub fn unify(&self, shape: &Shape, env: &mut ShapeEnv) -> bool {
        if self.rank() != shape.rank() {
            return false;
        }
        // First pass: check consistency without modifying env
        let mut new_bindings = Vec::new();
        for (pattern, &value) in self.dims.iter().zip(shape.dims().iter()) {
            match pattern {
                SymDim::Fixed(n) => {
                    if value != *n {
                        return false;
                    }
                }
                SymDim::Symbolic(name) => {
                    if let Some(bound) = env.get(name) {
                        if value != bound {
                            return false;
                        }
                    } else {
                        // Check consistency with other new bindings
                        if let Some(&prev) =
                            new_bindings.iter().find_map(|(n, v): &(&str, usize)| {
                                if *n == name.as_str() {
                                    Some(v)
                                } else {
                                    None
                                }
                            })
                        {
                            if value != prev {
                                return false;
                            }
                        } else {
                            new_bindings.push((name.as_str(), value));
                        }
                    }
                }
                SymDim::Dynamic => {} // always matches
            }
        }
        // Apply new bindings
        for (name, value) in new_bindings {
            env.bind(name, value);
        }
        true
    }

    /// Compute the output shape of a broadcasting operation between two
    /// symbolic shapes. Returns None if shapes are incompatible.
    pub fn broadcast(&self, other: &SymbolicShape) -> Option<SymbolicShape> {
        let rank = self.rank().max(other.rank());
        let mut result = Vec::with_capacity(rank);

        for i in 0..rank {
            let a = if i < rank - self.rank() {
                &SymDim::Fixed(1)
            } else {
                &self.dims[i - (rank - self.rank())]
            };
            let b = if i < rank - other.rank() {
                &SymDim::Fixed(1)
            } else {
                &other.dims[i - (rank - other.rank())]
            };

            match (a, b) {
                (SymDim::Fixed(1), _) => result.push(b.clone()),
                (_, SymDim::Fixed(1)) => result.push(a.clone()),
                (SymDim::Fixed(x), SymDim::Fixed(y)) if x == y => result.push(a.clone()),
                (SymDim::Fixed(_), SymDim::Fixed(_)) => return None, // incompatible
                (SymDim::Symbolic(s), SymDim::Symbolic(t)) if s == t => result.push(a.clone()),
                (SymDim::Dynamic, _) | (_, SymDim::Dynamic) => result.push(SymDim::Dynamic),
                (SymDim::Symbolic(_), _) | (_, SymDim::Symbolic(_)) => result.push(SymDim::Dynamic),
            }
        }
        Some(SymbolicShape::new(result))
    }
}

impl fmt::Display for SymbolicShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

impl From<Vec<SymDim>> for SymbolicShape {
    fn from(dims: Vec<SymDim>) -> Self {
        Self::new(dims)
    }
}

impl From<Shape> for SymbolicShape {
    fn from(shape: Shape) -> Self {
        Self::from_shape(&shape)
    }
}

// ShapeEnv — Environment for symbolic dimension bindings

/// Maps symbolic dimension names to concrete values.
///
/// Used during shape resolution to convert symbolic shapes to concrete ones.
///
/// # Examples
/// ```ignore
/// let mut env = ShapeEnv::new();
/// env.bind("Batch", 32);
/// env.bind("SeqLen", 128);
/// assert_eq!(env.get("Batch"), Some(32));
/// ```
#[derive(Debug, Clone)]
pub struct ShapeEnv {
    bindings: HashMap<String, usize>,
}

impl ShapeEnv {
    /// Create an empty shape environment.
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Bind a symbolic name to a concrete value.
    pub fn bind(&mut self, name: impl Into<String>, value: usize) {
        self.bindings.insert(name.into(), value);
    }

    /// Look up a symbolic name.
    pub fn get(&self, name: &str) -> Option<usize> {
        self.bindings.get(name).copied()
    }

    /// Check if a name is bound.
    pub fn is_bound(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    /// Get all bindings.
    pub fn bindings(&self) -> &HashMap<String, usize> {
        &self.bindings
    }

    /// Number of bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Is the environment empty?
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Merge another environment into this one.
    /// Returns an error if there are conflicting bindings.
    pub fn merge(&mut self, other: &ShapeEnv) -> Result<()> {
        for (name, &value) in &other.bindings {
            if let Some(&existing) = self.bindings.get(name) {
                if existing != value {
                    return Err(Error::msg(format!(
                        "conflicting binding for '{}': {} vs {}",
                        name, existing, value
                    )));
                }
            } else {
                self.bindings.insert(name.clone(), value);
            }
        }
        Ok(())
    }
}

impl Default for ShapeEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&[(&str, usize)]> for ShapeEnv {
    fn from(bindings: &[(&str, usize)]) -> Self {
        let mut env = ShapeEnv::new();
        for &(name, value) in bindings {
            env.bind(name, value);
        }
        env
    }
}

// ShapeGuard — Runtime shape validation against patterns

/// Validates concrete tensor shapes against symbolic patterns.
///
/// Useful for checking model inputs at runtime, ensuring shapes are
/// compatible before attempting operations that would fail.
///
/// # Examples
/// ```ignore
/// let guard = ShapeGuard::new("input")
///     .expect(SymbolicShape::from(vec![
///         SymDim::symbolic("Batch"),
///         SymDim::fixed(784),
///     ]));
///
/// // This shape is valid (batch=32, features=784)
/// guard.validate(&Shape::new(vec![32, 784]), &env)?;
///
/// // This would error (wrong feature dim)
/// guard.validate(&Shape::new(vec![32, 100]), &env); // Error!
/// ```
#[derive(Debug, Clone)]
pub struct ShapeGuard {
    /// Name of the tensor being guarded (for error messages).
    name: String,
    /// Expected shape pattern.
    pattern: SymbolicShape,
}

impl ShapeGuard {
    /// Create a shape guard with a name and expected pattern.
    pub fn new(name: impl Into<String>, pattern: SymbolicShape) -> Self {
        Self {
            name: name.into(),
            pattern,
        }
    }

    /// Validate a concrete shape against the expected pattern.
    ///
    /// Returns Ok(()) if the shape matches, or an error describing
    /// the mismatch.
    pub fn validate(&self, shape: &Shape, env: &ShapeEnv) -> Result<()> {
        if self.pattern.rank() != shape.rank() {
            return Err(Error::msg(format!(
                "shape mismatch for '{}': expected rank {} ({}), got rank {} ({:?})",
                self.name,
                self.pattern.rank(),
                self.pattern,
                shape.rank(),
                shape.dims()
            )));
        }
        for (i, (expected, &actual)) in self
            .pattern
            .dims()
            .iter()
            .zip(shape.dims().iter())
            .enumerate()
        {
            if !expected.matches(actual, env) {
                return Err(Error::msg(format!(
                    "shape mismatch for '{}' at dim {}: expected {}, got {}",
                    self.name, i, expected, actual
                )));
            }
        }
        Ok(())
    }

    /// Validate and bind — simultaneously validates the shape and binds
    /// any unbound symbolic dimensions. Returns Ok if consistent.
    pub fn validate_and_bind(&self, shape: &Shape, env: &mut ShapeEnv) -> Result<()> {
        if !self.pattern.unify(shape, env) {
            Err(Error::msg(format!(
                "shape mismatch for '{}': expected {}, got {:?}",
                self.name,
                self.pattern,
                shape.dims()
            )))
        } else {
            Ok(())
        }
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    // ── SymDim ──

    #[test]
    fn test_symdim_fixed() {
        let d = SymDim::fixed(32);
        assert!(d.is_fixed());
        assert!(!d.is_symbolic());
        assert!(!d.is_dynamic());
        assert_eq!(d.resolve(&ShapeEnv::new()), Some(32));
        assert_eq!(format!("{d}"), "32");
    }

    #[test]
    fn test_symdim_symbolic_bound() {
        let d = SymDim::symbolic("Batch");
        let mut env = ShapeEnv::new();
        env.bind("Batch", 64);

        assert!(d.is_symbolic());
        assert_eq!(d.resolve(&env), Some(64));
        assert!(d.matches(64, &env));
        assert!(!d.matches(32, &env));
    }

    #[test]
    fn test_symdim_symbolic_unbound() {
        let d = SymDim::symbolic("SeqLen");
        let env = ShapeEnv::new();

        assert_eq!(d.resolve(&env), None);
        assert!(d.matches(100, &env)); // unbound matches anything
        assert!(d.matches(200, &env));
    }

    #[test]
    fn test_symdim_dynamic() {
        let d = SymDim::dynamic();
        let env = ShapeEnv::new();
        assert!(d.is_dynamic());
        assert_eq!(d.resolve(&env), None);
        assert!(d.matches(999, &env));
    }

    #[test]
    fn test_symdim_unify() {
        let d = SymDim::symbolic("N");
        let mut env = ShapeEnv::new();
        assert!(d.unify(42, &mut env));
        assert_eq!(env.get("N"), Some(42));
        // Same value: unify again succeeds
        assert!(d.unify(42, &mut env));
        // Different value: unify fails
        assert!(!d.unify(99, &mut env));
    }

    #[test]
    fn test_symdim_from() {
        let d: SymDim = 32usize.into();
        assert_eq!(d, SymDim::Fixed(32));
        let d: SymDim = "Batch".into();
        assert_eq!(d, SymDim::Symbolic("Batch".to_string()));
    }

    // ── SymbolicShape ──

    #[test]
    fn test_symbolic_shape_basic() {
        let s = SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(784)]);
        assert_eq!(s.rank(), 2);
        assert!(!s.is_concrete());
        assert!(s.has_symbolic());
        assert_eq!(s.symbolic_names(), vec!["Batch"]);
        assert_eq!(format!("{s}"), "[Batch, 784]");
    }

    #[test]
    fn test_symbolic_shape_resolve() {
        let s = SymbolicShape::new(vec![
            SymDim::symbolic("Batch"),
            SymDim::symbolic("SeqLen"),
            SymDim::fixed(768),
        ]);
        let mut env = ShapeEnv::new();
        env.bind("Batch", 32);
        env.bind("SeqLen", 128);

        let concrete = s.resolve(&env).unwrap();
        assert_eq!(concrete.dims(), &[32, 128, 768]);
    }

    #[test]
    fn test_symbolic_shape_resolve_fails_unbound() {
        let s = SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(784)]);
        let env = ShapeEnv::new();
        assert!(s.resolve(&env).is_err());
    }

    #[test]
    fn test_symbolic_shape_resolve_dynamic_fails() {
        let s = SymbolicShape::new(vec![SymDim::dynamic(), SymDim::fixed(784)]);
        let env = ShapeEnv::new();
        assert!(s.resolve(&env).is_err());
    }

    #[test]
    fn test_symbolic_shape_resolve_with_default() {
        let s = SymbolicShape::new(vec![
            SymDim::dynamic(),
            SymDim::symbolic("N"),
            SymDim::fixed(768),
        ]);
        let mut env = ShapeEnv::new();
        env.bind("N", 100);
        let concrete = s.resolve_with_default(&env, 1);
        assert_eq!(concrete.dims(), &[1, 100, 768]);
    }

    #[test]
    fn test_symbolic_shape_matches() {
        let s = SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(784)]);
        let mut env = ShapeEnv::new();
        env.bind("Batch", 32);

        assert!(s.matches(&Shape::new(vec![32, 784]), &env));
        assert!(!s.matches(&Shape::new(vec![32, 100]), &env)); // wrong feature dim
        assert!(!s.matches(&Shape::new(vec![64, 784]), &env)); // wrong batch
        assert!(!s.matches(&Shape::new(vec![32, 784, 1]), &env)); // wrong rank
    }

    #[test]
    fn test_symbolic_shape_unify() {
        let s = SymbolicShape::new(vec![
            SymDim::symbolic("Batch"),
            SymDim::symbolic("SeqLen"),
            SymDim::fixed(768),
        ]);
        let shape = Shape::new(vec![16, 256, 768]);
        let mut env = ShapeEnv::new();

        assert!(s.unify(&shape, &mut env));
        assert_eq!(env.get("Batch"), Some(16));
        assert_eq!(env.get("SeqLen"), Some(256));
    }

    #[test]
    fn test_symbolic_shape_unify_consistency() {
        // Same symbolic name used twice must have same value
        let s = SymbolicShape::new(vec![SymDim::symbolic("N"), SymDim::symbolic("N")]);

        let mut env1 = ShapeEnv::new();
        assert!(s.unify(&Shape::new(vec![32, 32]), &mut env1));
        assert_eq!(env1.get("N"), Some(32));

        let mut env2 = ShapeEnv::new();
        assert!(!s.unify(&Shape::new(vec![32, 64]), &mut env2)); // inconsistent
    }

    #[test]
    fn test_symbolic_shape_unify_fixed_mismatch() {
        let s = SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(784)]);
        let mut env = ShapeEnv::new();
        assert!(!s.unify(&Shape::new(vec![32, 100]), &mut env)); // 100 != 784
    }

    #[test]
    fn test_symbolic_shape_from_concrete() {
        let shape = Shape::new(vec![2, 3, 4]);
        let sym = SymbolicShape::from_shape(&shape);
        assert!(sym.is_concrete());
        assert!(!sym.has_symbolic());
        let resolved = sym.resolve(&ShapeEnv::new()).unwrap();
        assert_eq!(resolved.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_symbolic_shape_broadcast() {
        let a = SymbolicShape::new(vec![SymDim::fixed(3), SymDim::fixed(1)]);
        let b = SymbolicShape::new(vec![SymDim::fixed(1), SymDim::fixed(4)]);
        let c = a.broadcast(&b).unwrap();
        assert_eq!(format!("{c}"), "[3, 4]");
    }

    #[test]
    fn test_symbolic_shape_broadcast_symbolic() {
        let a = SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(768)]);
        let b = SymbolicShape::new(vec![SymDim::fixed(1), SymDim::fixed(768)]);
        let c = a.broadcast(&b).unwrap();
        assert_eq!(format!("{c}"), "[Batch, 768]");
    }

    #[test]
    fn test_symbolic_shape_broadcast_incompatible() {
        let a = SymbolicShape::new(vec![SymDim::fixed(3)]);
        let b = SymbolicShape::new(vec![SymDim::fixed(4)]);
        assert!(a.broadcast(&b).is_none());
    }

    #[test]
    fn test_symbolic_shape_broadcast_rank_extension() {
        let a = SymbolicShape::new(vec![SymDim::fixed(768)]);
        let b = SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(768)]);
        let c = a.broadcast(&b).unwrap();
        assert_eq!(c.rank(), 2);
        assert_eq!(format!("{c}"), "[Batch, 768]");
    }

    // ── ShapeEnv ──

    #[test]
    fn test_shape_env_basic() {
        let mut env = ShapeEnv::new();
        assert!(env.is_empty());
        env.bind("Batch", 32);
        assert_eq!(env.len(), 1);
        assert_eq!(env.get("Batch"), Some(32));
        assert!(env.is_bound("Batch"));
        assert!(!env.is_bound("SeqLen"));
    }

    #[test]
    fn test_shape_env_merge() {
        let mut a = ShapeEnv::new();
        a.bind("Batch", 32);

        let mut b = ShapeEnv::new();
        b.bind("SeqLen", 128);
        b.bind("Batch", 32); // same value — ok

        a.merge(&b).unwrap();
        assert_eq!(a.get("Batch"), Some(32));
        assert_eq!(a.get("SeqLen"), Some(128));
    }

    #[test]
    fn test_shape_env_merge_conflict() {
        let mut a = ShapeEnv::new();
        a.bind("Batch", 32);

        let mut b = ShapeEnv::new();
        b.bind("Batch", 64); // different value — conflict

        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn test_shape_env_from_slice() {
        let env = ShapeEnv::from([("Batch", 32), ("SeqLen", 128)].as_slice());
        assert_eq!(env.get("Batch"), Some(32));
        assert_eq!(env.get("SeqLen"), Some(128));
    }

    // ── ShapeGuard ──

    #[test]
    fn test_shape_guard_validate() {
        let guard = ShapeGuard::new(
            "input",
            SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(784)]),
        );
        let mut env = ShapeEnv::new();
        env.bind("Batch", 32);

        assert!(guard.validate(&Shape::new(vec![32, 784]), &env).is_ok());
        assert!(guard.validate(&Shape::new(vec![32, 100]), &env).is_err());
        assert!(guard.validate(&Shape::new(vec![64, 784]), &env).is_err());
    }

    #[test]
    fn test_shape_guard_validate_and_bind() {
        let guard = ShapeGuard::new(
            "input",
            SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(784)]),
        );
        let mut env = ShapeEnv::new();

        // First call: binds Batch=32
        guard
            .validate_and_bind(&Shape::new(vec![32, 784]), &mut env)
            .unwrap();
        assert_eq!(env.get("Batch"), Some(32));

        // Second call with same batch: ok
        guard
            .validate_and_bind(&Shape::new(vec![32, 784]), &mut env)
            .unwrap();

        // Third call with different batch: error
        assert!(guard
            .validate_and_bind(&Shape::new(vec![64, 784]), &mut env)
            .is_err());
    }

    #[test]
    fn test_shape_guard_wrong_rank() {
        let guard = ShapeGuard::new(
            "x",
            SymbolicShape::new(vec![SymDim::dynamic(), SymDim::fixed(10)]),
        );
        let env = ShapeEnv::new();
        assert!(guard.validate(&Shape::new(vec![10]), &env).is_err());
        assert!(guard.validate(&Shape::new(vec![5, 10, 1]), &env).is_err());
    }

    // ── Integration: multi-tensor shape consistency ──

    #[test]
    fn test_multi_tensor_batch_consistency() {
        // Verify that batch dimensions stay consistent across tensors
        let input_guard = ShapeGuard::new(
            "input",
            SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(784)]),
        );
        let target_guard = ShapeGuard::new(
            "target",
            SymbolicShape::new(vec![SymDim::symbolic("Batch"), SymDim::fixed(10)]),
        );

        let mut env = ShapeEnv::new();

        // Input has batch=32
        input_guard
            .validate_and_bind(&Shape::new(vec![32, 784]), &mut env)
            .unwrap();

        // Target must also have batch=32
        target_guard
            .validate_and_bind(&Shape::new(vec![32, 10]), &mut env)
            .unwrap();

        // Target with batch=16 would fail
        let mut env2 = env.clone();
        assert!(target_guard
            .validate_and_bind(&Shape::new(vec![16, 10]), &mut env2)
            .is_err());
    }

    #[test]
    fn test_transformer_shape_pattern() {
        // A transformer model with batch, sequence, and hidden dimensions
        let pattern = SymbolicShape::new(vec![
            SymDim::symbolic("Batch"),
            SymDim::symbolic("SeqLen"),
            SymDim::fixed(768),
        ]);

        // Different batch sizes work
        let mut env1 = ShapeEnv::new();
        assert!(pattern.unify(&Shape::new(vec![1, 512, 768]), &mut env1));
        assert_eq!(env1.get("Batch"), Some(1));
        assert_eq!(env1.get("SeqLen"), Some(512));

        let mut env2 = ShapeEnv::new();
        assert!(pattern.unify(&Shape::new(vec![32, 128, 768]), &mut env2));
        assert_eq!(env2.get("Batch"), Some(32));
        assert_eq!(env2.get("SeqLen"), Some(128));

        // Wrong hidden dim fails
        let mut env3 = ShapeEnv::new();
        assert!(!pattern.unify(&Shape::new(vec![32, 128, 512]), &mut env3));
    }
}
