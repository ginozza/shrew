// AST — Abstract Syntax Tree for the .sw language
//
// Every node in the AST corresponds to a production in the grammar.
// The AST is purely syntactic — no type checking, no name resolution,
// no graph construction. Those happen in later IR lowering passes.
//
// DESIGN: Every node stores a Span for error reporting back to the user.

use crate::token::Span;

// Top-level program

/// A complete .sw program is a sequence of directives and imports.
#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<TopLevel>,
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Import(ImportStmt),
    Metadata(MetadataBlock),
    Config(ConfigBlock),
    Types(TypesBlock),
    Graph(GraphBlock),
    CustomOp(CustomOpBlock),
    Training(TrainingBlock),
    Inference(InferenceBlock),
    Metrics(MetricsBlock),
    Logging(LoggingBlock),
    Visualization(VisualizationBlock),
}

// Import

/// `@import "path/to/file.sw" as alias;`
#[derive(Debug, Clone)]
pub struct ImportStmt {
    pub path: String,
    pub alias: Option<String>,
    pub span: Span,
}

// Metadata (@model)

/// `@model { name: "GPT-2"; version: "1.0"; }`
#[derive(Debug, Clone)]
pub struct MetadataBlock {
    pub fields: Vec<Field>,
    pub span: Span,
}

// Config (@config)

/// `@config { d_model: 768; n_heads: 12; }`
#[derive(Debug, Clone)]
pub struct ConfigBlock {
    pub fields: Vec<ExprField>,
    pub span: Span,
}

// Types (@types)

/// `@types { type Hidden = Tensor<[Batch, 768], f32>; }`
#[derive(Debug, Clone)]
pub struct TypesBlock {
    pub defs: Vec<TypeDef>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

/// A type expression (Tensor<dims, dtype>, scalar, tuple, etc.).
#[derive(Debug, Clone)]
pub enum TypeExpr {
    /// `Tensor<[dim1, dim2, ...], dtype>`
    Tensor {
        dims: Vec<Dimension>,
        dtype: DTypeKind,
        span: Span,
    },
    /// A bare scalar dtype: `f32`, `i64`, etc.
    Scalar(DTypeKind, Span),
    /// `(TypeA, TypeB, ...)`
    Tuple(Vec<TypeExpr>, Span),
    /// `[TypeExpr]` — list of
    List(Box<TypeExpr>, Span),
    /// `{ field: Type, ... }` — dict/struct
    Dict(Vec<(String, TypeExpr)>, Span),
    /// A named type alias reference
    Named(String, Span),
    /// `?` — dynamic/unknown
    Dynamic(Span),
    /// An integer dimension used as a type: concrete dimension value
    IntDim(i64, Span),
    /// Arithmetic on dimensions: e.g. `D / 2`
    BinaryDim {
        left: Box<TypeExpr>,
        op: BinOp,
        right: Box<TypeExpr>,
        span: Span,
    },
}

/// Dimension in a Tensor type.
#[derive(Debug, Clone)]
pub enum Dimension {
    /// Named/symbolic: `Batch`, `SeqLen`
    Named(String, Span),
    /// Concrete: `768`
    Concrete(i64, Span),
    /// Dynamic: `?`
    Dynamic(Span),
    /// Inferred: `_`
    Inferred(Span),
    /// Computed: `D / 2`
    Computed(Box<Expr>, Span),
}

/// Data types supported by the language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeKind {
    F16,
    F32,
    F64,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Complex64,
    Complex128,
}

// Graph (@graph)

/// `@graph Forward(x: Tensor<[B,S,D], f32>) -> Tensor<[B,S,D], f32> { ... }`
#[derive(Debug, Clone)]
pub struct GraphBlock {
    pub name: String,
    pub params: Vec<ParamDef>,
    pub return_type: Option<TypeExpr>,
    pub body: Vec<GraphStmt>,
    pub span: Span,
}

/// A parameter definition: `name: Type [?]`
#[derive(Debug, Clone)]
pub struct ParamDef {
    pub name: String,
    pub ty: TypeExpr,
    pub optional: bool,
    pub span: Span,
}

/// Statements inside a graph body.
#[derive(Debug, Clone)]
pub enum GraphStmt {
    Input(InputDecl),
    Output(OutputDecl),
    Param(ParamDecl),
    Node(NodeDecl),
    Assert(AssertStmt),
    Check(CheckBlock),
}

/// `input x: Tensor<[B,S], f32>;`
#[derive(Debug, Clone)]
pub struct InputDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub optional: bool,
    pub span: Span,
}

/// `output logits: softmax(h);` or `output expr;`
#[derive(Debug, Clone)]
pub struct OutputDecl {
    pub name: Option<String>,
    pub expr: Expr,
    pub span: Span,
}

/// `param W: Tensor<[D,D], f32> { init: "normal(0,0.02)"; frozen: false; };`
#[derive(Debug, Clone)]
pub struct ParamDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub attrs: Vec<ParamAttr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ParamAttr {
    pub key: String,
    pub value: Expr,
    pub span: Span,
}

/// `node h { op: matmul(x, W); }`  or  `node h: Type { ... };`
#[derive(Debug, Clone)]
pub struct NodeDecl {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub stmts: Vec<NodeStmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum NodeStmt {
    Op(Expr, Span),
    InputRef(Expr, Span),
    OutputType(TypeExpr, Span),
    Hint(HintKind, Span),
    Attr(String, Expr, Span),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HintKind {
    RecomputeInBackward,
    MustPreserve,
    InPlace,
    NoGrad,
    Custom(String),
}

/// `@assert shape(x) == [B, S, D], "shape mismatch";`
#[derive(Debug, Clone)]
pub struct AssertStmt {
    pub condition: Expr,
    pub message: Option<String>,
    pub span: Span,
}

/// `@check name { assert ...; assert ...; }`
#[derive(Debug, Clone)]
pub struct CheckBlock {
    pub name: String,
    pub conditions: Vec<AssertStmt>,
    pub span: Span,
}

// Custom operators (@custom_op)

#[derive(Debug, Clone)]
pub struct CustomOpBlock {
    pub name: String,
    pub stmts: Vec<CustomOpStmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum CustomOpStmt {
    Signature {
        params: Vec<ParamDef>,
        return_type: TypeExpr,
        span: Span,
    },
    Impl {
        target: String,
        attrs: Vec<ExprField>,
        span: Span,
    },
    Gradient {
        target: String,
        body: Vec<CustomOpStmt>,
        span: Span,
    },
}

// Training (@training)

#[derive(Debug, Clone)]
pub struct TrainingBlock {
    pub fields: Vec<TrainingField>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TrainingField {
    Model(String, Span),
    Loss(String, Span),
    Optimizer(Vec<ExprField>, Span),
    LrSchedule(Vec<ExprField>, Span),
    GradClip(Vec<ExprField>, Span),
    Generic(ExprField),
}

// Inference (@inference)

#[derive(Debug, Clone)]
pub struct InferenceBlock {
    pub fields: Vec<InferenceField>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum InferenceField {
    Model(String, Span),
    Optimizations(Vec<Expr>, Span),
    Quantization(Vec<ExprField>, Span),
    Generation(Vec<ExprField>, Span),
    Generic(ExprField),
}

// Metrics & Logging

#[derive(Debug, Clone)]
pub struct MetricsBlock {
    pub name: String,
    pub defs: Vec<MetricDef>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct MetricDef {
    pub name: String,
    pub attrs: Vec<ExprField>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct LoggingBlock {
    pub fields: Vec<ExprField>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct VisualizationBlock {
    pub plots: Vec<PlotDef>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct PlotDef {
    pub name: String,
    pub attrs: Vec<ExprField>,
    pub span: Span,
}

// Shared field types

/// A field with a literal value: `name: "GPT-2";`
#[derive(Debug, Clone)]
pub struct Field {
    pub key: String,
    pub value: Literal,
    pub span: Span,
}

/// A field with an expression value: `d_model: 768;`
#[derive(Debug, Clone)]
pub struct ExprField {
    pub key: String,
    pub value: Expr,
    pub span: Span,
}

// Expressions

#[derive(Debug, Clone)]
pub enum Expr {
    /// Integer literal: `42`
    Int(i64, Span),
    /// Float literal: `3.14`
    Float(f64, Span),
    /// String literal: `"hello"`
    Str(String, Span),
    /// Boolean: `true`, `false`
    Bool(bool, Span),
    /// Null: `null`
    Null(Span),
    /// Identifier: `x`, `Batch`
    Ident(String, Span),
    /// Binary expression: `a + b`
    Binary {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
        span: Span,
    },
    /// Unary expression: `-x`, `!cond`
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
    /// Function call: `matmul(x, w)`
    Call {
        func: String,
        args: Vec<Arg>,
        span: Span,
    },
    /// Qualified call: `module.func(args)` or `mod::func(args)`
    QualifiedCall {
        path: Vec<String>,
        args: Vec<Arg>,
        span: Span,
    },
    /// Member access: `x.shape`
    Member {
        object: Box<Expr>,
        field: String,
        span: Span,
    },
    /// Index access: `x[0]` or `x[1:3]`
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
        end: Option<Box<Expr>>,
        span: Span,
    },
    /// List expression: `[1, 2, 3]`
    List(Vec<Expr>, Span),
    /// Dict expression: `{ key: value, ... }`
    Dict(Vec<(String, Expr)>, Span),
    /// Parenthesized: `(expr)`
    Paren(Box<Expr>, Span),
    /// Block operation — if
    IfExpr {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
        span: Span,
    },
    /// Block operation — repeat
    RepeatExpr {
        count: Box<Expr>,
        body: Box<Expr>,
        span: Span,
    },
    /// Closure: `|a, b| { expr }`
    Closure {
        params: Vec<String>,
        body: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Int(_, s)
            | Expr::Float(_, s)
            | Expr::Str(_, s)
            | Expr::Bool(_, s)
            | Expr::Null(s)
            | Expr::Ident(_, s) => *s,
            Expr::Binary { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Call { span, .. }
            | Expr::QualifiedCall { span, .. }
            | Expr::Member { span, .. }
            | Expr::Index { span, .. }
            | Expr::List(_, span)
            | Expr::Dict(_, span)
            | Expr::Paren(_, span)
            | Expr::IfExpr { span, .. }
            | Expr::RepeatExpr { span, .. }
            | Expr::Closure { span, .. } => *span,
        }
    }
}

/// Named or positional argument.
#[derive(Debug, Clone)]
pub struct Arg {
    pub name: Option<String>,
    pub value: Expr,
    pub span: Span,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    NullCoalesce,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

// Literals

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64, Span),
    Float(f64, Span),
    Str(String, Span),
    Bool(bool, Span),
    Null(Span),
    List(Vec<Literal>, Span),
    Dict(Vec<(String, Literal)>, Span),
}
