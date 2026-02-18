// Token — All lexical tokens of the .sw language
//
// The .sw format uses @ as the directive prefix (like Rust attributes).
// Tokens fall into these categories:
//
//   1. Directives    — @model, @graph, @training, etc.
//   2. Keywords      — input, output, param, node, op, type, etc.
//   3. Operators     — +, -, *, /, **, ==, !=, &&, ||, etc.
//   4. Punctuation   — { } ( ) [ ] : ; , . -> ? _ ::
//   5. Literals      — integers, floats, strings, booleans, null
//   6. DType names   — f32, f64, bf16, etc.
//   7. Identifiers   — user-defined names
//
// Each token carries a Span (byte offset + length) for error reporting.

use std::fmt;

/// Byte-level location in source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Byte offset from the start of the source.
    pub offset: usize,
    /// Length in bytes.
    pub len: usize,
    /// Line number (1-based).
    pub line: usize,
    /// Column number (1-based, in bytes).
    pub col: usize,
}

impl Span {
    pub fn new(offset: usize, len: usize, line: usize, col: usize) -> Self {
        Self {
            offset,
            len,
            line,
            col,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// A token with its kind and source location.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// Every possible token kind in the .sw language.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    //  Directives (@-prefixed blocks) 
    AtModel,          // @model
    AtConfig,         // @config
    AtTypes,          // @types
    AtGraph,          // @graph
    AtCustomOp,       // @custom_op
    AtTraining,       // @training
    AtInference,      // @inference
    AtMetrics,        // @metrics
    AtLogging,        // @logging
    AtVisualizations, // @visualizations
    AtImport,         // @import
    AtAssert,         // @assert
    AtCheck,          // @check
    AtHint,           // @hint

    //  Keywords 
    Input,
    Output,
    Param,
    Node,
    Op,
    Call,
    If,
    Else,
    Repeat,
    Scan,
    For,
    In,
    Type,
    Init,
    Gradient,
    Impl,
    Signature,
    As,
    Track,
    Plot,
    Range,
    Model,
    Loss,
    Optimizer,
    LrSchedule, // lr_schedule
    GradClip,   // grad_clip
    Precision,
    AccumulationSteps, // accumulation_steps
    Optimizations,
    Quantization,
    Generation,
    Backend,
    Checkpoints,
    Frozen,
    Device,
    Source,
    Compute,
    Aggregate,
    LogEvery, // log_every

    //  Hint values ─
    RecomputeInBackward, // recompute_in_backward
    MustPreserve,        // must_preserve
    InPlace,             // in_place
    NoGrad,              // no_grad

    //  DType keywords 
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

    //  Device keywords ─
    Cpu,
    Gpu,
    Tpu,

    //  Tensor keyword 
    Tensor,

    //  Literals 
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    True,
    False,
    Null,

    //  Operators ─
    Plus,             // +
    Minus,            // -
    Star,             // *
    Slash,            // /
    Percent,          // %
    StarStar,         // **
    EqEq,             // ==
    BangEq,           // !=
    Lt,               // <
    Gt,               // >
    LtEq,             // <=
    GtEq,             // >=
    AmpAmp,           // &&
    PipePipe,         // ||
    QuestionQuestion, // ??
    Amp,              // &
    Pipe,             // |
    Caret,            // ^
    LtLt,             // <<
    GtGt,             // >>
    Bang,             // !
    Tilde,            // ~
    Eq,               // =

    //  Punctuation ─
    LBrace,     // {
    RBrace,     // }
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]
    Colon,      // :
    ColonColon, // ::
    Semi,       // ;
    Comma,      // ,
    Dot,        // .
    Arrow,      // ->
    Question,   // ?
    Underscore, // _

    //  Identifiers ─
    Ident(String),

    //  Special ─
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::AtModel => write!(f, "@model"),
            TokenKind::AtConfig => write!(f, "@config"),
            TokenKind::AtTypes => write!(f, "@types"),
            TokenKind::AtGraph => write!(f, "@graph"),
            TokenKind::AtCustomOp => write!(f, "@custom_op"),
            TokenKind::AtTraining => write!(f, "@training"),
            TokenKind::AtInference => write!(f, "@inference"),
            TokenKind::AtMetrics => write!(f, "@metrics"),
            TokenKind::AtLogging => write!(f, "@logging"),
            TokenKind::AtVisualizations => write!(f, "@visualizations"),
            TokenKind::AtImport => write!(f, "@import"),
            TokenKind::AtAssert => write!(f, "@assert"),
            TokenKind::AtCheck => write!(f, "@check"),
            TokenKind::AtHint => write!(f, "@hint"),
            TokenKind::Ident(s) => write!(f, "{s}"),
            TokenKind::IntLit(n) => write!(f, "{n}"),
            TokenKind::FloatLit(n) => write!(f, "{n}"),
            TokenKind::StringLit(s) => write!(f, "\"{s}\""),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Null => write!(f, "null"),
            TokenKind::Eof => write!(f, "<eof>"),
            other => write!(f, "{other:?}"),
        }
    }
}

impl TokenKind {
    /// Return the string representation of a keyword token.
    /// Returns `None` for non-keyword tokens (literals, operators, idents, etc.).
    pub fn keyword_str(&self) -> Option<&'static str> {
        match self {
            TokenKind::Input => Some("input"),
            TokenKind::Output => Some("output"),
            TokenKind::Param => Some("param"),
            TokenKind::Node => Some("node"),
            TokenKind::Op => Some("op"),
            TokenKind::Call => Some("call"),
            TokenKind::If => Some("if"),
            TokenKind::Else => Some("else"),
            TokenKind::Repeat => Some("repeat"),
            TokenKind::Scan => Some("scan"),
            TokenKind::For => Some("for"),
            TokenKind::In => Some("in"),
            TokenKind::Type => Some("type"),
            TokenKind::Init => Some("init"),
            TokenKind::Gradient => Some("gradient"),
            TokenKind::Impl => Some("impl"),
            TokenKind::Signature => Some("signature"),
            TokenKind::As => Some("as"),
            TokenKind::Track => Some("track"),
            TokenKind::Plot => Some("plot"),
            TokenKind::Range => Some("range"),
            TokenKind::Model => Some("model"),
            TokenKind::Loss => Some("loss"),
            TokenKind::Optimizer => Some("optimizer"),
            TokenKind::LrSchedule => Some("lr_schedule"),
            TokenKind::GradClip => Some("grad_clip"),
            TokenKind::Precision => Some("precision"),
            TokenKind::AccumulationSteps => Some("accumulation_steps"),
            TokenKind::Optimizations => Some("optimizations"),
            TokenKind::Quantization => Some("quantization"),
            TokenKind::Generation => Some("generation"),
            TokenKind::Backend => Some("backend"),
            TokenKind::Checkpoints => Some("checkpoints"),
            TokenKind::Frozen => Some("frozen"),
            TokenKind::Device => Some("device"),
            TokenKind::Source => Some("source"),
            TokenKind::Compute => Some("compute"),
            TokenKind::Aggregate => Some("aggregate"),
            TokenKind::LogEvery => Some("log_every"),
            TokenKind::RecomputeInBackward => Some("recompute_in_backward"),
            TokenKind::MustPreserve => Some("must_preserve"),
            TokenKind::InPlace => Some("in_place"),
            TokenKind::NoGrad => Some("no_grad"),
            TokenKind::Cpu => Some("cpu"),
            TokenKind::Gpu => Some("gpu"),
            TokenKind::Tpu => Some("tpu"),
            _ => None,
        }
    }
}

/// Look up keyword/dtype/device from an identifier string.
/// Returns None if the string is a plain identifier.
pub fn keyword_lookup(s: &str) -> Option<TokenKind> {
    match s {
        // Keywords
        "input" => Some(TokenKind::Input),
        "output" => Some(TokenKind::Output),
        "param" => Some(TokenKind::Param),
        "node" => Some(TokenKind::Node),
        "op" => Some(TokenKind::Op),
        "call" => Some(TokenKind::Call),
        "if" => Some(TokenKind::If),
        "else" => Some(TokenKind::Else),
        "repeat" => Some(TokenKind::Repeat),
        "scan" => Some(TokenKind::Scan),
        "for" => Some(TokenKind::For),
        "in" => Some(TokenKind::In),
        "type" => Some(TokenKind::Type),
        "init" => Some(TokenKind::Init),
        "gradient" => Some(TokenKind::Gradient),
        "impl" => Some(TokenKind::Impl),
        "signature" => Some(TokenKind::Signature),
        "as" => Some(TokenKind::As),
        "track" => Some(TokenKind::Track),
        "plot" => Some(TokenKind::Plot),
        "range" => Some(TokenKind::Range),
        "model" => Some(TokenKind::Model),
        "loss" => Some(TokenKind::Loss),
        "optimizer" => Some(TokenKind::Optimizer),
        "lr_schedule" => Some(TokenKind::LrSchedule),
        "grad_clip" => Some(TokenKind::GradClip),
        "precision" => Some(TokenKind::Precision),
        "accumulation_steps" => Some(TokenKind::AccumulationSteps),
        "optimizations" => Some(TokenKind::Optimizations),
        "quantization" => Some(TokenKind::Quantization),
        "generation" => Some(TokenKind::Generation),
        "backend" => Some(TokenKind::Backend),
        "checkpoints" => Some(TokenKind::Checkpoints),
        "frozen" => Some(TokenKind::Frozen),
        "device" => Some(TokenKind::Device),
        "source" => Some(TokenKind::Source),
        "compute" => Some(TokenKind::Compute),
        "aggregate" => Some(TokenKind::Aggregate),
        "log_every" => Some(TokenKind::LogEvery),

        // Hint values
        "recompute_in_backward" => Some(TokenKind::RecomputeInBackward),
        "must_preserve" => Some(TokenKind::MustPreserve),
        "in_place" => Some(TokenKind::InPlace),
        "no_grad" => Some(TokenKind::NoGrad),

        // DTypes
        "f16" => Some(TokenKind::F16),
        "f32" => Some(TokenKind::F32),
        "f64" => Some(TokenKind::F64),
        "bf16" => Some(TokenKind::Bf16),
        "i8" => Some(TokenKind::I8),
        "i16" => Some(TokenKind::I16),
        "i32" => Some(TokenKind::I32),
        "i64" => Some(TokenKind::I64),
        "u8" => Some(TokenKind::U8),
        "u16" => Some(TokenKind::U16),
        "u32" => Some(TokenKind::U32),
        "u64" => Some(TokenKind::U64),
        "bool" => Some(TokenKind::Bool),
        "complex64" => Some(TokenKind::Complex64),
        "complex128" => Some(TokenKind::Complex128),

        // Devices
        "cpu" => Some(TokenKind::Cpu),
        "gpu" => Some(TokenKind::Gpu),
        "tpu" => Some(TokenKind::Tpu),

        // Tensor keyword
        "Tensor" => Some(TokenKind::Tensor),

        // Boolean & null literals
        "true" => Some(TokenKind::True),
        "false" => Some(TokenKind::False),
        "null" => Some(TokenKind::Null),

        _ => None,
    }
}
