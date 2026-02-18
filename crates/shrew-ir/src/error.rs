// Error types for shrew-ir

use crate::token::Span;
use std::fmt;

/// Result type for the IR crate.
pub type Result<T> = std::result::Result<T, Error>;

/// All errors that can occur during lexing, parsing, or IR construction.
#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub span: Option<Span>,
    pub source_line: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ErrorKind {
    // Lexer errors
    UnexpectedChar(char),
    UnterminatedString,
    UnterminatedComment,
    InvalidNumber(String),
    UnknownDirective(String),

    // Parser errors
    UnexpectedToken { expected: String, got: String },
    UnexpectedEof,
    InvalidDType(String),

    // General
    Message(String),
}

impl Error {
    pub fn new(kind: ErrorKind, span: Span) -> Self {
        Self {
            kind,
            span: Some(span),
            source_line: None,
        }
    }

    pub fn msg(s: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Message(s.into()),
            span: None,
            source_line: None,
        }
    }

    pub fn with_source_line(mut self, line: String) -> Self {
        self.source_line = Some(line);
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(span) = &self.span {
            write!(f, "[{}:{}] ", span.line, span.col)?;
        }
        match &self.kind {
            ErrorKind::UnexpectedChar(c) => write!(f, "unexpected character '{c}'"),
            ErrorKind::UnterminatedString => write!(f, "unterminated string literal"),
            ErrorKind::UnterminatedComment => write!(f, "unterminated block comment"),
            ErrorKind::InvalidNumber(s) => write!(f, "invalid number '{s}'"),
            ErrorKind::UnknownDirective(s) => write!(f, "unknown directive '@{s}'"),
            ErrorKind::UnexpectedToken { expected, got } => {
                write!(f, "expected {expected}, got {got}")
            }
            ErrorKind::UnexpectedEof => write!(f, "unexpected end of file"),
            ErrorKind::InvalidDType(s) => write!(f, "invalid dtype '{s}'"),
            ErrorKind::Message(s) => write!(f, "{s}"),
        }?;
        if let Some(line) = &self.source_line {
            write!(f, "\n  | {line}")?;
            if let Some(span) = &self.span {
                write!(f, "\n  | {}^", " ".repeat(span.col.saturating_sub(1)))?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for Error {}
