// Lexer — Converts .sw source text into a stream of Tokens
//
// The lexer is a hand-written scanner (no regex, no generator). It processes
// the source one byte at a time, producing Token values.
//
// DESIGN DECISIONS:
//
//   1. We allow identifiers to contain underscores (e.g. `lr_schedule`),
//      and these map to keyword tokens via `keyword_lookup`.
//
//   2. @ followed by an identifier is lexed as a single directive token.
//
//   3. Numbers: we support integers and floats (with optional exponent).
//      We don't try to distinguish i64 vs u64 at lex time — that's the
//      parser's job. All ints are stored as i64, floats as f64.
//
//   4. Strings use double quotes and support basic escapes: \n \t \\ \"
//
//   5. Comments: // line comments and /* block comments */ (no nesting).

use crate::error::{Error, ErrorKind, Result};
use crate::token::{keyword_lookup, Span, Token, TokenKind};

/// Lexer state over a source string.
pub struct Lexer<'src> {
    src: &'src str,
    bytes: &'src [u8],
    pos: usize,
    line: usize,
    col: usize,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    /// Tokenize the entire source, returning a Vec of Tokens.
    /// The last token is always Eof.
    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    /// Read the next token.
    fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace_and_comments()?;

        if self.pos >= self.bytes.len() {
            return Ok(Token::new(TokenKind::Eof, self.span(0)));
        }

        let start_pos = self.pos;
        let start_line = self.line;
        let start_col = self.col;
        let ch = self.bytes[self.pos] as char;

        //  Directive (@keyword) 
        if ch == '@' {
            self.advance();
            let ident_start = self.pos;
            while self.pos < self.bytes.len()
                && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_')
            {
                self.advance();
            }
            let name = &self.src[ident_start..self.pos];
            let kind = match name {
                "model" => TokenKind::AtModel,
                "config" => TokenKind::AtConfig,
                "types" => TokenKind::AtTypes,
                "graph" => TokenKind::AtGraph,
                "custom_op" => TokenKind::AtCustomOp,
                "training" => TokenKind::AtTraining,
                "inference" => TokenKind::AtInference,
                "metrics" => TokenKind::AtMetrics,
                "logging" => TokenKind::AtLogging,
                "visualizations" => TokenKind::AtVisualizations,
                "import" => TokenKind::AtImport,
                "assert" => TokenKind::AtAssert,
                "check" => TokenKind::AtCheck,
                "hint" => TokenKind::AtHint,
                _ => {
                    return Err(Error::new(
                        ErrorKind::UnknownDirective(name.to_string()),
                        Span::new(start_pos, self.pos - start_pos, start_line, start_col),
                    ))
                }
            };
            return Ok(Token::new(
                kind,
                Span::new(start_pos, self.pos - start_pos, start_line, start_col),
            ));
        }

        //  String literal 
        if ch == '"' {
            return self.lex_string(start_pos, start_line, start_col);
        }

        //  Number literal 
        if ch.is_ascii_digit() {
            return self.lex_number(start_pos, start_line, start_col);
        }

        //  Identifier or keyword 
        if ch.is_ascii_alphabetic() || ch == '_' {
            return self.lex_ident(start_pos, start_line, start_col);
        }

        //  Multi-char operators & punctuation 
        let kind = match ch {
            '{' => {
                self.advance();
                TokenKind::LBrace
            }
            '}' => {
                self.advance();
                TokenKind::RBrace
            }
            '(' => {
                self.advance();
                TokenKind::LParen
            }
            ')' => {
                self.advance();
                TokenKind::RParen
            }
            '[' => {
                self.advance();
                TokenKind::LBracket
            }
            ']' => {
                self.advance();
                TokenKind::RBracket
            }
            ';' => {
                self.advance();
                TokenKind::Semi
            }
            ',' => {
                self.advance();
                TokenKind::Comma
            }
            '~' => {
                self.advance();
                TokenKind::Tilde
            }
            ':' => {
                self.advance();
                if self.peek() == Some(':') {
                    self.advance();
                    TokenKind::ColonColon
                } else {
                    TokenKind::Colon
                }
            }
            '.' => {
                self.advance();
                TokenKind::Dot
            }
            '+' => {
                self.advance();
                TokenKind::Plus
            }
            '%' => {
                self.advance();
                TokenKind::Percent
            }
            '^' => {
                self.advance();
                TokenKind::Caret
            }
            '-' => {
                self.advance();
                if self.peek() == Some('>') {
                    self.advance();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            '*' => {
                self.advance();
                if self.peek() == Some('*') {
                    self.advance();
                    TokenKind::StarStar
                } else {
                    TokenKind::Star
                }
            }
            '/' => {
                self.advance();
                TokenKind::Slash
            }
            '=' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::EqEq
                } else {
                    TokenKind::Eq
                }
            }
            '!' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::BangEq
                } else {
                    TokenKind::Bang
                }
            }
            '<' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::LtEq
                } else if self.peek() == Some('<') {
                    self.advance();
                    TokenKind::LtLt
                } else {
                    TokenKind::Lt
                }
            }
            '>' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::GtEq
                } else if self.peek() == Some('>') {
                    self.advance();
                    TokenKind::GtGt
                } else {
                    TokenKind::Gt
                }
            }
            '&' => {
                self.advance();
                if self.peek() == Some('&') {
                    self.advance();
                    TokenKind::AmpAmp
                } else {
                    TokenKind::Amp
                }
            }
            '|' => {
                self.advance();
                if self.peek() == Some('|') {
                    self.advance();
                    TokenKind::PipePipe
                } else {
                    TokenKind::Pipe
                }
            }
            '?' => {
                self.advance();
                if self.peek() == Some('?') {
                    self.advance();
                    TokenKind::QuestionQuestion
                } else {
                    TokenKind::Question
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::UnexpectedChar(ch),
                    Span::new(start_pos, 1, start_line, start_col),
                ));
            }
        };

        Ok(Token::new(
            kind,
            Span::new(start_pos, self.pos - start_pos, start_line, start_col),
        ))
    }

    // Helpers 

    fn advance(&mut self) {
        if self.pos < self.bytes.len() {
            if self.bytes[self.pos] == b'\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        if self.pos < self.bytes.len() {
            Some(self.bytes[self.pos] as char)
        } else {
            None
        }
    }

    fn span(&self, len: usize) -> Span {
        Span::new(self.pos, len, self.line, self.col)
    }

    /// Skip whitespace, single-line comments (//), and block comments (/* */).
    fn skip_whitespace_and_comments(&mut self) -> Result<()> {
        loop {
            // Skip whitespace
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.advance();
            }

            if self.pos + 1 < self.bytes.len()
                && self.bytes[self.pos] == b'/'
                && self.bytes[self.pos + 1] == b'/'
            {
                // Single-line comment: skip to end of line
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.advance();
                }
                continue;
            }

            if self.pos + 1 < self.bytes.len()
                && self.bytes[self.pos] == b'/'
                && self.bytes[self.pos + 1] == b'*'
            {
                // Block comment: skip to */
                let start_pos = self.pos;
                let start_line = self.line;
                let start_col = self.col;
                self.advance(); // /
                self.advance(); // *
                loop {
                    if self.pos >= self.bytes.len() {
                        return Err(Error::new(
                            ErrorKind::UnterminatedComment,
                            Span::new(start_pos, 2, start_line, start_col),
                        ));
                    }
                    if self.bytes[self.pos] == b'*'
                        && self.pos + 1 < self.bytes.len()
                        && self.bytes[self.pos + 1] == b'/'
                    {
                        self.advance(); // *
                        self.advance(); // /
                        break;
                    }
                    self.advance();
                }
                continue;
            }

            break;
        }
        Ok(())
    }

    /// Lex a string literal (starting after the opening `"`).
    fn lex_string(
        &mut self,
        start_pos: usize,
        start_line: usize,
        start_col: usize,
    ) -> Result<Token> {
        self.advance(); // skip opening "
        let mut value = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                return Err(Error::new(
                    ErrorKind::UnterminatedString,
                    Span::new(start_pos, self.pos - start_pos, start_line, start_col),
                ));
            }
            let ch = self.bytes[self.pos] as char;
            if ch == '"' {
                self.advance(); // skip closing "
                break;
            }
            if ch == '\\' {
                self.advance();
                if self.pos >= self.bytes.len() {
                    return Err(Error::new(
                        ErrorKind::UnterminatedString,
                        Span::new(start_pos, self.pos - start_pos, start_line, start_col),
                    ));
                }
                let esc = self.bytes[self.pos] as char;
                match esc {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    _ => {
                        value.push('\\');
                        value.push(esc);
                    }
                }
                self.advance();
            } else {
                value.push(ch);
                self.advance();
            }
        }
        Ok(Token::new(
            TokenKind::StringLit(value),
            Span::new(start_pos, self.pos - start_pos, start_line, start_col),
        ))
    }

    /// Lex a number: integer or float.
    fn lex_number(
        &mut self,
        start_pos: usize,
        start_line: usize,
        start_col: usize,
    ) -> Result<Token> {
        let num_start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.advance();
        }

        let mut is_float = false;

        // Check for decimal point
        if self.pos < self.bytes.len()
            && self.bytes[self.pos] == b'.'
            && self.pos + 1 < self.bytes.len()
            && self.bytes[self.pos + 1].is_ascii_digit()
        {
            is_float = true;
            self.advance(); // skip .
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.advance();
            }
        }

        // Check for exponent
        if self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b'e' || self.bytes[self.pos] == b'E')
        {
            is_float = true;
            self.advance(); // skip e/E
            if self.pos < self.bytes.len()
                && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-')
            {
                self.advance();
            }
            if self.pos >= self.bytes.len() || !self.bytes[self.pos].is_ascii_digit() {
                let raw = &self.src[num_start..self.pos];
                return Err(Error::new(
                    ErrorKind::InvalidNumber(raw.to_string()),
                    Span::new(start_pos, self.pos - start_pos, start_line, start_col),
                ));
            }
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.advance();
            }
        }

        let raw = &self.src[num_start..self.pos];
        let span = Span::new(start_pos, self.pos - start_pos, start_line, start_col);

        if is_float {
            let val: f64 = raw
                .parse()
                .map_err(|_| Error::new(ErrorKind::InvalidNumber(raw.to_string()), span))?;
            Ok(Token::new(TokenKind::FloatLit(val), span))
        } else {
            let val: i64 = raw
                .parse()
                .map_err(|_| Error::new(ErrorKind::InvalidNumber(raw.to_string()), span))?;
            Ok(Token::new(TokenKind::IntLit(val), span))
        }
    }

    /// Lex an identifier or keyword.
    fn lex_ident(
        &mut self,
        start_pos: usize,
        start_line: usize,
        start_col: usize,
    ) -> Result<Token> {
        let id_start = self.pos;
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_')
        {
            self.advance();
        }
        let word = &self.src[id_start..self.pos];
        let span = Span::new(start_pos, self.pos - start_pos, start_line, start_col);

        // Check if it's _ alone (Underscore token for inferred dims)
        if word == "_" {
            return Ok(Token::new(TokenKind::Underscore, span));
        }

        let kind = keyword_lookup(word).unwrap_or_else(|| TokenKind::Ident(word.to_string()));
        Ok(Token::new(kind, span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<TokenKind> {
        Lexer::new(src)
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    #[test]
    fn test_directive() {
        let kinds = lex("@model { }");
        assert_eq!(
            kinds,
            vec![
                TokenKind::AtModel,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_ident_and_keyword() {
        let kinds = lex("input foo");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Input,
                TokenKind::Ident("foo".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_number_literals() {
        let kinds = lex("42 3.14 1e-4");
        assert_eq!(
            kinds,
            vec![
                TokenKind::IntLit(42),
                TokenKind::FloatLit(3.14),
                TokenKind::FloatLit(1e-4),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_string_literal() {
        let kinds = lex(r#""hello world""#);
        assert_eq!(
            kinds,
            vec![TokenKind::StringLit("hello world".into()), TokenKind::Eof,]
        );
    }

    #[test]
    fn test_string_escape() {
        let kinds = lex(r#""line\none""#);
        assert_eq!(
            kinds,
            vec![TokenKind::StringLit("line\none".into()), TokenKind::Eof,]
        );
    }

    #[test]
    fn test_operators() {
        let kinds = lex("+ - * / ** == != <= >= && || -> :: ?? <<");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::StarStar,
                TokenKind::EqEq,
                TokenKind::BangEq,
                TokenKind::LtEq,
                TokenKind::GtEq,
                TokenKind::AmpAmp,
                TokenKind::PipePipe,
                TokenKind::Arrow,
                TokenKind::ColonColon,
                TokenKind::QuestionQuestion,
                TokenKind::LtLt,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_dtype_keywords() {
        let kinds = lex("f32 f64 bf16 i64 u8 bool");
        assert_eq!(
            kinds,
            vec![
                TokenKind::F32,
                TokenKind::F64,
                TokenKind::Bf16,
                TokenKind::I64,
                TokenKind::U8,
                TokenKind::Bool,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_comment_skipping() {
        let kinds = lex("input // this is a comment\noutput");
        assert_eq!(
            kinds,
            vec![TokenKind::Input, TokenKind::Output, TokenKind::Eof,]
        );
    }

    #[test]
    fn test_block_comment() {
        let kinds = lex("input /* skip this */ output");
        assert_eq!(
            kinds,
            vec![TokenKind::Input, TokenKind::Output, TokenKind::Eof,]
        );
    }

    #[test]
    fn test_tensor_type_tokens() {
        let kinds = lex("Tensor<[Batch, 768], f32>");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Tensor,
                TokenKind::Lt,
                TokenKind::LBracket,
                TokenKind::Ident("Batch".into()),
                TokenKind::Comma,
                TokenKind::IntLit(768),
                TokenKind::RBracket,
                TokenKind::Comma,
                TokenKind::F32,
                TokenKind::Gt,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_full_model_block() {
        let src = r#"
            @model {
                name: "GPT-2";
                version: "1.0";
            }
        "#;
        let kinds = lex(src);
        assert_eq!(
            kinds,
            vec![
                TokenKind::AtModel,
                TokenKind::LBrace,
                TokenKind::Ident("name".into()),
                TokenKind::Colon,
                TokenKind::StringLit("GPT-2".into()),
                TokenKind::Semi,
                TokenKind::Ident("version".into()),
                TokenKind::Colon,
                TokenKind::StringLit("1.0".into()),
                TokenKind::Semi,
                TokenKind::RBrace,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_negative_number_as_minus_int() {
        // -3 is lexed as Minus + IntLit(3), NOT IntLit(-3)
        let kinds = lex("-3");
        assert_eq!(
            kinds,
            vec![TokenKind::Minus, TokenKind::IntLit(3), TokenKind::Eof,]
        );
    }

    #[test]
    fn test_underscore_and_question() {
        let kinds = lex("_ ? ??");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Underscore,
                TokenKind::Question,
                TokenKind::QuestionQuestion,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_span_tracking() {
        let tokens = Lexer::new("ab cd").tokenize().unwrap();
        assert_eq!(tokens[0].span.line, 1);
        assert_eq!(tokens[0].span.col, 1);
        assert_eq!(tokens[1].span.line, 1);
        assert_eq!(tokens[1].span.col, 4);
    }

    #[test]
    fn test_multiline_span() {
        let tokens = Lexer::new("ab\ncd").tokenize().unwrap();
        assert_eq!(tokens[0].span.line, 1);
        assert_eq!(tokens[1].span.line, 2);
        assert_eq!(tokens[1].span.col, 1);
    }
}
