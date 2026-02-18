// Parser — Recursive descent parser for the .sw language
//
// The parser consumes a Vec<Token> (from the Lexer) and produces an AST
// (Program). It's a classic hand-written recursive descent parser — one
// method per grammar production.
//
// ERROR RECOVERY: For now we use "panic mode" — on error, we return Err
// immediately. A future improvement could skip to the next @directive
// and collect multiple errors per parse.
//
// OPERATOR PRECEDENCE (lowest to highest):
//   1. ?? (null coalescing)
//   2. || (logical or)
//   3. && (logical and)
//   4. | (bitwise or)
//   5. ^ (bitwise xor)
//   6. & (bitwise and)
//   7. == != (equality)
//   8. < > <= >= (comparison)
//   9. << >> (shift)
//  10. + - (additive)
//  11. * / % (multiplicative)
//  12. ** (power, right-associative)
//  13. - ! ~ (unary prefix)
//  14. . [] () (postfix: member, index, call)

use crate::ast::*;
use crate::error::{Error, ErrorKind, Result};
use crate::token::{Span, Token, TokenKind};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parse the full program.
    pub fn parse_program(&mut self) -> Result<Program> {
        let mut items = Vec::new();
        while !self.at_eof() {
            items.push(self.parse_top_level()?);
        }
        Ok(Program { items })
    }

    // Top-level dispatch

    fn parse_top_level(&mut self) -> Result<TopLevel> {
        match self.peek_kind() {
            TokenKind::AtImport => Ok(TopLevel::Import(self.parse_import()?)),
            TokenKind::AtModel => Ok(TopLevel::Metadata(self.parse_metadata()?)),
            TokenKind::AtConfig => Ok(TopLevel::Config(self.parse_config()?)),
            TokenKind::AtTypes => Ok(TopLevel::Types(self.parse_types()?)),
            TokenKind::AtGraph => Ok(TopLevel::Graph(self.parse_graph()?)),
            TokenKind::AtCustomOp => Ok(TopLevel::CustomOp(self.parse_custom_op()?)),
            TokenKind::AtTraining => Ok(TopLevel::Training(self.parse_training()?)),
            TokenKind::AtInference => Ok(TopLevel::Inference(self.parse_inference()?)),
            TokenKind::AtMetrics => Ok(TopLevel::Metrics(self.parse_metrics()?)),
            TokenKind::AtLogging => Ok(TopLevel::Logging(self.parse_logging()?)),
            TokenKind::AtVisualizations => Ok(TopLevel::Visualization(self.parse_visualization()?)),
            _ => Err(self.error_unexpected("a directive (@model, @graph, etc.)")),
        }
    }

    // @import

    fn parse_import(&mut self) -> Result<ImportStmt> {
        let span = self.expect(TokenKind::AtImport)?.span;
        let path = self.expect_string()?;
        let alias = if self.check(&TokenKind::As) {
            self.advance();
            Some(self.expect_ident()?)
        } else {
            None
        };
        self.expect(TokenKind::Semi)?;
        Ok(ImportStmt { path, alias, span })
    }
  
    // @model

    fn parse_metadata(&mut self) -> Result<MetadataBlock> {
        let span = self.expect(TokenKind::AtModel)?.span;
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_field()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(MetadataBlock { fields, span })
    }

    fn parse_field(&mut self) -> Result<Field> {
        let span = self.current_span();
        let key = self.expect_key()?;
        self.expect(TokenKind::Colon)?;
        let value = self.parse_literal()?;
        self.expect(TokenKind::Semi)?;
        Ok(Field { key, value, span })
    }

    // @config

    fn parse_config(&mut self) -> Result<ConfigBlock> {
        let span = self.expect(TokenKind::AtConfig)?.span;
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_expr_field()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(ConfigBlock { fields, span })
    }

    fn parse_expr_field(&mut self) -> Result<ExprField> {
        let span = self.current_span();
        let key = self.expect_key()?;
        self.expect(TokenKind::Colon)?;
        let value = self.parse_expr()?;
        self.expect(TokenKind::Semi)?;
        Ok(ExprField { key, value, span })
    }
  
    // @types 

    fn parse_types(&mut self) -> Result<TypesBlock> {
        let span = self.expect(TokenKind::AtTypes)?.span;
        self.expect(TokenKind::LBrace)?;
        let mut defs = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            defs.push(self.parse_type_def()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(TypesBlock { defs, span })
    }

    fn parse_type_def(&mut self) -> Result<TypeDef> {
        let span = self.current_span();
        self.expect(TokenKind::Type)?;
        let name = self.expect_key()?;
        self.expect(TokenKind::Eq)?;
        let ty = self.parse_type_expr()?;
        self.expect(TokenKind::Semi)?;
        Ok(TypeDef { name, ty, span })
    }
 
    // Type expressions 

    fn parse_type_expr(&mut self) -> Result<TypeExpr> {
        let span = self.current_span();

        // Tensor<[dims], dtype>
        if self.check(&TokenKind::Tensor) {
            return self.parse_tensor_type();
        }

        // Tuple: (Type, Type, ...)
        if self.check(&TokenKind::LParen) {
            self.advance();
            let mut types = Vec::new();
            if !self.check(&TokenKind::RParen) {
                types.push(self.parse_type_expr()?);
                while self.check(&TokenKind::Comma) {
                    self.advance();
                    types.push(self.parse_type_expr()?);
                }
            }
            self.expect(TokenKind::RParen)?;
            return Ok(TypeExpr::Tuple(types, span));
        }

        // List: [Type]
        if self.check(&TokenKind::LBracket) {
            self.advance();
            let inner = self.parse_type_expr()?;
            self.expect(TokenKind::RBracket)?;
            return Ok(TypeExpr::List(Box::new(inner), span));
        }

        // Dynamic: ?
        if self.check(&TokenKind::Question) {
            self.advance();
            return Ok(TypeExpr::Dynamic(span));
        }

        // Scalar dtype
        if let Some(dt) = self.try_parse_dtype() {
            return Ok(TypeExpr::Scalar(dt, span));
        }

        // Concrete int dimension
        if let TokenKind::IntLit(n) = self.peek_kind() {
            self.advance();
            return Ok(TypeExpr::IntDim(n, span));
        }

        // Named type alias
        if self.is_ident_or_keyword_as_ident() {
            let name = self.expect_key()?;
            return Ok(TypeExpr::Named(name, span));
        }

        Err(self.error_unexpected("a type expression"))
    }

    fn parse_tensor_type(&mut self) -> Result<TypeExpr> {
        let span = self.current_span();
        self.expect(TokenKind::Tensor)?;
        self.expect(TokenKind::Lt)?;
        self.expect(TokenKind::LBracket)?;

        let mut dims = Vec::new();
        if !self.check(&TokenKind::RBracket) {
            dims.push(self.parse_dimension()?);
            while self.check(&TokenKind::Comma) {
                self.advance();
                if self.check(&TokenKind::RBracket) {
                    break;
                }
                dims.push(self.parse_dimension()?);
            }
        }
        self.expect(TokenKind::RBracket)?;
        self.expect(TokenKind::Comma)?;
        let dtype = self.parse_dtype()?;
        self.expect(TokenKind::Gt)?;
        Ok(TypeExpr::Tensor { dims, dtype, span })
    }

    fn parse_dimension(&mut self) -> Result<Dimension> {
        let span = self.current_span();

        if self.check(&TokenKind::Question) {
            self.advance();
            return Ok(Dimension::Dynamic(span));
        }
        if self.check(&TokenKind::Underscore) {
            self.advance();
            return Ok(Dimension::Inferred(span));
        }
        if let TokenKind::IntLit(n) = self.peek_kind() {
            self.advance();
            return Ok(Dimension::Concrete(n, span));
        }
        if self.is_ident_or_keyword_as_ident() {
            let name = self.expect_key()?;
            return Ok(Dimension::Named(name, span));
        }
        Err(self.error_unexpected("a dimension (name, integer, ?, or _)"))
    }

    fn parse_dtype(&mut self) -> Result<DTypeKind> {
        self.try_parse_dtype()
            .ok_or_else(|| self.error_unexpected("a dtype (f32, f64, i64, etc.)"))
    }

    fn try_parse_dtype(&mut self) -> Option<DTypeKind> {
        let dt = match self.peek_kind() {
            TokenKind::F16 => DTypeKind::F16,
            TokenKind::F32 => DTypeKind::F32,
            TokenKind::F64 => DTypeKind::F64,
            TokenKind::Bf16 => DTypeKind::Bf16,
            TokenKind::I8 => DTypeKind::I8,
            TokenKind::I16 => DTypeKind::I16,
            TokenKind::I32 => DTypeKind::I32,
            TokenKind::I64 => DTypeKind::I64,
            TokenKind::U8 => DTypeKind::U8,
            TokenKind::U16 => DTypeKind::U16,
            TokenKind::U32 => DTypeKind::U32,
            TokenKind::U64 => DTypeKind::U64,
            TokenKind::Bool => DTypeKind::Bool,
            TokenKind::Complex64 => DTypeKind::Complex64,
            TokenKind::Complex128 => DTypeKind::Complex128,
            _ => return None,
        };
        self.advance();
        Some(dt)
    }

    // @graph

    fn parse_graph(&mut self) -> Result<GraphBlock> {
        let span = self.expect(TokenKind::AtGraph)?.span;
        let name = self.expect_key()?;

        // Optional param list
        let params = if self.check(&TokenKind::LParen) {
            self.advance();
            let ps = self.parse_param_list()?;
            self.expect(TokenKind::RParen)?;
            ps
        } else {
            Vec::new()
        };

        // Optional return type
        let return_type = if self.check(&TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type_expr()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;
        let mut body = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            body.push(self.parse_graph_stmt()?);
        }
        self.expect(TokenKind::RBrace)?;

        Ok(GraphBlock {
            name,
            params,
            return_type,
            body,
            span,
        })
    }

    fn parse_param_list(&mut self) -> Result<Vec<ParamDef>> {
        let mut params = Vec::new();
        if !self.check(&TokenKind::RParen) {
            params.push(self.parse_param_def()?);
            while self.check(&TokenKind::Comma) {
                self.advance();
                if self.check(&TokenKind::RParen) {
                    break;
                }
                params.push(self.parse_param_def()?);
            }
        }
        Ok(params)
    }

    fn parse_param_def(&mut self) -> Result<ParamDef> {
        let span = self.current_span();
        let name = self.expect_key()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type_expr()?;
        let optional = if self.check(&TokenKind::Question) {
            self.advance();
            true
        } else {
            false
        };
        Ok(ParamDef {
            name,
            ty,
            optional,
            span,
        })
    }

    fn parse_graph_stmt(&mut self) -> Result<GraphStmt> {
        match self.peek_kind() {
            TokenKind::Input => {
                let span = self.advance().span;
                let name = self.expect_key()?;
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type_expr()?;
                let optional = if self.check(&TokenKind::Question) {
                    self.advance();
                    true
                } else {
                    false
                };
                self.expect(TokenKind::Semi)?;
                Ok(GraphStmt::Input(InputDecl {
                    name,
                    ty,
                    optional,
                    span,
                }))
            }
            TokenKind::Output => {
                let span = self.advance().span;
                // output [name :] expr ;
                let (name, expr) = if self.is_ident_or_keyword_as_ident()
                    && self.peek_ahead_kind(1) == TokenKind::Colon
                {
                    let n = self.expect_key()?;
                    self.expect(TokenKind::Colon)?;
                    (Some(n), self.parse_expr()?)
                } else {
                    (None, self.parse_expr()?)
                };
                self.expect(TokenKind::Semi)?;
                Ok(GraphStmt::Output(OutputDecl { name, expr, span }))
            }
            TokenKind::Param => {
                let span = self.advance().span;
                let name = self.expect_key()?;
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type_expr()?;
                let attrs = if self.check(&TokenKind::LBrace) {
                    self.advance();
                    let mut attrs = Vec::new();
                    while !self.check(&TokenKind::RBrace) {
                        let aspan = self.current_span();
                        let key = self.expect_key()?;
                        self.expect(TokenKind::Colon)?;
                        let value = self.parse_expr()?;
                        self.expect(TokenKind::Semi)?;
                        attrs.push(ParamAttr {
                            key,
                            value,
                            span: aspan,
                        });
                    }
                    self.expect(TokenKind::RBrace)?;
                    attrs
                } else {
                    Vec::new()
                };
                self.expect(TokenKind::Semi)?;
                Ok(GraphStmt::Param(ParamDecl {
                    name,
                    ty,
                    attrs,
                    span,
                }))
            }
            TokenKind::Node => {
                let span = self.advance().span;
                let name = self.expect_key()?;
                let ty = if self.check(&TokenKind::Colon) {
                    self.advance();
                    // Could be a type_expr OR a node_body starting with {
                    if self.check(&TokenKind::LBrace) {
                        None // the { is the node body, not a dict type
                    } else {
                        Some(self.parse_type_expr()?)
                    }
                } else {
                    None
                };
                let stmts = if self.check(&TokenKind::LBrace) {
                    self.advance();
                    let mut stmts = Vec::new();
                    while !self.check(&TokenKind::RBrace) {
                        stmts.push(self.parse_node_stmt()?);
                    }
                    self.expect(TokenKind::RBrace)?;
                    stmts
                } else {
                    Vec::new()
                };
                self.expect(TokenKind::Semi)?;
                Ok(GraphStmt::Node(NodeDecl {
                    name,
                    ty,
                    stmts,
                    span,
                }))
            }
            TokenKind::AtAssert => {
                let span = self.advance().span;
                let condition = self.parse_expr()?;
                let message = if self.check(&TokenKind::Comma) {
                    self.advance();
                    Some(self.expect_string()?)
                } else {
                    None
                };
                self.expect(TokenKind::Semi)?;
                Ok(GraphStmt::Assert(AssertStmt {
                    condition,
                    message,
                    span,
                }))
            }
            TokenKind::AtCheck => {
                let span = self.advance().span;
                let name = self.expect_key()?;
                self.expect(TokenKind::LBrace)?;
                let mut conditions = Vec::new();
                while !self.check(&TokenKind::RBrace) {
                    // Each line is: assert expr [, "msg"] ;
                    let aspan = self.current_span();
                    // "assert" is not a keyword in our grammar but appears
                    // inside @check. We'll accept it as an ident or keyword.
                    let word = self.expect_key()?;
                    if word != "assert" {
                        return Err(Error::new(
                            ErrorKind::UnexpectedToken {
                                expected: "assert".to_string(),
                                got: word,
                            },
                            aspan,
                        ));
                    }
                    let condition = self.parse_expr()?;
                    let message = if self.check(&TokenKind::Comma) {
                        self.advance();
                        Some(self.expect_string()?)
                    } else {
                        None
                    };
                    self.expect(TokenKind::Semi)?;
                    conditions.push(AssertStmt {
                        condition,
                        message,
                        span: aspan,
                    });
                }
                self.expect(TokenKind::RBrace)?;
                Ok(GraphStmt::Check(CheckBlock {
                    name,
                    conditions,
                    span,
                }))
            }
            _ => Err(self.error_unexpected(
                "a graph statement (input, output, param, node, @assert, @check)",
            )),
        }
    }

    fn parse_node_stmt(&mut self) -> Result<NodeStmt> {
        let span = self.current_span();

        // @hint
        if self.check(&TokenKind::AtHint) {
            self.advance();
            let kind = match self.peek_kind() {
                TokenKind::RecomputeInBackward => {
                    self.advance();
                    HintKind::RecomputeInBackward
                }
                TokenKind::MustPreserve => {
                    self.advance();
                    HintKind::MustPreserve
                }
                TokenKind::InPlace => {
                    self.advance();
                    HintKind::InPlace
                }
                TokenKind::NoGrad => {
                    self.advance();
                    HintKind::NoGrad
                }
                _ if self.is_ident() => {
                    let name = self.expect_ident()?;
                    HintKind::Custom(name)
                }
                _ => return Err(self.error_unexpected("a hint type")),
            };
            self.expect(TokenKind::Semi)?;
            return Ok(NodeStmt::Hint(kind, span));
        }

        // keyword : expr ;
        let key = self.expect_key()?;
        self.expect(TokenKind::Colon)?;

        let stmt = match key.as_str() {
            "op" => {
                let expr = self.parse_expr()?;
                self.expect(TokenKind::Semi)?;
                NodeStmt::Op(expr, span)
            }
            "input" => {
                let expr = self.parse_expr()?;
                self.expect(TokenKind::Semi)?;
                NodeStmt::InputRef(expr, span)
            }
            "output" => {
                let ty = self.parse_type_expr()?;
                self.expect(TokenKind::Semi)?;
                NodeStmt::OutputType(ty, span)
            }
            _ => {
                let expr = self.parse_expr()?;
                self.expect(TokenKind::Semi)?;
                NodeStmt::Attr(key, expr, span)
            }
        };
        Ok(stmt)
    }

    // @custom_op

    fn parse_custom_op(&mut self) -> Result<CustomOpBlock> {
        let span = self.expect(TokenKind::AtCustomOp)?.span;
        let name = self.expect_key()?;
        self.expect(TokenKind::LBrace)?;
        let mut stmts = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            stmts.push(self.parse_custom_op_stmt()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(CustomOpBlock { name, stmts, span })
    }

    fn parse_custom_op_stmt(&mut self) -> Result<CustomOpStmt> {
        let span = self.current_span();
        match self.peek_kind() {
            TokenKind::Signature => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                self.expect(TokenKind::LParen)?;
                let params = self.parse_param_list()?;
                self.expect(TokenKind::RParen)?;
                self.expect(TokenKind::Arrow)?;
                let return_type = self.parse_type_expr()?;
                self.expect(TokenKind::Semi)?;
                Ok(CustomOpStmt::Signature {
                    params,
                    return_type,
                    span,
                })
            }
            TokenKind::Impl => {
                self.advance();
                let target = self.expect_key()?;
                self.expect(TokenKind::LBrace)?;
                let mut attrs = Vec::new();
                while !self.check(&TokenKind::RBrace) {
                    attrs.push(self.parse_expr_field()?);
                }
                self.expect(TokenKind::RBrace)?;
                Ok(CustomOpStmt::Impl {
                    target,
                    attrs,
                    span,
                })
            }
            TokenKind::Gradient => {
                self.advance();
                let target = self.expect_key()?;
                self.expect(TokenKind::LBrace)?;
                let mut body = Vec::new();
                while !self.check(&TokenKind::RBrace) {
                    body.push(self.parse_custom_op_stmt()?);
                }
                self.expect(TokenKind::RBrace)?;
                Ok(CustomOpStmt::Gradient { target, body, span })
            }
            _ => Err(self.error_unexpected("signature, impl, or gradient")),
        }
    }

    // @training

    fn parse_training(&mut self) -> Result<TrainingBlock> {
        let span = self.expect(TokenKind::AtTraining)?.span;
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_training_field()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(TrainingBlock { fields, span })
    }

    fn parse_training_field(&mut self) -> Result<TrainingField> {
        let span = self.current_span();
        match self.peek_kind() {
            TokenKind::Model => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                let name = self.expect_key()?;
                self.expect(TokenKind::Semi)?;
                Ok(TrainingField::Model(name, span))
            }
            TokenKind::Loss => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                let name = self.expect_key()?;
                self.expect(TokenKind::Semi)?;
                Ok(TrainingField::Loss(name, span))
            }
            TokenKind::Optimizer => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                let fields = self.parse_brace_fields()?;
                Ok(TrainingField::Optimizer(fields, span))
            }
            TokenKind::LrSchedule => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                let fields = self.parse_brace_fields()?;
                Ok(TrainingField::LrSchedule(fields, span))
            }
            TokenKind::GradClip => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                let fields = self.parse_brace_fields()?;
                Ok(TrainingField::GradClip(fields, span))
            }
            _ => {
                let f = self.parse_expr_field()?;
                Ok(TrainingField::Generic(f))
            }
        }
    }

    // @inference

    fn parse_inference(&mut self) -> Result<InferenceBlock> {
        let span = self.expect(TokenKind::AtInference)?.span;
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_inference_field()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(InferenceBlock { fields, span })
    }

    fn parse_inference_field(&mut self) -> Result<InferenceField> {
        let span = self.current_span();
        match self.peek_kind() {
            TokenKind::Model => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                let name = self.expect_key()?;
                self.expect(TokenKind::Semi)?;
                Ok(InferenceField::Model(name, span))
            }
            TokenKind::Optimizations => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                let list = self.parse_list_expr_items()?;
                self.expect(TokenKind::Semi)?;
                Ok(InferenceField::Optimizations(list, span))
            }
            TokenKind::Quantization => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                let fields = self.parse_brace_fields()?;
                Ok(InferenceField::Quantization(fields, span))
            }
            TokenKind::Generation => {
                self.advance();
                self.expect(TokenKind::Colon)?;
                let fields = self.parse_brace_fields()?;
                Ok(InferenceField::Generation(fields, span))
            }
            _ => {
                let f = self.parse_expr_field()?;
                Ok(InferenceField::Generic(f))
            }
        }
    }

    // @metrics

    fn parse_metrics(&mut self) -> Result<MetricsBlock> {
        let span = self.expect(TokenKind::AtMetrics)?.span;
        let name = self.expect_key()?;
        self.expect(TokenKind::LBrace)?;
        let mut defs = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            defs.push(self.parse_metric_def()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(MetricsBlock { name, defs, span })
    }

    fn parse_metric_def(&mut self) -> Result<MetricDef> {
        let span = self.current_span();
        self.expect(TokenKind::Track)?;
        let name = self.expect_key()?;
        self.expect(TokenKind::LBrace)?;
        let mut attrs = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            attrs.push(self.parse_expr_field()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(MetricDef { name, attrs, span })
    }
 
    // @logging

    fn parse_logging(&mut self) -> Result<LoggingBlock> {
        let span = self.expect(TokenKind::AtLogging)?.span;
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_expr_field()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(LoggingBlock { fields, span })
    }
 
    // @visualizations

    fn parse_visualization(&mut self) -> Result<VisualizationBlock> {
        let span = self.expect(TokenKind::AtVisualizations)?.span;
        self.expect(TokenKind::LBrace)?;
        let mut plots = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            plots.push(self.parse_plot_def()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(VisualizationBlock { plots, span })
    }

    fn parse_plot_def(&mut self) -> Result<PlotDef> {
        let span = self.current_span();
        self.expect(TokenKind::Plot)?;
        let name = self.expect_key()?;
        self.expect(TokenKind::LBrace)?;
        let mut attrs = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            attrs.push(self.parse_expr_field()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(PlotDef { name, attrs, span })
    }
 
    // Expressions (Pratt parser with precedence climbing)

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_expr_bp(0)
    }

    /// Pratt parser: parse expression with minimum binding power `min_bp`.
    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr> {
        let mut lhs = self.parse_prefix()?;

        loop {
            // Postfix: . [] ()
            lhs = match self.peek_kind() {
                TokenKind::Dot => {
                    let span = lhs.span();
                    self.advance();
                    let field = self.expect_key()?;
                    // Check for method call: expr.method(args)
                    if self.check(&TokenKind::LParen) {
                        self.advance();
                        let args = self.parse_arg_list()?;
                        self.expect(TokenKind::RParen)?;
                        Expr::QualifiedCall {
                            path: vec![format!("{}", self.expr_to_string(&lhs)), field],
                            args,
                            span,
                        }
                    } else {
                        Expr::Member {
                            object: Box::new(lhs),
                            field,
                            span,
                        }
                    }
                }
                TokenKind::LBracket => {
                    let span = lhs.span();
                    self.advance();
                    let index = self.parse_expr()?;
                    let end = if self.check(&TokenKind::Colon) {
                        self.advance();
                        Some(Box::new(self.parse_expr()?))
                    } else {
                        None
                    };
                    self.expect(TokenKind::RBracket)?;
                    Expr::Index {
                        object: Box::new(lhs),
                        index: Box::new(index),
                        end,
                        span,
                    }
                }
                _ => break,
            };
        }

        loop {
            let (op, bp) = match self.peek_kind() {
                TokenKind::QuestionQuestion => (BinOp::NullCoalesce, (1, 2)),
                TokenKind::PipePipe => (BinOp::Or, (3, 4)),
                TokenKind::AmpAmp => (BinOp::And, (5, 6)),
                TokenKind::Pipe => (BinOp::BitOr, (7, 8)),
                TokenKind::Caret => (BinOp::BitXor, (9, 10)),
                TokenKind::Amp => (BinOp::BitAnd, (11, 12)),
                TokenKind::EqEq => (BinOp::Eq, (13, 14)),
                TokenKind::BangEq => (BinOp::Ne, (13, 14)),
                TokenKind::Lt => (BinOp::Lt, (15, 16)),
                TokenKind::Gt => (BinOp::Gt, (15, 16)),
                TokenKind::LtEq => (BinOp::Le, (15, 16)),
                TokenKind::GtEq => (BinOp::Ge, (15, 16)),
                TokenKind::LtLt => (BinOp::Shl, (17, 18)),
                TokenKind::GtGt => (BinOp::Shr, (17, 18)),
                TokenKind::Plus => (BinOp::Add, (19, 20)),
                TokenKind::Minus => (BinOp::Sub, (19, 20)),
                TokenKind::Star => (BinOp::Mul, (21, 22)),
                TokenKind::Slash => (BinOp::Div, (21, 22)),
                TokenKind::Percent => (BinOp::Mod, (21, 22)),
                TokenKind::StarStar => (BinOp::Pow, (24, 23)), // right-assoc
                _ => break,
            };

            let (l_bp, r_bp) = bp;
            if l_bp < min_bp {
                break;
            }

            self.advance();
            let span = lhs.span();
            let rhs = self.parse_expr_bp(r_bp)?;
            lhs = Expr::Binary {
                left: Box::new(lhs),
                op,
                right: Box::new(rhs),
                span,
            };
        }

        Ok(lhs)
    }

    /// Parse a prefix expression (unary or primary).
    fn parse_prefix(&mut self) -> Result<Expr> {
        let span = self.current_span();
        match self.peek_kind() {
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_expr_bp(25)?; // unary binds tighter than binary
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                    span,
                })
            }
            TokenKind::Bang => {
                self.advance();
                let operand = self.parse_expr_bp(25)?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                    span,
                })
            }
            TokenKind::Tilde => {
                self.advance();
                let operand = self.parse_expr_bp(25)?;
                Ok(Expr::Unary {
                    op: UnaryOp::BitNot,
                    operand: Box::new(operand),
                    span,
                })
            }
            _ => self.parse_primary(),
        }
    }

    /// Parse a primary expression (literals, identifiers, calls, etc.).
    fn parse_primary(&mut self) -> Result<Expr> {
        let span = self.current_span();
        match self.peek_kind() {
            TokenKind::IntLit(n) => {
                self.advance();
                Ok(Expr::Int(n, span))
            }
            TokenKind::FloatLit(n) => {
                self.advance();
                Ok(Expr::Float(n, span))
            }
            TokenKind::StringLit(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Expr::Str(s, span))
            }
            TokenKind::True => {
                self.advance();
                Ok(Expr::Bool(true, span))
            }
            TokenKind::False => {
                self.advance();
                Ok(Expr::Bool(false, span))
            }
            TokenKind::Null => {
                self.advance();
                Ok(Expr::Null(span))
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(Expr::Paren(Box::new(expr), span))
            }
            TokenKind::LBracket => {
                let items = self.parse_list_expr_items()?;
                Ok(Expr::List(items, span))
            }
            TokenKind::LBrace => {
                self.advance();
                let mut entries = Vec::new();
                while !self.check(&TokenKind::RBrace) {
                    let key = self.expect_key()?;
                    self.expect(TokenKind::Colon)?;
                    let value = self.parse_expr()?;
                    entries.push((key, value));
                    if !self.check(&TokenKind::RBrace) {
                        self.expect(TokenKind::Comma)?;
                    }
                }
                self.expect(TokenKind::RBrace)?;
                Ok(Expr::Dict(entries, span))
            }
            TokenKind::If => {
                self.advance();
                let cond = self.parse_expr()?;
                self.expect(TokenKind::LBrace)?;
                let then_branch = self.parse_expr()?;
                self.expect(TokenKind::RBrace)?;
                let else_branch = if self.check(&TokenKind::Else) {
                    self.advance();
                    self.expect(TokenKind::LBrace)?;
                    let e = self.parse_expr()?;
                    self.expect(TokenKind::RBrace)?;
                    Some(Box::new(e))
                } else {
                    None
                };
                Ok(Expr::IfExpr {
                    cond: Box::new(cond),
                    then_branch: Box::new(then_branch),
                    else_branch,
                    span,
                })
            }
            TokenKind::Repeat => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let count = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                self.expect(TokenKind::LBrace)?;
                let body = self.parse_expr()?;
                self.expect(TokenKind::RBrace)?;
                Ok(Expr::RepeatExpr {
                    count: Box::new(count),
                    body: Box::new(body),
                    span,
                })
            }
            TokenKind::Pipe => {
                // Closure: |a, b| { expr }
                self.advance();
                let mut params = Vec::new();
                if !self.check(&TokenKind::Pipe) {
                    params.push(self.expect_ident()?);
                    while self.check(&TokenKind::Comma) {
                        self.advance();
                        params.push(self.expect_ident()?);
                    }
                }
                self.expect(TokenKind::Pipe)?;
                self.expect(TokenKind::LBrace)?;
                let body = self.parse_expr()?;
                self.expect(TokenKind::RBrace)?;
                Ok(Expr::Closure {
                    params,
                    body: Box::new(body),
                    span,
                })
            }
            _ if self.is_ident_or_keyword_as_ident() => {
                let name = self.consume_as_ident()?;
                // Check for function call
                if self.check(&TokenKind::LParen) {
                    self.advance();
                    let args = self.parse_arg_list()?;
                    self.expect(TokenKind::RParen)?;
                    Ok(Expr::Call {
                        func: name,
                        args,
                        span,
                    })
                } else if self.check(&TokenKind::ColonColon) {
                    // Qualified path: mod::func(args)
                    let mut path = vec![name];
                    while self.check(&TokenKind::ColonColon) {
                        self.advance();
                        path.push(self.expect_key()?);
                    }
                    if self.check(&TokenKind::LParen) {
                        self.advance();
                        let args = self.parse_arg_list()?;
                        self.expect(TokenKind::RParen)?;
                        Ok(Expr::QualifiedCall { path, args, span })
                    } else {
                        // Just a qualified ident; return last as ident
                        let full = path.join("::");
                        Ok(Expr::Ident(full, span))
                    }
                } else {
                    Ok(Expr::Ident(name, span))
                }
            }
            _ => Err(self.error_unexpected("an expression")),
        }
    }

    fn parse_arg_list(&mut self) -> Result<Vec<Arg>> {
        let mut args = Vec::new();
        if self.check(&TokenKind::RParen) {
            return Ok(args);
        }
        args.push(self.parse_arg()?);
        while self.check(&TokenKind::Comma) {
            self.advance();
            if self.check(&TokenKind::RParen) {
                break;
            }
            args.push(self.parse_arg()?);
        }
        Ok(args)
    }

    fn parse_arg(&mut self) -> Result<Arg> {
        let span = self.current_span();
        // Try named arg: key : expr (key can be ident or keyword)
        let is_key = matches!(self.peek_kind(), TokenKind::Ident(_))
            || self.peek_kind().keyword_str().is_some();
        if is_key && self.peek_ahead_kind(1) == TokenKind::Colon {
            let name = self.expect_key()?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_expr()?;
            return Ok(Arg {
                name: Some(name),
                value,
                span,
            });
        }
        let value = self.parse_expr()?;
        Ok(Arg {
            name: None,
            value,
            span,
        })
    }

    fn parse_list_expr_items(&mut self) -> Result<Vec<Expr>> {
        self.expect(TokenKind::LBracket)?;
        let mut items = Vec::new();
        if !self.check(&TokenKind::RBracket) {
            items.push(self.parse_expr()?);
            while self.check(&TokenKind::Comma) {
                self.advance();
                if self.check(&TokenKind::RBracket) {
                    break;
                }
                items.push(self.parse_expr()?);
            }
        }
        self.expect(TokenKind::RBracket)?;
        Ok(items)
    }

    
    // Literals
    

    fn parse_literal(&mut self) -> Result<Literal> {
        let span = self.current_span();
        match self.peek_kind() {
            TokenKind::IntLit(n) => {
                self.advance();
                Ok(Literal::Int(n, span))
            }
            TokenKind::FloatLit(n) => {
                self.advance();
                Ok(Literal::Float(n, span))
            }
            TokenKind::StringLit(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Literal::Str(s, span))
            }
            TokenKind::True => {
                self.advance();
                Ok(Literal::Bool(true, span))
            }
            TokenKind::False => {
                self.advance();
                Ok(Literal::Bool(false, span))
            }
            TokenKind::Null => {
                self.advance();
                Ok(Literal::Null(span))
            }
            TokenKind::LBracket => {
                self.advance();
                let mut items = Vec::new();
                if !self.check(&TokenKind::RBracket) {
                    items.push(self.parse_literal()?);
                    while self.check(&TokenKind::Comma) {
                        self.advance();
                        if self.check(&TokenKind::RBracket) {
                            break;
                        }
                        items.push(self.parse_literal()?);
                    }
                }
                self.expect(TokenKind::RBracket)?;
                Ok(Literal::List(items, span))
            }
            TokenKind::LBrace => {
                self.advance();
                let mut entries = Vec::new();
                while !self.check(&TokenKind::RBrace) {
                    let key = self.expect_key()?;
                    self.expect(TokenKind::Colon)?;
                    let val = self.parse_literal()?;
                    entries.push((key, val));
                    if !self.check(&TokenKind::RBrace) {
                        self.expect(TokenKind::Comma)?;
                    }
                }
                self.expect(TokenKind::RBrace)?;
                Ok(Literal::Dict(entries, span))
            }
            _ => Err(self.error_unexpected("a literal value")),
        }
    }

    // Helpers: { key: expr; ... } blocks

    fn parse_brace_fields(&mut self) -> Result<Vec<ExprField>> {
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_expr_field()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(fields)
    }

    // Token stream helpers

    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn peek_kind(&self) -> TokenKind {
        self.peek().kind.clone()
    }

    fn peek_ahead_kind(&self, offset: usize) -> TokenKind {
        let idx = (self.pos + offset).min(self.tokens.len() - 1);
        self.tokens[idx].kind.clone()
    }

    fn current_span(&self) -> Span {
        self.peek().span
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Eof)
    }

    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek_kind()) == std::mem::discriminant(kind)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.pos.min(self.tokens.len() - 1)].clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token> {
        if self.check(&kind) {
            Ok(self.advance())
        } else {
            Err(Error::new(
                ErrorKind::UnexpectedToken {
                    expected: format!("{}", kind),
                    got: format!("{}", self.peek_kind()),
                },
                self.current_span(),
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.peek_kind() {
            TokenKind::Ident(s) => {
                self.advance();
                Ok(s)
            }
            _ => Err(self.error_unexpected("an identifier")),
        }
    }

    /// Consume an identifier OR keyword token as a string key.
    /// Used in field-key positions where keywords like `init`, `source`,
    /// `frozen`, `op`, `backend`, etc. can appear as dictionary keys.
    fn expect_key(&mut self) -> Result<String> {
        match self.peek_kind() {
            TokenKind::Ident(s) => {
                self.advance();
                Ok(s)
            }
            ref kind if kind.keyword_str().is_some() => {
                let s = kind.keyword_str().unwrap().to_string();
                self.advance();
                Ok(s)
            }
            _ => Err(self.error_unexpected("a key (identifier or keyword)")),
        }
    }

    fn expect_string(&mut self) -> Result<String> {
        match self.peek_kind() {
            TokenKind::StringLit(s) => {
                self.advance();
                Ok(s)
            }
            _ => Err(self.error_unexpected("a string literal")),
        }
    }

    fn is_ident(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Ident(_))
    }

    /// Check if current token is an identifier or a keyword that can double
    /// as an identifier in expression context.
    fn is_ident_or_keyword_as_ident(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Ident(_)) || self.peek_kind().keyword_str().is_some()
    }

    /// Consume the current token as an identifier string, even if it's a keyword
    /// that can be used as an ident in expression context.
    fn consume_as_ident(&mut self) -> Result<String> {
        self.expect_key()
    }

    fn error_unexpected(&self, expected: &str) -> Error {
        Error::new(
            ErrorKind::UnexpectedToken {
                expected: expected.to_string(),
                got: format!("{}", self.peek_kind()),
            },
            self.current_span(),
        )
    }

    fn expr_to_string(&self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(name, _) => name.clone(),
            _ => "<expr>".to_string(),
        }
    }
}

// Public convenience function

/// Parse a .sw source string into an AST Program.
pub fn parse(source: &str) -> Result<Program> {
    let tokens = crate::lexer::Lexer::new(source).tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}
