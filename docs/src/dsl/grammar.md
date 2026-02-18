# Grammar Specification

The **Shrew** language grammar is defined below in Extended Backus-Naur Form (EBNF).

```ebnf
SHREW LANGUAGE GRAMMAR v0.1

TOP LEVEL
program ::= { directive | import_stmt }

directive ::= metadata_block
            | config_block
            | types_block
            | graph_block
            | custom_op_block
            | training_block
            | inference_block
            | metrics_block
            | logging_block
            | visualization_block

IMPORTS
import_stmt ::= "@import" string_literal [ "as" identifier ] ";"

METADATA
metadata_block ::= "@model" "{" { metadata_field } "}"
metadata_field ::= identifier ":" literal ";"

CONFIGURATION
config_block ::= "@config" "{" { config_field } "}"
config_field ::= identifier ":" expr ";"

TYPE SYSTEM
types_block ::= "@types" "{" { type_def } "}"
type_def    ::= "type" identifier "=" type_expr ";"

type_expr ::= tensor_type
            | scalar_type
            | tuple_type
            | list_type
            | dict_type
            | identifier
            | "?"
            | integer_literal
            | binary_expr

tensor_type    ::= "Tensor" "<" "[" dimension_list "]" "," dtype ">"
dimension_list ::= dimension { "," dimension }
dimension      ::= identifier | integer_literal | "?" | "_" | binary_expr

dtype ::= "f32" | "f64" | "f16" | "bf16"
        | "i8" | "i16" | "i32" | "i64"
        | "u8" | "u16" | "u32" | "u64"
        | "bool" | "complex64" | "complex128"

scalar_type ::= dtype
tuple_type  ::= "(" type_expr { "," type_expr } ")"
list_type   ::= "[" type_expr "]"
dict_type   ::= "{" identifier ":" type_expr { "," identifier ":" type_expr } "}"

GRAPH DEFINITION
graph_block ::= "@graph" identifier [ "(" param_list ")" [ "->" type_expr ] ] "{" graph_body "}"
param_list  ::= param_def { "," param_def }
param_def   ::= identifier ":" type_expr [ "?" ]

graph_body ::= { graph_stmt }
graph_stmt ::= input_decl
             | output_decl
             | param_decl
             | node_decl
             | assert_stmt
             | check_stmt

input_decl  ::= "input" identifier ":" type_expr [ "?" ] ";"
output_decl ::= "output" [ identifier ":" ] expr ";"

param_decl  ::= "param" identifier ":" type_expr [ param_attrs ] ";"
param_attrs ::= "{" { param_attr } "}"
param_attr  ::= "init" ":" init_expr
              | "frozen" ":" bool_literal
              | "device" ":" device_expr
              | identifier ":" literal

init_expr ::= string_literal | expr

node_decl ::= "node" identifier [ ":" type_expr ] [ node_body ] ";"
node_body ::= "{" { node_stmt } "}"
node_stmt ::= "op" ":" operation ";"
            | "input" ":" expr ";"
            | "output" ":" type_expr ";"
            | hint_directive
            | identifier ":" expr ";"

operation ::= identifier "(" [ arg_list ] ")"
            | block_operation
            | call_operation
            | binary_expr
            | unary_expr

block_operation ::= "if" expr "{" operation "}" [ "else" "{" operation "}" ]
                  | "repeat" "(" expr ")" "{" operation "}"

call_operation       ::= "call" qualified_identifier "(" [ arg_list ] ")"
qualified_identifier ::= identifier { "." identifier | "::" identifier }
arg_list             ::= arg { "," arg }
arg                  ::= [ identifier ":" ] expr

assert_stmt     ::= "@assert" expr [ "," string_literal ] ";"
check_stmt      ::= "@check" identifier "{" { check_condition } "}"
check_condition ::= "assert" expr [ "," string_literal ] ";"

hint_directive ::= "@hint" hint_type ";"
hint_type      ::= "recompute_in_backward" | "must_preserve" | "in_place" | "no_grad" | identifier

CUSTOM OPERATORS
custom_op_block ::= "@custom_op" identifier "{" { custom_op_stmt } "}"
custom_op_stmt  ::= "signature" ":" signature ";"
                  | "impl" identifier "{" { impl_attr } "}"
                  | "gradient" identifier "{" gradient_body "}"

signature     ::= "(" param_list ")" "->" type_expr
impl_attr     ::= identifier ":" expr ";"
gradient_body ::= { gradient_stmt }
gradient_stmt ::= "impl" identifier "{" { impl_attr } "}"
                | "call" operation ";"

TRAINING CONFIGURATION
training_block ::= "@training" "{" { training_stmt } "}"
training_stmt  ::= "model" ":" identifier ";"
                 | "loss" ":" identifier ";"
                 | "optimizer" ":" optimizer_config ";"
                 | "lr_schedule" ":" schedule_config ";"
                 | "grad_clip" ":" clip_config ";"
                 | "precision" ":" string_literal ";"
                 | "accumulation_steps" ":" integer_literal ";"
                 | identifier ":" expr ";"

optimizer_config ::= "{" { optimizer_attr } "}"
optimizer_attr   ::= identifier ":" expr ";"
schedule_config  ::= "{" { schedule_attr } "}"
schedule_attr    ::= identifier ":" expr ";"
clip_config      ::= "{" { clip_attr } "}"
clip_attr        ::= identifier ":" expr ";"

INFERENCE CONFIGURATION
inference_block ::= "@inference" "{" { inference_stmt } "}"
inference_stmt  ::= "model" ":" identifier ";"
                  | "optimizations" ":" list_literal ";"
                  | "quantization" ":" quant_config ";"
                  | "generation" ":" generation_config ";"
                  | identifier ":" expr ";"

quant_config      ::= "{" { quant_attr } "}"
quant_attr        ::= identifier ":" expr ";"
generation_config ::= "{" { generation_attr } "}"
generation_attr   ::= identifier ":" expr ";"

METRICS & LOGGING
metrics_block ::= "@metrics" identifier "{" { metric_def } "}"
metric_def    ::= "track" identifier "{" { metric_attr } "}"
metric_attr   ::= "source" ":" source_expr ";"
                | "compute" ":" compute_expr ";"
                | "aggregate" ":" string_literal ";"
                | "type" ":" string_literal ";"
                | "log_every" ":" integer_literal ";"
                | identifier ":" expr ";"

source_expr    ::= qualified_identifier | "[" qualified_identifier { "," qualified_identifier } "]"
compute_expr   ::= expr | "{" iteration_expr "}"
iteration_expr ::= "for" identifier "in" expr "{" expr "}"

logging_block     ::= "@logging" "{" { logging_stmt } "}"
logging_stmt      ::= "backend" ":" string_literal ";"
                    | identifier ":" config_literal ";"
                    | "checkpoints" ":" checkpoint_config ";"

checkpoint_config ::= "{" { checkpoint_attr } "}"
checkpoint_attr   ::= identifier ":" expr ";"

visualization_block ::= "@visualizations" "{" { visualization_def } "}"
visualization_def   ::= "plot" identifier "{" { plot_attr } "}"
plot_attr           ::= identifier ":" expr ";"

EXPRESSIONS
expr ::= binary_expr | unary_expr | primary_expr

binary_expr ::= expr binary_op expr
binary_op   ::= "+" | "-" | "*" | "/" | "%" | "**"
              | "==" | "!=" | "<" | ">" | "<=" | ">="
              | "&&" | "||" | "??"
              | "&" | "|" | "^" | "<<" | ">>"

unary_expr ::= unary_op expr
unary_op   ::= "-" | "!" | "~"

primary_expr ::= identifier
               | literal
               | function_call
               | member_access
               | index_access
               | range_expr
               | list_expr
               | dict_expr
               | paren_expr
               | tensor_literal

function_call  ::= identifier "(" [ arg_list ] ")"
member_access  ::= expr "." identifier
index_access   ::= expr "[" expr [ ":" expr ] "]"
range_expr     ::= "range" "(" expr [ "," expr [ "," expr ] ] ")"
list_expr      ::= "[" [ expr { "," expr } ] "]"
dict_expr      ::= "{" [ dict_entry { "," dict_entry } ] "}"
dict_entry     ::= identifier ":" expr
paren_expr     ::= "(" expr ")"
tensor_literal ::= "[[" expr { "," expr } "]]"

LITERALS
literal ::= integer_literal
          | float_literal
          | bool_literal
          | string_literal
          | list_literal
          | config_literal
          | "null"

integer_literal ::= digit { digit }
float_literal   ::= digit { digit } "." digit { digit } [ exponent ]
                  | digit { digit } exponent
exponent        ::= ("e" | "E") [ "+" | "-" ] digit { digit }
bool_literal    ::= "true" | "false"
string_literal  ::= '"' { string_char } '"'
list_literal    ::= "[" [ literal { "," literal } ] "]"
config_literal  ::= "{" { identifier ":" literal { "," identifier ":" literal } } "}"

DEVICE EXPRESSIONS
device_expr ::= "cpu"
              | "gpu" [ ":" integer_literal ]
              | "tpu" [ ":" integer_literal ]
              | identifier

IDENTIFIERS
identifier ::= letter { letter | digit | "_" }
letter     ::= "a" .. "z" | "A" .. "Z"
digit      ::= "0" .. "9"
```
