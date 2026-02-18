# Graphs & Operations

The `@graph` directive defines a computation graph, similar to a function in other languages.

## Anatomy of a Graph

```sw
@graph GraphName(input1: Type1) -> OutputType {
    // 1. Parameter Declarations (Weights)
    param w: Type2 { init: "start_value"; };

    // 2. Nodes (Operations)
    node x: Type3 {
        op: input1 * w;
    };

    // 3. Output Statement
    output x;
}
```

## Nodes

Nodes represent intermediate computations. They must have an `op` field defining the operation.

```sw
node activation {
    op: relu(x);
}
```

## Standard Operations

Shrew supports standard element-wise arithmetic and matrix operations:
- `+`, `-`, `*`, `/`
- `matmul` or `@` operator
- `pow` (`**`)

And common neural network functions:
- `relu()`, `sigmoid()`, `tanh()`
- `softmax()`
- `conv2d()`, `maxpool2d()`

## Attributes

Nodes can have additional attributes passed in the block:

```sw
node conv_layer {
    op: conv2d(input, weight);
    padding: 1;
    stride: 2;
}
```
