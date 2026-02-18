// Shrew v1.0 Demo — MLP for XOR classification
//
// A simple 2-layer MLP that learns the XOR function:
//   [0,0] → 0, [0,1] → 1, [1,0] → 1, [1,1] → 0
//
// Run with:  cargo run -p shrew-cli -- run examples/demo_mlp.sw --verbose
// Inspect:   cargo run -p shrew-cli -- info examples/demo_mlp.sw
// Validate:  cargo run -p shrew-cli -- validate examples/demo_mlp.sw

@model {
    name: "XOR_MLP";
    version: "1.0";
}

@graph Forward {
    // ── Inputs ──
    input x: Tensor<[4, 2], f64>;     // 4 XOR samples, 2 features each

    // ── Hidden layer (2 → 8) ──
    param w1: Tensor<[2, 8], f64>  { init: "xavier_uniform"; };
    param b1: Tensor<[1, 8], f64>  { init: "zeros"; };

    // ── Output layer (8 → 1) ──
    param w2: Tensor<[8, 1], f64>  { init: "xavier_uniform"; };
    param b2: Tensor<[1, 1], f64>  { init: "zeros"; };

    // ── Forward pass ──
    node h1     { op: matmul(x, w1) + b1; };   // [4,2] @ [2,8] + [1,8] = [4,8]
    node a1     { op: relu(h1); };               // activation
    node logits { op: matmul(a1, w2) + b2; };   // [4,8] @ [8,1] + [1,1] = [4,1]
    node out    { op: sigmoid(logits); };         // probability output

    output out;
}
