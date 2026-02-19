// Wine Classifier — Multi-class classification for the Wine dataset
//
// Input:  13 features (alcohol, malic_acid, ash, ...)
// Output: 3 logits  (class_0, class_1, class_2)
//
// Architecture: 13 → 64 (relu) → 32 (relu) → 3
// Loss: cross_entropy

@model {
    name: "WineClassifier";
    version: "1.0";
    author: "ginozza";
}

// When @config includes `device`, the executor can select the backend.
// Possible values: "cpu", "cuda", "cuda:0", "cuda:1", "auto"
// "auto" will pick CUDA if available, otherwise CPU.
@config {
    device: "auto";
    dtype: "f32";
    seed: 42;
}

@graph forward {
    //  Input
    input x: Tensor<[?, 13], f32>;

    //  Hidden layer 1: 13 → 64
    param w1: Tensor<[13, 64], f32>  { init: "xavier_uniform"; };
    param b1: Tensor<[1, 64], f32>   { init: "zeros"; };

    node h1 { op: matmul(x, w1) + b1; };
    node a1 { op: relu(h1); };

    //  Hidden layer 2: 64 → 32
    param w2: Tensor<[64, 32], f32>  { init: "xavier_uniform"; };
    param b2: Tensor<[1, 32], f32>   { init: "zeros"; };

    node h2 { op: matmul(a1, w2) + b2; };
    node a2 { op: relu(h2); };

    //  Output layer: 32 → 3
    param w3: Tensor<[32, 3], f32>   { init: "xavier_uniform"; };
    param b3: Tensor<[1, 3], f32>    { init: "zeros"; };

    node out { op: matmul(a2, w3) + b3; };

    output out;
}

@training {
    model: forward;
    loss: cross_entropy;

    optimizer: {
        type: "AdamW";
        lr: 0.005;
        weight_decay: 0.01;
    }

    epochs: 400;
    batch_size: 32;
}
