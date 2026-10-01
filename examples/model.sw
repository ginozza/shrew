// Shrew Deep Learning DSL — Self-Contained XOR Classification Model
// 
// All architecture, hyperparameters, training epochs, optimizer, 
// and dataset paths are declared directly inside this .sw file.
// The host programming language only needs 2-5 lines of code to run or train it.

@model {
    name: "XOR_Model";
    version: "1.0";
    author: "Shrew";
}

@config {
    device: "auto";
    dtype: "f64";
    seed: 42;
}

@graph Forward {
    // Inputs: arbitrary batch dimension, 2 features
    input x: Tensor<[?, 2], f64>;

    // Hidden layer: 2 inputs -> 8 hidden neurons
    param w1: Tensor<[2, 8], f64>  { init: "xavier_uniform"; };
    param b1: Tensor<[1, 8], f64>  { init: "zeros"; };

    // Output layer: 8 hidden -> 1 probability
    param w2: Tensor<[8, 1], f64>  { init: "xavier_uniform"; };
    param b2: Tensor<[1, 1], f64>  { init: "zeros"; };

    // Computation graph
    node h1     { op: matmul(x, w1) + b1; };
    node a1     { op: relu(h1); };
    node logits { op: matmul(a1, w2) + b2; };
    node out    { op: sigmoid(logits); };

    output out;
}

@training {
    model: Forward;
    loss: mse;
    epochs: 200;
    batch_size: 4;

    optimizer: {
        type: "Adam";
        lr: 0.05;
    }

    dataset: {
        path: "examples/data/xor.csv";
        has_header: true;
        features: [0, 1];
        targets: [2];
    }
}
