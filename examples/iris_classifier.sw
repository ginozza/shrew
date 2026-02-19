// Iris Classifier — 3-class classification for the Iris dataset
//
// Input:  4 features (sepal_length, sepal_width, petal_length, petal_width)
// Output: 3 logits  (setosa, versicolor, virginica)
//
// Architecture: 4 → 16 (relu) → 16 (relu) → 3
// Loss: cross_entropy

@model {
    name: "IrisClassifier";
    version: "1.0";
    author: "ginozza";
}

@graph forward {
    //  Input 
    input x: Tensor<[?, 4], f32>;

    //  Hidden layer 1: 4 → 16 
    param w1: Tensor<[4, 16], f32>  { init: "xavier_uniform"; };
    param b1: Tensor<[1, 16], f32>  { init: "zeros"; };

    node h1 { op: matmul(x, w1) + b1; };
    node a1 { op: relu(h1); };

    //  Hidden layer 2: 16 → 16 
    param w2: Tensor<[16, 16], f32> { init: "xavier_uniform"; };
    param b2: Tensor<[1, 16], f32>  { init: "zeros"; };

    node h2 { op: matmul(a1, w2) + b2; };
    node a2 { op: relu(h2); };

    //  Output layer: 16 → 3 
    param w3: Tensor<[16, 3], f32>  { init: "xavier_uniform"; };
    param b3: Tensor<[1, 3], f32>   { init: "zeros"; };

    node out { op: matmul(a2, w3) + b3; };

    output out;
}
