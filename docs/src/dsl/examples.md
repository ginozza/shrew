# Examples

## Simple Linear Regression

```sw
@model {
    name: "LinearRegression";
    version: "1.0";
}

@config {
    batch_size: 32;
    learning_rate: 0.01;
}

@graph Forward(x: Tensor<[Batch, In], f32>) -> Tensor<[Batch, Out], f32> {
    param w: Tensor<[In, Out], f32> { init: "normal(0, 0.02)"; };
    param b: Tensor<[Out], f32> { init: "zeros"; };
    
    node y: Tensor<[Batch, Out], f32> {
        op: x * w + b;
    };
    
    output y;
}
```

## Matrix Multiplication

```sw
@graph MatMul(A: Tensor<[M, K], f32>, B: Tensor<[K, N], f32>) {
    node C {
        op: A @ B; 
    };
    output C;
}
```
