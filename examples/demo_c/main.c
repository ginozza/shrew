#include <stdio.h>
#include <stdlib.h>
#include "../../crates/shrew-c/include/shrew.h"

int main() {
    printf("=================================================\n");
    printf("  Shrew C Native API Demo\n");
    printf("=================================================\n");

    printf("Shrew Engine Version: %s\n", shrew_version());
    printf("CUDA Available: %s\n", shrew_cuda_is_available() ? "true" : "false");

    // Matrix A [2, 2] = [[1, 2], [3, 4]]
    double data_a[] = {1.0, 2.0, 3.0, 4.0};
    size_t shape_a[] = {2, 2};
    shrew_tensor_t* a = shrew_tensor_from_data(data_a, shape_a, 2, SHREW_DTYPE_F64, SHREW_DEVICE_CPU);

    // Matrix B [2, 2] = [[5, 6], [7, 8]]
    double data_b[] = {5.0, 6.0, 7.0, 8.0};
    size_t shape_b[] = {2, 2};
    shrew_tensor_t* b = shrew_tensor_from_data(data_b, shape_b, 2, SHREW_DTYPE_F64, SHREW_DEVICE_CPU);

    // C = A @ B
    shrew_tensor_t* c = shrew_tensor_matmul(a, b);

    printf("\nMatrix Multiplication Result printed by Shrew:\n");
    shrew_tensor_print(c);

    double out[4];
    int count = shrew_tensor_to_data(c, out, 4);
    printf("Extracted %d elements in C: [%.1f, %.1f, %.1f, %.1f]\n", count, out[0], out[1], out[2], out[3]);

    if (out[0] == 19.0 && out[1] == 22.0 && out[2] == 43.0 && out[3] == 50.0) {
        printf("\n>>> SUCCESS: Shrew successfully executed directly from C! <<<\n\n");
    }

    // ReLU test: relu([-2.0, 5.0])
    double relu_in[] = {-2.0, 5.0};
    size_t relu_shape[] = {2};
    shrew_tensor_t* r_in = shrew_tensor_from_data(relu_in, relu_shape, 1, SHREW_DTYPE_F64, SHREW_DEVICE_CPU);
    shrew_tensor_t* r_out = shrew_tensor_relu(r_in);
    shrew_tensor_to_data(r_out, out, 2);
    printf("ReLU([-2.0, 5.0]) = [%.1f, %.1f]\n", out[0], out[1]);

    // Free
    shrew_tensor_free(a);
    shrew_tensor_free(b);
    shrew_tensor_free(c);
    shrew_tensor_free(r_in);
    shrew_tensor_free(r_out);

    return 0;
}
