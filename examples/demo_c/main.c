#include <stdio.h>
#include "../../crates/shrew-c/include/shrew.h"

int main() {
    double loss = 0.0;
    int epochs = shrew_train_file("examples/model.sw", SHREW_DTYPE_F64, &loss);
    printf("Epochs: %d, Loss: %f\n", epochs, loss);
    return 0;
}
