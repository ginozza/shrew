#ifndef SHREW_H
#define SHREW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(SHREW_EXPORTS)
    #define SHREW_API __declspec(dllexport)
  #else
    #define SHREW_API __declspec(dllimport)
  #endif
#else
  #define SHREW_API __attribute__((visibility("default")))
#endif

// Status codes
#define SHREW_OK 0
#define SHREW_ERROR -1

// Data types
typedef enum {
    SHREW_DTYPE_F32 = 0,
    SHREW_DTYPE_F64 = 1,
    SHREW_DTYPE_I64 = 2,
    SHREW_DTYPE_U8  = 3,
} shrew_dtype_t;

// Device types
typedef enum {
    SHREW_DEVICE_CPU  = 0,
    SHREW_DEVICE_CUDA = 1,
} shrew_device_t;

// Opaque handles
typedef struct shrew_tensor shrew_tensor_t;
typedef struct shrew_grad_store shrew_grad_store_t;
typedef struct shrew_executor shrew_executor_t;

// --- Error Handling ---
SHREW_API const char* shrew_last_error(void);

// --- Tensor Lifecycle & Creation ---
SHREW_API shrew_tensor_t* shrew_tensor_from_data(
    const double* data,
    const size_t* shape,
    size_t ndim,
    shrew_dtype_t dtype,
    shrew_device_t device
);

SHREW_API shrew_tensor_t* shrew_tensor_zeros(
    const size_t* shape,
    size_t ndim,
    shrew_dtype_t dtype,
    shrew_device_t device
);

SHREW_API shrew_tensor_t* shrew_tensor_ones(
    const size_t* shape,
    size_t ndim,
    shrew_dtype_t dtype,
    shrew_device_t device
);

SHREW_API shrew_tensor_t* shrew_tensor_randn(
    const size_t* shape,
    size_t ndim,
    shrew_dtype_t dtype,
    shrew_device_t device
);

SHREW_API void shrew_tensor_free(shrew_tensor_t* tensor);

// --- Tensor Metadata & Inspection ---
SHREW_API size_t shrew_tensor_ndim(const shrew_tensor_t* tensor);
SHREW_API int shrew_tensor_shape(const shrew_tensor_t* tensor, size_t* out_shape, size_t max_dim);
SHREW_API size_t shrew_tensor_numel(const shrew_tensor_t* tensor);
SHREW_API int shrew_tensor_to_data(const shrew_tensor_t* tensor, double* out_data, size_t max_len);
SHREW_API void shrew_tensor_print(const shrew_tensor_t* tensor);

// --- Tensor Operations ---
SHREW_API shrew_tensor_t* shrew_tensor_add(const shrew_tensor_t* a, const shrew_tensor_t* b);
SHREW_API shrew_tensor_t* shrew_tensor_sub(const shrew_tensor_t* a, const shrew_tensor_t* b);
SHREW_API shrew_tensor_t* shrew_tensor_mul(const shrew_tensor_t* a, const shrew_tensor_t* b);
SHREW_API shrew_tensor_t* shrew_tensor_matmul(const shrew_tensor_t* a, const shrew_tensor_t* b);
SHREW_API shrew_tensor_t* shrew_tensor_relu(const shrew_tensor_t* a);
SHREW_API shrew_tensor_t* shrew_tensor_sigmoid(const shrew_tensor_t* a);
SHREW_API shrew_tensor_t* shrew_tensor_tanh(const shrew_tensor_t* a);

// --- Autograd ---
SHREW_API shrew_tensor_t* shrew_tensor_set_variable(shrew_tensor_t* a);
SHREW_API shrew_grad_store_t* shrew_tensor_backward(const shrew_tensor_t* loss);
SHREW_API shrew_tensor_t* shrew_grad_store_get(const shrew_grad_store_t* store, const shrew_tensor_t* param);
SHREW_API void shrew_grad_store_free(shrew_grad_store_t* store);

// --- .sw Model Execution ---
SHREW_API shrew_executor_t* shrew_executor_load_source(const char* sw_source, shrew_dtype_t dtype);
SHREW_API shrew_executor_t* shrew_executor_load_file(const char* sw_path, shrew_dtype_t dtype);
SHREW_API shrew_tensor_t* shrew_executor_run_single(
    const shrew_executor_t* exec,
    const char* graph_name,
    const char* input_name,
    const shrew_tensor_t* input
);
SHREW_API void shrew_executor_free(shrew_executor_t* exec);

// --- Hardware & Diagnostics ---
SHREW_API int shrew_cuda_is_available(void);
SHREW_API size_t shrew_cuda_device_count(void);
SHREW_API const char* shrew_version(void);

#ifdef __cplusplus
}
#endif

#endif // SHREW_H
