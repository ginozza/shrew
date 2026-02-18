// CUDA Kernel Source Code — Compiled to PTX at runtime via NVRTC
//
// All CUDA kernels for the Shrew GPU backend live here as string constants.
// They are compiled once when a CudaDevice is created, and cached in the device.
//
// DESIGN DECISIONS:
// - All data must be made contiguous before calling most kernels (simple flat indexing)
// - The `to_contiguous` kernel handles strided → contiguous copy
// - Reductions use a per-thread loop over the reduction dimension (not parallel reduction)
// - cuBLAS handles matmul (not a custom kernel)
// - F16/BF16 kernels promote to F32 for computation, then demote back (portable, correct)
// - F16 conversions use inline PTX assembly; BF16 uses bit manipulation

/// All kernel source code in one compilation unit.
/// Functions are prefixed by operation and suffixed by dtype (_f32, _f64, _f16, _bf16).
pub const KERNEL_SOURCE: &str = r#"

//  F16 / BF16 CONVERSION HELPERS 
//
// F16 ↔ F32: Uses inline PTX assembly (cvt.f32.f16 / cvt.rn.f16.f32)
// BF16 ↔ F32: Uses bit manipulation (BF16 = upper 16 bits of F32)
// All F16/BF16 data is stored as unsigned short (u16) on device.

__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ unsigned short f32_to_f16(float f) {
    unsigned short h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    unsigned int bits = ((unsigned int)h) << 16;
    return __int_as_float(bits);
}

__device__ __forceinline__ unsigned short f32_to_bf16(float f) {
    unsigned int bits = __float_as_int(f);
    unsigned int rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return (unsigned short)((bits + rounding_bias) >> 16);
}

//  FILL 

extern "C" __global__ void fill_f32(float* out, float val, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = val;
}

extern "C" __global__ void fill_f64(double* out, double val, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = val;
}

extern "C" __global__ void fill_u8(unsigned char* out, unsigned char val, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = val;
}

extern "C" __global__ void fill_u32(unsigned int* out_data, unsigned int val, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out_data[idx] = val;
}

extern "C" __global__ void fill_i64(long long* out, long long val, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = val;
}

extern "C" __global__ void fill_f16(unsigned short* out, float val, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(val);
}

extern "C" __global__ void fill_bf16(unsigned short* out, float val, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(val);
}

//  BINARY OPS 

extern "C" __global__ void binary_add_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}
extern "C" __global__ void binary_sub_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}
extern "C" __global__ void binary_mul_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}
extern "C" __global__ void binary_div_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / b[idx];
}

extern "C" __global__ void binary_add_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}
extern "C" __global__ void binary_sub_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}
extern "C" __global__ void binary_mul_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}
extern "C" __global__ void binary_div_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / b[idx];
}

//  F16 binary ops (promote to F32, compute, demote) 

extern "C" __global__ void binary_add_f16(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(f16_to_f32(a[idx]) + f16_to_f32(b[idx]));
}
extern "C" __global__ void binary_sub_f16(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(f16_to_f32(a[idx]) - f16_to_f32(b[idx]));
}
extern "C" __global__ void binary_mul_f16(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(f16_to_f32(a[idx]) * f16_to_f32(b[idx]));
}
extern "C" __global__ void binary_div_f16(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(f16_to_f32(a[idx]) / f16_to_f32(b[idx]));
}

//  BF16 binary ops 

extern "C" __global__ void binary_add_bf16(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(bf16_to_f32(a[idx]) + bf16_to_f32(b[idx]));
}
extern "C" __global__ void binary_sub_bf16(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(bf16_to_f32(a[idx]) - bf16_to_f32(b[idx]));
}
extern "C" __global__ void binary_mul_bf16(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(bf16_to_f32(a[idx]) * bf16_to_f32(b[idx]));
}
extern "C" __global__ void binary_div_bf16(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(bf16_to_f32(a[idx]) / bf16_to_f32(b[idx]));
}

//  BROADCAST BINARY OPS 
//
// These kernels handle element-wise binary ops when the two operands have
// different shapes (broadcasting). Each thread computes one output element,
// decomposing the flat index into multi-dimensional coordinates and then
// computing source offsets using broadcast strides (stride=0 for broadcast dims).
//
// Parameters:
//   a, b:        source arrays (may have fewer elements than output)
//   out:         output array (broadcast result)
//   out_dims:    shape of the output (GPU buffer, length = rank)
//   a_strides:   per-dim strides for 'a' (0 = broadcast, GPU buffer)
//   b_strides:   per-dim strides for 'b' (0 = broadcast, GPU buffer)
//   rank:        number of dimensions
//   n:           total output elements

#define BCAST_BINARY_KERNEL(name, type, expr) \
extern "C" __global__ void bcast_binary_##name( \
    const type* a, const type* b, type* out, \
    const unsigned int* out_dims, const unsigned int* a_strides, const unsigned int* b_strides, \
    unsigned int rank, unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    unsigned int remaining = idx; \
    unsigned int a_off = 0, b_off = 0; \
    for (int d = (int)rank - 1; d >= 0; d--) { \
        unsigned int coord = remaining % out_dims[d]; \
        remaining /= out_dims[d]; \
        a_off += coord * a_strides[d]; \
        b_off += coord * b_strides[d]; \
    } \
    out[idx] = expr; \
}

BCAST_BINARY_KERNEL(add_f32, float, a[a_off] + b[b_off])
BCAST_BINARY_KERNEL(sub_f32, float, a[a_off] - b[b_off])
BCAST_BINARY_KERNEL(mul_f32, float, a[a_off] * b[b_off])
BCAST_BINARY_KERNEL(div_f32, float, a[a_off] / b[b_off])

BCAST_BINARY_KERNEL(add_f64, double, a[a_off] + b[b_off])
BCAST_BINARY_KERNEL(sub_f64, double, a[a_off] - b[b_off])
BCAST_BINARY_KERNEL(mul_f64, double, a[a_off] * b[b_off])
BCAST_BINARY_KERNEL(div_f64, double, a[a_off] / b[b_off])

// F16 broadcast: promote to F32 for computation
#define BCAST_BINARY_KERNEL_F16(name, op) \
extern "C" __global__ void bcast_binary_##name##_f16( \
    const unsigned short* a, const unsigned short* b, unsigned short* out, \
    const unsigned int* out_dims, const unsigned int* a_strides, const unsigned int* b_strides, \
    unsigned int rank, unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    unsigned int remaining = idx; \
    unsigned int a_off = 0, b_off = 0; \
    for (int d = (int)rank - 1; d >= 0; d--) { \
        unsigned int coord = remaining % out_dims[d]; \
        remaining /= out_dims[d]; \
        a_off += coord * a_strides[d]; \
        b_off += coord * b_strides[d]; \
    } \
    out[idx] = f32_to_f16(f16_to_f32(a[a_off]) op f16_to_f32(b[b_off])); \
}

BCAST_BINARY_KERNEL_F16(add, +)
BCAST_BINARY_KERNEL_F16(sub, -)
BCAST_BINARY_KERNEL_F16(mul, *)
BCAST_BINARY_KERNEL_F16(div, /)

// BF16 broadcast
#define BCAST_BINARY_KERNEL_BF16(name, op) \
extern "C" __global__ void bcast_binary_##name##_bf16( \
    const unsigned short* a, const unsigned short* b, unsigned short* out, \
    const unsigned int* out_dims, const unsigned int* a_strides, const unsigned int* b_strides, \
    unsigned int rank, unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    unsigned int remaining = idx; \
    unsigned int a_off = 0, b_off = 0; \
    for (int d = (int)rank - 1; d >= 0; d--) { \
        unsigned int coord = remaining % out_dims[d]; \
        remaining /= out_dims[d]; \
        a_off += coord * a_strides[d]; \
        b_off += coord * b_strides[d]; \
    } \
    out[idx] = f32_to_bf16(bf16_to_f32(a[a_off]) op bf16_to_f32(b[b_off])); \
}

BCAST_BINARY_KERNEL_BF16(add, +)
BCAST_BINARY_KERNEL_BF16(sub, -)
BCAST_BINARY_KERNEL_BF16(mul, *)
BCAST_BINARY_KERNEL_BF16(div, /)

//  UNARY OPS 

extern "C" __global__ void unary_neg_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -inp[idx];
}
extern "C" __global__ void unary_abs_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fabsf(inp[idx]);
}
extern "C" __global__ void unary_exp_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(inp[idx]);
}
extern "C" __global__ void unary_log_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = logf(inp[idx]);
}
extern "C" __global__ void unary_sqrt_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrtf(inp[idx]);
}
extern "C" __global__ void unary_relu_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(inp[idx], 0.0f);
}
extern "C" __global__ void unary_sigmoid_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / (1.0f + expf(-inp[idx]));
}
extern "C" __global__ void unary_tanh_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(inp[idx]);
}
extern "C" __global__ void unary_gelu_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = inp[idx];
        out[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
extern "C" __global__ void unary_silu_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = inp[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}
extern "C" __global__ void unary_sin_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sinf(inp[idx]);
}
extern "C" __global__ void unary_cos_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = cosf(inp[idx]);
}
extern "C" __global__ void unary_square_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = inp[idx] * inp[idx];
}
extern "C" __global__ void unary_floor_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = floorf(inp[idx]);
}
extern "C" __global__ void unary_ceil_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = ceilf(inp[idx]);
}
extern "C" __global__ void unary_round_f32(const float* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = roundf(inp[idx]);
}

//  f64 unary ops 

extern "C" __global__ void unary_neg_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -inp[idx];
}
extern "C" __global__ void unary_abs_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fabs(inp[idx]);
}
extern "C" __global__ void unary_exp_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = exp(inp[idx]);
}
extern "C" __global__ void unary_log_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = log(inp[idx]);
}
extern "C" __global__ void unary_sqrt_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrt(inp[idx]);
}
extern "C" __global__ void unary_relu_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmax(inp[idx], 0.0);
}
extern "C" __global__ void unary_sigmoid_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0 / (1.0 + exp(-inp[idx]));
}
extern "C" __global__ void unary_tanh_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanh(inp[idx]);
}
extern "C" __global__ void unary_gelu_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = inp[idx];
        out[idx] = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    }
}
extern "C" __global__ void unary_silu_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = inp[idx];
        out[idx] = x / (1.0 + exp(-x));
    }
}
extern "C" __global__ void unary_sin_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sin(inp[idx]);
}
extern "C" __global__ void unary_cos_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = cos(inp[idx]);
}
extern "C" __global__ void unary_square_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = inp[idx] * inp[idx];
}
extern "C" __global__ void unary_floor_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = floor(inp[idx]);
}
extern "C" __global__ void unary_ceil_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = ceil(inp[idx]);
}
extern "C" __global__ void unary_round_f64(const double* inp, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (double)llround(inp[idx]);
    }
}

//  F16 unary ops (promote to F32, compute, demote) 

extern "C" __global__ void unary_neg_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(-f16_to_f32(inp[idx]));
}
extern "C" __global__ void unary_abs_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(fabsf(f16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_exp_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(expf(f16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_log_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(logf(f16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_sqrt_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(sqrtf(f16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_relu_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(fmaxf(f16_to_f32(inp[idx]), 0.0f));
}
extern "C" __global__ void unary_sigmoid_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float x = f16_to_f32(inp[idx]); out[idx] = f32_to_f16(1.0f / (1.0f + expf(-x))); }
}
extern "C" __global__ void unary_tanh_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(tanhf(f16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_gelu_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = f16_to_f32(inp[idx]);
        out[idx] = f32_to_f16(0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))));
    }
}
extern "C" __global__ void unary_silu_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float x = f16_to_f32(inp[idx]); out[idx] = f32_to_f16(x / (1.0f + expf(-x))); }
}
extern "C" __global__ void unary_sin_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(sinf(f16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_cos_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(cosf(f16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_square_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float x = f16_to_f32(inp[idx]); out[idx] = f32_to_f16(x * x); }
}
extern "C" __global__ void unary_floor_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(floorf(f16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_ceil_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(ceilf(f16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_round_f16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(roundf(f16_to_f32(inp[idx])));
}

//  BF16 unary ops 

extern "C" __global__ void unary_neg_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(-bf16_to_f32(inp[idx]));
}
extern "C" __global__ void unary_abs_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(fabsf(bf16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_exp_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(expf(bf16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_log_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(logf(bf16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_sqrt_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(sqrtf(bf16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_relu_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(fmaxf(bf16_to_f32(inp[idx]), 0.0f));
}
extern "C" __global__ void unary_sigmoid_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float x = bf16_to_f32(inp[idx]); out[idx] = f32_to_bf16(1.0f / (1.0f + expf(-x))); }
}
extern "C" __global__ void unary_tanh_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(tanhf(bf16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_gelu_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = bf16_to_f32(inp[idx]);
        out[idx] = f32_to_bf16(0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))));
    }
}
extern "C" __global__ void unary_silu_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float x = bf16_to_f32(inp[idx]); out[idx] = f32_to_bf16(x / (1.0f + expf(-x))); }
}
extern "C" __global__ void unary_sin_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(sinf(bf16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_cos_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(cosf(bf16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_square_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float x = bf16_to_f32(inp[idx]); out[idx] = f32_to_bf16(x * x); }
}
extern "C" __global__ void unary_floor_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(floorf(bf16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_ceil_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(ceilf(bf16_to_f32(inp[idx])));
}
extern "C" __global__ void unary_round_bf16(const unsigned short* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(roundf(bf16_to_f32(inp[idx])));
}

//  COMPARISON OPS 

extern "C" __global__ void cmp_eq_f32(const float* a, const float* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] == b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_ne_f32(const float* a, const float* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] != b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_gt_f32(const float* a, const float* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] > b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_ge_f32(const float* a, const float* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] >= b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_lt_f32(const float* a, const float* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] < b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_le_f32(const float* a, const float* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] <= b[idx]) ? 1 : 0;
}

extern "C" __global__ void cmp_eq_f64(const double* a, const double* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] == b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_ne_f64(const double* a, const double* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] != b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_gt_f64(const double* a, const double* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] > b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_ge_f64(const double* a, const double* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] >= b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_lt_f64(const double* a, const double* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] < b[idx]) ? 1 : 0;
}
extern "C" __global__ void cmp_le_f64(const double* a, const double* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (a[idx] <= b[idx]) ? 1 : 0;
}

//  F16 comparison ops 

extern "C" __global__ void cmp_eq_f16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (f16_to_f32(a[idx]) == f16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_ne_f16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (f16_to_f32(a[idx]) != f16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_gt_f16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (f16_to_f32(a[idx]) > f16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_ge_f16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (f16_to_f32(a[idx]) >= f16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_lt_f16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (f16_to_f32(a[idx]) < f16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_le_f16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (f16_to_f32(a[idx]) <= f16_to_f32(b[idx])) ? 1 : 0;
}

//  BF16 comparison ops 

extern "C" __global__ void cmp_eq_bf16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (bf16_to_f32(a[idx]) == bf16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_ne_bf16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (bf16_to_f32(a[idx]) != bf16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_gt_bf16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (bf16_to_f32(a[idx]) > bf16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_ge_bf16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (bf16_to_f32(a[idx]) >= bf16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_lt_bf16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (bf16_to_f32(a[idx]) < bf16_to_f32(b[idx])) ? 1 : 0;
}
extern "C" __global__ void cmp_le_bf16(const unsigned short* a, const unsigned short* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (bf16_to_f32(a[idx]) <= bf16_to_f32(b[idx])) ? 1 : 0;
}

//  AFFINE 

extern "C" __global__ void affine_f32(const float* inp, float* out, float mul, float add, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = inp[idx] * mul + add;
}
extern "C" __global__ void affine_f64(const double* inp, double* out, double mul, double add, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = inp[idx] * mul + add;
}
extern "C" __global__ void affine_f16(const unsigned short* inp, unsigned short* out, float mul, float add, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(f16_to_f32(inp[idx]) * mul + add);
}
extern "C" __global__ void affine_bf16(const unsigned short* inp, unsigned short* out, float mul, float add, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(bf16_to_f32(inp[idx]) * mul + add);
}

//  POWF 

extern "C" __global__ void powf_f32(const float* inp, float* out, float exponent, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = powf(inp[idx], exponent);
}
extern "C" __global__ void powf_f64(const double* inp, double* out, double exponent, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = pow(inp[idx], exponent);
}
extern "C" __global__ void powf_f16(const unsigned short* inp, unsigned short* out, float exponent, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(powf(f16_to_f32(inp[idx]), exponent));
}
extern "C" __global__ void powf_bf16(const unsigned short* inp, unsigned short* out, float exponent, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(powf(bf16_to_f32(inp[idx]), exponent));
}

//  CLAMP 

extern "C" __global__ void clamp_f32(const float* inp, float* out, float lo, float hi, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fminf(fmaxf(inp[idx], lo), hi);
}
extern "C" __global__ void clamp_f64(const double* inp, double* out, double lo, double hi, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmin(fmax(inp[idx], lo), hi);
}
extern "C" __global__ void clamp_f16(const unsigned short* inp, unsigned short* out, float lo, float hi, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(fminf(fmaxf(f16_to_f32(inp[idx]), lo), hi));
}
extern "C" __global__ void clamp_bf16(const unsigned short* inp, unsigned short* out, float lo, float hi, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(fminf(fmaxf(bf16_to_f32(inp[idx]), lo), hi));
}

//  WHERE_COND 

extern "C" __global__ void where_cond_f32(
    const unsigned char* mask, const float* on_true, const float* on_false,
    float* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = mask[idx] ? on_true[idx] : on_false[idx];
}
extern "C" __global__ void where_cond_f64(
    const unsigned char* mask, const double* on_true, const double* on_false,
    double* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = mask[idx] ? on_true[idx] : on_false[idx];
}
extern "C" __global__ void where_cond_f16(
    const unsigned char* mask, const unsigned short* on_true, const unsigned short* on_false,
    unsigned short* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = mask[idx] ? on_true[idx] : on_false[idx];
}
extern "C" __global__ void where_cond_bf16(
    const unsigned char* mask, const unsigned short* on_true, const unsigned short* on_false,
    unsigned short* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = mask[idx] ? on_true[idx] : on_false[idx];
}

//  REDUCTION 

extern "C" __global__ void reduce_sum_f32(
    const float* inp, float* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = 0.0f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        acc += inp[outer * reduce_size * inner_size + r * inner_size + inner];
    }
    out[idx] = acc;
}

extern "C" __global__ void reduce_mean_f32(
    const float* inp, float* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = 0.0f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        acc += inp[outer * reduce_size * inner_size + r * inner_size + inner];
    }
    out[idx] = acc / (float)reduce_size;
}

extern "C" __global__ void reduce_max_f32(
    const float* inp, float* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = -1e38f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = inp[outer * reduce_size * inner_size + r * inner_size + inner];
        if (v > acc) acc = v;
    }
    out[idx] = acc;
}

extern "C" __global__ void reduce_min_f32(
    const float* inp, float* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = 1e38f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = inp[outer * reduce_size * inner_size + r * inner_size + inner];
        if (v < acc) acc = v;
    }
    out[idx] = acc;
}

extern "C" __global__ void reduce_argmax_f32(
    const float* inp, long long* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float best_v = -1e38f;
    long long best_k = 0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = inp[outer * reduce_size * inner_size + r * inner_size + inner];
        if (v > best_v) { best_v = v; best_k = (long long)r; }
    }
    out[idx] = best_k;
}

extern "C" __global__ void reduce_argmin_f32(
    const float* inp, long long* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float best_v = 1e38f;
    long long best_k = 0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = inp[outer * reduce_size * inner_size + r * inner_size + inner];
        if (v < best_v) { best_v = v; best_k = (long long)r; }
    }
    out[idx] = best_k;
}

//  f64 reductions 

extern "C" __global__ void reduce_sum_f64(
    const double* inp, double* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    double acc = 0.0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        acc += inp[outer * reduce_size * inner_size + r * inner_size + inner];
    }
    out[idx] = acc;
}

extern "C" __global__ void reduce_mean_f64(
    const double* inp, double* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    double acc = 0.0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        acc += inp[outer * reduce_size * inner_size + r * inner_size + inner];
    }
    out[idx] = acc / (double)reduce_size;
}

extern "C" __global__ void reduce_max_f64(
    const double* inp, double* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    double acc = -1e308;
    for (unsigned int r = 0; r < reduce_size; r++) {
        double v = inp[outer * reduce_size * inner_size + r * inner_size + inner];
        if (v > acc) acc = v;
    }
    out[idx] = acc;
}

extern "C" __global__ void reduce_min_f64(
    const double* inp, double* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    double acc = 1e308;
    for (unsigned int r = 0; r < reduce_size; r++) {
        double v = inp[outer * reduce_size * inner_size + r * inner_size + inner];
        if (v < acc) acc = v;
    }
    out[idx] = acc;
}

extern "C" __global__ void reduce_argmax_f64(
    const double* inp, long long* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    double best_v = -1e308;
    long long best_k = 0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        double v = inp[outer * reduce_size * inner_size + r * inner_size + inner];
        if (v > best_v) { best_v = v; best_k = (long long)r; }
    }
    out[idx] = best_k;
}

extern "C" __global__ void reduce_argmin_f64(
    const double* inp, long long* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    double best_v = 1e308;
    long long best_k = 0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        double v = inp[outer * reduce_size * inner_size + r * inner_size + inner];
        if (v < best_v) { best_v = v; best_k = (long long)r; }
    }
    out[idx] = best_k;
}

//  F16 reductions (accumulate in F32 for precision) 

extern "C" __global__ void reduce_sum_f16(
    const unsigned short* inp, unsigned short* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = 0.0f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        acc += f16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
    }
    out[idx] = f32_to_f16(acc);
}

extern "C" __global__ void reduce_mean_f16(
    const unsigned short* inp, unsigned short* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = 0.0f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        acc += f16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
    }
    out[idx] = f32_to_f16(acc / (float)reduce_size);
}

extern "C" __global__ void reduce_max_f16(
    const unsigned short* inp, unsigned short* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = -1e38f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = f16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
        if (v > acc) acc = v;
    }
    out[idx] = f32_to_f16(acc);
}

extern "C" __global__ void reduce_min_f16(
    const unsigned short* inp, unsigned short* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = 1e38f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = f16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
        if (v < acc) acc = v;
    }
    out[idx] = f32_to_f16(acc);
}

extern "C" __global__ void reduce_argmax_f16(
    const unsigned short* inp, long long* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float best_v = -1e38f;
    long long best_k = 0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = f16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
        if (v > best_v) { best_v = v; best_k = (long long)r; }
    }
    out[idx] = best_k;
}

extern "C" __global__ void reduce_argmin_f16(
    const unsigned short* inp, long long* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float best_v = 1e38f;
    long long best_k = 0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = f16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
        if (v < best_v) { best_v = v; best_k = (long long)r; }
    }
    out[idx] = best_k;
}

//  BF16 reductions 

extern "C" __global__ void reduce_sum_bf16(
    const unsigned short* inp, unsigned short* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = 0.0f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        acc += bf16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
    }
    out[idx] = f32_to_bf16(acc);
}

extern "C" __global__ void reduce_mean_bf16(
    const unsigned short* inp, unsigned short* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = 0.0f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        acc += bf16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
    }
    out[idx] = f32_to_bf16(acc / (float)reduce_size);
}

extern "C" __global__ void reduce_max_bf16(
    const unsigned short* inp, unsigned short* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = -1e38f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = bf16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
        if (v > acc) acc = v;
    }
    out[idx] = f32_to_bf16(acc);
}

extern "C" __global__ void reduce_min_bf16(
    const unsigned short* inp, unsigned short* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float acc = 1e38f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = bf16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
        if (v < acc) acc = v;
    }
    out[idx] = f32_to_bf16(acc);
}

extern "C" __global__ void reduce_argmax_bf16(
    const unsigned short* inp, long long* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float best_v = -1e38f;
    long long best_k = 0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = bf16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
        if (v > best_v) { best_v = v; best_k = (long long)r; }
    }
    out[idx] = best_k;
}

extern "C" __global__ void reduce_argmin_bf16(
    const unsigned short* inp, long long* out,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * inner_size;
    if (idx >= total) return;
    unsigned int outer = idx / inner_size;
    unsigned int inner = idx % inner_size;
    float best_v = 1e38f;
    long long best_k = 0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        float v = bf16_to_f32(inp[outer * reduce_size * inner_size + r * inner_size + inner]);
        if (v < best_v) { best_v = v; best_k = (long long)r; }
    }
    out[idx] = best_k;
}

//  GATHER 

extern "C" __global__ void gather_f32(
    const float* inp, const long long* index, float* out,
    unsigned int pre, unsigned int inp_dim, unsigned int idx_dim,
    unsigned int post, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int pre_idx = id / (idx_dim * post);
    unsigned int post_idx = id % post;
    long long gi = index[id];
    out[id] = inp[pre_idx * inp_dim * post + (unsigned int)gi * post + post_idx];
}

extern "C" __global__ void gather_f64(
    const double* inp, const long long* index, double* out,
    unsigned int pre, unsigned int inp_dim, unsigned int idx_dim,
    unsigned int post, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int pre_idx = id / (idx_dim * post);
    unsigned int post_idx = id % post;
    long long gi = index[id];
    out[id] = inp[pre_idx * inp_dim * post + (unsigned int)gi * post + post_idx];
}

extern "C" __global__ void gather_f16(
    const unsigned short* inp, const long long* index, unsigned short* out,
    unsigned int pre, unsigned int inp_dim, unsigned int idx_dim,
    unsigned int post, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int pre_idx = id / (idx_dim * post);
    unsigned int post_idx = id % post;
    long long gi = index[id];
    out[id] = inp[pre_idx * inp_dim * post + (unsigned int)gi * post + post_idx];
}

extern "C" __global__ void gather_bf16(
    const unsigned short* inp, const long long* index, unsigned short* out,
    unsigned int pre, unsigned int inp_dim, unsigned int idx_dim,
    unsigned int post, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int pre_idx = id / (idx_dim * post);
    unsigned int post_idx = id % post;
    long long gi = index[id];
    out[id] = inp[pre_idx * inp_dim * post + (unsigned int)gi * post + post_idx];
}

//  INDEX SELECT 

extern "C" __global__ void index_select_f32(
    const float* inp, const long long* indices, float* out,
    unsigned int pre_dim, unsigned int src_dim, unsigned int post_dim,
    unsigned int idx_len, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int pre = id / (idx_len * post_dim);
    unsigned int rem = id % (idx_len * post_dim);
    unsigned int idx_pos = rem / post_dim;
    unsigned int post = rem % post_dim;
    long long src_pos = indices[idx_pos];
    out[id] = inp[pre * src_dim * post_dim + (unsigned int)src_pos * post_dim + post];
}

extern "C" __global__ void index_select_f64(
    const double* inp, const long long* indices, double* out,
    unsigned int pre_dim, unsigned int src_dim, unsigned int post_dim,
    unsigned int idx_len, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int pre = id / (idx_len * post_dim);
    unsigned int rem = id % (idx_len * post_dim);
    unsigned int idx_pos = rem / post_dim;
    unsigned int post = rem % post_dim;
    long long src_pos = indices[idx_pos];
    out[id] = inp[pre * src_dim * post_dim + (unsigned int)src_pos * post_dim + post];
}

extern "C" __global__ void index_select_f16(
    const unsigned short* inp, const long long* indices, unsigned short* out,
    unsigned int pre_dim, unsigned int src_dim, unsigned int post_dim,
    unsigned int idx_len, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int pre = id / (idx_len * post_dim);
    unsigned int rem = id % (idx_len * post_dim);
    unsigned int idx_pos = rem / post_dim;
    unsigned int post = rem % post_dim;
    long long src_pos = indices[idx_pos];
    out[id] = inp[pre * src_dim * post_dim + (unsigned int)src_pos * post_dim + post];
}

extern "C" __global__ void index_select_bf16(
    const unsigned short* inp, const long long* indices, unsigned short* out,
    unsigned int pre_dim, unsigned int src_dim, unsigned int post_dim,
    unsigned int idx_len, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int pre = id / (idx_len * post_dim);
    unsigned int rem = id % (idx_len * post_dim);
    unsigned int idx_pos = rem / post_dim;
    unsigned int post = rem % post_dim;
    long long src_pos = indices[idx_pos];
    out[id] = inp[pre * src_dim * post_dim + (unsigned int)src_pos * post_dim + post];
}

//  TO CONTIGUOUS (strided copy) 

extern "C" __global__ void to_contiguous_f32(
    const float* src, float* dst,
    const int* shape, const int* strides,
    int offset, int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int src_idx = offset;
    int remaining = (int)idx;
    for (int d = ndim - 1; d >= 0; d--) {
        src_idx += (remaining % shape[d]) * strides[d];
        remaining /= shape[d];
    }
    dst[idx] = src[src_idx];
}

extern "C" __global__ void to_contiguous_f64(
    const double* src, double* dst,
    const int* shape, const int* strides,
    int offset, int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int src_idx = offset;
    int remaining = (int)idx;
    for (int d = ndim - 1; d >= 0; d--) {
        src_idx += (remaining % shape[d]) * strides[d];
        remaining /= shape[d];
    }
    dst[idx] = src[src_idx];
}

extern "C" __global__ void to_contiguous_u8(
    const unsigned char* src, unsigned char* dst,
    const int* shape, const int* strides,
    int offset, int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int src_idx = offset;
    int remaining = (int)idx;
    for (int d = ndim - 1; d >= 0; d--) {
        src_idx += (remaining % shape[d]) * strides[d];
        remaining /= shape[d];
    }
    dst[idx] = src[src_idx];
}

extern "C" __global__ void to_contiguous_u16(
    const unsigned short* src, unsigned short* dst,
    const int* shape, const int* strides,
    int offset, int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int src_idx = offset;
    int remaining = (int)idx;
    for (int d = ndim - 1; d >= 0; d--) {
        src_idx += (remaining % shape[d]) * strides[d];
        remaining /= shape[d];
    }
    dst[idx] = src[src_idx];
}

//  CAT COPY 

extern "C" __global__ void cat_copy_f32(
    const float* src, float* dst,
    unsigned int outer, unsigned int this_dim, unsigned int inner,
    unsigned int total_dim, unsigned int dim_offset, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int o = id / (this_dim * inner);
    unsigned int rem = id % (this_dim * inner);
    unsigned int d = rem / inner;
    unsigned int i = rem % inner;
    unsigned int dst_idx = o * total_dim * inner + (dim_offset + d) * inner + i;
    dst[dst_idx] = src[id];
}

extern "C" __global__ void cat_copy_f64(
    const double* src, double* dst,
    unsigned int outer, unsigned int this_dim, unsigned int inner,
    unsigned int total_dim, unsigned int dim_offset, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int o = id / (this_dim * inner);
    unsigned int rem = id % (this_dim * inner);
    unsigned int d = rem / inner;
    unsigned int i = rem % inner;
    unsigned int dst_idx = o * total_dim * inner + (dim_offset + d) * inner + i;
    dst[dst_idx] = src[id];
}

extern "C" __global__ void cat_copy_f16(
    const unsigned short* src, unsigned short* dst,
    unsigned int outer, unsigned int this_dim, unsigned int inner,
    unsigned int total_dim, unsigned int dim_offset, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int o = id / (this_dim * inner);
    unsigned int rem = id % (this_dim * inner);
    unsigned int d = rem / inner;
    unsigned int i = rem % inner;
    unsigned int dst_idx = o * total_dim * inner + (dim_offset + d) * inner + i;
    dst[dst_idx] = src[id];
}

extern "C" __global__ void cat_copy_bf16(
    const unsigned short* src, unsigned short* dst,
    unsigned int outer, unsigned int this_dim, unsigned int inner,
    unsigned int total_dim, unsigned int dim_offset, unsigned int n
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int o = id / (this_dim * inner);
    unsigned int rem = id % (this_dim * inner);
    unsigned int d = rem / inner;
    unsigned int i = rem % inner;
    unsigned int dst_idx = o * total_dim * inner + (dim_offset + d) * inner + i;
    dst[dst_idx] = src[id];
}

//  F16/F32 CAST KERNELS 

extern "C" __global__ void cast_f16_to_f32(const unsigned short* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f16_to_f32(inp[idx]);
}

extern "C" __global__ void cast_f32_to_f16(const float* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_f16(inp[idx]);
}

extern "C" __global__ void cast_bf16_to_f32(const unsigned short* inp, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = bf16_to_f32(inp[idx]);
}

extern "C" __global__ void cast_f32_to_bf16(const float* inp, unsigned short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f32_to_bf16(inp[idx]);
}

"#;

/// All kernel function names used in load_ptx. Must match the extern "C" names above.
pub const KERNEL_NAMES: &[&str] = &[
    // fill
    "fill_f32",
    "fill_f64",
    "fill_u8",
    "fill_u32",
    "fill_i64",
    "fill_f16",
    "fill_bf16",
    // binary f32
    "binary_add_f32",
    "binary_sub_f32",
    "binary_mul_f32",
    "binary_div_f32",
    // binary f64
    "binary_add_f64",
    "binary_sub_f64",
    "binary_mul_f64",
    "binary_div_f64",
    // binary f16
    "binary_add_f16",
    "binary_sub_f16",
    "binary_mul_f16",
    "binary_div_f16",
    // binary bf16
    "binary_add_bf16",
    "binary_sub_bf16",
    "binary_mul_bf16",
    "binary_div_bf16",
    // broadcast binary f32
    "bcast_binary_add_f32",
    "bcast_binary_sub_f32",
    "bcast_binary_mul_f32",
    "bcast_binary_div_f32",
    // broadcast binary f64
    "bcast_binary_add_f64",
    "bcast_binary_sub_f64",
    "bcast_binary_mul_f64",
    "bcast_binary_div_f64",
    // broadcast binary f16
    "bcast_binary_add_f16",
    "bcast_binary_sub_f16",
    "bcast_binary_mul_f16",
    "bcast_binary_div_f16",
    // broadcast binary bf16
    "bcast_binary_add_bf16",
    "bcast_binary_sub_bf16",
    "bcast_binary_mul_bf16",
    "bcast_binary_div_bf16",
    // unary f32
    "unary_neg_f32",
    "unary_abs_f32",
    "unary_exp_f32",
    "unary_log_f32",
    "unary_sqrt_f32",
    "unary_relu_f32",
    "unary_sigmoid_f32",
    "unary_tanh_f32",
    "unary_gelu_f32",
    "unary_silu_f32",
    "unary_sin_f32",
    "unary_cos_f32",
    "unary_square_f32",
    "unary_floor_f32",
    "unary_ceil_f32",
    "unary_round_f32",
    // unary f64
    "unary_neg_f64",
    "unary_abs_f64",
    "unary_exp_f64",
    "unary_log_f64",
    "unary_sqrt_f64",
    "unary_relu_f64",
    "unary_sigmoid_f64",
    "unary_tanh_f64",
    "unary_gelu_f64",
    "unary_silu_f64",
    "unary_sin_f64",
    "unary_cos_f64",
    "unary_square_f64",
    "unary_floor_f64",
    "unary_ceil_f64",
    "unary_round_f64",
    // unary f16
    "unary_neg_f16",
    "unary_abs_f16",
    "unary_exp_f16",
    "unary_log_f16",
    "unary_sqrt_f16",
    "unary_relu_f16",
    "unary_sigmoid_f16",
    "unary_tanh_f16",
    "unary_gelu_f16",
    "unary_silu_f16",
    "unary_sin_f16",
    "unary_cos_f16",
    "unary_square_f16",
    "unary_floor_f16",
    "unary_ceil_f16",
    "unary_round_f16",
    // unary bf16
    "unary_neg_bf16",
    "unary_abs_bf16",
    "unary_exp_bf16",
    "unary_log_bf16",
    "unary_sqrt_bf16",
    "unary_relu_bf16",
    "unary_sigmoid_bf16",
    "unary_tanh_bf16",
    "unary_gelu_bf16",
    "unary_silu_bf16",
    "unary_sin_bf16",
    "unary_cos_bf16",
    "unary_square_bf16",
    "unary_floor_bf16",
    "unary_ceil_bf16",
    "unary_round_bf16",
    // cmp f32
    "cmp_eq_f32",
    "cmp_ne_f32",
    "cmp_gt_f32",
    "cmp_ge_f32",
    "cmp_lt_f32",
    "cmp_le_f32",
    // cmp f64
    "cmp_eq_f64",
    "cmp_ne_f64",
    "cmp_gt_f64",
    "cmp_ge_f64",
    "cmp_lt_f64",
    "cmp_le_f64",
    // cmp f16
    "cmp_eq_f16",
    "cmp_ne_f16",
    "cmp_gt_f16",
    "cmp_ge_f16",
    "cmp_lt_f16",
    "cmp_le_f16",
    // cmp bf16
    "cmp_eq_bf16",
    "cmp_ne_bf16",
    "cmp_gt_bf16",
    "cmp_ge_bf16",
    "cmp_lt_bf16",
    "cmp_le_bf16",
    // affine
    "affine_f32",
    "affine_f64",
    "affine_f16",
    "affine_bf16",
    // powf
    "powf_f32",
    "powf_f64",
    "powf_f16",
    "powf_bf16",
    // clamp
    "clamp_f32",
    "clamp_f64",
    "clamp_f16",
    "clamp_bf16",
    // where_cond
    "where_cond_f32",
    "where_cond_f64",
    "where_cond_f16",
    "where_cond_bf16",
    // reduce f32
    "reduce_sum_f32",
    "reduce_mean_f32",
    "reduce_max_f32",
    "reduce_min_f32",
    "reduce_argmax_f32",
    "reduce_argmin_f32",
    // reduce f64
    "reduce_sum_f64",
    "reduce_mean_f64",
    "reduce_max_f64",
    "reduce_min_f64",
    "reduce_argmax_f64",
    "reduce_argmin_f64",
    // reduce f16
    "reduce_sum_f16",
    "reduce_mean_f16",
    "reduce_max_f16",
    "reduce_min_f16",
    "reduce_argmax_f16",
    "reduce_argmin_f16",
    // reduce bf16
    "reduce_sum_bf16",
    "reduce_mean_bf16",
    "reduce_max_bf16",
    "reduce_min_bf16",
    "reduce_argmax_bf16",
    "reduce_argmin_bf16",
    // gather
    "gather_f32",
    "gather_f64",
    "gather_f16",
    "gather_bf16",
    // index_select
    "index_select_f32",
    "index_select_f64",
    "index_select_f16",
    "index_select_bf16",
    // to_contiguous
    "to_contiguous_f32",
    "to_contiguous_f64",
    "to_contiguous_u8",
    "to_contiguous_u16",
    // cat
    "cat_copy_f32",
    "cat_copy_f64",
    "cat_copy_f16",
    "cat_copy_bf16",
    // cast
    "cast_f16_to_f32",
    "cast_f32_to_f16",
    "cast_bf16_to_f32",
    "cast_f32_to_bf16",
];

/// Module name used in cudarc's PTX loading.
pub const MODULE_NAME: &str = "shrew_kernels";
