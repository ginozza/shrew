//! shrew-c — C FFI and Native Shared Library for Shrew
//!
//! Provides a stable C ABI (`shrew.h`) enabling Shrew to be used from:
//! - C / C++
//! - Java (Project Panama / FFM, JNA, JNI)
//! - C# / .NET (P/Invoke)
//! - Node.js / Bun (node-ffi)
//! - Go, Julia, Python (ctypes)

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;

use shrew_core::backprop::GradStore;
use shrew_core::dtype::DType;
use shrew_core::tensor::Tensor;
use shrew_cpu::{CpuBackend, CpuDevice};

type B = CpuBackend;
type ShrewTensor = Tensor<B>;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: impl Into<String>) {
    LAST_ERROR.with(|err| {
        *err.borrow_mut() = CString::new(msg.into()).ok();
    });
}

fn clear_error() {
    LAST_ERROR.with(|err| {
        *err.borrow_mut() = None;
    });
}

// C-compatible opaque wrapper types

#[repr(C)]
pub struct shrew_tensor {
    pub(crate) inner: ShrewTensor,
}

#[repr(C)]
pub struct shrew_grad_store {
    pub(crate) inner: GradStore<B>,
}

#[repr(C)]
pub struct shrew_executor {
    pub(crate) inner: shrew::exec::Executor<B>,
}

// Helper converters

fn parse_c_dtype(dt: i32) -> Option<DType> {
    match dt {
        0 => Some(DType::F32),
        1 => Some(DType::F64),
        2 => Some(DType::I64),
        3 => Some(DType::U8),
        _ => None,
    }
}

// --- Error Handling ---

#[no_mangle]
pub extern "C" fn shrew_last_error() -> *const c_char {
    LAST_ERROR.with(|err| {
        err.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

// --- Tensor Lifecycle & Creation ---

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_from_data(
    data: *const f64,
    shape: *const usize,
    ndim: usize,
    dtype: i32,
    _device: i32,
) -> *mut shrew_tensor {
    clear_error();
    if data.is_null() || shape.is_null() || ndim == 0 {
        set_error("Null pointer or zero ndim passed to shrew_tensor_from_data");
        return ptr::null_mut();
    }

    let dt = match parse_c_dtype(dtype) {
        Some(d) => d,
        None => {
            set_error(format!("Unsupported dtype: {dtype}"));
            return ptr::null_mut();
        }
    };

    let shape_slice = slice::from_raw_parts(shape, ndim);
    let numel: usize = shape_slice.iter().product();
    let data_slice = slice::from_raw_parts(data, numel);

    match ShrewTensor::from_f64_slice(data_slice, shape_slice.to_vec(), dt, &CpuDevice) {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_zeros(
    shape: *const usize,
    ndim: usize,
    dtype: i32,
    _device: i32,
) -> *mut shrew_tensor {
    clear_error();
    if shape.is_null() || ndim == 0 {
        set_error("Null shape pointer or zero ndim");
        return ptr::null_mut();
    }

    let dt = match parse_c_dtype(dtype) {
        Some(d) => d,
        None => {
            set_error(format!("Unsupported dtype: {dtype}"));
            return ptr::null_mut();
        }
    };

    let shape_slice = slice::from_raw_parts(shape, ndim);
    match ShrewTensor::zeros(shape_slice.to_vec(), dt, &CpuDevice) {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_ones(
    shape: *const usize,
    ndim: usize,
    dtype: i32,
    _device: i32,
) -> *mut shrew_tensor {
    clear_error();
    if shape.is_null() || ndim == 0 {
        set_error("Null shape pointer or zero ndim");
        return ptr::null_mut();
    }

    let dt = match parse_c_dtype(dtype) {
        Some(d) => d,
        None => {
            set_error(format!("Unsupported dtype: {dtype}"));
            return ptr::null_mut();
        }
    };

    let shape_slice = slice::from_raw_parts(shape, ndim);
    match ShrewTensor::ones(shape_slice.to_vec(), dt, &CpuDevice) {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_randn(
    shape: *const usize,
    ndim: usize,
    dtype: i32,
    _device: i32,
) -> *mut shrew_tensor {
    clear_error();
    if shape.is_null() || ndim == 0 {
        set_error("Null shape pointer or zero ndim");
        return ptr::null_mut();
    }

    let dt = match parse_c_dtype(dtype) {
        Some(d) => d,
        None => {
            set_error(format!("Unsupported dtype: {dtype}"));
            return ptr::null_mut();
        }
    };

    let shape_slice = slice::from_raw_parts(shape, ndim);
    match ShrewTensor::randn(shape_slice.to_vec(), dt, &CpuDevice) {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_free(tensor: *mut shrew_tensor) {
    if !tensor.is_null() {
        drop(Box::from_raw(tensor));
    }
}

// --- Tensor Metadata & Inspection ---

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_ndim(tensor: *const shrew_tensor) -> usize {
    if tensor.is_null() {
        return 0;
    }
    (*tensor).inner.dims().len()
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_shape(
    tensor: *const shrew_tensor,
    out_shape: *mut usize,
    max_dim: usize,
) -> i32 {
    clear_error();
    if tensor.is_null() || out_shape.is_null() {
        set_error("Null pointer to shrew_tensor_shape");
        return -1;
    }

    let dims = (*tensor).inner.dims();
    let to_copy = dims.len().min(max_dim);
    let dest = slice::from_raw_parts_mut(out_shape, to_copy);
    dest.copy_from_slice(&dims[..to_copy]);
    dims.len() as i32
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_numel(tensor: *const shrew_tensor) -> usize {
    if tensor.is_null() {
        return 0;
    }
    (*tensor).inner.elem_count()
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_to_data(
    tensor: *const shrew_tensor,
    out_data: *mut f64,
    max_len: usize,
) -> i32 {
    clear_error();
    if tensor.is_null() || out_data.is_null() {
        set_error("Null pointer to shrew_tensor_to_data");
        return -1;
    }

    match (*tensor).inner.to_f64_vec() {
        Ok(vec) => {
            let to_copy = vec.len().min(max_len);
            let dest = slice::from_raw_parts_mut(out_data, to_copy);
            dest.copy_from_slice(&vec[..to_copy]);
            vec.len() as i32
        }
        Err(e) => {
            set_error(format!("{e}"));
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_print(tensor: *const shrew_tensor) {
    if !tensor.is_null() {
        println!("{:?}", (*tensor).inner);
    }
}

// --- Tensor Operations ---

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_add(
    a: *const shrew_tensor,
    b: *const shrew_tensor,
) -> *mut shrew_tensor {
    clear_error();
    if a.is_null() || b.is_null() {
        set_error("Null pointer in shrew_tensor_add");
        return ptr::null_mut();
    }
    match (*a).inner.add(&(*b).inner) {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_sub(
    a: *const shrew_tensor,
    b: *const shrew_tensor,
) -> *mut shrew_tensor {
    clear_error();
    if a.is_null() || b.is_null() {
        set_error("Null pointer in shrew_tensor_sub");
        return ptr::null_mut();
    }
    match (*a).inner.sub(&(*b).inner) {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_mul(
    a: *const shrew_tensor,
    b: *const shrew_tensor,
) -> *mut shrew_tensor {
    clear_error();
    if a.is_null() || b.is_null() {
        set_error("Null pointer in shrew_tensor_mul");
        return ptr::null_mut();
    }
    match (*a).inner.mul(&(*b).inner) {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_matmul(
    a: *const shrew_tensor,
    b: *const shrew_tensor,
) -> *mut shrew_tensor {
    clear_error();
    if a.is_null() || b.is_null() {
        set_error("Null pointer in shrew_tensor_matmul");
        return ptr::null_mut();
    }
    match (*a).inner.matmul(&(*b).inner) {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_relu(a: *const shrew_tensor) -> *mut shrew_tensor {
    clear_error();
    if a.is_null() {
        set_error("Null pointer in shrew_tensor_relu");
        return ptr::null_mut();
    }
    match (*a).inner.relu() {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_sigmoid(a: *const shrew_tensor) -> *mut shrew_tensor {
    clear_error();
    if a.is_null() {
        set_error("Null pointer in shrew_tensor_sigmoid");
        return ptr::null_mut();
    }
    match (*a).inner.sigmoid() {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_tanh(a: *const shrew_tensor) -> *mut shrew_tensor {
    clear_error();
    if a.is_null() {
        set_error("Null pointer in shrew_tensor_tanh");
        return ptr::null_mut();
    }
    match (*a).inner.tanh() {
        Ok(t) => Box::into_raw(Box::new(shrew_tensor { inner: t })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

// --- Autograd ---

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_set_variable(a: *mut shrew_tensor) -> *mut shrew_tensor {
    clear_error();
    if a.is_null() {
        set_error("Null pointer in shrew_tensor_set_variable");
        return ptr::null_mut();
    }
    (*a).inner = (*a).inner.clone().set_variable();
    a
}

#[no_mangle]
pub unsafe extern "C" fn shrew_tensor_backward(loss: *const shrew_tensor) -> *mut shrew_grad_store {
    clear_error();
    if loss.is_null() {
        set_error("Null loss pointer in shrew_tensor_backward");
        return ptr::null_mut();
    }
    match (*loss).inner.backward() {
        Ok(store) => Box::into_raw(Box::new(shrew_grad_store { inner: store })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_grad_store_get(
    store: *const shrew_grad_store,
    param: *const shrew_tensor,
) -> *mut shrew_tensor {
    clear_error();
    if store.is_null() || param.is_null() {
        set_error("Null pointer in shrew_grad_store_get");
        return ptr::null_mut();
    }

    match (*store).inner.get(&(*param).inner) {
        Some(g) => Box::into_raw(Box::new(shrew_tensor { inner: g.clone() })),
        None => {
            set_error("No gradient found for given tensor");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_grad_store_free(store: *mut shrew_grad_store) {
    if !store.is_null() {
        drop(Box::from_raw(store));
    }
}

// --- .sw Model Execution ---

#[no_mangle]
pub unsafe extern "C" fn shrew_executor_load_source(
    sw_source: *const c_char,
    dtype: i32,
) -> *mut shrew_executor {
    clear_error();
    if sw_source.is_null() {
        set_error("Null source string in shrew_executor_load_source");
        return ptr::null_mut();
    }

    let c_str = CStr::from_ptr(sw_source);
    let src = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("Invalid UTF-8 source string: {e}"));
            return ptr::null_mut();
        }
    };

    let dt = parse_c_dtype(dtype).unwrap_or(DType::F64);
    let config = shrew::exec::RuntimeConfig::default().with_dtype(dt);

    match shrew::exec::load_program::<B>(src, CpuDevice, config) {
        Ok(exec) => Box::into_raw(Box::new(shrew_executor { inner: exec })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_executor_load_file(
    sw_path: *const c_char,
    dtype: i32,
) -> *mut shrew_executor {
    clear_error();
    if sw_path.is_null() {
        set_error("Null path in shrew_executor_load_file");
        return ptr::null_mut();
    }

    let c_str = CStr::from_ptr(sw_path);
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("Invalid UTF-8 path string: {e}"));
            return ptr::null_mut();
        }
    };

    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("Failed to read '{path}': {e}"));
            return ptr::null_mut();
        }
    };

    let dt = parse_c_dtype(dtype).unwrap_or(DType::F64);
    let config = shrew::exec::RuntimeConfig::default().with_dtype(dt);

    match shrew::exec::load_program::<B>(&source, CpuDevice, config) {
        Ok(exec) => Box::into_raw(Box::new(shrew_executor { inner: exec })),
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_executor_run_single(
    exec: *const shrew_executor,
    graph_name: *const c_char,
    input_name: *const c_char,
    input: *const shrew_tensor,
) -> *mut shrew_tensor {
    clear_error();
    if exec.is_null() || graph_name.is_null() || input_name.is_null() || input.is_null() {
        set_error("Null pointer passed to shrew_executor_run_single");
        return ptr::null_mut();
    }

    let g_name = match CStr::from_ptr(graph_name).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("{e}"));
            return ptr::null_mut();
        }
    };

    let in_name = match CStr::from_ptr(input_name).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("{e}"));
            return ptr::null_mut();
        }
    };

    let mut inputs = std::collections::HashMap::new();
    inputs.insert(in_name.to_string(), (*input).inner.clone());

    match (*exec).inner.run(g_name, &inputs) {
        Ok(res) => {
            if let Some(out) = res.output() {
                Box::into_raw(Box::new(shrew_tensor { inner: out.clone() }))
            } else {
                set_error("Graph produced no primary output");
                ptr::null_mut()
            }
        }
        Err(e) => {
            set_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_executor_train(
    exec: *mut shrew_executor,
    out_final_loss: *mut f64,
) -> i32 {
    clear_error();
    if exec.is_null() {
        set_error("Null pointer passed to shrew_executor_train");
        return -1;
    }

    match (*exec).inner.train() {
        Ok(res) => {
            if !out_final_loss.is_null() {
                *out_final_loss = res.final_loss;
            }
            res.epochs.len() as i32
        }
        Err(e) => {
            set_error(format!("{e}"));
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_train_file(
    sw_path: *const c_char,
    dtype: i32,
    out_final_loss: *mut f64,
) -> i32 {
    clear_error();
    if sw_path.is_null() {
        set_error("Null path in shrew_train_file");
        return -1;
    }

    let c_str = CStr::from_ptr(sw_path);
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("Invalid UTF-8 path string: {e}"));
            return -1;
        }
    };

    let dt = parse_c_dtype(dtype).unwrap_or(DType::F64);
    let config = shrew::exec::RuntimeConfig::default().with_dtype(dt);

    match shrew::exec::train_file::<B>(path, CpuDevice, config) {
        Ok((_trainer, res)) => {
            if !out_final_loss.is_null() {
                *out_final_loss = res.final_loss;
            }
            res.epochs.len() as i32
        }
        Err(e) => {
            set_error(format!("{e}"));
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn shrew_executor_free(exec: *mut shrew_executor) {
    if !exec.is_null() {
        drop(Box::from_raw(exec));
    }
}

// --- Hardware & Diagnostics ---

#[no_mangle]
pub extern "C" fn shrew_cuda_is_available() -> i32 {
    #[cfg(feature = "cuda")]
    {
        if shrew_cuda::CudaDevice::new(0).is_ok() {
            1
        } else {
            0
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

#[no_mangle]
pub extern "C" fn shrew_cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        let mut count = 0;
        while shrew_cuda::CudaDevice::new(count).is_ok() {
            count += 1;
        }
        count
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

#[no_mangle]
pub extern "C" fn shrew_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}
