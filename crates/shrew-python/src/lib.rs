// shrew-python — Complete PyO3 bindings for the Shrew deep learning library
//
// Exposes the full Shrew API to Python:
//   - Tensor (creation, arithmetic, reductions, shape ops, comparisons, autograd)
//   - NN Layers: Linear, Conv2d, Conv1d, BatchNorm2d, LayerNorm, GroupNorm,
//     RMSNorm, Embedding, Dropout, Flatten, MaxPool2d, AvgPool2d,
//     AdaptiveAvgPool2d, MultiHeadAttention, TransformerBlock,
//     RNNCell, RNN, LSTMCell, LSTM, GRUCell, GRU
//   - Activations: ReLU, GeLU, SiLU, Sigmoid, Tanh, LeakyReLU, ELU, Mish
//   - Optimizers: SGD, Adam, AdamW, RMSProp, RAdam
//   - LR Schedulers: StepLR, ExponentialLR, LinearLR, CosineAnnealingLR,
//     CosineWarmupLR, ReduceLROnPlateau
//   - Losses: mse_loss, cross_entropy_loss, l1_loss, smooth_l1_loss,
//     bce_loss, bce_with_logits_loss, nll_loss
//   - Gradient utilities: clip_grad_norm, clip_grad_value, grad_norm
//   - Data: MnistDataset, CsvDataset
//   - I/O: save_safetensors, load_safetensors
//
// Build: `maturin develop --release` or `pip install .`
// Usage: `import shrew_python as shrew`

use numpy::ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use shrew_core::backprop::GradStore;
use shrew_core::DType;
use shrew_cpu::{CpuBackend, CpuDevice};

type B = CpuBackend;
type ShrewTensor = shrew_core::tensor::Tensor<B>;

// Helpers

fn to_py_err(e: shrew_core::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{}", e))
}

fn parse_dtype(s: &str) -> PyResult<DType> {
    match s {
        "f32" | "float32" => Ok(DType::F32),
        "f64" | "float64" => Ok(DType::F64),
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "u8" | "uint8" => Ok(DType::U8),
        "u32" | "uint32" => Ok(DType::U32),
        "i64" | "int64" => Ok(DType::I64),
        _ => Err(PyValueError::new_err(format!("Unknown dtype: {}", s))),
    }
}

fn dtype_to_str(dt: DType) -> &'static str {
    match dt {
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::U8 => "uint8",
        DType::U32 => "uint32",
        DType::I64 => "int64",
    }
}

// PyTensor — Full Python wrapper around Tensor<CpuBackend>

/// A multi-dimensional tensor, backed by the Shrew CPU engine.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
struct PyTensor {
    inner: ShrewTensor,
}

#[pymethods]
impl PyTensor {
    //  Creation 

    /// Create a tensor from a flat list and shape.
    #[staticmethod]
    #[pyo3(signature = (data, shape, dtype="f32"))]
    fn from_list(data: Vec<f64>, shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let t = ShrewTensor::from_f64_slice(&data, shape, dt, &CpuDevice).map_err(to_py_err)?;
        Ok(PyTensor { inner: t })
    }

    /// Create a tensor from a NumPy array.
    #[staticmethod]
    fn from_numpy(_py: Python<'_>, arr: &Bound<'_, PyArrayDyn<f64>>) -> PyResult<Self> {
        let readonly = arr
            .try_readonly()
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot read array: {}", e)))?;
        let view = readonly.as_array();
        let shape: Vec<usize> = view.shape().to_vec();
        let data: Vec<f64> = view.iter().cloned().collect();
        let t =
            ShrewTensor::from_f64_slice(&data, shape, DType::F64, &CpuDevice).map_err(to_py_err)?;
        Ok(PyTensor { inner: t })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype="f32"))]
    fn zeros(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::zeros(shape, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype="f32"))]
    fn ones(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::ones(shape, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, val, dtype="f32"))]
    fn full(shape: Vec<usize>, val: f64, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::full(shape, val, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype="f32"))]
    fn rand(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::rand(shape, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype="f32"))]
    fn randn(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::randn(shape, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (start, end, steps, dtype="f32"))]
    fn linspace(start: f64, end: f64, steps: usize, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::linspace(start, end, steps, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (n, dtype="f32"))]
    fn eye(n: usize, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::eye(n, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (n, dtype="f32"))]
    fn arange(n: usize, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::arange(n, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (start, end, step, dtype="f32"))]
    fn arange_step(start: f64, end: f64, step: f64, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::arange_step(start, end, step, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (n, m, diagonal=0, dtype="f32"))]
    fn triu(n: usize, m: usize, diagonal: i64, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::triu(n, m, diagonal, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (n, m, diagonal=0, dtype="f32"))]
    fn tril(n: usize, m: usize, diagonal: i64, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: ShrewTensor::tril(n, m, diagonal, dt, &CpuDevice).map_err(to_py_err)?,
        })
    }

    fn zeros_like(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: ShrewTensor::zeros_like(&self.inner).map_err(to_py_err)?,
        })
    }

    fn ones_like(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: ShrewTensor::ones_like(&self.inner).map_err(to_py_err)?,
        })
    }

    fn full_like(&self, val: f64) -> PyResult<Self> {
        Ok(PyTensor {
            inner: ShrewTensor::full_like(&self.inner, val).map_err(to_py_err)?,
        })
    }

    //  Properties 

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.dims().to_vec()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_to_str(self.inner.dtype())
    }

    #[getter]
    fn numel(&self) -> usize {
        self.inner.elem_count()
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.is_variable()
    }

    #[getter]
    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    //  Conversions 

    fn to_list(&self) -> PyResult<Vec<f64>> {
        self.inner.to_f64_vec().map_err(to_py_err)
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let data = self.inner.to_f64_vec().map_err(to_py_err)?;
        let shape: Vec<usize> = self.inner.dims().to_vec();
        let arr = ArrayD::from_shape_vec(numpy::ndarray::IxDyn(&shape), data)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(arr.into_pyarray(py))
    }

    fn item(&self) -> PyResult<f64> {
        self.inner.to_scalar_f64().map_err(to_py_err)
    }

    fn requires_grad_(&self) -> Self {
        PyTensor {
            inner: self.inner.clone().set_variable(),
        }
    }

    fn detach(&self) -> Self {
        PyTensor {
            inner: self.inner.detach(),
        }
    }

    fn freeze(&self) -> Self {
        PyTensor {
            inner: self.inner.freeze(),
        }
    }

    fn unfreeze(&self) -> Self {
        PyTensor {
            inner: self.inner.unfreeze(),
        }
    }

    fn to_dtype(&self, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(PyTensor {
            inner: self.inner.to_dtype(dt).map_err(to_py_err)?,
        })
    }

    fn contiguous(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.contiguous().map_err(to_py_err)?,
        })
    }

    //  Arithmetic 

    fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.add(&other.inner).map_err(to_py_err)?,
        })
    }
    fn __sub__(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.sub(&other.inner).map_err(to_py_err)?,
        })
    }
    fn __mul__(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.mul(&other.inner).map_err(to_py_err)?,
        })
    }
    fn __truediv__(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.div(&other.inner).map_err(to_py_err)?,
        })
    }
    fn __neg__(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.neg().map_err(to_py_err)?,
        })
    }
    fn __matmul__(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.matmul(&other.inner).map_err(to_py_err)?,
        })
    }

    //  Comparisons 

    fn eq(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.eq(&other.inner).map_err(to_py_err)?,
        })
    }
    fn ne(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.ne(&other.inner).map_err(to_py_err)?,
        })
    }
    fn gt(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.gt(&other.inner).map_err(to_py_err)?,
        })
    }
    fn ge(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.ge(&other.inner).map_err(to_py_err)?,
        })
    }
    fn lt(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.lt(&other.inner).map_err(to_py_err)?,
        })
    }
    fn le(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.le(&other.inner).map_err(to_py_err)?,
        })
    }

    //  Unary ops 

    fn exp(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.exp().map_err(to_py_err)?,
        })
    }
    fn log(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.log().map_err(to_py_err)?,
        })
    }
    fn sqrt(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.sqrt().map_err(to_py_err)?,
        })
    }
    fn abs(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.abs().map_err(to_py_err)?,
        })
    }
    fn relu(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.relu().map_err(to_py_err)?,
        })
    }
    fn sigmoid(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.sigmoid().map_err(to_py_err)?,
        })
    }
    fn tanh(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.tanh().map_err(to_py_err)?,
        })
    }
    fn gelu(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.gelu().map_err(to_py_err)?,
        })
    }
    fn silu(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.silu().map_err(to_py_err)?,
        })
    }
    fn sin(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.sin().map_err(to_py_err)?,
        })
    }
    fn cos(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.cos().map_err(to_py_err)?,
        })
    }
    fn square(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.square().map_err(to_py_err)?,
        })
    }
    fn floor(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.floor().map_err(to_py_err)?,
        })
    }
    fn ceil(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.ceil().map_err(to_py_err)?,
        })
    }
    fn round(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.round().map_err(to_py_err)?,
        })
    }
    fn reciprocal(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.reciprocal().map_err(to_py_err)?,
        })
    }
    fn rsqrt(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.rsqrt().map_err(to_py_err)?,
        })
    }
    fn sign(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.sign().map_err(to_py_err)?,
        })
    }

    //  Special element-wise 

    fn powf(&self, exponent: f64) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.powf(exponent).map_err(to_py_err)?,
        })
    }

    fn clamp(&self, min_val: f64, max_val: f64) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.clamp(min_val, max_val).map_err(to_py_err)?,
        })
    }

    fn affine(&self, mul: f64, add: f64) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.affine(mul, add).map_err(to_py_err)?,
        })
    }

    fn masked_fill(&self, mask: &PyTensor, value: f64) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self
                .inner
                .masked_fill(&mask.inner, value)
                .map_err(to_py_err)?,
        })
    }

    //  Composite ops 

    fn softmax(&self, dim: usize) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.softmax(dim).map_err(to_py_err)?,
        })
    }
    fn log_softmax(&self, dim: usize) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.log_softmax(dim).map_err(to_py_err)?,
        })
    }

    //  Reductions 

    fn sum_all(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.sum_all().map_err(to_py_err)?,
        })
    }
    fn mean_all(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.mean_all().map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (dim, keep_dim=false))]
    fn sum(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.sum(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    #[pyo3(signature = (dim, keep_dim=false))]
    fn mean(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.mean(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    #[pyo3(signature = (dim, keep_dim=false))]
    fn max(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.max(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    #[pyo3(signature = (dim, keep_dim=false))]
    fn min(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.min(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    #[pyo3(signature = (dim, keep_dim=false))]
    fn argmax(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.argmax(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    #[pyo3(signature = (dim, keep_dim=false))]
    fn argmin(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.argmin(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    #[pyo3(signature = (dim, keep_dim=false))]
    fn var(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.var(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    #[pyo3(signature = (dim, keep_dim=false))]
    fn std(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.std(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    #[pyo3(signature = (dim, keep_dim=false))]
    fn logsumexp(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.logsumexp(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    #[pyo3(signature = (dim, keep_dim=false))]
    fn prod(&self, dim: usize, keep_dim: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.prod(dim, keep_dim).map_err(to_py_err)?,
        })
    }
    fn cumsum(&self, dim: usize) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.cumsum(dim).map_err(to_py_err)?,
        })
    }

    //  Shape ops 

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.reshape(shape).map_err(to_py_err)?,
        })
    }
    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.transpose(dim0, dim1).map_err(to_py_err)?,
        })
    }
    #[getter]
    fn t(&self) -> PyResult<Self> {
        let r = self.inner.rank();
        if r < 2 {
            return Err(PyValueError::new_err(
                "Cannot transpose tensor with < 2 dims",
            ));
        }
        Ok(PyTensor {
            inner: self.inner.transpose(r - 2, r - 1).map_err(to_py_err)?,
        })
    }
    fn permute(&self, dims: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.permute(&dims).map_err(to_py_err)?,
        })
    }
    fn squeeze(&self, dim: usize) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.squeeze(dim).map_err(to_py_err)?,
        })
    }
    fn squeeze_all(&self) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.squeeze_all(),
        })
    }
    fn unsqueeze(&self, dim: usize) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.unsqueeze(dim).map_err(to_py_err)?,
        })
    }
    fn flatten(&self, start_dim: usize, end_dim: usize) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.flatten(start_dim, end_dim).map_err(to_py_err)?,
        })
    }
    fn narrow(&self, dim: usize, start: usize, len: usize) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.narrow(dim, start, len).map_err(to_py_err)?,
        })
    }
    fn expand(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.expand(shape.as_slice()).map_err(to_py_err)?,
        })
    }
    fn chunk(&self, n: usize, dim: usize) -> PyResult<Vec<Self>> {
        let chunks = self.inner.chunk(n, dim).map_err(to_py_err)?;
        Ok(chunks.into_iter().map(|c| PyTensor { inner: c }).collect())
    }
    fn split(&self, split_size: usize, dim: usize) -> PyResult<Vec<Self>> {
        let parts = self.inner.split(split_size, dim).map_err(to_py_err)?;
        Ok(parts.into_iter().map(|p| PyTensor { inner: p }).collect())
    }
    #[pyo3(signature = (padding, value=0.0))]
    fn pad(&self, padding: Vec<[usize; 2]>, value: f64) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.pad(&padding, value).map_err(to_py_err)?,
        })
    }

    //  Static shape ops 

    #[staticmethod]
    fn cat(tensors: Vec<PyRef<PyTensor>>, dim: usize) -> PyResult<Self> {
        let inners: Vec<ShrewTensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        Ok(PyTensor {
            inner: ShrewTensor::cat(&inners, dim).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn stack(tensors: Vec<PyRef<PyTensor>>, dim: usize) -> PyResult<Self> {
        let inners: Vec<ShrewTensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        Ok(PyTensor {
            inner: ShrewTensor::stack(&inners, dim).map_err(to_py_err)?,
        })
    }

    //  Indexing / selection 

    fn index_select(&self, dim: usize, indices: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self
                .inner
                .index_select(dim, &indices.inner)
                .map_err(to_py_err)?,
        })
    }
    fn gather(&self, dim: usize, index: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.gather(dim, &index.inner).map_err(to_py_err)?,
        })
    }
    fn where_cond(&self, on_true: &PyTensor, on_false: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: ShrewTensor::where_cond(&self.inner, &on_true.inner, &on_false.inner)
                .map_err(to_py_err)?,
        })
    }

    //  Sorting 

    #[pyo3(signature = (dim, descending=false))]
    fn sort(&self, dim: usize, descending: bool) -> PyResult<(Self, Self)> {
        let (vals, idxs) = self.inner.sort(dim, descending).map_err(to_py_err)?;
        Ok((PyTensor { inner: vals }, PyTensor { inner: idxs }))
    }
    #[pyo3(signature = (dim, descending=false))]
    fn argsort(&self, dim: usize, descending: bool) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.argsort(dim, descending).map_err(to_py_err)?,
        })
    }
    fn topk(&self, k: usize, dim: usize) -> PyResult<(Self, Vec<usize>)> {
        let (vals, idxs) = self.inner.topk(k, dim).map_err(to_py_err)?;
        Ok((PyTensor { inner: vals }, idxs))
    }

    //  Conv/Pool convenience 

    #[pyo3(signature = (weight, bias=None, stride=[1,1], padding=[0,0]))]
    fn conv2d(
        &self,
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self
                .inner
                .conv2d(&weight.inner, bias.map(|b| &b.inner), stride, padding)
                .map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (weight, bias=None, stride=1, padding=0))]
    fn conv1d(
        &self,
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        stride: usize,
        padding: usize,
    ) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self
                .inner
                .conv1d(&weight.inner, bias.map(|b| &b.inner), stride, padding)
                .map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (kernel_size, stride=[1,1], padding=[0,0]))]
    fn max_pool2d(
        &self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self
                .inner
                .max_pool2d(kernel_size, stride, padding)
                .map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (kernel_size, stride=[1,1], padding=[0,0]))]
    fn avg_pool2d(
        &self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self
                .inner
                .avg_pool2d(kernel_size, stride, padding)
                .map_err(to_py_err)?,
        })
    }

    //  Matmul 

    fn matmul(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor {
            inner: self.inner.matmul(&other.inner).map_err(to_py_err)?,
        })
    }

    //  Autograd 

    fn backward(&self) -> PyResult<PyGradStore> {
        Ok(PyGradStore {
            inner: self.inner.backward().map_err(to_py_err)?,
        })
    }

    //  Display 

    fn __repr__(&self) -> String {
        let dt = dtype_to_str(self.inner.dtype());
        let shape = self.inner.dims();
        if self.inner.elem_count() <= 10 {
            if let Ok(data) = self.inner.to_f64_vec() {
                return format!("Tensor({:?}, shape={:?}, dtype={})", data, shape, dt);
            }
        }
        format!("Tensor(shape={:?}, dtype={})", shape, dt)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// PyGradStore

#[pyclass(name = "GradStore")]
#[derive(Clone)]
struct PyGradStore {
    inner: GradStore<B>,
}

#[pymethods]
impl PyGradStore {
    fn grad(&self, tensor: &PyTensor) -> Option<PyTensor> {
        self.inner
            .get(&tensor.inner)
            .map(|g| PyTensor { inner: g.clone() })
    }
}

// Executor — Load and run .sw model files from Python

#[pyclass(name = "Executor")]
struct PyExecutor {
    inner: shrew::exec::Executor<B>,
}

#[pymethods]
impl PyExecutor {
    /// Load a .sw model from source code string.
    ///
    /// ```python
    /// source = open("model.sw").read()
    /// exec = shrew.Executor.from_source(source)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (source, dtype="f64", training=false))]
    fn from_source(source: &str, dtype: &str, training: bool) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let config = shrew::exec::RuntimeConfig::default()
            .with_dtype(dt)
            .with_training(training);
        let exec = shrew::exec::load_program::<B>(source, CpuDevice, config)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(PyExecutor { inner: exec })
    }

    /// Load a .sw model from a file path.
    ///
    /// ```python
    /// exec = shrew.Executor.load("examples/demo_mlp.sw")
    /// ```
    #[staticmethod]
    #[pyo3(signature = (path, dtype="f64", training=false))]
    fn load(path: &str, dtype: &str, training: bool) -> PyResult<Self> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("Cannot read file '{}': {}", path, e)))?;
        Self::from_source(&source, dtype, training)
    }

    /// List graph names in the program.
    fn graph_names(&self) -> Vec<String> {
        self.inner
            .program()
            .graphs
            .iter()
            .map(|g| g.name.clone())
            .collect()
    }

    /// Run a named graph with provided input tensors.
    ///
    /// ```python
    /// result = exec.run("Forward", {"x": input_tensor})
    /// output = result["out"]
    /// ```
    fn run(
        &self,
        graph_name: &str,
        inputs: std::collections::HashMap<String, PyRef<PyTensor>>,
    ) -> PyResult<std::collections::HashMap<String, PyTensor>> {
        let rust_inputs: std::collections::HashMap<String, ShrewTensor> = inputs
            .into_iter()
            .map(|(k, v)| (k, v.inner.clone()))
            .collect();
        let result = self
            .inner
            .run(graph_name, &rust_inputs)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        let outputs = result
            .outputs
            .into_iter()
            .map(|(k, v)| (k, PyTensor { inner: v }))
            .collect();
        Ok(outputs)
    }

    /// Get all named parameters as a dict {"graph/param": Tensor}.
    fn named_params(&self) -> std::collections::HashMap<String, PyTensor> {
        self.inner
            .named_params()
            .into_iter()
            .map(|(k, v)| (k, PyTensor { inner: v }))
            .collect()
    }

    /// Set a parameter by key (e.g. "Forward/w1").
    fn set_param(&mut self, key: &str, tensor: &PyTensor) -> bool {
        self.inner.set_param_by_key(key, tensor.inner.clone())
    }

    /// Get the list of input names for a graph.
    fn input_names(&self, graph_name: &str) -> PyResult<Vec<String>> {
        let graph =
            self.inner.program().get_graph(graph_name).ok_or_else(|| {
                PyValueError::new_err(format!("Graph '{}' not found", graph_name))
            })?;
        Ok(graph
            .inputs
            .iter()
            .map(|id| graph.nodes[id.0].name.clone())
            .collect())
    }

    /// Get the list of output names for a graph.
    fn output_names(&self, graph_name: &str) -> PyResult<Vec<String>> {
        let graph =
            self.inner.program().get_graph(graph_name).ok_or_else(|| {
                PyValueError::new_err(format!("Graph '{}' not found", graph_name))
            })?;
        Ok(graph.outputs.iter().map(|o| o.name.clone()).collect())
    }

    /// Get param count for a graph.
    fn param_count(&self, graph_name: &str) -> PyResult<usize> {
        let graph =
            self.inner.program().get_graph(graph_name).ok_or_else(|| {
                PyValueError::new_err(format!("Graph '{}' not found", graph_name))
            })?;
        Ok(graph.params.len())
    }

    fn __repr__(&self) -> String {
        let graphs: Vec<String> = self
            .inner
            .program()
            .graphs
            .iter()
            .map(|g| {
                format!(
                    "{}({} nodes, {} params)",
                    g.name,
                    g.nodes.len(),
                    g.params.len()
                )
            })
            .collect();
        format!("Executor(graphs=[{}])", graphs.join(", "))
    }
}

// NN Layers

//  Linear 

#[pyclass(name = "Linear")]
struct PyLinear {
    inner: shrew_nn::Linear<B>,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true, dtype="f32"))]
    fn new(in_features: usize, out_features: usize, bias: bool, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer = shrew_nn::Linear::new(in_features, out_features, bias, dt, &CpuDevice)
            .map_err(to_py_err)?;
        Ok(PyLinear { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }

    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  Conv2d 

#[pyclass(name = "Conv2d")]
struct PyConv2d {
    inner: shrew_nn::Conv2d<B>,
}

#[pymethods]
impl PyConv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=[1,1], padding=[0,0], bias=true, dtype="f32"))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        bias: bool,
        dtype: &str,
    ) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer = shrew_nn::Conv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            dt,
            &CpuDevice,
        )
        .map_err(to_py_err)?;
        Ok(PyConv2d { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  Conv1d 

#[pyclass(name = "Conv1d")]
struct PyConv1d {
    inner: shrew_nn::Conv1d<B>,
}

#[pymethods]
impl PyConv1d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, bias=true, dtype="f32"))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        dtype: &str,
    ) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer = shrew_nn::Conv1d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            dt,
            &CpuDevice,
        )
        .map_err(to_py_err)?;
        Ok(PyConv1d { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  BatchNorm2d 

#[pyclass(name = "BatchNorm2d", unsendable)]
struct PyBatchNorm2d {
    inner: shrew_nn::BatchNorm2d<B>,
}

#[pymethods]
impl PyBatchNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, dtype="f32"))]
    fn new(num_features: usize, eps: f64, momentum: f64, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer = shrew_nn::BatchNorm2d::new(num_features, eps, momentum, dt, &CpuDevice)
            .map_err(to_py_err)?;
        Ok(PyBatchNorm2d { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  LayerNorm 

#[pyclass(name = "LayerNorm")]
struct PyLayerNorm {
    inner: shrew_nn::LayerNorm<B>,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (normalized_size, eps=1e-5, dtype="f32"))]
    fn new(normalized_size: usize, eps: f64, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer =
            shrew_nn::LayerNorm::new(normalized_size, eps, dt, &CpuDevice).map_err(to_py_err)?;
        Ok(PyLayerNorm { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  GroupNorm 

#[pyclass(name = "GroupNorm")]
struct PyGroupNorm {
    inner: shrew_nn::GroupNorm<B>,
}

#[pymethods]
impl PyGroupNorm {
    #[new]
    #[pyo3(signature = (num_groups, num_channels, eps=1e-5, dtype="f32"))]
    fn new(num_groups: usize, num_channels: usize, eps: f64, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer = shrew_nn::GroupNorm::new(num_groups, num_channels, eps, dt, &CpuDevice)
            .map_err(to_py_err)?;
        Ok(PyGroupNorm { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  RMSNorm 

#[pyclass(name = "RMSNorm")]
struct PyRMSNorm {
    inner: shrew_nn::RMSNorm<B>,
}

#[pymethods]
impl PyRMSNorm {
    #[new]
    #[pyo3(signature = (normalized_size, eps=1e-5, dtype="f32"))]
    fn new(normalized_size: usize, eps: f64, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer =
            shrew_nn::RMSNorm::new(normalized_size, eps, dt, &CpuDevice).map_err(to_py_err)?;
        Ok(PyRMSNorm { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  Embedding 

#[pyclass(name = "Embedding")]
struct PyEmbedding {
    inner: shrew_nn::Embedding<B>,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, dtype="f32"))]
    fn new(num_embeddings: usize, embedding_dim: usize, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer = shrew_nn::Embedding::new(num_embeddings, embedding_dim, dt, &CpuDevice)
            .map_err(to_py_err)?;
        Ok(PyEmbedding { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  Dropout 

#[pyclass(name = "Dropout", unsendable)]
struct PyDropout {
    inner: shrew_nn::Dropout,
}

#[pymethods]
impl PyDropout {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f64) -> Self {
        PyDropout {
            inner: shrew_nn::Dropout::new(p),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.forward_t::<B>(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

//  Flatten 

#[pyclass(name = "FlattenLayer")]
struct PyFlatten {
    inner: shrew_nn::Flatten,
}

#[pymethods]
impl PyFlatten {
    #[new]
    #[pyo3(signature = (start_dim=1))]
    fn new(start_dim: usize) -> Self {
        PyFlatten {
            inner: shrew_nn::Flatten::new(start_dim),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

//  MaxPool2d 

#[pyclass(name = "MaxPool2d")]
struct PyMaxPool2d {
    inner: shrew_nn::MaxPool2d,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=[1,1], padding=[0,0]))]
    fn new(kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> Self {
        PyMaxPool2d {
            inner: shrew_nn::MaxPool2d::new(kernel_size, stride, padding),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

//  AvgPool2d 

#[pyclass(name = "AvgPool2d")]
struct PyAvgPool2d {
    inner: shrew_nn::AvgPool2d,
}

#[pymethods]
impl PyAvgPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=[1,1], padding=[0,0]))]
    fn new(kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> Self {
        PyAvgPool2d {
            inner: shrew_nn::AvgPool2d::new(kernel_size, stride, padding),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

//  AdaptiveAvgPool2d 

#[pyclass(name = "AdaptiveAvgPool2d")]
struct PyAdaptiveAvgPool2d {
    inner: shrew_nn::AdaptiveAvgPool2d,
}

#[pymethods]
impl PyAdaptiveAvgPool2d {
    #[new]
    fn new(output_size: [usize; 2]) -> Self {
        PyAdaptiveAvgPool2d {
            inner: shrew_nn::AdaptiveAvgPool2d::new(output_size),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

//  MultiHeadAttention 

#[pyclass(name = "MultiHeadAttention")]
struct PyMultiHeadAttention {
    inner: shrew_nn::MultiHeadAttention<B>,
}

#[pymethods]
impl PyMultiHeadAttention {
    #[new]
    #[pyo3(signature = (d_model, num_heads, causal=false, dtype="f32"))]
    fn new(d_model: usize, num_heads: usize, causal: bool, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer = shrew_nn::MultiHeadAttention::new(d_model, num_heads, dt, &CpuDevice)
            .map_err(to_py_err)?
            .with_causal(causal);
        Ok(PyMultiHeadAttention { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  TransformerBlock 

#[pyclass(name = "TransformerBlock")]
struct PyTransformerBlock {
    inner: shrew_nn::TransformerBlock<B>,
}

#[pymethods]
impl PyTransformerBlock {
    #[new]
    #[pyo3(signature = (d_model, num_heads, d_ff, causal=false, dtype="f32"))]
    fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        causal: bool,
        dtype: &str,
    ) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let layer =
            shrew_nn::TransformerBlock::new(d_model, num_heads, d_ff, causal, dt, &CpuDevice)
                .map_err(to_py_err)?;
        Ok(PyTransformerBlock { inner: layer })
    }

    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<PyTensor> {
        use shrew_nn::Module;
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  RNNCell

#[pyclass(name = "RNNCell")]
struct PyRNNCell {
    inner: shrew_nn::RNNCell<B>,
}

#[pymethods]
impl PyRNNCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, dtype="f32"))]
    fn new(input_size: usize, hidden_size: usize, bias: bool, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let cell = shrew_nn::RNNCell::new(input_size, hidden_size, bias, dt, &CpuDevice)
            .map_err(to_py_err)?;
        Ok(PyRNNCell { inner: cell })
    }

    fn forward(&self, x: &PyTensor, h: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner, &h.inner).map_err(to_py_err)?,
        })
    }
    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  RNN 

#[pyclass(name = "RNN")]
struct PyRNN {
    inner: shrew_nn::RNN<B>,
}

#[pymethods]
impl PyRNN {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, dtype="f32"))]
    fn new(input_size: usize, hidden_size: usize, bias: bool, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let rnn =
            shrew_nn::RNN::new(input_size, hidden_size, bias, dt, &CpuDevice).map_err(to_py_err)?;
        Ok(PyRNN { inner: rnn })
    }

    /// Returns (output, h_n).
    #[pyo3(signature = (x, h0=None))]
    fn forward(&self, x: &PyTensor, h0: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
        let (out, hn) = self
            .inner
            .forward(&x.inner, h0.map(|h| &h.inner))
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: out }, PyTensor { inner: hn }))
    }
    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  LSTMCell 

#[pyclass(name = "LSTMCell")]
struct PyLSTMCell {
    inner: shrew_nn::LSTMCell<B>,
}

#[pymethods]
impl PyLSTMCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, dtype="f32"))]
    fn new(input_size: usize, hidden_size: usize, bias: bool, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let cell = shrew_nn::LSTMCell::new(input_size, hidden_size, bias, dt, &CpuDevice)
            .map_err(to_py_err)?;
        Ok(PyLSTMCell { inner: cell })
    }

    /// Returns (h', c').
    fn forward(&self, x: &PyTensor, h: &PyTensor, c: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
        let (h_new, c_new) = self
            .inner
            .forward(&x.inner, &h.inner, &c.inner)
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: h_new }, PyTensor { inner: c_new }))
    }
    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  LSTM 

#[pyclass(name = "LSTM")]
struct PyLSTM {
    inner: shrew_nn::LSTM<B>,
}

#[pymethods]
impl PyLSTM {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, dtype="f32"))]
    fn new(input_size: usize, hidden_size: usize, bias: bool, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let lstm = shrew_nn::LSTM::new(input_size, hidden_size, bias, dt, &CpuDevice)
            .map_err(to_py_err)?;
        Ok(PyLSTM { inner: lstm })
    }

    /// Returns (output, (h_n, c_n)).
    #[pyo3(signature = (x, h0=None, c0=None))]
    fn forward(
        &self,
        x: &PyTensor,
        h0: Option<&PyTensor>,
        c0: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
        let hc0 = match (h0, c0) {
            (Some(h), Some(c)) => Some((&h.inner, &c.inner)),
            _ => None,
        };
        let (out, (hn, cn)) = self.inner.forward(&x.inner, hc0).map_err(to_py_err)?;
        Ok((
            PyTensor { inner: out },
            PyTensor { inner: hn },
            PyTensor { inner: cn },
        ))
    }
    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  GRUCell 

#[pyclass(name = "GRUCell")]
struct PyGRUCell {
    inner: shrew_nn::GRUCell<B>,
}

#[pymethods]
impl PyGRUCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, dtype="f32"))]
    fn new(input_size: usize, hidden_size: usize, bias: bool, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let cell = shrew_nn::GRUCell::new(input_size, hidden_size, bias, dt, &CpuDevice)
            .map_err(to_py_err)?;
        Ok(PyGRUCell { inner: cell })
    }

    fn forward(&self, x: &PyTensor, h: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner, &h.inner).map_err(to_py_err)?,
        })
    }
    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

//  GRU 

#[pyclass(name = "GRU")]
struct PyGRU {
    inner: shrew_nn::GRU<B>,
}

#[pymethods]
impl PyGRU {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, dtype="f32"))]
    fn new(input_size: usize, hidden_size: usize, bias: bool, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let gru =
            shrew_nn::GRU::new(input_size, hidden_size, bias, dt, &CpuDevice).map_err(to_py_err)?;
        Ok(PyGRU { inner: gru })
    }

    /// Returns (output, h_n).
    #[pyo3(signature = (x, h0=None))]
    fn forward(&self, x: &PyTensor, h0: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
        let (out, hn) = self
            .inner
            .forward(&x.inner, h0.map(|h| &h.inner))
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: out }, PyTensor { inner: hn }))
    }
    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| PyTensor { inner: p })
            .collect()
    }
}

// Activation Modules

macro_rules! py_activation_unit {
    ($py_name:ident, $name:literal, $rust_ty:ty, $ctor:expr) => {
        #[pyclass(name = $name)]
        struct $py_name {
            inner: $rust_ty,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new() -> Self {
                $py_name { inner: $ctor }
            }

            fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
                use shrew_nn::Module;
                Ok(PyTensor {
                    inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
                })
            }

            fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
                self.forward(x)
            }
        }
    };
}

py_activation_unit!(PyReLU, "ReLU", shrew_nn::ReLU, shrew_nn::ReLU);
py_activation_unit!(PyGeLU, "GeLU", shrew_nn::GeLU, shrew_nn::GeLU);
py_activation_unit!(PySiLU, "SiLU", shrew_nn::SiLU, shrew_nn::SiLU);
py_activation_unit!(
    PySigmoidAct,
    "SigmoidAct",
    shrew_nn::Sigmoid,
    shrew_nn::Sigmoid
);
py_activation_unit!(PyTanhAct, "TanhAct", shrew_nn::Tanh, shrew_nn::Tanh);
py_activation_unit!(PyMish, "Mish", shrew_nn::Mish, shrew_nn::Mish);

#[pyclass(name = "LeakyReLU")]
struct PyLeakyReLU {
    inner: shrew_nn::LeakyReLU,
}

#[pymethods]
impl PyLeakyReLU {
    #[new]
    #[pyo3(signature = (negative_slope=0.01))]
    fn new(negative_slope: f64) -> Self {
        PyLeakyReLU {
            inner: shrew_nn::LeakyReLU::with_slope(negative_slope),
        }
    }
    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[pyclass(name = "ELU")]
struct PyELU {
    inner: shrew_nn::ELU,
}

#[pymethods]
impl PyELU {
    #[new]
    #[pyo3(signature = (alpha=1.0))]
    fn new(alpha: f64) -> Self {
        PyELU {
            inner: shrew_nn::ELU::with_alpha(alpha),
        }
    }
    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        use shrew_nn::Module;
        Ok(PyTensor {
            inner: self.inner.forward(&x.inner).map_err(to_py_err)?,
        })
    }
    fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

// Optimizers

//  SGD 

#[pyclass(name = "SGD")]
struct PySGD {
    inner: shrew_optim::SGD<B>,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr, momentum=0.0, weight_decay=0.0))]
    fn new(
        params: Vec<PyRef<PyTensor>>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        let ps: Vec<ShrewTensor> = params.iter().map(|t| t.inner.clone()).collect();
        Ok(PySGD {
            inner: shrew_optim::SGD::new(ps, lr, momentum, weight_decay),
        })
    }

    fn step(&mut self, grads: &PyGradStore) -> PyResult<()> {
        use shrew_optim::Optimizer;
        self.inner.step(&grads.inner).map_err(to_py_err)?;
        Ok(())
    }
}

//  Adam 

#[pyclass(name = "Adam")]
struct PyAdam {
    inner: shrew_optim::Adam<B>,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0))]
    fn new(
        params: Vec<PyRef<PyTensor>>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        let ps: Vec<ShrewTensor> = params.iter().map(|t| t.inner.clone()).collect();
        let opt = shrew_optim::Adam::new(ps, lr)
            .beta1(beta1)
            .beta2(beta2)
            .epsilon(eps)
            .weight_decay(weight_decay);
        Ok(PyAdam { inner: opt })
    }

    fn step(&mut self, grads: &PyGradStore) -> PyResult<()> {
        use shrew_optim::Optimizer;
        self.inner.step(&grads.inner).map_err(to_py_err)?;
        Ok(())
    }
}

//  AdamW 

#[pyclass(name = "AdamW")]
struct PyAdamW {
    inner: shrew_optim::AdamW<B>,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (params, lr=1e-3, weight_decay=0.01, beta1=0.9, beta2=0.999))]
    fn new(
        params: Vec<PyRef<PyTensor>>,
        lr: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
    ) -> PyResult<Self> {
        let ps: Vec<ShrewTensor> = params.iter().map(|t| t.inner.clone()).collect();
        let opt = shrew_optim::AdamW::new(ps, lr, weight_decay)
            .beta1(beta1)
            .beta2(beta2);
        Ok(PyAdamW { inner: opt })
    }

    fn step(&mut self, grads: &PyGradStore) -> PyResult<()> {
        use shrew_optim::Optimizer;
        self.inner.step(&grads.inner).map_err(to_py_err)?;
        Ok(())
    }
}

//  RMSProp 

#[pyclass(name = "RMSProp")]
struct PyRMSProp {
    inner: shrew_optim::RMSProp<B>,
}

#[pymethods]
impl PyRMSProp {
    #[new]
    #[pyo3(signature = (params, lr=1e-3, alpha=0.99, eps=1e-8, momentum=0.0, weight_decay=0.0))]
    fn new(
        params: Vec<PyRef<PyTensor>>,
        lr: f64,
        alpha: f64,
        eps: f64,
        momentum: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        let ps: Vec<ShrewTensor> = params.iter().map(|t| t.inner.clone()).collect();
        let opt = shrew_optim::RMSProp::new(ps, lr)
            .alpha(alpha)
            .epsilon(eps)
            .momentum(momentum)
            .weight_decay(weight_decay);
        Ok(PyRMSProp { inner: opt })
    }

    fn step(&mut self, grads: &PyGradStore) -> PyResult<()> {
        use shrew_optim::Optimizer;
        self.inner.step(&grads.inner).map_err(to_py_err)?;
        Ok(())
    }
}

//  RAdam 

#[pyclass(name = "RAdam")]
struct PyRAdam {
    inner: shrew_optim::RAdam<B>,
}

#[pymethods]
impl PyRAdam {
    #[new]
    #[pyo3(signature = (params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0))]
    fn new(
        params: Vec<PyRef<PyTensor>>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        let ps: Vec<ShrewTensor> = params.iter().map(|t| t.inner.clone()).collect();
        let opt = shrew_optim::RAdam::new(ps, lr)
            .beta1(beta1)
            .beta2(beta2)
            .epsilon(eps)
            .weight_decay(weight_decay);
        Ok(PyRAdam { inner: opt })
    }

    fn step(&mut self, grads: &PyGradStore) -> PyResult<()> {
        use shrew_optim::Optimizer;
        self.inner.step(&grads.inner).map_err(to_py_err)?;
        Ok(())
    }
}

// LR Schedulers

#[pyclass(name = "StepLR")]
struct PyStepLR {
    inner: shrew_optim::StepLR,
}

#[pymethods]
impl PyStepLR {
    #[new]
    fn new(initial_lr: f64, step_size: u64, gamma: f64) -> Self {
        PyStepLR {
            inner: shrew_optim::StepLR::new(initial_lr, step_size, gamma),
        }
    }

    /// Advance one step and return the new learning rate.
    fn step(&mut self) -> f64 {
        use shrew_optim::LrScheduler;
        self.inner.step()
    }
}

#[pyclass(name = "ExponentialLR")]
struct PyExponentialLR {
    inner: shrew_optim::ExponentialLR,
}

#[pymethods]
impl PyExponentialLR {
    #[new]
    fn new(initial_lr: f64, gamma: f64) -> Self {
        PyExponentialLR {
            inner: shrew_optim::ExponentialLR::new(initial_lr, gamma),
        }
    }
    fn step(&mut self) -> f64 {
        use shrew_optim::LrScheduler;
        self.inner.step()
    }
}

#[pyclass(name = "LinearLR")]
struct PyLinearLR {
    inner: shrew_optim::LinearLR,
}

#[pymethods]
impl PyLinearLR {
    #[new]
    fn new(initial_lr: f64, start_factor: f64, end_factor: f64, total_steps: u64) -> Self {
        PyLinearLR {
            inner: shrew_optim::LinearLR::new(initial_lr, start_factor, end_factor, total_steps),
        }
    }
    fn step(&mut self) -> f64 {
        use shrew_optim::LrScheduler;
        self.inner.step()
    }
}

#[pyclass(name = "CosineAnnealingLR")]
struct PyCosineAnnealingLR {
    inner: shrew_optim::CosineAnnealingLR,
}

#[pymethods]
impl PyCosineAnnealingLR {
    #[new]
    #[pyo3(signature = (initial_lr, total_steps, min_lr=0.0))]
    fn new(initial_lr: f64, total_steps: u64, min_lr: f64) -> Self {
        PyCosineAnnealingLR {
            inner: shrew_optim::CosineAnnealingLR::new(initial_lr, total_steps, min_lr),
        }
    }
    fn step(&mut self) -> f64 {
        use shrew_optim::LrScheduler;
        self.inner.step()
    }
}

#[pyclass(name = "CosineWarmupLR")]
struct PyCosineWarmupLR {
    inner: shrew_optim::CosineWarmupLR,
}

#[pymethods]
impl PyCosineWarmupLR {
    #[new]
    #[pyo3(signature = (initial_lr, warmup_steps, total_steps, min_lr=0.0))]
    fn new(initial_lr: f64, warmup_steps: u64, total_steps: u64, min_lr: f64) -> Self {
        PyCosineWarmupLR {
            inner: shrew_optim::CosineWarmupLR::new(initial_lr, warmup_steps, total_steps, min_lr),
        }
    }
    fn step(&mut self) -> f64 {
        use shrew_optim::LrScheduler;
        self.inner.step()
    }
}

#[pyclass(name = "ReduceLROnPlateau")]
struct PyReduceLROnPlateau {
    inner: shrew_optim::ReduceLROnPlateau,
}

#[pymethods]
impl PyReduceLROnPlateau {
    #[new]
    #[pyo3(signature = (initial_lr, factor=0.1, patience=10, min_lr=0.0, threshold=1e-4))]
    fn new(initial_lr: f64, factor: f64, patience: u64, min_lr: f64, threshold: f64) -> Self {
        PyReduceLROnPlateau {
            inner: shrew_optim::ReduceLROnPlateau::new(initial_lr)
                .factor(factor)
                .patience(patience)
                .min_lr(min_lr)
                .threshold(threshold),
        }
    }

    /// Feed a metric value and return the updated learning rate.
    fn step_metric(&mut self, metric: f64) -> f64 {
        self.inner.step_metric(metric)
    }
}

// Loss Functions (module-level)

#[pyfunction]
fn mse_loss(pred: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: shrew_nn::loss::mse_loss::<B>(&pred.inner, &target.inner).map_err(to_py_err)?,
    })
}

#[pyfunction]
fn cross_entropy_loss(logits: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: shrew_nn::loss::cross_entropy_loss::<B>(&logits.inner, &target.inner)
            .map_err(to_py_err)?,
    })
}

#[pyfunction]
fn l1_loss(pred: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: shrew_nn::loss::l1_loss::<B>(&pred.inner, &target.inner).map_err(to_py_err)?,
    })
}

#[pyfunction]
fn smooth_l1_loss(pred: &PyTensor, target: &PyTensor, beta: f64) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: shrew_nn::loss::smooth_l1_loss::<B>(&pred.inner, &target.inner, beta)
            .map_err(to_py_err)?,
    })
}

#[pyfunction]
fn bce_loss(pred: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: shrew_nn::loss::bce_loss::<B>(&pred.inner, &target.inner).map_err(to_py_err)?,
    })
}

#[pyfunction]
fn bce_with_logits_loss(logits: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: shrew_nn::loss::bce_with_logits_loss::<B>(&logits.inner, &target.inner)
            .map_err(to_py_err)?,
    })
}

#[pyfunction]
fn nll_loss(log_probs: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: shrew_nn::loss::nll_loss::<B>(&log_probs.inner, &targets.inner)
            .map_err(to_py_err)?,
    })
}

// Gradient Utilities (module-level)

/// Clip gradients by global norm, returns (clipped_grads, total_norm).
#[pyfunction]
fn clip_grad_norm(
    grads: &PyGradStore,
    params: Vec<PyRef<PyTensor>>,
    max_norm: f64,
) -> PyResult<(PyGradStore, f64)> {
    let ps: Vec<ShrewTensor> = params.iter().map(|t| t.inner.clone()).collect();
    let (new_grads, total) =
        shrew_optim::clip_grad_norm::<B>(&grads.inner, &ps, max_norm).map_err(to_py_err)?;
    Ok((PyGradStore { inner: new_grads }, total))
}

/// Clip gradients by value.
#[pyfunction]
fn clip_grad_value(
    grads: &PyGradStore,
    params: Vec<PyRef<PyTensor>>,
    max_value: f64,
) -> PyResult<PyGradStore> {
    let ps: Vec<ShrewTensor> = params.iter().map(|t| t.inner.clone()).collect();
    let new_grads =
        shrew_optim::clip_grad_value::<B>(&grads.inner, &ps, max_value).map_err(to_py_err)?;
    Ok(PyGradStore { inner: new_grads })
}

/// Compute the global gradient norm.
#[pyfunction]
fn grad_norm(grads: &PyGradStore, params: Vec<PyRef<PyTensor>>) -> PyResult<f64> {
    let ps: Vec<ShrewTensor> = params.iter().map(|t| t.inner.clone()).collect();
    shrew_optim::grad_norm::<B>(&grads.inner, &ps).map_err(to_py_err)
}

// Data Loading

//  MnistDataset 

#[pyclass(name = "MnistDataset")]
struct PyMnistDataset {
    inner: shrew_data::MnistDataset,
}

#[pymethods]
impl PyMnistDataset {
    /// Load MNIST from a directory containing IDX files.
    #[staticmethod]
    #[pyo3(signature = (dir, split="train"))]
    fn load(dir: &str, split: &str) -> PyResult<Self> {
        let s = match split {
            "train" => shrew_data::mnist::MnistSplit::Train,
            "test" => shrew_data::mnist::MnistSplit::Test,
            _ => return Err(PyValueError::new_err("split must be 'train' or 'test'")),
        };
        let ds = shrew_data::MnistDataset::load(dir, s)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(PyMnistDataset { inner: ds })
    }

    /// Create a synthetic MNIST dataset (for testing).
    #[staticmethod]
    #[pyo3(signature = (n, split="train"))]
    fn synthetic(n: usize, split: &str) -> PyResult<Self> {
        let s = match split {
            "train" => shrew_data::mnist::MnistSplit::Train,
            "test" => shrew_data::mnist::MnistSplit::Test,
            _ => return Err(PyValueError::new_err("split must be 'train' or 'test'")),
        };
        Ok(PyMnistDataset {
            inner: shrew_data::MnistDataset::synthetic(n, s),
        })
    }

    fn __len__(&self) -> usize {
        use shrew_data::Dataset;
        self.inner.len()
    }

    /// Get the i-th sample as (features_list, target_list).
    fn get(&self, index: usize) -> (Vec<f64>, Vec<f64>) {
        use shrew_data::Dataset;
        let sample = self.inner.get(index);
        (sample.features, sample.target)
    }

    // Note: take() consumes — not exposed directly.
    // If needed, load a fresh dataset and take.
}

//  CsvDataset 

#[pyclass(name = "CsvDataset")]
struct PyCsvDataset {
    inner: shrew_data::CsvDataset,
}

#[pymethods]
impl PyCsvDataset {
    /// Load a CSV dataset from a file.
    #[staticmethod]
    #[pyo3(signature = (path, has_header=true, feature_cols=None, target_cols=None, delimiter=","))]
    fn load(
        path: &str,
        has_header: bool,
        feature_cols: Option<Vec<usize>>,
        target_cols: Option<Vec<usize>>,
        delimiter: &str,
    ) -> PyResult<Self> {
        let mut config = shrew_data::CsvConfig::default().has_header(has_header);
        if let Some(fc) = feature_cols {
            config = config.feature_cols(fc);
        }
        if let Some(tc) = target_cols {
            config = config.target_cols(tc);
        }
        if let Some(d) = delimiter.as_bytes().first() {
            config = config.delimiter(*d);
        }
        let ds = shrew_data::CsvDataset::load(path, config).map_err(PyRuntimeError::new_err)?;
        Ok(PyCsvDataset { inner: ds })
    }

    fn __len__(&self) -> usize {
        use shrew_data::Dataset;
        self.inner.len()
    }

    fn get(&self, index: usize) -> (Vec<f64>, Vec<f64>) {
        use shrew_data::Dataset;
        let sample = self.inner.get(index);
        (sample.features, sample.target)
    }
}

// Safetensors I/O

#[pyfunction]
fn save_safetensors(path: &str, names: Vec<String>, tensors: Vec<PyRef<PyTensor>>) -> PyResult<()> {
    if names.len() != tensors.len() {
        return Err(PyValueError::new_err(
            "names and tensors must have same length",
        ));
    }
    let pairs: Vec<(String, ShrewTensor)> = names
        .into_iter()
        .zip(tensors.iter().map(|t| t.inner.clone()))
        .collect();
    shrew::safetensors::save::<B>(path, &pairs).map_err(to_py_err)
}

#[pyfunction]
fn load_safetensors(path: &str) -> PyResult<Vec<(String, PyTensor)>> {
    let loaded = shrew::safetensors::load::<B>(path, &CpuDevice).map_err(to_py_err)?;
    Ok(loaded
        .into_iter()
        .map(|(name, t)| (name, PyTensor { inner: t }))
        .collect())
}

// Metrics

/// Classification accuracy from predicted & true class indices.
#[pyfunction]
#[pyo3(name = "accuracy")]
fn py_accuracy(predictions: Vec<usize>, targets: Vec<usize>) -> f64 {
    shrew_nn::metrics::accuracy(&predictions, &targets)
}

/// Precision with averaging strategy ("macro", "micro", "weighted").
#[pyfunction]
#[pyo3(name = "precision", signature = (predictions, targets, n_classes, average="macro"))]
fn py_precision(
    predictions: Vec<usize>,
    targets: Vec<usize>,
    n_classes: usize,
    average: &str,
) -> PyResult<f64> {
    let avg = parse_average(average)?;
    Ok(shrew_nn::metrics::precision(
        &predictions,
        &targets,
        n_classes,
        avg,
    ))
}

/// Recall with averaging strategy.
#[pyfunction]
#[pyo3(name = "recall", signature = (predictions, targets, n_classes, average="macro"))]
fn py_recall(
    predictions: Vec<usize>,
    targets: Vec<usize>,
    n_classes: usize,
    average: &str,
) -> PyResult<f64> {
    let avg = parse_average(average)?;
    Ok(shrew_nn::metrics::recall(
        &predictions,
        &targets,
        n_classes,
        avg,
    ))
}

/// F1 score — harmonic mean of precision and recall.
#[pyfunction]
#[pyo3(name = "f1_score", signature = (predictions, targets, n_classes, average="macro"))]
fn py_f1_score(
    predictions: Vec<usize>,
    targets: Vec<usize>,
    n_classes: usize,
    average: &str,
) -> PyResult<f64> {
    let avg = parse_average(average)?;
    Ok(shrew_nn::metrics::f1_score(
        &predictions,
        &targets,
        n_classes,
        avg,
    ))
}

/// Confusion matrix NxN as list of lists.
#[pyfunction]
#[pyo3(name = "confusion_matrix")]
fn py_confusion_matrix(
    predictions: Vec<usize>,
    targets: Vec<usize>,
    n_classes: usize,
) -> Vec<Vec<u64>> {
    let cm =
        shrew_nn::metrics::ConfusionMatrix::from_predictions(&predictions, &targets, n_classes);
    cm.matrix
}

/// Per-class classification report: list of (class, precision, recall, f1, support) tuples.
#[pyfunction]
#[pyo3(name = "classification_report")]
fn py_classification_report(
    predictions: Vec<usize>,
    targets: Vec<usize>,
    n_classes: usize,
) -> Vec<(usize, f64, f64, f64, u64)> {
    let report = shrew_nn::metrics::classification_report(&predictions, &targets, n_classes);
    report
        .into_iter()
        .map(|c| (c.class, c.precision, c.recall, c.f1, c.support))
        .collect()
}

/// Top-K accuracy from flat score array and targets.
#[pyfunction]
#[pyo3(name = "top_k_accuracy")]
fn py_top_k_accuracy(scores: Vec<f64>, targets: Vec<usize>, n_classes: usize, k: usize) -> f64 {
    shrew_nn::metrics::top_k_accuracy(&scores, &targets, n_classes, k)
}

/// R² (coefficient of determination).
#[pyfunction]
#[pyo3(name = "r2_score")]
fn py_r2_score(predictions: Vec<f64>, targets: Vec<f64>) -> f64 {
    shrew_nn::metrics::r2_score(&predictions, &targets)
}

/// Mean Absolute Error.
#[pyfunction]
#[pyo3(name = "mae")]
fn py_mae(predictions: Vec<f64>, targets: Vec<f64>) -> f64 {
    shrew_nn::metrics::mae(&predictions, &targets)
}

/// Root Mean Squared Error.
#[pyfunction]
#[pyo3(name = "rmse")]
fn py_rmse(predictions: Vec<f64>, targets: Vec<f64>) -> f64 {
    shrew_nn::metrics::rmse(&predictions, &targets)
}

/// Mean Absolute Percentage Error.
#[pyfunction]
#[pyo3(name = "mape")]
fn py_mape(predictions: Vec<f64>, targets: Vec<f64>) -> f64 {
    shrew_nn::metrics::mape(&predictions, &targets)
}

/// Perplexity from cross-entropy loss.
#[pyfunction]
#[pyo3(name = "perplexity")]
fn py_perplexity(cross_entropy_loss: f64) -> f64 {
    shrew_nn::metrics::perplexity(cross_entropy_loss)
}

/// Tensor accuracy: logits [batch, classes] vs targets [batch] or [batch, classes].
#[pyfunction]
#[pyo3(name = "tensor_accuracy")]
fn py_tensor_accuracy(logits: &PyTensor, targets: &PyTensor) -> PyResult<f64> {
    shrew_nn::metrics::tensor_accuracy::<B>(&logits.inner, &targets.inner).map_err(to_py_err)
}

/// Argmax classes from logits tensor [batch, classes] -> Vec<usize>.
#[pyfunction]
#[pyo3(name = "argmax_classes")]
fn py_argmax_classes(logits: &PyTensor) -> PyResult<Vec<usize>> {
    shrew_nn::metrics::argmax_classes::<B>(&logits.inner).map_err(to_py_err)
}

fn parse_average(s: &str) -> PyResult<shrew_nn::metrics::Average> {
    match s {
        "macro" => Ok(shrew_nn::metrics::Average::Macro),
        "micro" => Ok(shrew_nn::metrics::Average::Micro),
        "weighted" => Ok(shrew_nn::metrics::Average::Weighted),
        _ => Err(PyValueError::new_err(format!(
            "Unknown average: '{}'. Use 'macro', 'micro', or 'weighted'.",
            s
        ))),
    }
}

// Module Registration

#[pymodule]
fn shrew_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core
    m.add_class::<PyTensor>()?;
    m.add_class::<PyGradStore>()?;
    m.add_class::<PyExecutor>()?;

    // NN Layers
    m.add_class::<PyLinear>()?;
    m.add_class::<PyConv2d>()?;
    m.add_class::<PyConv1d>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyLayerNorm>()?;
    m.add_class::<PyGroupNorm>()?;
    m.add_class::<PyRMSNorm>()?;
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyDropout>()?;
    m.add_class::<PyFlatten>()?;
    m.add_class::<PyMaxPool2d>()?;
    m.add_class::<PyAvgPool2d>()?;
    m.add_class::<PyAdaptiveAvgPool2d>()?;
    m.add_class::<PyMultiHeadAttention>()?;
    m.add_class::<PyTransformerBlock>()?;
    m.add_class::<PyRNNCell>()?;
    m.add_class::<PyRNN>()?;
    m.add_class::<PyLSTMCell>()?;
    m.add_class::<PyLSTM>()?;
    m.add_class::<PyGRUCell>()?;
    m.add_class::<PyGRU>()?;

    // Activations
    m.add_class::<PyReLU>()?;
    m.add_class::<PyGeLU>()?;
    m.add_class::<PySiLU>()?;
    m.add_class::<PySigmoidAct>()?;
    m.add_class::<PyTanhAct>()?;
    m.add_class::<PyLeakyReLU>()?;
    m.add_class::<PyELU>()?;
    m.add_class::<PyMish>()?;

    // Optimizers
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PyRMSProp>()?;
    m.add_class::<PyRAdam>()?;

    // LR Schedulers
    m.add_class::<PyStepLR>()?;
    m.add_class::<PyExponentialLR>()?;
    m.add_class::<PyLinearLR>()?;
    m.add_class::<PyCosineAnnealingLR>()?;
    m.add_class::<PyCosineWarmupLR>()?;
    m.add_class::<PyReduceLROnPlateau>()?;

    // Losses
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(bce_loss, m)?)?;
    m.add_function(wrap_pyfunction!(bce_with_logits_loss, m)?)?;
    m.add_function(wrap_pyfunction!(nll_loss, m)?)?;

    // Gradient utilities
    m.add_function(wrap_pyfunction!(clip_grad_norm, m)?)?;
    m.add_function(wrap_pyfunction!(clip_grad_value, m)?)?;
    m.add_function(wrap_pyfunction!(grad_norm, m)?)?;

    // Data
    m.add_class::<PyMnistDataset>()?;
    m.add_class::<PyCsvDataset>()?;

    // I/O
    m.add_function(wrap_pyfunction!(save_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(load_safetensors, m)?)?;

    // Metrics
    m.add_function(wrap_pyfunction!(py_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(py_precision, m)?)?;
    m.add_function(wrap_pyfunction!(py_recall, m)?)?;
    m.add_function(wrap_pyfunction!(py_f1_score, m)?)?;
    m.add_function(wrap_pyfunction!(py_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_classification_report, m)?)?;
    m.add_function(wrap_pyfunction!(py_top_k_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(py_r2_score, m)?)?;
    m.add_function(wrap_pyfunction!(py_mae, m)?)?;
    m.add_function(wrap_pyfunction!(py_rmse, m)?)?;
    m.add_function(wrap_pyfunction!(py_mape, m)?)?;
    m.add_function(wrap_pyfunction!(py_perplexity, m)?)?;
    m.add_function(wrap_pyfunction!(py_tensor_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(py_argmax_classes, m)?)?;

    Ok(())
}
