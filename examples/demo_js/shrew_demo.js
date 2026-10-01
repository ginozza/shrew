import koffi from 'koffi';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dllPath = path.resolve(__dirname, '../../target/release/shrew.dll');

console.log("=================================================");
console.log("  Shrew JavaScript / Node.js FFI Demo");
console.log("=================================================");
console.log("Loading Shrew native library:", dllPath);

const lib = koffi.load(dllPath);

// Define C types and functions
const shrew_tensor_t = koffi.opaque('shrew_tensor_t');

const shrew_version = lib.func('const char* shrew_version()');
const shrew_cuda_is_available = lib.func('int shrew_cuda_is_available()');

const shrew_tensor_from_data = lib.func('shrew_tensor_t* shrew_tensor_from_data(double* data, size_t* shape, size_t ndim, int dtype, int device)');
const shrew_tensor_matmul = lib.func('shrew_tensor_t* shrew_tensor_matmul(shrew_tensor_t* a, shrew_tensor_t* b)');
const shrew_tensor_to_data = lib.func('int shrew_tensor_to_data(shrew_tensor_t* tensor, _Out_ double* out_data, size_t max_len)');
const shrew_tensor_print = lib.func('void shrew_tensor_print(shrew_tensor_t* tensor)');
const shrew_tensor_free = lib.func('void shrew_tensor_free(shrew_tensor_t* tensor)');

console.log("Shrew Version:", shrew_version());
console.log("CUDA Available in C API:", shrew_cuda_is_available() === 1);

// Create Matrix A: 2x2 [[1, 2], [3, 4]]
const dataA = [1.0, 2.0, 3.0, 4.0];
const shapeA = [2, 2];
const tensorA = shrew_tensor_from_data(dataA, shapeA, 2, 1, 0); // F64, CPU

// Create Matrix B: 2x2 [[5, 6], [7, 8]]
const dataB = [5.0, 6.0, 7.0, 8.0];
const shapeB = [2, 2];
const tensorB = shrew_tensor_from_data(dataB, shapeB, 2, 1, 0); // F64, CPU

// Multiply: C = A @ B
const tensorC = shrew_tensor_matmul(tensorA, tensorB);

console.log("\nResult of Matrix Multiplication C = A @ B computed by Shrew:");
shrew_tensor_print(tensorC);

const outData = new Array(4);
const count = shrew_tensor_to_data(tensorC, outData, 4);
console.log(`Extracted ${count} elements in JavaScript: [ ${outData.join(', ')} ]`);

if (outData[0] === 19 && outData[1] === 22 && outData[2] === 43 && outData[3] === 50) {
  console.log("\n>>> SUCCESS: Shrew successfully executed directly from JavaScript / Node.js! <<<\n");
}

shrew_tensor_free(tensorA);
shrew_tensor_free(tensorB);
shrew_tensor_free(tensorC);
