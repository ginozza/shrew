import koffi from 'koffi';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Locate shrew.dll automatically across standard project locations
const searchPaths = [
  path.resolve(__dirname, '../../target/release/shrew.dll'),
  path.resolve(__dirname, '../target/release/shrew.dll'),
  path.resolve(process.cwd(), 'target/release/shrew.dll'),
  path.resolve(process.cwd(), 'shrew.dll'),
];

let dllPath = searchPaths.find(p => fs.existsSync(p)) || 'shrew.dll';
const lib = koffi.load(dllPath);

// C API function bindings
const _version = lib.func('const char* shrew_version()');
const _cuda_is_available = lib.func('int shrew_cuda_is_available()');
const _train_file = lib.func('int shrew_train_file(const char* sw_path, int dtype, _Out_ double* out_loss)');

const _last_error = lib.func('const char* shrew_last_error()');

export function version() {
  return _version();
}

export function isCudaAvailable() {
  return _cuda_is_available() === 1;
}

export function train(swPath, dtype = 1) {
  let resolved = swPath;
  if (!fs.existsSync(resolved)) {
    const candidates = [
      path.resolve(process.cwd(), swPath),
      path.resolve(process.cwd(), '../..', swPath),
      path.resolve(process.cwd(), '..', swPath),
    ];
    resolved = candidates.find(p => fs.existsSync(p)) || swPath;
  }
  const outLoss = [0.0];
  const epochs = _train_file(path.resolve(resolved), dtype, outLoss);
  if (epochs < 0) {
    const err = _last_error();
    throw new Error(`Failed to train model at ${swPath}: ${err}`);
  }
  return { epochs, loss: outLoss[0] };
}

export default {
  version,
  isCudaAvailable,
  train,
};
