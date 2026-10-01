import koffi from 'koffi';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const lib = koffi.load(path.resolve(__dirname, '../../target/release/shrew.dll'));
const _train = lib.func('int shrew_train_file(const char* sw_path, int dtype, _Out_ double* out_loss)');

export function train(swPath) {
  const loss = [0.0];
  const epochs = _train(path.resolve(swPath), 1, loss);
  return { epochs, loss: loss[0] };
}
