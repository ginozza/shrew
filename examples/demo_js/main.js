import { train } from './shrew.js';

const res = train('examples/model.sw');
console.log(`Epochs: ${res.epochs}, Loss: ${res.loss.toFixed(6)}`);
