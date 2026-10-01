import shrew from 'shrew';

const res = shrew.train('examples/model.sw');
console.log(`Epochs: ${res.epochs}, Loss: ${res.loss.toFixed(6)}`);
