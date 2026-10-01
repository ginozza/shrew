import shrew_python as shrew

res = shrew.train("examples/model.sw")
print(f"Epochs: {int(res['epochs'])}, Loss: {res['final_loss']:.6f}")
