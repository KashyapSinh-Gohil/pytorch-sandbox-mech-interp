import torch
import math
import os
from model_architectures import DeadNeuronMLP, CausalAblationMLP, GrokkingTransformer

os.makedirs("../artifacts", exist_ok=True)

task1 = DeadNeuronMLP()
torch.save(task1, "../artifacts/task1.pt")
print("Task 1 saved.")

task2 = CausalAblationMLP(hidden_dim=10, mult_idx=2, add_idx=3)
torch.save(task2, "../artifacts/task2.pt")
print("Task 2 saved.")

task3 = GrokkingTransformer(p=97, d_model=128)
with torch.no_grad():
    p = 97
    d_model = 128
    W_E = torch.randn(p, d_model) * 0.01  
    freqs = task3.secret_freqs
    for i, f in enumerate(freqs):
        for x in range(p):
            W_E[x, i*2] += math.sin(2 * math.pi * f * x / p)
            W_E[x, i*2+1] += math.cos(2 * math.pi * f * x / p)
    task3.W_E.weight.data = W_E

torch.save(task3, "../artifacts/task3.pt")
print("Task 3 saved.")
