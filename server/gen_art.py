import math
import os

import torch

try:
    from .model_architectures import CausalAblationMLP, DeadNeuronMLP, GrokkingTransformer
except ImportError:
    from model_architectures import CausalAblationMLP, DeadNeuronMLP, GrokkingTransformer


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    task1 = DeadNeuronMLP()
    torch.save(task1, os.path.join(artifacts_dir, "task1.pt"))
    print("Task 1 saved.")

    task2 = CausalAblationMLP(hidden_dim=10, mult_idx=2, add_idx=3)
    torch.save(task2, os.path.join(artifacts_dir, "task2.pt"))
    print("Task 2 saved.")

    task3 = GrokkingTransformer(p=97, d_model=128)
    with torch.no_grad():
        p = task3.W_E.num_embeddings
        d_model = task3.W_E.embedding_dim
        embedding = torch.randn(p, d_model) * 0.01
        positions = torch.arange(p, dtype=torch.float32)

        for offset, freq in enumerate(task3.secret_freqs):
            angle = 2.0 * math.pi * float(freq) * positions / float(p)
            embedding[:, offset * 2] += torch.sin(angle)
            embedding[:, offset * 2 + 1] += torch.cos(angle)

        task3.W_E.weight.copy_(embedding)

    torch.save(task3, os.path.join(artifacts_dir, "task3.pt"))
    print("Task 3 saved.")


if __name__ == "__main__":
    main()
