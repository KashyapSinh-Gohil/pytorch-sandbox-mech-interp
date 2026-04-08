---
title: PyTorchSandbox Mechanistic Interpretability
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "0.2.2"
app_port: 8000
app_file: server.app:app
pinned: false
license: bsd-3-clause
---

# 🔬 PyTorchSandbox Mechanistic Interpretability RL Environment

## Overview

PyTorchSandbox is an OpenEnv environment that evaluates whether an LLM agent can reverse-engineer neural network internals through PyTorch code execution. It presents a **3-task curriculum** of increasing difficulty.

---

## The 3-Task Curriculum

| Task | Difficulty | Domain | Objective |
|------|-----------|--------|-----------|
| **1. Dead Neuron Detection** | Easy | Weight Inspection | Find zero-weight indices in Linear(10,1) |
| **2. Causal Ablation** | Medium | Activation Patching | Identify multiplication neuron in y=x1*x2+x3 |
| **3. Fourier Analysis** | Hard | Synthetic Features | Extract 5 planted frequencies from embeddings |

---

## How to Use

### Option 1: Web Interface
Visit: **https://kashyapsinh-pytorch-sandbox-mech-interp.hf.space**

1. Click **Reset** to start a new episode
2. Enter Python code to inspect the model, OR
3. Enter Solution Target as JSON array (e.g., `[i1, i2, i3]`)
4. Click **Step** to execute

### Option 2: Python API
```python
from openai import OpenAI
from mech_interp import MechInterpAction, MechInterpEnv
import asyncio

# Configure
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)
MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"  # Or any HF model

# Connect to environment
env = MechInterpEnv(base_url="https://kashyapsinh-pytorch-sandbox-mech-interp.hf.space")

# Run episode
result = await env.reset()
print(result.observation.stdout_or_error)

# Submit solution
action = MechInterpAction(solution_target=[2, 5, 8])
result = await env.step(action)
print(f"Reward: {result.reward}")
```

---

## Environment Variables

Configure the inference script with these variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | Required |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model to use for inference | `deepseek-ai/DeepSeek-V3-0324` |
| `ENV_URL` | Environment server URL | Space URL |

---

## Action Formats

```python
# Execute Python code:
{"python_code": "print(model.layer.weight)"}

# Submit solution:
{"solution_target": [2, 5, 8]}
```

---

## Scoring

- **Task 1**: Exact match gives full credit; partial credit is available for partially correct sets
- **Task 2**: Exact identification of the multiplication neuron
- **Task 3**: Score is based on mean-squared error against the planted frequencies
- **Final Score**: Average of all 3 tasks

---

## Determinism

The benchmark uses deterministic model artifacts. `OPENENV_SEED` is surfaced in metadata for reproducibility, but the task answers are derived from the loaded models rather than randomized per reset.

---

## Health Checks

- `GET /health` returns service health
- `GET /info` returns a non-secret summary of the three tasks

---

## Security

- Code execution runs in a separate constrained subprocess with a 30-second timeout
- Max 30 steps per episode
- Restricted builtins are exposed to agent code instead of the full Python builtins

---

Built with [OpenEnv](https://github.com/meta-pytorch/openenv) by Meta.
