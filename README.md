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

# PyTorchSandbox

Minimal OpenEnv benchmark for mechanistic interpretability.

## Tasks

1. Dead Neuron Detection
2. Causal Ablation
3. Fourier Analysis

## Actions

- `{"python_code": "..."}`
- `{"solution_target": [..]}`

## Run

```bash
python inference.py
```

## Endpoints

- `/health`
- `/reset`
- `/step`
- `/tasks`
