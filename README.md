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
import asyncio
import os

from openai import OpenAI
from mech_interp import MechInterpAction, MechInterpEnv

async def main() -> None:
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        api_key=os.getenv("API_KEY", os.getenv("HF_TOKEN")),
    )
    model_name = "deepseek-ai/DeepSeek-V3-0324"

    async with MechInterpEnv(
        base_url="https://kashyapsinh-pytorch-sandbox-mech-interp.hf.space"
    ) as env:
        result = await env.reset()
        print(result.observation.stdout_or_error)

        action = MechInterpAction(solution_target=[2, 5, 8])
        result = await env.step(action)
        print(f"Reward: {result.reward}")

asyncio.run(main())
```

---

## Environment Variables

Configure the inference script with these variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | Validator-injected LLM proxy key | Required for hackathon submission |
| `HF_TOKEN` | Optional local HuggingFace token fallback | Unset |
| `API_BASE_URL` | LLM API endpoint or validator proxy URL | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model to use for inference | `deepseek-ai/DeepSeek-V3-0324` |
| `ENV_URL` | Environment server URL | Space URL |
| `LOCAL_IMAGE_NAME` | Optional local Docker image for `from_docker_image()` runs | Unset |

For hackathon submissions, do not override the injected `API_BASE_URL` / `API_KEY` pair with your own provider settings.

---

## Action Formats

```python
# Execute Python code:
{"python_code": "print(model.layer.weight)"}

# Submit solution:
{"solution_target": [2, 5, 8]}
```

---

## Action And Observation Spaces

```python
class MechInterpAction(Action):
    python_code: Optional[str]
    solution_target: Optional[List[Any]]

class MechInterpObservation(Observation):
    stdout_or_error: str
    task_level: int
    done: bool
    reward: float
```

- `python_code` executes sandboxed inspection code against the current task model.
- `solution_target` submits the final integer answer list for grading.
- `stdout_or_error` returns captured stdout/stderr or grader feedback.
- `task_level` tracks progression through the 3-task curriculum.

---

## Scoring

- **Task 1**: Exact match gives full credit; partial credit is available for partially correct sets
- **Task 2**: Exact identification gives full credit; including the correct neuron in a longer list earns partial credit
- **Task 3**: Score is based on mean-squared error against the planted frequencies
- **Validator-facing task scores**: Each task's grader reports a score strictly inside `(0, 1)` for compatibility with Phase 2 validation
- **Final Score**: Average of all 3 tasks

---

## Baseline Reproduction

Run the provided baseline agent with:

```bash
python inference.py
```

The script emits the required `[START]`, `[STEP]`, and `[END]` lines and reports a final score in `[0, 1]`. Record the exact score from your target deployment before final submission so the README and submission notes reflect the same environment build.

---

## Determinism

The benchmark uses deterministic model artifacts. `OPENENV_SEED` is surfaced in metadata for reproducibility, but the task answers are derived from the loaded models rather than randomized per reset.

---

## Health Checks

- `GET /health` returns service health
- `GET /info` returns a non-secret summary of the three tasks and their graders
- `GET /tasks` returns the explicit task/grader manifest used by validators and clients

---

## Security

- Code execution runs in a separate constrained subprocess with a 30-second timeout
- Max 30 steps per episode
- Restricted builtins are exposed to agent code instead of the full Python builtins

---

Built with [OpenEnv](https://github.com/meta-pytorch/openenv) by Meta.
