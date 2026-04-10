---
title: PyTorchSandbox Mechanistic Interpretability
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "0.2.2"
app_port: 8000
app_file: server.app:app
pinned: false
license: bsd-3-clause
---

# PyTorchSandbox — Mechanistic Interpretability Benchmark

An OpenEnv environment for mechanistic interpretability research. Agents inspect live PyTorch models, execute diagnostic code, and submit circuit-level answers across a four-task curriculum.

## Tasks

| # | Task | Difficulty | Objective | Reset Payload |
|---|------|-----------|-----------|---------------|
| 1 | Dead Neuron Detection | Easy | Find zero-weight input indices in Linear(10,1) | `{"task_id": "task1"}` |
| 2 | Causal Ablation | Medium | Identify the multiplication neuron in y = x1·x2 + x3 | `{"task_id": "task2"}` |
| 3 | Fourier Analysis | Hard | Recover planted frequencies from transformer embedding spectrum | `{"task_id": "task3"}` |
| 4 | Additive Bypass Attribution | Medium | Identify the x3 bypass neuron in the hidden layer | `{"task_id": "task4"}` |

## Actions

```python
# Inspect model internals
{"python_code": "print(model.layer.weight)"}

# Submit final answer
{"solution_target": [2, 5, 8]}
```

## Observation

```python
MechInterpObservation:
  stdout_or_error: str   # Captured output or error from code execution
  task_level: int        # Current task (1–4)
  done: bool             # Episode terminated
  reward: float         # Score in (0.01, 0.99)
  metadata: dict        # task_id, grader info, seed, step
```

## Setup

```bash
# Install dependencies
python3.10+ -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Generate model artifacts (if needed)
python -c "
import sys; sys.path.insert(0, 'server')
import os; os.makedirs('artifacts', exist_ok=True)
from gen_art import main; main()
"

# Run locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference
HF_TOKEN=your_token \
  ENV_URL=http://localhost:8000 \
  MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  python inference.py
```

## Docker

```bash
docker build -t mech_interp .
docker run -p 8000:8000 mech_interp
```

## Graders

| Task | Module | Function |
|------|--------|----------|
| 1 | `tasks.task1.grader` | `grade` |
| 2 | `tasks.task2.grader` | `grade` |
| 3 | `tasks.task3.grader` | `grade` |
| 4 | `tasks.task4.grader` | `grade` |
| Router | `tasks.graders` | `grade_task` |

All graders return scores strictly within `(0.01, 0.99)`.

## Submission Checklist

- [ ] `inference.py` in project root
- [ ] OpenAI client used for all LLM calls
- [ ] `HF_TOKEN` environment variable required at runtime
- [ ] Strict `[START]` / `[STEP]` / `[END]` stdout format
- [ ] 4 tasks with programmatic graders
- [ ] Scores in `[0.0, 1.0]` range
- [ ] `docker build` succeeds
- [ ] HF Space URL submitted before deadline

## Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `HF_TOKEN` | Yes | — |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` |
| `ENV_URL` | No* | — |
| `LOCAL_IMAGE_NAME` | No* | — |

*Either `ENV_URL` (for HF Space) or `LOCAL_IMAGE_NAME` (for Docker image) must be set.

## Validation

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space .
```
