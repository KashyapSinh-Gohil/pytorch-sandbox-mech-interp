---
title: PyTorchSandbox Mechanistic Interpretability
emoji: 🔬
colorFrom: teal
colorTo: blue
sdk: docker
sdk_version: "0.2.2"
app_port: 8000
app_file: server.app:app
pinned: false
license: bsd-3-clause
---

# PyTorchSandbox

PyTorchSandbox is an OpenEnv benchmark for mechanistic interpretability workflows. An agent inspects PyTorch models, runs targeted probes, and submits graded circuit-level answers across a four-step curriculum.

## Overview

The environment is designed around the kind of debugging and attribution work ML researchers actually do when they inspect learned representations and hidden circuits. Every task exposes a deterministic grader, a direct reset payload, and scores strictly inside `(0, 1)`.

## Tasks

1. Dead Neuron Detection
2. Causal Ablation
3. Fourier Analysis
4. Additive Bypass Attribution

## Task Map

| Task | Goal | Difficulty | Reset Payload |
| --- | --- | --- | --- |
| 1 | Find zero-weight input features in a sparse linear probe | Easy | `{"task_id":"task1"}` |
| 2 | Identify the multiplication neuron in a hidden circuit | Medium | `{"task_id":"task2"}` |
| 3 | Recover planted frequencies from a transformer embedding spectrum | Hard | `{"task_id":"task3"}` |
| 4 | Identify the hidden neuron carrying the additive `x3` bypass | Medium | `{"task_id":"task4"}` |

## Graders

- Task 1: `tasks.task1.grader:grade`
- Task 2: `tasks.task2.grader:grade`
- Task 3: `tasks.task3.grader:grade`
- Task 4: `tasks.task4.grader:grade`
- Top-level router: `tasks.graders:grade_task`
- Static registry: `tasks.json`
- Python registry: `tasks.TASKS`

## Actions

- `{"python_code": "..."}`
- `{"solution_target": [..]}`

## Observation

- `stdout_or_error`
- `task_level`
- `reward`
- `done`
- `metadata.task_id`

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Run

```bash
python inference.py
```

The baseline agent uses the OpenAI client against the Hugging Face router via `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.

## Submission Compliance

- `inference.py` is in the project root
- all LLM calls use the OpenAI Python client
- `API_BASE_URL` and `MODEL_NAME` have defaults
- `HF_TOKEN` is required at runtime
- stdout follows the strict `[START]`, `[STEP]`, `[END]` format expected by the validator

## Validation

- `87/87` local tests passing
- `openenv validate` passes
- Live Space endpoints verified: `/health`, `/reset`, `/tasks`
- Validator-facing task rewards are clamped strictly inside `(0, 1)`

## Baseline

- Latest local smoke run started correctly and produced valid `[START]`, `[STEP]`, and `[END]` logs
- That run stopped early because the local Hugging Face Inference Provider credits were exhausted
- During evaluation, `HF_TOKEN` is required and `API_BASE_URL` / `MODEL_NAME` can use their defaults or injected overrides

## Endpoints

- `/health`
- `/reset`
- `/step`
- `/tasks`
- `/info`
