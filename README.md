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

## Overview

This environment evaluates whether an agent can inspect neural-network internals, execute PyTorch code, and submit graded answers across three tasks.

## Tasks

1. Dead Neuron Detection
2. Causal Ablation
3. Fourier Analysis

## Graders

- Task 1: `tasks.task1.grader:grade`
- Task 2: `tasks.task2.grader:grade`
- Task 3: `tasks.task3.grader:grade`
- Top-level router: `tasks.graders:grade_task`

## Actions

- `{"python_code": "..."}`
- `{"solution_target": [..]}`

## Observation

- `stdout_or_error`
- `task_level`
- `reward`
- `done`

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Run

```bash
python inference.py
```

## Validation

- `79/79` local tests passing
- `openenv validate` passes
- Live Space endpoints verified: `/health`, `/reset`, `/tasks`

## Baseline

- Latest local smoke run started correctly and reached `score=0.337`
- That run stopped early because the local Hugging Face Inference Provider credits were exhausted
- During evaluation, the script is configured to prefer validator-injected `API_KEY` / `API_BASE_URL`

## Endpoints

- `/health`
- `/reset`
- `/step`
- `/tasks`
