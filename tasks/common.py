"""Shared grading helpers for task manifest entrypoints."""

from functools import lru_cache
import importlib
from pathlib import Path
import sys
from typing import Any, Optional

import torch

MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def clamp_task_score(score: float) -> float:
    return round(min(MAX_TASK_SCORE, max(MIN_TASK_SCORE, float(score))), 4)


def _get_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def normalize_submission(action: Any) -> tuple[Optional[list[int]], Optional[str]]:
    raw_submission = _get_field(action, "solution_target")
    if not isinstance(raw_submission, list):
        return None, "Invalid submission: solution_target must be a list of integers."

    if not raw_submission:
        return None, "Invalid submission: solution_target cannot be empty."

    normalized: list[int] = []
    for item in raw_submission:
        if isinstance(item, bool):
            return None, "Invalid submission: boolean values are not valid indices."

        if isinstance(item, int):
            normalized.append(item)
            continue

        if isinstance(item, float) and item.is_integer():
            normalized.append(int(item))
            continue

        if isinstance(item, str) and item.strip().lstrip("-").isdigit():
            normalized.append(int(item.strip()))
            continue

        return None, "Invalid submission: solution_target must contain only integers."

    return normalized, None


def _import_model_architectures():
    module = importlib.import_module("server.model_architectures")
    sys.modules.setdefault("model_architectures", module)
    return module


@lru_cache(maxsize=None)
def _load_task_model(task_id: str):
    model_architectures = _import_model_architectures()
    artifact_name = "task2.pt" if task_id == "task4" else f"{task_id}.pt"
    artifact_path = ARTIFACTS_DIR / artifact_name

    try:
        return torch.load(artifact_path, weights_only=False)
    except Exception:
        if task_id == "task1":
            return model_architectures.DeadNeuronMLP()
        if task_id in {"task2", "task4"}:
            return model_architectures.CausalAblationMLP()
        if task_id == "task3":
            return model_architectures.GrokkingTransformer()
        raise ValueError(f"Unknown task id: {task_id}")


def _infer_task1_ground_truth(model: Any) -> list[int]:
    weight = getattr(getattr(model, "layer", None), "weight", None)
    if weight is None:
        raise ValueError("Task 1 model is missing layer.weight.")
    dead_indices = torch.where(torch.all(weight.detach() == 0, dim=0))[0].tolist()
    return sorted(int(index) for index in dead_indices)


def _infer_task2_ground_truth(model: Any) -> list[int]:
    hidden_layer = getattr(model, "hidden", None)
    mult_idx = getattr(hidden_layer, "mult_idx", None)
    if mult_idx is not None:
        return [int(mult_idx)]

    if hidden_layer is None:
        raise ValueError("Task 2 model is missing the hidden layer used for ablation.")

    inputs = torch.tensor(
        [
            [2.0, 3.0, 0.5],
            [-1.0, 4.0, 1.5],
            [0.25, -2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        baseline = model(inputs)
        hidden_dim = model.hidden(inputs).shape[1]

    best_idx = 0
    best_score = -1.0
    for candidate_idx in range(hidden_dim):
        def _ablate(_module, _args, output):
            patched_output = output.clone()
            patched_output[:, candidate_idx] = 0
            return patched_output

        handle = model.hidden.register_forward_hook(_ablate)
        try:
            with torch.no_grad():
                ablated = model(inputs)
        finally:
            handle.remove()

        score = (baseline - ablated).abs().sum().item()
        if score > best_score:
            best_idx = candidate_idx
            best_score = score

    return [int(best_idx)]


def _infer_task3_ground_truth(model: Any) -> list[int]:
    secret_freqs = getattr(model, "secret_freqs", None)
    if secret_freqs is not None:
        return sorted(int(freq) for freq in secret_freqs)

    embedding = getattr(getattr(model, "W_E", None), "weight", None)
    if embedding is None:
        raise ValueError("Task 3 model is missing W_E.weight.")

    spectrum = torch.fft.rfft(embedding.detach().float(), dim=0)
    energy = spectrum.abs().pow(2).sum(dim=1)
    if energy.numel() > 0:
        energy[0] = 0

    top_k = min(5, max(0, energy.shape[0] - 1))
    if top_k == 0:
        return []

    frequency_indices = torch.topk(energy, k=top_k).indices.tolist()
    return sorted(int(index) for index in frequency_indices if int(index) > 0)


def _infer_task4_ground_truth(model: Any) -> list[int]:
    hidden_layer = getattr(model, "hidden", None)
    add_idx = getattr(hidden_layer, "add_idx", None)
    if add_idx is not None:
        return [int(add_idx)]

    if hidden_layer is None:
        raise ValueError("Task 4 model is missing the hidden layer used for attribution.")

    inputs = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -3.5],
            [0.0, 0.0, 5.25],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        baseline = model(inputs)
        hidden_dim = model.hidden(inputs).shape[1]

    best_idx = 0
    best_score = -1.0
    for candidate_idx in range(hidden_dim):
        def _ablate(_module, _args, output):
            patched_output = output.clone()
            patched_output[:, candidate_idx] = 0
            return patched_output

        handle = model.hidden.register_forward_hook(_ablate)
        try:
            with torch.no_grad():
                ablated = model(inputs)
        finally:
            handle.remove()

        score = (baseline - ablated).abs().sum().item()
        if score > best_score:
            best_idx = candidate_idx
            best_score = score

    return [int(best_idx)]


@lru_cache(maxsize=None)
def get_task_ground_truth(task_id: str) -> list[int]:
    model = _load_task_model(task_id)
    if task_id == "task1":
        return _infer_task1_ground_truth(model)
    if task_id == "task2":
        return _infer_task2_ground_truth(model)
    if task_id == "task3":
        return _infer_task3_ground_truth(model)
    if task_id == "task4":
        return _infer_task4_ground_truth(model)
    raise ValueError(f"Unknown task id: {task_id}")
