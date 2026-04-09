import base64
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

try:
    from openenv.core.env_server.interfaces import Environment
except Exception:
    class Environment:  # type: ignore[override]
        def __init__(self, rubric: Optional["Rubric"] = None):
            self.rubric = rubric

        def _apply_rubric(self, action: Any, observation: Any) -> Optional[float]:
            if self.rubric is None:
                return None
            return self.rubric(action, observation)

        def _reset_rubric(self) -> None:
            if self.rubric is not None and hasattr(self.rubric, "reset"):
                self.rubric.reset()

try:
    from openenv.core.env_server.types import EnvironmentMetadata
except Exception:
    @dataclass
    class EnvironmentMetadata:  # type: ignore[override]
        name: str
        description: str
        readme_content: Optional[str] = None
        version: str = "0.1.0"
        author: Optional[str] = None
        documentation_url: Optional[str] = None

try:
    from openenv.core.rubrics import Rubric
except Exception:
    class Rubric:  # type: ignore[override]
        def __init__(self) -> None:
            self.last_score = 0.5

        def forward(self, action: Any, observation: Any) -> float:
            raise NotImplementedError

        def __call__(self, action: Any, observation: Any) -> float:
            score = float(self.forward(action, observation))
            self.last_score = score
            return score

        def named_children(self):
            for name, value in self.__dict__.items():
                if isinstance(value, Rubric):
                    yield name, value

        def named_rubrics(self):
            return self.named_children()

from . import model_architectures

sys.modules.setdefault("model_architectures", model_architectures)

try:
    from ..models import MechInterpAction, MechInterpObservation, InterpState
except ImportError:
    from models import MechInterpAction, MechInterpObservation, InterpState

EXEC_TIMEOUT = 30
MAX_EPISODE_STEPS = 30
TASK3_NUM_FREQUENCIES = 5
MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

TASK_SPECS = {
    "task1": {
        "id": "task1",
        "level": 1,
        "name": "Dead Neuron Detection",
        "description": "Find all zero-weight input indices in the Linear(10,1) model.",
        "grader_name": "dead_neuron_detection_grader",
        "grader_module": "tasks.task1.grader",
        "grader_function": "grade",
    },
    "task2": {
        "id": "task2",
        "level": 2,
        "name": "Causal Ablation",
        "description": "Identify the hidden neuron responsible for the multiplicative circuit.",
        "grader_name": "causal_ablation_grader",
        "grader_module": "tasks.task2.grader",
        "grader_function": "grade",
    },
    "task3": {
        "id": "task3",
        "level": 3,
        "name": "Fourier Analysis",
        "description": "Recover the planted frequencies from the embedding spectrum.",
        "grader_name": "fourier_frequency_recovery_grader",
        "grader_module": "tasks.task3.grader",
        "grader_function": "grade",
    },
    "task4": {
        "id": "task4",
        "level": 4,
        "name": "Additive Bypass Attribution",
        "description": "Identify the hidden neuron that directly carries the additive x3 bypass.",
        "grader_name": "additive_bypass_attribution_grader",
        "grader_module": "tasks.task4.grader",
        "grader_function": "grade",
    },
}

TASK_ALIASES = {
    "task1": "task1",
    "1": "task1",
    "dead_neuron_detection": "task1",
    "dead neuron detection": "task1",
    "task2": "task2",
    "2": "task2",
    "causal_ablation": "task2",
    "causal ablation": "task2",
    "task3": "task3",
    "3": "task3",
    "fourier_analysis": "task3",
    "fourier analysis": "task3",
    "task4": "task4",
    "4": "task4",
    "additive_bypass_attribution": "task4",
    "additive bypass attribution": "task4",
    "additive_bypass": "task4",
    "additive bypass": "task4",
}

EXEC_RUNNER = """
import base64
import contextlib
import io
import math
import sys
import traceback

import torch
import torch.nn as nn

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

buffer = io.StringIO()
model_path = sys.argv[1]
encoded_code = sys.argv[2]
code = base64.b64decode(encoded_code.encode("ascii")).decode("utf-8")
model = torch.load(model_path, weights_only=False)
namespace = {
    "__builtins__": SAFE_BUILTINS,
    "model": model,
    "torch": torch,
    "nn": nn,
    "math": math,
}

with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
    try:
        compiled_code = compile(code, "<agent_code>", "exec")
        exec(compiled_code, namespace, namespace)
    except Exception:
        traceback.print_exc(file=buffer)

sys.stdout.write(buffer.getvalue())
"""


def _coerce_seed(seed: Optional[int]) -> int:
    """Normalize an optional seed value into a stable integer."""
    if seed is not None:
        return int(seed)

    raw_seed = os.getenv("OPENENV_SEED", "42")
    try:
        return int(raw_seed)
    except ValueError:
        return 42


def _build_task3_model() -> nn.Module:
    """Build a deterministic fallback model when the serialized artifact is unavailable."""
    model = model_architectures.GrokkingTransformer(p=97, d_model=128)
    p = model.W_E.num_embeddings
    d_model = model.W_E.embedding_dim
    positions = torch.arange(p, dtype=torch.float32)
    embeddings = torch.randn(p, d_model) * 0.01
    secret_freqs = list(getattr(model, "secret_freqs", [2, 17, 23, 44, 47]))

    with torch.no_grad():
        for offset, freq in enumerate(secret_freqs):
            angle = 2.0 * math.pi * float(freq) * positions / float(p)
            embeddings[:, offset * 2] += torch.sin(angle)
            embeddings[:, offset * 2 + 1] += torch.cos(angle)
        model.W_E.weight.copy_(embeddings)

    return model


def _load_model(artifact_path: str, fallback_factory: Callable[[], nn.Module]) -> nn.Module:
    """Load a serialized model artifact, or fall back to a deterministic in-memory build."""
    try:
        return torch.load(artifact_path, weights_only=False)
    except Exception:
        return fallback_factory()


def _infer_task1_ground_truth(model: nn.Module) -> list[int]:
    """Extract dead input-feature indices from the task 1 linear layer."""
    weight = getattr(getattr(model, "layer", None), "weight", None)
    if weight is None:
        raise ValueError("Task 1 model is missing layer.weight.")
    dead_indices = torch.where(torch.all(weight.detach() == 0, dim=0))[0].tolist()
    return sorted(int(index) for index in dead_indices)


def _infer_task2_ground_truth(model: nn.Module) -> list[int]:
    """Identify the multiplication neuron from model structure or ablation behavior."""
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
        def _ablate(_module: nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
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


def _infer_task3_ground_truth(model: nn.Module) -> list[int]:
    """Recover the planted Fourier frequencies from the embedding matrix."""
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

    top_k = min(TASK3_NUM_FREQUENCIES, max(0, energy.shape[0] - 1))
    if top_k == 0:
        return []

    frequency_indices = torch.topk(energy, k=top_k).indices.tolist()
    return sorted(int(index) for index in frequency_indices if int(index) > 0)


def _infer_task4_ground_truth(model: nn.Module) -> list[int]:
    """Identify the hidden neuron that carries the additive x3 signal."""
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
        def _ablate(_module: nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
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


def _normalize_submission(raw_submission: object) -> tuple[Optional[list[int]], Optional[str]]:
    """Convert a raw submission into a clean list of integers, or return a validation error."""
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


def _clamp_task_score(score: float) -> float:
    """Map any raw task score into the validator-safe open interval (0, 1)."""
    return round(min(MAX_TASK_SCORE, max(MIN_TASK_SCORE, float(score))), 4)


def _score_task1_raw(submission: list[int], ground_truth: list[int]) -> float:
    gt_set = set(ground_truth)
    submission_set = set(submission)
    matches = len(submission_set & gt_set)
    false_positives = len(submission_set - gt_set)

    if submission == ground_truth:
        return 1.0

    return max(0.0, (matches / len(ground_truth)) * 0.33 - (false_positives * 0.1))


def _score_task2_raw(submission: list[int], ground_truth: list[int]) -> float:
    if submission == ground_truth:
        return 1.0
    if ground_truth[0] in submission:
        return max(0.1, 1.0 / len(submission))
    return 0.0


def _task3_mse(submission: list[int], ground_truth: list[int]) -> Optional[float]:
    if len(submission) != len(ground_truth):
        return None

    return sum((candidate - expected) ** 2 for candidate, expected in zip(sorted(submission), ground_truth)) / len(ground_truth)


def _score_task3_raw(submission: list[int], ground_truth: list[int]) -> float:
    mse = _task3_mse(submission, ground_truth)
    if mse is None:
        return 0.0
    return max(0.0, 1.0 - mse)


def _score_task4_raw(submission: list[int], ground_truth: list[int]) -> float:
    if submission == ground_truth:
        return 1.0
    if ground_truth[0] in submission:
        return max(0.1, 0.8 / len(submission))
    return 0.0


def _task_key_for_level(task_level: int) -> str:
    return f"task{task_level}"


def _resolve_task_key(task_id: Any = None, task_level: Any = None) -> str:
    """Resolve a task selection from reset kwargs into a stable task key."""
    if task_id is not None:
        normalized = str(task_id).strip().lower()
        resolved = TASK_ALIASES.get(normalized)
        if resolved is not None:
            return resolved

    if task_level is not None:
        normalized = str(task_level).strip().lower()
        resolved = TASK_ALIASES.get(normalized)
        if resolved is not None:
            return resolved

    return "task1"


def resolve_task_selection(task_id: Any = None, task_level: Any = None) -> dict[str, Any]:
    """Return the canonical task identifier and level for a task selection hint."""
    task_key = _resolve_task_key(task_id=task_id, task_level=task_level)
    spec = TASK_SPECS[task_key]
    return {
        "task_id": spec["id"],
        "task_level": spec["level"],
        "task_name": spec["name"],
    }


def _task_metadata(task_level: int) -> dict[str, Any]:
    spec = TASK_SPECS[_task_key_for_level(task_level)]
    return {
        "task_id": spec["id"],
        "task_level": spec["level"],
        "task_name": spec["name"],
        "grader_name": spec["grader_name"],
    }


def get_task_catalog() -> list[dict[str, Any]]:
    """Return a validator-friendly public task manifest with explicit grader paths."""
    return [
        {
            "id": spec["id"],
            "level": spec["level"],
            "name": spec["name"],
            "description": spec["description"],
            "has_grader": True,
            "reset_payload": {"task_id": spec["id"]},
            "grader": {
                "name": spec["grader_name"],
                "module": spec["grader_module"],
                "function": spec["grader_function"],
                "score_range": {"min_exclusive": 0.0, "max_exclusive": 1.0},
            },
        }
        for spec in TASK_SPECS.values()
    ]


def _read_readme(base_dir: str) -> Optional[str]:
    readme_path = Path(base_dir) / "README.md"
    if not readme_path.exists():
        return None
    return readme_path.read_text(encoding="utf-8")


def _build_reset_prompt(task_key: str, seed: int) -> str:
    """Return a task-specific reset prompt so validators can enter tasks directly."""
    if task_key == "task2":
        return (
            f"PyTorchSandbox environment ready. (Seed: {seed})\n"
            "Task 2: Causal Ablation.\n"
            "The model computes y = (x1*x2) + x3 through a hidden layer available via `model.hidden`.\n"
            "Use forward hooks to ablate hidden neurons, then submit "
            '{"solution_target": [neuron_index]}.'
        )

    if task_key == "task3":
        return (
            f"PyTorchSandbox environment ready. (Seed: {seed})\n"
            "Task 3: Fourier Analysis.\n"
            "Analyze `model.W_E.weight` across the 97 token positions and recover the 5 planted frequencies.\n"
            "Submit them as a sorted JSON list like "
            '{"solution_target": [f1, f2, f3, f4, f5]}.'
        )

    if task_key == "task4":
        return (
            f"PyTorchSandbox environment ready. (Seed: {seed})\n"
            "Task 4: Additive Bypass Attribution.\n"
            "The same causal-ablation MLP has one hidden neuron that directly carries the additive x3 pathway.\n"
            "Use targeted ablations on `model.hidden` to identify the single additive bypass neuron, then submit "
            '{"solution_target": [neuron_index]}.'
        )

    return (
        f"PyTorchSandbox environment ready. (Seed: {seed})\n"
        "Task 1: Dead Neuron Detection.\n"
        "Find all zero-weight input indices in the Linear(10,1) model available as `model`.\n"
        "Use print() to inspect the weights, then submit a sorted JSON list of indices as "
        '{"solution_target": [i1, i2, i3]}.'
    )


class Task1Rubric(Rubric):
    def __init__(self, ground_truth: list[int]):
        super().__init__()
        self.ground_truth = list(ground_truth)
        self.last_score = 0.5

    def forward(self, action: Any, observation: Any) -> float:
        submission, error_message = _normalize_submission(getattr(action, "solution_target", None))
        if error_message is not None or submission is None:
            return MIN_TASK_SCORE
        return _clamp_task_score(_score_task1_raw(submission, self.ground_truth))


class Task2Rubric(Rubric):
    def __init__(self, ground_truth: list[int]):
        super().__init__()
        self.ground_truth = list(ground_truth)
        self.last_score = 0.5

    def forward(self, action: Any, observation: Any) -> float:
        submission, error_message = _normalize_submission(getattr(action, "solution_target", None))
        if error_message is not None or submission is None:
            return MIN_TASK_SCORE
        return _clamp_task_score(_score_task2_raw(submission, self.ground_truth))


class Task3Rubric(Rubric):
    def __init__(self, ground_truth: list[int]):
        super().__init__()
        self.ground_truth = list(ground_truth)
        self.last_score = 0.5

    def forward(self, action: Any, observation: Any) -> float:
        submission, error_message = _normalize_submission(getattr(action, "solution_target", None))
        if error_message is not None or submission is None:
            return MIN_TASK_SCORE
        return _clamp_task_score(_score_task3_raw(submission, self.ground_truth))


class Task4Rubric(Rubric):
    def __init__(self, ground_truth: list[int]):
        super().__init__()
        self.ground_truth = list(ground_truth)
        self.last_score = 0.5

    def forward(self, action: Any, observation: Any) -> float:
        submission, error_message = _normalize_submission(getattr(action, "solution_target", None))
        if error_message is not None or submission is None:
            return MIN_TASK_SCORE
        return _clamp_task_score(_score_task4_raw(submission, self.ground_truth))


class CurriculumRubric(Rubric):
    def __init__(self, ground_truths: dict[str, list[int]]):
        super().__init__()
        self.task1 = Task1Rubric(ground_truths["task1"])
        self.task2 = Task2Rubric(ground_truths["task2"])
        self.task3 = Task3Rubric(ground_truths["task3"])
        self.task4 = Task4Rubric(ground_truths["task4"])
        self.last_score = 0.5

    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {}) or {}
        task_id = metadata.get("task_id", "task1")
        rubric = getattr(self, task_id, self.task1)
        return rubric(action, observation)

    def reset(self) -> None:
        self.last_score = 0.5
        for _name, rubric in self.named_children():
            rubric.last_score = 0.5


class MechInterpEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: Optional[int] = None):
        self.seed = _coerce_seed(seed)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.artifacts_dir = os.path.join(self.base_dir, "artifacts")

        self.task1_model = _load_model(
            os.path.join(self.artifacts_dir, "task1.pt"),
            model_architectures.DeadNeuronMLP,
        )
        self.task2_model = _load_model(
            os.path.join(self.artifacts_dir, "task2.pt"),
            lambda: model_architectures.CausalAblationMLP(hidden_dim=10, mult_idx=2, add_idx=3),
        )
        self.task3_model = _load_model(
            os.path.join(self.artifacts_dir, "task3.pt"),
            _build_task3_model,
        )

        self.ground_truths = {
            "task1": _infer_task1_ground_truth(self.task1_model),
            "task2": _infer_task2_ground_truth(self.task2_model),
            "task3": _infer_task3_ground_truth(self.task3_model),
            "task4": _infer_task4_ground_truth(self.task2_model),
        }
        super().__init__(rubric=CurriculumRubric(self.ground_truths))

        self._state = InterpState(episode_id=str(uuid4()), step_count=0, task_level=1)
        self.task_level = 1

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MechInterpObservation:
        if seed is not None:
            self.seed = _coerce_seed(seed)

        task_key = _resolve_task_key(
            task_id=kwargs.get("task_id"),
            task_level=kwargs.get("task_level"),
        )
        task_spec = TASK_SPECS[task_key]

        self._reset_rubric()
        self._state = InterpState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_level=task_spec["level"],
        )
        self.task_level = task_spec["level"]

        return MechInterpObservation(
            stdout_or_error=_build_reset_prompt(task_key, self.seed),
            task_level=task_spec["level"],
            done=False,
            reward=MIN_TASK_SCORE,
            metadata={**_task_metadata(task_spec["level"]), "seed": self.seed, "step": 0},
        )

    def step(
        self,
        action: MechInterpAction,
        task_id: Any = None,
        task_level: Any = None,
        **_: Any,
    ) -> MechInterpObservation:
        if task_id is not None or task_level is not None:
            selection = resolve_task_selection(task_id=task_id, task_level=task_level)
            self.task_level = int(selection["task_level"])
            self._state.task_level = self.task_level

        self._state.step_count += 1
        active_task_level = self.task_level

        if self._state.step_count > MAX_EPISODE_STEPS:
            return MechInterpObservation(
                stdout_or_error=f"Episode ended. Max steps ({MAX_EPISODE_STEPS}) reached.",
                task_level=self.task_level,
                done=True,
                reward=MIN_TASK_SCORE,
                metadata={"reason": "max_steps_reached", "seed": self.seed, "step": self._state.step_count},
            )

        has_code = action.python_code is not None and action.python_code.strip() != ""
        has_solution = action.solution_target is not None

        stdout_or_error = ""
        reward = MIN_TASK_SCORE
        done = False

        if has_code and has_solution:
            stdout_or_error = "Invalid action: provide either python_code or solution_target, not both."
        elif has_code:
            stdout_or_error = self._execute_python_code(action.python_code or "", self._current_model())
        elif has_solution:
            submission, error_message = _normalize_submission(action.solution_target)
            if error_message is not None:
                stdout_or_error = error_message
                reward = MIN_TASK_SCORE
            else:
                stdout_or_error, reward, done = self._grade_solution(submission or [])
        else:
            stdout_or_error = "No action provided. Please submit either python_code or solution_target."

        metadata = {
            **_task_metadata(active_task_level),
            "seed": self.seed,
            "step": self._state.step_count,
            "current_task_level": self.task_level,
        }
        observation = MechInterpObservation(
            stdout_or_error=stdout_or_error,
            task_level=self.task_level,
            done=done,
            reward=reward,
            metadata=metadata,
        )
        if has_solution:
            self._apply_rubric(action, observation)
        return observation

    def _current_model(self) -> nn.Module:
        if self.task_level == 1:
            return self.task1_model
        if self.task_level in (2, 4):
            return self.task2_model
        return self.task3_model

    def _execute_python_code(self, python_code: str, model: nn.Module) -> str:
        """Execute agent code in a separate process so timeouts are enforced reliably."""
        if model is None:
            return "Current task model is unavailable."

        fd, model_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)

        try:
            torch.save(model, model_path)
            encoded_code = base64.b64encode(python_code.encode("utf-8")).decode("ascii")
            child_env = os.environ.copy()
            pythonpath_entries = [self.base_dir]
            existing_pythonpath = child_env.get("PYTHONPATH")
            if existing_pythonpath:
                pythonpath_entries.append(existing_pythonpath)
            child_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

            completed = subprocess.run(
                [sys.executable, "-c", EXEC_RUNNER, model_path, encoded_code],
                capture_output=True,
                cwd=self.base_dir,
                env=child_env,
                text=True,
                timeout=EXEC_TIMEOUT,
            )
            return f"{completed.stdout}{completed.stderr}".strip()
        except subprocess.TimeoutExpired:
            return "ERROR: Code execution timed out after 30 seconds. Avoid infinite loops."
        except Exception as exc:
            return f"ERROR: Failed to execute code safely: {exc}"
        finally:
            try:
                os.remove(model_path)
            except FileNotFoundError:
                pass

    def _grade_solution(self, submission: list[int]) -> tuple[str, float, bool]:
        """Grade a normalized solution submission for the current task."""
        done = False

        if self.task_level == 1:
            gt = self.ground_truths["task1"]
            reward = _clamp_task_score(_score_task1_raw(submission, gt))

            message = f"Task 1 graded. Score: {reward:.2f}"
            if submission == gt:
                self.task_level = 2
                self._state.task_level = 2
                message += (
                    "\n\nMoving to Task 2: Causal Ablation.\n"
                    "Find the hidden neuron responsible for the multiplicative term in y = x1*x2 + x3.\n"
                    "Use forward hooks on model.hidden, then submit {\"solution_target\": [neuron_index]}."
                )
            return message, reward, done

        if self.task_level == 2:
            gt = self.ground_truths["task2"]
            raw_reward = _score_task2_raw(submission, gt)
            reward = _clamp_task_score(raw_reward)
            if submission == gt:
                self.task_level = 3
                self._state.task_level = 3
                message = (
                    f"Task 2 graded. Score: {reward:.2f}\n\n"
                    "Moving to Task 3: Fourier Analysis of Planted Features.\n"
                    "Analyze model.W_E.weight across the 97 token positions and submit the 5 dominant "
                    "frequency indices as {\"solution_target\": [f1, f2, f3, f4, f5]}."
                )
            elif gt[0] in submission:
                message = (
                    f"Task 2 graded. Partial credit: {reward:.2f}\n"
                    "You found the correct multiplication neuron but included extra candidates. "
                    "Submit just the single hidden-neuron index for full credit and advancement."
                )
            else:
                message = (
                    f"Task 2 incorrect. Score: {reward:.2f}\n"
                    "Submit the single hidden-neuron index responsible for the multiplicative circuit."
                )
            return message, reward, done

        gt = self.ground_truths["task3"]
        if self.task_level == 3:
            if len(submission) != len(gt):
                return (
                    f"Task 3 failed. Expected exactly {len(gt)} frequency indices, got {len(submission)}.",
                    MIN_TASK_SCORE,
                    False,
                )

            mse = _task3_mse(submission, gt) or 0.0
            reward = _clamp_task_score(_score_task3_raw(submission, gt))

            if submission == gt:
                self.task_level = 4
                self._state.task_level = 4
                message = (
                    f"Task 3 graded. MSE={mse:.4f}, Score: {reward:.4f}\n\n"
                    "Moving to Task 4: Additive Bypass Attribution.\n"
                    "Return to the causal-ablation MLP and isolate the hidden neuron that carries the direct x3 bypass.\n"
                    "Submit it as {\"solution_target\": [neuron_index]}."
                )
            else:
                message = f"Task 3 graded. MSE={mse:.4f}, Score: {reward:.4f}"

            return message, reward, done

        gt = self.ground_truths["task4"]
        raw_reward = _score_task4_raw(submission, gt)
        reward = _clamp_task_score(raw_reward)
        if submission == gt:
            done = True
            message = (
                f"Task 4 graded. Score: {reward:.2f}\n\n"
                "All four tasks completed!"
            )
        elif gt[0] in submission:
            message = (
                f"Task 4 graded. Partial credit: {reward:.2f}\n"
                "You included the correct additive bypass neuron along with extra candidates. "
                "Submit the single hidden-neuron index for full credit."
            )
        else:
            message = (
                f"Task 4 incorrect. Score: {reward:.2f}\n"
                "Submit the single hidden-neuron index that directly carries x3 through the hidden layer."
            )
        return message, reward, done

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="mech_interp",
            description=(
                "PyTorchSandbox mechanistic interpretability benchmark with four graded tasks. "
                "Each task exposes a rubric-backed grader and reports scores strictly within (0, 1)."
            ),
            readme_content=_read_readme(self.base_dir),
            version="0.1.0",
            author="Kashyapsinh Gohil",
            documentation_url="https://github.com/KashyapSinh-Gohil/pytorch-sandbox-mech-interp",
        )

    @property
    def state(self) -> InterpState:
        return self._state
