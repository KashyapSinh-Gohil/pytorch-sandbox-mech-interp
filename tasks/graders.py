"""Top-level grader router for validators that expect a single grader entrypoint."""

from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.task1.grader import grade as grade_task1
from tasks.task2.grader import grade as grade_task2
from tasks.task3.grader import grade as grade_task3
from tasks.task4.grader import grade as grade_task4


def _extract_task_id_from_stdout(stdout_or_error: Any) -> str | None:
    if not isinstance(stdout_or_error, str):
        return None

    match = re.search(r"\bTask\s+([1-4])\b", stdout_or_error)
    if match is None:
        return None

    return f"task{match.group(1)}"


def _extract_task_id(observation: Any) -> str:
    """Resolve task identifier from observation metadata or task level hints."""
    if isinstance(observation, dict):
        metadata = observation.get("metadata", {}) or {}
        task_id = (
            metadata.get("task_id")
            or observation.get("task_id")
            or observation.get("graded_task_id")
        )
        if task_id:
            return str(task_id)
        stdout_or_error = observation.get("stdout_or_error")
        task_id_from_stdout = _extract_task_id_from_stdout(stdout_or_error)
        if task_id_from_stdout is not None:
            return task_id_from_stdout
        task_level = observation.get("task_level")
    else:
        metadata = getattr(observation, "metadata", {}) or {}
        task_id = (
            metadata.get("task_id")
            or getattr(observation, "task_id", None)
            or getattr(observation, "graded_task_id", None)
        )
        if task_id:
            return str(task_id)
        stdout_or_error = getattr(observation, "stdout_or_error", None)
        task_id_from_stdout = _extract_task_id_from_stdout(stdout_or_error)
        if task_id_from_stdout is not None:
            return task_id_from_stdout
        task_level = getattr(observation, "task_level", None)

    if task_level == 2:
        return "task2"
    if task_level == 3:
        return "task3"
    if task_level == 4:
        return "task4"
    return "task1"


def grade_task(action: Any, observation: Any = None) -> float:
    """Dispatch grading to the per-task grader using observation context."""
    task_id = _extract_task_id(observation)
    if task_id == "task2":
        return grade_task2(action, observation)
    if task_id == "task3":
        return grade_task3(action, observation)
    if task_id == "task4":
        return grade_task4(action, observation)
    return grade_task1(action, observation)
