"""Top-level grader router for validators that expect a single grader entrypoint."""

from typing import Any

from tasks.task1.grader import grade as grade_task1
from tasks.task2.grader import grade as grade_task2
from tasks.task3.grader import grade as grade_task3


def _extract_task_id(observation: Any) -> str:
    """Resolve task identifier from observation metadata or task level hints."""
    if isinstance(observation, dict):
        metadata = observation.get("metadata", {}) or {}
        task_id = metadata.get("task_id")
        if task_id:
            return str(task_id)
        task_level = observation.get("task_level")
    else:
        metadata = getattr(observation, "metadata", {}) or {}
        task_id = metadata.get("task_id")
        if task_id:
            return str(task_id)
        task_level = getattr(observation, "task_level", None)

    if task_level == 2:
        return "task2"
    if task_level == 3:
        return "task3"
    return "task1"


def grade_task(action: Any, observation: Any = None) -> float:
    """Dispatch grading to the per-task grader using observation context."""
    task_id = _extract_task_id(observation)
    if task_id == "task2":
        return grade_task2(action, observation)
    if task_id == "task3":
        return grade_task3(action, observation)
    return grade_task1(action, observation)
