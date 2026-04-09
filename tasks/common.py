"""Shared grading helpers for task manifest entrypoints."""

from typing import Any, Optional

MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99


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
