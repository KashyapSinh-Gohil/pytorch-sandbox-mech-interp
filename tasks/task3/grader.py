"""Task 3 manifest grader entrypoint."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.common import clamp_task_score, normalize_submission

GROUND_TRUTH = [2, 17, 23, 44, 47]


def grade(action, observation=None) -> float:
    submission, error_message = normalize_submission(action)
    if error_message is not None or submission is None:
        return 0.01

    if len(submission) != len(GROUND_TRUTH):
        return 0.01

    mse = sum((candidate - expected) ** 2 for candidate, expected in zip(sorted(submission), GROUND_TRUTH)) / len(GROUND_TRUTH)
    raw_score = max(0.0, 1.0 - mse)
    return clamp_task_score(raw_score)
