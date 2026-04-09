"""Task 2 manifest grader entrypoint."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.common import clamp_task_score, normalize_submission

GROUND_TRUTH = [2]


def grade(action, observation=None) -> float:
    submission, error_message = normalize_submission(action)
    if error_message is not None or submission is None:
        return 0.01

    if submission == GROUND_TRUTH:
        raw_score = 1.0
    elif GROUND_TRUTH[0] in submission:
        raw_score = max(0.1, 1.0 / len(submission))
    else:
        raw_score = 0.0
    return clamp_task_score(raw_score)
