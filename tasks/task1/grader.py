"""Task 1 manifest grader entrypoint."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.common import clamp_task_score, get_task_ground_truth, normalize_submission


def grade(action, observation=None) -> float:
    submission, error_message = normalize_submission(action)
    if error_message is not None or submission is None:
        return 0.01

    ground_truth = get_task_ground_truth("task1")

    gt_set = set(ground_truth)
    submission_set = set(submission)
    matches = len(submission_set & gt_set)
    false_positives = len(submission_set - gt_set)

    if submission == ground_truth:
        return 0.99

    raw_score = max(0.0, (matches / len(ground_truth)) * 0.33 - (false_positives * 0.1))
    return clamp_task_score(raw_score)
