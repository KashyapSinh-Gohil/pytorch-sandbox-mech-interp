"""Task 2 manifest grader entrypoint."""

from ..common import clamp_task_score, normalize_submission

GROUND_TRUTH = [2]


def grade(action, observation=None) -> float:
    submission, error_message = normalize_submission(action)
    if error_message is not None or submission is None:
        return 0.01

    raw_score = 1.0 if submission == GROUND_TRUTH else 0.0
    return clamp_task_score(raw_score)
