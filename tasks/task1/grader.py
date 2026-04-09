"""Task 1 manifest grader entrypoint."""

from ..common import clamp_task_score, normalize_submission

GROUND_TRUTH = [2, 5, 8]


def grade(action, observation=None) -> float:
    submission, error_message = normalize_submission(action)
    if error_message is not None or submission is None:
        return 0.01

    gt_set = set(GROUND_TRUTH)
    submission_set = set(submission)
    matches = len(submission_set & gt_set)
    false_positives = len(submission_set - gt_set)

    if submission == GROUND_TRUTH:
        return 0.99

    raw_score = max(0.0, (matches / len(GROUND_TRUTH)) * 0.33 - (false_positives * 0.1))
    return clamp_task_score(raw_score)
