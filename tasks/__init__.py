"""Task-specific grader registry for hackathon validation."""

from .task1.grader import grade as grade_task1
from .task2.grader import grade as grade_task2
from .task3.grader import grade as grade_task3

TASKS = [
    {
        "id": "task1",
        "level": 1,
        "name": "Dead Neuron Detection",
        "grader_name": "dead_neuron_detection_grader",
        "grader_module": "tasks.task1.grader",
        "grader_function": "grade",
        "grader_callable": grade_task1,
    },
    {
        "id": "task2",
        "level": 2,
        "name": "Causal Ablation",
        "grader_name": "causal_ablation_grader",
        "grader_module": "tasks.task2.grader",
        "grader_function": "grade",
        "grader_callable": grade_task2,
    },
    {
        "id": "task3",
        "level": 3,
        "name": "Fourier Analysis",
        "grader_name": "fourier_frequency_recovery_grader",
        "grader_module": "tasks.task3.grader",
        "grader_function": "grade",
        "grader_callable": grade_task3,
    },
]

TASK_COUNT = len(TASKS)
GRADER_COUNT = len(TASKS)

__all__ = [
    "GRADER_COUNT",
    "TASK_COUNT",
    "TASKS",
    "grade_task1",
    "grade_task2",
    "grade_task3",
]
