from typing import Optional, List, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class MechInterpAction(Action):
    """Agent action: either execute code or submit a solution."""
    python_code: Optional[str] = Field(None, description="Python code to execute in the sandbox.")
    solution_target: Optional[List[Any]] = Field(None, description="Sorted list of integer indices/frequencies.")


class MechInterpObservation(Observation):
    """Environment observation returned after each step."""
    stdout_or_error: str = Field(default="", description="Captured stdout/stderr from code execution or grading feedback.")
    task_level: int = Field(default=1, description="Current task level (1, 2, or 3).")


class InterpState(State):
    """Extended state tracking the current task level."""
    task_level: int = 1
