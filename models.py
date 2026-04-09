from typing import Any, Dict, List, Optional
import json
from pydantic import BaseModel, ConfigDict, Field, field_validator
from openenv.core.env_server.types import Action, Observation, State


def _is_pydantic_model(candidate: Any) -> bool:
    """Return True when the imported OpenEnv base type is a Pydantic model."""
    try:
        return isinstance(candidate, type) and issubclass(candidate, BaseModel)
    except TypeError:
        return False


class _FallbackAction(BaseModel):
    """Compatibility base used when tests monkeypatch OpenEnv with plain classes."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class _FallbackObservation(BaseModel):
    """Compatibility observation base mirroring the OpenEnv fields we rely on."""

    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class _FallbackState(BaseModel):
    """Compatibility state base for mocked OpenEnv test environments."""

    episode_id: Optional[str] = None
    step_count: int = 0
    model_config = ConfigDict(extra="allow")


ActionBase = Action if _is_pydantic_model(Action) else _FallbackAction
ObservationBase = Observation if _is_pydantic_model(Observation) else _FallbackObservation
StateBase = State if _is_pydantic_model(State) else _FallbackState


class MechInterpAction(ActionBase):
    """Agent action: either execute code or submit a solution."""

    python_code: Optional[str] = Field(None, description="Python code to execute in the sandbox.")
    solution_target: Optional[List[Any]] = Field(None, description="Sorted list of integer indices/frequencies.")
    model_config = ConfigDict(extra="forbid")

    @field_validator("solution_target", mode="before")
    @classmethod
    def parse_solution_target(cls, v):
        """Convert string JSON to list if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return v
        return v


class MechInterpObservation(ObservationBase):
    """Environment observation returned after each step."""

    stdout_or_error: str = Field(default="", description="Captured stdout/stderr from code execution or grading feedback.")
    task_level: int = Field(default=1, description="Current task level (1, 2, 3, or 4).")
    model_config = ConfigDict(extra="forbid")


class InterpState(StateBase):
    """Extended state tracking the current task level."""

    task_level: int = 1
    model_config = ConfigDict(extra="allow")
