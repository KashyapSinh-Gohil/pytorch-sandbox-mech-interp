# Copyright (c) 2026, Kashyapsinh Gohil
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mech Interp Environment Client."""

from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar

try:
    from openenv.core import EnvClient
except Exception:
    ActionT = TypeVar("ActionT")
    ObservationT = TypeVar("ObservationT")
    StateT = TypeVar("StateT")

    class EnvClient(Generic[ActionT, ObservationT, StateT]):  # type: ignore[override]
        pass

try:
    from openenv.core.client_types import StepResult
except Exception:
    ObservationResultT = TypeVar("ObservationResultT")

    @dataclass
    class StepResult(Generic[ObservationResultT]):  # type: ignore[override]
        observation: ObservationResultT
        reward: float | None = None
        done: bool = False
        info: Dict[str, Any] | None = None

try:
    from .models import InterpState, MechInterpAction, MechInterpObservation
except ImportError:
    from models import InterpState, MechInterpAction, MechInterpObservation


DEFAULT_CLIENT_REWARD = 0.01


class MechInterpEnv(
    EnvClient[MechInterpAction, MechInterpObservation, InterpState]
):
    """
    Client for the PyTorchSandbox Mechanistic Interpretability Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with MechInterpEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.stdout_or_error)
        ...
        ...     # Execute PyTorch code
        ...     result = client.step(MechInterpAction(python_code="print(model.layer.weight)"))
        ...     print(result.observation.stdout_or_error)
        ...
        ...     # Submit a solution
        ...     result = client.step(MechInterpAction(solution_target=[2, 5, 8]))
        ...     print(result.observation.stdout_or_error, result.reward)
    """

    def _step_payload(self, action: MechInterpAction) -> Dict:
        """
        Convert MechInterpAction to JSON payload for step message.

        Args:
            action: MechInterpAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {}
        if action.python_code is not None:
            payload["python_code"] = action.python_code
        if action.solution_target is not None:
            payload["solution_target"] = action.solution_target
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[MechInterpObservation]:
        """
        Parse server response into StepResult[MechInterpObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MechInterpObservation
        """
        obs_data = payload.get("observation", {})
        observation = MechInterpObservation(
            stdout_or_error=obs_data.get("stdout_or_error", ""),
            task_level=obs_data.get("task_level", 1),
            done=payload.get("done", False),
            reward=payload.get("reward", DEFAULT_CLIENT_REWARD),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", DEFAULT_CLIENT_REWARD),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> InterpState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id, step_count, and task level
        """
        return InterpState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_level=payload.get("task_level", 1),
        )
