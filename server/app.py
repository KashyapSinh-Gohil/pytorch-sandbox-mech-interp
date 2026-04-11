# Copyright (c) 2026, Kashyapsinh Gohil
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Mech Interp environment."""

import json
import logging
from threading import Lock
from typing import Any, Optional, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    )

try:
    from mech_interp.models import MechInterpAction, MechInterpObservation
    from server.mech_interp_environment import (
        MechInterpEnvironment,
        get_task_catalog,
        resolve_task_selection,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from models import MechInterpAction, MechInterpObservation
        from mech_interp_environment import (
            MechInterpEnvironment,
            get_task_catalog,
            resolve_task_selection,
        )
    except (ImportError, ModuleNotFoundError):
        from ..models import MechInterpAction, MechInterpObservation
        from .mech_interp_environment import (
            MechInterpEnvironment,
            get_task_catalog,
            resolve_task_selection,
        )

logger = logging.getLogger(__name__)

_TASK_ID_COOKIE = "mech_interp_task_id"
_TASK_LEVEL_COOKIE = "mech_interp_task_level"
_HTTP_TASK_SELECTION = resolve_task_selection()
_HTTP_TASK_SELECTION_LOCK = Lock()


def _set_http_task_selection(selection: dict[str, Any]) -> None:
    with _HTTP_TASK_SELECTION_LOCK:
        _HTTP_TASK_SELECTION.update(selection)


def _get_http_task_selection() -> dict[str, Any]:
    with _HTTP_TASK_SELECTION_LOCK:
        return dict(_HTTP_TASK_SELECTION)


def _selection_payload(selection: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Return only request-body-safe task selectors expected by reset/step handlers."""
    if not selection:
        return {}
    payload: dict[str, Any] = {}
    if "task_id" in selection:
        payload["task_id"] = selection["task_id"]
    if "task_level" in selection:
        payload["task_level"] = selection["task_level"]
    return payload


def _extract_task_selection(data: Any) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None

    action = data.get("action") if isinstance(data.get("action"), dict) else {}
    metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}

    task_id = data.get("task_id", metadata.get("task_id"))
    task_level = data.get("task_level", metadata.get("task_level"))
    if task_id is None and task_level is None:
        return None

    return resolve_task_selection(task_id=task_id, task_level=task_level)


def _selection_from_request(request: Request) -> Optional[dict[str, Any]]:
    task_id = request.cookies.get(_TASK_ID_COOKIE)
    task_level = request.cookies.get(_TASK_LEVEL_COOKIE)
    if task_id is None and task_level is None:
        return None
    return resolve_task_selection(task_id=task_id, task_level=task_level)


class JSONStringParserMiddleware(BaseHTTPMiddleware):
    """Normalize task-aware HTTP payloads for stateless validator requests."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path in {"/reset", "/step"}:
            try:
                body = await request.body()
                data = json.loads(body.decode("utf-8")) if body else {}
                if not isinstance(data, dict):
                    data = {}

                selection = _extract_task_selection(data)
                if request.url.path == "/reset":
                    selection = selection or resolve_task_selection()
                else:
                    selection = (
                        selection
                        or _selection_from_request(request)
                        or _get_http_task_selection()
                    )
                    data.update(_selection_payload(selection))

                candidate_actions = [data]
                if isinstance(data.get("action"), dict):
                    candidate_actions.append(data["action"])

                for action in candidate_actions:
                    if "solution_target" in action and isinstance(action["solution_target"], str):
                        try:
                            parsed = json.loads(action["solution_target"])
                            if isinstance(parsed, list):
                                action["solution_target"] = parsed
                            else:
                                logger.warning(
                                    "solution_target parsed to %s, expected list",
                                    type(parsed).__name__,
                                )
                        except (json.JSONDecodeError, TypeError) as exc:
                            logger.warning("Failed to parse solution_target JSON string: %s", exc)

                if selection is not None:
                    data.update(_selection_payload(selection))
                    _set_http_task_selection(selection)

                updated_body = json.dumps(data).encode("utf-8")

                async def receive():
                    return {
                        "type": "http.request",
                        "body": updated_body,
                        "more_body": False,
                    }

                request._body = updated_body
                request._receive = receive
                response = await call_next(request)
                if selection is not None:
                    response.set_cookie(_TASK_ID_COOKIE, str(selection["task_id"]))
                    response.set_cookie(_TASK_LEVEL_COOKIE, str(selection["task_level"]))
                return response
            except Exception as exc:
                logger.warning(
                    "Failed to preprocess %s request body: %s",
                    request.url.path,
                    exc,
                )

        return await call_next(request)


def create_env_factory(seed: Optional[int] = None) -> Callable[[], MechInterpEnvironment]:
    """Create environment factory with optional seed."""
    def factory() -> MechInterpEnvironment:
        return MechInterpEnvironment(seed=seed)
    return factory


# Create app with default OpenEnv web interface
app = create_app(
    create_env_factory(),
    MechInterpAction,
    MechInterpObservation,
    env_name="mech_interp",
    max_concurrent_envs=1,
)

app.add_middleware(JSONStringParserMiddleware)


@app.get("/health")
async def health_check():
    """Lightweight deployment health endpoint."""
    return {"status": "healthy", "env_name": "mech_interp"}


@app.get("/info")
async def environment_info():
    """Describe the curriculum at a high level without leaking task answers."""
    tasks = get_task_catalog()
    return {
        "env_name": "mech_interp",
        "task_count": len(tasks),
        "tasks": tasks,
    }


@app.get("/tasks")
async def task_manifest():
    """Return an explicit task manifest for validators and clients."""
    tasks = get_task_catalog()
    return {
        "env_name": "mech_interp",
        "task_count": len(tasks),
        "grader_count": len(tasks),
        "tasks": tasks,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
