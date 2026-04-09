# Copyright (c) 2026, Kashyapsinh Gohil
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Mech Interp environment."""

import json
import logging
from typing import Optional, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    )

try:
    from ..models import MechInterpAction, MechInterpObservation
    from .mech_interp_environment import MechInterpEnvironment, get_task_catalog
except (ImportError, ModuleNotFoundError):
    from models import MechInterpAction, MechInterpObservation
    from server.mech_interp_environment import MechInterpEnvironment, get_task_catalog

logger = logging.getLogger(__name__)


class JSONStringParserMiddleware(BaseHTTPMiddleware):
    """Middleware to parse JSON strings in solution_target field."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path == "/step":
            try:
                body = await request.body()
                data = json.loads(body.decode("utf-8")) if body else {}

                candidate_actions = []
                if isinstance(data, dict):
                    candidate_actions.append(data)
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

                async def receive():
                    return {
                        "type": "http.request",
                        "body": json.dumps(data).encode("utf-8"),
                        "more_body": False,
                    }

                request._receive = receive
            except Exception as exc:
                logger.warning("Failed to preprocess /step request body: %s", exc)

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
    return {
        "env_name": "mech_interp",
        "task_count": 3,
        "tasks": get_task_catalog(),
    }


@app.get("/tasks")
async def task_manifest():
    """Return an explicit task/grader manifest for validators and clients."""
    return {
        "env_name": "mech_interp",
        "task_count": 3,
        "tasks": get_task_catalog(),
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
