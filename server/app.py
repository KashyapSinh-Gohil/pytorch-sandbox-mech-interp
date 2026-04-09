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
    from .mech_interp_environment import MechInterpEnvironment
except (ImportError, ModuleNotFoundError):
    from models import MechInterpAction, MechInterpObservation
    from server.mech_interp_environment import MechInterpEnvironment

logger = logging.getLogger(__name__)


class JSONStringParserMiddleware(BaseHTTPMiddleware):
    """Middleware to parse JSON strings in solution_target field."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path == "/step":
            try:
                body = await request.body()
                data = json.loads(body.decode("utf-8")) if body else {}

                if isinstance(data.get("action"), dict):
                    action = data["action"]
                    if "solution_target" in action and isinstance(action["solution_target"], str):
                        try:
                            action["solution_target"] = json.loads(action["solution_target"])
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
        "tasks": [
            {
                "id": 1,
                "name": "Dead Neuron Detection",
                "description": "Find zeroed input features in a linear layer.",
            },
            {
                "id": 2,
                "name": "Causal Ablation",
                "description": "Identify the hidden neuron implementing the multiplicative circuit.",
            },
            {
                "id": 3,
                "name": "Fourier Analysis",
                "description": "Recover the planted frequencies from the embedding spectrum.",
            },
        ],
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
