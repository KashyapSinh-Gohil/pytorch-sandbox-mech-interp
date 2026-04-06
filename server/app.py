# Copyright (c) 2026, Kashyapsinh Gohil
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Mech Interp Environment.

This module creates an HTTP server that exposes the MechInterpEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import MechInterpAction, MechInterpObservation
    from .mech_interp_environment import MechInterpEnvironment
except (ImportError, ModuleNotFoundError):
    from models import MechInterpAction, MechInterpObservation
    from server.mech_interp_environment import MechInterpEnvironment


# Create the app with web interface and README integration
app = create_app(
    MechInterpEnvironment,
    MechInterpAction,
    MechInterpObservation,
    env_name="mech_interp",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


class JSONStringParserMiddleware(BaseHTTPMiddleware):
    """
    Middleware to parse JSON strings in solution_target field.
    
    The Gradio web interface sends solution_target as a JSON string (e.g., "[2, 5, 8]")
    but the backend expects a list. This middleware intercepts /step requests and
    converts JSON strings to lists before they reach Pydantic validation.
    """
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path == "/step":
            try:
                body = await request.body()
                data = json.loads(body.decode("utf-8")) if body else {}
                
                # Parse solution_target if it's a string JSON array
                if isinstance(data.get("action"), dict):
                    action = data["action"]
                    if "solution_target" in action and isinstance(action["solution_target"], str):
                        try:
                            action["solution_target"] = json.loads(action["solution_target"])
                        except (json.JSONDecodeError, TypeError):
                            pass  # Leave as-is if parsing fails
                
                # Create new request with modified body
                async def receive():
                    return {"type": "http.request", "body": json.dumps(data).encode("utf-8")}
                
                request._receive = receive
            except Exception:
                pass  # On any error, proceed with original request
        
        return await call_next(request)


# Add middleware to the app
app.add_middleware(JSONStringParserMiddleware)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m mech_interp.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn mech_interp.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
