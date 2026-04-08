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

import os
import json
from typing import Any, Optional, Callable, Dict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

try:
    from openenv.core.env_server.http_server import create_app
    import gradio as gr
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import MechInterpAction, MechInterpObservation
    from .mech_interp_environment import MechInterpEnvironment
except (ImportError, ModuleNotFoundError):
    from models import MechInterpAction, MechInterpObservation
    from server.mech_interp_environment import MechInterpEnvironment


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
                
                if isinstance(data.get("action"), dict):
                    action = data["action"]
                    if "solution_target" in action and isinstance(action["solution_target"], str):
                        try:
                            action["solution_target"] = json.loads(action["solution_target"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                async def receive():
                    return {"type": "http.request", "body": json.dumps(data).encode("utf-8")}
                
                request._receive = receive
            except Exception:
                pass
        
        return await call_next(request)


def build_custom_gradio_ui(
    web_manager: Any,
    action_fields: Dict[str, Any],
    metadata: Dict[str, Any],
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """
    Custom Gradio UI builder with model selector and improved layout.
    """
    POPULAR_MODELS = [
        "deepseek-ai/DeepSeek-V3-0324",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
    ]
    
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# 🔬 {title}")
        gr.Markdown("""
        ## Mechanistic Interpretability RL Environment
        
        This environment tests an LLM agent's ability to reverse-engineer neural network internals through PyTorch code execution.
        
        ### The 3-Task Curriculum:
        
        **Task 1 (Easy): Dead Neuron Detection**
        - Find all zero-weight input indices in a Linear(10,1) model
        
        **Task 2 (Medium): Causal Ablation**
        - Identify the hidden neuron responsible for multiplication in y = x1*x2 + x3
        
        **Task 3 (Hard): Fourier Analysis of Planted Features**
        - Compute the DFT of model.W_E.weight across 97 token positions
        - Find the 5 frequency indices with highest total energy
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                model_dropdown = gr.Dropdown(
                    choices=POPULAR_MODELS,
                    value="deepseek-ai/DeepSeek-V3-0324",
                    label="Select Model",
                    info="Choose an LLM from HuggingFace for agentic inference"
                )
                seed_input = gr.Number(
                    value=42,
                    label="Random Seed",
                    info="Seed for reproducible ground truths"
                )
                gr.Markdown("""
                ### 📖 How to Use
                
                1. **Select a model** from the dropdown above
                2. Click **Reset** to start a new episode
                3. Enter **Python code** to inspect the model, OR
                4. Enter **Solution Target** as a JSON array (e.g., `[2, 5, 8]`)
                5. Click **Step** to execute
                
                ### 🔑 Action Formats
                
                ```python
                # Execute Python code:
                {"python_code": "print(model.layer.weight)"}
                
                # Submit solution:
                {"solution_target": [2, 5, 8]}
                ```
                
                ### ⚠️ Important Notes
                
                - Ground truths are randomized based on seed
                - Code execution has 30-second timeout
                - Max 30 steps per episode
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎮 Playground")
                step_btn = gr.Button("Step", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")
                state_btn = gr.Button("Get State", variant="secondary")
                
                with gr.Row():
                    python_code_input = gr.Textbox(
                        label="Python Code",
                        placeholder="Enter Python code to execute...",
                        lines=3
                    )
                    solution_target_input = gr.Textbox(
                        label="Solution Target",
                        placeholder='Enter JSON array, e.g., [2, 5, 8]',
                        lines=3
                    )
                
                output_md = gr.Markdown("**Status:** Click Reset to start...")
                output_json = gr.Code(label="Raw JSON Response", language="json")
                state_json = gr.Code(label="Environment State", language="json")
        
        def step_fn(python_code: str, solution_target: str):
            action = {}
            if python_code and python_code.strip():
                action["python_code"] = python_code
            if solution_target and solution_target.strip():
                try:
                    action["solution_target"] = json.loads(solution_target)
                except:
                    action["solution_target"] = solution_target
            
            if not action:
                return "**Error:** No action provided", "{}", "No action to submit"
            
            try:
                result = web_manager.step_environment(action)
                obs = result.get("observation", {})
                status = f"**Task Level:** {obs.get('task_level', '?')} | **Reward:** {obs.get('reward', 0)} | **Done:** {obs.get('done', False)}"
                status += f"\n\n{obs.get('stdout_or_error', '')}"
                return status, json.dumps(result, indent=2), json.dumps(web_manager.get_state(), indent=2)
            except Exception as e:
                return f"**Error:** {str(e)}", "{}", "{}"
        
        step_btn.click(
            step_fn,
            inputs=[python_code_input, solution_target_input],
            outputs=[output_md, output_json, state_json]
        )
        
        def reset_fn():
            try:
                result = web_manager.reset_environment()
                obs = result.get("observation", {})
                status = f"**Reset Complete!** Task Level: {obs.get('task_level', 1)}"
                status += f"\n\n{obs.get('stdout_or_error', '')}"
                return "", "", status, "{}", json.dumps(result, indent=2)
            except Exception as e:
                return "", "", f"**Error:** {str(e)}", "{}", "{}"
        
        reset_btn.click(
            reset_fn,
            outputs=[python_code_input, solution_target_input, output_md, state_json, output_json]
        )
        
        def state_fn():
            try:
                state = web_manager.get_state()
                return json.dumps(state, indent=2)
            except Exception as e:
                return f"Error: {str(e)}"
        
        state_btn.click(
            state_fn,
            outputs=[state_json]
        )
    
    return demo


def create_env_factory(seed: Optional[int] = None) -> Callable[[], MechInterpEnvironment]:
    """Create environment factory with optional seed."""
    def factory() -> MechInterpEnvironment:
        return MechInterpEnvironment(seed=seed)
    return factory


app = create_app(
    create_env_factory(),
    MechInterpAction,
    MechInterpObservation,
    env_name="mech_interp",
    max_concurrent_envs=1,
    gradio_builder=build_custom_gradio_ui,
)

app.add_middleware(JSONStringParserMiddleware)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()