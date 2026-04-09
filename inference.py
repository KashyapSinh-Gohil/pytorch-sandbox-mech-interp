"""Agentic inference script for the PyTorchSandbox OpenEnv submission."""

import asyncio
import json
import os
import re
from typing import List, Optional

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3-0324")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "https://kashyapsinh-pytorch-sandbox-mech-interp.hf.space")
TASK_NAME = "mech_interp_curriculum"
BENCHMARK = "pytorch_sandbox"
MAX_STEPS = 30

MAX_RETRIES = 3
RETRY_DELAY = 2
MIN_TASK_SCORE = 0.01


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _single_line(value: object) -> str:
    return str(value).replace("\n", "\\n").replace("\r", "\\r")


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = _single_line(error) if error else "null"
    done_val = str(done).lower()
    action_val = _single_line(action)
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


SYSTEM_PROMPT = """You are an expert AI Safety researcher specializing in Mechanistic Interpretability of neural networks. You are interacting with a PyTorch sandbox environment that presents you with a 4-task curriculum.

## Your Tools
You interact by returning a single JSON object (no markdown fences) matching one of these two schemas:

### Option A: Run Python code to inspect the model
{"python_code": "<your pytorch code here>"}

The sandbox exposes `model` (the current task's nn.Module) and `torch` / `nn` in scope.
Use `print()` to see outputs. Use `model.named_parameters()`, forward hooks, and tensor investigation.

### Option B: Submit your final answer for the current task
{"solution_target": [<list of integers>]}

## Curriculum

### Task 1 (Easy)  Dead Neuron Detection
The model is a 1-layer Linear MLP (10 inputs -> 1 output). Some weight entries are exactly 0.0.
Find ALL input indices where model.layer.weight[0, i] == 0.0.
Submit as: {"solution_target": [i1, i2, ...]}  (sorted ascending)

### Task 2 (Medium)  Causal Ablation
The model computes y = (x1 * x2) + x3 through a 2-layer MLP with 10 hidden neurons.
One hidden neuron is the "multiplication neuron"  ablating it destroys the multiplicative component.
Use forward hooks on model.hidden to zero out each hidden neurons one at a time, measure the output change on test inputs, and identify which neuron index causes the largest error increase.
Submit as: {"solution_target": [neuron_index]}

### Task 3 (Hard)  Fourier Analysis of Planted Features
The model is a 1-layer Transformer engineered with synthetic modular arithmetic features.
The key frequencies are synthetically planted into the embedding matrix model.W_E.weight.
Compute the DFT of each embedding dimension across the 97 token positions, find the 5 frequency indices with the highest total energy.
Submit as: {"solution_target": [f1, f2, f3, f4, f5]}  (sorted ascending)

### Task 4 (Medium)  Additive Bypass Attribution
Return to the causal-ablation MLP and identify the hidden neuron that directly carries the additive x3 pathway.
Use forward hooks on model.hidden, design test inputs that isolate the x3 contribution, and identify which neuron index destroys the additive bypass when ablated.
Submit as: {"solution_target": [neuron_index]}

## CRITICAL RULES
1. Return ONLY raw JSON. No markdown code fences. No explanation text outside the JSON.
2. After running code, read the observation carefully before deciding next steps.
3. When using forward hooks, ALWAYS call handle.remove() after use.
4. You may take multiple exploration steps before submitting.
5. Sort your solution arrays in ascending order.
"""


def extract_json(text: str) -> Optional[dict]:
    """Robustly extract a JSON object from LLM output, handling markdown fences."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


async def call_llm_with_retry(client: OpenAI, messages: List[dict], model: str) -> str:
    """Call LLM with exponential backoff retry logic."""
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(wait_time)
    
    raise last_error


async def main() -> None:
    env = None
    rewards: List[float] = []
    steps_taken = 0
    success = False
    task_scores = {1: MIN_TASK_SCORE, 2: MIN_TASK_SCORE, 3: MIN_TASK_SCORE, 4: MIN_TASK_SCORE}

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN environment variable is required")

        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

        try:
            from mech_interp import MechInterpAction, MechInterpEnv
        except ModuleNotFoundError:
            import sys

            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from client import MechInterpEnv
            from models import MechInterpAction

        if LOCAL_IMAGE_NAME:
            env = await MechInterpEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = MechInterpEnv(base_url=ENV_URL)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        result = await env.reset()
        obs = result.observation

        messages.append({
            "role": "user",
            "content": (
                f"Environment initialized. Current task level: {obs.task_level}\n"
                f"Observation: {obs.stdout_or_error}\n\n"
                "Begin by inspecting the model for Task 1. Return your action as JSON."
            )
        })

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            try:
                llm_output = await call_llm_with_retry(client, messages, MODEL_NAME)
            except Exception as e:
                error_msg = f"LLM API error after {MAX_RETRIES} retries: {e}"
                log_step(step, "api_error", MIN_TASK_SCORE, False, error_msg)
                rewards.append(MIN_TASK_SCORE)
                steps_taken = step
                break

            parsed = extract_json(llm_output)
            action_for_log = "invalid_json"

            if parsed is None:
                action = MechInterpAction(python_code="print('LLM returned invalid JSON')")
                action_label = "parse_error_retry"
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "ERROR: Your response was not valid JSON. "
                        "Return ONLY a raw JSON object like {\"python_code\": \"...\"} "
                        "or {\"solution_target\": [...]}. No markdown, no explanation."
                    )
                })
            else:
                try:
                    action = MechInterpAction(**parsed)
                    action_label = "submit" if action.solution_target is not None else "code_exec"
                    action_for_log = json.dumps(parsed, separators=(",", ":"), ensure_ascii=True)
                except Exception as e:
                    action = MechInterpAction(python_code=f"print('Schema error: {e}')")
                    action_label = "schema_error_retry"
                    messages.append({"role": "assistant", "content": llm_output})
                    messages.append({
                        "role": "user",
                        "content": f"ERROR: JSON schema mismatch: {e}. Use keys 'python_code' (str) or 'solution_target' (list of ints)."
                    })

            prev_task = obs.task_level
            result = await env.step(action)
            obs = result.observation
            reward = result.reward if result.reward is not None else MIN_TASK_SCORE
            rewards.append(reward)
            steps_taken = step

            log_step(step, action_for_log if action_for_log != "invalid_json" else action_label, reward, obs.done, None)

            if action.solution_target is not None and reward > task_scores.get(prev_task, MIN_TASK_SCORE):
                task_scores[prev_task] = reward

            if parsed is not None and action_label not in ("parse_error_retry", "schema_error_retry"):
                messages.append({"role": "assistant", "content": llm_output})

                task_change_note = ""
                if obs.task_level != prev_task:
                    task_change_note = f"\nYou advanced to Task {obs.task_level}! Read the new task description from the system prompt."

                messages.append({
                    "role": "user",
                    "content": (
                        f"Step result (task_level={obs.task_level}, reward={reward:.2f}, done={obs.done}):\n"
                        f"{obs.stdout_or_error}"
                        f"{task_change_note}\n\n"
                        "What is your next action? Return JSON only."
                    )
                })

            if len(messages) > 22:
                messages = [messages[0]] + messages[-20:]

        success = (sum(task_scores.values()) / 4.0) > 0.9

    except Exception as e:
        log_step(steps_taken + 1, "fatal_error", MIN_TASK_SCORE, True, str(e))
        rewards.append(MIN_TASK_SCORE)
        steps_taken += 1

    finally:
        if env is not None:
            await env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
