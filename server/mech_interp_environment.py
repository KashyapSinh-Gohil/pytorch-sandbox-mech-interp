import os
import io
import sys
import signal
import traceback
import contextlib
import math
import random
from uuid import uuid4
from typing import Optional, List

import torch
import torch.nn as nn
from openenv.core.env_server.interfaces import Environment

from . import model_architectures
sys.modules['model_architectures'] = model_architectures

try:
    from ..models import MechInterpAction, MechInterpObservation, InterpState
except ImportError:
    from models import MechInterpAction, MechInterpObservation, InterpState

EXEC_TIMEOUT = 30
MAX_EPISODE_STEPS = 30


class ExecTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ExecTimeoutError("Code execution timed out after 30 seconds.")


def _generate_ground_truths(seed: int = 42) -> dict:
    """Generate ground truths with reproducible randomness."""
    random.seed(seed)
    torch.manual_seed(seed)
    
    task1_gt = sorted(random.sample(range(10), 3))
    
    task2_gt = [random.randint(0, 9)]
    
    task3_candidates = list(range(1, 48))
    random.shuffle(task3_candidates)
    task3_gt = sorted(task3_candidates[:5])
    
    return {
        "task1": task1_gt,
        "task2": task2_gt,
        "task3": task3_gt
    }


SAFE_BUILTINS = {
    'print': print,
    'len': len,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'sum': sum,
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'sorted': sorted,
    'list': list,
    'dict': dict,
    'set': set,
    'tuple': tuple,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'type': type,
    'isinstance': isinstance,
    'hasattr': hasattr,
    'getattr': getattr,
    'setattr': setattr,
    'zip': zip,
    'any': any,
    'all': all,
}


class MechInterpEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: Optional[int] = None):
        self._state = InterpState(episode_id=str(uuid4()), step_count=0)
        self.task_level = 1
        self.seed = seed if seed is not None else int(os.getenv("OPENENV_SEED", "42"))
        self.ground_truths = _generate_ground_truths(self.seed)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_dir = os.path.join(base_dir, "artifacts")
        
        try:
            self.task1_model = torch.load(os.path.join(artifacts_dir, "task1.pt"), weights_only=False)
        except Exception:
            self.task1_model = None
            
        try:
            self.task2_model = torch.load(os.path.join(artifacts_dir, "task2.pt"), weights_only=False)
        except Exception:
            self.task2_model = None
            
        try:
            self.task3_model = torch.load(os.path.join(artifacts_dir, "task3.pt"), weights_only=False)
        except Exception:
            self.task3_model = None

    def reset(self) -> MechInterpObservation:
        self._state = InterpState(episode_id=str(uuid4()), step_count=0)
        self.task_level = 1
        self._state.task_level = 1
        
        gt = self.ground_truths["task1"]
        task_instructions = (
            f"PyTorchSandbox environment ready. (Seed: {self.seed})\n"
            f"Task 1: Dead Neuron Detection - Find all zero-weight input indices in a Linear(10,1) model.\n"
            f"The model is available as `model`. Use print() to inspect it.\n"
            f"Expected answer format: {{\"solution_target\": [{gt[0]}, {gt[1]}, {gt[2]}]}}\n"
            f"Hint: Check model.layer.weight[0, :] for zero values."
        )

        return MechInterpObservation(
            stdout_or_error=task_instructions,
            task_level=1,
            done=False,
            reward=0.0,
        )

    def step(self, action: MechInterpAction) -> MechInterpObservation:
        self._state.step_count += 1
        
        stdout_err = ""
        reward = 0.0
        done = False
        
        if self._state.step_count >= MAX_EPISODE_STEPS:
            return MechInterpObservation(
                stdout_or_error=f"Episode ended. Max steps ({MAX_EPISODE_STEPS}) reached.",
                task_level=self.task_level,
                done=True,
                reward=reward,
                metadata={"step": self._state.step_count, "reason": "max_steps_reached"},
            )
        
        if self.task_level == 1:
            current_model = self.task1_model
        elif self.task_level == 2:
            current_model = self.task2_model
        else:
            current_model = self.task3_model

        if action.python_code is not None and action.python_code.strip():
            safe_globals = {"__builtins__": SAFE_BUILTINS}
            safe_locals = {
                "model": current_model, 
                "torch": torch, 
                "nn": nn,
                "math": math,
            }
            
            f = io.StringIO()
            
            old_handler = None
            try:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(EXEC_TIMEOUT)
            except (AttributeError, ValueError):
                pass

            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                try:
                    compiled_code = compile(action.python_code, "<agent_code>", "exec")
                    exec(compiled_code, safe_globals, safe_locals)
                except ExecTimeoutError:
                    f.write("\nERROR: Code execution timed out after 30 seconds. Avoid infinite loops.\n")
                except Exception:
                    traceback.print_exc(file=f)

            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass

            stdout_err = f.getvalue()
            
            if current_model is not None:
                for module in current_model.modules():
                    module._forward_hooks.clear()
                    module._forward_pre_hooks.clear()
                    if hasattr(module, '_backward_hooks'):
                        module._backward_hooks.clear()
                    if hasattr(module, '_backward_pre_hooks'):
                        module._backward_pre_hooks.clear()

        elif action.solution_target is not None:
            if not isinstance(action.solution_target, list):
                stdout_err = "Invalid submission: solution_target must be a list of integers."
                reward = 0.0
            elif len(action.solution_target) == 0:
                stdout_err = "Invalid submission: solution_target cannot be empty."
                reward = 0.0
            else:
                submission = action.solution_target

                if self.task_level == 1:
                    gt = self.ground_truths["task1"]
                    matches = sum(1 for x in submission if x in gt)
                    false_positives = sum(1 for x in submission if x not in gt)
                    
                    if matches == len(gt) and false_positives == 0 and len(submission) == len(gt):
                        reward = 1.0
                    else:
                        reward = round(max(0.0, (matches / len(gt)) - (false_positives * 0.1)), 2)

                    stdout_err = f"Task 1 graded. Score: {reward}"
                    if reward >= 0.9:
                        self.task_level = 2
                        self._state.task_level = 2
                        gt2 = self.ground_truths["task2"]
                        stdout_err += f"\n\nMoving to Task 2: Causal Ablation.\n"
                        stdout_err += f"Find the hidden neuron responsible for multiplication in y = x1*x2 + x3.\n"
                        stdout_err += f"Use forward hooks on model.hidden to ablate neurons one at a time.\n"
                        stdout_err += f"Expected answer format: {{\"solution_target\": [{gt2[0]}]}}"

                elif self.task_level == 2:
                    gt = self.ground_truths["task2"]
                    
                    if submission == gt:
                        reward = 1.0
                        self.task_level = 3
                        self._state.task_level = 3
                        gt3 = self.ground_truths["task3"]
                        stdout_err = f"Task 2 graded. Maximum reward attained.\n\n"
                        stdout_err += f"Moving to Task 3: Fourier Analysis of Planted Features.\n"
                        stdout_err += f"Compute the DFT of model.W_E.weight across 97 token positions.\n"
                        stdout_err += f"Find the 5 frequency indices with highest total energy.\n"
                        stdout_err += f"Expected answer format: {{\"solution_target\": {gt3}}}"
                    elif gt[0] in submission:
                        false_positives = len(submission) - 1
                        reward = round(max(0.1, 1.0 / (false_positives + 1)), 2)
                        stdout_err = f"Task 2 graded. Partial credit ({reward:.2f}). "
                        stdout_err += f"You correctly identified neuron {gt[0]}, but included {false_positives} false positive(s). Narrow it down!"
                    else:
                        stdout_err = f"Task 2 incorrect. Expected neuron index {gt[0]}, got {submission}."

                elif self.task_level == 3:
                    gt = self.ground_truths["task3"]
                    
                    if len(submission) != len(gt):
                        stdout_err = f"Task 3 failed. Expected exactly {len(gt)} frequency indices, got {len(submission)}."
                        reward = 0.0
                    else:
                        try:
                            mse = sum((s - g) ** 2 for s, g in zip(sorted(submission), sorted(gt))) / len(gt)
                            reward = round(max(0.0, 1.0 - mse / 100.0), 4)
                            
                            if reward >= 0.99:
                                done = True
                                stdout_err = f"Task 3 graded. MSE={mse:.4f}, Score: {reward}\n\nAll tasks completed! 🎉"
                            else:
                                stdout_err = f"Task 3 graded. MSE={mse:.4f}, Score: {reward}"
                        except Exception:
                            reward = 0.0
                            stdout_err = "Error calculating score. Ensure all values are numeric."
        else:
            stdout_err = "No action provided. Please submit either python_code or solution_target."
            reward = 0.0

        return MechInterpObservation(
            stdout_or_error=stdout_err,
            task_level=self.task_level,
            done=done,
            reward=reward,
            metadata={"step": self._state.step_count, "seed": self.seed},
        )

    @property
    def state(self) -> InterpState:
        return self._state
