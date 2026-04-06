import os
import io
import sys
import signal
import traceback
import contextlib
from uuid import uuid4

import torch
import torch.nn as nn
from openenv.core.env_server.interfaces import Environment

from . import model_architectures
sys.modules['model_architectures'] = model_architectures

try:
    from ..models import MechInterpAction, MechInterpObservation, InterpState
except ImportError:
    from models import MechInterpAction, MechInterpObservation, InterpState

# Timeout for exec() calls (seconds)
EXEC_TIMEOUT = 30


class ExecTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ExecTimeoutError("Code execution timed out after 30 seconds.")


class MechInterpEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = InterpState(episode_id=str(uuid4()), step_count=0)
        self.task_level = 1

        # Load the models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_dir = os.path.join(base_dir, "artifacts")
        try:
            self.task1_model = torch.load(os.path.join(artifacts_dir, "task1.pt"), weights_only=False)
            self.task2_model = torch.load(os.path.join(artifacts_dir, "task2.pt"), weights_only=False)
            self.task3_model = torch.load(os.path.join(artifacts_dir, "task3.pt"), weights_only=False)
        except Exception as e:
            self.task1_model = None
            self.task2_model = None
            self.task3_model = None
            print(f"Failed to load artifacts: {e}")

    def reset(self) -> MechInterpObservation:
        self._state = InterpState(episode_id=str(uuid4()), step_count=0)
        self.task_level = 1
        self._state.task_level = 1

        return MechInterpObservation(
            stdout_or_error=(
                "PyTorchSandbox environment ready.\n"
                "Task 1: Dead Neuron Detection  Find all zero-weight input indices in a Linear(10,1) model.\n"
                "The model is available as `model`. Use print() to inspect it.\n"
                "Submit your answer as: {\"solution_target\": [sorted list of indices]}"
            ),
            task_level=1,
            done=False,
            reward=0.0,
        )

    def step(self, action: MechInterpAction) -> MechInterpObservation:
        self._state.step_count += 1

        stdout_err = ""
        reward = 0.0
        done = False

        if self.task_level == 1:
            current_model = self.task1_model
        elif self.task_level == 2:
            current_model = self.task2_model
        else:
            current_model = self.task3_model

        if action.python_code is not None:
            local_namespace = {"model": current_model, "torch": torch, "nn": nn}
            global_namespace = local_namespace

            f = io.StringIO()

            # Set alarm-based timeout (Unix only)
            old_handler = None
            try:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(EXEC_TIMEOUT)
            except (AttributeError, ValueError):
                # Windows or non-main thread  skip timeout
                pass

            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                try:
                    compiled_code = compile(action.python_code, "<agent_code>", "exec")
                    exec(compiled_code, global_namespace, local_namespace)
                except ExecTimeoutError:
                    f.write("\nERROR: Code execution timed out after 30 seconds. Avoid infinite loops.\n")
                except Exception:
                    traceback.print_exc(file=f)

            # Cancel alarm
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass

            stdout_err = f.getvalue()

            # Purge all hooks to prevent OOM on long sessions
            if current_model is not None:
                for module in current_model.modules():
                    module._forward_hooks.clear()
                    module._forward_pre_hooks.clear()
                    if hasattr(module, '_backward_hooks'):
                        module._backward_hooks.clear()
                    if hasattr(module, '_backward_pre_hooks'):
                        module._backward_pre_hooks.clear()

        elif action.solution_target is not None:
            submission = action.solution_target

            if self.task_level == 1:
                gt = [2, 5, 8]
                matches = sum(1 for x in submission if x in gt)
                false_positives = sum(1 for x in submission if x not in gt)
                if matches == len(gt) and false_positives == 0:
                    reward = 1.0
                else:
                    reward = round(max(0.0, matches * 0.33 - false_positives * 0.1), 2)

                stdout_err = f"Task 1 graded. Score: {reward}"
                if reward >= 0.99:
                    self.task_level = 2
                    self._state.task_level = 2
                    stdout_err += "\nMoving to Task 2: Causal Ablation."
                    stdout_err += "\nFind the hidden neuron responsible for multiplication in y = x1*x2 + x3."
                    stdout_err += "\nUse forward hooks on model.hidden to ablate neurons one at a time."

            elif self.task_level == 2:
                gt = [2]
                if not isinstance(submission, list):
                    reward = 0.0
                    stdout_err = "Task 2 failed. Expected a list containing the neuron index."
                elif submission == gt:
                    reward = 1.0
                    self.task_level = 3
                    self._state.task_level = 3
                    stdout_err = "Task 2 graded. Maximum reward attained."
                    stdout_err += "\nMoving to Task 3: Fourier Analysis of Planted Features."
                    stdout_err += "\nCompute the DFT of model.W_E.weight across 97 token positions."
                    stdout_err += "\nFind the 5 frequency indices with highest total energy."
                elif 2 in submission:
                    # Partial credit: Agent narrowed it down but included false positives
                    reward = round(max(0.1, 1.0 / len(submission)), 2)
                    stdout_err = f"Task 2 graded. Partial credit ({reward:.2f}). You correctly identified the neuron, but included {len(submission)-1} false positives. Narrow it down!"
                else:
                    reward = 0.0
                    stdout_err = f"Task 2 incorrect. You submitted {submission}. Expected the specific causal neuron index."

            elif self.task_level == 3:
                gt = [2, 17, 23, 44, 47]
                if not isinstance(submission, list) or len(submission) != len(gt):
                    reward = 0.0
                    stdout_err = f"Task 3 failed. Expected exactly 5 frequency indices, got {len(submission) if isinstance(submission, list) else 'non-list'}."
                else:
                    try:
                        mse = sum((s - g) ** 2 for s, g in zip(sorted(submission), sorted(gt))) / len(gt)
                        reward = round(max(0.0, 1.0 - mse), 4)
                        stdout_err = f"Task 3 graded. MSE={mse:.4f}, Score: {reward}"
                        if reward >= 0.99:
                            done = True
                            stdout_err += "\nAll tasks completed."
                    except Exception:
                        reward = 0.0
                        stdout_err = "Error calculating score. Ensure all values are numeric."

        return MechInterpObservation(
            stdout_or_error=stdout_err,
            task_level=self.task_level,
            done=done,
            reward=reward,
            metadata={"step": self._state.step_count},
        )

    @property
    def state(self) -> InterpState:
        return self._state
