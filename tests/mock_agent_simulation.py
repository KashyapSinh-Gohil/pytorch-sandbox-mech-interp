import asyncio
import os
import sys

# Ensure we can import the local client and models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client import MechInterpEnv
from models import MechInterpAction

async def run_correct_agent():
    print("--- Starting Correct Mock Agent Test ---")
    env = MechInterpEnv(base_url="http://localhost:8000")
    
    try:
        # 1. Reset
        print("\n[Action] Resetting environment...")
        result = await env.reset()
        obs = result.observation
        print(f"[Observation] Task Level: {obs.task_level}")
        print(f"[Observation] Output: {obs.stdout_or_error}")

        # 2. Solve Task 1
        print("\n[Action] Submitting solution for Task 1: [2, 5, 8]")
        action = MechInterpAction(solution_target=[2, 5, 8])
        result = await env.step(action)
        obs = result.observation
        print(f"[Result] Reward: {result.reward}, Done: {result.done}")
        print(f"[Observation] Task Level: {obs.task_level}")
        print(f"[Observation] Output: {obs.stdout_or_error}")

        # 3. Solve Task 2
        print("\n[Action] Submitting solution for Task 2: [2]")
        action = MechInterpAction(solution_target=[2])
        result = await env.step(action)
        obs = result.observation
        print(f"[Result] Reward: {result.reward}, Done: {result.done}")
        print(f"[Observation] Task Level: {obs.task_level}")
        print(f"[Observation] Output: {obs.stdout_or_error}")

        # 4. Solve Task 3
        print("\n[Action] Submitting solution for Task 3: [2, 17, 23, 44, 47]")
        action = MechInterpAction(solution_target=[2, 17, 23, 44, 47])
        result = await env.step(action)
        obs = result.observation
        print(f"[Result] Reward: {result.reward}, Done: {result.done}")
        print(f"[Observation] Task Level: {obs.task_level}")
        print(f"[Observation] Output: {obs.stdout_or_error}")

        if result.done and obs.task_level == 3 and result.reward >= 0.99:
            print("\n--- TEST PASSED: Correct Curriculum Completion ---")
        else:
            print("\n--- TEST FAILED: Curriculum not completed as expected ---")

    except Exception as e:
        print(f"\n[Error] Test failed with exception: {e}")
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(run_correct_agent())
