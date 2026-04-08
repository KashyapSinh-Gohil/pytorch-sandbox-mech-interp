import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MechInterpAction
from server.mech_interp_environment import MechInterpEnvironment


class TestMechInterpEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = MechInterpEnvironment()

    def test_environment_reset_and_initial_state(self):
        obs = self.env.reset()
        self.assertEqual(obs.task_level, 1)
        self.assertIn("Task 1", obs.stdout_or_error)
        self.assertNotIn(str(self.env.ground_truths["task1"]), obs.stdout_or_error)
        self.assertEqual(obs.reward, 0.0)
        self.assertFalse(obs.done)

    def test_code_execution_updates_observation(self):
        self.env.reset()
        action = MechInterpAction(python_code="print('hello test')")
        obs = self.env.step(action)
        self.assertIn("hello test", obs.stdout_or_error.strip())

    def test_task1_grading_logic(self):
        self.env.reset()
        
        # Wrong / Random guess
        obs = self.env.step(MechInterpAction(solution_target=[1, 2, 3]))
        self.assertLess(obs.reward, 1.0)
        self.assertEqual(obs.task_level, 1)
        
        # Correct guess
        obs = self.env.step(MechInterpAction(solution_target=[2, 5, 8]))
        self.assertEqual(obs.reward, 1.0)
        self.assertEqual(obs.task_level, 2)
        self.assertIn("Moving to Task 2", obs.stdout_or_error)

    def test_task2_grading_logic(self):
        self.env.reset()
        # Skip to task 2
        self.env.task_level = 2
        self.env._state.task_level = 2

        # Wrong guess
        obs = self.env.step(MechInterpAction(solution_target=[7]))
        self.assertEqual(obs.reward, 0.0)
        self.assertEqual(obs.task_level, 2)

        # False positives should not advance the task
        obs = self.env.step(MechInterpAction(solution_target=[2, 5]))
        self.assertEqual(obs.reward, 0.0)
        self.assertEqual(obs.task_level, 2)

        # Correct guess
        obs = self.env.step(MechInterpAction(solution_target=[2]))
        self.assertEqual(obs.reward, 1.0)
        self.assertEqual(obs.task_level, 3)

    def test_task3_grading_and_completion(self):
        self.env.reset()
        # Skip to task 3
        self.env.task_level = 3
        self.env._state.task_level = 3
        
        # Expected target: [2, 17, 23, 44, 47]
        obs = self.env.step(MechInterpAction(solution_target=[2, 17, 23, 44, 47]))
        self.assertEqual(obs.reward, 1.0)
        self.assertTrue(obs.done)
        self.assertIn("All tasks completed", obs.stdout_or_error)


if __name__ == '__main__':
    unittest.main()
