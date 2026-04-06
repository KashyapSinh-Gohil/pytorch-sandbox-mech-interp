import unittest
import sys
import os
import time

# Ensure we can import models and server from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MechInterpAction
from server.mech_interp_environment import MechInterpEnvironment


class TestMechInterpStress(unittest.TestCase):
    def setUp(self):
        self.env = MechInterpEnvironment()

    def test_infinite_loop_timeout(self):
        """Verify that an infinite loop is killed within 30s + grace."""
        self.env.reset()
        # Note: In real testing we might want a shorter timeout for the test suite,
        # but here we test the real EXEC_TIMEOUT (30s).
        action = MechInterpAction(python_code="while True: pass")
        
        start_time = time.time()
        obs = self.env.step(action)
        end_time = time.time()
        
        # Check that it didn't hang forever
        self.assertLess(end_time - start_time, 35) # 30s + 5s grace
        self.assertIn("ERROR: Code execution timed out after 30 seconds", obs.stdout_or_error)

    def test_syntax_and_runtime_errors(self):
        """Verify the sandbox catches and reports various Python errors."""
        self.env.reset()
        
        # 1. Syntax Error
        action = MechInterpAction(python_code="  print(1)") # IndentationError
        obs = self.env.step(action)
        self.assertIn("IndentationError", obs.stdout_or_error)
        
        # 2. Runtime Error (ZeroDivision)
        action = MechInterpAction(python_code="x = 1 / 0")
        obs = self.env.step(action)
        self.assertIn("ZeroDivisionError", obs.stdout_or_error)
        
        # 3. Name Error
        action = MechInterpAction(python_code="print(undefined_variable)")
        obs = self.env.step(action)
        self.assertIn("NameError", obs.stdout_or_error)

    def test_hook_cleanup_integrity(self):
        """Verify that hooks are cleared after every step to prevent memory leaks."""
        self.env.reset()
        # Task 2 has model with hooks support
        self.env.task_level = 2
        
        # Step 1: Add a hook
        code = (
            "def hook_fn(module, input, output): return output\n"
            "handle = model.hidden.register_forward_hook(hook_fn)\n"
            "print(f'Hooks before: {len(model.hidden._forward_hooks)}')"
        )
        obs = self.env.step(MechInterpAction(python_code=code))
        self.assertIn("Hooks before: 1", obs.stdout_or_error)
        
        # Step 2: Check if hook was cleared by the environment guard
        code_check = "print(f'Hooks after cleanup: {len(model.hidden._forward_hooks)}')"
        obs = self.env.step(MechInterpAction(python_code=code_check))
        self.assertIn("Hooks after cleanup: 0", obs.stdout_or_error)

    def test_stdout_buffer_stress(self):
        """Verify the environment handles large stdout streams (1MB)."""
        self.env.reset()
        large_string_size = 1024 * 1024 # 1MB
        code = f"print('A' * {large_string_size})"
        
        obs = self.env.step(MechInterpAction(python_code=code))
        # Capturing stdout should not crash or truncate significantly
        self.assertEqual(len(obs.stdout_or_error.strip()), large_string_size)

    def test_state_persistence_and_step_counting(self):
        """Verify episode state is preserved accurately across complex steps."""
        self.env.reset()
        self.assertEqual(self.env.state.step_count, 0)
        
        # Take 5 actions
        for i in range(1, 6):
            self.env.step(MechInterpAction(python_code=f"print({i})"))
            self.assertEqual(self.env.state.step_count, i)
        
        # Check task level persists
        self.env.task_level = 2
        self.env.step(MechInterpAction(python_code="print('still level 2')"))
        self.assertEqual(self.env.task_level, 2)


if __name__ == '__main__':
    unittest.main()
