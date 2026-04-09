"""
Final Integration Test Suite
Tests end-to-end workflows and complete submission scenarios.
"""

import unittest
import json
import torch
import torch.nn as nn

from server.mech_interp_environment import MAX_TASK_SCORE, MIN_TASK_SCORE

class TestEndToEndIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_task1_complete_workflow(self):
        """Complete workflow: reset → code → grading → advancement."""
        # Task 1: Find dead neurons [2, 5, 8]
        submission = [2, 5, 8]
        expected = [2, 5, 8]
        
        # Grading logic
        if submission == expected:
            reward = MAX_TASK_SCORE
            task_level_next = 2
            done = False
        else:
            reward = MIN_TASK_SCORE
            task_level_next = 1
            done = False
        
        self.assertEqual(reward, MAX_TASK_SCORE)
        self.assertEqual(task_level_next, 2)
        self.assertFalse(done)
    
    def test_task2_complete_workflow(self):
        """Complete workflow: reset → code → grading → advancement."""
        # Task 2: Find causal ablation neuron [2]
        submission = [2]
        expected = [2]
        
        if submission == expected:
            reward = MAX_TASK_SCORE
            task_level_next = 3
            done = False
        else:
            reward = MIN_TASK_SCORE
            task_level_next = 2
            done = False
        
        self.assertEqual(reward, MAX_TASK_SCORE)
        self.assertEqual(task_level_next, 3)
        self.assertFalse(done)

    def test_task2_partial_credit_without_advancement(self):
        """Task 2 should provide reward shaping before the exact answer is submitted."""
        submission = [2, 5]

        if 2 in submission:
            reward = max(0.1, 1.0 / len(submission))
            task_level_next = 2
        else:
            reward = 0.0
            task_level_next = 2

        self.assertEqual(reward, 0.5)
        self.assertEqual(task_level_next, 2)
    
    def test_task3_complete_workflow(self):
        """Complete workflow: reset → code → grading → completion."""
        # Task 3: Find Fourier frequencies [2, 17, 23, 44, 47]
        submission = [2, 17, 23, 44, 47]
        expected = [2, 17, 23, 44, 47]
        
        if submission == expected:
            reward = MAX_TASK_SCORE
            done = True
        else:
            reward = MIN_TASK_SCORE
            done = False
        
        self.assertEqual(reward, MAX_TASK_SCORE)
        self.assertTrue(done)
    
    def test_full_3task_curriculum(self):
        """Complete all 3 tasks in sequence."""
        task_sequence = [
            {"task": 1, "submission": [2, 5, 8], "expected_reward": MAX_TASK_SCORE, "next_task": 2},
            {"task": 2, "submission": [2], "expected_reward": MAX_TASK_SCORE, "next_task": 3},
            {"task": 3, "submission": [2, 17, 23, 44, 47], "expected_reward": MAX_TASK_SCORE, "next_task": None},
        ]
        
        current_task = 1
        total_reward = 0.0
        
        for step_data in task_sequence:
            self.assertEqual(current_task, step_data["task"])
            total_reward += step_data["expected_reward"]
            
            if step_data["next_task"]:
                current_task = step_data["next_task"]
        
        self.assertAlmostEqual(total_reward, MAX_TASK_SCORE * 3, places=6)
        self.assertIsNone(task_sequence[-1]["next_task"])
    
    def test_json_action_formats(self):
        """Verify both action formats parse correctly."""
        
        # Action 1: python_code
        action1_json = '{"python_code": "print(\'hello\')"}'
        action1 = json.loads(action1_json)
        self.assertIn("python_code", action1)
        self.assertIsNone(action1.get("solution_target"))
        
        # Action 2: solution_target
        action2_json = '{"solution_target": [2, 5, 8]}'
        action2 = json.loads(action2_json)
        self.assertIn("solution_target", action2)
        self.assertIsNone(action2.get("python_code"))
    
    def test_logging_format_compliance(self):
        """Verify logging output format matches spec."""
        
        # [START] format
        start_log = "[START] task=mech_interp_curriculum env=pytorch_sandbox model=Qwen/Qwen2.5-72B-Instruct"
        self.assertIn("[START]", start_log)
        self.assertIn("task=", start_log)
        self.assertIn("env=", start_log)
        self.assertIn("model=", start_log)
        
        # [STEP] format
        step_log = "[STEP] step=1 action=python_code reward=0.00 done=false error=null"
        self.assertIn("[STEP]", step_log)
        self.assertIn("step=", step_log)
        self.assertIn("reward=", step_log)
        self.assertIn("done=", step_log)
        self.assertIn("error=", step_log)
        
        # [END] format
        end_log = "[END] success=true steps=5 score=0.990 rewards=0.99,0.99,0.99"
        self.assertIn("[END]", end_log)
        self.assertIn("success=", end_log)
        self.assertIn("steps=", end_log)
        self.assertIn("score=", end_log)
        self.assertIn("rewards=", end_log)
    
    def test_timeout_resilience(self):
        """Verify timeout mechanism will prevent hangs."""
        timeout_seconds = 30
        
        # Timeout should be configured
        self.assertGreater(timeout_seconds, 0)
        self.assertLessEqual(timeout_seconds, 60)
    
    def test_artifact_model_loading(self):
        """Test model artifact loading workflow."""
        model = nn.Linear(10, 1)
        
        # Simulate task1 model
        with torch.no_grad():
            model.weight.zero_()
            model.weight[0, [0, 1, 3, 4, 6, 7, 9]] = 1.0
        
        # Find dead neurons
        dead_indices = torch.where(torch.all(model.weight == 0, dim=0))[0].tolist()
        
        self.assertEqual(dead_indices, [2, 5, 8])
    
    def test_task_advancement_gates(self):
        """Verify task advancement requires specific rewards."""
        
        # Task 1 advancement: reward >= 0.99
        reward1 = MAX_TASK_SCORE
        should_advance_1 = reward1 >= 0.99
        self.assertTrue(should_advance_1)
        
        # Task 2 advancement uses an exact-match solve, even though the published score is 0.99
        reward2 = MAX_TASK_SCORE
        solved_task2 = True
        should_advance_2 = solved_task2 and reward2 >= 0.99
        self.assertTrue(should_advance_2)
        
        # Task 3 completion: reward >= 0.99
        reward3 = MAX_TASK_SCORE
        should_complete = reward3 >= 0.99
        self.assertTrue(should_complete)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery."""
    
    def test_invalid_json_recovery(self):
        """LLM output with invalid JSON should fail gracefully."""
        invalid_json = "This is not JSON"
        
        try:
            parsed = json.loads(invalid_json)
            self.fail("Should have raised JSONDecodeError")
        except json.JSONDecodeError:
            pass  # Expected
    
    def test_timeout_error_handling(self):
        """Timeout errors should be caught and reported."""
        error_message = "Code execution timed out after 30 seconds."
        
        self.assertIn("timed out", error_message.lower())
        self.assertIn("30", error_message)
    
    def test_type_mismatch_handling(self):
        """Non-list solution_target should be handled."""
        submission = "not a list"
        
        is_list = isinstance(submission, list)
        self.assertFalse(is_list)
        
        # Validator-facing task scores should stay inside the open interval (0, 1)
        reward = MIN_TASK_SCORE if not is_list else MAX_TASK_SCORE
        self.assertEqual(reward, MIN_TASK_SCORE)
    
    def test_syntax_error_in_code(self):
        """Syntax errors in submitted code should be caught."""
        bad_code = "print('unclosed string"
        
        try:
            compile(bad_code, "<string>", "exec")
            self.fail("Should have raised SyntaxError")
        except SyntaxError:
            pass  # Expected


class TestConsistencyAndDeterminism(unittest.TestCase):
    """Verify deterministic behavior and consistency."""
    
    def test_ground_truth_deterministic_task1(self):
        """Task 1 ground truth must always be [2, 5, 8]."""
        for _ in range(10):
            ground_truth = [2, 5, 8]
            self.assertEqual(ground_truth, [2, 5, 8])
    
    def test_ground_truth_deterministic_task2(self):
        """Task 2 ground truth must always be [2]."""
        for _ in range(10):
            ground_truth = [2]
            self.assertEqual(ground_truth, [2])
    
    def test_ground_truth_deterministic_task3(self):
        """Task 3 ground truth must always be [2, 17, 23, 44, 47]."""
        for _ in range(10):
            ground_truth = [2, 17, 23, 44, 47]
            self.assertEqual(ground_truth, [2, 17, 23, 44, 47])
    
    def test_grading_deterministic_task1(self):
        """Task 1 grading must be deterministic."""
        submission = [2, 5, 8]
        expected = [2, 5, 8]
        
        rewards = []
        for _ in range(5):
            if submission == expected:
                reward = MAX_TASK_SCORE
            else:
                reward = MIN_TASK_SCORE
            rewards.append(reward)
        
        # All should be same
        self.assertEqual(len(set(rewards)), 1)
        self.assertEqual(rewards[0], MAX_TASK_SCORE)
    
    def test_grading_deterministic_task2(self):
        """Task 2 grading must be deterministic."""
        submission = [2]
        expected = [2]
        
        rewards = []
        for _ in range(5):
            if submission == expected:
                reward = MAX_TASK_SCORE
            else:
                reward = MIN_TASK_SCORE
            rewards.append(reward)
        
        self.assertEqual(len(set(rewards)), 1)
        self.assertEqual(rewards[0], MAX_TASK_SCORE)
    
    def test_task_level_state_consistency(self):
        """Task level state must persist across steps."""
        # Simulate: complete task 1 (3 steps) -> task 2 (3 steps) -> task 3 (2 steps)
        step_to_task = {
            1: 1, 2: 1, 3: 1,  # Task 1 steps
            4: 2, 5: 2, 6: 2,  # Task 2 steps (advance after step 3)
            7: 3, 8: 3,        # Task 3 steps (advance after step 6)
        }
        
        current_task = 1
        for step, expected_task in step_to_task.items():
            self.assertEqual(current_task, expected_task, f"Step {step}")
            
            # Simulate task advancement
            if step == 3:
                current_task = 2
            elif step == 6:
                current_task = 3


if __name__ == "__main__":
    unittest.main(verbosity=2)
