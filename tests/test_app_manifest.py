import os
import sys
import unittest

from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.app import app


class TestAppManifest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_info_exposes_three_tasks_with_graders(self):
        response = self.client.get("/info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_count"], 3)
        self.assertEqual(len(payload["tasks"]), 3)
        self.assertTrue(all("grader" in task for task in payload["tasks"]))

    def test_tasks_endpoint_matches_validator_shape(self):
        response = self.client.get("/tasks")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        grader_names = [task["grader"]["name"] for task in payload["tasks"]]
        self.assertEqual(
            grader_names,
            [
                "dead_neuron_detection_grader",
                "causal_ablation_grader",
                "fourier_frequency_recovery_grader",
            ],
        )

    def test_metadata_mentions_three_graded_tasks(self):
        response = self.client.get("/metadata")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "mech_interp")
        self.assertIn("three graded tasks", payload["description"])


if __name__ == "__main__":
    unittest.main()
