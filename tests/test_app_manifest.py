import importlib
import json
import os
import sys
import unittest

from fastapi.testclient import TestClient
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.app import app


class TestAppManifest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_info_exposes_four_tasks_with_graders(self):
        response = self.client.get("/info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_count"], 4)
        self.assertEqual(len(payload["tasks"]), 4)
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
                "additive_bypass_attribution_grader",
            ],
        )
        reset_payloads = [task["reset_payload"]["task_id"] for task in payload["tasks"]]
        self.assertEqual(reset_payloads, ["task1", "task2", "task3", "task4"])

    def test_metadata_mentions_four_graded_tasks(self):
        response = self.client.get("/metadata")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "mech_interp")
        self.assertIn("four graded tasks", payload["description"])

    def test_openenv_manifest_declares_four_task_graders(self):
        manifest_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openenv.yaml")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)

        self.assertEqual(len(manifest["tasks"]), 4)
        grader_paths = [
            f"{task['grader']['module']}:{task['grader']['function']}"
            for task in manifest["tasks"]
        ]
        self.assertEqual(
            grader_paths,
            [
                "tasks.task1.grader:grade",
                "tasks.task2.grader:grade",
                "tasks.task3.grader:grade",
                "tasks.task4.grader:grade",
            ],
        )
        self.assertEqual(manifest["grader"]["module"], "tasks.graders")
        self.assertEqual(manifest["grader"]["function"], "grade_task")

    def test_manifest_grader_entrypoints_are_importable(self):
        manifest_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openenv.yaml")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)

        for task in manifest["tasks"]:
            module_name = task["grader"]["module"]
            func_name = task["grader"]["function"]
            module = importlib.import_module(module_name)
            grader = getattr(module, func_name)
            self.assertTrue(callable(grader))

        top_module = importlib.import_module(manifest["grader"]["module"])
        top_grader = getattr(top_module, manifest["grader"]["function"])
        self.assertTrue(callable(top_grader))

    def test_static_tasks_json_matches_four_task_shape(self):
        manifest_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks.json")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.assertEqual(manifest["task_count"], 4)
        self.assertEqual(manifest["grader_count"], 4)
        self.assertEqual(len(manifest["tasks"]), 4)
        self.assertTrue(all(task["has_grader"] for task in manifest["tasks"]))
        self.assertEqual(
            [task["reset_payload"]["task_id"] for task in manifest["tasks"]],
            ["task1", "task2", "task3", "task4"],
        )


if __name__ == "__main__":
    unittest.main()
