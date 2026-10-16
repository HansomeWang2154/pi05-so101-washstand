import copy
import json
import tempfile
import unittest
from pathlib import Path

from washstand_vla.cli import write_result
from washstand_vla.config import merge_dicts, read_json, validate_config
from washstand_vla.planning import build_plan

ROOT = Path(__file__).resolve().parents[1]


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / "configs/project.json")

    def test_template_valid_but_devices_incomplete(self):
        result = validate_config(self.config)
        self.assertTrue(result["valid"])
        self.assertFalse(result["complete"])
        self.assertIn("robot.port", result["missing"])

    def test_reject_bad_action_dim_and_camera_fps(self):
        self.config["policy"]["action_dim"] = 7
        self.config["cameras"]["top"]["fps"] = 10
        self.assertFalse(validate_config(self.config)["valid"])

    def test_nested_override_keeps_other_camera_fields(self):
        original = copy.deepcopy(self.config)
        merged = merge_dicts(self.config, {"cameras": {"top": {"index_or_path": 2}}})
        self.assertEqual(merged["cameras"]["top"]["width"], 640)
        self.assertEqual(merged["cameras"]["top"]["index_or_path"], 2)
        self.assertEqual(self.config, original)

    def test_reject_same_camera_and_serial_ids(self):
        self.config["robot"]["port"] = "/dev/example"
        self.config["robot"]["leader_port"] = "/dev/example"
        for camera in self.config["cameras"].values():
            camera["index_or_path"] = 0
        self.assertFalse(validate_config(self.config)["valid"])

    def test_reject_missing_required_config_fields(self):
        del self.config["robot"]["id"]
        self.assertFalse(validate_config(self.config)["valid"])

    def test_malformed_joint_names_are_reported_without_crashing(self):
        self.config["robot"]["joint_names"] = [{"bad": 1}] * 6
        self.assertFalse(validate_config(self.config)["valid"])

    def test_plan_includes_configured_chunk_and_module_entrypoint(self):
        result = build_plan(self.config, "train", ROOT)
        self.assertEqual(result["argv"][:3], ["python", "-m", "lerobot.scripts.lerobot_train"])
        self.assertIn("--policy.chunk_size=50", result["argv"])

    def test_train_plan_blocks_unverified_stats_and_split(self):
        result = build_plan(self.config, "train", ROOT)
        self.assertFalse(result["execution_supported"])
        self.assertIn("data.split_path", result["missing"])
        self.assertIn("data.train_statistics_verified", result["missing"])

    def test_train_plan_uses_only_train_episode_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"train": [0, 1], "validation": [2]}))
            result = build_plan(self.config, "train", ROOT, split_path=path)
            self.assertIn("--dataset.episodes=[0,1]", result["argv"])
            self.assertNotIn("--dataset.episodes=[0,1,2]", result["argv"])

    def test_reject_split_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"train": [0, 1], "validation": [1]}))
            with self.assertRaises(ValueError):
                build_plan(self.config, "train", ROOT, split_path=path)

    def test_all_plans_are_non_executing(self):
        for stage in ("record", "train", "server", "client"):
            self.assertIs(build_plan(self.config, stage, ROOT)["execution_supported"], False)

    def test_output_does_not_overwrite_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("original")
            with self.assertRaises(ValueError):
                write_result({"new": True}, path)
            self.assertEqual(path.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
