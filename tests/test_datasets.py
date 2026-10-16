"""Metadata and split checks with synthetic fixtures, no robot or GPU required."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from washstand_vla.datasets import inspect_dataset, make_episode_split


class InspectDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.meta = self.root / "meta"
        self.meta.mkdir()
        self.info = {
            "codebase_version": "v2.1", "fps": 30, "total_episodes": 2,
            "features": {
                "action": {"dtype": "float32", "shape": [6]},
                "observation.state": {"dtype": "float32", "shape": [6]},
                "observation.images.top": {"dtype": "video", "shape": [3, 480, 640]},
                "observation.images.wrist": {"dtype": "video", "shape": [3, 480, 640]},
            },
        }
        self.stats = {name: {"q01": [0.0] * 6, "q99": [1.0] * 6}
                      for name in ("action", "observation.state")}
        self.write_json("info.json", self.info)
        self.write_json("stats.json", self.stats)
        (self.meta / "tasks.jsonl").write_text('{"task_index":0,"task":"red brush to left cup"}\n', encoding="utf-8")
        (self.meta / "episodes.jsonl").write_text('{"episode_index":0,"length":30}\n{"episode_index":1,"length":30}\n', encoding="utf-8")

    def write_json(self, name, value):
        (self.meta / name).write_text(json.dumps(value), encoding="utf-8")

    def test_v21_metadata_is_read_without_claiming_full_validation(self):
        before = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = inspect_dataset(self.root)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["summary"]["format"], "lerobot-v2")
        self.assertEqual(result["summary"]["action_dim"], 6)
        self.assertEqual(result["summary"]["state_dim"], 6)
        self.assertEqual(result["summary"]["jsonl_episode_count"], 2)
        self.assertEqual(result["summary"]["top_image_features"], ["observation.images.top"])
        self.assertTrue(result["summary"]["quantile_keys"]["action"]["q01"])
        self.assertFalse(result["summary"]["video_decoded"])
        self.assertFalse(result["summary"]["temporal_alignment_verified"])
        self.assertTrue(any("alignment" in warning for warning in result["warnings"]))
        after = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_v3_parquet_is_only_discovered(self):
        self.info["codebase_version"] = "v3.0"
        self.write_json("info.json", self.info)
        (self.meta / "episodes.jsonl").unlink()
        (self.meta / "tasks.jsonl").unlink()
        parquet = self.meta / "episodes" / "chunk-000" / "file-000.parquet"
        parquet.parent.mkdir(parents=True)
        parquet.write_bytes(b"not a real parquet file; inspection must not decode it")
        (self.meta / "tasks.parquet").write_bytes(b"unread")
        result = inspect_dataset(self.root)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["summary"]["format"], "lerobot-v3")
        self.assertEqual(result["summary"]["episode_metadata_files"], ["meta/episodes/chunk-000/file-000.parquet"])
        self.assertIsNone(result["summary"]["jsonl_episode_count"])
        self.assertFalse(result["summary"]["parquet_rows_verified"])
        self.assertTrue(any("discovered only" in item for item in result["warnings"]))

    def test_missing_required_metadata_has_clear_errors(self):
        (self.meta / "stats.json").unlink()
        result = inspect_dataset(self.root)
        self.assertEqual(result["status"], "error")
        self.assertTrue(any("stats.json" in item and "missing" in item for item in result["errors"]))

    def test_missing_directory_does_not_create_files(self):
        missing = self.root / "absent"
        result = inspect_dataset(missing)
        self.assertEqual(result["status"], "error")
        self.assertFalse(missing.exists())

    def test_malformed_info_and_jsonl_include_filename_and_line(self):
        (self.meta / "info.json").write_text("{invalid", encoding="utf-8")
        (self.meta / "episodes.jsonl").write_text('{"episode_index":0}\n{bad}\n', encoding="utf-8")
        result = inspect_dataset(self.root)
        self.assertTrue(any("info.json" in item and "parse" in item for item in result["errors"]))
        self.assertTrue(any("episodes.jsonl" in item and "line 2" in item for item in result["errors"]))

    def test_malformed_stats_and_non_object_json_are_errors(self):
        self.write_json("info.json", [])
        (self.meta / "stats.json").write_text("{bad", encoding="utf-8")
        result = inspect_dataset(self.root)
        self.assertEqual(len(result["errors"]), 2)

    def test_wrong_dimensions_and_quantile_lengths_are_errors(self):
        self.info["features"]["action"]["shape"] = [7]
        self.write_json("info.json", self.info)
        result = inspect_dataset(self.root)
        self.assertEqual(result["summary"]["action_dim"], 7)
        self.assertTrue(any("expects 6" in item for item in result["errors"]))
        self.assertTrue(any("action.q01" in item for item in result["errors"]))

    def test_missing_quantiles_and_camera_are_warnings(self):
        del self.stats["action"]["q01"]
        del self.info["features"]["observation.images.wrist"]
        self.write_json("info.json", self.info)
        self.write_json("stats.json", self.stats)
        result = inspect_dataset(self.root)
        self.assertEqual(result["errors"], [])
        self.assertFalse(result["summary"]["quantile_keys"]["action"]["q01"])
        self.assertTrue(any("q01" in item for item in result["warnings"]))
        self.assertTrue(any("wrist" in item for item in result["warnings"]))


class EpisodeSplitTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [
            {"episode_id": f"{group}-{index}", "session_id": group}
            for group, count in (("morning", 3), ("afternoon", 2), ("evening", 1))
            for index in range(count)
        ]

    def test_split_preserves_complete_groups_and_is_input_order_independent(self):
        original = copy.deepcopy(self.episodes)
        result = make_episode_split(self.episodes, 0.33, 42)
        self.assertEqual(result, make_episode_split(list(reversed(self.episodes)), 0.33, 42))
        train = set(result["train"])
        validation = set(result["validation"])
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(train.isdisjoint(validation))
        self.assertEqual(train | validation, {episode["episode_id"] for episode in self.episodes})
        for group in {episode["session_id"] for episode in self.episodes}:
            members = {episode["episode_id"] for episode in self.episodes if episode["session_id"] == group}
            self.assertTrue(members <= train or members <= validation)
        self.assertEqual(original, self.episodes)

    def test_mixed_string_and_integer_ids_are_not_coerced(self):
        episodes = [{"episode_id": 1, "layout": "a"}, {"episode_id": "1", "layout": "b"}]
        result = make_episode_split(episodes, 0.5, 7, "layout")
        identifiers = result["train"] + result["validation"]
        self.assertIn(1, identifiers)
        self.assertIn("1", identifiers)
        self.assertEqual(result["counts"]["episodes"], 2)

    def test_duplicate_or_missing_id_or_group_is_rejected(self):
        cases = [
            ([{"session_id": "a"}], "episode_id"),
            ([{"episode_id": "x"}], "group"),
            ([{"episode_id": "x", "session_id": "a"}, {"episode_id": "x", "session_id": "b"}], "Duplicate"),
            ([{"episode_id": None, "session_id": "a"}], "episode_id"),
            ([{"episode_id": "x", "session_id": ""}], "session_id"),
            ([{"episode_id": True, "session_id": "a"}], "episode_id"),
        ]
        for episodes, message in cases:
            with self.subTest(episodes=episodes), self.assertRaisesRegex(ValueError, message):
                make_episode_split(episodes, 0.2, 0)

    def test_one_group_cannot_be_split(self):
        with self.assertRaisesRegex(ValueError, "two distinct groups"):
            make_episode_split([{"episode_id": 1, "session_id": "only"}], 0.2, 0)

    def test_invalid_fractions_and_seed_are_rejected(self):
        for fraction in (0, 1, -0.2, float("nan"), float("inf"), True, "0.2"):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                make_episode_split(self.episodes, fraction, 0)
        with self.assertRaisesRegex(ValueError, "seed"):
            make_episode_split(self.episodes, 0.2, True)

    def test_uneven_groups_report_actual_episode_fraction(self):
        result = make_episode_split(self.episodes, 0.5, 10)
        self.assertEqual(result["actual_validation_fraction"], len(result["validation"]) / len(self.episodes))
        self.assertEqual(len(result["groups"]["validation"]), 2)
        self.assertTrue(any("Instruction balance" in note for note in result["notes"]))


if __name__ == "__main__":
    unittest.main()
