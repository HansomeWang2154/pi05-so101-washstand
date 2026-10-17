"""Optional simulation data checks; no model/robot/GPU imports or motion.

NumPy/PyAV are optional here. The core preparation tool suite still works in a
standard-library-only installation and skips these optional checks.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sim.washstand.export_lerobot import export_episodes, qualification_errors, source_entry
from sim.washstand.inspect_dataset import expected_window_indices, preview_indices
from sim.washstand.recording import EpisodeRecorder, load_episode, validate_episode


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_AV = importlib.util.find_spec("av") is not None


class InspectionReferenceTests(unittest.TestCase):
    def test_action_window_is_clipped_to_current_episode(self):
        indices, padding = expected_window_indices(2, 3, [0, 1, 2])
        self.assertEqual(indices, [2, 2, 2])
        self.assertEqual(padding, [False, True, True])
        self.assertEqual(expected_window_indices(0, 3, [-1, 0]), ([0, 0], [True, False]))

    def test_preview_contains_first_grasp_release_and_last(self):
        selected = preview_indices(["PREGRASP", "CLOSE", "CLOSE", "LIFT", "RELEASE", "SETTLE"])
        self.assertEqual(selected, {0: "first", 2: "grasp", 4: "release", 5: "last"})

    def test_source_manifest_keeps_environment_provenance(self):
        metadata = {
            "instruction": "synthetic", "frame_count": 4, "layout_id": "layout-a", "group_id": "group-a",
            "xml_sha256": "xml-sha", "asset_sha256": {"mesh": "asset-sha"}, "project_commit": "commit",
            "working_tree_dirty": True, "expert_version": "expert-v1", "scene_schema": 1,
            "task_id": "red_left", "config": {"layout_jitter_m": 0},
        }
        entry = source_entry({"root": Path("/tmp/synthetic"), "metadata": metadata}, 2)
        for field in metadata:
            self.assertEqual(entry[field], metadata[field])
        self.assertEqual(entry["episode_index"], 2)
        self.assertEqual(entry["raw_path"], "/tmp/synthetic")
        self.assertNotIn("model_hash", entry)


@unittest.skipUnless(HAS_NUMPY, "Optional simulation NumPy dependency not installed")
class RawRecordingTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def recorder(self, name="attempt", **kwargs):
        return EpisodeRecorder(self.root / name,
                               {"instruction": "Put the red toothbrush into the left cup.",
                                "layout_id": "synthetic-layout", "evidence_kind": "synthetic_unit_fixture"},
                               {"integration_state": self.np.zeros(30), "state_spec": 1},
                               width=32, height=32, **kwargs)

    def append(self, recorder, timestamp, image=None, state=None):
        self.state = self.np.arange(6, dtype=float) if state is None else state
        recorder.append(timestamp, self.state, self.state + 0.1,
                        self.np.zeros(20), self.np.zeros(18), image, image, "synthetic")

    def test_native_action_is_sent_target_and_arrays_are_copied(self):
        recorder = self.recorder()
        state = self.np.arange(6, dtype=float)
        self.append(recorder, 0, state=state)
        state[:] = 100
        self.append(recorder, 1 / 30)
        metadata = recorder.finish({"physics_success": False, "training_eligible": True})
        episode = load_episode(recorder.root)
        self.np.testing.assert_array_equal(episode["arrays"]["state"][0], self.np.arange(6))
        self.np.testing.assert_allclose(episode["arrays"]["action"][0], self.np.arange(6) + 0.1)
        self.assertFalse(metadata["training_eligible"])
        self.assertFalse(metadata["real_contract_verified"])
        self.assertFalse(metadata["images_present"])
        self.assertTrue(metadata["data_checks_passed"])
        self.assertEqual(validate_episode(episode, check_videos=False), [])
        self.assertEqual(recorder.finish({}), metadata)

    def test_fps_mismatch_cannot_be_exported(self):
        recorder = self.recorder()
        self.append(recorder, 0)
        self.append(recorder, 0.1)
        metadata = recorder.finish({"data_checks_passed": True})
        self.assertFalse(metadata["data_checks_passed"])
        self.assertIn("Timestamps", metadata["data_check_errors"][0])

    def test_bad_joint_value_and_timestamp_are_rejected_before_write(self):
        recorder = self.recorder()
        with self.assertRaisesRegex(ValueError, "finite"):
            self.append(recorder, 0, state=self.np.full(6, self.np.nan))
        self.append(recorder, 0)
        with self.assertRaisesRegex(ValueError, "increase"):
            self.append(recorder, 0)
        recorder.finish({"physics_success": False})

    def test_failed_debug_episode_does_not_create_export(self):
        recorder = self.recorder()
        self.append(recorder, 0)
        self.append(recorder, 1 / 30)
        recorder.finish({"physics_success": False})
        with self.assertRaisesRegex(ValueError, "No qualified episodes"):
            export_episodes([recorder.root], self.root / "export")
        self.assertFalse((self.root / "export").exists())

    def test_never_overwrites_an_existing_attempt(self):
        recorder = self.recorder()
        recorder.finish({})
        with self.assertRaises(FileExistsError):
            self.recorder()

    @unittest.skipUnless(HAS_AV, "Optional simulation PyAV dependency not installed")
    def test_dual_stream_video_roundtrip_and_metadata_revalidation(self):
        recorder = self.recorder()
        for index in range(4):
            self.append(recorder, index / 30, self.np.full((32, 32, 3), index * 30, dtype=self.np.uint8))
        recorder.finish({"physics_success": True, "replay_passed": True,
                         "evidence_kind": "synthetic_unit_fixture_not_expert"})
        episode = load_episode(recorder.root)
        self.assertEqual(qualification_errors(episode), [])
        episode["arrays"]["action"][1, 0] = self.np.nan
        self.assertTrue(any("action" in item for item in qualification_errors(episode)))
        diagnostics = (recorder.root / "diagnostics.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(row)["frame_index"] for row in diagnostics], list(range(4)))

    @unittest.skipUnless(HAS_AV, "Optional simulation PyAV dependency not installed")
    def test_no_missing_or_mixed_camera_frames(self):
        recorder = self.recorder()
        image = self.np.zeros((32, 32, 3), dtype=self.np.uint8)
        self.append(recorder, 0)
        with self.assertRaisesRegex(ValueError, "mix"):
            self.append(recorder, 1 / 30, image)
        with self.assertRaisesRegex(ValueError, "together"):
            recorder.append(1 / 30, self.np.zeros(6), self.np.zeros(6), self.np.zeros(20),
                            self.np.zeros(18), image, None, "bad")
        recorder.finish({})


if __name__ == "__main__":
    unittest.main()
