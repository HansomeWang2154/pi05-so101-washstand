"""Small optional pipeline regressions; no expert rollouts or rendering.

Short fixtures intentionally do not finish the task. Their purpose is to check
observation/action timing, failure preservation and rejection of damaged data.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim.washstand import WashstandEnv, load_config
from sim.washstand.export_lerobot import qualification_errors
from sim.washstand.recording import EpisodeRecorder, load_episode
from sim.washstand.replay import replay_episode
from sim.washstand.validators import numeric_health


class LayoutGroupingTests(unittest.TestCase):
    def test_zero_jitter_duplicate_seeds_and_instructions_share_group(self):
        from sim.washstand.generate import layout_group

        with WashstandEnv(seed=100, task_id="red_left") as first:
            identity = layout_group(first)
            offsets = first.layout_offsets
        with WashstandEnv(seed=101, task_id="blue_right") as second:
            self.assertEqual(offsets, second.layout_offsets)
            self.assertEqual(identity, layout_group(second))

    def test_actual_randomized_layouts_receive_distinct_groups(self):
        from sim.washstand.generate import layout_group

        config = load_config()
        config["layout_jitter_m"] = .001
        with WashstandEnv(seed=100, config=config) as first:
            identity = layout_group(first)
        with WashstandEnv(seed=101, config=config) as second:
            self.assertNotEqual(identity, layout_group(second))


class PhysicsMonitoringTests(unittest.TestCase):
    def test_transient_substep_penetration_is_rejected_after_contact_recovers(self):
        with WashstandEnv() as env:
            native_step = __import__("mujoco").mj_step
            calls = 0

            def transient_contact(model, data):
                nonlocal calls
                native_step(model, data)
                calls += 1
                if calls == 3:
                    # Fault injection only: mimic an 8mm contact overlap at
                    # one substep. Subsequent native steps recompute contacts;
                    # no qpos, velocity, force or control is modified.
                    self.assertGreater(data.ncon, 0)
                    data.contact[0].dist = -.008

            with patch("sim.washstand.env.mujoco.mj_step", side_effect=transient_contact):
                env.step_targets(env.config["home_targets_rad"])
            self.assertEqual(calls, env.config["substeps"])
            health = numeric_health(env)
            self.assertLess(health["current_penetration_m"], .006)
            self.assertAlmostEqual(health["maximum_penetration_m"], .008)
            self.assertFalse(health["ok"])

    def test_peak_monitor_is_snapshot_state_and_reset_clears_history(self):
        with WashstandEnv() as env:
            reset_peak = env.peak_penetration_m
            env.peak_penetration_m = .004
            snapshot = env.snapshot()
            self.assertEqual(snapshot["peak_penetration_m"], .004)
            env.peak_penetration_m = .009
            env.restore(snapshot)
            self.assertEqual(env.peak_penetration_m, .004)
            env.reset()
            self.assertAlmostEqual(env.peak_penetration_m, reset_peak)
            snapshot = env.snapshot()
            snapshot["peak_penetration_m"] = np.nan
            with self.assertRaises(ValueError):
                env.restore(snapshot)


class RecordingReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="washstand-pipeline-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "attempt"
        with WashstandEnv(seed=17) as env:
            metadata = env.metadata()
            metadata.update(instruction=env.task_text, evidence_kind="short_test_fixture_not_expert")
            recorder = EpisodeRecorder(self.path, metadata, env.snapshot(), fps=30)
            self.observed_qpos = []
            self.observed_state = []
            self.actions = []
            self.qpos_ids = env.qpos_ids.copy()
            start = float(env.data.time)
            for index in range(3):
                target = np.asarray(env.config["home_targets_rad"], dtype=np.float32)
                target[0] += np.float32(index * .001)
                self.observed_qpos.append(env.data.qpos.copy())
                self.observed_state.append(env.state())
                self.actions.append(target.copy())
                recorder.append(timestamp=env.data.time - start, state=env.state(), action=target,
                                qpos=env.data.qpos, qvel=env.data.qvel, top=None, wrist=None,
                                stage="SHORT_FIXTURE", diagnostics={"fixture": True})
                env.step_targets(target)
            recorder.finish({"physics_success": False, "failure_reason": "short_fixture_not_task_completion"})

    def damage(self, name):
        episode = load_episode(self.path)
        arrays = episode["arrays"]
        arrays[name][1, 0] += .1
        np.savez_compressed(self.path / "trajectory.npz", **arrays)

    def test_observations_are_copied_before_sent_controls_advance_physics(self):
        arrays = load_episode(self.path)["arrays"]
        np.testing.assert_array_equal(arrays["qpos"], self.observed_qpos)
        np.testing.assert_allclose(arrays["state"], self.observed_state, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(arrays["action"], self.actions)
        np.testing.assert_allclose(arrays["state"], arrays["qpos"][:, self.qpos_ids], atol=1e-7, rtol=0)
        np.testing.assert_allclose(arrays["timestamps"], [0, 1 / 30, 2 / 30], atol=1e-12, rtol=0)

    def test_failure_data_remains_available_but_cannot_export(self):
        episode = load_episode(self.path)
        self.assertFalse(episode["metadata"]["physics_success"])
        self.assertEqual(episode["metadata"]["failure_reason"], "short_fixture_not_task_completion")
        self.assertEqual(episode["arrays"]["state"].shape, (3, 6))
        diagnostics = [json.loads(line) for line in (self.path / "diagnostics.jsonl").read_text().splitlines()]
        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(qualification_errors(episode))

    def test_replay_has_no_expert_feedback_and_short_fixture_is_not_success(self):
        with patch("sim.washstand.expert.ScriptedExpert.run", side_effect=AssertionError("expert called during replay")):
            result = replay_episode(self.path, update_metadata=False)
        self.assertFalse(result["replay_passed"])
        self.assertIn("Independent final physical success check failed", result["replay_errors"])
        self.assertFalse(any("mismatch" in error for error in result["replay_errors"]), result)
        self.assertLess(result["replay_max_state_error"], 1e-6)
        self.assertLess(result["replay_max_qvel_error"], 1e-6)

    def test_replay_rejects_saved_state_damage(self):
        self.damage("state")
        result = replay_episode(self.path, update_metadata=False)
        self.assertFalse(result["replay_passed"])
        self.assertTrue(any("state mismatch" in error for error in result["replay_errors"]), result)

    def test_replay_rejects_saved_velocity_damage(self):
        self.damage("qvel")
        result = replay_episode(self.path, update_metadata=False)
        self.assertFalse(result["replay_passed"])
        self.assertTrue(any("qvel mismatch" in error for error in result["replay_errors"]), result)

    def test_replay_rejects_saved_qpos_damage(self):
        self.damage("qpos")
        result = replay_episode(self.path, update_metadata=False)
        self.assertFalse(result["replay_passed"])
        self.assertTrue(any("qpos mismatch" in error for error in result["replay_errors"]), result)

    def test_replay_rejects_nonfinite_arrays_before_numeric_comparison(self):
        for name in ("state", "action", "qpos", "qvel"):
            with self.subTest(name=name):
                episode = load_episode(self.path)
                arrays = episode["arrays"]
                original = arrays[name][1, 0]
                arrays[name][1, 0] = np.nan
                np.savez_compressed(self.path / "trajectory.npz", **arrays)
                result = replay_episode(self.path, update_metadata=False)
                self.assertFalse(result["replay_passed"])
                self.assertTrue(any(f"Invalid {name}" in error for error in result["replay_errors"]), result)
                arrays[name][1, 0] = original
                np.savez_compressed(self.path / "trajectory.npz", **arrays)


if __name__ == "__main__":
    unittest.main()
