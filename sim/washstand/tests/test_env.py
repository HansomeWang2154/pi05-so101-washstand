"""Optional runtime integration tests; never imported by ordinary make test.

Run physics: .venv/bin/python -m unittest discover -s sim/washstand/tests -v
Add WASHSTAND_RENDER_TESTS=1 MUJOCO_GL=egl for both camera render checks.
"""

import os
import unittest

import mujoco
import numpy as np

from sim.washstand import WashstandEnv, load_config


class PhysicalEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = WashstandEnv(seed=7)
        self.addCleanup(self.env.close)

    def test_state_uses_named_joint_addresses_and_returns_copy(self):
        e = self.env
        expected = [e.model.jnt_qposadr[mujoco.mj_name2id(
            e.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in e.joint_names]
        np.testing.assert_array_equal(e.qpos_ids, expected)
        self.assertEqual(e.state().shape, (6,))
        observed = e.state()
        observed[:] = 123
        self.assertFalse(np.any(e.state() == 123))

    def test_only_robot_is_actuated_and_objects_are_free(self):
        e = self.env
        self.assertEqual(e.model.neq, 0)
        self.assertEqual(e.model.nu, 6)
        for prefix in ("brush_red", "brush_blue", "cup_left", "cup_right"):
            body_id = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_BODY, prefix)
            joint_id = e.model.body_jntadr[body_id]
            self.assertEqual(e.model.body_jntnum[body_id], 1)
            self.assertEqual(e.model.jnt_type[joint_id], mujoco.mjtJoint.mjJNT_FREE)
            self.assertFalse(np.any(e.model.actuator_trnid[:, 0] == joint_id))

    def test_invalid_targets_do_not_advance_and_clamping_is_recorded(self):
        e = self.env
        for targets in ([0] * 5, [0] * 7, [0, 0, 0, 0, np.nan, 0], [0, 0, 0, 0, np.inf, 0]):
            before = e.data.time
            with self.assertRaises(ValueError):
                e.step_targets(targets)
            self.assertEqual(e.data.time, before)
        e.step_targets([99] * 6)
        np.testing.assert_array_equal(e.last_action, e.control_bounds[:, 1])
        np.testing.assert_array_equal(e.data.ctrl[e.actuator_ids], e.last_action)

    def test_control_frequency_matches_elapsed_physics_time(self):
        e = self.env
        before = e.data.time
        for _ in range(30):
            e.step_targets(e.config["home_targets_rad"])
        self.assertAlmostEqual(e.data.time - before, 1.0, places=11)
        config = load_config()
        config["physics_timestep"] = 0.005
        with self.assertRaises(ValueError):
            WashstandEnv(config=config)

    def test_full_snapshot_and_target_replay_are_deterministic(self):
        e = self.env
        initial = e.snapshot()
        action = np.array(e.config["home_targets_rad"], dtype=float)
        action[0] = 0.1
        for _ in range(12):
            e.step_targets(action)
        result = e.snapshot()
        e.restore(initial)
        for _ in range(12):
            e.step_targets(action)
        np.testing.assert_allclose(e.snapshot()["state"], result["state"], rtol=0, atol=1e-12)

    def test_cup_is_a_physical_cavity_and_accepts_a_dropped_brush(self):
        e = self.env
        body = e.brush_body("red")
        joint = e.model.body_jntadr[body]
        qpos = e.model.jnt_qposadr[joint]
        dof = e.model.jnt_dofadr[joint]
        cup = e.cup_body("left")
        center = e.data.xpos[cup].copy()
        # Independent reset-only test: place a vertical free brush above the
        # opening, then use gravity/contact for all subsequent evolution.
        e.data.qpos[qpos:qpos + 7] = [center[0], center[1], .165,
                                     np.sqrt(.5), 0, -np.sqrt(.5), 0]
        e.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(e.model, e.data)
        for _ in range(120):
            e.step_targets(e.config["home_targets_rad"])
        tip_id = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_SITE, "brush_red_tip")
        local_tip = e.data.xmat[cup].reshape(3, 3).T @ (e.data.site_xpos[tip_id] - e.data.xpos[cup])
        self.assertLess(local_tip[2], e.config["cup"]["height"] - .01)
        self.assertGreater(local_tip[2], .002)
        self.assertLess(np.linalg.norm(local_tip[:2]), e.config["cup"]["inner_radius"] + .003)
        self.assertLess(np.linalg.norm(e.data.cvel[body]), .01)
        cup_contact = False
        for index in range(e.data.ncon):
            contact = e.data.contact[index]
            bodies = {int(e.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
            cup_contact |= bodies == {body, cup}
        self.assertTrue(cup_contact)

    def test_seed_reset_and_scene_identity(self):
        e = self.env
        first_state, first_metadata = e.state(), e.metadata()
        e.step_targets([0] * 6)
        e.reset(seed=7)
        np.testing.assert_array_equal(first_state, e.state())
        self.assertEqual(first_metadata["xml_sha256"], e.metadata()["xml_sha256"])
        self.assertFalse(first_metadata["hidden_grasp_assistance"])
        self.assertFalse(first_metadata["real_interface_mapping_verified"])
        with self.assertRaises(ValueError):
            e.reset(task_id="unknown")

    @unittest.skipUnless(os.environ.get("WASHSTAND_RENDER_TESTS") == "1", "explicit render check only")
    def test_top_and_wrist_rgb(self):
        for camera in ("top", "wrist"):
            pixels = self.env.render(camera)
            self.assertEqual(pixels.shape, (240, 320, 3))
            self.assertEqual(pixels.dtype, np.uint8)
            self.assertGreater(pixels.max(), pixels.min())
        with self.assertRaises(ValueError):
            self.env.render("unknown")


if __name__ == "__main__":
    unittest.main()
