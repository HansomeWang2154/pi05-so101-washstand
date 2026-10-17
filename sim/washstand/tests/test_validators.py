"""Optional MuJoCo tests: python -m unittest discover -s sim/washstand/tests."""

import math
import unittest

import mujoco
import numpy as np

from sim.washstand.env import WashstandEnv
from sim.washstand.validators import SuccessValidator, contact_groups, cup_geometry, numeric_health


class GeometryTests(unittest.TestCase):
    def geometry(self, tip, axis=(0, 0, 1), rotation=None):
        return cup_geometry(tip, axis, [0, 0, 0], np.eye(3) if rotation is None else rotation,
                            inner_radius=.04, height=.076, bottom_thickness=.006)

    def test_actual_handle_segment_inside(self):
        self.assertTrue(self.geometry([0, 0, .008])["inserted"])

    def test_tip_near_rim_is_insufficient(self):
        self.assertFalse(self.geometry([0, 0, .073])["inserted"])

    def test_beside_cup_and_through_bottom_fail(self):
        self.assertFalse(self.geometry([.045, 0, .02])["inserted"])
        self.assertFalse(self.geometry([0, 0, -.003])["inserted"])

    def test_current_cup_local_frame(self):
        angle = .12
        rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                             [-math.sin(angle), 0, math.cos(angle)]])
        tip, axis = rotation @ [0, 0, .008], rotation @ [0, 0, 1]
        self.assertTrue(self.geometry(tip, axis, rotation)["inserted"])


class PhysicalValidatorTests(unittest.TestCase):
    """Construct reset states only, then let true contact/gravity settle them.

    These are checker fixtures, not demonstrations or expert success evidence.
    No object state is modified during any settling interval.
    """

    def setUp(self):
        self.env = WashstandEnv()
        self.validator = SuccessValidator(self.env, "red_left")

    def tearDown(self):
        self.env.close()

    def place(self, body_id, position, quaternion=(1, 0, 0, 0)):
        model, data = self.env.model, self.env.data
        joint = model.body_jntadr[body_id]
        qpos, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        data.qpos[qpos:qpos + 7] = np.r_[position, quaternion]
        data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(model, data)

    def drop_into(self, cup="left", brush="red"):
        position = self.env.data.xpos[self.env.cup_body(cup)].copy() + [0, 0, .085]
        self.place(self.env.brush_body(brush), position, (math.sqrt(.5), 0, -math.sqrt(.5), 0))

    def advance(self, steps=240, targets=None):
        for _ in range(steps):
            self.env.step_targets(targets if targets is not None else self.env.config["home_targets_rad"])
            result = self.validator.update()
        return result

    def test_tabletop_is_not_success(self):
        result = self.advance(90)
        self.assertFalse(result["success"])
        self.assertIn("handle_not_inserted", result["reasons"])

    def test_supported_released_stable_true_contact_positive(self):
        self.drop_into()
        result = self.advance()
        self.assertTrue(result["success"], result)
        self.assertGreaterEqual(result["stable_seconds"], 2)
        self.assertGreater(result["details"]["contacts"]["cup_left"], 0)
        self.assertGreater(result["details"]["support_weight_fraction"], .8)
        self.assertEqual(self.env.model.neq, 0)

    def test_wrong_cup_fails(self):
        self.drop_into("right")
        result = self.advance()
        self.assertFalse(result["success"])
        self.assertIn("wrong_cup", result["reasons"])

    def test_wrong_brush_in_a_cup_fails(self):
        self.drop_into("left", "blue")
        result = self.advance()
        self.assertFalse(result["success"])
        self.assertIn("wrong_object_stored", result["reasons"])

    def test_closed_gripper_rejected_despite_stable_brush(self):
        self.drop_into()
        targets = self.env.state()
        targets[-1] = 0
        result = self.advance(targets=targets)
        self.assertFalse(result["success"])
        self.assertIn("gripper_not_open", result["reasons"])

    def test_tipped_and_moved_cup_rejected(self):
        cup = self.env.cup_body("left")
        position = self.env.data.xpos[cup].copy() + [0.06, 0, 0.045]
        self.place(cup, position, (math.sqrt(.5), math.sqrt(.5), 0, 0))
        result = self.validator.update()
        self.assertFalse(result["success"])
        self.assertIn("cup_tipped", result["reasons"])
        self.assertIn("cup_moved", result["reasons"])

    def test_gripper_contact_identified_independently(self):
        fixed = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_box1")
        self.place(self.env.brush_body("red"), self.env.data.geom_xpos[fixed].copy())
        contacts = contact_groups(self.env)
        self.assertGreater(contacts["robot_total"], 0)
        result = self.validator.update()
        self.assertFalse(result["success"])
        self.assertIn("robot_still_touching", result["reasons"])

    def test_repeated_same_time_does_not_advance_stability(self):
        self.drop_into()
        result = self.advance()
        for _ in range(10):
            repeated = self.validator.update()
        self.assertEqual(repeated["stable_frames"], result["stable_frames"])
        self.assertEqual(repeated["stable_seconds"], result["stable_seconds"])

    def test_sampling_gap_resets_stability_and_success_is_not_sticky(self):
        self.drop_into()
        self.assertTrue(self.advance()["success"])
        for _ in range(5):
            self.env.step_targets(self.env.config["home_targets_rad"])
        result = self.validator.update()
        self.assertFalse(result["success"])
        self.assertIn("sampling_gap", result["reasons"])
        self.assertEqual(result["stable_seconds"], 0)

    def test_numeric_error_remains_rejected_after_values_recover(self):
        original = self.env.data.qvel[0]
        self.env.data.qvel[0] = np.nan
        self.assertFalse(numeric_health(self.env)["finite"])
        self.assertIn("numeric_error", self.validator.update()["reasons"])
        self.env.data.qvel[0] = original
        self.assertIn("numeric_error", self.validator.update()["reasons"])

    def test_joint_limit_violation_is_detected(self):
        self.env.data.qpos[self.env.qpos_ids[0]] = 10
        health = numeric_health(self.env)
        self.assertFalse(health["ok"])
        self.assertIn("shoulder_pan", health["joint_limit_violations"])

    def test_wrong_cup_tip_is_terminal_after_recovery(self):
        self.drop_into()
        self.assertTrue(self.advance()["success"])
        body = self.env.cup_body("right")
        original = self.env.data.xpos[body].copy()
        self.place(body, original + [0, 0, .12], (math.sqrt(.5), math.sqrt(.5), 0, 0))
        bad = self.validator.update()
        self.assertFalse(bad["success"])
        self.assertIn("cup_tipped", bad["reasons"])
        self.place(body, original)
        self.assertFalse(self.advance(90)["success"])
        self.validator.reset()
        self.assertTrue(self.advance(90)["success"])

    def test_target_cup_movement_is_terminal_after_recovery(self):
        self.drop_into()
        self.assertTrue(self.advance()["success"])
        body = self.env.cup_body("left")
        original = self.env.data.xpos[body].copy()
        self.place(body, original + [.1, 0, .25])
        bad = self.validator.update()
        self.assertFalse(bad["success"])
        self.assertIn("cup_moved", bad["reasons"])
        self.place(body, original)
        self.assertFalse(self.advance(90)["success"])

    def test_wrong_object_storage_is_terminal_after_removal(self):
        self.drop_into()
        self.assertTrue(self.advance()["success"])
        body = self.env.brush_body("blue")
        original = self.env.data.xpos[body].copy()
        self.drop_into("right", "blue")
        bad = self.advance(150)
        self.assertFalse(bad["success"])
        self.assertIn("wrong_object_stored", bad["reasons"])
        self.place(body, original)
        self.assertFalse(self.advance(90)["success"])


if __name__ == "__main__":
    unittest.main()
