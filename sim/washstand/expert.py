"""Privileged, contact-driven scripted expert for the nominal tabletop task.

Object states are never written here: only six robot joint targets are sent.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .kinematics import PoseSolver, grasp_rotation, upright_rotation


class ExpertFailure(RuntimeError):
    pass


class ScriptedExpert:
    VERSION = "nominal-contact-expert-v1"

    def __init__(self, env, validator, on_frame=None):
        self.env, self.validator, self.on_frame = env, validator, on_frame
        self.solver = PoseSolver(env)
        self.color, self.cup = env.task_id.split("_")
        self.brush = env.brush_body(self.color)
        self.start_time = float(env.data.time)
        self.frames = 0
        self.stages = []
        self.last_check = {}
        self.closed = -.15
        self.open = .85

    def _step(self, target, stage):
        # Quantize before both sending and recording: float32 dataset targets
        # then represent the exact controls that the simulator received.
        action = np.clip(np.asarray(target, dtype=np.float32).astype(np.float64),
                         self.env.control_bounds[:, 0], self.env.control_bounds[:, 1])
        if not self.stages or self.stages[-1] != stage:
            self.stages.append(stage)
        if self.on_frame:
            self.on_frame(stage, action, self)
        self.env.step_targets(action)
        self.frames += 1
        self.last_check = self.validator.update()
        if not np.isfinite(self.env.data.qpos).all() or not np.isfinite(self.env.data.qvel).all():
            raise ExpertFailure("nonfinite_state")
        if self.env.data.time-self.start_time > 45:
            raise ExpertFailure("timeout")

    def hold(self, seconds, gripper, stage):
        goal = self.env.data.ctrl[self.env.actuator_ids].copy()
        goal[5] = gripper
        for _ in range(math.ceil(seconds*self.env.config["control_hz"])):
            self._step(goal, stage)

    def move_joints(self, goal, gripper, stage, *, seconds=.6, max_speed=1.0):
        start = self.env.state()[:5]
        goal = np.asarray(goal)[:5]
        duration = max(seconds, 1.5*float(np.max(np.abs(goal-start)))/max_speed)
        frames = max(2, math.ceil(duration*self.env.config["control_hz"]))
        for index in range(1, frames+1):
            fraction = index/frames
            blend = fraction*fraction*(3-2*fraction)
            self._step(np.r_[start+(goal-start)*blend, gripper], stage)
        self.hold(.12, gripper, stage)
        error = float(np.max(np.abs(self.env.state()[:5]-goal)))
        if error > .06:
            raise ExpertFailure(f"pose_tracking_failed_in_{stage}: {error:.4f}rad")

    def move_pose(self, position, rotation, gripper, stage, **kwargs):
        goal = self.solver.solve(position, rotation, self.env.state())
        self.move_joints(goal, gripper, stage, **kwargs)

    def run(self):
        from .validators import contact_groups

        outcome = {"physics_success": False, "expert_version": self.VERSION}
        try:
            data = self.env.data
            brush_axis = data.xmat[self.brush].reshape(3, 3)[:, 0].copy()
            grasp = data.site(f"brush_{self.color}_grasp").xpos.copy()
            rg = grasp_rotation(brush_axis)
            self.move_pose(grasp+[0, 0, .060], rg, self.open, "PREGRASP", seconds=1.2)
            grasp = data.site(f"brush_{self.color}_grasp").xpos.copy()
            self.move_pose(grasp+[0, 0, .002], rg, self.open, "APPROACH", seconds=1.1)
            self.hold(.45, self.closed, "CLOSE")
            groups = contact_groups(self.env, self.color)
            outcome["close_contacts"] = groups
            if not (groups["fixed_jaw"] and groups["moving_jaw"]):
                raise ExpertFailure("no_two_jaw_grasp")
            self.move_pose(np.r_[grasp[:2], .043], rg, self.closed, "LIFT", seconds=.9)
            height = float(data.xpos[self.brush, 2])
            outcome["lift_center_height_m"] = height
            if height < .022:
                raise ExpertFailure("no_grasp_after_trial_lift")
            lift_groups = contact_groups(self.env, self.color)
            if not (lift_groups["fixed_jaw"] and lift_groups["moving_jaw"]):
                raise ExpertFailure("lost_two_jaw_contact_after_lift")

            # Raising the TCP vertically while pointing down exceeds this
            # five-axis arm's wrist range. Reorient and raise together, keeping
            # the lower end above the tabletop throughout the nominal arc.
            turn_end = grasp + [.035, 0, 0]
            ru = upright_rotation(turn_end)
            rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([rg, ru])))
            for fraction in np.linspace(1/12, 1, 12):
                position = np.r_[grasp[:2]+[.035*fraction, 0], .043+.140*math.sin(fraction*math.pi/2)]
                self.move_pose(position, rotations(fraction).as_matrix(), self.closed,
                               "REORIENT", seconds=.14, max_speed=.8)
            if data.xpos[self.brush, 2] < .035:
                raise ExpertFailure("dropped_during_reorientation")

            cup = self.env.cup_body(self.cup)
            center = data.xpos[cup].copy()
            r_place = upright_rotation(center)
            # Privileged geometry corrects the held object/TCP offset; no
            # attachment is created and slipping still causes physical failure.
            tcp_r = data.site_xmat[self.env.tcp_id].reshape(3, 3)
            tip = data.site(f"brush_{self.color}_tip").xpos.copy()
            relative_tip = tcp_r.T @ (tip-data.site_xpos[self.env.tcp_id])
            offset = r_place @ relative_tip
            above = center + [0, 0, self.env.config["cup"]["height"]+.035] - offset
            self.move_pose(above, r_place, self.closed, "ABOVE_CUP", seconds=1.6)
            if np.linalg.norm(data.xpos[self.brush]-data.site_xpos[self.env.tcp_id]) > .12:
                raise ExpertFailure("lost_object_before_insertion")
            insert = above.copy()
            insert[2] = center[2]+self.env.config["cup"]["bottom_thickness"]+.014-offset[2]
            self.move_pose(insert, r_place, self.closed, "INSERT", seconds=2.4, max_speed=.5)
            self.hold(.35, self.closed, "INSERT")
            self.hold(.5, self.open, "RELEASE")
            retract = insert + [-.025, 0, .09]
            self.move_pose(retract, r_place, self.open, "RETRACT", seconds=1.3)
            self.hold(3.0, self.open, "SETTLE")
            outcome["physics_success"] = bool(self.last_check.get("success"))
            outcome["failure_reason"] = None if outcome["physics_success"] else "final_validation_failed"
        except Exception as exc:
            outcome["failure_reason"] = f"{type(exc).__name__}: {exc}"
        outcome.update({"frames": self.frames, "simulation_seconds": float(self.env.data.time-self.start_time),
                        "stages": self.stages, "final_check": self.last_check})
        return outcome
