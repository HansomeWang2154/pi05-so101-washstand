"""Independent physical checks for simulated toothbrush-in-cup episodes.

These checks do not consume an expert stage/done flag. Thresholds describe the
nominal prototype, not measured real hardware. All distances are metres and
all velocities are SI units. Call ``update`` after each control step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class ValidatorThresholds:
    geometry_margin: float = 0.001
    contact_geometry_tolerance: float = 0.0015
    handle_segment_length: float = 0.016
    minimum_tip_depth: float = 0.018
    bottom_tolerance: float = 0.0015
    maximum_cup_translation: float = 0.025
    maximum_cup_tilt_rad: float = math.radians(10)
    maximum_brush_linear_speed: float = 0.025
    maximum_brush_angular_speed: float = 0.30
    maximum_cup_linear_speed: float = 0.015
    maximum_cup_angular_speed: float = 0.20
    minimum_gripper_open_rad: float = 0.35
    minimum_tcp_clearance: float = 0.05
    minimum_support_weight_fraction: float = 0.20
    maximum_penetration: float = 0.006
    joint_limit_tolerance: float = 0.015


def _id(env, kind, name):
    value = mujoco.mj_name2id(env.model, kind, name)
    if value < 0:
        raise ValueError(f"Missing {kind}: {name}")
    return value


def _descends(model, body_id, ancestor_id):
    while body_id:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return ancestor_id == 0


def _body_velocity(env, body_id):
    """World angular/linear velocity about the body centre, without qvel slices."""
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY,
                           body_id, velocity, 0)
    return velocity[3:].copy(), velocity[:3].copy()


def cup_geometry(tip_world, axis_world, cup_position, cup_rotation, *,
                 inner_radius, height, bottom_thickness, handle_radius=0.004,
                 thresholds=None):
    """Check an actual handle segment inside a cup's current local cavity.

    ``axis_world`` points from the handle tip towards the brush head. A tip
    close to the rim alone does not satisfy this test. The whole sampled
    segment, including its radial thickness, must lie inside the cavity.
    """
    t = thresholds or ValidatorThresholds()
    tip = np.asarray(tip_world, dtype=float)
    axis = np.asarray(axis_world, dtype=float)
    rotation = np.asarray(cup_rotation, dtype=float).reshape(3, 3)
    position = np.asarray(cup_position, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or not np.isfinite(np.r_[tip, axis, position, rotation.ravel()]).all():
        return {"inserted": False, "reason": "invalid_geometry"}
    # The semantic tip is the capsule's physical endpoint, not the centre of
    # its rounded cap. Its cross-section starts at zero radius and grows to
    # handle_radius; subtracting the full radius at the tip wrongly rejects a
    # normally leaning brush whose cap touches the inner cup wall.
    distances = np.unique(np.r_[0.0, handle_radius,
                                np.linspace(0, t.handle_segment_length, 7)])
    segment = tip + distances[:, None] * axis / norm
    local = (segment - position) @ rotation
    radial = np.linalg.norm(local[:, :2], axis=1)
    cap_radii = np.sqrt(np.maximum(0.0, handle_radius ** 2
                                   - np.maximum(0.0, handle_radius - distances) ** 2))
    radial_limits = inner_radius - cap_radii + t.contact_geometry_tolerance
    conditions = {
        "radially_inside": bool(np.all(radial <= radial_limits)),
        "below_rim": bool(np.all(local[:, 2] <= height - t.geometry_margin)),
        "above_bottom": bool(np.all(local[:, 2] >= bottom_thickness - t.bottom_tolerance)),
        "tip_deep_enough": bool(local[0, 2] <= height - t.minimum_tip_depth),
    }
    return {"inserted": bool(all(conditions.values())), "conditions": conditions,
            "tip_local": local[0].tolist(), "segment_local": local.tolist(),
            "maximum_radial": float(radial.max()), "radial_limits": radial_limits.tolist(),
            "tip_depth": float(height - local[0, 2])}


def contact_groups(env, brush="red"):
    """Physical contact counts and upward cup load acting on one toothbrush.

    Fixed/moving finger contacts are distinct; touching the table with a closed
    gripper is never evidence of a grasp. Contact force is transformed from the
    contact frame, with sign chosen for the force acting on the brush body.
    """
    model, data = env.model, env.data
    brush_id = env.brush_body(brush)
    cup_ids = {name: env.cup_body(name) for name in ("left", "right")}
    robot_id = _id(env, mujoco.mjtObj.mjOBJ_BODY, "base")
    fixed_id = _id(env, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    moving_id = _id(env, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    counts = {key: 0 for key in ("fixed_jaw", "moving_jaw", "cup_left", "cup_right", "table", "robot", "other")}
    upward = {"left": 0.0, "right": 0.0}
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if brush_id not in (b1, b2):
            continue
        other = b2 if b1 == brush_id else b1
        other_geom = g2 if b1 == brush_id else g1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
        group = "other"
        if other == cup_ids["left"]:
            group = "cup_left"
        elif other == cup_ids["right"]:
            group = "cup_right"
        elif name == "table":
            group = "table"
        elif _descends(model, other, moving_id):
            group = "moving_jaw"
        elif other == fixed_id and name.startswith("fixed_jaw"):
            group = "fixed_jaw"
        elif _descends(model, other, robot_id):
            group = "robot"
        counts[group] += 1
        force_local = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force_local)
        force_world = contact.frame.reshape(3, 3).T @ force_local[:3]
        if b1 == brush_id:
            force_world = -force_world
        if group.startswith("cup_"):
            upward[group.removeprefix("cup_")] += float(force_world[2])
        rows.append({"group": group, "other_geom": name, "distance": float(contact.dist),
                     "force_world_on_brush": force_world.tolist()})
    return {**counts, "cup_upward_force": upward,
            "robot_total": counts["robot"] + counts["fixed_jaw"] + counts["moving_jaw"],
            "contacts": rows}


def grasp_evidence(env, brush="red", previous_relative=None, *,
                   minimum_lift=0.018, maximum_slip=0.012):
    """Read-only grasp evidence; the expert may use this for stage transitions.

    ``previous_relative`` is a 3-vector returned by an earlier call after grasp.
    A dual contact is necessary but the expert must additionally verify lifting
    and low slip; a single sample is never a demonstrated stable grasp.
    """
    body_id = env.brush_body(brush)
    tcp_pos = env.data.site_xpos[env.tcp_id]
    tcp_rot = env.data.site_xmat[env.tcp_id].reshape(3, 3)
    relative = tcp_rot.T @ (env.data.xpos[body_id] - tcp_pos)
    contact = contact_groups(env, brush)
    position = env.data.xpos[body_id]
    initial_height = env.config["brush"]["centers"][brush][2]
    lift = float(position[2] - initial_height)
    slip = (float(np.linalg.norm(relative - np.asarray(previous_relative)))
            if previous_relative is not None else None)
    dual = bool(contact["fixed_jaw"] and contact["moving_jaw"])
    return {"dual_finger_contact": dual, "relative_position_tcp": relative.tolist(),
            "lift_m": lift, "lifted": lift >= minimum_lift, "slip_m": slip,
            "low_slip": None if slip is None else slip <= maximum_slip,
            "contacts": contact}


def numeric_health(env, *, warning_baseline=None, thresholds=None):
    """Return current values plus the environment's substep penetration peak."""
    model, data = env.model, env.data
    t = thresholds or ValidatorThresholds()
    finite = bool(all(np.isfinite(values).all() for values in
                      (data.qpos, data.qvel, data.qacc, data.ctrl, data.xpos)))
    joint_violations = []
    for joint in range(model.njnt):
        if model.jnt_limited[joint]:
            value = float(data.qpos[model.jnt_qposadr[joint]])
            low, high = model.jnt_range[joint]
            if not low - t.joint_limit_tolerance <= value <= high + t.joint_limit_tolerance:
                joint_violations.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint))
    warning_counts = {}
    for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL"):
        enum = getattr(mujoco.mjtWarning, name)
        count = int(data.warning[int(enum)].number)
        if count > (warning_baseline or {}).get(name, 0):
            warning_counts[name] = count
    current_penetration = max((max(0.0, -float(data.contact[i].dist)) for i in range(data.ncon)), default=0.0)
    penetration = max(current_penetration, float(getattr(env, "peak_penetration_m", 0.0)))
    return {"ok": bool(finite and not joint_violations and not warning_counts
                       and penetration <= t.maximum_penetration),
            "finite": finite, "joint_limit_violations": joint_violations,
            "numeric_warnings": warning_counts, "maximum_penetration_m": penetration,
            "current_penetration_m": current_penetration}


def _segment_distance(point, start, end):
    delta = end - start
    fraction = np.clip(np.dot(point - start, delta) / max(np.dot(delta, delta), 1e-15), 0, 1)
    return float(np.linalg.norm(point - start - fraction * delta))


class SuccessValidator:
    """Require released, supported insertion to remain valid for two seconds.

    Time accumulates from MuJoCo's simulation clock, not call count or wall
    time. Any invalid sample resets the stability window. ``success`` is not
    sticky: dropping the brush on the next step revokes it. Numeric errors are
    sticky until reset, as are wrong-container/object and cup-movement faults.
    Both cups are checked. Call at least once every control step. The environment
    preserves penetration peaks at every physics substep; MuJoCo preserves
    numeric warning counts between these control-rate checks.
    """

    def __init__(self, env, task_id=None, stable_seconds=2.0, *, thresholds=None):
        self.env = env
        self.task_id = task_id or env.task_id
        try:
            self.brush, self.cup = self.task_id.split("_")
        except ValueError as exc:
            raise ValueError("task_id must be red_left/red_right/blue_left/blue_right") from exc
        if self.brush not in ("red", "blue") or self.cup not in ("left", "right"):
            raise ValueError("task_id must be red_left/red_right/blue_left/blue_right")
        if stable_seconds <= 0:
            raise ValueError("stable_seconds must be positive")
        self.stable_seconds_required = float(stable_seconds)
        self.thresholds = thresholds or ValidatorThresholds()
        self.reset()

    def reset(self):
        self._start_time = None
        self._last_time = None
        self._stable_frames = 0
        self._numeric_error = False
        self._physics_error = False
        self._terminal_faults = set()
        self._cup_initial = {name: self.env.data.xpos[self.env.cup_body(name)].copy()
                             for name in ("left", "right")}
        self._warning_baseline = {name: int(self.env.data.warning[int(getattr(mujoco.mjtWarning, name))].number)
                                  for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL")}
        return self

    def _geometry(self, brush, cup):
        env = self.env
        brush_id, cup_id = env.brush_body(brush), env.cup_body(cup)
        tip_id = _id(env, mujoco.mjtObj.mjOBJ_SITE, f"brush_{brush}_tip")
        cfg = env.config["cup"]
        return cup_geometry(env.data.site_xpos[tip_id], env.data.xmat[brush_id].reshape(3, 3)[:, 0],
                            env.data.xpos[cup_id], env.data.xmat[cup_id],
                            inner_radius=cfg["inner_radius"], height=cfg["height"],
                            bottom_thickness=cfg["bottom_thickness"],
                            handle_radius=env.config["brush"]["handle_radius"], thresholds=self.thresholds)

    def update(self):
        env, t = self.env, self.thresholds
        now = float(env.data.time)
        if self._last_time is not None and now < self._last_time - 1e-10:
            raise ValueError("Simulation clock went backwards; reset the validator after restoring/resetting the environment")
        health = numeric_health(env, warning_baseline=self._warning_baseline, thresholds=t)
        self._numeric_error |= not health["finite"] or bool(health["numeric_warnings"])
        self._physics_error |= not health["ok"]
        details = {"health": health}
        reasons = []
        if self._numeric_error:
            reasons.append("numeric_error")
        elif self._physics_error:
            reasons.append("invalid_physics")
        if self._last_time is not None and now - self._last_time > 1.5 / env.config["control_hz"]:
            reasons.append("sampling_gap")
        if health["finite"]:
            geometry = self._geometry(self.brush, self.cup)
            details["target_geometry"] = geometry
            if not geometry["inserted"]:
                reasons.append("handle_not_inserted")
            wrong_cup = "right" if self.cup == "left" else "left"
            if self._geometry(self.brush, wrong_cup)["inserted"]:
                reasons.insert(0, "wrong_cup")
                self._terminal_faults.add("wrong_cup")
            wrong_brush = "blue" if self.brush == "red" else "red"
            if any(self._geometry(wrong_brush, cup)["inserted"] for cup in ("left", "right")):
                reasons.insert(0, "wrong_object_stored")
                self._terminal_faults.add("wrong_object_stored")
            body, cup_id = env.brush_body(self.brush), env.cup_body(self.cup)
            cup_rotation = env.data.xmat[cup_id].reshape(3, 3)
            tilt = math.acos(float(np.clip(cup_rotation[2, 2], -1, 1)))
            translation = float(np.linalg.norm(env.data.xpos[cup_id] - self._cup_initial[self.cup]))
            details.update(cup_tilt_rad=tilt, cup_translation_m=translation)
            # Both cups belong to the task. A wrong cup knocked over, or a
            # terminal mistake later recovered, must not become a success.
            cup_states = {}
            for name in ("left", "right"):
                other_id = env.cup_body(name)
                other_rotation = env.data.xmat[other_id].reshape(3, 3)
                other_tilt = math.acos(float(np.clip(other_rotation[2, 2], -1, 1)))
                other_translation = float(np.linalg.norm(env.data.xpos[other_id]-self._cup_initial[name]))
                cup_states[name] = {"tilt_rad": other_tilt, "translation_m": other_translation}
                if other_tilt > t.maximum_cup_tilt_rad:
                    self._terminal_faults.add("cup_tipped")
                if other_translation > t.maximum_cup_translation:
                    self._terminal_faults.add("cup_moved")
            details["cup_states"] = cup_states
            contacts = contact_groups(env, self.brush)
            details["contacts"] = contacts
            gravity = float(np.linalg.norm(env.model.opt.gravity))
            weight = float(env.model.body_mass[body] * gravity)
            support = contacts["cup_upward_force"][self.cup]
            details["support_weight_fraction"] = support / max(weight, 1e-12)
            if gravity < 1 or env.model.body_gravcomp[body] > 0.01:
                reasons.append("gravity_disabled")
            if not contacts[f"cup_{self.cup}"] or support < weight * t.minimum_support_weight_fraction:
                reasons.append("not_supported_by_cup")
            if contacts["robot_total"]:
                reasons.append("robot_still_touching")
            joint = _id(env, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
            opening = float(env.data.qpos[env.model.jnt_qposadr[joint]])
            details["gripper_open_rad"] = opening
            if opening < t.minimum_gripper_open_rad:
                reasons.append("gripper_not_open")
            tip = env.data.site_xpos[_id(env, mujoco.mjtObj.mjOBJ_SITE, f"brush_{self.brush}_tip")]
            head = env.data.site_xpos[_id(env, mujoco.mjtObj.mjOBJ_SITE, f"brush_{self.brush}_head")]
            clearance = _segment_distance(env.data.site_xpos[env.tcp_id], tip, head)
            details["tcp_brush_clearance_m"] = clearance
            if clearance < t.minimum_tcp_clearance:
                reasons.append("gripper_not_retracted")
            linear, angular = _body_velocity(env, body)
            cup_linear, cup_angular = _body_velocity(env, cup_id)
            speeds = {"brush_linear": float(np.linalg.norm(linear)), "brush_angular": float(np.linalg.norm(angular)),
                      "cup_linear": float(np.linalg.norm(cup_linear)), "cup_angular": float(np.linalg.norm(cup_angular))}
            details["speeds"] = speeds
            if (speeds["brush_linear"] > t.maximum_brush_linear_speed
                    or speeds["brush_angular"] > t.maximum_brush_angular_speed
                    or speeds["cup_linear"] > t.maximum_cup_linear_speed
                    or speeds["cup_angular"] > t.maximum_cup_angular_speed):
                reasons.append("unstable")
        details["terminal_faults"] = sorted(self._terminal_faults)
        reasons.extend(reason for reason in sorted(self._terminal_faults) if reason not in reasons)
        fresh_time = self._last_time is None or now > self._last_time + 1e-10
        if reasons:
            self._start_time = None
            self._stable_frames = 0
        elif self._start_time is None:
            self._start_time = now
            self._stable_frames = 1
        elif fresh_time:
            self._stable_frames += 1
        stable = 0.0 if self._start_time is None else max(0.0, now - self._start_time)
        success = bool(not reasons and stable >= self.stable_seconds_required - 1e-9)
        self._last_time = now
        return {"success": success, "stable_frames": self._stable_frames, "stable_seconds": stable,
                "reason": "success" if success else (reasons[0] if reasons else "settling"),
                "reasons": reasons, "details": details}
