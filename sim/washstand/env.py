"""Single-environment MuJoCo runtime, with six joint target controls only."""

from __future__ import annotations

import copy
import hashlib

import mujoco
import numpy as np

from .scene import ASSET_DIR, JOINT_NAMES, TASKS, asset_manifest, build_xml, load_config


class WashstandEnv:
    """Physical prototype. Render backend is chosen by MUJOCO_GL before import."""

    def __init__(self, seed=0, task_id="red_left", width=320, height=240, *, config=None):
        self.config = copy.deepcopy(config or load_config())
        self.control_dt = self.config["physics_timestep"] * self.config["substeps"]
        if not np.isclose(self.control_dt * self.config["control_hz"], 1.0, atol=1e-12):
            raise ValueError("physics_timestep * substeps must equal 1/control_hz")
        self.width, self.height = int(width), int(height)
        self.joint_names = JOINT_NAMES
        self._renderer = None
        self.model = None
        self.reset(seed=seed, task_id=task_id)

    def _id(self, kind, name):
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"Missing model element {name}")
        return value

    def reset(self, seed=0, task_id="red_left"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.close()
        self.seed, self.task_id = int(seed), task_id
        self.task_text = TASKS[task_id]
        self.rng = np.random.default_rng(seed)
        jitter = self.config["layout_jitter_m"]
        self.layout_offsets = {name: self.rng.uniform(-jitter, jitter, size=2).tolist()
                               for name in ("brush_red", "brush_blue", "cup_left", "cup_right")}
        self.xml = build_xml(self.config, offsets=self.layout_offsets)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.peak_penetration_m = 0.0
        joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.qpos_ids = np.array([self.model.jnt_qposadr[j] for j in joints], dtype=int)
        self.dof_ids = np.array([self.model.jnt_dofadr[j] for j in joints], dtype=int)
        self.actuator_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES], dtype=int)
        self.tcp_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.control_bounds = self.model.actuator_ctrlrange[self.actuator_ids].copy()
        home = np.asarray(self.config["home_targets_rad"], dtype=float)
        self.data.qpos[self.qpos_ids] = home
        self.data.ctrl[self.actuator_ids] = home
        mujoco.mj_forward(self.model, self.data)
        self._track_penetration()
        for _ in range(round(self.config["settle_seconds"] * self.config["control_hz"])):
            self.step_targets(home)
        self.reset_snapshot = self.snapshot()
        return self.state()

    def state(self):
        return self.data.qpos[self.qpos_ids].copy()

    def _track_penetration(self):
        # Read contacts already computed by MuJoCo. An extra forward call at
        # each substep would alter solver scheduling and is unnecessary here.
        penetration = max(0.0, -float(np.min(self.data.contact.dist, initial=0.0)))
        self.peak_penetration_m = max(self.peak_penetration_m, penetration)

    def step_targets(self, target6):
        values = np.asarray(target6, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("target6 must contain six finite joint targets in radians")
        self.last_action = np.clip(values, self.control_bounds[:, 0], self.control_bounds[:, 1])
        self.data.ctrl[self.actuator_ids] = self.last_action
        for _ in range(self.config["substeps"]):
            mujoco.mj_step(self.model, self.data)
            self._track_penetration()
        # MuJoCo position-dependent fields otherwise lag the final integration.
        mujoco.mj_forward(self.model, self.data)
        self._track_penetration()

    def render(self, camera="top"):
        if camera not in ("top", "wrist"):
            raise ValueError("camera must be 'top' or 'wrist'")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0  # Hide robot collision-only geometry and debug sites.
        option.sitegroup[:] = 0
        self._renderer.update_scene(self.data, camera=camera, scene_option=option)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.config["render"].get("shadows", False)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.config["render"].get("reflections", False)
        return self._renderer.render().copy()

    def brush_body(self, name="red"):
        if name not in ("red", "blue"):
            raise ValueError("brush name must be red/blue")
        return self._id(mujoco.mjtObj.mjOBJ_BODY, f"brush_{name}")

    def cup_body(self, name="left"):
        if name not in ("left", "right"):
            raise ValueError("cup name must be left/right")
        return self._id(mujoco.mjtObj.mjOBJ_BODY, f"cup_{name}")

    def snapshot(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, spec), dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, spec)
        return {"state_spec": int(spec), "state": state.copy(),
                "last_action": self.data.ctrl[self.actuator_ids].copy(),
                "peak_penetration_m": self.peak_penetration_m}

    def restore(self, snapshot):
        spec = mujoco.mjtState(int(snapshot["state_spec"]))
        state = np.asarray(snapshot["state"], dtype=np.float64)
        if state.shape != (mujoco.mj_stateSize(self.model, spec),):
            raise ValueError("Snapshot shape does not match this model")
        peak = float(snapshot.get("peak_penetration_m", 0.0))
        if not np.isfinite(peak) or peak < 0:
            raise ValueError("Snapshot penetration peak must be finite and nonnegative")
        mujoco.mj_setState(self.model, self.data, state, spec)
        self.last_action = np.asarray(snapshot["last_action"], dtype=float).copy()
        self.peak_penetration_m = peak
        mujoco.mj_forward(self.model, self.data)
        self._track_penetration()

    def metadata(self):
        return {"mujoco_version": mujoco.__version__, "scene_schema": 1,
                "measurement_status": self.config["measurement_status"],
                "seed": self.seed, "task_id": self.task_id, "task": self.task_text,
                "joint_names": list(self.joint_names), "state_action_units": "radians",
                "control_hz": self.config["control_hz"], "substeps": self.config["substeps"],
                "physics_timestep": self.model.opt.timestep,
                "layout_offsets": self.layout_offsets, "config": self.config,
                "xml_sha256": hashlib.sha256(
                    self.xml.replace(str(ASSET_DIR), "${SO101_ASSET_DIR}").encode()).hexdigest(),
                "asset_sha256": asset_manifest(),
                "real_interface_mapping_verified": False,
                "hidden_grasp_assistance": False}

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
