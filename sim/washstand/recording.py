"""Bounded-image-memory recording for native-unit MuJoCo attempts.

Only scalar/joint arrays are buffered; camera frames are encoded immediately.
These raw radians are deliberately not advertised as a verified real SO101
driver contract. A successful simulation does not establish sim-to-real fit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


JOINT_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
SCHEMA_VERSION = "washstand-sim-raw-v1"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path, value):
    """Atomically replace a small UTF-8 metadata/manifest file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


class _VideoWriter:
    def __init__(self, path, fps, width, height):
        import av

        self.av = av
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "veryfast"}
        self.count = 0

    def append(self, rgb):
        frame = self.av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self.count
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


class EpisodeRecorder:
    """Record observations at t, actual sent targets at t, before stepping.

    ``initial_state`` must contain numeric arrays/scalars (for example MuJoCo's
    complete integration state and state spec); object arrays are prohibited.
    ``top`` and ``wrist`` are uint8 HWC RGB or both None for a debug-only attempt.
    Do not mix image and image-less samples inside one attempt.
    """

    def __init__(self, output_dir, metadata, initial_state, fps=30, width=320, height=240):
        import numpy as np

        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v % 2
               for v in (width, height)):
            raise ValueError("width and height must be positive even integers")
        self.root = Path(output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.fps, self.width, self.height = fps, width, height
        self.metadata = dict(metadata)
        self.metadata.update({
            "schema_version": SCHEMA_VERSION, "source": "sim", "robot_type": "so101_sim",
            "joint_names": JOINT_NAMES, "state_units": "rad", "action_units": "rad",
            "action_semantics": "actual_sent_position_target",
            "sample_alignment": "observation_t_and_action_t_before_step",
            "real_contract_verified": False, "training_eligible": False,
            "fps": fps, "width": width, "height": height,
            "images_present": False, "physics_success": False,
            "replay_passed": False, "data_checks_passed": False, "status": "recording",
        })
        initial = {key: np.asarray(value).copy() for key, value in initial_state.items()}
        if any(value.dtype.kind not in "biuf" or not np.isfinite(value).all()
               for value in initial.values()):
            raise ValueError("initial_state must only contain finite numeric arrays")
        np.savez_compressed(self.root / "initial_state.npz", **initial)
        write_json(self.root / "metadata.json", self.metadata)
        self.rows = {key: [] for key in ("timestamps", "state", "action", "qpos", "qvel", "stage")}
        self._writers = {}
        self._image_mode = None
        self._finished = False
        self._diagnostics = (self.root / "diagnostics.jsonl").open("w", encoding="utf-8")

    def append(self, timestamp, state, action, qpos, qvel, top, wrist, stage, diagnostics=None):
        import numpy as np

        if self._finished:
            raise RuntimeError("Recorder is already finished")
        if not math.isfinite(float(timestamp)):
            raise ValueError("timestamp must be finite")
        if self.rows["timestamps"] and timestamp <= self.rows["timestamps"][-1]:
            raise ValueError("timestamps must increase strictly")
        values = {}
        for name, value in (("state", state), ("action", action), ("qpos", qpos), ("qvel", qvel)):
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite 1D array")
            if name in ("state", "action") and array.shape != (6,):
                raise ValueError(f"{name} must contain six native-unit joint values")
            if self.rows[name] and array.shape != self.rows[name][0].shape:
                raise ValueError(f"{name} size changed inside an episode")
            values[name] = array.copy()
        image_mode = top is not None and wrist is not None
        if (top is None) != (wrist is None):
            raise ValueError("Both top and wrist must be provided together")
        if self._image_mode is not None and self._image_mode != image_mode:
            raise ValueError("Cannot mix image and image-less samples")
        images = {}
        if image_mode:
            for camera, value in (("top", top), ("wrist", wrist)):
                array = np.asarray(value)
                if array.dtype != np.uint8 or array.shape != (self.height, self.width, 3):
                    raise ValueError(f"{camera} must be uint8 RGB {self.height}x{self.width}x3")
                images[camera] = np.ascontiguousarray(array)
        diagnostics_line = json.dumps({"frame_index": len(self.rows["timestamps"]),
                                       "timestamp": float(timestamp), "stage": str(stage),
                                       "diagnostics": diagnostics or {}},
                                      allow_nan=False, default=_json_default)
        if self._image_mode is None:
            self._image_mode = image_mode
            if image_mode:
                self._writers = {camera: _VideoWriter(self.root / f"{camera}.mp4", self.fps,
                                                       self.width, self.height)
                                 for camera in ("top", "wrist")}
        for camera, rgb in images.items():
            self._writers[camera].append(rgb)
        for name, value in values.items():
            self.rows[name].append(value)
        self.rows["timestamps"].append(float(timestamp))
        self.rows["stage"].append(str(stage))
        self._diagnostics.write(diagnostics_line + "\n")

    def finish(self, outcome):
        import numpy as np

        if self._finished:
            return self.metadata
        self._finished = True
        errors = []
        for camera, writer in self._writers.items():
            try:
                writer.close()
            except Exception as exc:
                errors.append(f"{camera} video finalization: {exc}")
        self._diagnostics.close()
        arrays = {}
        for key, rows in self.rows.items():
            dtype = str if key == "stage" else (np.float32 if key in ("state", "action") else np.float64)
            arrays[key] = np.asarray(rows, dtype=dtype)
        np.savez_compressed(self.root / "trajectory.npz", **arrays)
        count = len(arrays["timestamps"])
        if count < 2:
            errors.append("At least two frames are required")
        if count >= 2 and not np.allclose(np.diff(arrays["timestamps"]), 1 / self.fps,
                                          rtol=0, atol=1e-6):
            errors.append("Timestamps do not match the declared constant fps")
        self.metadata.update(dict(outcome))
        # Qualification cannot be granted by an expert's outcome dictionary.
        self.metadata.update({
            "source": "sim", "robot_type": "so101_sim", "state_units": "rad", "action_units": "rad",
            "schema_version": SCHEMA_VERSION, "joint_names": JOINT_NAMES,
            "fps": self.fps, "width": self.width, "height": self.height,
            "real_contract_verified": False, "training_eligible": False,
            "frame_count": count, "images_present": bool(self._image_mode) and not errors,
            "data_checks_passed": not errors, "data_check_errors": errors,
            "duration_s": (arrays["timestamps"][-1] - arrays["timestamps"][0]) if count else 0,
            "status": "finished",
            "video_encoding": {"codec": "h264", "pixel_format": "yuv420p", "crf": 18,
                               "lossless": False} if self._image_mode else None,
        })
        write_json(self.root / "metadata.json", self.metadata)
        return self.metadata


def load_episode(path):
    """Read a raw attempt. ``root`` is its resolved directory; arrays are copied.

    Images stay on disk; use ``iter_video_frames`` for bounded-memory decoding.
    This loader does not certify physics, replay or eligibility.
    """
    import numpy as np

    root = Path(path).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    with np.load(root / "trajectory.npz", allow_pickle=False) as source:
        arrays = {key: value.copy() for key, value in source.items()}
    with np.load(root / "initial_state.npz", allow_pickle=False) as source:
        initial = {key: value.copy() for key, value in source.items()}
    return {"root": root, "metadata": metadata, "arrays": arrays, "initial_state": initial}


def iter_video_frames(path):
    """Yield decoded RGB frames without accumulating a whole episode in RAM."""
    import av

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def validate_episode(episode, check_videos=True):
    """Recheck stored data, independent of metadata's own pass flag."""
    import numpy as np

    errors = []
    meta, arrays = episode["metadata"], episode["arrays"]
    if meta.get("schema_version") != SCHEMA_VERSION:
        errors.append("Unknown raw schema")
    if meta.get("joint_names") != JOINT_NAMES:
        errors.append("Unexpected joint order")
    count = len(arrays.get("timestamps", []))
    if count < 2 or meta.get("frame_count") != count:
        errors.append("Invalid frame_count")
    for name in ("state", "action", "qpos", "qvel"):
        value = arrays.get(name)
        if value is None or value.ndim != 2 or value.shape[0] != count or not np.isfinite(value).all():
            errors.append(f"Invalid {name} shape or values")
        elif name in ("state", "action") and value.shape[1] != 6:
            errors.append(f"Invalid {name} dimension")
    timestamps = arrays.get("timestamps")
    fps = meta.get("fps", 0)
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        errors.append("Invalid fps")
    elif timestamps is None or timestamps.ndim != 1 or not np.isfinite(timestamps).all() or (
        count >= 2 and not np.allclose(np.diff(timestamps), 1 / fps, rtol=0, atol=1e-6)
    ):
        errors.append("Invalid timestamp cadence")
    if not isinstance(meta.get("instruction"), str) or not meta["instruction"].strip():
        errors.append("Missing instruction")
    if len(arrays.get("stage", [])) != count:
        errors.append("Invalid stage count")
    if meta.get("source") != "sim" or meta.get("state_units") != "rad" or meta.get("action_units") != "rad":
        errors.append("Unexpected source or units")
    if check_videos:
        import av

        for camera in ("top", "wrist"):
            frames = 0
            try:
                with av.open(str(episode["root"] / f"{camera}.mp4")) as container:
                    for frame in container.decode(video=0):
                        if frame.time is None or abs(float(frame.time) - frames / fps) > 1e-5:
                            raise ValueError("Video timestamps disagree with control cadence")
                        if (frame.height, frame.width) != (meta.get("height"), meta.get("width")):
                            raise ValueError("Unexpected image dimensions")
                        frames += 1
                if frames != count:
                    errors.append(f"{camera} decoded frame count {frames} != {count}")
            except Exception as exc:
                errors.append(f"{camera} decoding failed: {exc}")
    return errors
