"""Read a local exported dataset, compare every sample to raw, and preview it.

Run in a new process after export, with a writable HF_DATASETS_CACHE (for example
under this project's outputs). No model or robot is loaded and no upload occurs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from .recording import iter_video_frames, load_episode, write_json


class ImageComparison:
    """Bounded-memory same-frame comparison allowing a second H.264 encode.

    Errors use RGB values on the 0..255 scale. These prototype thresholds can
    reject obvious image/camera substitutions; similar adjacent or static
    images can still pass, so passing does not prove strict temporal identity.
    NumPy is imported only when comparing, preserving lightweight imports.
    """

    def __init__(self, mae_threshold=3.0, rmse_threshold=6.0):
        import math

        if not all(math.isfinite(value) and value >= 0 for value in (mae_threshold, rmse_threshold)):
            raise ValueError("Image thresholds must be finite and nonnegative")
        self.mae_threshold, self.rmse_threshold = mae_threshold, rmse_threshold
        self.frames = self.failed_frames = 0
        self.mean_mae = self.mean_rmse = self.maximum_mae = self.maximum_rmse = 0.0

    def compare(self, exported_chw, raw_hwc):
        import numpy as np

        exported = np.asarray(exported_chw)
        raw = np.asarray(raw_hwc)
        if (exported.ndim != 3 or raw.ndim != 3 or exported.shape[0] != 3
                or raw.shape != (exported.shape[1], exported.shape[2], 3)
                or raw.dtype != np.uint8 or not np.isfinite(exported).all()
                or np.any(exported < 0) or np.any(exported > 1)):
            raise ValueError("Expected finite CHW exported RGB in 0..1 and matching uint8 HWC raw RGB")
        difference = exported.transpose(1, 2, 0).astype(np.float64) * 255 - raw.astype(np.float64)
        mae = float(np.abs(difference).mean())
        rmse = float(np.sqrt(np.square(difference).mean()))
        passed = mae <= self.mae_threshold and rmse <= self.rmse_threshold
        self.frames += 1
        self.failed_frames += int(not passed)
        self.mean_mae += (mae - self.mean_mae) / self.frames
        self.mean_rmse += (rmse - self.mean_rmse) / self.frames
        self.maximum_mae = max(self.maximum_mae, mae)
        self.maximum_rmse = max(self.maximum_rmse, rmse)
        return {"mae": mae, "rmse": rmse, "passed": passed}

    def summary(self):
        return {"frames_compared": self.frames, "failed_frames": self.failed_frames,
                "mean_frame_mae": self.mean_mae, "maximum_frame_mae": self.maximum_mae,
                "mean_frame_rmse": self.mean_rmse, "maximum_frame_rmse": self.maximum_rmse,
                "mae_threshold": self.mae_threshold, "rmse_threshold": self.rmse_threshold,
                "pixel_scale": "RGB 0..255", "passed": self.frames > 0 and self.failed_frames == 0}


def preview_indices(stages):
    """Keep first/last and representative grasp/release frames, in time order."""
    selected = {0: "first", len(stages) - 1: "last"}
    for stage, label in (("CLOSE", "grasp"), ("RELEASE", "release")):
        candidates = [index for index, value in enumerate(stages) if value == stage]
        if candidates:
            selected[candidates[len(candidates) // 2]] = label
    return dict(sorted(selected.items()))


def expected_window_indices(frame_index, length, offsets):
    """Reference clipping/padding within ONE episode, never its neighbor."""
    indices = [max(0, min(length - 1, frame_index + offset)) for offset in offsets]
    padding = [frame_index + offset < 0 or frame_index + offset >= length for offset in offsets]
    return indices, padding


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    import numpy as np

    return np.asarray(value)


def _save_preview(path, samples, instruction):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    columns = len(samples)
    first = next(iter(samples.values()))
    height, width = first["top"].shape[1:]
    header, label_height = 34, 24
    image = Image.new("RGB", (width * columns, header + (height + label_height) * 2), "#202124")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), instruction, fill="white", font=font)
    for column, (frame_index, sample) in enumerate(samples.items()):
        for row, camera in enumerate(("top", "wrist")):
            rgb = np.clip(sample[camera].transpose(1, 2, 0) * 255, 0, 255).round().astype(np.uint8)
            x, y = column * width, header + row * (height + label_height)
            image.paste(Image.fromarray(rgb), (x, y + label_height))
            draw.text((x + 5, y + 4), f"{camera} {sample['label']} #{frame_index}", fill="white", font=font)
    image.save(path, quality=92)


def inspect_dataset(dataset_root, output, *, repo_id=None, action_horizon=50):
    """Check all numeric/video frames plus sampled, episode-bounded windows."""
    # Local inspection must never fetch absent metadata/videos from the Hub.
    # These flags affect this process; they do not edit any user configuration.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    started = time.monotonic()
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    if output == dataset_root:
        raise ValueError("Inspection output must be separate from the dataset")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_root / "source_manifest.json").read_text(encoding="utf-8"))
    repo_id = repo_id or manifest["repo_id"]
    report = {
        "dataset_root": str(dataset_root), "process_id": os.getpid(),
        "source": "sim", "video_backend": "pyav", "repo_id": repo_id,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
        "action_horizon": action_horizon, "episodes": [], "errors": [],
        "frames_checked": 0, "window_samples_checked": 0, "upload_performed": False,
        "downloads_permitted": False,
        "model_loaded": False, "model_training_batch_verified": False,
        "real_contract_verified": False, "training_eligible": False,
        "visual_preview_created": False, "human_visual_reviewed": False,
        "image_comparison_note": "Every exported frame is compared to the same indexed raw camera frame; "
                                 "H.264 re-encoding tolerance does not prove strict temporal identity.",
    }
    def check(condition, message):
        if not condition and len(report["errors"]) < 100:
            report["errors"].append(message)

    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav")
    check(manifest.get("status") == "complete", "Source manifest is incomplete")
    check(dataset.meta.total_episodes == len(manifest["episodes"]), "Episode count differs from source manifest")
    expected_total = sum(entry["frame_count"] for entry in manifest["episodes"])
    check(len(dataset) == expected_total, "Total frame count differs from source manifest")
    fps = dataset.fps
    delta = {"action": [index / fps for index in range(action_horizon)],
             "observation.state": [-1 / fps, 0]}
    window_dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav", delta_timestamps=delta)
    image_totals = {camera: ImageComparison() for camera in ("top", "wrist")}
    absolute_start = 0
    for entry in manifest["episodes"]:
        episode_index = entry["episode_index"]
        raw = load_episode(entry["raw_path"])
        arrays, metadata = raw["arrays"], raw["metadata"]
        length = entry["frame_count"]
        check(length == len(arrays["state"]), f"episode {episode_index}: raw length mismatch")
        check(fps == metadata["fps"], f"episode {episode_index}: fps mismatch")
        for field in ("xml_sha256", "asset_sha256", "project_commit", "working_tree_dirty",
                      "expert_version", "scene_schema", "task_id", "group_id", "layout_id", "config",
                      "replay_validator_sha256", "replay_code_sha256"):
            check(entry.get(field) == metadata.get(field), f"episode {episode_index}: manifest {field} mismatch")
        selected = preview_indices(arrays["stage"])
        preview = {}
        raw_videos = {camera: iter_video_frames(raw["root"] / f"{camera}.mp4") for camera in ("top", "wrist")}
        image_comparisons = {camera: ImageComparison() for camera in ("top", "wrist")}
        max_state_error, max_action_error = 0.0, 0.0
        for frame_index in range(length):
            item = dataset[absolute_start + frame_index]
            prefix = f"episode {episode_index}, frame {frame_index}"
            state, action = _array(item["observation.state"]), _array(item["action"])
            check(state.shape == (6,) and action.shape == (6,), f"{prefix}: bad state/action shape")
            state_error = float(np.max(np.abs(state - arrays["state"][frame_index])))
            action_error = float(np.max(np.abs(action - arrays["action"][frame_index])))
            max_state_error, max_action_error = max(max_state_error, state_error), max(max_action_error, action_error)
            check(np.isfinite(state).all() and np.isfinite(action).all(), f"{prefix}: nonfinite state/action")
            check(state_error <= 1e-7 and action_error <= 1e-7, f"{prefix}: state/action differs from raw")
            check(int(item["episode_index"]) == episode_index, f"{prefix}: wrong episode index")
            check(int(item["frame_index"]) == frame_index, f"{prefix}: wrong frame index")
            check(abs(float(item["timestamp"]) - frame_index / fps) <= 2e-6, f"{prefix}: timestamp mismatch")
            check(item["task"] == metadata["instruction"], f"{prefix}: instruction mismatch")
            sample = {"label": selected.get(frame_index, "")}
            for camera in ("top", "wrist"):
                rgb = _array(item[f"observation.images.{camera}"])
                check(rgb.shape == (3, metadata["height"], metadata["width"]), f"{prefix}: {camera} image shape")
                check(np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1,
                      f"{prefix}: invalid {camera} RGB values")
                raw_rgb = next(raw_videos[camera], None)
                check(raw_rgb is not None, f"{prefix}: missing raw {camera} frame")
                if raw_rgb is not None:
                    difference = image_comparisons[camera].compare(rgb, raw_rgb)
                    image_totals[camera].compare(rgb, raw_rgb)
                    check(difference["passed"], f"{prefix}: {camera} differs from same-index raw image "
                          f"(MAE={difference['mae']:.4f}, RMSE={difference['rmse']:.4f} on 0..255 scale)")
                if frame_index in selected:
                    sample[camera] = rgb.copy()
            if frame_index in selected:
                preview[frame_index] = sample
            report["frames_checked"] += 1
        for camera, video in raw_videos.items():
            check(next(video, None) is None, f"episode {episode_index}: extra raw {camera} frames")
            video.close()
        preview_path = output / f"episode-{episode_index:04d}-preview.jpg"
        _save_preview(preview_path, preview, metadata["instruction"])
        window_samples = sorted({0, min(1, length - 1), max(0, length - action_horizon), max(0, length - 2), length - 1})
        for frame_index in window_samples:
            item = window_dataset[absolute_start + frame_index]
            for key, raw_key, offsets in (("action", "action", list(range(action_horizon))),
                                          ("observation.state", "state", [-1, 0])):
                indices, padding = expected_window_indices(frame_index, length, offsets)
                actual, expected = _array(item[key]), arrays[raw_key][indices]
                check(actual.shape == expected.shape and np.allclose(actual, expected, rtol=0, atol=1e-7),
                      f"episode {episode_index}, frame {frame_index}: {key} window crossed boundary or differs")
                check(np.array_equal(_array(item[f"{key}_is_pad"]), padding),
                      f"episode {episode_index}, frame {frame_index}: incorrect {key} padding mask")
            report["window_samples_checked"] += 1
        report["episodes"].append({
            "episode_index": episode_index, "frame_count": length, "task_id": metadata.get("task_id"),
            "instruction": metadata["instruction"], "group_id": metadata.get("group_id"),
            "raw_state_max_abs_error": max_state_error, "raw_action_max_abs_error": max_action_error,
            "image_comparison": {camera: comparison.summary() for camera, comparison in image_comparisons.items()},
            "preview": str(preview_path), "preview_frames": {str(k): v for k, v in selected.items()},
            "window_samples_checked": window_samples,
        })
        absolute_start += length
    report["image_comparison"] = {camera: comparison.summary() for camera, comparison in image_totals.items()}
    batch = next(iter(DataLoader(window_dataset, batch_size=min(2, len(window_dataset)), shuffle=False, num_workers=0)))
    report["data_loader_batch_shapes"] = {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")}
    report["data_loader_batch_verified"] = True
    report["visual_preview_created"] = True
    report["wall_seconds"] = time.monotonic() - started
    report["passed"] = not report["errors"]
    write_json(output / "inspection.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--cache-dir", type=Path, help="Set HF_DATASETS_CACHE for this process only")
    args = parser.parse_args(argv)
    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = str(args.cache_dir.resolve())
    try:
        report = inspect_dataset(args.dataset, args.output, repo_id=args.repo_id, action_horizon=args.action_horizon)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"passed": False, "fatal_error": f"{type(exc).__name__}: {exc}",
                   "dataset_root": str(args.dataset.resolve()), "model_loaded": False,
                   "downloads_permitted": False, "upload_performed": False}
        write_json(args.output / "inspection.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({key: report[key] for key in ("passed", "frames_checked", "window_samples_checked",
                                                 "data_loader_batch_shapes", "errors")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
