"""Export qualified native-unit simulation attempts, locally and without upload.

This produces robot_type=so101_sim, not a verified real-driver training dataset.
The independent source manifest keeps instruction/layout groups out of policy
features. LeRobot 0.6.2 writer timestamps are generated from the verified fps.
"""

from __future__ import annotations

import argparse
import json
from itertools import zip_longest
from pathlib import Path

from .recording import JOINT_NAMES, iter_video_frames, load_episode, validate_episode, write_json


def qualification_errors(episode):
    meta = episode["metadata"]
    errors = [f"{key} is not true" for key in (
        "physics_success", "replay_passed", "data_checks_passed", "images_present"
    ) if meta.get(key) is not True]
    if errors:
        return errors
    return validate_episode(episode, check_videos=True)


def source_entry(episode, episode_index):
    """Keep the actual environment provenance names alongside exported IDs."""
    meta = episode["metadata"]
    return {
        "episode_index": episode_index, "raw_path": str(episode["root"]),
        "instruction": meta["instruction"], "layout_id": meta.get("layout_id"),
        "group_id": meta.get("group_id", meta.get("layout_id")), "seed": meta.get("seed"),
        "task_id": meta.get("task_id"), "scene_schema": meta.get("scene_schema"),
        "xml_sha256": meta.get("xml_sha256"), "asset_sha256": meta.get("asset_sha256"),
        "project_commit": meta.get("project_commit"),
        "working_tree_dirty": meta.get("working_tree_dirty"),
        "expert_version": meta.get("expert_version"), "config": meta.get("config"),
        "metadata_maintenance": meta.get("metadata_maintenance"),
        "replay_validator_sha256": meta.get("replay_validator_sha256"),
        "frame_count": meta["frame_count"], "evidence_kind": meta.get("evidence_kind"),
    }


def export_episodes(episode_paths, output, repo_id="local/washstand_sim_v1", max_episodes=None):
    """Filter, verify, write and finalize a local LeRobot v3 simulation dataset."""
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    accepted, rejected = [], []
    seen = set()
    for path in episode_paths:
        root = Path(path).resolve()
        if root in seen:
            continue
        seen.add(root)
        try:
            episode = load_episode(root)
            errors = qualification_errors(episode)
        except Exception as exc:
            errors, episode = [str(exc)], None
        if errors:
            rejected.append({"raw_path": str(root), "reasons": errors})
        elif max_episodes is None or len(accepted) < max_episodes:
            # Keep only lightweight descriptors, not every episode's arrays.
            accepted.append({"root": root, "metadata": episode["metadata"]})
    if not accepted:
        raise ValueError("No qualified episodes; no output created. Rejections: " + json.dumps(rejected))
    first = accepted[0]["metadata"]
    for episode in accepted:
        if any(episode["metadata"].get(key) != first.get(key) for key in ("fps", "height", "width", "joint_names")):
            raise ValueError("Qualified episodes have incompatible fps/image/joint contracts")
    # Lazy imports preserve the lightweight project CLI and ordinary test suite.
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        **{f"observation.images.{camera}": {
            "dtype": "video", "shape": (first["height"], first["width"], 3),
            "names": ["height", "width", "channels"],
        } for camera in ("top", "wrist")},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id, root=output, fps=first["fps"], features=features,
        robot_type="so101_sim", use_videos=True, video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="veryfast", g=2),
        image_writer_threads=2, streaming_encoding=False, encoder_threads=2,
    )
    manifest = {
        "source": "sim", "repo_id": repo_id, "robot_type": "so101_sim",
        "real_contract_verified": False, "training_eligible": False,
        "unit_contract": "native MuJoCo radians; NOT real driver calibrated units",
        "qualification": "physics_success + replay_passed + data_checks_passed + full dual video decode",
        "normalization_scope": "exported simulation episodes only; no real train/test equivalence implied",
        "episodes": [], "rejected": rejected, "upload_performed": False,
        "status": "writing",
    }
    try:
        for index, descriptor in enumerate(accepted):
            episode = load_episode(descriptor["root"])
            arrays, meta = episode["arrays"], episode["metadata"]
            top = iter_video_frames(episode["root"] / "top.mp4")
            wrist = iter_video_frames(episode["root"] / "wrist.mp4")
            for frame_index, (top_frame, wrist_frame) in enumerate(zip_longest(top, wrist)):
                if top_frame is None or wrist_frame is None or frame_index >= len(arrays["state"]):
                    raise ValueError("Video length changed after validation")
                dataset.add_frame({
                    "observation.state": arrays["state"][frame_index].astype("float32"),
                    "action": arrays["action"][frame_index].astype("float32"),
                    "observation.images.top": top_frame, "observation.images.wrist": wrist_frame,
                    "task": meta["instruction"],
                })
            dataset.save_episode(parallel_encoding=False)
            manifest["episodes"].append(source_entry(episode, index))
        manifest["status"] = "finalizing"
    except Exception as exc:
        manifest["status"], manifest["error"] = "incomplete", str(exc)
        raise
    finally:
        try:
            dataset.finalize()
            if manifest["status"] == "finalizing":
                manifest["status"] = "complete"
        except Exception as exc:
            manifest["status"], manifest["error"] = "incomplete", f"Finalization failed: {exc}"
            raise
        finally:
            write_json(output / "source_manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--raw-root", type=Path, help="Find raw metadata.json files recursively")
    selection.add_argument("--episodes", nargs="+", type=Path, help="Explicit raw attempt directories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/washstand_sim_v1")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args(argv)
    paths = args.episodes or [path.parent for path in sorted(args.raw_root.rglob("metadata.json"))]
    manifest = export_episodes(paths, args.output, args.repo_id, args.max_episodes)
    print(json.dumps({"output": str(args.output.resolve()), "episodes": len(manifest["episodes"]),
                      "rejected": len(manifest["rejected"]), "training_eligible": False,
                      "upload_performed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
