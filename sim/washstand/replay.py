"""Replay recorded controls without the privileged expert or object rewrites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .env import WashstandEnv
from .recording import load_episode, validate_episode, write_json
from .validators import SuccessValidator


def replay_provenance():
    """Record the checker actually used, independently of the recording commit."""
    root = Path(__file__).resolve().parent
    return {"replay_validator_sha256": hashlib.sha256((root / "validators.py").read_bytes()).hexdigest(),
            "replay_code_sha256": hashlib.sha256((root / "replay.py").read_bytes()).hexdigest()}


def replay_episode(path, *, update_metadata=True, tolerance=2e-5):
    episode = load_episode(path)
    meta, arrays = episode["metadata"], episode["arrays"]
    errors = []
    data_errors = validate_episode(episode, check_videos=False)
    if data_errors:
        result = {**replay_provenance(), "replay_passed": False,
                  "replay_errors": ["Invalid saved data: "+error for error in data_errors]}
        if update_metadata:
            meta.update(result)
            write_json(episode["root"] / "metadata.json", meta)
        return result
    maximum_error = 0.
    maximum_state_error = maximum_velocity_error = 0.
    with WashstandEnv(seed=meta["seed"], task_id=meta["task_id"], config=meta["config"],
                      width=meta["width"], height=meta["height"]) as env:
        if env.metadata()["xml_sha256"] != meta["xml_sha256"]:
            raise ValueError("Scene content changed; replay cannot validate this episode")
        if env.metadata()["asset_sha256"] != meta["asset_sha256"]:
            raise ValueError("Robot asset content changed")
        if env.metadata()["mujoco_version"] != meta["mujoco_version"]:
            raise ValueError("Use the recorded MuJoCo version for validation")
        env.restore(episode["initial_state"])
        checker = SuccessValidator(env)
        start_time = float(env.data.time)
        check = {}
        for index, target in enumerate(arrays["action"]):
            error = float(np.max(np.abs(env.data.qpos-arrays["qpos"][index])))
            state_error = float(np.max(np.abs(env.state()-arrays["state"][index])))
            velocity_error = float(np.max(np.abs(env.data.qvel-arrays["qvel"][index])))
            maximum_error = max(maximum_error, error)
            maximum_state_error = max(maximum_state_error, state_error)
            maximum_velocity_error = max(maximum_velocity_error, velocity_error)
            if abs(env.data.time-start_time-arrays["timestamps"][index]) > 1e-6:
                errors.append(f"timestamp mismatch at frame {index}")
                break
            if error > tolerance:
                errors.append(f"qpos mismatch at frame {index}: {error}")
                break
            if state_error > tolerance:
                errors.append(f"state mismatch at frame {index}: {state_error}")
                break
            if velocity_error > tolerance*10:
                errors.append(f"qvel mismatch at frame {index}: {velocity_error}")
                break
            env.step_targets(target)
            check = checker.update()
        if not check.get("success"):
            errors.append("Independent final physical success check failed")
    result = {**replay_provenance(), "replay_passed": not errors, "replay_max_qpos_error": maximum_error,
              "replay_max_state_error": maximum_state_error,
              "replay_max_qvel_error": maximum_velocity_error,
              "replay_tolerance": tolerance, "replay_errors": errors,
              "replay_final_check": check}
    if update_metadata:
        meta.update(result)
        write_json(episode["root"] / "metadata.json", meta)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    args = parser.parse_args()
    result = replay_episode(args.episode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["replay_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
