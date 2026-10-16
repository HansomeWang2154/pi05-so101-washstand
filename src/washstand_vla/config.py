"""Load project JSON separately from upstream LeRobot configuration."""

import copy
import json
from pathlib import Path


def read_json(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def merge_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path, local_path: Path | None = None) -> dict:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Project config must be a JSON object")
    if local_path is not None:
        override = read_json(local_path)
        if not isinstance(override, dict):
            raise ValueError("Local config must be a JSON object")
        config = merge_dicts(config, override)
    return config


def validate_config(config: dict) -> dict:
    errors, missing = [], []
    sections = ("project", "runtime", "robot", "cameras", "data", "tasks", "policy", "training", "deployment", "simulation")
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for name in sections:
        if not isinstance(config.get(name), dict):
            errors.append(f"{name} must be an object")
    if errors:
        return {"valid": False, "complete": False, "errors": errors, "missing": missing}
    required = {
        "robot": ("type", "id", "port", "leader_type", "leader_id", "leader_port", "use_degrees", "joint_names"),
        "data": ("root", "repo_id", "fps", "episodes_per_record", "episode_time_s", "reset_time_s"),
        "policy": ("pretrained_path",),
        "training": ("num_workers", "seed", "output_dir"),
        "deployment": ("checkpoint", "server_host", "server_port", "server_address", "policy_device", "actions_per_chunk"),
    }
    for name, fields in required.items():
        for field in fields:
            if field not in config[name]:
                errors.append(f"Missing required field: {name}.{field}")
    if errors:
        return {"valid": False, "complete": False, "errors": errors, "missing": missing}
    if config["robot"]["type"] != "so101_follower" or config["robot"]["leader_type"] != "so101_leader":
        errors.append("The initial profile requires so101_follower and so101_leader")
    if type(config["robot"]["use_degrees"]) is not bool:
        errors.append("robot.use_degrees must be boolean")
    if config["robot"]["port"] and config["robot"]["port"] == config["robot"]["leader_port"]:
        errors.append("Follower and leader must use different serial ports")
    policy = config["policy"]
    if policy.get("type") != "pi05":
        errors.append("policy.type must be pi05 for this project")
    for field in ("action_dim", "state_dim"):
        if policy.get(field) != 6:
            errors.append(f"policy.{field} must be 6 for SO101")
    joints = config["robot"].get("joint_names", [])
    if (not isinstance(joints, list) or len(joints) != 6
            or any(not isinstance(j, str) or not j for j in joints) or len(set(joints)) != 6):
        errors.append("robot.joint_names must contain six distinct names")
    for section, field in (("policy", "chunk_size"), ("policy", "n_action_steps"), ("policy", "num_inference_steps"),
                           ("training", "batch_size"), ("training", "steps"), ("training", "save_freq"),
                           ("training", "log_freq"), ("data", "fps"), ("data", "episodes_per_record")):
        value = config[section].get(field)
        if type(value) is not int or value <= 0:
            errors.append(f"{section}.{field} must be a positive integer")
    if all(type(policy.get(k)) is int for k in ("chunk_size", "n_action_steps")):
        if policy["n_action_steps"] > policy["chunk_size"]:
            errors.append("policy.n_action_steps cannot exceed chunk_size")
    if policy.get("dtype") not in ("bfloat16", "float32"):
        errors.append("policy.dtype must be bfloat16 or float32")
    for field in ("train_expert_only", "freeze_vision_encoder", "gradient_checkpointing"):
        if type(policy.get(field)) is not bool:
            errors.append(f"policy.{field} must be boolean")
    if not policy.get("train_expert_only"):
        errors.append("The initial project profile requires train_expert_only=true")
    for name in ("top", "wrist"):
        camera = config["cameras"].get(name)
        if not isinstance(camera, dict):
            errors.append(f"cameras.{name} must be an object")
            continue
        for field in ("width", "height", "fps"):
            if type(camera.get(field)) is not int or camera[field] <= 0:
                errors.append(f"cameras.{name}.{field} must be a positive integer")
        if camera.get("fps") != config["data"].get("fps"):
            errors.append(f"cameras.{name}.fps must match data.fps")
        if camera.get("index_or_path") is None:
            missing.append(f"cameras.{name}.index_or_path")
    top, wrist = config["cameras"].get("top", {}), config["cameras"].get("wrist", {})
    if isinstance(top, dict) and isinstance(wrist, dict):
        top_id, wrist_id = top.get("index_or_path"), wrist.get("index_or_path")
        if top_id is not None and top_id == wrist_id:
            errors.append("top and wrist cannot refer to the same camera identifier")
    for field in ("port", "leader_port"):
        if not config["robot"].get(field):
            missing.append(f"robot.{field}")
    if not config["data"].get("repo_id"):
        missing.append("data.repo_id")
    if not config["tasks"] or any(not isinstance(v, str) or not v.strip() for v in config["tasks"].values()):
        errors.append("tasks must map task IDs to non-empty instructions")
    probability = config["simulation"].get("training_source_real_probability")
    if isinstance(probability, bool) or not isinstance(probability, (float, int)) or not 0 <= probability <= 1:
        errors.append("simulation.training_source_real_probability must be in [0,1]")
    threshold = config["deployment"].get("chunk_size_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
        errors.append("deployment.chunk_size_threshold must be in [0,1]")
    port = config["deployment"].get("server_port")
    if type(port) is not int or not 1 <= port <= 65535:
        errors.append("deployment.server_port must be in [1,65535]")
    return {"valid": not errors, "complete": not errors and not missing,
            "errors": errors, "missing": missing,
            "note": "Structural validation does not verify hardware, calibration, dependencies or training readiness."}
