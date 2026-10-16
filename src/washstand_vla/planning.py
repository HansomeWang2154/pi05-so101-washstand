"""Produce command previews only. No generated command is executed here."""

import json
import shlex
from pathlib import Path

from .config import read_json, validate_config


def build_plan(config: dict, stage: str, root: Path, task: str = "red_left", split_path: Path | None = None) -> dict:
    report = validate_config(config)
    if not report["valid"]:
        raise ValueError("; ".join(report["errors"]))
    if task not in config["tasks"]:
        raise ValueError(f"Unknown task: {task}")
    missing = []

    def value(section, key, placeholder):
        result = config[section].get(key)
        if result is None or result == "":
            missing.append(f"{section}.{key}")
            return placeholder
        return result

    def flag(key, val):
        if isinstance(val, bool):
            val = str(val).lower()
        elif isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        return f"--{key}={val}"

    def robot_flags(with_leader=False):
        cameras = {}
        for name in ("top", "wrist"):
            cameras[name] = dict(config["cameras"][name])
            if cameras[name]["index_or_path"] is None:
                missing.append(f"cameras.{name}.index_or_path")
                cameras[name]["index_or_path"] = f"__{name.upper()}_CAMERA__"
        args = [flag("robot.type", config["robot"]["type"]),
                flag("robot.id", config["robot"]["id"]),
                flag("robot.port", value("robot", "port", "__FOLLOWER_PORT__")),
                flag("robot.use_degrees", config["robot"]["use_degrees"]), flag("robot.cameras", cameras)]
        if config["robot"].get("max_relative_target") is not None:
            args.append(flag("robot.max_relative_target", config["robot"]["max_relative_target"]))
        if with_leader:
            args += [flag("teleop.type", config["robot"]["leader_type"]),
                     flag("teleop.id", config["robot"]["leader_id"]),
                     flag("teleop.port", value("robot", "leader_port", "__LEADER_PORT__"))]
        if not config["robot"].get("calibration_verified"):
            missing.append("robot.calibration_verified")
        return args

    if stage == "train":
        argv = ["python", "-m", "lerobot.scripts.lerobot_train", flag("policy.type", "pi05"),
                flag("policy.pretrained_path", config["policy"]["pretrained_path"]),
                flag("dataset.repo_id", value("data", "repo_id", "__DATASET_REPO_ID__")),
                flag("dataset.root", config["data"]["root"]),
                flag("dataset.video_backend", config["data"].get("video_backend", "pyav"))]
        selected_split = split_path or config["data"].get("split_path")
        if selected_split:
            selected_split = Path(selected_split)
            if not selected_split.is_absolute():
                selected_split = root / selected_split
            split = read_json(selected_split)
            episodes, validation = split.get("train", []), split.get("validation", [])
            if not episodes or any(type(x) is not int or x < 0 for x in episodes + validation):
                raise ValueError("LeRobot train/validation IDs must be non-negative integers and train must be non-empty")
            if len(set(episodes)) != len(episodes) or set(episodes) & set(validation):
                raise ValueError("Duplicate training IDs or overlap between train and validation")
            argv.append(flag("dataset.episodes", episodes))
        else:
            missing.append("data.split_path")
        for key in ("dtype", "train_expert_only", "freeze_vision_encoder", "gradient_checkpointing", "chunk_size", "n_action_steps", "num_inference_steps"):
            argv.append(flag(f"policy.{key}", config["policy"][key]))
        argv += [flag("policy.device", "cuda"), flag("policy.push_to_hub", False),
                 flag("policy.compile_model", False), flag("env_eval_freq", 0), flag("wandb.enable", False)]
        argv += [flag(k, config["training"][k]) for k in ("batch_size", "steps", "num_workers", "seed", "save_freq", "log_freq", "output_dir")]
        if not config["data"].get("train_statistics_verified"):
            missing.append("data.train_statistics_verified")
        if not config["robot"].get("action_contract_verified"):
            missing.append("robot.action_contract_verified")
    elif stage == "record":
        argv = ["python", "-m", "lerobot.scripts.lerobot_record"] + robot_flags(with_leader=True)
        argv += [flag("dataset.repo_id", value("data", "repo_id", "__DATASET_REPO_ID__")),
                 flag("dataset.root", config["data"]["root"]), flag("dataset.single_task", config["tasks"][task]),
                 flag("dataset.fps", config["data"]["fps"]),
                 flag("dataset.num_episodes", config["data"]["episodes_per_record"]),
                 flag("dataset.episode_time_s", config["data"]["episode_time_s"]),
                 flag("dataset.reset_time_s", config["data"]["reset_time_s"]),
                 flag("dataset.push_to_hub", False), flag("display_data", False)]
    elif stage == "server":
        argv = ["python", "-m", "lerobot.async_inference.policy_server",
                flag("host", config["deployment"]["server_host"]), flag("port", config["deployment"]["server_port"])]
    elif stage == "client":
        argv = ["python", "-m", "lerobot.async_inference.robot_client"] + robot_flags()
        argv += [flag("policy_type", "pi05"),
                 flag("pretrained_name_or_path", value("deployment", "checkpoint", "__CHECKPOINT_PATH__")),
                 flag("server_address", config["deployment"]["server_address"]),
                 flag("policy_device", config["deployment"]["policy_device"]),
                 flag("client_device", "cpu"), flag("fps", config["data"]["fps"]),
                 flag("actions_per_chunk", config["deployment"]["actions_per_chunk"]),
                 flag("chunk_size_threshold", config["deployment"]["chunk_size_threshold"]),
                 flag("task", config["tasks"][task])]
        if not config["robot"].get("action_contract_verified"):
            missing.append("robot.action_contract_verified")
    else:
        raise ValueError(f"Unsupported plan stage: {stage}")
    if not config["runtime"].get("environment_verified"):
        missing.append("runtime.environment_verified")
    return {"stage": stage, "execution_supported": False, "configuration_complete": not missing,
            "missing": sorted(set(missing)), "argv": argv, "command_preview": shlex.join(argv),
            "notes": ["Preview only; does not start LeRobot, rent GPUs, upload data or move hardware.",
                      "Verify flags against the installed LeRobot commit before execution.",
                      "Training episode selection alone does not recompute train-only normalization statistics.",
                      "Record preview creates a new session; multi-session merge/resume must be planned explicitly."]}
