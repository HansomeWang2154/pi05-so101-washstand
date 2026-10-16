"""Command-line entry points for local preparation and offline checks."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, read_json, validate_config
from .doctor import inspect_environment
from .planning import build_plan


def write_result(result, output: Path | None = None):
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if output is not None:
        if output.exists():
            raise ValueError(f"Refusing to overwrite {output}; choose a new output path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SO101 pi05 project: preparation tools, no hardware execution")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: working directory)")
    parser.add_argument("--config", type=Path, help="Project JSON config; not a LeRobot train_config")
    parser.add_argument("--local-config", type=Path, help="Explicit local override JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Read-only environment report")
    doctor.add_argument("--gpu", action="store_true")
    doctor.add_argument("--output", type=Path)
    validate = commands.add_parser("validate-config", help="Check structure and missing device fields")
    validate.add_argument("--require-complete", action="store_true")
    validate.add_argument("--output", type=Path)
    plan = commands.add_parser("plan", help="Preview a command without executing it")
    plan.add_argument("stage", choices=("record", "train", "server", "client"))
    plan.add_argument("--task", default="red_left")
    plan.add_argument("--split", type=Path)
    plan.add_argument("--output", type=Path)
    inspect = commands.add_parser("inspect-dataset", help="Inspect metadata; not a complete video QA")
    inspect.add_argument("dataset_root", type=Path)
    inspect.add_argument("--output", type=Path)
    split = commands.add_parser("split-dataset", help="Split episode manifest by whole groups")
    split.add_argument("manifest", type=Path)
    split.add_argument("--group-key", default="session_id")
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=1000)
    split.add_argument("--output", type=Path, required=True)
    commands.add_parser("status", help="Display recorded project progress")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "doctor":
            write_result(inspect_environment(root, args.gpu), args.output)
        elif args.command == "inspect-dataset":
            from .datasets import inspect_dataset
            result = inspect_dataset(args.dataset_root)
            write_result(result, args.output)
            return 2 if result["status"] == "error" else 0
        elif args.command == "split-dataset":
            from .datasets import make_episode_split
            episodes = read_json(args.manifest)
            write_result(make_episode_split(episodes, args.validation_fraction, args.seed, args.group_key), args.output)
        elif args.command == "status":
            write_result(read_json(root / "project_status.json"))
        else:
            config_path = args.config or root / "configs/project.json"
            local_path = args.local_config
            if local_path is None and (root / "configs/local.json").exists():
                local_path = root / "configs/local.json"
            config = load_config(config_path, local_path)
            if args.command == "validate-config":
                result = validate_config(config)
                write_result(result, args.output)
                return 2 if not result["valid"] or (args.require_complete and not result["complete"]) else 0
            write_result(build_plan(config, args.stage, root, args.task, args.split), args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0
