"""Generate auditable nominal SO101 contact-expert attempts locally."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

from .env import WashstandEnv
from .expert import ScriptedExpert
from .recording import EpisodeRecorder, write_json
from .replay import replay_episode
from .scene import TASKS, load_config
from .validators import SuccessValidator


def layout_group(env):
    """Identical actual initial layouts share a group even with different seeds."""
    poses = {name: env.data.qpos[env.model.jnt_qposadr[env.model.joint(name+"_free").id]:
                                env.model.jnt_qposadr[env.model.joint(name+"_free").id]+7].round(6).tolist()
             for name in ("brush_red", "brush_blue", "cup_left", "cup_right")}
    identity = json.dumps(poses, sort_keys=True, separators=(",", ":"))
    return "layout-"+hashlib.sha256(identity.encode()).hexdigest()[:16]


def git_version():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"project_commit": git("rev-parse", "HEAD"),
            "project_branch": git("branch", "--show-current"),
            "working_tree_dirty": bool(git("status", "--porcelain"))}


def generate_attempt(path, *, seed, task_id, config, images=True, width=320, height=240):
    started = time.monotonic()
    with WashstandEnv(seed=seed, task_id=task_id, width=width, height=height, config=config) as env:
        meta = env.metadata()
        meta.update(git_version())
        group = layout_group(env)
        meta.update({"instruction": TASKS[task_id], "layout_id": group,
                     "group_id": group, "expert_version": ScriptedExpert.VERSION,
                     "record_images": bool(images),
                     "evidence_kind": "scripted_contact_expert",
                     "qualification_scope": "native-radian simulation only; real mapping unverified"})
        recorder = EpisodeRecorder(path, meta, env.snapshot(), fps=config["control_hz"],
                                   width=width, height=height)
        checker = SuccessValidator(env)
        def on_frame(stage, target, expert):
            recorder.append(timestamp=env.data.time-expert.start_time,
                            state=env.state(), action=target, qpos=env.data.qpos,
                            qvel=env.data.qvel, top=env.render("top") if images else None,
                            wrist=env.render("wrist") if images else None, stage=stage,
                            diagnostics={"previous_check": expert.last_check})
        expert = ScriptedExpert(env, checker, on_frame)
        outcome = expert.run()
        outcome["wall_seconds"] = time.monotonic()-started
        result = recorder.finish(outcome)
    if result["physics_success"]:
        result.update(replay_episode(path))
    return result


@contextmanager
def run_lock(output):
    """Lock the whole run, including reconciliation, without waiting on a peer.

    Keep the lock file after release: unlinking it can let two processes hold
    locks on different inodes while believing they own the same run.
    """
    with (Path(output) / ".run.lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Run is already locked by another generator: {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n"
                                  for row in rows), encoding="utf-8")
    temporary.replace(path)


def _run_config(args, config):
    return {"schema_version": "washstand-generation-run-v1", "attempts": args.attempts,
            "tasks": list(args.tasks), "seed": args.seed,
            "width": args.width, "height": args.height, "images": not args.no_images,
            "expert_version": ScriptedExpert.VERSION, "effective_config": config}


def _planned_attempt(args, index):
    task = args.tasks[index % len(args.tasks)]
    return {"attempt": f"attempt-{index:04d}-{task}", "task_id": task,
            "seed": args.seed + index // len(args.tasks)}


def _image_intent(path, metadata):
    """Infer legacy intent without treating recording/images_present=False as no-images.

    Old recorders only set images_present at finish. An empty/incomplete legacy
    attempt cannot independently prove its image mode; the other completed
    metadata or run manifest establish that mode, while its bytes are archived.
    """
    if isinstance(metadata.get("record_images"), bool):
        return metadata["record_images"]
    if metadata.get("images_present") or metadata.get("video_encoding"):
        return True
    if any((path / f"{camera}.mp4").exists() for camera in ("top", "wrist")):
        return True
    if (metadata.get("status") in ("finished", "failed")
            and metadata.get("frame_count", 0) > 0
            and not metadata.get("data_check_errors")):
        return False
    return None


def _existing_attempts(args, config):
    """Validate every directory before changing manifests or archiving anything."""
    rows = {}
    for path in sorted(args.output.glob("attempt-*")):
        match = re.fullmatch(r"attempt-(\d+)-(.+)", path.name)
        if not path.is_dir() or not match:
            raise ValueError(f"Unexpected attempt entry: {path}")
        index = int(match[1])
        if index >= args.attempts or path.name != _planned_attempt(args, index)["attempt"]:
            raise ValueError(f"Existing attempt does not match requested tasks/attempts: {path.name}")
        if index in rows:
            raise ValueError(f"Duplicate attempt index: {index}")
        metadata_path = path / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.exists() else None
        if metadata is not None:
            expected = {**_planned_attempt(args, index), "width": args.width,
                        "height": args.height, "config": config}
            for key in ("task_id", "seed", "width", "height", "config"):
                if metadata.get(key) != expected[key]:
                    raise ValueError(f"Resume mismatch in {path.name}: {key}")
            if metadata.get("expert_version", ScriptedExpert.VERSION) != ScriptedExpert.VERSION:
                raise ValueError(f"Resume mismatch in {path.name}: expert_version")
            intent = _image_intent(path, metadata)
            if intent is not None and intent != (not args.no_images):
                raise ValueError(f"Resume mismatch in {path.name}: images")
            if metadata.get("status") == "finished":
                required = ["initial_state.npz", "trajectory.npz"]
                if metadata.get("images_present"):
                    required += ["top.mp4", "wrist.mp4"]
                missing = [name for name in required if not (path / name).is_file()]
                if missing:
                    raise ValueError(f"Finished attempt has missing artifacts {missing}: {path.name}")
        rows[index] = (path, metadata)
    return rows


def _archive_interrupted(output, path, metadata):
    """Rename intact episode bytes; interruption bookkeeping stays outside them."""
    directory = output / "interrupted"
    directory.mkdir(exist_ok=True)
    identity = f"{path.name}--{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    destination = directory / identity
    path.rename(destination)
    event = {"interruption_id": identity, "event": "system_interruption",
             "attempt": path.name, "archive": str(destination.relative_to(output)),
             "detected_at_utc": datetime.now(timezone.utc).isoformat(),
             "previous_status": metadata.get("status") if metadata else None,
             "reason": "unfinished_attempt_detected_on_resume",
             "exit_code": None}
    write_json(directory / f"{identity}.json", event)
    return event


def _interruptions(output):
    """Recover an archive even if the process stopped between rename and event write."""
    directory = output / "interrupted"
    if not directory.exists():
        return []
    events = []
    for archive in sorted(path for path in directory.iterdir() if path.is_dir()):
        event_path = directory / f"{archive.name}.json"
        if event_path.exists():
            event = _read_json(event_path)
        else:
            event = {"interruption_id": archive.name, "event": "system_interruption",
                     "attempt": archive.name.split("--", 1)[0],
                     "archive": str(archive.relative_to(output)),
                     "detected_at_utc": datetime.now(timezone.utc).isoformat(),
                     "previous_status": None, "reason": "recovered_archive_without_event",
                     "exit_code": None}
            write_json(event_path, event)
        events.append(event)
    return events


def _attempt_entry(plan, result):
    ok = all(result.get(key) for key in
             ("physics_success", "replay_passed", "data_checks_passed", "images_present"))
    return {**plan, "physics_success": bool(result.get("physics_success")),
            "qualified": bool(ok), "failure_reason": result.get("failure_reason"),
            "frames": result.get("frame_count", 0),
            "simulation_seconds": result.get("simulation_seconds", 0),
            "wall_seconds": result.get("wall_seconds", 0)}


def _progress(output, entries, interruptions, wall_seconds):
    attempts = [entries[index] for index in sorted(entries)]
    qualified = [entry["attempt"] for entry in attempts if entry["qualified"]]
    _write_jsonl(output / "attempts.jsonl", attempts)
    _write_jsonl(output / "interruptions.jsonl", interruptions)
    write_json(output / "qualified.json", qualified)
    write_json(output / "summary.json", {
        "attempted": len(attempts), "physical_successes": sum(a["physics_success"] for a in attempts),
        "qualified_count": len(qualified), "qualified": qualified,
        "wall_seconds": wall_seconds, "attempts": attempts,
        "system_interruptions": len(interruptions), "interruptions": interruptions,
        "total_execution_attempts": len(attempts) + len(interruptions),
        "real_contract_verified": False, "training_eligible": False})
    return qualified


def run_generation(args):
    """Generate missing logical attempts, retaining all prior failures and interruptions."""
    config = load_config(args.config)
    config["layout_jitter_m"] = args.jitter_m
    expected = _run_config(args, config)
    if args.resume:
        if not args.output.is_dir():
            raise ValueError(f"--resume requires an existing run directory: {args.output}")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with run_lock(args.output):
        if args.resume:
            if _read_json(args.output / "effective_config.json") != config:
                raise ValueError("Resume mismatch: effective_config.json")
            manifest = args.output / "run_config.json"
            if manifest.exists() and _read_json(manifest) != expected:
                raise ValueError("Resume mismatch: run_config.json (attempts/tasks/seed/resolution/images/config)")
            existing = _existing_attempts(args, config)
        else:
            write_json(args.output / "effective_config.json", config)
            existing = {}
        # Legacy runs are imported only after every available metadata record
        # passes identity/config checks. No existing episode metadata is edited.
        write_json(args.output / "run_config.json", expected)
        started = time.monotonic()
        previous_wall = 0
        if (args.output / "summary.json").exists():
            previous_wall = float(_read_json(args.output / "summary.json").get("wall_seconds", 0))
        entries = {}
        for index, (path, metadata) in existing.items():
            if metadata and metadata.get("status") in ("finished", "failed"):
                entries[index] = _attempt_entry(_planned_attempt(args, index), metadata)
            else:
                _archive_interrupted(args.output, path, metadata)
        interruptions = _interruptions(args.output)
        previous_wall = max(previous_wall, sum(float(row["wall_seconds"]) for row in entries.values()))
        qualified = _progress(args.output, entries, interruptions, previous_wall)
        for index in range(args.attempts):
            if index in entries:
                continue
            plan = _planned_attempt(args, index)
            path = args.output / plan["attempt"]
            result = generate_attempt(path, seed=plan["seed"], task_id=plan["task_id"], config=config,
                                      images=not args.no_images, width=args.width, height=args.height)
            entries[index] = _attempt_entry(plan, result)
            qualified = _progress(args.output, entries, interruptions,
                                  previous_wall + time.monotonic() - started)
            print(json.dumps(entries[index], ensure_ascii=False), flush=True)
    return 0 if qualified or args.no_images else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New run directory; existing directories require --resume")
    parser.add_argument("--resume", action="store_true",
                        help="Validate and continue an existing run; archive unfinished attempts intact")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--jitter-m", type=float, default=0.)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--no-images", action="store_true", help="Debug-only, cannot export to training dataset")
    args = parser.parse_args(argv)
    if args.attempts < 1 or not 0 <= args.jitter_m <= .01:
        parser.error("attempts must be positive; prototype jitter must be 0..0.01m")
    if any(value <= 0 or value % 2 for value in (args.width, args.height)):
        parser.error("width and height must be positive even integers")
    try:
        return run_generation(args)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
