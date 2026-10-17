"""Generate auditable nominal SO101 contact-expert attempts locally."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New run directory; existing directories are never overwritten")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--jitter-m", type=float, default=0.)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--no-images", action="store_true", help="Debug-only, cannot export to training dataset")
    args = parser.parse_args()
    if args.attempts < 1 or not 0 <= args.jitter_m <= .01:
        parser.error("attempts must be positive; prototype jitter must be 0..0.01m")
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    config["layout_jitter_m"] = args.jitter_m
    write_json(args.output/"effective_config.json", config)
    attempts, qualified = [], []
    started = time.monotonic()
    for index in range(args.attempts):
        task = args.tasks[index % len(args.tasks)]
        # Counterfactual instructions share an initial seed and group; keep
        # their related trajectories on one side of any later dataset split.
        seed = args.seed + index//len(args.tasks)
        path = args.output/f"attempt-{index:04d}-{task}"
        result = generate_attempt(path, seed=seed, task_id=task, config=config,
                                  images=not args.no_images, width=args.width, height=args.height)
        ok = all(result.get(key) for key in ("physics_success", "replay_passed", "data_checks_passed", "images_present"))
        entry = {"attempt": path.name, "task_id": task, "seed": seed,
                 "physics_success": result["physics_success"], "qualified": ok,
                 "failure_reason": result.get("failure_reason"),
                 "frames": result["frame_count"], "simulation_seconds": result["simulation_seconds"],
                 "wall_seconds": result["wall_seconds"]}
        attempts.append(entry)
        if ok:
            qualified.append(path.name)
        with (args.output/"attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False)+"\n")
        write_json(args.output/"qualified.json", qualified)
        write_json(args.output/"summary.json", {
            "attempted": len(attempts), "physical_successes": sum(a["physics_success"] for a in attempts),
            "qualified_count": len(qualified), "qualified": qualified,
            "wall_seconds": time.monotonic()-started, "attempts": attempts,
            "real_contract_verified": False, "training_eligible": False})
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    return 0 if qualified or args.no_images else 1


if __name__ == "__main__":
    raise SystemExit(main())
