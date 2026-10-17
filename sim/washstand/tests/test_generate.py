"""Resume/locking regressions with small fake episodes; no simulation is run."""

import argparse
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from sim.washstand import generate


class GenerationResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(output=self.root / "run", attempts=4,
                                       tasks=["red_left", "red_right"], seed=200,
                                       jitter_m=.002, config=None, width=320, height=240,
                                       no_images=False, resume=False)
        self.config = generate.load_config()
        self.config["layout_jitter_m"] = self.args.jitter_m
        self.calls = []
        self.quiet = io.StringIO()

    def write_episode(self, path, *, seed, task_id, config, images=True,
                      width=320, height=240, success=True, status="finished", legacy=False):
        """Only manifests/artifact presence are relevant to these run-manager tests."""
        path.mkdir()
        metadata = {"seed": seed, "task_id": task_id, "config": config,
                    "width": width, "height": height, "record_images": images,
                    "expert_version": generate.ScriptedExpert.VERSION,
                    "status": status, "physics_success": success,
                    "replay_passed": success, "data_checks_passed": True,
                    "images_present": images, "frame_count": 2,
                    "simulation_seconds": 2/30, "wall_seconds": .25,
                    "failure_reason": None if success else "no_grasp_after_trial_lift",
                    "data_check_errors": [],
                    "video_encoding": {"codec": "h264"} if images else None}
        (path / "initial_state.npz").write_bytes(b"fake-initial-state\x00")
        if status == "recording":
            metadata.update(images_present=False, physics_success=False,
                            replay_passed=False, data_checks_passed=False,
                            frame_count=0, video_encoding=None)
            (path / "diagnostics.jsonl").write_bytes(b'{"partial": true}\n')
        else:
            (path / "trajectory.npz").write_bytes(b"fake-trajectory\x00")
        if images:
            for camera in ("top", "wrist"):
                (path / f"{camera}.mp4").write_bytes(b"partial-video" if status == "recording" else b"fake-video")
        if legacy:
            del metadata["record_images"]
        generate.write_json(path / "metadata.json", metadata)
        return metadata

    def fake_generate(self, path, **kwargs):
        self.calls.append(path.name)
        return self.write_episode(path, **kwargs)

    def legacy_attempt(self, index, *, success=True, status="finished", output=None):
        output = output or self.args.output
        plan = generate._planned_attempt(self.args, index)
        return self.write_episode(output / plan["attempt"], seed=plan["seed"],
                                  task_id=plan["task_id"], config=self.config,
                                  success=success, status=status, legacy=True)

    @staticmethod
    def bytes_in(path):
        return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}

    def invoke(self, args=None, fake=None):
        with patch.object(generate, "generate_attempt", side_effect=fake or self.fake_generate), redirect_stdout(self.quiet):
            return generate.run_generation(args or self.args)

    def test_new_run_rejects_existing_directory_without_touching_it(self):
        self.args.output.mkdir()
        (self.args.output / "precious.txt").write_bytes(b"keep")
        before = self.bytes_in(self.args.output)
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.bytes_in(self.args.output), before)
        self.assertEqual(self.calls, [])

    def test_new_run_manifest_and_counterfactual_seed_schedule(self):
        self.args.attempts = 5
        self.assertEqual(self.invoke(), 0)
        manifest = json.loads((self.args.output / "run_config.json").read_text())
        self.assertEqual(manifest["images"], True)
        self.assertEqual(manifest["effective_config"], self.config)
        entries = [json.loads(line) for line in (self.args.output / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual([e["seed"] for e in entries], [200, 200, 201, 201, 202])
        self.assertEqual([e["task_id"] for e in entries], ["red_left", "red_right", "red_left", "red_right", "red_left"])
        self.assertEqual(len({e["attempt"] for e in entries}), 5)

    def test_legacy_resume_keeps_failed_and_finished_and_archives_partial_intact(self):
        self.args.output.mkdir()
        generate.write_json(self.args.output / "effective_config.json", self.config)
        self.legacy_attempt(0)
        self.legacy_attempt(1, success=False)
        self.legacy_attempt(2, status="recording")
        finished = {index: self.bytes_in(self.args.output / generate._planned_attempt(self.args, index)["attempt"])
                    for index in (0, 1)}
        partial_name = generate._planned_attempt(self.args, 2)["attempt"]
        original_partial = self.bytes_in(self.args.output / partial_name)
        # Stale or duplicated aggregate logs are rebuilt from per-episode truth.
        (self.args.output / "attempts.jsonl").write_text('garbage\ngarbage\n')
        generate.write_json(self.args.output / "summary.json", {"wall_seconds": 1.5, "attempted": 99})
        self.args.resume = True
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.calls, [partial_name, "attempt-0003-red_right"])
        for index in (0, 1):
            path = self.args.output / generate._planned_attempt(self.args, index)["attempt"]
            self.assertEqual(self.bytes_in(path), finished[index])
        archives = [p for p in (self.args.output / "interrupted").iterdir() if p.is_dir()]
        self.assertEqual(len(archives), 1)
        self.assertEqual(self.bytes_in(archives[0]), original_partial)
        summary = json.loads((self.args.output / "summary.json").read_text())
        self.assertEqual(summary["attempted"], 4)
        self.assertEqual(summary["physical_successes"], 3)
        self.assertEqual(summary["qualified_count"], 3)
        self.assertEqual(summary["system_interruptions"], 1)
        self.assertEqual(summary["total_execution_attempts"], 5)
        self.assertEqual(summary["attempts"][1]["failure_reason"], "no_grasp_after_trial_lift")
        self.assertEqual(len({e["attempt"] for e in summary["attempts"]}), 4)
        self.assertTrue((self.args.output / "run_config.json").is_file())
        self.calls.clear()
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(len((self.args.output / "interruptions.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len((self.args.output / "attempts.jsonl").read_text().splitlines()), 4)

    def test_finished_episode_not_in_aggregate_log_is_recovered_without_rerunning(self):
        self.args.attempts = 2
        self.args.output.mkdir()
        generate.write_json(self.args.output / "effective_config.json", self.config)
        self.legacy_attempt(0)
        self.legacy_attempt(1)
        self.args.resume = True
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.calls, [])
        summary = json.loads((self.args.output / "summary.json").read_text())
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["qualified_count"], 2)

    def test_manifest_mismatches_rejected_before_generation_or_archiving(self):
        self.invoke()
        self.args.resume = True
        before = self.bytes_in(self.args.output)
        changes = {"seed": 201, "tasks": ["red_right", "red_left"], "width": 640,
                   "height": 480, "no_images": True, "attempts": 5, "jitter_m": .001}
        for field, value in changes.items():
            with self.subTest(field=field):
                args = copy.deepcopy(self.args)
                setattr(args, field, value)
                self.calls.clear()
                with self.assertRaisesRegex(ValueError, "Resume mismatch"):
                    self.invoke(args)
                self.assertEqual(self.calls, [])
                self.assertEqual(self.bytes_in(self.args.output), before)

    def test_legacy_metadata_mismatches_never_archive_or_add_manifest(self):
        for field, value in {"seed": 9, "task_id": "blue_left", "width": 640,
                             "height": 480, "config": {}, "record_images": False}.items():
            with self.subTest(field=field):
                output = self.root / field
                output.mkdir()
                generate.write_json(output / "effective_config.json", self.config)
                metadata = self.legacy_attempt(0, output=output)
                self.legacy_attempt(1, status="recording", output=output)
                metadata[field] = value
                generate.write_json(output / "attempt-0000-red_left" / "metadata.json", metadata)
                args = copy.deepcopy(self.args)
                args.output, args.resume = output, True
                before = self.bytes_in(output)
                with self.assertRaisesRegex(ValueError, "Resume mismatch"):
                    self.invoke(args)
                after = self.bytes_in(output)
                after.pop(".run.lock", None)
                self.assertEqual(after, before)
                self.assertFalse((output / "run_config.json").exists())
                self.assertFalse((output / "interrupted").exists())

    def test_finished_failed_video_checks_are_preserved_using_explicit_image_intent(self):
        self.args.attempts = 1
        self.args.output.mkdir()
        generate.write_json(self.args.output / "effective_config.json", self.config)
        plan = generate._planned_attempt(self.args, 0)
        path = self.args.output / plan["attempt"]
        metadata = self.write_episode(path, seed=200, task_id="red_left", config=self.config, success=False)
        metadata.update(images_present=False, data_checks_passed=False, video_encoding=None,
                        data_check_errors=["top video finalization failed"])
        generate.write_json(path / "metadata.json", metadata)
        before = self.bytes_in(path)
        self.args.resume = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.bytes_in(path), before)
        summary = json.loads((self.args.output / "summary.json").read_text())
        self.assertEqual(summary["attempted"], 1)
        self.assertEqual(summary["qualified_count"], 0)

    def test_missing_finished_artifact_is_rejected_without_overwriting_metadata(self):
        self.args.output.mkdir()
        generate.write_json(self.args.output / "effective_config.json", self.config)
        self.legacy_attempt(0)
        (self.args.output / "attempt-0000-red_left" / "trajectory.npz").unlink()
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "missing artifacts"):
            self.invoke()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.args.output / "interrupted").exists())

    def test_repeated_interruption_keeps_unique_archives_and_releases_lock(self):
        self.args.attempts = 1
        counter = [0]
        def interrupted(path, **kwargs):
            counter[0] += 1
            self.write_episode(path, **kwargs, status="recording")
            (path / "marker").write_text(str(counter[0]))
            raise RuntimeError("simulated interruption")
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.invoke(fake=interrupted)
        self.args.resume = True
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.invoke(fake=interrupted)
        self.assertEqual(self.invoke(), 0)
        archives = [p for p in (self.args.output / "interrupted").iterdir() if p.is_dir()]
        self.assertEqual({(p / "marker").read_text() for p in archives}, {"1", "2"})
        events = [json.loads(line) for line in (self.args.output / "interruptions.jsonl").read_text().splitlines()]
        self.assertEqual(len({event["interruption_id"] for event in events}), 2)
        self.assertTrue(all(event["event"] == "system_interruption" for event in events))
        summary = json.loads((self.args.output / "summary.json").read_text())
        self.assertEqual(summary["attempted"], 1)
        self.assertEqual(summary["total_execution_attempts"], 3)

    def test_orphan_archive_event_is_recovered_without_modifying_archived_bytes(self):
        self.args.attempts = 1
        self.invoke()
        archive = self.args.output / "interrupted" / "attempt-0000-red_left--orphan"
        archive.mkdir(parents=True)
        (archive / "partial.bin").write_bytes(b"preserve")
        self.args.resume = True
        self.invoke()
        self.invoke()
        self.assertEqual(self.bytes_in(archive), {"partial.bin": b"preserve"})
        events = [json.loads(line) for line in (self.args.output / "interruptions.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reason"], "recovered_archive_without_event")

    def test_other_process_lock_fails_immediately_without_generation(self):
        self.args.output.mkdir()
        self.args.resume = True
        program = ("import fcntl,sys; f=open(sys.argv[1],'a+'); "
                   "fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); sys.stdin.read()")
        process = subprocess.Popen([sys.executable, "-c", program,
                                    str(self.args.output / ".run.lock")],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "locked")
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "already locked"):
                self.invoke()
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertEqual(self.calls, [])
        finally:
            process.stdin.close()
            process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
