#!/usr/bin/env python3
"""Fault-inject the actual record function with in-memory devices and dataset.

Requires Python 3.12+. No hardware modules are imported or devices opened.
"""

import argparse
import ast
import copy
import sys
import types
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import Mock


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runtime", type=Path)
options, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
source = options.runtime / "src/lerobot"
cleanup_namespace = {}
exec(compile((source / "utils/cleanup.py").read_text(), "cleanup.py", "exec"), cleanup_namespace)
tree = ast.parse((source / "scripts/lerobot_record.py").read_text())
record_node = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "record"))
record_node.decorator_list = []
record_node.returns = None
for argument in record_node.args.args:
    argument.annotation = None
record_ast = ast.fix_missing_locations(ast.Module(body=[record_node], type_ignores=[]))
record_code = compile(record_ast, "lerobot_record.py", "exec")


def leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [item for child in error.exceptions for item in leaves(child)]
    return [error]


class RecordCleanupTests(unittest.TestCase):
    def run_record(self, *, failure=None, cleanup_failures=(), upload=False):
        events = []
        robot = types.SimpleNamespace(
            is_connected=False, name="fake_robot", cameras={"top": None, "wrist": None},
            action_features={}, observation_features={},
        )
        leader = types.SimpleNamespace(is_connected=False)

        def action(name):
            def call(*args, **kwargs):
                events.append(name)
                if name in cleanup_failures:
                    raise OSError(name)
            return call

        def connect_leader():
            events.append("leader.connect")
            leader.is_connected = True
            if failure == "leader.connect":
                raise ValueError(failure)

        def connect_robot():
            events.append("robot.connect")
            if failure == "robot.connect":
                # Model a failed second camera: whole-device flag stays false.
                raise ValueError(failure)
            robot.is_connected = True

        robot.connect = connect_robot
        robot.disconnect = action("robot.disconnect")
        leader.connect = connect_leader
        leader.disconnect = action("leader.disconnect")

        class Dataset:
            def __init__(self):
                self.num_episodes = 0
                self.writer = None

            def __bool__(self):
                return self.num_episodes > 0

            def save_episode(self):
                events.append("save")
                self.num_episodes += 1

            finalize = staticmethod(action("finalize"))
            push_to_hub = staticmethod(action("upload"))

        dataset = Dataset()

        def create(*args, **kwargs):
            events.append("create")
            if failure == "create":
                raise ValueError(failure)
            return dataset

        class Manager:
            def __init__(self, data):
                self.data = data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.data.finalize()
                return False

        def loop(**kwargs):
            events.append("record_loop")
            if failure == "record_loop":
                raise ValueError(failure)
            kwargs["events"]["stop_recording"] = True  # q saves the current episode.

        def start_view(*args, **kwargs):
            events.append("view.start")
            if failure == "view.start":
                raise ValueError(failure)

        def notify(message, *args, **kwargs):
            if message == "Stop recording" and "notification" in cleanup_failures:
                raise OSError("notification")

        dataset_cfg = types.SimpleNamespace(
            repo_id="local/fake", root="UNUSED", fps=30, video=True, num_episodes=1,
            episode_time_s=45, reset_time_s=0, push_to_hub=upload, tags=None, private=None,
            video_encoding_batch_size=1, rgb_encoder=None, depth_encoder=None,
            encoder_threads=None, streaming_encoding=False, encoder_queue_maxsize=30,
            num_image_writer_processes=0, num_image_writer_threads_per_camera=4,
            single_task="Put the red toothbrush into the left cup.", stamp_repo_id=lambda: None,
        )
        cfg = types.SimpleNamespace(
            display_data=True, display_mode="rerun", display_ip=None, display_port=None,
            display_compressed_images=False, robot=None, teleop=object(), dataset=dataset_cfg,
            resume=False, play_sounds=False,
        )
        timer = types.SimpleNamespace(
            log_run_summary=action("summary"), log_episode_summary=lambda *args: None,
            restart=lambda: None,
        )
        namespace = dict(
            partial=partial, init_logging=lambda: None, logging=Mock(), pformat=lambda cfg: "",
            asdict=lambda cfg: {}, init_visualization=start_view,
            shutdown_visualization=action("view.stop"), make_robot_from_config=lambda cfg: robot,
            make_teleoperator_from_config=lambda cfg: leader,
            make_default_processors=lambda: (object(), object(), object()),
            combine_feature_dicts=lambda *args: {}, aggregate_pipeline_dataset_features=lambda **kw: {},
            create_initial_features=lambda **kw: {}, CycleTimer=lambda fps: timer,
            LeRobotDataset=types.SimpleNamespace(create=create), VideoEncodingManager=Manager,
            init_keyboard_listener=lambda: (
                types.SimpleNamespace(stop=action("listener.stop")),
                {"stop_recording": False, "exit_early": False, "rerecord_episode": False},
            ),
            log_say=notify, record_loop=loop,
            run_cleanup=cleanup_namespace["run_cleanup"],
        )
        exec(record_code, namespace)
        try:
            result = namespace["record"](cfg)
        except BaseException as error:
            return events, dataset, error
        self.assertIs(result, dataset)
        return events, dataset, None

    def test_one_episode_q_path_saves_and_cleans_every_resource(self):
        events, dataset, error = self.run_record()
        self.assertIsNone(error)
        self.assertEqual(dataset.num_episodes, 1)
        self.assertEqual(events.count("record_loop"), 1)
        self.assertLess(events.index("save"), events.index("robot.disconnect"))
        self.assertLess(events.index("robot.disconnect"), events.index("leader.disconnect"))
        for name in ["listener.stop", "view.stop", "summary", "finalize"]:
            self.assertIn(name, events)
        self.assertNotIn("upload", events)

    def test_partial_camera_connect_still_disconnects_robot_and_finalizes_empty_dataset(self):
        events, dataset, error = self.run_record(failure="robot.connect")
        self.assertFalse(bool(dataset))
        self.assertIsInstance(error, ValueError)
        self.assertIn("robot.disconnect", events)
        self.assertIn("leader.disconnect", events)
        self.assertIn("finalize", events)

    def test_leader_connect_failure_finalizes_empty_dataset(self):
        events, _, error = self.run_record(failure="leader.connect")
        self.assertIsInstance(error, ValueError)
        self.assertIn("leader.disconnect", events)
        self.assertIn("finalize", events)
        self.assertNotIn("record_loop", events)

    def test_all_hardware_cleanup_failures_still_attempt_later_actions(self):
        failures = ["robot.disconnect", "leader.disconnect", "listener.stop", "view.stop", "summary"]
        events, _, error = self.run_record(cleanup_failures=failures)
        self.assertIsInstance(error, ExceptionGroup)
        self.assertEqual(len(leaves(error)), len(failures))
        self.assertEqual(events[-1], "finalize")

    def test_manager_finalize_failure_still_unloads_and_retries_finalization(self):
        events, _, error = self.run_record(cleanup_failures=["finalize"])
        self.assertIsInstance(error, ExceptionGroup)
        self.assertIn("robot.disconnect", events)
        self.assertIn("leader.disconnect", events)
        self.assertIn("view.stop", events)
        self.assertEqual(events.count("finalize"), 2)

    def test_loop_and_cleanup_errors_are_both_retained(self):
        events, _, error = self.run_record(failure="record_loop", cleanup_failures=["robot.disconnect"])
        errors = leaves(error)
        self.assertEqual([str(item) for item in errors], ["record_loop", "robot.disconnect"])
        self.assertIn("finalize", events)
        self.assertNotIn("save", events)

    def test_loop_error_remains_in_context_if_manager_finalize_also_fails(self):
        events, _, error = self.run_record(failure="record_loop", cleanup_failures=["finalize"])
        first = leaves(error)[0]
        self.assertEqual(str(first), "finalize")
        self.assertIsInstance(first.__context__, ValueError)
        self.assertEqual(str(first.__context__), "record_loop")
        self.assertIn("robot.disconnect", events)

    def test_stop_notification_failure_occurs_after_hardware_and_finalize(self):
        events, _, error = self.run_record(cleanup_failures=["notification"])
        self.assertEqual([str(item) for item in leaves(error)], ["notification"])
        for action in ["robot.disconnect", "leader.disconnect", "view.stop", "finalize"]:
            self.assertIn(action, events)

    def test_creation_error_cleans_unconnected_follower_and_viewer(self):
        events, _, error = self.run_record(failure="create")
        self.assertIsInstance(error, ValueError)
        self.assertIn("robot.disconnect", events)
        self.assertIn("view.stop", events)
        self.assertNotIn("finalize", events)

    def test_viewer_init_error_preserved_alongside_shutdown_error(self):
        events, _, error = self.run_record(failure="view.start", cleanup_failures=["view.stop"])
        self.assertEqual([str(item) for item in leaves(error)], ["view.start", "view.stop"])
        self.assertEqual(events, ["view.start", "view.stop"])

    def test_upload_occurs_only_after_successful_cleanup(self):
        events, _, error = self.run_record(upload=True)
        self.assertIsNone(error)
        self.assertEqual(events[-1], "upload")
        events, _, error = self.run_record(upload=True, cleanup_failures=["robot.disconnect"])
        self.assertIsInstance(error, ExceptionGroup)
        self.assertNotIn("upload", events)


if __name__ == "__main__":
    unittest.main(verbosity=2)
