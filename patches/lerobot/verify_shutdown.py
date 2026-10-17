#!/usr/bin/env python3
"""Offline fault injection against patched source; never import hardware modules.

Usage: python verify_shutdown.py /path/to/lerobot
Requires Python 3.12+, matching this LeRobot revision.
"""

import argparse
import ast
import copy
import logging
import sys
import types
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import Mock


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runtime", type=Path, help="LeRobot checkout containing the applied patch")
arguments, unittest_arguments = parser.parse_known_args()
sys.argv = [sys.argv[0], *unittest_arguments]
RUNTIME = arguments.runtime.resolve()
SOURCE = RUNTIME / "src/lerobot"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_function(relative, name, class_name=None, **namespace):
    """Compile only actual cleanup methods, with hardware imports excluded."""
    tree = ast.parse((SOURCE / relative).read_text())
    owner = tree
    if class_name:
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = copy.deepcopy(
        next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    env = {"partial": partial, "logger": Mock(), **namespace}
    exec(compile(module, str(SOURCE / relative), "exec"), env)
    return env[name]


cleanup_globals = {}
exec(compile((SOURCE / "utils/cleanup.py").read_text(), "cleanup.py", "exec"), cleanup_globals)
run_cleanup = cleanup_globals["run_cleanup"]
bus_disconnect = load_function("motors/motors_bus.py", "disconnect", "SerialMotorsBus", run_cleanup=run_cleanup)
write_packet = load_function("motors/motors_bus.py", "_write", "SerialMotorsBus")
follower_disconnect = load_function(
    "robots/so_follower/so_follower.py", "disconnect", "SOFollower", run_cleanup=run_cleanup
)


def leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in leaves(child)]
    return [error]


def make_bus(events, failures=(), clear_fails=False, close_fails=False):
    bus = types.SimpleNamespace(is_connected=True, motors=dict.fromkeys(JOINTS))

    def clear():
        events.append("clear")
        if clear_fails:
            raise OSError("buffer failure")

    def disable(motors, num_retry):
        events.append(motors)
        assert num_retry == 5
        if motors in failures:
            raise RuntimeError(f"servo failure: {motors}")

    def close():
        events.append("close")
        bus.is_connected = False
        if close_fails:
            raise OSError("port close failure")

    bus.port_handler = types.SimpleNamespace(clearPort=clear, closePort=close, is_using=True)
    bus.disable_torque = disable
    bus.disconnect = lambda torque=True: bus_disconnect(bus, torque)
    return bus


def make_camera(name, events, connected=True, thread=None, fails=False):
    camera = types.SimpleNamespace(is_connected=connected, thread=thread)

    def disconnect():
        events.append(name)
        camera.is_connected = False
        camera.thread = None
        if fails:
            raise OSError(f"camera failure: {name}")

    camera.disconnect = disconnect
    return camera


def make_follower(bus, cameras):
    robot = types.SimpleNamespace(
        bus=bus, cameras=cameras, config=types.SimpleNamespace(disable_torque_on_disconnect=True)
    )
    robot.disconnect = lambda: follower_disconnect(robot)
    return robot


class ShutdownTests(unittest.TestCase):
    def test_id2_failure_does_not_skip_remaining_motors_or_port(self):
        events = []
        bus = make_bus(events, failures=["shoulder_lift"])
        with self.assertRaises(ExceptionGroup) as caught:
            bus.disconnect()
        self.assertEqual(events, ["clear", *JOINTS, "close"])
        self.assertFalse(bus.is_connected)
        self.assertFalse(bus.port_handler.is_using)
        self.assertEqual(len(leaves(caught.exception)), 1)
        self.assertIn("shoulder_lift", str(leaves(caught.exception)[0]))

    def test_multiple_motor_buffer_and_close_failures_are_all_retained(self):
        events = []
        bus = make_bus(events, failures=[JOINTS[1], JOINTS[3]], clear_fails=True, close_fails=True)
        with self.assertRaises(ExceptionGroup) as caught:
            bus.disconnect()
        self.assertEqual(events, ["clear", *JOINTS, "close"])
        self.assertEqual(len(leaves(caught.exception)), 4)

    def test_explicit_torque_false_keeps_existing_option_semantics(self):
        events = []
        make_bus(events).disconnect(False)
        self.assertEqual(events, ["close"])

    def test_bus_and_top_failures_do_not_skip_wrist(self):
        events = []
        robot = make_follower(
            make_bus(events, failures=[JOINTS[1]]),
            {"top": make_camera("top", events, fails=True), "wrist": make_camera("wrist", events)},
        )
        with self.assertRaises(ExceptionGroup) as caught:
            robot.disconnect()
        self.assertEqual(events, ["clear", *JOINTS, "close", "top", "wrist"])
        self.assertEqual(len(leaves(caught.exception)), 2)

    def test_partial_camera_connection_and_repeat_cleanup(self):
        events = []
        robot = make_follower(
            make_bus(events),
            {"top": make_camera("top", events), "wrist": make_camera("wrist", events, connected=False)},
        )
        # Whole-device decorator would block this partial-state cleanup.
        tree = ast.parse((SOURCE / "robots/so_follower/so_follower.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SOFollower")
        node = next(
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "disconnect"
        )
        self.assertEqual(node.decorator_list, [])
        robot.disconnect()
        once = list(events)
        robot.disconnect()
        self.assertEqual(events, once)
        self.assertEqual(events, ["clear", *JOINTS, "close", "top"])

    def test_disconnected_camera_with_remaining_thread_is_cleaned(self):
        events = []
        bus = make_bus(events)
        bus.is_connected = False
        robot = make_follower(bus, {"wrist": make_camera("wrist", events, connected=False, thread=object())})
        robot.disconnect()
        self.assertEqual(events, ["wrist"])

    def test_interrupt_during_cleanup_still_attempts_remaining_actions(self):
        events = []

        def interrupted():
            raise KeyboardInterrupt()

        with self.assertRaises(BaseExceptionGroup):
            run_cleanup([("interrupt", interrupted), ("remaining", lambda: events.append("remaining"))])
        self.assertEqual(events, ["remaining"])

    def entry(
        self, events, robot, *, leader_fails=False, loop_error=None,
        view_fails=False, init_error=None, leader_stop_fails=False,
    ):
        leader = types.SimpleNamespace(is_connected=False)

        def connect():
            events.append("leader-connect")
            leader.is_connected = True
            if leader_fails:
                raise ValueError("leader calibration failure")

        leader.connect = connect
        def leader_stop():
            events.append("leader-disconnect")
            if leader_stop_fails:
                raise OSError("leader disconnect failure")

        leader.disconnect = leader_stop

        def view_start(*args, **kwargs):
            events.append("view-start")
            if init_error is not None:
                raise init_error

        def loop(**kwargs):
            events.append("loop")
            if loop_error is not None:
                raise loop_error

        def view_stop(mode):
            events.append("view-stop")
            if view_fails:
                raise OSError("viewer failure")

        entry = load_function(
            "scripts/lerobot_teleoperate.py", "teleoperate", TeleoperateConfig=object,
            init_logging=lambda: None, logging=Mock(), pformat=lambda value: "", asdict=lambda cfg: {},
            init_visualization=view_start,
            shutdown_visualization=view_stop, make_teleoperator_from_config=lambda cfg: leader,
            make_robot_from_config=lambda cfg: robot, make_default_processors=lambda: (None, None, None),
            teleop_loop=loop, run_cleanup=run_cleanup,
        )
        cfg = types.SimpleNamespace(
            display_data=True, display_mode="rerun", display_ip=None, display_port=None,
            display_compressed_images=False, teleop=None, robot=None, fps=30, teleop_time_s=None,
        )
        entry(cfg)

    def test_ctrl_c_cleans_follower_before_leader_and_viewer(self):
        events = []
        robot = make_follower(make_bus(events), {})
        robot.connect = lambda: events.append("robot-connect")
        self.entry(events, robot, loop_error=KeyboardInterrupt())
        self.assertEqual(events[-3:], ["close", "leader-disconnect", "view-stop"])

    def test_second_camera_connect_failure_still_releases_partial_robot(self):
        events = []
        robot = make_follower(
            make_bus(events),
            {"top": make_camera("top", events), "wrist": make_camera("wrist", events, connected=False)},
        )

        def fail_connect():
            raise ValueError("second camera connect failure")

        robot.connect = fail_connect
        with self.assertRaisesRegex(ValueError, "second camera"):
            self.entry(events, robot)
        self.assertEqual(events[-4:], ["close", "top", "leader-disconnect", "view-stop"])

    def test_leader_partial_connect_failure_releases_its_open_port(self):
        events = []
        bus = make_bus(events)
        bus.is_connected = False
        robot = make_follower(bus, {})
        with self.assertRaisesRegex(ValueError, "leader calibration"):
            self.entry(events, robot, leader_fails=True)
        self.assertEqual(events, ["view-start", "leader-connect", "leader-disconnect", "view-stop"])

    def test_loop_and_cleanup_errors_are_preserved_and_viewer_still_stops(self):
        events = []
        robot = make_follower(
            make_bus(events, failures=[JOINTS[1]]), {"top": make_camera("top", events)}
        )
        robot.connect = lambda: None
        primary = ValueError("original loop failure")
        with self.assertRaises(ExceptionGroup) as caught:
            self.entry(events, robot, loop_error=primary, view_fails=True)
        errors = leaves(caught.exception)
        self.assertIs(errors[0], primary)
        self.assertEqual(len(errors), 3)
        self.assertEqual(events[-3:], ["top", "leader-disconnect", "view-stop"])

    def test_viewer_start_and_stop_failures_preserve_both_errors(self):
        events = []
        primary = ValueError("viewer initialization failure")
        with self.assertRaises(ExceptionGroup) as caught:
            self.entry(events, None, init_error=primary, view_fails=True)
        self.assertEqual(events, ["view-start", "view-stop"])
        errors = leaves(caught.exception)
        self.assertIs(errors[0], primary)
        self.assertEqual(len(errors), 2)

    def test_leader_cleanup_failure_does_not_skip_viewer(self):
        events = []
        robot = make_follower(make_bus(events), {})
        robot.connect = lambda: None
        with self.assertRaises(ExceptionGroup):
            self.entry(events, robot, leader_stop_fails=True)
        self.assertEqual(events[-3:], ["close", "leader-disconnect", "view-stop"])

    def packet_bus(self, results):
        handler = types.SimpleNamespace(
            writeTxRx=Mock(side_effect=results), getTxRxResult=lambda code: f"comm {code}",
            getRxPacketError=lambda error: "",
        )
        return types.SimpleNamespace(
            port="FAKE-PORT-NOT-A-DEVICE", packet_handler=handler, port_handler=object(),
            _serialize_data=lambda value, length: [value], _is_comm_success=lambda comm: comm == 0,
            _is_error=lambda error: error != 0,
        )

    def test_unmapped_status_reports_raw_code_and_actual_single_attempt(self):
        bus = self.packet_bus([(0, 16)])
        with self.assertRaises(RuntimeError) as caught:
            write_packet(bus, 40, 1, 2, 0, num_retry=5)
        message = str(caught.exception)
        for value in [
            "motor_id=2", "comm=0", "error=16 (0x10)", "actual_attempts=1",
            "max_attempts=6", "unmapped device status",
        ]:
            self.assertIn(value, message)
        self.assertEqual(bus.packet_handler.writeTxRx.call_count, 1)

    def test_timeout_reports_six_actual_attempts(self):
        bus = self.packet_bus([(-6, 0)] * 6)
        with self.assertRaises(ConnectionError) as caught:
            write_packet(bus, 40, 1, 2, 0, num_retry=5)
        self.assertIn("actual_attempts=6", str(caught.exception))
        self.assertEqual(bus.packet_handler.writeTxRx.call_count, 6)

    def test_successful_write_and_suppressed_error_keep_return_values(self):
        bus = self.packet_bus([(0, 0), (0, 16)])
        self.assertEqual(write_packet(bus, 40, 1, 2, 0), (0, 0))
        self.assertEqual(write_packet(bus, 40, 1, 2, 0, raise_on_error=False), (0, 16))


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
