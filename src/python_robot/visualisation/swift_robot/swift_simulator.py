"""
Swift-based visualization for URDF-backed manipulators.

The public ``SwiftSimulator`` class bridges python-robot motion schemes to the
Robotics Toolbox Swift backend. Motion planning remains the responsibility of
python-robot; Swift is used only as a rich 3D renderer for the underlying RTB
``ERobot`` that was created from the URDF/xacro model.

The private ``_SwiftSession`` helper owns the low-level Swift environment,
browser lifecycle, and websocket cleanup details. Those details are deliberately
kept out of ``SwiftSimulator`` so the public API can stay focused on robot
playback rather than Swift's transport mechanics.
"""

from copy import deepcopy
import json
import logging
import time
from queue import Empty

import numpy as np

from roboticstoolbox import ERobot
from roboticstoolbox.backends.swift import Swift
from swift import Button, Slider

from python_robot.base.types import NumpyArray
from python_robot.motion import JointSpaceScheme
from python_robot.manipulator import URDFManipulator

__all__ = ["SwiftSimulator"]


class SwiftSimulator:
    """
    High-level Swift playback interface for a URDF manipulator.

    ``URDFManipulator`` is the python-robot model used for kinematics, inverse
    kinematics, and trajectory generation. Swift, however, knows how to display
    Robotics Toolbox robots. Therefore, this simulator uses the manipulator's
    ``source_erobot`` internally and sends sampled joint configurations to that
    RTB robot.

    The class intentionally exposes playback operations such as
    ``play_joint_scheme`` and ``play_joint_trajectory``. Details such as opening
    the browser, keeping the session alive, and suppressing expected websocket
    shutdown noise are delegated to ``_SwiftSession``.
    """

    def __init__(self, robot: URDFManipulator | None = None) -> None:
        """
        Create a simulator, optionally bound to a URDF manipulator.

        Parameters
        ----------
        robot : URDFManipulator | None, default=None
            Manipulator whose Robotics Toolbox ``ERobot`` should be visualized.
            If omitted, ``play_joint_scheme`` will take the robot from the given
            joint-space scheme.
        """
        if robot is not None:
            if not isinstance(robot, URDFManipulator):
                raise TypeError(
                    "The robot must be an instance of URDFManipulator."
                )
            self._rtb_robot: ERobot | None = robot.source_erobot
        else:
            self._rtb_robot = None

        self._q_array = None
        self._dt = None
        self._speed = 1.0
        self._controls = False
        self._session = _SwiftSession(self)

    def play_joint_trajectory(
        self,
        q_array: NumpyArray,
        t_array: NumpyArray,
        hold: bool = False,
        speed: float = 1.0,
        controls: bool = False,
    ) -> None:
        """
        Play a sampled joint trajectory in Swift.

        Parameters
        ----------
        q_array : NumpyArray
            Joint configurations ordered row-wise. Each row is sent to the RTB
            robot and rendered by Swift.
        t_array : NumpyArray
            Time samples corresponding with ``q_array``. The average timestep is
            used as Swift playback interval.
        hold : bool, default=False
            If True, keep the script alive after playback until the browser
            window is closed.
        speed : float, default=1.0
            Playback speed multiplier. ``1.0`` uses the sampled trajectory
            timing, ``2.0`` plays twice as fast, and ``0.5`` plays twice as
            slow. Must be strictly positive and finite.
        controls : bool, default=False
            If True, add browser playback controls for changing speed and
            replaying the trajectory after it finishes. The browser session
            stays open until the browser window is closed.
        """
        self._q_array: NumpyArray = q_array
        self._dt: float = self._get_timestep(t_array)
        self._speed: float = self._validate_speed(speed)
        self._controls: bool = bool(controls)

        self._session.run(hold=hold, controls=controls)

    def play_joint_scheme(
        self,
        joint_scheme: JointSpaceScheme,
        step: int = 1,
        hold: bool = False,
        speed: float = 1.0,
        controls: bool = False,
    ) -> None:
        """
        Play a python-robot joint-space motion scheme in Swift.

        ``JointSpaceScheme`` contains the planned robot motion. Swift does not
        compute that motion; it only visualizes the sampled joint positions.

        Parameters
        ----------
        joint_scheme : JointSpaceScheme
            Motion scheme whose manipulator must be a ``URDFManipulator``.
        step : int, default=1
            Use every ``step``-th sample for playback. This is useful because
            Swift animation can become sluggish when every dense planning sample
            is rendered.
        hold : bool, default=False
            If True, keep the browser session open after playback and exit
            cleanly when the browser window is closed.
        speed : float, default=1.0
            Playback speed multiplier. ``1.0`` uses the sampled trajectory
            timing, ``2.0`` plays twice as fast, and ``0.5`` plays twice as
            slow. Must be strictly positive and finite.
        controls : bool, default=False
            If True, add browser playback controls for changing speed and
            replaying the trajectory after it finishes. The browser session
            stays open until the browser window is closed.
        """
        if not isinstance(joint_scheme.manipulator, URDFManipulator):
            raise TypeError(
                "The robot must be an instance of URDFManipulator."
            )

        if self._rtb_robot is not None:
            rtb_robot_copy = deepcopy(self._rtb_robot)
        else:
            rtb_robot_copy = None

        self._rtb_robot = joint_scheme.manipulator.source_erobot

        q_array: NumpyArray = joint_scheme.positions[::step]
        t_array: NumpyArray = joint_scheme.time_samples[::step]

        self.play_joint_trajectory(
            q_array,
            t_array,
            hold=hold,
            speed=speed,
            controls=controls,
        )

        if rtb_robot_copy is not None:
            self._rtb_robot = rtb_robot_copy

    @staticmethod
    def _get_timestep(t_array: NumpyArray) -> float:
        """
        Estimate a constant playback timestep from sampled trajectory times.
        """
        return float(np.mean(np.diff(t_array)))

    @staticmethod
    def _validate_speed(speed: float) -> float:
        """
        Return a normalized playback speed multiplier.
        """
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be a strictly positive finite number.")
        return speed


class _SwiftSession:
    """
    Low-level Swift environment wrapper.

    This helper encapsulates the parts of Swift that are transport- and
    browser-specific: launching the backend, adding the RTB robot, stepping the
    scene, keeping the browser alive after playback, and cleaning up websocket
    logging.

    Swift's own ``hold()`` method is an endless sleep loop. It does not notice
    when the browser tab is closed. This class therefore implements its own hold
    loop by periodically asking Swift for ``shape_poses``. If the browser no
    longer responds, the session is closed.
    """

    def __init__(self, parent: SwiftSimulator) -> None:
        """
        Create a session owned by a ``SwiftSimulator``.

        The parent provides the prepared trajectory and RTB robot. Keeping this
        class private avoids making Swift's queue and websocket details part of
        the public python-robot API.
        """
        self._parent = parent
        self._env = None
        self._websocket_logger_levels: dict[str, int] | None = None
        self._replay_requested = False

    def run(self, hold: bool = False, controls: bool = False) -> None:
        """
        Launch Swift, play the prepared trajectory, and optionally keep it open.
        """
        if self._parent._q_array is None or self._parent._dt is None:
            raise RuntimeError("Call setup() before run().")

        self._create_env()

        if controls:
            self._add_playback_controls()

        while True:
            self._replay_requested = False
            self._reset_sim_time()
            self._run_loop()

            if not controls:
                break

            if not self._wait_for_replay_or_browser_close():
                return

        if hold and not controls:
            self._hold()

    def close(self) -> None:
        """
        Close the Swift environment and restore logging state.
        """
        try:
            if self._env is not None:
                self._env.close()
                self._env = None
        finally:
            self._restore_websocket_logging()

    def _create_env(self) -> None:
        """
        Create the Swift backend and add the RTB robot to the scene.
        """
        self._suppress_expected_websocket_close_tracebacks()
        self._env = Swift()
        self._env.launch(realtime=True)
        self._env.add(
            self._parent._rtb_robot,
            robot_alpha=1.0,
            collision_alpha=0.0,
            readonly=True
        )

    def _reset_sim_time(self) -> None:
        """
        Reset Swift's browser timer before a playback run.
        """
        if self._env is None:
            raise RuntimeError("No Swift environment is running.")

        self._env.sim_time = 0.0
        self._env.outq.put([False, ["sim_time", self._env.sim_time]])

    def _run_loop(self) -> None:
        """
        Send every sampled joint configuration to Swift.
        """
        for q in self._parent._q_array:
            self._parent._rtb_robot.q = q
            self._env.step(self._parent._dt / self._parent._speed)

    def _add_playback_controls(self) -> None:
        """
        Add browser controls for playback speed and replay.
        """
        if self._env is None:
            raise RuntimeError("No Swift environment is running.")

        self._env.add(
            Slider(
                cb=self._set_speed,
                min=0.25,
                max=4.0,
                step=0.25,
                value=self._parent._speed,
                desc="Playback speed",
                unit="x",
            )
        )
        self._env.add(Button(cb=self._request_replay, desc="Replay"))

    def _set_speed(self, value: float) -> None:
        """
        Update the playback speed from the browser speed slider.
        """
        self._parent._speed = self._parent._validate_speed(value)

    def _request_replay(self, _) -> None:
        """
        Mark the current trajectory for replay.
        """
        self._replay_requested = True

    def _wait_for_replay_or_browser_close(
        self,
        poll_interval: float = 0.5,
        timeout: float = 2.0,
    ) -> bool:
        """
        Wait until the replay button is clicked or the browser closes.
        """
        while self._env is not None:
            if self._replay_requested:
                return True
            if not self._browser_is_responsive(timeout):
                self.close()
                return False
            time.sleep(poll_interval)

        return False

    def _hold(self, poll_interval: float = 0.5, timeout: float = 2.0) -> None:
        """
        Keep the script alive until the browser stops responding.
        """
        if self._env is None:
            raise RuntimeError("No Swift environment is running.")

        try:
            while self._env is not None and self._browser_is_responsive(timeout):
                time.sleep(poll_interval)
        finally:
            self.close()

    def _browser_is_responsive(self, timeout: float) -> bool:
        """
        Return whether the Swift browser client still answers websocket calls.
        """
        if self._env is None:
            return False

        self._env.outq.put([True, ["shape_poses", []]])
        try:
            events = json.loads(self._env.inq.get(timeout=timeout))
        except Empty:
            return False

        self._env.process_events(events)
        return True

    def _suppress_expected_websocket_close_tracebacks(self) -> None:
        """
        Suppress expected websocket tracebacks caused by closing the browser.

        Closing the Swift browser tab can make Swift's websocket server thread
        log ``ConnectionClosedError`` tracebacks. They are expected during manual
        shutdown and do not indicate a failed robot simulation, so the simulator
        temporarily raises that logger's threshold while Swift is active.
        """
        if self._websocket_logger_levels is not None:
            return

        logger_names = ("websockets.server",)
        self._websocket_logger_levels = {
            name: logging.getLogger(name).level
            for name in logger_names
        }

        for name in logger_names:
            logging.getLogger(name).setLevel(logging.CRITICAL)

    def _restore_websocket_logging(self) -> None:
        """
        Restore websocket logger levels changed for the Swift session.
        """
        if self._websocket_logger_levels is None:
            return

        for name, level in self._websocket_logger_levels.items():
            logging.getLogger(name).setLevel(level)

        self._websocket_logger_levels = None
