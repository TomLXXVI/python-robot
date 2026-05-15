from copy import deepcopy

import numpy as np

from roboticstoolbox import ERobot
from roboticstoolbox.backends.swift import Swift

from python_robot.base.types import NumpyArray
from python_robot.motion import JointSpaceScheme
from python_robot.manipulator import URDFManipulator

__all__ = ["SwiftSimulator"]


class SwiftSimulator:

    def __init__(self, robot: URDFManipulator | None = None) -> None:
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
        self._env = None

    @staticmethod
    def _get_timestep(t_array: NumpyArray) -> float:
        return float(np.mean(np.diff(t_array)))

    def play_joint_trajectory(
        self,
        q_array: NumpyArray,
        t_array: NumpyArray
    ) -> None:
        self._q_array: NumpyArray = q_array
        self._dt: float = self._get_timestep(t_array)

        self._run()

    def play_joint_scheme(
        self,
        joint_scheme: JointSpaceScheme,
        step: int = 1
    ) -> None:
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

        self.play_joint_trajectory(q_array, t_array)

        if rtb_robot_copy is not None:
            self._rtb_robot = rtb_robot_copy

    def _run(self) -> None:
        if self._q_array is None or self._dt is None:
            raise RuntimeError("Call setup() before run().")

        self._create_env()
        self._run_loop()

    def _create_env(self) -> None:
        self._env = Swift()
        self._env.launch(realtime=True)
        self._env.add(
            self._rtb_robot,
            robot_alpha=1.0,
            collision_alpha=0.0,
            readonly=True
        )

    def _run_loop(self) -> None:
        for q in self._q_array:
            self._rtb_robot.q = q
            self._env.step(self._dt)
