from typing import Sequence, Any

import numpy as np
from roboticstoolbox import ERobot

from python_robot.base.types import NumpyArray
from python_robot.base import Frame

from .kinematic_chain import AbstractLink, KinematicChain

__all__ = ["SerialLinkManipulator"]


class SerialLinkManipulator(KinematicChain):
    """
    Models a serial-link, single-arm manipulator for kinematic and dynamic
    simulations.
    """
    def __init__(
        self,
        links: Sequence[AbstractLink],
        joint_coords: Sequence[float] | None = None,
        base_frame: Frame | None = None,
        tool_frame: Frame | None = None,
        *,
        plot_options: dict[str, Any] | None = None,
        anim_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates a SerialLinkManipulator object.

        Parameters
        ----------
        links: Sequence[AbstractLink]
            List of links in the chain, ordered from base to tool.
        joint_coords: Sequence[float], optional
            Initial values for the joint variables of the links. If None, all
            joint variables are set to zero.
        base_frame: Frame, optional.
            Fixed base frame of the kinematic chain relative to the station
            frame or world frame. If None, the base frame coincides with the
            world frame.
        tool_frame: Frame, optional.
            Tool-center point (TCP) frame or end-effector frame relative to the
            frame of the last link in the kinematic chain. If None, the tool
            frame coincides with the last link frame.
        plot_options: dict[str, Any], optional
            Global plot options used with every call to plot() or plot_async().
        anim_options: dict[str, Any], optional
            Global animation options used with every call to animate() or
            animate_async().
        """
        super().__init__(
            links, joint_coords, base_frame, tool_frame,
            plot_options=plot_options,
            anim_options=anim_options
        )

    @property
    def erobot(self) -> ERobot:
        """
        Returns a ERobot instance of the serial-links manipulator (see
        roboticstoolbox.robot.ERobot.py).
        """
        return self._erobot

    def has_dynamics(self) -> bool:
        return all([link.dynamics is not None for link in self])

    def inv_dyn(
        self,
        joint_coords: Sequence[float],
        joint_velocities: Sequence[float],
        joint_accelerations: Sequence[float],
        gravity: Sequence[float] | None = None
    ) -> NumpyArray:
        """
        Computes inverse dynamics using the recursive Newton-Euler algorithm.

        Parameters
        ----------
        joint_coords:
            Joint coordinates ordered from base to tool. Revolute joint
            coordinates may use the angle unit assigned to the corresponding
            links.
        joint_velocities:
            Joint velocities ordered from base to tool. Revolute joint
            velocities must be expressed in rad/s.
        joint_accelerations:
            Joint accelerations ordered from base to tool. Revolute joint
            accelerations must be expressed in rad/s².
        gravity:
            Optional override for the physical gravitational acceleration vector,
            expressed in the robot base frame, in m/s².
            If None, the gravity vector stored in the underlying Robotics
            Toolbox robot object is used. For a base frame whose positive z-axis
            points upward, this is typically [0, 0, -9.81].
            Use [0, 0, 0] to ignore the effect of gravity on the computed joint
            torques/forces.

        Returns
        -------
        NumpyArray
            Joint torques/forces ordered from base to tool.
        """
        q = self._check_number_of_joint_coords(joint_coords)
        qd = self._check_number_of_joint_coords(joint_velocities)
        qdd = self._check_number_of_joint_coords(joint_accelerations)

        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)

        robot = self.erobot

        if gravity is None:
            tau = robot.rne(q, qd, qdd)
        else:
            tau = robot.rne(q, qd, qdd, gravity=np.asarray(gravity, dtype=float))

        return np.asarray(tau, dtype=float)
