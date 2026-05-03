from typing import Sequence

import numpy as np
from roboticstoolbox import ERobot

from python_robot.base.types import NumpyArray

from .kinematic_chain import AbstractLink, KinematicChain

__all__ = ["SerialLinkManipulator"]


class SerialLinkManipulator(KinematicChain):
    """
    Represents a serial-links manipulator with kinematic and dynamic properties.
    """
    def __init__(
        self,
        links: Sequence[AbstractLink],
        joint_coords: Sequence[float] | None = None
    ) -> None:
        super().__init__(links, joint_coords)
        self._erobot = self._create_erobot()

    def _create_erobot(self) -> ERobot:
        links = [link.rtb_link.copy() for link in self]
        for j, link in enumerate(links):
            link.jindex = j
        return ERobot(links)

    @property
    def erobot(self) -> ERobot:
        """
        Returns a ERobot instance of the serial-links manipulator (see
        roboticstoolbox.robot.ERobot.py).
        """
        return self._erobot

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

        q = np.asarray(self._convert_to(q), dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)

        robot = self.erobot

        if gravity is None:
            tau = robot.rne(q, qd, qdd)
        else:
            tau = robot.rne(q, qd, qdd, gravity=np.asarray(gravity, dtype=float))

        return np.asarray(tau, dtype=float)
