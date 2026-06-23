"""
Serial-link manipulator model with kinematics and dynamics helpers.

The module defines :class:`SerialLinkManipulator`, a kinematic chain with
additional dynamic calculations such as inverse dynamics, gravity loading, mass
matrices, friction loads, and end-effector wrench mapping.
"""

from typing import Sequence, Any

import numpy as np

from python_robot.base.types import NumpyArray, ArrayLike3, ArrayLike6
from python_robot.base import Frame, Wrench

from .kinematic_chain import AbstractLink, KinematicChain, RefFrame
from .links import LinkDynamicParams


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

    def has_dynamics(self) -> bool:
        """
        Returns True if all links have been assigned dynamic properties,
        otherwise False.
        """
        return all([link.dynamics is not None for link in self])

    def print_dynamics(self) -> None:
        """
        Directly prints the dynamics properties of all links to standard output.
        """
        if self.has_dynamics():
            for i, link in enumerate(self):
                s = f"link {i}:"
                print(s)
                print("-" * len(s))
                print(link.dynamics)
                print()
        else:
            raise AttributeError(
                "No dynamic properties have been assigned to the links."
            )

    @property
    def gravity(self) -> Sequence[float] | None:
        """
        Returns the gravity vector as seen from the manipulator's base frame.
        """
        return self._erobot.gravity

    @gravity.setter
    def gravity(self, v: ArrayLike3) -> None:
        """
        Sets the (x, y, z)-components of the gravity vector w.r.t. the
        manipulator's base frame.
        """
        self._erobot.gravity = v

    def inv_dyn(
        self,
        joint_coords: Sequence[float],
        joint_velocities: Sequence[float],
        joint_accelerations: Sequence[float],
        gravity: Sequence[float] | None = None
    ) -> NumpyArray:
        """
        Computes the joint forces or torques using the recursive Newton-Euler
        algorithm.

        tau = M(q) * qdd + C(q, qd) * qd + f(qd) + g(q) [+ J^T(q) * W_ee]

        Note: the joint torques/forces due to an external wrench W_ee applied to
        the end-effector are not included in the returned result of ``inv_dyn()``.
        Use method ``get_ee_wrench_joint_torques()`` to get these and add them
        to the result of ``inv_dyn()``.

        Parameters
        ----------
        joint_coords: Sequence[float]
            Joint coordinates ordered from base to tool. Revolute joint angles
            must be expressed in radians.
        joint_velocities: Sequence[float]
            Joint velocities ordered from base to tool. Revolute joint
            velocities must be expressed in rad/s.
        joint_accelerations: Sequence[float]
            Joint accelerations ordered from base to tool. Revolute joint
            accelerations must be expressed in rad/s².
        gravity: Sequence[float], optional
            Overrides the gravitational acceleration vector expressed in the
            robot base frame, in m/s².
            If None, the gravity vector stored in the underlying Robotics
            Toolbox robot object is used (for a base frame whose positive z-axis
            points upward, this is typically [0, 0, -9.81].)
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

    def links_dynamics(self) -> dict[str, LinkDynamicParams]:
        """
        Returns the dynamic parameters of the links.

        Returns
        -------
        dict[str, LinkDynamicParams]
        """
        if self.has_dynamics():
            return {f"link {i}": link.dynamics for i, link in enumerate(self)}  # type: ignore
        raise AttributeError("No dynamic parameters have been assigned to the links.")

    def gravity_load(
        self,
        joint_coords: Sequence[float],
        gravity: Sequence[float] | None = None
    ) -> NumpyArray:
        """
        Returns the joint torques/forces due to gravity alone, i.e. g(q) in the
        dynamic equation below.

        tau = M(q) * qdd + C(q, qd) * qd + f(qd) + g(q) + J^T(q) * W_ee

        Parameters
        ----------
        joint_coords: Sequence[float]
            Joint coordinates ordered from base to tool. Revolute joint angles
            must be expressed in radians.
        gravity: Sequence[float], optional
            Overrides the gravitational acceleration vector expressed in the
            robot base frame, in m/s².
            If None, the gravity vector stored in the underlying Robotics
            Toolbox robot object is used (for a base frame whose positive z-axis
            points upward, this is typically [0, 0, -9.81].)
            Use [0, 0, 0] to ignore the effect of gravity on the computed joint
            torques/forces.

        Returns
        -------
        NumpyArray
        """
        q = self._check_number_of_joint_coords(joint_coords)

        return self.erobot.gravload(
            np.asarray(q, dtype=float),
            np.asarray(gravity, dtype=float)
        )

    def mass_matrix(self, joint_coords: Sequence[float]) -> NumpyArray:
        """
        Returns the mass matrix, aka joint-space inertia matrix M(q).

        tau = M(q) * qdd + C(q, qd) * qd + f(qd) + g(q) + J^T(q) * W_ee

        The diagonal elements m_jj describe the inertia experienced by joint j
        (tau_j = m_jj * qdd_j). The off-diagonal elements m_ij = m_ji are the
        products of inertia which couple the acceleration of joint j (qdd_j) to
        the force/torque on joint i (tau_i).

        Parameters
        ----------
        joint_coords: Sequence[float]
            Joint coordinates ordered from base to tool. Revolute joint angles
            must be expressed in radians.

        Returns
        -------
        NumpyArray
        """
        q = self._check_number_of_joint_coords(joint_coords)
        return self.erobot.inertia(np.asarray(q, dtype=float))

    def friction_load(self, joint_velocities: Sequence[float]) -> NumpyArray:
        """
        Returns the friction load on the joints f(qd) due to viscous friction
        and Coulomb friction. The friction values are lumped, and apply to the
        motor side of the gearbox.

        tau = M(q) * qdd + C(q, qd) * qd + f(qd) + g(q) + J^T(q) * W_ee

        Parameters
        ----------
        joint_velocities: Sequence[float]
            Joint velocities ordered from base to tool. Revolute joint
            velocities must be expressed in rad/s.

        Returns
        -------
        NumpyArray
        """
        qd = self._check_number_of_joint_coords(joint_velocities)
        return self.erobot.friction(np.asarray(qd, dtype=float))

    def gyroscopic_matrix(
        self,
        joint_coords: Sequence[float],
        joint_velocities: Sequence[float],
    ) -> NumpyArray:
        """
        Returns the matrix C(q, qd). Element c_ij couples the velocity of joint
        j to the force/torque acting on joint i. The coupling is due to
        gyroscopic effects: the centripetal torques are proportional to qd_j^2,
        while the Coriolis torques are proportional to qd_i * qd_j.

        tau = M(q) * qdd + C(q, qd) * qd + f(qd) + g(q) + J^T(q) * W_ee

        Parameters
        ----------
        joint_coords: Sequence[float]
            Joint coordinates ordered from base to tool. Revolute joint angles
            must be expressed in radians.
        joint_velocities: Sequence[float]
            Joint velocities ordered from base to tool. Revolute joint
            velocities must be expressed in rad/s.

        Returns
        -------
        NumpyArray
        """
        q = self._check_number_of_joint_coords(joint_coords)
        qd = self._check_number_of_joint_coords(joint_velocities)
        return self.erobot.coriolis(
            np.asarray(q, dtype=float),
            np.asarray(qd, dtype=float)
        )

    def payload(
        self,
        mass: float,
        position: ArrayLike3 | None = None
    ) -> None:
        """
        Adds a payload to the end-effector.

        To set the payload back to zero, call ``payload(0)``.

        Parameters
        ----------
        mass: float
            Point mass in kg.
        position: ArrayLike3, optional
            Position of the point mass w.r.t. the end-effector frame.
            If None, the point mass is considered to be in the origin of the
            end-effector frame.

        Returns
        -------
        None
        """
        if position is None:
            self.erobot.payload(mass)
        else:
            self.erobot.payload(mass, position)

    def get_ee_wrench_joint_torques(
        self,
        W_ee: Wrench | ArrayLike6,
        ref_frame: RefFrame = "world"
    ) -> NumpyArray:
        """
        Returns the joint torques (or forces) due to an external wrench applied
        at the end-effector of the manipulator. This method is actually an alias
        of the method ``get_static_joint_torques()`` of superclass
        `KinematicChain`.
        """
        return self.get_static_joint_torques(W_ee, ref_frame)
