"""
Implements a class representing a "Simple Planar Three-Link Manipulator" (RRR).
"""
from typing import Sequence, Literal

import numpy as np

from ..base.types import NumpyArray
from ..base import Frame
from ..manipulator.links.denavit_hartenberg import RevoluteMDHLink
from ..manipulator.links import LinkDynamicParams
from ..manipulator import SerialLinkManipulator, RefFrame, IKSolverSpec


__all__ = ["Planar3R"]


class Planar3R(SerialLinkManipulator):
    """
    Implements a simple planar three-link manipulator with three revolute
    joints (RRR).

    The first joint, near the fixed base, is analog to the shoulder of an arm.
    The second joint is analog to the elbow. And the third joint is analog to
    the wrist of the arm.
    """
    def __init__(
        self,
        l1: float,
        l2: float,
        point_masses: Sequence[float] | None = None,
        q_lim: Sequence[Sequence[float]] | None = None,
        base_frame: Frame | None = None,
        tool_frame: Frame | None = None,
    ) -> None:
        """
        Creates an instance of SP3RLinkManipulator.

        Parameters
        ----------
        l1: float
            Length of links 1 in the chain.
        l2: float
            Length of links 2 in the chain.
        point_masses: Sequence[float] | None, optional
            Point masses attached to the distal end of each modeled link.
            If two masses are given, they are assigned to the two non-zero
            planar arm segments. If three masses are given, they are assigned
            to all three revolute links in the kinematic chain.
        q_lim: Sequence[Sequence[float]] | None, optional
            Mechanical joint limits for the three revolute joints, in radians.
            If provided, it must contain three pairs: one lower and upper limit
            for each joint.
        """
        self.l1 = l1
        self.l2 = l2

        self.point_masses = self._normalize_point_masses(point_masses)
        self.q_lim = self._normalize_q_lim(q_lim)
        links = self._create_links()

        super().__init__(links, base_frame=base_frame, tool_frame=tool_frame)

    @staticmethod
    def _normalize_point_masses(point_masses: Sequence[float] | None) -> tuple[float, ...]:
        if point_masses is None:
            return 0.0, 0.0, 0.0

        masses = tuple(float(m) for m in point_masses)
        if len(masses) == 2:
            return 0.0, masses[0], masses[1]
        if len(masses) == 3:
            return masses

        raise ValueError(
            "Point_masses must contain either two masses for the two non-zero "
            "segments, or three masses for all links."
        )

    @staticmethod
    def _normalize_q_lim(q_lim: Sequence[Sequence[float]] | None) -> tuple[tuple[float, float], ...] | None:
        if q_lim is None:
            return None

        limits = tuple(
            tuple(float(elem) for elem in seq)
            for seq in q_lim
        )
        if len(limits) != 3 or any(len(tup) != 2 for tup in limits):
            raise ValueError(
                "q_lim must contain three joint-limit pairs: one lower and "
                "upper limit for each revolute joint."
            )
        return limits  # type: ignore[return-value]

    def _create_links(self) -> list[RevoluteMDHLink]:

        def _create_point_mass_dynamics(self_) -> list[LinkDynamicParams]:
            link_lengths = (0.0, self_.l1, self_.l2)
            return [
                LinkDynamicParams(
                    mass=mass,
                    center_of_mass=(length, 0.0, 0.0),
                    inertia=np.zeros((3, 3)),
                )
                for mass, length in zip(self_.point_masses, link_lengths)
            ]

        dynamics = _create_point_mass_dynamics(self)

        link1 = RevoluteMDHLink(
            link_length=0.0,  # length between J0 (fixed base) and J1
            twist_angle=0.0,
            link_offset=0.0,
            dynamics=dynamics[0],
            limits_joint_angle=None if self.q_lim is None else self.q_lim[0],
        )
        link2 = RevoluteMDHLink(
            link_length=self.l1,  # length between J1 and J2
            twist_angle=0.0,
            link_offset=0.0,
            dynamics=dynamics[1],
            limits_joint_angle=None if self.q_lim is None else self.q_lim[1],
        )
        link3 = RevoluteMDHLink(
            link_length=self.l2,  # length between J2 and J3
            twist_angle=0.0,
            link_offset=0.0,
            dynamics=dynamics[2],
            limits_joint_angle=None if self.q_lim is None else self.q_lim[2],
        )
        return [link1, link2, link3]

    def inv_kin_analytic(
        self,
        ee_frame: Frame,
        elbow_up: bool = True,
    ) -> NumpyArray:
        """
        Given the desired position and orientation of the end-effector frame,
        calculates and returns the corresponding joint angles.

        Parameters
        ----------
        ee_frame: Frame
            Desired end-effector pose.
        elbow_up: bool, optional
            If True, the "elbow-up" solution is returned (this is the default).
            Otherwise, the "elbow-down" solution is returned.

        Returns
        -------
        NumpyArray
            The calculated joint angles in radians.

        Notes
        -----
        Calling method ikine_analytical(...) does not change the configuration
        state of the underlying KinematicChain object. To update the
        configuration state, call method fwd_kin(...) with the returned joint
        angles.
        """
        link_frame = self._tcp_frame_to_link_frame(ee_frame)
        x, y = link_frame.origin[0], link_frame.origin[1]
        phi = link_frame.orient_angles[2]

        c2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        if c2 > 1 or c2 < -1:
            raise ValueError(
                "The goal is too far for the manipulator to reach."
            )

        s2 = np.sqrt(1 - c2 ** 2)
        if not elbow_up:
            s2 = -s2

        theta2 = np.arctan2(s2, c2)

        k1 = self.l1 + self.l2 * c2
        k2 = self.l2 * s2
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        theta3 = phi - theta1 - theta2
        return np.array([theta1, theta2, theta3])

    def inv_kin(
        self,
        ee_frame: Frame,
        ini_guess: Sequence[float] | None = None,
        which_solver: IKSolverSpec = "LM",
        **kwargs
    ) -> NumpyArray:
        return super().inv_kin(
            ee_frame=ee_frame,
            ini_guess=ini_guess,
            which_solver=which_solver,
            mask=[1, 1, 0, 0, 0, 1],  # [x, y, z, alpha, beta, gamma] -> to indicate that this is a planar robot
            **kwargs
        )

    def jacobian(
        self,
        joint_coords: Sequence[float] | None = None,
        ref_frame: RefFrame = "world"
    ) -> NumpyArray:
        """
        Adapts the jacobian() method from the parent class to the 2D-case.
        """
        jac = super().jacobian(joint_coords, ref_frame)
        M = jac[[0, 1, -1], :]
        return M

    def jacobian_dot(
        self,
        joint_coords: Sequence[float],
        joint_velocities: Sequence[float],
        representation: Literal["rpy/xyz", "rpy/zyx", "eul", "exp"] | None = None,
    ) -> NumpyArray:
        """
        Adapts the jacobian_dot() method from the parent class to the 2D-case.
        """
        jac_dot = super().jacobian_dot(joint_coords, joint_velocities, representation)
        M = jac_dot[[0, 1, -1], :]
        return M
