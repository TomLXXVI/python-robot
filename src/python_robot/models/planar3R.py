"""
Implements a class representing a "Simple Planar Three-Link Manipulator" (RRR).
"""
from typing import Sequence

import numpy as np

from ..base.types import NumpyArray
from ..base import Frame
from ..manipulator.links.denavit_hartenberg import ModifiedRevoluteLink
from ..manipulator.links import LinkDynamicParams
from ..manipulator import SerialLinkManipulator, RefFrame, IKSolverSpec
from ..manipulator import KinematicChainViewer


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
        point_masses: Sequence[float] | None = None
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
        """
        self.l1 = l1
        self.l2 = l2

        self.point_masses = self._normalize_point_masses(point_masses)
        links = self._create_links()

        super().__init__(links)
        self._viewer = KinematicChainViewer(self)

    @staticmethod
    def _normalize_point_masses(
        point_masses: Sequence[float] | None
    ) -> tuple[float, ...]:
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

    def _create_links(self) -> list[ModifiedRevoluteLink]:
        # All links have angles in radians.

        def _create_point_mass_dynamics(self) -> list[LinkDynamicParams]:
            link_lengths = (0.0, self.l1, self.l2)
            return [
                LinkDynamicParams(
                    mass=mass,
                    center_of_mass=(length, 0.0, 0.0),
                    inertia=np.zeros((3, 3)),
                )
                for mass, length in zip(self.point_masses, link_lengths)
            ]

        dynamics = _create_point_mass_dynamics(self)

        link1 = ModifiedRevoluteLink(
            length=0.0, twist=0.0, offset=0.0,  # length between J0 (fixed base) and J1
            dynamics=dynamics[0]
        )
        link2 = ModifiedRevoluteLink(
            length=self.l1, twist=0.0, offset=0.0,  # length between J1 and J2
            dynamics=dynamics[1]
        )
        link3 = ModifiedRevoluteLink(
            length=self.l2, twist=0.0, offset=0.0,  # length between J2 and J3
            dynamics=dynamics[2]
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
        x, y = ee_frame.origin[0], ee_frame.origin[1]
        phi = ee_frame.orient_angles[2]

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
            mask=[1, 1, 0, 0, 0, 1],
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

    def plot(self, **kwargs) -> None:
        """
        Plots the current joint configuration of the kinematic chain in
        3D-space.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments for 3D scene configuration (for details
            see docstring of class WorldScene in visualisation.scene.py).

        Returns
        -------
        None
        """
        self._viewer.plot(**kwargs)

    async def plot_async(self, **kwargs) -> None:
        """
        Plots the current joint-and-links configuration of the kinematic chain in
        3D-space.

        This is an asynchronous version of the plot method that can be used in
        Jupyter notebooks. (When calling this function, you need keyword await
        in front of the method call.)

        Parameters
        ----------
        kwargs: dict
            Additional keyword arguments for scene configuration (for details
            see docstring of class WorldScene  in visualisation.scene.py).

        Returns
        -------
        None
        """
        await self._viewer.plot_async(**kwargs)

    def animate(
        self,
        joint_angle_sets: Sequence[Sequence[float]],
        fps: int = 20,
        step: int = 1,
        gif_path: str | None = None,
        mp4_path: str | None = None,
        show: bool = True,
        show_ee_path: bool = False,
        ee_path_color: str = "orange",
        ee_path_line_width: float = 3.0,
        **kwargs
    ) -> None:
        """
        Animate a sequence of manipulator joint configurations.

        For info about the parameters of this function, see the docstring of
        `manipulator.visualisation.KinematicChainViewer`.
        """
        self._viewer.animate(
            joint_coord_sets=joint_angle_sets,
            fps=fps,
            step=step,
            gif_path=gif_path,
            mp4_path=mp4_path,
            show=show,
            show_ee_path=show_ee_path,
            ee_path_color=ee_path_color,
            ee_path_line_width=ee_path_line_width,
            **kwargs
        )
