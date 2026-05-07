"""
Implements a class representing an XYZ gantry robot with three prismatic joints.
"""
from typing import Sequence

import numpy as np

from ..base import Frame
from ..base.types import NumpyArray
from ..manipulator import IKSolverSpec, KinematicChainViewer, SerialLinkManipulator
from ..manipulator.links import LinkDynamicParams
from ..manipulator.links.ets import PrismaticETSLink


__all__ = ["XYZGantry"]


class XYZGantry(SerialLinkManipulator):
    """
    Implements a Cartesian XYZ gantry robot with three orthogonal prismatic
    joints.

    Joint coordinates are ordered as [x, y, z] and directly correspond to the
    end-effector position in the base frame. The end-effector orientation stays
    aligned with the base frame.
    """
    def __init__(
        self,
        q_lim: Sequence[Sequence[float]] | None = None,
        dynamics: Sequence[LinkDynamicParams | None] | None = None,
        joint_coords: Sequence[float] | None = None,
    ) -> None:
        """
        Creates an XYZ gantry robot.

        Parameters
        ----------
        q_lim: Sequence[Sequence[float]] | None, optional
            Mechanical limits for the three prismatic joints. If provided, it
            must contain three pairs: one lower and upper limit for each axis.
        dynamics: Sequence[LinkDynamicParams | None] | None, optional
            Dynamic parameters for the three prismatic links.
        joint_coords: Sequence[float] | None, optional
            Initial joint coordinates [x, y, z]. If None, all joints are set to
            zero.
        """
        self.q_lim = self._normalize_q_lim(q_lim)
        self.dynamics = self._normalize_dynamics(dynamics)

        super().__init__(self._create_links(), joint_coords=joint_coords)
        self._viewer = KinematicChainViewer(self)

    @staticmethod
    def _normalize_q_lim(
        q_lim: Sequence[Sequence[float]] | None,
    ) -> tuple[tuple[float, float], ...] | None:
        if q_lim is None:
            return None

        limits = tuple(tuple(float(elem) for elem in seq) for seq in q_lim)
        if len(limits) != 3 or any(len(tup) != 2 for tup in limits):
            raise ValueError(
                "q_lim must contain three joint-limit pairs: one lower and "
                "upper limit for each prismatic joint."
            )
        return limits  # type: ignore[return-value]

    @staticmethod
    def _normalize_dynamics(
        dynamics: Sequence[LinkDynamicParams | None] | None,
    ) -> tuple[LinkDynamicParams | None, ...]:
        if dynamics is None:
            return None, None, None

        dynamics_ = tuple(dynamics)
        if len(dynamics_) != 3:
            raise ValueError("dynamics must contain exactly three elements.")
        return dynamics_

    def _create_links(self) -> list[PrismaticETSLink]:
        axes = ("x", "y", "z")
        return [
            PrismaticETSLink(
                axis,  # type: ignore
                dynamics=self.dynamics[i],
                q_lim=None if self.q_lim is None else self.q_lim[i],
            )
            for i, axis in enumerate(axes)
        ]

    def inv_kin_analytic(self, ee_frame: Frame) -> NumpyArray:
        """
        Returns the XYZ joint coordinates for a desired end-effector frame.
        """
        q = np.asarray(ee_frame.origin, dtype=float)
        self._check_joint_limits(q)
        return q

    def inv_kin(
        self,
        ee_frame: Frame,
        ini_guess: Sequence[float] | None = None,
        which_solver: IKSolverSpec = "LM",
        **kwargs,
    ) -> NumpyArray:
        return super().inv_kin(
            ee_frame=ee_frame,
            ini_guess=ini_guess,
            which_solver=which_solver,
            mask=[1, 1, 1, 0, 0, 0],
            **kwargs,
        )
