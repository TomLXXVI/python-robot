from abc import ABC

import numpy as np
from spatialmath import SE3
from roboticstoolbox import ET, ETS
from roboticstoolbox import Link as RTBLink

from ....base.types import ArrayLike3
from ....base import Axis, Vector
from ..link import (
    AbstractLink,
    AbstractRevoluteLink, AbstractPrismaticLink,
    LinkDynamicParams
)

__all__ = ["RevolutePoELink", "PrismaticPoELink"]


class AbstractPoELink(AbstractLink, ABC):
    """
    Defines a link using the PoE-formulation (Product of Exponentials).
    Internally this is then converted to an ETS-link.
    """
    def __init__(
        self,
        axis: ArrayLike3,
        position: ArrayLike3 | None,
        joint_limits: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams | None = None
    ) -> None:
        self._axis = axis
        self._position = position
        self._joint_limits = joint_limits
        link_length = 0.0

        ets = self._create_link_ETS()
        rtb_link = RTBLink(ets)

        super().__init__(link_length, rtb_link, dynamics)

    def _create_link_ETS(self) -> ETS:
        pole = Vector(self._position) if self._position is not None else None
        ets = self._screw_to_ETS(Axis(self._axis), pole, self._joint_limits)
        return ets

    @staticmethod
    def _screw_to_ETS(
        axis: Axis,
        pole: Vector | None = None,
        qlim: tuple[float, float] | None = None,
    ) -> ETS:
        """
        Transforms a screw to ETS.

        Parameters
        ----------
        axis : Axis
            Defines the direction of the screw axis in 3D space.
        pole : Vector, optional
            Point on the screw axis that specifies the location of the screw axis in
            3D space.
        qlim : tuple[float, float], optional
            Lower and upper mechanical limit of the joint.

        Returns
        -------
        ETS
        """
        def _axis_frame(axis: Axis, pole: Vector | None = None) -> SE3:
            z_axis = np.asarray(axis.direction, dtype=float)
            x_hint = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            y_axis = np.cross(z_axis, x_hint)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)

            R = np.column_stack((x_axis, y_axis, z_axis))
            t = np.zeros(3) if pole is None else np.asarray(pole, dtype=float)
            return SE3.Rt(R, t)

        qlim_arr = np.asarray(qlim, dtype=float) if qlim is not None else None
        axis_frame = _axis_frame(axis, pole)
        axis_frame_inv = axis_frame.inv()

        ets_parts: list[ET] = []
        if not np.allclose(axis_frame.A, np.eye(4)):
            ets_parts.append(ET.SE3(axis_frame))

        ets_parts.append(ET.tz(qlim=qlim_arr) if pole is None else ET.Rz(qlim=qlim_arr))

        if not np.allclose(axis_frame_inv.A, np.eye(4)):
            ets_parts.append(ET.SE3(axis_frame_inv))

        return ETS(ets_parts)


class RevolutePoELink(AbstractPoELink, AbstractRevoluteLink):

    def __init__(
        self,
        rotation_axis: ArrayLike3,
        position: ArrayLike3,
        limits_joint_angle: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams | None = None
    ) -> None:
        """
        Creates a revolute PoE-link.

        Parameters
        ----------
        rotation_axis: ArrayLike3
            Unit vector that specifies the direction of the revolute axis w.r.t.
            the base frame of the manipulator.
        position: ArrayLike3
            Position of the rotation axis w.r.t. the base frame of the
            manipulator.
        limits_joint_angle: tuple[float, float], optional
            Lower and upper mechanical limit of the revolute joint.
        dynamics: LinkDynamicParams, optional
            Data-object containing dynamical parameters of the revolute link.
        """
        super().__init__(rotation_axis, position, limits_joint_angle, dynamics)


class PrismaticPoELink(AbstractPoELink, AbstractPrismaticLink):

    def __init__(
        self,
        translation_axis: ArrayLike3,
        limits_link_offset: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams | None = None
    ) -> None:
        """
        Creates a prismatic PoE-link.

        Parameters
        ----------
        translation_axis: ArrayLike3
            Unit vector that specifies the direction of the prismatic axis
            w.r.t. the base frame of the manipulator.
        limits_link_offset: tuple[float, float], optional
            Lower and upper mechanical limit of the prismatic joint.
        dynamics: LinkDynamicParams, optional
            Data-object containing dynamical parameters of the prismatic link.
        """
        super().__init__(translation_axis, None, limits_link_offset, dynamics)
