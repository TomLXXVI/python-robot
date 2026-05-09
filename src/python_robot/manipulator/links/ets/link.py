"""
ETS links.

Link description is done with an Elementary Transform Sequence (ETS).
"""
from typing import Literal, Sequence
from abc import ABC, abstractmethod

import numpy as np

from roboticstoolbox import ET, ETS
from roboticstoolbox import Link as RBTLink

from python_robot.base.types import AngleUnit, NumpyArray
from python_robot.base import Frame

from ..link import AbstractLink, LinkDynamicParams
from .elementary_transform import ElementaryTransform


__all__ = ["RevoluteETSLink", "PrismaticETSLink", "RLink", "PLink"]


JointAxis = Literal["x", "y", "z"]


class AbstractETSLink(AbstractLink, ABC):

    def __init__(
        self,
        joint_axis: JointAxis,
        delta_x: float | None = None,
        delta_y: float | None = None,
        delta_z: float | None = None,
        theta_x: float | None = None,
        theta_y: float | None = None,
        theta_z: float | None = None,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        super().__init__(angle_unit, dynamics, q_lim)

        self.joint_axis = joint_axis
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z
        self.theta_x = self._set_theta(theta_x, angle_unit)
        self.theta_y = self._set_theta(theta_y, angle_unit)
        self.theta_z = self._set_theta(theta_z, angle_unit)

        self.et_seq = self._get_et_sequence()
        self._rbt_link = RBTLink(self.ets, **self._rbt_q_lim_kwargs())
        self._apply_dynamics_to_rbt_link(self._rbt_link)

    def _set_theta(self, theta: float | None, angle_unit: AngleUnit | None) -> float | None:
        if theta is not None and (angle_unit == "deg" or self.defaults.angle_unit == "deg"):
            return float(np.deg2rad(theta))
        return theta

    @abstractmethod
    def _get_var_et(self, joint_axis: JointAxis) -> ET:
        ...

    @abstractmethod
    def _get_const_et_sequence(self) -> list[ET]:
        ...

    def _get_et_sequence(self) -> list[ET]:
        et_seq = self._get_const_et_sequence()
        var_et = self._get_var_et(self.joint_axis)
        et_seq.append(var_et)
        return et_seq

    @property
    def ets(self) -> ETS:
        """
        Returns the underlying ETS object from Robotics Toolbox.
        """
        return ETS(list(self.et_seq))

    @property
    def rbt_link(self) -> RBTLink:
        """
        Returns the underlying Link object from Robotics Toolbox.
        """
        return self._rbt_link

    def _get_frame(self) -> Frame:
        if self._rbt_link is not None and self._variable is not None:
            SE3_mat = self._rbt_link.A(self._variable)
            return Frame.from_matrix(SE3_mat, angle_unit="rad")
        raise ValueError("link is not configured.")

    @property
    def link_length(self) -> float:
        return np.sqrt(
            np.sum(
                np.square(
                    np.asarray(
                        [self.delta_x, self.delta_y, self.delta_z],
                        dtype=float
        ))))


class RevoluteETSLink(AbstractETSLink):

    def __init__(
        self,
        joint_axis: JointAxis,
        delta_x: float | None = None,
        delta_y: float | None = None,
        delta_z: float | None = None,
        theta_x: float | None = None,
        theta_y: float | None = None,
        theta_z: float | None = None,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        """
        Configures a revolute ETS link. The joint variable of the link is a
        rotation angle.

        Parameters
        ----------
        joint_axis: JointAxis
            Specifies the axis of the revolute joint. Either "x", "y" or "z".
        delta_x: float | None, default=None
            Translation distance along the x-axis, if needed.
        delta_y: float | None, default=None
            Translation distance along the y-axis, if needed.
        delta_z: float | None, default=None
            Translation distance along the z-axis, if needed.
        theta_x: float | None, default=None
            Rotation angle about the x-axis, if needed. Note: if for example the
            joint_axis is specified as "x", theta_x is the corresponding joint
            variable and must be left to None at instantiation.
        theta_y: float | None, default=None
            Rotation angle about the y-axis, if needed. See note at theta_x.
        theta_z: float | None, default=None
            Rotation angle about the z-axis, if needed. See note at theta_x.
        angle_unit: AngleUnit | None, default=None
            The unit used for *all* type of angles (theta_x, theta_y, and
            theta_z) entered by the user (either "deg" or "rad").
            If None, uses the default unit set at the class level (which is
            "rad", but this can be modified through class attribute
            defaults.angle_unit).
            Note that internally all angles will be converted to radians.
        dynamics: LinkDynamics | None, default=None
            Adds dynamic link properties to the link object.
        q_lim: Sequence[float] | NumpyArray | None, default=None
            Mechanical joint limits. Revolute limits use angle_unit and are
            converted to radians internally.
        """
        super().__init__(
            joint_axis,
            delta_x, delta_y, delta_z,
            theta_x, theta_y, theta_z,
            angle_unit,
            dynamics,
            q_lim
        )
        self._angle: float | None = None

    def _get_var_et(self, joint_axis: JointAxis) -> ET:
        match joint_axis:
            case "x":
                if self.theta_x is None:
                    return ElementaryTransform.rotx(self.theta_x)
                raise ValueError("'theta_x' is not None.")
            case "y":
                if self.theta_y is None:
                    return ElementaryTransform.roty(self.theta_y)
                raise ValueError("'theta_y' is not None.")
            case "z":
                if self.theta_z is None:
                    return ElementaryTransform.rotz(self.theta_z)
                raise ValueError("'theta_z' is not None.")
            case _:
                raise ValueError(f"Wrong joint axis.")

    def _get_const_et_sequence(self) -> list[ET]:
        et_seq = []
        if self.joint_axis != "x" and self.theta_x is not None:
            et_seq.append(ElementaryTransform.rotx(self.theta_x))
        if self.joint_axis != "y" and self.theta_y is not None:
            et_seq.append(ElementaryTransform.roty(self.theta_y))
        if self.joint_axis != "z" and self.theta_z is not None:
            et_seq.append(ElementaryTransform.rotz(self.theta_z))
        if self.delta_x is not None:
            et_seq.append(ElementaryTransform.translx(self.delta_x))
        if self.delta_y is not None:
            et_seq.append(ElementaryTransform.transly(self.delta_y))
        if self.delta_z is not None:
            et_seq.append(ElementaryTransform.translz(self.delta_z))
        return et_seq

    @property
    def angle(self) -> float | None:
        """
        Returns the rotation angle of the link.
        If self.angle_unit == "deg", the angle is returned in degrees.
        """
        if self._angle is not None:
            if self.angle_unit == "rad":
                return self._angle
            else:
                return float(np.rad2deg(self._angle))
        return None

    @angle.setter
    def angle(self, v: float) -> None:
        """
        Sets the rotation angle of the link.
        If self.angle_unit == "deg", the angle must be set in degrees.
        """
        self._angle = v if self.angle_unit == "rad" else float(np.deg2rad(v))
        self._variable = self._angle

    @property
    def variable(self) -> float | None:
        return super().variable

    @variable.setter
    def variable(self, v: float) -> None:
        self.angle = v

    @property
    def is_revolute(self) -> bool:
        return True

    @property
    def is_prismatic(self) -> bool:
        return False


class PrismaticETSLink(AbstractETSLink):

    def __init__(
        self,
        joint_axis: JointAxis,
        delta_x: float | None = None,
        delta_y: float | None = None,
        delta_z: float | None = None,
        theta_x: float | None = None,
        theta_y: float | None = None,
        theta_z: float | None = None,
        angle_unit: AngleUnit | None = None,
        dynamics: LinkDynamicParams | None = None,
        q_lim: Sequence[float] | NumpyArray | None = None,
    ) -> None:
        """
        Configures a prismatic ETS link. The joint angle of the link is a
        linear offset.

        Parameters
        ----------
        joint_axis: JointAxis
            Specifies the axis of the prismatic joint. Either "x", "y" or "z".
        delta_x: float | None, default=None
            Translation distance along the x-axis, if needed. Note: if for
            example the joint_axis is specified as "x", delta_x is the
            corresponding joint variable and must be left to None at
            instantiation.
        delta_y: float | None, default=None
            Translation distance along the y-axis, if needed. See note at delta_x.
        delta_z: float | None, default=None
            Translation distance along the z-axis, if needed. See note at delta_x.
        theta_x: float | None, default=None
            Rotation angle about the x-axis, if needed.
        theta_y: float | None, default=None
            Rotation angle about the y-axis, if needed.
        theta_z: float | None, default=None
            Rotation angle about the z-axis, if needed.
        angle_unit: AngleUnit | None, default=None
            The unit used for *all* type of angles (theta_x, theta_y, and
            theta_z) entered by the user (either "deg" or "rad").
            If None, uses the default unit set at the class level (which is
            "rad", but this can be modified through class attribute
            defaults.angle_unit).
            Note that internally all angles will be converted to radians.
        dynamics: LinkDynamics | None, default=None
            Adds dynamic link properties to the link object.
        q_lim: Sequence[float] | NumpyArray | None, default=None
            Mechanical joint limits.
        """
        super().__init__(
            joint_axis,
            delta_x, delta_y, delta_z,
            theta_x, theta_y, theta_z,
            angle_unit,
            dynamics,
            q_lim
        )
        self._offset: float | None = None

    def _get_var_et(self, joint_axis: JointAxis) -> ET:
        match joint_axis:
            case "x":
                if self.delta_x is None:
                    return ElementaryTransform.translx(self.delta_x)
                raise ValueError("'delta_x' is not None.")
            case "y":
                if self.delta_y is None:
                    return ElementaryTransform.transly(self.delta_y)
                raise ValueError("'delta_y' is not None.")
            case "z":
                if self.delta_z is None:
                    return ElementaryTransform.translz(self.delta_z)
                raise ValueError("'delta_z' is not None.")
            case _:
                raise ValueError(f"Wrong joint axis.")

    def _get_const_et_sequence(self) -> list[ET]:
        et_seq = []
        if self.joint_axis != "x" and self.delta_x is not None:
            et_seq.append(ElementaryTransform.translx(self.delta_x))
        if self.joint_axis != "y" and self.delta_y is not None:
            et_seq.append(ElementaryTransform.transly(self.delta_y))
        if self.joint_axis != "z" and self.delta_z is not None:
            et_seq.append(ElementaryTransform.translz(self.delta_z))
        if self.theta_x is not None:
            et_seq.append(ElementaryTransform.rotx(self.theta_x))
        if self.theta_y is not None:
            et_seq.append(ElementaryTransform.roty(self.theta_y))
        if self.theta_z is not None:
            et_seq.append(ElementaryTransform.rotz(self.theta_z))
        return et_seq

    @property
    def offset(self) -> float | None:
        """Returns the linear offset of the link."""
        return self._offset

    @offset.setter
    def offset(self, v: float) -> None:
        """Sets the linear offset of the link."""
        self._offset = v
        self._variable = self._offset

    @property
    def variable(self) -> float | None:
        return super().variable

    @variable.setter
    def variable(self, v: float) -> None:
        self.offset = v

    @property
    def is_revolute(self) -> bool:
        return False

    @property
    def is_prismatic(self) -> bool:
        return True


RLink = RevoluteETSLink
PLink = PrismaticETSLink
