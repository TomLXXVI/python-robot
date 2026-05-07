from __future__ import annotations
import typing

from dataclasses import dataclass

import numpy as np
import spatialmath.base as smb

from .types import *

__all__ = [
    "Vector",
    "Axis",
    "PrincipalAxis",
    "TranslationalVelocity",
    "AngularVelocity",
    "TranslationalAcceleration",
    "AngularAcceleration",
    "SpatialVelocity",
    "SpatialAcceleration",
    "Force",
    "Torque",
    "Wrench",
]


class Vector:
    """
    Represents a Cartesian vector (x, y, z) in 3D space (starting in the
    origin of its reference frame).

    Attributes
    ----------
    coords : ArrayLike3
        Cartesian coordinates in 3D space.
    x : float
        The X component of the vector in 3D space.
    y : float
        The Y component of the vector in 3D space.
    z : float
        The Z component of the vector in 3D space.
    magnitude : float
        Magnitude of the vector.
    direction : NumpyArray
        Cartesian components (x, y, z) of the direction of the vector. Will be
        None if the vector is null, i.e. when the magnitude of the vector is 0.
    theta : float
        Angle with the Z-axis in radians, or None if the vector is null.
    phi : float, optional
        Angle from the X-axis in the XY-plane in radians, or None if the vector
        is null.
    axis : Axis
        The axis of the vector in 3D space, which is actually equivalent to the
        vector's direction. Will be None if the vector is null.
    """
    def __init__(self, coords: ArrayLike3) -> None:
        """
        Creates a Vector object from Cartesian coordinates.

        Parameters
        ----------
        coords : ArrayLike3
            Cartesian coordinates in 3D space.
        """
        self.coords = coords
        self._array: NumpyArray = np.asarray(self.coords, dtype=float)
        self.x = float(self._array[0])
        self.y = float(self._array[1])
        self.z = float(self._array[2])
        self.magnitude = float(np.linalg.norm(self._array))
        if self.isnull():
            self.direction: NumpyArray = np.zeros(3)
            self.theta = 0.0
            self.phi = 0.0
            self.axis = Axis(self)
        else:
            self.direction: NumpyArray = self._array / self.magnitude
            self.theta = float(np.arccos(self._array[2] / self.magnitude))
            self.phi = float(np.arctan2(self._array[1], self._array[0]))
            self.axis: Axis = Axis(self)

    def isnull(self) -> bool:
        """
        Returns True if the vector is null, False otherwise.
        """
        return bool(np.isclose(self.magnitude, 0))

    @classmethod
    def from_spherical(
        cls,
        magnitude: float,
        phi: float,
        theta: float,
        angle_unit: AngleUnit = "rad",
    ) -> Vector:
        """
        Creates a Cartesian vector from spherical coordinates.

        Parameters
        ----------
        magnitude : float
            Magnitude of the vector.
        theta : float
            Angle with the Z-axis.
        phi : float
            Angle from the X-axis in the XY-plane.
        angle_unit : AngleUnits, default = "deg"
            Unit of the angles.
        """
        if angle_unit == "deg":
            phi = np.deg2rad(phi)
            theta = np.deg2rad(theta)
        x = magnitude * np.sin(theta) * np.cos(phi)
        y = magnitude * np.sin(theta) * np.sin(phi)
        z = magnitude * np.cos(theta)
        return cls([x, y, z])

    @classmethod
    def from_axis(cls, axis: Axis, magnitude: float) -> Vector:
        """
        Creates a Cartesian vector with the given magnitude along the given
        axis.

        Parameters
        ----------
        axis : Axis
            Axis (direction) of the vector in 3D space.
        magnitude : float
            Magnitude of the vector.
        """
        return cls(typing.cast(ArrayLike3, axis.direction * magnitude))

    def array(self, extended: bool = False) -> NumpyArray:
        """
        Returns the NumPy array representation of the vector.

        Parameters
        ----------
        extended : bool, default = False
            If True, appends 1 at the end of the array.

        Returns
        -------
        NumpyArray
        """
        if extended:
           return np.append(self._array, 1.0)
        return self._array

    def skew(self) -> NumpyArray:
        """
        Returns the skew symmetric matrix representation of the vector.

        Returns
        -------
        NumpyArray
        """
        return smb.skew(self._array)

    def __str__(self) -> str:
        return f"Vector({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"


class Axis:
    """
    Represents an axis in 3D space.

    Attributes
    ----------
    direction : NumpyArray
        The direction of the axis in space represented as a Cartesian
        unit-vector.
    """
    def __init__(self, coords: Vector | ArrayLike3) -> None:
        """
        Creates an axis in space from Cartesian coordinates.

        Parameters
        ----------
        coords : ArrayLike3 | Vector
            Specifies the direction of the axis in space using a Cartesian
            vector. Only the direction of this vector is considered.

        Raises
        ------
        ValueError
            If coords is a null vector.
        """
        if not isinstance(coords, Vector):
            coords: Vector = Vector(typing.cast(ArrayLike3, coords))
        if coords.isnull():
            self.direction: NumpyArray = np.zeros(3)
        else:
            self.direction: NumpyArray = coords.direction

    @classmethod
    def from_spherical(
        cls,
        phi: float,
        theta: float,
        angle_unit: AngleUnit = "rad"
    ) -> Axis:
        """
        Creates an axis in space from spherical coordinates.

        Parameters
        ----------
        phi : float
            Angle from the X-axis in the XY-plane.
        theta : float
            Angle with the Z-axis.
        angle_unit : AngleUnits, default = "deg"
            Unit of the angles.

        Returns
        -------
        Axis
        """
        v = Vector.from_spherical(1, phi, theta, angle_unit)
        return cls(typing.cast(ArrayLike3, v.direction))

    def __mul__(self, magnitude: float) -> ArrayLike3:
        """
        Creates a Cartesian vector by multiplying the direction of the axis with
        the given magnitude.

        Returns
        -------
        ArrayLike3
        """
        return Vector(typing.cast(ArrayLike3, self.direction * magnitude)).coords

    def __rmul__(self, magnitude: float) -> ArrayLike3:
        """
        Creates a Cartesian vector by multiplying the direction of the axis with
        the given magnitude.

        Returns
        -------
        ArrayLike3
        """
        return Vector(typing.cast(ArrayLike3, self.direction * magnitude)).coords

    def array(self, extended: bool = False) -> NumpyArray:
        """
        Returns the NumPy array representation of the axis.

        Parameters
        ----------
        extended : bool, default = False
            If True, appends 0 at the end of the array.

        Returns
        -------
        NumpyArray
        """
        if extended:
           return np.append(self.direction, 0.0)
        return self.direction

    def skew(self) -> NumpyArray:
        """
        Returns the skew symmetric matrix representation of the axis.

        Returns
        -------
        NumpyArray
        """
        return smb.skew(self.direction)


@dataclass(frozen=True)
class PrincipalAxis:
    """
    Represents the principal axes of the "World Reference Frame" or any other
    frame when seen from that frame itself.
    """
    X = Axis([1, 0, 0])
    Y = Axis([0, 1, 0])
    Z = Axis([0, 0, 1])


class TranslationalVelocity(Vector):
    """
    Represents a translational velocity vector (v_x, v_y, v_z).
    """
    pass


class AngularVelocity(Vector):
    """
    Represents an angular velocity vector (omega_x, omega_y, omega_z).
    """
    @classmethod
    def from_axis(
        cls,
        axis: Axis,
        magnitude: float,
        unit: AngularSpeedUnit = "rad/s"
    ) -> AngularVelocity:
        """
        Creates an angular velocity vector by specifying the rotation axis and
        the magnitude of the angular velocity about this axis.

        Parameters
        ----------
        axis : Axis
            Axis of rotation.
        magnitude : float
            Rotational speed about the axis.
        unit : AngularSpeedUnits, default = "deg/s"
            Unit of the angular velocity.

        Returns
        -------
        AngularVelocity

        Notes
        -----
        If the unit of angular velocity is set to deg/s, the magnitude will be
        internally converted to rad/s.
        """
        angular_speed = magnitude
        if unit == "deg/s":
            magnitude = np.deg2rad(magnitude)
        else:
            magnitude = angular_speed
        return cls(axis * magnitude)

    @staticmethod
    def jacobian(r: ArrayLike3) -> NumpyArray:
        """
        Calculates the analytical Jacobian that maps the time derivative of a
        rotation vector or angle-axis vector (aka the exponential coordinates of
        a rotation matrix) to angular velocity.

        omega = J(r) * r_dot

        Parameters
        ----------
        r : ArrayLike3
            Rotation vector.

        Returns
        -------
        NumpyArray
            (3 x 3) analytical Jacobian matrix.
        """
        r: NumpyArray = np.asarray(r, dtype=float)
        theta = float(np.linalg.norm(r))
        S = smb.skew(r)

        if np.isclose(theta, 0.0):
            return np.eye(3) + 0.5 * S

        return (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta ** 2 * S
            + (theta - np.sin(theta)) / theta ** 3 * (S @ S)
        )


class TranslationalAcceleration(Vector):
    """
    Represents a translational acceleration vector (a_x, a_y, a_z).
    """
    pass


class AngularAcceleration(Vector):
    """
    Represents an angular acceleration vector (alpha_x, alpha_y, alpha_z).
    """
    @staticmethod
    def jacobian_dot(r: ArrayLike3, r_dot: ArrayLike3) -> NumpyArray:
        """
        Time derivative of the analytical Jacobian that maps the first and
        second time derivatives of a rotation vector or angle-axis vector to
        angular acceleration.

        omega = J(r) * r_dot
        =>
        alpha = d(omega)/dt = J(r) * r_ddot + J_dot(r, r_dot) * r_dot

        Parameters
        ----------
        r: ArrayLike3
            Rotation vector.
        r_dot: ArrayLike3
            First time derivative of the rotation vector.

        Returns
        -------
        NumpyArray
            (3 x 3) time derivative of the analytical Jacobian matrix.
        """
        r: NumpyArray = np.asarray(r, dtype=float)
        r_dot: NumpyArray = np.asarray(r_dot, dtype=float)

        theta = float(np.linalg.norm(r))
        S_dot = smb.skew(r_dot)

        if np.isclose(theta, 0.0):
            return 0.5 * S_dot

        theta_dot = float(r @ r_dot) / theta
        S = smb.skew(r)

        A = (1.0 - np.cos(theta)) / theta ** 2
        B = (theta - np.sin(theta)) / theta ** 3

        A_dot = (
            theta_dot
            * (theta * np.sin(theta) - 2.0 * (1.0 - np.cos(theta)))
            / theta ** 3
        )
        B_dot = (
            theta_dot
            * (theta * (1.0 - np.cos(theta)) - 3.0 * (theta - np.sin(theta)))
            / theta ** 4
        )
        return (
            A_dot * S
            + A * S_dot
            + B_dot * (S @ S)
            + B * (S_dot @ S + S @ S_dot)
        )


class SpatialVelocity:
    """
    Represents the spatial velocity vector of a frame combining the
    translational velocity of its origin and the angular velocity of its
    orientation: (v_x, v_y, v_z, omega_x, omega_y, omega_z).

    Attributes
    ----------
    coords : ArrayLike6
        Coordinates of the spatial velocity vector. By convention, the first
        three coordinates represent the translational velocity of the
        frame's origin and the last three coordinates represent the angular
        velocity of the frame.
    """
    def __init__(self, coords: ArrayLike6) -> None:
        """
        Creates a spatial velocity vector from the given coordinates.

        Parameters
        ----------
        coords : ArrayLike6
            Coordinates of the spatial velocity vector. By convention, the first
            three coordinates represent the translational velocity of the
            frame's origin and the last three coordinates represent the angular
            velocity of the frame.
        """
        self.coords = coords
        self._array: NumpyArray = np.asarray(self.coords, dtype=float)
        self._v = TranslationalVelocity(self._array[0:3])
        self._omega = AngularVelocity(self._array[3:6])

    @classmethod
    def from_components(
        cls,
        v: TranslationalVelocity,
        omega: AngularVelocity
    ) -> SpatialVelocity:
        """
        Creates a spatial velocity vector from the given components.

        Parameters
        ----------
        v : TranslationalVelocity
            The translational velocity component of the spatial velocity vector.
        omega : AngularVelocity
            The angular velocity component of the spatial velocity vector.

        Returns
        -------
        SpatialVelocity
        """
        v_arr = v.array()
        omega_arr = omega.array()
        V_arr = np.hstack((v_arr, omega_arr))
        return cls(V_arr)

    def array(self) -> NumpyArray:
        """
        Returns the NumPy array representation of the spatial velocity vector.

        Returns
        -------
        NumpyArray
        """
        return self._array

    @property
    def v(self) -> TranslationalVelocity:
        """
        Returns the translational velocity component of the spatial velocity.

        Returns
        -------
        TranslationalVelocity
        """
        return self._v

    @property
    def omega(self) -> AngularVelocity:
        """
        Returns the angular velocity component of the spatial velocity.

        Returns
        -------
        AngularVelocity
        """
        return self._omega

    @classmethod
    def from_pose(
        cls,
        p: ArrayLike6,
        p_dot: ArrayLike6
    ) -> SpatialVelocity:
        """
        Given a pose vector p and its first time derivative, returns the
        corresponding spatial velocity vector of the frame.

        Parameters
        ----------
        p: ArrayLike6
            Pose vector (x, y, z, rx, ry, rz).
        p_dot: ArrayLike6
            First time derivative of the pose vector (x_dot, y_dot, z_dot,
            rx_dot, ry_dot, rz_dot).

        Returns
        -------
        SpatialVelocity
        """
        v = np.asarray(p_dot[:3], dtype=float)
        r = np.asarray(p[3:], dtype=float)
        r_dot = np.asarray(p_dot[3:], dtype=float)
        omega = AngularVelocity.jacobian(r) @ r_dot
        V = np.concatenate((v, omega))
        return SpatialVelocity(V)


class SpatialAcceleration:
    """
    Represents the spatial acceleration vector of a frame combining the
    translational acceleration of its origin and the angular acceleration of its
    orientation: (a_x, a_y, a_z, alpha_x, alpha_y, alpha_z).

    Attributes
    ----------
    coords : ArrayLike6
        Coordinates of the spatial acceleration vector. By convention, the first
        three coordinates represent the translational acceleration of the
        frame's origin and the last three coordinates represent the angular
        acceleration of the frame.
    """
    def __init__(self, coords: ArrayLike6) -> None:
        self.coords = coords
        self._array: NumpyArray = np.asarray(self.coords, dtype=float)
        self._a = TranslationalAcceleration(self._array[0:3])
        self._alpha = AngularAcceleration(self._array[3:6])

    @classmethod
    def from_components(
        cls,
        a: TranslationalAcceleration,
        alpha: AngularAcceleration
    ) -> SpatialAcceleration:
        """
        Creates a spatial acceleration vector from the given components.

        Parameters
        ----------
        a : TranslationalAcceleration
            The translational acceleration component of the spatial acceleration
            vector.
        alpha : AngularAcceleration
            The angular acceleration component of the spatial acceleration
            vector.

        Returns
        -------
        SpatialAcceleration
        """
        a_arr = a.array()
        alpha_arr = alpha.array()
        A_arr = np.hstack((a_arr, alpha_arr))
        return cls(A_arr)

    def array(self) -> NumpyArray:
        """
        Returns the NumPy array representation of the spatial acceleration
        vector.

        Returns
        -------
        NumpyArray
        """
        return self._array

    @property
    def a(self) -> TranslationalAcceleration:
        """
        Returns the translational acceleration component of the spatial
        acceleration.

        Returns
        -------
        TranslationalAcceleration
        """
        return self._a

    @property
    def alpha(self) -> AngularAcceleration:
        """
        Returns the angular acceleration component of the spatial acceleration.

        Returns
        -------
        AngularAcceleration
        """
        return self._alpha

    @classmethod
    def from_pose(
        cls,
        p: ArrayLike6,
        p_dot: ArrayLike6,
        p_ddot: ArrayLike6
    ) -> SpatialAcceleration:
        """
        Given a pose vector p and its first and second time derivative, returns
        the corresponding spatial acceleration vector of the frame.

        Parameters
        ----------
        p: ArrayLike6
            Pose vector (x, y, z, rx, ry, rz).
        p_dot: ArrayLike6
            First time derivative of the pose vector (x_dot, y_dot, z_dot,
            rx_dot, ry_dot, rz_dot).
        p_ddot: ArrayLike6
            Second time derivative of the pose vector (x_ddot, y_ddot, z_ddot,
            rx_ddot, ry_ddot, rz_ddot).

        Returns
        -------
        SpatialAcceleration
        """
        a = np.asarray(p_ddot[:3], dtype=float)
        r = np.asarray(p[3:], dtype=float)
        r_dot = np.asarray(p_dot[3:], dtype=float)
        r_ddot = np.asarray(p_ddot[3:], dtype=float)

        J = AngularVelocity.jacobian(r)
        J_dot = AngularAcceleration.jacobian_dot(r, r_dot)

        alpha = J @ r_ddot + J_dot * r_dot
        A = np.concatenate((a, alpha))
        return SpatialAcceleration(A)


class Force(Vector):
    """
    Represents a force vector (F_x, F_y, F_z).
    """
    pass


class Torque(Vector):
    """
    Represents a torque vector (T_x, T_y, T_z).
    """
    @classmethod
    def from_axis(
        cls,
        axis: Axis,
        magnitude: float,
    ) -> Torque:
        """
        Creates a torque vector by specifying the rotation axis and the
        magnitude of the torque (moment) about this axis.

        Parameters
        ----------
        axis : Axis
            Axis of rotation.
        magnitude : float
            Magnitude of the torque about the axis.

        Returns
        -------
        Torque
        """
        return cls(axis * magnitude)


class Wrench:
    """
    Force-torque vector combining force and torque vectors
    (F_x, F_y, F_z, Tx, Ty, Tz).

    Attributes
    ----------
    coords : ArrayLike6
        Coordinates of the force-torque vector. By convention, the first
        three coordinates represent the force vector and the last three
        coordinates represent the torque vector.
    """
    def __init__(self, coords: ArrayLike6) -> None:
        self.coords = coords
        self._array: NumpyArray = np.asarray(self.coords, dtype=float)
        self._F = Force(self._array[0:3])
        self._T = Torque(self._array[3:6])

    @classmethod
    def from_components(
        cls,
        F: Force,
        T: Torque
    ) -> Wrench:
        """
        Creates a wrench vector from the given components.

        Parameters
        ----------
        F : Force
            The force component of the wrench vector.
        T : Torque
            The torque component of the wrench vector.

        Returns
        -------
        Wrench
        """
        F_arr = F.array()
        T_arr = T.array()
        W_arr = np.hstack((F_arr, T_arr))
        return cls(W_arr)

    def array(self) -> NumpyArray:
        """
        Returns the NumPy array representation of the spatial velocity vector.

        Returns
        -------
        NumpyArray
        """
        return self._array

    @property
    def F(self) -> Force:
        """
        Returns the force component of the wrench vector.

        Returns
        -------
        Force
        """
        return self._F

    @property
    def T(self) -> Torque:
        """
        Returns the torque component of the wrench vector.

        Returns
        -------
        Torque
        """
        return self._T
