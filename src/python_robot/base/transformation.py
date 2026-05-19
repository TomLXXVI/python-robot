from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from spatialmath import SE3, Twist3
import spatialmath.base as smb

from .types import AngleUnit, NumpyArray
from .vector import Vector, Axis, AngularVelocity, SpatialVelocity


__all__ = ["Transformation", "Translation", "Rotation", "Screw"]


class Transformation(ABC):

    def __init__(self) -> None:
        self._matrix: SE3 = SE3()

    @property
    @abstractmethod
    def matrix(self) -> SE3:
        """
        Returns the transformation matrix (4x4 homogeneous matrix) associated
        with the transformation.

        Returns
        -------
        SE3
        """
        pass

    @classmethod
    @abstractmethod
    def from_matrix(cls, matrix: SE3) -> Transformation:
        """
        Creates a Transformation object from a 4x4 homogeneous matrix.
        """
        pass

    def inv(self) -> Transformation:
        """
        Returns the inverse of the transformation.
        """
        inv_matrix = self._matrix.inv()
        return self.from_matrix(inv_matrix)


class Translation(Transformation):
    """
    Represents the operation of translating a frame.

    Attributes
    ----------
    vector: Vector
        Cartesian translation vector.
    """
    def __init__(self, vector: Vector) -> None:
        """
        Creates a Translation object.

        Parameters
        ----------
        vector: Vector
            Cartesian translation vector.
        """
        super().__init__()
        self.vector = vector
        # noinspection PyArgumentList
        self._matrix = SE3.Trans(np.asarray(vector, dtype=float))

    @classmethod
    def from_axis(cls, axis: Axis, distance: float) -> Translation:
        """
        Creates a Translation object defined by an Axis object and a distance.
        """
        return cls(Vector(distance * axis))

    @classmethod
    def from_matrix(cls, matrix: SE3) -> Translation:
        return cls(Vector(matrix.t))

    @property
    def matrix(self) -> SE3:
        """
        Returns the transformation matrix (4x4 homogeneous matrix) associated
        with the translation.

        Returns
        -------
        SE3
        """
        return self._matrix

    @property
    def distance(self) -> float:
        return float(np.sqrt(np.square(self.vector).sum()))


class Rotation(Transformation):
    """
    Represents the operation of rotating a frame.

    Attributes
    ----------
    vector: Vector
        Combines the rotation angle and the rotation axis in a single
        vector. The magnitude of this vector represents the rotation angle,
        and its axis or direction represents the rotation axis.
    angle: float
        Rotation angle.
    axis: Axis
        Rotation axis.
    pole: Vector, optional
        Point on the rotation axis, i.e. the center of rotation. If None,
        the rotation axis is taken through the origin of the reference
        frame.
    angle_unit: AngleUnits, default = "deg"
        Units of the rotation angle.
    """
    def __init__(
        self,
        vector: Vector,
        pole: Vector | None = None,
        angle_unit: AngleUnit = "rad"
    ) -> None:
        """
        Creates a Rotation object associated with a rotation about a given
        rotation axis by a given rotation angle.

        Parameters
        ----------
        vector: Vector
            Combines the rotation angle and the rotation axis in a single
            vector. The magnitude of this vector represents the rotation angle,
            and its axis or direction represents the rotation axis.
        pole: Vector, optional
            Point on the rotation axis, i.e. the center of rotation. If None,
            the rotation axis is taken through the origin of the reference
            frame.
        """
        super().__init__()
        self.vector = vector
        self.angle = vector.magnitude
        self.axis = vector.axis
        self.pole = pole
        self.angle_unit = angle_unit

        self._matrix = self._create_matrix()

    def _create_matrix(self) -> SE3:
        R_mat = SE3.AngleAxis(self.angle, self.axis.direction, unit=self.angle_unit)
        if self.pole is not None and not self.pole.isnull():
            # noinspection PyArgumentList
            T_mat = SE3.Trans(np.asarray(self.pole, dtype=float))
            return T_mat * R_mat * T_mat.inv()
        return R_mat

    @classmethod
    def about_axis(
        cls,
        axis: Axis,
        angle: float,
        pole: Vector | None = None,
        angle_unit: AngleUnit = "rad"
    ) -> Rotation:
        """
        Creates a Rotation object associated with a rotation about a given
        rotation axis by a given rotation angle.

        Parameters
        ----------
        axis: Axis
            Rotation axis.
        angle: float
            Rotation angle.
        pole: Vector, optional
            Point on the rotation axis, i.e. the center of rotation. If None,
            the rotation axis is taken through the origin of the reference frame.
        angle_unit: AngleUnits, default = "deg"
            Units of the rotation angle.
        """
        return cls(Vector(angle * axis), pole, angle_unit)

    @classmethod
    def from_matrix(cls, matrix: SE3, angle_unit: AngleUnit = "rad") -> Rotation:
        theta, axis = matrix.angvec(angle_unit)
        vector = Vector.from_axis(Axis(axis), theta)
        pole = Vector(matrix.t)
        return cls(vector, pole, angle_unit=angle_unit)

    @property
    def matrix(self) -> SE3:
        """
        Returns the transformation matrix (4x4 homogeneous matrix) associated
        with the rotation.

        Returns
        -------
        SE3
        """
        return self._matrix


class Screw(Transformation):
    """
    Represents a screw motion.

    Generally, a screw motion combines rotation and translation along a screw
    axis. However, a screw motion can also describe a pure rotation or a pure
    translation depending on the initialization parameters.

    Attributes
    ----------
    axis : Axis
       Axis of the screw motion.
    pole : Vector | None
        Point on the screw axis, i.e. the center of rotation. If None, the
        screw motion is purely translational (i.e. the screw axis is at
        infinity).
    magnitude : float
        The rotation angle of the screw motion, or the translation distance
        along the screw axis in case the screw motion is purely
        translational.
    pitch : float, optional
        The pitch of the screw axis. This has no meaning (can be None) in
        case the screw motion is purely rotational or purely translational.
    angle_unit : AngleUnits, optional, default = "deg"
        Units of the rotation angle. This has no meaning (can be None) in
        case the screw motion is purely translational.
    """
    def __init__(
        self,
        axis: Axis,
        pole: Vector | None,
        magnitude: float,
        pitch: float | None = None,
        angle_unit: AngleUnit = "rad"
    ) -> None:
        """
        Creates a Screw object.

        Parameters
        ----------
        axis : Axis
           Axis of the screw motion.
        pole : Vector | None
            Point on the screw axis, i.e. the center of rotation. If None, the
            screw motion is purely translational (i.e. the screw axis is at
            infinity).
        magnitude : float
            The rotation angle of the screw motion, or the translation distance
            along the screw axis in case the screw motion is purely
            translational.
        pitch : float, optional
            The pitch of the screw axis, i.e., the translation distance per
            angle unit. The pitch has no meaning (and can be None) in case the
            screw motion is purely rotational or purely translational.
        angle_unit : AngleUnits, optional, default = "deg"
            Units of the rotation angle. This has no meaning (can be None) in
            case the screw motion is purely translational.
        """
        super().__init__()
        self.axis = axis
        self.pole = pole
        self.magnitude = magnitude

        if self.pole is None:
            self.pitch = None
            self.angle_unit = None
            self._unit_twist = Twist3.UnitPrismatic(self.axis.direction)
            self._matrix = self._unit_twist.exp(self.magnitude)
        else:
            self.pitch = pitch
            self.angle_unit = angle_unit
            self._unit_twist = Twist3.UnitRevolute(
                self.axis.direction,
                np.asarray(self.pole, dtype=float),
                self.pitch
            )
            self._matrix = self._unit_twist.exp(self.magnitude, self.angle_unit)

    @property
    def matrix(self) -> SE3:
        """
        Returns the transformation matrix (4x4 homogeneous matrix) associated
        with the screw motion.

        Returns
        -------
        SE3
        """
        return self._matrix

    @property
    def unit_twist(self) -> Twist3:
        """
        Returns the unit twist of the screw motion.

        Returns
        -------
        Twist3
        """
        return self._unit_twist

    @property
    def twist(self) -> NumpyArray:
        """
        Returns the total twist of the screw motion. This is the product of the
        magnitude of the screw motion and the unit twist of the screw motion.

        Returns
        -------
        NumpyArray
        """
        return self.magnitude * self._unit_twist.S

    @property
    def v(self) -> NumpyArray:
        """
        Returns the total moment (v) of the twist.

        Returns
        -------
        NumpyArray
        """
        return self.magnitude * self._unit_twist.v

    @property
    def omega(self) -> NumpyArray:
        """
        Returns the total omega of the twist.

        Returns
        -------
        NumpyArray
        """
        return self.magnitude * self._unit_twist.w

    def is_prismatic(self) -> bool:
        """
        Returns True if the screw motion is purely translational, else False.
        """
        return bool(self.pole is None)

    def is_revolute(self) -> bool:
        """
        Returns True if the screw motion is not purely translational, else False.
        """
        return bool(self.pole is not None)

    @classmethod
    def from_matrix(cls, matrix: SE3) -> Screw:
        """
        Creates a Screw object from the given transformation matrix (4x4
        homogeneous matrix).
        """
        S = Twist3(matrix)
        axis = Axis(S.w)
        pole = Vector(S.pole)
        magnitude = S.theta
        pitch = S.pitch
        angle_unit: AngleUnit = "rad"
        return cls(axis, pole, magnitude, pitch, angle_unit)


def incremental_rotation_exact(omega: AngularVelocity, dt: float) -> SE3:
    """
    Returns the transformation matrix (4x4 homogeneous matrix) associated with a
    small rotation at a given time instant t.

    Parameters
    ----------
    omega : AngularVelocity
        Angular velocity vector at time instant t.
    dt : float
        Sample time interval in seconds, which should be a small value (ideally
        an infinitesimal time step).

    Returns
    -------
    SE3
    """
    if omega.isnull():
        return SE3()
    else:
        return SE3.Rt(smb.trexp(omega.axis.skew(), omega.magnitude * dt))


def incremental_rotation_approx(omega: AngularVelocity, dt: float) -> SE3:
    """
    Returns the transformation matrix (4x4 homogeneous matrix) associated with a
    small rotation at a given time instant t. The small rotation is represented
    by an approximation which is faster to calculate than the exact
    representation.

    Parameters
    ----------
    omega : AngularVelocity
        Angular velocity vector at time instant t.
    dt : float
        Sample time interval in seconds, which should be a small value (ideally
        an infinitesimal time step).

    Returns
    -------
    SE3

    Notes
    -----
    Repeated application of the approximation will result in an improper
    rotation matrix. However, if the sample rate is high (dt is small), then the
    error can be largely corrected by normalization.
    """
    if omega.isnull():
        return SE3()
    else:
        return SE3.Rt(
            np.eye(3, dtype=np.floating) + omega.magnitude * dt * omega.axis.skew(),
            check=False
        )


def incremental_motion_approx(V: SpatialVelocity, dt: float) -> SE3:
    """
    Returns the transformation matrix (4x4 homogeneous matrix) associated with a
    small motion at a given time instant t.

    Parameters
    ----------
    V : SpatialVelocity
        Spatial velocity vector at time instant t.
    dt : float
        Sample time interval in seconds, which should be a small value (ideally
        an infinitesimal time step).

    Returns
    -------
    SE3

    Notes
    -----
    Repeated application of the approximation will result in an improper
    rotation matrix. However, if the sample rate is high (dt is small), then the
    error can be largely corrected by normalization.
    """
    delta: NumpyArray = V * dt
    T_delta = smb.delta2tr(delta)
    return SE3(T_delta, check=False)
