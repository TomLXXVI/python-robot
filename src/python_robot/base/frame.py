from __future__ import annotations
from typing import Literal, cast

import numpy as np
from spatialmath import SO3, SE3

from .types import *
from .transformation import Transformation, AngleUnit
from .vector import Vector, Axis, AngularVelocity, TranslationalVelocity, SpatialVelocity, Wrench

__all__ = [
    "FrameReference",
    "Frame",
    "WREF_FRAME",
    "orientation_rate_of_change",
    "pose_rate_of_change",
]


FrameReference = Literal["parent", "this"]


class Frame:
    """
    Represents an orthogonal, right-handed, 3-axis coordinate system for
    describing the position and orientation (the pose) of a rigid body in 3D
    space.

    A frame is not a fixed or absolute property of the rigid body. In fact, it
    is a (mathematical) **description** of an observer who observes the position
    and orientation of the rigid body from his own frame of reference.
    """
    def __init__(
        self,
        origin: ArrayLike3,
        orient_angles: ArrayLike3,
        angle_unit: AngleUnit = "rad"
    ) -> None:
        """
        Creates a Frame object.

        Parameters
        ----------
        origin : ArrayLike3
            Origin (x, y, z) of the frame in space as seen from its reference
            frame.
        orient_angles : ArrayLike3
            Defines the orientation of the frame in space by three angles. The
            first angle is the rotation angle about the X-axis, the second angle
            about the Y-axis, and the third angle about the Z-axis. The sign of
            the angles is determined by the right-hand rule.
        angle_unit : Literal["deg", "rad"], default = "rad"
            Unit of the orientation angles.
        """
        self.origin = origin
        self.orient_angles = orient_angles
        self.angle_unit = angle_unit

        self._origin_mat = SE3.Trans(*self.origin)
        self._orient_mat = SO3.RPY(*self.orient_angles, unit=self.angle_unit, order="zyx")
        self._matrix = SE3.Rt(self._orient_mat, self.origin)

    def __str__(self) -> str:
        return (
            f"Frame("
            f"[x: {float(self.origin[0]):.4g}, "
            f"y: {float(self.origin[1]):.4g}, "
            f"z: {float(self.origin[2]):.4g}], "
            f"[alpha: {float(self.orient_angles[0]):.4g} {self.angle_unit}, "
            f"beta: {float(self.orient_angles[1]):.4g} {self.angle_unit}, "
            f"gamma: {float(self.orient_angles[2]):.4g} {self.angle_unit}])"
        )

    @property
    def origin_mat(self) -> SE3:
        """
        Description in SE3-matrix form (4x4 homogeneous matrix) of the frame's
        origin as seen from its reference frame.

        Returns
        -------
        SE3
        """
        return self._origin_mat

    @property
    def orient_mat(self) -> SO3:
        """
        Description in SO3-matrix form (3x3 orthonormal matrix) of the frame's
        orientation as seen from its reference frame.

        Returns
        -------
        SO3
        """
        return self._orient_mat

    @property
    def matrix(self) -> SE3:
        """
        Description in SE3-matrix form (4x4 homogeneous matrix) of the frame's
        origin and orientation as seen from its reference frame.

        Returns
        -------
        SE3
        """
        return self._matrix

    @classmethod
    def from_matrix(cls, matrix: SE3, angle_unit: AngleUnit = "rad") -> Frame:
        origin = matrix.t
        orient_mat = SO3(matrix.R)
        orient_angles = orient_mat.rpy(angle_unit, order="zyx")
        return cls(origin, orient_angles, angle_unit)

    def to_pose_vector(self) -> NumpyArray:
        """
        Returns a 6x1 vector representation of the frame [x, y, z, rx, ry, rz].
        The first 3 elements are the position of the frame's origin. The last 3
        elements are the Cartesian components of the angle-axis vector
        describing the frame's orientation.

        Returns
        -------
        NumpyArray
        """
        position = np.asarray(self.origin, dtype=float)
        theta, axis = self.orient_mat.angvec()
        rotvec = theta * axis
        return np.concatenate((position, rotvec))

    @classmethod
    def from_pose_vector(cls, pose_vector: ArrayLike6) -> Frame:
        """
        Given a pose vector [x, y, z, rx, ry, rz], returns the corresponding
        Frame object.
        """
        pose_vector_ = np.asarray(pose_vector, dtype=float)

        position = pose_vector_[:3]
        rotvec = pose_vector_[3:]
        theta = np.linalg.norm(rotvec)

        if np.isclose(theta, 0.0):
            R = SO3()
        else:
            axis = rotvec / theta
            R = SO3.AngleAxis(float(theta), axis, unit="rad")

        T = SE3.Rt(R, position)
        return cls.from_matrix(T)

    def __mul__(self, other: Transformation | Frame) -> Frame:
        """
        Frame transformation via post-multiplication. The transformation of this
        frame is applied w.r.t. itself.
        """
        return self.from_matrix(self.matrix * other.matrix)

    def __rmul__(self, other: Transformation | Frame) -> Frame:
        """
        Frame transformation via pre-multiplication. The transformation of this
        frame is applied w.r.t. its reference frame.
        """
        return self.from_matrix(other.matrix * self.matrix)

    def __invert__(self) -> Frame:
        """
        Frame inversion. Returns the description of the reference frame of
        this frame as it is seen from this frame.
        """
        return self.from_matrix(self.matrix.inv())

    @property
    def x_axis(self) -> Axis:
        """
        Returns the x-axis of the frame as seen from its reference frame.
        """
        return Axis(self._orient_mat.A[:, 0])

    @property
    def y_axis(self) -> Axis:
        """
        Returns the y-axis of the frame as seen from its reference frame.
        """
        return Axis(self._orient_mat.A[:, 1])

    @property
    def z_axis(self) -> Axis:
        """
        Returns the z-axis of the frame as seen from its reference frame.
        """
        return Axis(self._orient_mat.A[:, 2])

    def _transform_angular(self, omega: AngularVelocity) -> AngularVelocity:
        """
        Transforms the angular velocity omega as observed in this frame to the
        angular velocity as it is observed from the reference frame of this
        frame.

        Returns
        -------
        AngularVelocity
        """
        omega_ = omega.array()
        omega_ = np.asarray(self.matrix.A[:3, :3]) @ omega_[np.newaxis, :].T
        return type(omega)(omega_.flatten())

    def _transform_spatial(self, V: SpatialVelocity, is_frame: bool) -> SpatialVelocity:
        """
        Transforms the spatial velocity V as observed in this frame to the
        spatial velocity as it is observed from the reference frame of this
        frame.

        Parameters
        ----------
        V : SpatialVelocity
            The spatial velocity as observed in this frame.
        is_frame : bool
            Indicates whether the spatial velocity V is the space velocity of
            this frame or the space velocity of a moving object as it is
            observed in this frame.

        Returns
        -------
        SpatialVelocity
        """
        V_ = V.array()
        if not is_frame:
            V_other = self.jacobian() @ V_[np.newaxis, :].T
            return type(V)(coords=V_other.flatten())
        else:
            V_other = self.adjoint() @ V_[np.newaxis, :].T
            return type(V)(coords=V_other.flatten())

    def _transform_wrench(self, W: Wrench) -> Wrench:
        """
        Transforms the wrench W as observed in this frame to the wrench as it
        is observed from the reference frame of this frame.

        Parameters
        ----------
        W : Wrench
            The wrench as observed in this frame.

        Returns
        -------
        Wrench
        """
        W_ = W.array()
        Ad_T = np.transpose(self.adjoint())
        W_other = Ad_T @ W_[np.newaxis, :].T
        return type(W)(coords=W_other.flatten())

    def _transform_position(self, v: Vector) -> Vector:
        arr1 = self._transform_angular(cast(AngularVelocity, v)).array()
        arr2 = np.asarray(self.origin)
        arr = arr1 + arr2
        return type(v)(arr)

    def transform(
        self,
        v: Vector | AngularVelocity | SpatialVelocity | Wrench,
        is_frame: bool | None = None
    ) -> Vector | AngularVelocity | SpatialVelocity | Wrench:
        """
        Transforms the vector v as observed in this frame to the vector as it is
        observed from the reference frame of this frame.

        Parameters
        ----------
        v : Vector | AngularVelocity | SpatialVelocity | Wrench
            The vector as observed in this frame. This vector can be either an
            AngularVelocity object, a SpatialVelocity object, a Wrench object or
            a Vector object.
        is_frame : bool | None
            Only has meaning when v is a SpatialVelocity object.
            Indicates whether the spatial velocity is the space velocity of this
            frame itself (in that case this frame must be fixed) or the space
            velocity of a moving object as it is observed in this frame.
        """
        if isinstance(v, AngularVelocity):
            return self._transform_angular(v)
        elif isinstance(v, SpatialVelocity) and is_frame is not None:
            return self._transform_spatial(v, is_frame)
        elif isinstance(v, SpatialVelocity) and is_frame is None:
            raise TypeError(
                "An object of type 'SpatialVelocity' requires parameter "
                "'is_frame' either to be True or False."
            )
        elif isinstance(v, Wrench):
            return self._transform_wrench(v)
        elif isinstance(v, Vector):
            return self._transform_position(v)
        else:
            raise TypeError(f"An object of type {type(v)} is not supported.")

    def jacobian(self) -> NumpyArray:
        """
        Returns the Jacobian of the frame as seen from its reference frame.

        The Jacobian of frame B w.r.t. frame A:
        ^AJ_B = [
            [ ^AR_B     0(3x3)],
            [ 0(3x3)    ^AR_B]
        ]
        ^AR_B is the orientation matrix describing the orientation of frame B
        as seen from its reference frame A.

        Returns
        -------
        NumpyArray
        """
        return self.matrix.jacob()

    def adjoint(self) -> NumpyArray:
        """
        Returns the adjoint of the frame as seen from its reference frame.

        The adjoint of frame B w.r.t. frame A:
        ^AAd_B = [
            [ ^AR_B     [^At_B]_x * ^AR_B],
            [ 0(3x3)    ^AR_B            ]
        ]
        ^AR_B is the orientation matrix describing the orientation of frame B as
        seen from its reference frame A. [^At_B]_x is the skew-symmetric matrix
        of the translation vector ^At_B, the translation vector between the
        origin of the reference frame A and the origin of frame B.

        Returns
        -------
        NumpyArray
        """
        return self.matrix.Ad()


# The "World Reference Frame" can be any frame that is used as a reference frame
# for one or more other frames.
WREF_FRAME = Frame(origin=(0, 0, 0), orient_angles=(0, 0, 0), angle_unit="deg")


def orientation_rate_of_change(
    frame: Frame,
    omega: AngularVelocity,
    ref: FrameReference = "parent"
) -> NumpyArray:
    """
    Returns the rate of change of the orientation of the given frame at a
    certain instant of time t.

    Parameters
    ----------
    frame : Frame
        Frame pose at the instant of time t.
    omega : AngularVelocity
        Angular velocity vector at the instant of time t.
    ref : FrameReference
        Indicates the frame where the angular velocity is measured. The angular
        velocity vector can be measured either in the reference frame (parent
        frame) of the given frame or in the given frame itself.

    Returns
    -------
    NumpyArray
    """
    if ref == "parent":
        # noinspection PyUnresolvedReferences
        return omega.skew() @ frame.orient_mat.A
    elif ref == "this":
        # noinspection PyUnresolvedReferences
        return frame.orient_mat.A @ omega.skew()
    else:
        raise ValueError("Unknown frame reference. Should be 'parent' or 'this'.")


def pose_rate_of_change(
    frame: Frame,
    omega: AngularVelocity,
    v: TranslationalVelocity,
) -> NumpyArray:
    """
    Returns the rate of change of the pose of the given frame at a certain
    instant of time t.

    Parameters
    ----------
    frame : Frame
        Frame pose (position of origin and orientation) at the instant of time t.
    omega : AngularVelocity
        Angular velocity vector at the instant of time t measured in the
        reference frame (parent frame) of the given frame.
    v : TranslationalVelocity
        Linear velocity vector at the instant of time t of the frame's origin
        measured in the reference frame (parent frame) of the given frame.

    Returns
    -------
    NumpyArray
    """
    T_dot = np.zeros((4, 4))
    R_dot = orientation_rate_of_change(frame, omega, ref="parent")
    T_dot[0:3, 0:3] = R_dot
    T_dot[0:3, 3] = v.coords
    return T_dot
