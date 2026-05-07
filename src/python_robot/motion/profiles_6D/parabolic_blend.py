from typing import Sequence

from dataclasses import dataclass

import numpy as np

from ...base.types import NumpyArray
from ...base import SpatialVelocity, SpatialAcceleration
from ._profile_abc import MultiPointVectorMotionProfile


__all__ = ["MultiLinearVectorPath"]


@dataclass
class VectorPathPiece:
    """
    Represents a piece of a 6-dimensional multisegment path.

    A path piece is either a parabolic blend or a linear piece. Each piece has
    vectorial kinematic time-functions for position, velocity, and
    acceleration. In a parabolic blend, acceleration is constant. In a linear
    piece, acceleration is zero and velocity is constant.
    """
    t0: float
    dt: float
    x0: NumpyArray
    v0: NumpyArray
    a: NumpyArray

    @property
    def t_f(self) -> float:
        return self.t0 + self.dt

    def position(self, t: float) -> NumpyArray:
        tau = min(max(t - self.t0, 0.0), self.dt)
        return self.x0 + self.v0 * tau + 0.5 * self.a * tau**2  # type: ignore

    def velocity(self, t: float) -> NumpyArray:
        tau = min(max(t - self.t0, 0.0), self.dt)
        return self.v0 + self.a * tau

    # noinspection PyUnusedLocal
    def acceleration(self, t: float) -> NumpyArray:
        return self.a


class MultiLinearVectorPath(MultiPointVectorMotionProfile):
    """
    Builds a multisegment path in time which is composed of linear segments
    interconnected by parabolic blends.

    This class is the vectorial counterpart of MultiLinearPath. It is intended
    for motion variables that are naturally represented as vectors, like pose
    vectors of the form

        [x, y, z, rx, ry, rz]

    where [rx, ry, rz] is an angle-axis vector, representing the orientation of
    a frame.

    The blend times are specified explicitly. This is useful when all vector
    components must share the same blend time, as is the case for Cartesian
    straight-line motion.

    Attributes
    ----------
    pose_vectors: NumpyArray
        Two-dimensional array with the pose vectors [x, y, z, rx, ry, rz]. Each
        row represents one pose vector.
    dt_segments: NumpyArray
        One-dimensional array with the travel durations of each segment.
    dt_blends: NumpyArray
        One-dimensional array with the blend time at each path point.
    segment_velocities: NumpyArray
        Two-dimensional array with the constant velocity vector of each linear
        segment.
    pieces: list[VectorPathPiece]
        List of the pieces that make up the vectorial multisegment path.
    """
    def __init__(
        self,
        pose_vectors: Sequence[Sequence[float]],
        dt_segments: Sequence[float],
        dt_blends: float | Sequence[float],
    ) -> None:
        """
        Creates a MultiLinearVectorPath object.

        Parameters
        ----------
        pose_vectors: Sequence[Sequence[float]]
            Sequence of vectorial path points of the form
            [x, y, z, rx, ry, rz]. Each path point must have the same number
            of components.
        dt_segments: Sequence[float]
            Sequence of the durations of the path segments.
        dt_blends: float | Sequence[float]
            Single value or a sequence of blend times. If a single value is
            provided, it is applied to all path points. If a sequence is
            provided, its length must match the number of path points.
        """
        super().__init__()

        _pose_vectors = np.asarray(pose_vectors, dtype=float)

        if _pose_vectors.ndim != 2:
            raise ValueError(
                "Path points must be a two-dimensional array-like object. "
                "Each row must be one vectorial path point."
            )

        if _pose_vectors.shape[0] < 2:
            raise ValueError("At least two path points are required.")

        if len(dt_segments) != _pose_vectors.shape[0] - 1:
            raise ValueError(
                f"Number of durations ({len(dt_segments)}) does not match "
                f"the number of segments ({_pose_vectors.shape[0] - 1})."
            )

        if any(dt <= 0.0 for dt in dt_segments):
            raise ValueError("All segment durations must be strictly positive.")

        if isinstance(dt_blends, float | int):
            blend_times = np.full(_pose_vectors.shape[0], float(dt_blends))
        else:
            if len(dt_blends) != _pose_vectors.shape[0]:
                raise ValueError(
                    f"Number of blend times ({len(dt_blends)}) does not match "
                    f"the number of path points ({_pose_vectors.shape[0]})."
                )
            blend_times = np.asarray(dt_blends, dtype=float)

        if np.any(blend_times < 0.0):
            raise ValueError("Blend times must be greater than or equal to zero.")

        self.pose_vectors = _pose_vectors
        self.dt_segments = np.asarray(dt_segments, dtype=float)
        self.dt_blends = blend_times

        self.n_points = self.pose_vectors.shape[0]
        self.n_segments = self.n_points - 1
        self.n_dim = self.pose_vectors.shape[1]
        self.dt_tot = float(np.sum(self.dt_segments))

        self.knot_times = self._calc_knot_times()
        self.segment_velocities = self._calc_segment_velocities()

        self._validate_blend_times()

        self.pieces: list[VectorPathPiece] = []
        self._build_pieces()

    def _calc_knot_times(self) -> NumpyArray:
        """
        Calculates the nominal time moments of the path points.
        """
        t_arr = np.zeros(self.n_points)
        t_arr[1:] = np.cumsum(self.dt_segments)
        return t_arr

    def _calc_segment_velocities(self) -> NumpyArray:
        """
        Calculates the constant velocity vector for each linear segment.

        The first and last segment are treated differently because the start
        and end blends are not centered around the start and end path points.
        """
        velocities = np.zeros((self.n_segments, self.n_dim))

        for i in range(self.n_segments):
            dx = self.pose_vectors[i + 1] - self.pose_vectors[i]
            dt = self.dt_segments[i]

            if self.n_segments == 1:
                dt_eff = dt - 0.5 * self.dt_blends[0] - 0.5 * self.dt_blends[-1]
            elif i == 0:
                dt_eff = dt - 0.5 * self.dt_blends[0]
            elif i == self.n_segments - 1:
                dt_eff = dt - 0.5 * self.dt_blends[-1]
            else:
                dt_eff = dt

            if dt_eff <= 0.0:
                raise ValueError(
                    f"Blend times are too large for segment {i}. "
                    f"The effective segment duration is {dt_eff:.6g}."
                )

            velocities[i] = dx / dt_eff

        return velocities

    def _validate_blend_times(self) -> None:
        """
        Checks whether the blend times leave a non-negative linear piece in
        every segment.
        """
        for i in range(self.n_segments):
            t0 = self._linear_start_time(i)
            tf = self._linear_end_time(i)

            if tf < t0:
                raise ValueError(
                    f"Blend times are too large for segment {i}. "
                    f"The resulting linear duration is {tf - t0:.6g}."
                )

    def _line_position(self, segment_index: int, t: float) -> NumpyArray:
        """
        Returns the position on the nominal linear function of a segment.

        For the first segment, the nominal linear function is shifted backward
        because the path starts with a blend from zero velocity. For all other
        segments, the nominal linear function passes through the path point at
        the start of the segment.
        """
        v = self.segment_velocities[segment_index]
        t_i = self.knot_times[segment_index]
        x_i = self.pose_vectors[segment_index]

        if segment_index == 0:
            x_i = x_i - 0.5 * v * self.dt_blends[0]

        return x_i + v * (t - t_i)

    def _linear_start_time(self, segment_index: int) -> float:
        """
        Returns the start time of the linear piece of a segment.
        """
        t_i = self.knot_times[segment_index]
        dt_b = self.dt_blends[segment_index]

        if segment_index == 0:
            return t_i + dt_b
        return t_i + 0.5 * dt_b

    def _linear_end_time(self, segment_index: int) -> float:
        """
        Returns the end time of the linear piece of a segment.
        """
        t_f = self.knot_times[segment_index + 1]
        dt_b = self.dt_blends[segment_index + 1]

        if segment_index == self.n_segments - 1:
            return t_f - dt_b
        return t_f - 0.5 * dt_b

    def _add_piece(
        self,
        t0: float,
        dt: float,
        x0: NumpyArray,
        v0: NumpyArray,
        a: NumpyArray,
    ) -> None:
        """
        Adds a path piece to the path if its duration is strictly positive.
        """
        if dt <= 0.0:
            return

        piece = VectorPathPiece(
            t0=t0,
            dt=dt,
            x0=x0,
            v0=v0,
            a=a
        )
        self.pieces.append(piece)

    def _build_start_blend(self) -> None:
        """
        Builds the blend at the start point of the path.
        """
        dt_b = self.dt_blends[0]
        if dt_b <= 0.0:
            return

        v_next = self.segment_velocities[0]
        a = v_next / dt_b

        self._add_piece(
            t0=0.0,
            dt=dt_b,
            x0=self.pose_vectors[0],
            v0=np.zeros(self.n_dim),
            a=a
        )

    def _build_interior_blend(self, point_index: int) -> None:
        """
        Builds the blend around an interior path point.
        """
        dt_b = self.dt_blends[point_index]
        if dt_b <= 0.0:
            return

        t0 = self.knot_times[point_index] - 0.5 * dt_b
        v_prev = self.segment_velocities[point_index - 1]
        v_next = self.segment_velocities[point_index]
        a = (v_next - v_prev) / dt_b
        x0 = self._line_position(point_index - 1, t0)

        self._add_piece(
            t0=t0,
            dt=dt_b,
            x0=x0,
            v0=v_prev,
            a=a
        )

    def _build_end_blend(self) -> None:
        """
        Builds the blend at the end point of the path.
        """
        dt_b = self.dt_blends[-1]
        if dt_b <= 0.0:
            return

        t0 = self.dt_tot - dt_b
        v_prev = self.segment_velocities[-1]
        a = -v_prev / dt_b
        x0 = self._line_position(self.n_segments - 1, t0)

        self._add_piece(
            t0=t0,
            dt=dt_b,
            x0=x0,
            v0=v_prev,
            a=a
        )

    def _build_linear_piece(self, segment_index: int) -> None:
        """
        Builds the linear piece of a segment.
        """
        t0 = self._linear_start_time(segment_index)
        tf = self._linear_end_time(segment_index)
        dt = tf - t0

        if dt <= 0.0:
            return

        v = self.segment_velocities[segment_index]
        x0 = self._line_position(segment_index, t0)

        self._add_piece(
            t0=t0,
            dt=dt,
            x0=x0,
            v0=v,
            a=np.zeros(self.n_dim)
        )

    def _build_pieces(self) -> None:
        """
        Builds all pieces of the vectorial multisegment path.
        """
        self._build_start_blend()

        for i in range(self.n_segments):
            self._build_linear_piece(i)

            if i < self.n_segments - 1:
                self._build_interior_blend(i + 1)

        self._build_end_blend()

        self.pieces.sort(key=lambda piece: piece.t0)

    def _locate_piece(self, t: float) -> VectorPathPiece:
        """
        Locates the path piece that corresponds to the given time moment.
        """
        t = min(max(t, 0.0), self.dt_tot)

        for piece in self.pieces:
            if t <= piece.t_f:
                return piece

        return self.pieces[-1]

    def position(self, t: float) -> NumpyArray:
        """
        Returns the pose vector p (x, y, z, rx, ry, rz) at time t.
        """
        piece = self._locate_piece(t)
        return piece.position(t)

    def velocity(self, t: float) -> NumpyArray:
        """
        Returns the pose vector velocity pd (x_dot, y_dot, z_dot, rx_dot,
        ry_dot, rz_dot) at time t.
        """
        piece = self._locate_piece(t)
        return piece.velocity(t)

    def acceleration(self, t: float) -> NumpyArray:
        """
        Returns the pose vector acceleration pdd (x_ddot, y_ddot, z_ddot,
        rx_ddot, ry_ddot, rz_ddot) at time t.
        """
        piece = self._locate_piece(t)
        return piece.acceleration(t)

    def spatial_velocity(self, t: float) -> NumpyArray:
        """
        Returns the spatial velocity V (v_x, v_y, v_z, omega_x, omega_y,
        omega_z) at time t.
        """
        piece = self._locate_piece(t)
        p = piece.position(t)
        pd = piece.velocity(t)
        V = SpatialVelocity.from_pose(p, pd)
        return V.array()

    def spatial_acceleration(self, t: float) -> NumpyArray:
        """
        Returns the spatial acceleration A (a_x, a_y, a_z, alpha_x, alpha_y,
        alpha_z) at time t.
        """
        piece = self._locate_piece(t)
        p = piece.position(t)
        pd = piece.velocity(t)
        pdd = piece.acceleration(t)
        A = SpatialAcceleration.from_pose(p, pd, pdd)
        return A.array()
