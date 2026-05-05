"""
Multisegment motion path with parabolic blends: class MultiLinearSegmentPath.

Implementation of a motion path (1D kinematic time-path) having multiple linear
segments, smoothly interconnected by parabolic blends.

Classes PathSegment and PathPoint contain the formulas for preliminary planning
of the motion path. After path planning, the path is generated. The result of
the path generation is a list of ordered PathPiece objects, representing the
parabolic blends around path points and the linear sections that connect them.
"""
from __future__ import annotations
from typing import Sequence

from dataclasses import dataclass, field

import numpy as np

from ._profile_abc import MultiPointMotionProfile

__all__ = ["MultiLinearSegmentPath"]


@dataclass
class PathSegment:
    """
    Represents a path segment of the multisegment path.

    Parameters
    ----------
    i: int
        Index of the path segment.
    start_point: PathPoint
        Start point of the path segment.
    end_point: PathPoint
        End point of the path segment.
    dt: float
        Time duration of the path segment.
    """
    i: int
    start_point: PathPoint
    end_point: PathPoint
    dt: float

    def is_start(self) -> bool:
        """
        Returns True if the path segment is the first segment of the multisegment
        path.
        """
        return self.start_point.is_start()

    def is_end(self) -> bool:
        """
        Returns True if the path segment is the last segment of the multisegment
        path.
        """
        return self.end_point.is_end()

    @property
    def v_lin(self) -> float:
        """Velocity in the linear section of the path segment."""
        s1 = self.start_point.s
        s2 = self.end_point.s
        if self.is_start():
            dt_b = self.start_point.dt_b
            v = (s2 - s1) / (self.dt - dt_b / 2)
            return v
        if self.is_end():
            dt_b = self.end_point.dt_b
            v = (s2 - s1) / (self.dt - dt_b / 2)
            return v
        # interior segment
        v = (s2 - s1) / self.dt
        return v

    @property
    def dt_lin(self) -> float:
        """Time duration of the linear section of the path segment."""
        dt_b1 = self.start_point.dt_b
        dt_b2 = self.end_point.dt_b
        if self.is_start():
            return self.dt - dt_b1 - dt_b2 / 2
        if self.is_end():
            return self.dt - dt_b2 - dt_b1 / 2
        # interior segment
        return self.dt - 0.5 * (dt_b1 + dt_b2)


@dataclass
class PathPoint:
    """
    Represents a path point of a multisegment path.

    Parameters
    ----------
    s: float
        Position of the path point.
    a_abs: float
        Magnitude of the acceleration for making the parabolic blend at the
        path point.
    """
    s: float
    a_abs: float

    segment_in: PathSegment | None = field(init=False)
    segment_out: PathSegment | None = field(init=False)

    def is_interior(self) -> bool:
        """
        Returns True if the path point is an interior point of the multisegment
        path.
        """
        return all([self.segment_in, self.segment_out])

    def is_start(self) -> bool:
        """
        Returns True if the path point is the startpoint of the multisegment
        path.
        """
        return self.segment_in is None and self.segment_out is not None

    def is_end(self) -> bool:
        """
        Returns True if the path point is the endpoint of the multisegment
        path.
        """
        return self.segment_in is not None and self.segment_out is None

    @property
    def a(self) -> float:
        """
        Constant acceleration during the blend. The sign of the acceleration is
        determined based on the direction of the path (either ascending or
        descending) or the speed difference between the next and the previous
        segment if the path point is an interior point.
        """
        if self.is_start() and self.segment_out is not None:
            s1 = self.s
            s2 = self.segment_out.end_point.s
            a = np.sign(s2 - s1) * self.a_abs
            return a
        if self.is_end() and self.segment_in is not None:
            s1 = self.segment_in.start_point.s
            s2 = self.s
            a = np.sign(s1 - s2) * self.a_abs
            return a
        # interior path point
        v1 = self.segment_in.v_lin if self.segment_in else 0.0
        v2 = self.segment_out.v_lin if self.segment_out else 0.0
        a = np.sign(v2 - v1) * self.a_abs
        return a

    @property
    def dt_b(self) -> float:
        """
        Blend time, i.e. the time it takes to make the parabolic blend at the
        path point.
        """
        if self.is_start() and self.segment_out is not None:
            dt = self.segment_out.dt
            s1 = self.s
            s2 = self.segment_out.end_point.s
            a = self.a
            D = np.sqrt(dt**2 - 2 * (s2 - s1) / a)
            if D < 0.0 or np.isnan(D):
                a_min = 2 * (s2 - s1) / dt**2
                raise ValueError(
                    f"The start acceleration ({a:.6g}) is too small. "
                    f"Minimum acceleration is {a_min:.6g}."
                )
            dt_b = dt - D
            return dt_b
        if self.is_end() and self.segment_in is not None:
            dt = self.segment_in.dt
            s1 = self.segment_in.start_point.s
            s2 = self.s
            a = self.a
            D = np.sqrt(dt**2 + 2 * (s2 - s1) / a)
            if D < 0.0 or np.isnan(D):
                a_min = -2 * (s2 - s1) / dt ** 2
                raise ValueError(
                    f"The end acceleration ({a:.6g}) is too small. "
                    f"Minimum acceleration is {a_min:.6g}."
                )
            dt_b = dt - D
            return dt_b
        # interior path point
        v1 = self.segment_in.v_lin if self.segment_in else 0.0
        v2 = self.segment_out.v_lin if self.segment_out else 0.0
        dt_b = (v2 - v1) / self.a
        return dt_b


@dataclass
class PathPiece:
    """
    Represents a piece of the multisegment path: either a parabolic blend or
    a linear piece. Each piece has kinematic time-functions for position,
    velocity, and acceleration. In a parabolic blend acceleration is constant.
    In a linear piece acceleration is zero and velocity is constant.
    """
    t0: float
    dt: float
    s0: float
    v0: float
    a: float

    @property
    def t_f(self) -> float:
        return self.t0 + self.dt

    def position(self, t: float) -> float:
        tau = min(max(t - self.t0, 0.0), self.dt)
        return self.s0 + self.v0 * tau + 0.5 * self.a * tau**2

    def velocity(self, t: float) -> float:
        tau = min(max(t - self.t0, 0.0), self.dt)
        return self.v0 + self.a * tau

    # noinspection PyUnusedLocal
    def acceleration(self, t: float) -> float:
        return self.a


class MultiLinearSegmentPath(MultiPointMotionProfile):
    """
    Represents a multisegment path composed of linear segments interconnected
    by parabolic blends.

    Attributes
    ----------
    pieces: list[PathPiece]
        The pieces that make up the multisegment path. Pieces are parabolic
        blends around the path points and linear sections that connect them
        (see class PathPiece in this module).
    """
    def __init__(
        self,
        path_points: Sequence[float],
        dt_segments: Sequence[float],
        blend_accels: float | Sequence[float],
    ) -> None:
        """
        Creates a MultiSegmentPath object.

        Parameters
        ----------
        path_points: Sequence[float]
            Sequence of the positions of the path points.
        dt_segments: Sequence[float]
            Sequence of the durations of the path segments.
        blend_accels: float | Sequence[float]
            Single value or a sequence of the magnitudes of accelerations for
            parabolic blends at the path points. If a single value is provided,
            it is applied to all blends.
        """
        super().__init__()

        if len(path_points) < 2:
            raise ValueError("At least two points are required.")

        if len(dt_segments) != len(path_points) - 1:
            raise ValueError(
                f"Number of durations ({len(dt_segments)}) does not match "
                f"the number of segments ({len(path_points) - 1})."
            )

        if any(dt <= 0.0 for dt in dt_segments):
            raise ValueError("All durations must be strictly positive.")

        if not isinstance(blend_accels, float) and len(blend_accels) != len(path_points):
            raise ValueError(
                f"Number of accelerations ({len(blend_accels)}) does not match "
                f"the number of points ({len(path_points)})."
            )
        elif isinstance(blend_accels, float):
            self.a_abs_seq = [blend_accels] * len(path_points)
        else:
            self.a_abs_seq = blend_accels

        self.s_seq = path_points
        self.dt_seq = dt_segments
        self.dt_tot = sum(self.dt_seq)      # total path time
        self.n_segments = len(self.dt_seq)  # total number of path segments

        self.segments: list[PathSegment] = []
        self._create_segments()
        self.pieces: list[PathPiece] = []
        self._build_pieces()

    def _create_segment(
        self,
        i: int,
        s_tup: tuple[float, float],
        a_abs_tup: tuple[float, float],
        dt: float,
    ) -> None:
        """
        Creates a single PathSegment object.
        """
        start_point = PathPoint(s_tup[0], a_abs_tup[0])
        end_point = PathPoint(s_tup[-1], a_abs_tup[-1])
        segment = PathSegment(i, start_point, end_point, dt)
        if i == 0:
            # first segment of path
            segment.start_point.segment_in = None
            segment.start_point.segment_out = segment
            segment.end_point.segment_in = segment
            segment.end_point.segment_out = None  # we don't know yet
        elif i == self.n_segments - 1:
            # last segment of path
            self.segments[i - 1].end_point.segment_out = segment  # now we know
            segment.start_point.segment_in = self.segments[i - 1]
            segment.start_point.segment_out = segment
            segment.end_point.segment_in = segment
            segment.end_point.segment_out = None
        else:
            # interior segment of path
            self.segments[i - 1].end_point.segment_out = segment  # now we know
            segment.start_point.segment_in = self.segments[i - 1]
            segment.start_point.segment_out = segment
            segment.end_point.segment_in = segment
            segment.end_point.segment_out = None  # we don't know yet
        self.segments.append(segment)

    def _create_segments(self) -> None:
        """
        Creates the list of PathSegments that make up the MultiSegmentPath
        object.
        """
        s_tuples = list(zip(self.s_seq[:-1], self.s_seq[1:]))
        a_tuples = list(zip(self.a_abs_seq[:-1], self.a_abs_seq[1:]))
        for i in range(self.n_segments):
            self._create_segment(i, s_tuples[i], a_tuples[i], self.dt_seq[i])

    def _build_pieces(self) -> None:
        """
        Using the PathPoint objects and the PathSegment objects, the time
        functions s(t), v(t) and a(t) can be calculated in the parabolic blends
        and linear pieces of the multisegment path.
        """
        # Initial values at the start point of the multisegment path.
        t = 0.0
        s = self.segments[0].start_point.s
        v = 0.0

        def add_piece(dt: float, a: float) -> None:
            nonlocal t, s, v
            if dt <= 0.0:
                return
            piece = PathPiece(t0=t, dt=dt, s0=s, v0=v, a=a)
            self.pieces.append(piece)
            s = piece.position(piece.t_f)
            v = piece.velocity(piece.t_f)
            t = piece.t_f

        for i in range(self.n_segments):
            blend = self.segments[i].start_point
            segment = self.segments[i]
            add_piece(blend.dt_b, blend.a)  # blend
            add_piece(segment.dt_lin, 0.0)  # linear piece of segment
            if i == self.n_segments - 1:
                # end point of the multisegment path
                blend = self.segments[i].end_point
                add_piece(blend.dt_b, blend.a)

    def _locate_piece(self, t: float) -> PathPiece:
        t = min(max(t, 0.0), self.dt_tot)
        for piece in self.pieces:
            if t <= piece.t_f:
                return piece
        return self.pieces[-1]

    def position(self, t: float) -> float:
        piece = self._locate_piece(t)
        return piece.position(t)

    def velocity(self, t: float) -> float:
        piece = self._locate_piece(t)
        return piece.velocity(t)

    def acceleration(self, t: float) -> float:
        piece = self._locate_piece(t)
        return piece.acceleration(t)
