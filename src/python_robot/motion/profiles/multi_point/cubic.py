"""
Multipoint motion path of which the path points are interconnected by cubic
polynomials: class MultiPointCubicPath.

Path planning involves the calculation of velocities at the via points of the
path so that the final acceleration on the previous segment is equal to the
initial acceleration on the next segment. After path planning, the path is
generated of which the result is an ordered list of CubicMotionProfile objects,
representing the cubic polynomials that connect the path points.
"""
from typing import Sequence

import numpy as np

from ..point_to_point import CubicPolynomial, CubicMotionProfile
from ._profile_abc import MultiPointMotionProfile


__all__ = ["MultiPointCubicPath"]


class MultiPointCubicPath(MultiPointMotionProfile):
    """
    Represents a multipoint motion path of which the path points are
    interconnected by cubic polynomials.

    Attributes
    ----------
    segments: list[CubicMotionProfile]
        List of the segments that make up the path. Each segment is defined by
        a CubicMotionProfile object (see motion.point_to_point.poly_profiles).
    """
    def __init__(
        self,
        path_points: Sequence[float],
        dt_segments: Sequence[float],
        v_start: float = 0.0,
        v_end: float = 0.0,
    ) -> None:
        """
        Creates a MultiPointCubicPath object.

        Parameters
        ----------
        path_points: Sequence[float]
            Sequence with the positions of the path points that make up the path.
        dt_segments: Sequence[float]
            Sequence with the travel durations of each path segment.
        v_start: float, default = 0.0
            Initial velocity at the start of the path.
        v_end: float, default = 0.0
            Final velocity at the end of the path.
        """
        super().__init__()

        if len(path_points) < 2:
            raise ValueError("At least two points are required.")

        if len(dt_segments) != len(path_points) - 1:
            raise ValueError(
                f"Number of durations ({len(dt_segments)}) does not match "
                f"the number of segments ({len(path_points) - 1})."
            )

        if any(h <= 0.0 for h in dt_segments):
            raise ValueError("All segment durations must be strictly positive.")

        self.points = np.asarray(path_points, dtype=float)
        self.durations = np.asarray(dt_segments, dtype=float)
        self.v_start = float(v_start)
        self.v_end = float(v_end)

        self.n_segments = len(dt_segments)
        self.dt_tot = sum(dt_segments)  # total path time

        self.knot_velocities = self._solve_knot_velocities()
        self.segments = self._build_segments()

    def _solve_knot_velocities(self) -> np.ndarray:
        """
        Solve the internal knot velocities from acceleration continuity.
        """
        n = self.n_segments

        velocities = np.zeros(n + 1)
        velocities[0] = self.v_start
        velocities[-1] = self.v_end

        if n == 1:
            return velocities

        # Unknowns are v_1, ..., v_{n-1}
        A = np.zeros((n - 1, n - 1))
        b = np.zeros(n - 1)

        for i in range(1, n):
            row = i - 1

            h_prev = self.durations[i - 1]
            h_next = self.durations[i]

            ds_prev = self.points[i] - self.points[i - 1]
            ds_next = self.points[i + 1] - self.points[i]

            # Right-hand side
            b[row] = (
                6.0 * ds_prev / h_prev**2
                + 6.0 * ds_next / h_next**2
            )

            # Coefficient of v_{i-1}
            coeff_prev = 2.0 / h_prev

            # Coefficient of v_i
            coeff_curr = 4.0 / h_prev + 4.0 / h_next

            # Coefficient of v_{i+1}
            coeff_next = 2.0 / h_next

            # v_i is unknown
            A[row, row] = coeff_curr

            if i - 1 == 0:
                # v_0 is known, move to right-hand side
                b[row] -= coeff_prev * self.v_start
            else:
                A[row, row - 1] = coeff_prev

            if i + 1 == n:
                # v_n is known, move to right-hand side
                b[row] -= coeff_next * self.v_end
            else:
                A[row, row + 1] = coeff_next

        internal_velocities = np.linalg.solve(A, b)
        velocities[1:n] = internal_velocities

        return velocities

    def _build_segments(self) -> list[CubicMotionProfile]:
        """
        Build all cubic segments once the knot velocities are known.
        """
        segments = []
        t_i = 0.0
        for i in range(self.n_segments):
            s0 = self.points[i]
            sf = self.points[i + 1]
            h = self.durations[i]

            v0 = self.knot_velocities[i]
            vf = self.knot_velocities[i + 1]

            ds = sf - s0

            c0 = s0
            c1 = v0
            c2 = 3.0 * ds / h**2 - (2.0 * v0 + vf) / h
            c3 = -2.0 * ds / h**3 + (v0 + vf) / h**2

            cubic_poly = CubicPolynomial(T=h, c0=c0, c1=c1, c2=c2, c3=c3)
            cubic_mopf = CubicMotionProfile(
                s_i=cubic_poly.s0,
                s_f=cubic_poly.sT,
                v_i=cubic_poly.v0,
                v_f=cubic_poly.vT,
                dt_tot=cubic_poly.T,
                t_i=t_i
            )
            segments.append(cubic_mopf)
            t_i += h
        return segments

    def _locate_piece(self, t: float) -> CubicMotionProfile:
        t = min(max(t, 0.0), self.dt_tot)
        for seg in self.segments:
            if t <= seg.t_f:
                return seg
        return self.segments[-1]

    def position(self, t: float) -> float:
        seg = self._locate_piece(t)
        return float(seg.position(t))

    def velocity(self, t: float) -> float:
        seg = self._locate_piece(t)
        return float(seg.velocity(t))

    def acceleration(self, t: float) -> float:
        seg = self._locate_piece(t)
        return float(seg.acceleration(t))
