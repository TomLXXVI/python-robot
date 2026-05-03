"""
Point-to-point motion profiles based on a polynomial function s(t) connecting
a start position s_i at a given time moment t_i with an end position s_f at a
given later time moment t_f.
"""
import typing

from abc import ABC, abstractmethod

import numpy as np

from ....base.types import NumpyArray
from ._profile_abc import MotionProfile
from .polynomials import Polynomial, CubicPolynomial, QuinticPolynomial


__all__ = [
    "PolyMotionProfile",
    "CubicMotionProfile",
    "QuinticMotionProfile"
]


NUM_SAMPLES = 100


class PolyMotionProfile(MotionProfile, ABC):
    """
    Represents a motion profile based on a polynomial function s(t).
    """
    def __init__(
        self,
        s_i: float = 0.0,
        s_f: float = 0.0,
        v_i: float = 0.0,
        v_f: float = 0.0,
        a_i: float = 0.0,
        a_f: float = 0.0,
        dt_tot: float = 0.0,
        t_i: float = 0.0,
        n_samples: int = NUM_SAMPLES,
    ) -> None:
        """
        Creates a PolyMotionProfile object based on the specified boundary
        conditions or constraints.

        Parameters
        ----------
        s_i: float, default 0.0
            Start position of the motion.
        s_f: float, default 0.0
            End position of the motion.
        v_i: float, default 0.0
            Initial velocity of the motion at the startpoint s_i.
        v_f: float, default 0.0
            Final velocity of the motion at the endpoint s_f.
        a_i: float, default 0.0
            Initial acceleration of the motion at the startpoint s_i.
        a_f: float, default 0.0
            Final acceleration of the motion profile at the endpoint s_f.
        dt_tot: float, default 0.0
            Time duration of the motion (travel time).
        t_i: float, default 0.0
            Initial time or start time of the motion.
        n_samples: int, default NUM_SAMPLES
            Number of time moments for calculating the position, velocity and
            acceleration profile.
        """
        super().__init__()
        self.s_i = s_i
        self.s_f = s_f
        self.v_i = v_i
        self.v_f = v_f
        self.s_i = s_i
        self.a_i = a_i
        self.a_f = a_f
        self.dt_tot = dt_tot
        self.t_i = t_i

        self._t_arr = np.linspace(0, self.dt_tot, n_samples)
        self._polynomial = None
        self._calc_motion_profile()

    def __str__(self) -> str:
        d = {
            "s_i": self._polynomial.s0,
            "v_i": self._polynomial.v0,
            "a_i": self._polynomial.a0,
            "j_i": self._polynomial.j0,
            "s_f": self._polynomial.sT,
            "v_f": self._polynomial.vT,
            "a_f": self._polynomial.aT,
            "j_f": self._polynomial.jT,
            "dt_tot": self._polynomial.T,
        }
        lines = "\n".join([f"{k}: {v:.6g}" for k, v in d.items()])
        return f"{self.__class__.__name__}:\n{lines}\n"

    @property
    def polynomial(self) -> Polynomial:
        """
        Returns the underlying polynomial that defines the polynomial motion
        profile.
        """
        return self._polynomial

    @property
    def t_f(self) -> float:
        """Returns the end time of the motion profile."""
        return self.t_i + self.dt_tot

    @abstractmethod
    def _calc_motion_profile(self) -> None:
        pass

    def position(self, t: float | NumpyArray) -> float | NumpyArray:
        """
        Returns the position of the motion profile at time t. Time t should be
        between the start time t_i and the end time t_f of the motion profile.
        If time t is less than t_i, the position at t_i is returned. If time t
        is greater than t_f, the position at t_f is returned.
        """
        tau = min(max(t - self.t_i, 0.0), self.dt_tot)
        return self._polynomial.position(tau)

    def velocity(self, t: float | NumpyArray) -> float | NumpyArray:
        """
        Returns the velocity of the motion profile at time t. Time t should be
        between the start time t_i and the end time t_f of the motion profile.
        If time t is less than t_i, the velocity at t_i is returned. If time t
        is greater than t_f, the velocity at t_f is returned.
        """
        tau = min(max(t - self.t_i, 0.0), self.dt_tot)
        return self._polynomial.velocity(tau)

    def acceleration(self, t: float | NumpyArray) -> float | NumpyArray:
        """
        Returns the acceleration of the motion profile at time t. Time t should
        be between the start time t_i and the end time t_f of the motion
        profile. If time t is less than t_i, the acceleration at t_i is
        returned. If time t is greater than t_f, the acceleration at t_f is
        returned.
        """
        tau = min(max(t - self.t_i, 0.0), self.dt_tot)
        return self._polynomial.acceleration(tau)

    def position_profile(self) -> tuple[NumpyArray, NumpyArray]:
        """
        Calculates the position profile of the motion.

        Returns
        -------
        t_arr :
            Numpy array with values on the time-axis.
        s_arr:
            Numpy array with the corresponding values on the position-axis.
        """
        s_arr = self._polynomial.position(self._t_arr)
        t_arr: NumpyArray = typing.cast(NumpyArray, self.t_i + self._t_arr)
        return t_arr, s_arr

    def velocity_profile(self) -> tuple[NumpyArray, NumpyArray]:
        """
        Calculates the velocity profile of the motion.

        Returns
        -------
        t_arr :
            Numpy array with values on the time-axis.
        v_arr:
            Numpy array with the corresponding values on the velocity-axis.
        """
        v_arr = self._polynomial.velocity(self._t_arr)
        t_arr: NumpyArray = typing.cast(NumpyArray, self.t_i + self._t_arr)
        return t_arr, v_arr

    def acceleration_profile(self) -> tuple[NumpyArray, NumpyArray]:
        """
        Calculates the acceleration profile of the motion.

        Returns
        -------
        t_arr :
            Numpy array with values on the time-axis.
        a_arr:
            Numpy array with the corresponding values on the acceleration-axis.
        """
        a_arr = self._polynomial.acceleration(self._t_arr)
        t_arr: NumpyArray = typing.cast(NumpyArray, self.t_i + self._t_arr)
        return t_arr, a_arr


class CubicMotionProfile(PolyMotionProfile):
    """
    Motion profile based on a cubic polynomial
    s(t) = c0 + c1.t + c2.t^2 + c3.t^3.
    """
    def __init__(
        self,
        s_i: float = 0.0,
        s_f: float = 0.0,
        v_i: float = 0.0,
        v_f: float = 0.0,
        dt_tot: float = 0.0,
        t_i: float = 0.0,
        n_samples: int = NUM_SAMPLES,
    ) -> None:
        """
        Creates a CubicMotionProfile object based on the specified boundary
        conditions or constraints.

        Parameters
        ----------
        s_i: float, default 0.0
            Start position of the motion.
        s_f: float, default 0.0
            End position of the motion.
        v_i: float, default 0.0
            Initial velocity of the motion at the startpoint s_i.
        v_f: float, default 0.0
            Final velocity of the motion at the endpoint s_f.
        dt_tot: float, default 0.0
            Time duration of the motion (travel time).
        t_i: float, default 0.0
            Initial time or start time of the motion.
        n_samples: int, default NUM_SAMPLES
            Number of time moments for calculating the position, velocity and
            acceleration profile.
        """
        super().__init__(s_i, s_f, v_i, v_f, 0.0, 0.0, dt_tot, t_i, n_samples)

    def _calc_motion_profile(self) -> None:
        self._polynomial = CubicPolynomial.from_constraints(
            T=self.dt_tot,
            s0=self.s_i,
            sT=self.s_f,
            v0=self.v_i,
            vT=self.v_f,
        )


class QuinticMotionProfile(PolyMotionProfile):
    """
    Motion profile based on a quintic polynomial
    s(t) = c0 + c1.t + c2.t^2 + c3.t^3 + c4.t^4 + c5.t^5.
    """
    def _calc_motion_profile(self) -> None:
        self._polynomial = QuinticPolynomial.from_constraints(
            T=self.dt_tot,
            s0=self.s_i,
            sT=self.s_f,
            v0=self.v_i,
            vT=self.v_f,
            a0=self.a_i,
            aT=self.a_f,
        )
