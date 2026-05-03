"""
Point-to-point motion profiles composed of three phases: acceleration,
constant velocity, and deceleration.

Following classes are available:
-   TrapezoidalProfile, representing a trapezoidal motion profile.
-   SCurvedProfile, representing an S-curved motion profile.
"""
from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

from ....base.types import NumpyArray
from ....utils.math import find_zero_crossings
from ._profile_abc import MotionProfile


__all__ = [
    "TriPhaseMotionProfile",
    "TrapezoidalProfile",
    "SCurvedProfile",
]


NUM_SAMPLES = 100


class TriPhaseMotionProfile(MotionProfile, ABC):
    """
    Represents a motion profile consisting of three phases: an acceleration
    phase, a constant-velocity phase, and a deceleration phase.
    """
    dt_acc_min: float = 0.1

    def __init__(
        self,
        ds_tot: float,
        a_max: float,
        v_max: float,
        v_i: float | None,
        v_f: float | None,
        s_i: float = 0.0,
        dt_tot: float | None = None
    ) -> None:
        """
        Creates a `MotionProfile` object.

        Parameters
        ----------
        ds_tot: float
            Total travel distance.
        a_max: float
            (Maximum) acceleration (of the motor). This value remains unaltered
            in the calculation of the motion profile (i.e. `self.a_top` is
            always equal to `a_max`).
        v_max: float
            (Maximum) speed limit (of the motor). `v_max` needs to be
            distinguished from the top velocity `self.v_top` of the motion
            profile. `v_max` poses a maximum limit when calculating `self.v_top`.
        v_i: float, optional
            Start velocity of the movement, if known.
        v_f: float, optional
            End velocity of the movement, if known.
        s_i: float, default=0.0
            Initial position of the axis or motor shaft.
        dt_tot: float, optional
            Total travel time. When total travel time is specified, a motion
            profile is calculated so that total travel distance `ds_tot` is
            covered in `dt_tot` (seconds).

        Units can be chosen freely, but must be consistent. Usually time is
        chosen to be expressed in seconds. The units of position (displacement),
        velocity, and acceleration have units of length in common. For example,
        if time is in seconds and position is in units of mm, then velocity must
        be in mm/s, and acceleration in mm/s².

        A motion profile can be calculated for two use cases:
        (1) Travel distance `ds_tot` is known, but travel time `dt_tot` is not
            (`dt_tot` is `None`).
            The top velocity `self.v_top` is calculated so that the total travel
            distance `ds_tot` is accomplished in the shortest possible travel
            time `dt_tot`, taking into account the maximum velocity `v_max`, the
            (maximum) acceleration `a_max`, and the initial and final boundary
            velocities `v_i` and `v_f`.
        (2) Travel distance `ds_tot` and travel time `dt_tot` are both known.
            The travel time `dt_tot` adds an extra constraint to the calculation
            of `self.v_top`.

        The top acceleration `self.a_top` used in the calculations is always
        equal to the (maximum) acceleration `a_max`.

        If the initial boundary velocity `v_i` is undetermined (`None`), there
        is no initial acceleration phase (no initial velocity change) and the
        initial velocity will be equal to the calculated top velocity `v_top`.
        The same also applies to the final boundary velocity `v_f`.

        If neither the initial, nor the final boundary velocity is determined
        (`v_i` and `v_f` are both `None`), the motion profile only has a
        constant-velocity phase where the velocity is taken to be equal to the
        value of `v_max`.
        """
        super().__init__()
        self.ds_tot = ds_tot
        self.a_max = a_max
        self.v_max = v_max
        self.v_i = v_i
        self.v_f = v_f
        self.s_i = s_i
        self.dt_tot = dt_tot

        self.v_top = 0.0
        self.dv_i = 0.0
        self.dv_f = 0.0
        self.a_top_i = 0.0
        self.a_top_f = 0.0
        self.dt_i = 0.0
        self.dt_f = 0.0
        self.ds_i = 0.0
        self.ds_f = 0.0
        self.ds_cov = 0.0
        self.dt_cov = 0.0

        self._calc_motion_profile()

        self._pos_fn = self.get_position_from_time_fn()
        self._vel_fn = self.get_velocity_from_time_fn()
        self._acc_fn = self.get_acceleration_from_time_fn()

    def __str__(self) -> str:
        d = {
            "s_i": self.s_i,
            "v_i": self.v_i,
            "ds_i": self.ds_i,
            "dv_i": self.dv_i,
            "dt_i": self.dt_i,
            "a_top_i": self.a_top_i,
            "ds_cov": self.ds_cov,
            "v_top": self.v_top,
            "dt_cov": self.dt_cov,
            "ds_f": self.ds_f,
            "dv_f": self.dv_f,
            "dt_f": self.dt_f,
            "a_top_f": self.a_top_f,
            "ds_tot": self.ds_tot,
            "dt_tot": self.dt_tot,
            "s_f": self.s_i + self.ds_tot,
            "v_f": self.v_f,
            "v_max": self.v_max,
            "a_max": self.a_max,
        }
        lines = "\n".join([f"{k}: {v:.6g}" for k, v in d.items()])
        return f"{self.__class__.__name__}:\n{lines}\n"

    @abstractmethod
    def get_dv_min(self) -> float:
        """
        Returns the smallest achievable velocity change with acceleration
        `self.a_max` where the acceleration time equals the minimum allowable
        acceleration time `self.dt_acc_min`.
        """
        ...

    @ abstractmethod
    def _calc_a_top(self, dv: float) -> tuple[float, float]:
        """
        Calculates the top acceleration of the motion and the accleration time.

        Parameters
        ----------
        dv :
            Required velocity change.

        Returns
        -------
        a_top :
            In case the acceleration time that follows from `dv` and `self.a_max`
            is smaller than the minimum permitted acceleration time
            `self.dt_acc_min`, the top acceleration `a_top` of the motion is
            calculated so that acceleration time would equal the minimum
            permitted acceleration time (`a_top` < `a_max`). Otherwise, `a_top`
            is kept equal to `self.a_max`.
        dt_acc :
            Resulting acceleration time.
        """
        ...

    @abstractmethod
    def _acceleration(
        self,
        dv: float,
        t0: float = 0.0,
        n_samples: int = NUM_SAMPLES
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Calculates the acceleration values at given time moments during
        acceleration of the motion.

        Parameters
        ----------
        dv :
            Required velocity change.
        t0:
            Initial time moment of the acceleration.
        n_samples:
            Number of time samples where the acceleration is to be calculated.

        Returns
        -------
        t_arr :
            Numpy array with the time values during acceleration of the motion.
        a_arr:
            Numpy array with the corresponding values of the acceleration.
        """
        ...

    def velocity_acc(
        self,
        dv: float,
        t0: float = 0.0,
        v0: float = 0.0
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Calculates the velocity values at given time moments during
        acceleration of the motion.

        Parameters
        ----------
        dv :
            Required velocity change.
        t0 :
            Initial time moment.
        v0 :
            Initial velocity.

        Returns
        -------
        t_arr :
            Numpy array with the time values during acceleration of the motion.
        v_arr:
            Numpy array with the corresponding values of the velocity.
        """
        t_arr, a_arr = self._acceleration(dv, t0)
        v_arr = cumulative_trapezoid(a_arr, t_arr, initial=0)
        v_arr += v0
        return t_arr, v_arr

    def position_acc(
        self,
        dv: float,
        t0: float = 0.0,
        s0: float = 0.0,
        v0: float = 0.0
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Calculates the position values at given time moments during
        acceleration of the motion.

        Parameters
        ----------
        dv :
            Required velocity change.
        t0 :
            Initial time moment.
        s0 :
            Initial position.
        v0 :
            Initial velocity.

        Returns
        -------
        t_arr :
            Numpy array with the time values during acceleration of the motion.
        s_arr:
            Numpy array with the corresponding values of the position.
        """
        t_arr, v_arr = self.velocity_acc(dv, t0, v0)
        s_arr = cumulative_trapezoid(v_arr, t_arr, initial=0)
        s_arr += s0
        return t_arr, s_arr

    def _calc_ds(self, dv: float, v0: float = 0.0):
        """
        Calculates the distance traveled during acceleration of the motion.

        Parameters
        ----------
        dv :
            Required velocity change.

        Returns
        -------
        ds :
            Distance traveled during acceleration of the motion.
        """
        _, s_arr = self.position_acc(dv, v0=v0)
        ds = abs(s_arr[-1] - s_arr[0])
        return ds

    def _calc_dv_i(self, v_top: float) -> float:
        dv = v_top - self.v_i if self.v_i is not None else 0.0  # assumes self.v_i = v_top
        return dv

    def _calc_dv_f(self, v_top: float) -> float:
        dv = self.v_f - v_top if self.v_f is not None else 0.0  # assumes self.v_f = v_top
        return dv

    def _calc_v_top_without_time_constraint(self) -> float:
        """
        Calculates `v_top` so that `ds_cov` = 0 (i.e. without constant-velocity
        phase).
        """
        def fn(v_top: float) -> float:
            dv_i = self._calc_dv_i(v_top)
            dv_f = self._calc_dv_f(v_top)
            ds_i = self._calc_ds(dv_i, v_top - dv_i)
            ds_f = self._calc_ds(dv_f, v_top)
            ds_cov = self.ds_tot - (ds_i + ds_f)
            return ds_cov

        v_top_ini = self.v_max
        if self.v_f is not None and self.v_i is not None:
            dv = self.v_f - self.v_i
            if dv > 0.0:
                v_top_ini = self.v_f
            if dv < 0.0:
                v_top_ini = self.v_i
        ds_cov = fn(v_top_ini)
        if ds_cov > 0.0:
            return v_top_ini

        try:
            sol = root_scalar(fn, bracket=(0.0, self.v_max))
            return sol.root
        except ValueError:
            raise ValueError("Top velocity cannot be determined.")

    def _calc_v_top_with_time_constraint(self) -> float:
        """
        Calculates `v_top` so that `ds_tot` is finished after `dt_tot` s when
        acceleration is `self.a_top`.
        """
        def fn(v_top: float) -> float:
            dv_i = self._calc_dv_i(v_top)
            dv_f = self._calc_dv_f(v_top)
            _, dt_i = self._calc_a_top(dv_i)
            _, dt_f = self._calc_a_top(dv_f)
            dt_cov = self.dt_tot - (dt_i + dt_f)  #type: ignore
            ds_i = self._calc_ds(dv_i, v_top - dv_i)
            ds_f = self._calc_ds(dv_f, v_top)
            ds_cov1 = self.ds_tot - (ds_i + ds_f)
            ds_cov2 = v_top * dt_cov
            dev = ds_cov2 - ds_cov1
            return dev

        def check(v_top: float) -> float | None:
            dv_i = self._calc_dv_i(v_top)
            dv_f = self._calc_dv_f(v_top)
            a_top_i, dt_i = self._calc_a_top(dv_i)
            a_top_f, dt_f = self._calc_a_top(dv_f)
            ds_i = self._calc_ds(dv_i, v_top - dv_i)
            ds_f = self._calc_ds(dv_f, v_top)
            ds_cov = self.ds_tot - (ds_i + ds_f)
            if ds_cov > 0.0:
                dt_cov = ds_cov / v_top
            else:
                dt_cov = 0.0
            dt_tot = dt_i + dt_cov + dt_f
            dev = abs(dt_tot - self.dt_tot)
            if dev < 1.e-6:
                return v_top
            return None

        v_top_arr = np.linspace(0.0, self.v_max, max(10, int(0.5 * self.v_max)))
        v_top_lst = find_zero_crossings(fn, v_top_arr)

        v_top_lst = [
            v_top
            for v_top in [check(v_top_) for v_top_ in v_top_lst]
            if v_top is not None
        ]
        try:
            return v_top_lst[0]
        except IndexError:
            raise ValueError("Top velocity cannot be determined.")

    def _calc_motion_profile(self) -> None:
        if self.ds_tot == 0.0 and self.dt_tot is not None:
            self.dt_cov = self.dt_tot
            return

        # noinspection PyUnreachableCode
        if self.ds_tot > 0.0 and self.dt_tot is None:
            self.v_top = self._calc_v_top_without_time_constraint()
        elif self.ds_tot > 0.0 and self.dt_tot is not None and self.dt_tot > 0.0:
            self.v_top = self._calc_v_top_with_time_constraint()
        else:
            raise ValueError("Motion profile cannot be determined.")

        self.dv_i = self._calc_dv_i(self.v_top)
        self.dv_f = self._calc_dv_f(self.v_top)
        self.a_top_i, self.dt_i = self._calc_a_top(self.dv_i)
        self.a_top_f, self.dt_f = self._calc_a_top(self.dv_f)
        self.ds_i = self._calc_ds(self.dv_i, self.v_top - self.dv_i)
        self.ds_f = self._calc_ds(self.dv_f, self.v_top)
        self.ds_cov = self.ds_tot - (self.ds_i + self.ds_f)
        if self.ds_cov > 0.0:
            self.dt_cov = self.ds_cov / self.v_top
        self.dt_tot = self.dt_i + self.dt_cov + self.dt_f

    # noinspection PyTypeChecker
    def velocity_profile(self, n_samples: int = NUM_SAMPLES) -> tuple[NumpyArray, NumpyArray]:
        """
        Calculates the velocity profile of the motion.

        Returns
        -------
        t_arr :
            Numpy array with values on the time-axis.
        v_arr:
            Numpy array with the corresponding values on the velocity-axis.
        """
        # initial acceleration phase
        t0 = 0.0
        v0 = self.v_i if self.v_i is not None else self.v_top
        if self.dt_i > 0.0:
            t1_arr, v1_arr = self.velocity_acc(self.dv_i, v0=v0)
        else:
            t1_arr, v1_arr = np.array([t0]), np.array([v0])

        # constant-velocity phase
        t1 = float(t1_arr[-1])
        v1 = float(v1_arr[-1])
        if self.dt_cov > 0.0:
            t2 = t1 + self.dt_cov
            t2_arr = np.linspace(t1, t2, n_samples)
            v2_arr = np.full_like(t2_arr, self.v_top)
        else:
            t2_arr, v2_arr = None, None
            t2, v2 = t1, v1

        # final acceleration phase
        if self.dt_f > 0:
            t3_arr, v3_arr = self.velocity_acc(self.dv_f, t0=t2, v0=self.v_top)
        else:
            t3_arr, v3_arr = None, None

        if t2_arr is None and t3_arr is not None:
            t_arr = np.concatenate((t1_arr[:-1], t3_arr))
            v_arr = np.concatenate((v1_arr[:-1], v3_arr))
        elif t2_arr is not None and t3_arr is None:
            t_arr = np.concatenate((t1_arr[:-1], t2_arr))
            v_arr = np.concatenate((v1_arr[:-1], v2_arr))
        elif t2_arr is None and t3_arr is None:
            t_arr = t1_arr
            v_arr = v1_arr
        else:
            t_arr = np.concatenate((t1_arr[:-1], t2_arr[:-1], t3_arr))  #type: ignore
            v_arr = np.concatenate((v1_arr[:-1], v2_arr[:-1], v3_arr))  #type: ignore

        return t_arr, v_arr

    # noinspection PyTypeChecker
    def position_profile(self, n_samples: int = NUM_SAMPLES) -> tuple[NumpyArray, NumpyArray]:
        """
        Calculates the position profile of the motion.

        Returns
        -------
        t_arr :
            Numpy array with values on the time-axis.
        s_arr:
            Numpy array with the corresponding values on the position-axis.
        """
        # initial acceleration phase
        s0 = self.s_i
        v0 = self.v_i if self.v_i is not None else self.v_top
        if self.dt_i > 0.0:
            t1_arr, s1_arr = self.position_acc(self.dv_i, s0=s0, v0=v0)
        else:
            t1_arr, s1_arr = np.array([0.0]), np.array([s0])

        # constant-velocity phase
        t1, s1 = float(t1_arr[-1]), float(s1_arr[-1])
        if self.dt_cov > 0.0:
            t2 = t1 + self.dt_cov
            t2_arr = np.linspace(t1, t2, n_samples)
            s2_arr = s1 + self.v_top * (t2_arr - t1)
            s2 = float(s2_arr[-1])  #type: ignore
        else:
            t2_arr, s2_arr = None, None
            t2, s2 = t1, s1

        # final acceleration phase
        if self.dt_f > 0.0:
            t3_arr, s3_arr = self.position_acc(self.dv_f, t0=t2, s0=s2, v0=self.v_top)
        else:
            t3_arr, s3_arr = None, None

        if t2_arr is None and t3_arr is not None:
            t_arr = np.concatenate((t1_arr[:-1], t3_arr))
            s_arr = np.concatenate((s1_arr[:-1], s3_arr))
        elif t2_arr is not None and t3_arr is None:
            t_arr = np.concatenate((t1_arr[:-1], t2_arr))
            s_arr = np.concatenate((s1_arr[:-1], s2_arr))
        elif t2_arr is None and t3_arr is None:
            t_arr = t1_arr
            s_arr = s1_arr
        else:
            t_arr = np.concatenate((t1_arr[:-1], t2_arr[:-1], t3_arr))  #type: ignore
            s_arr = np.concatenate((s1_arr[:-1], s2_arr[:-1], s3_arr))  #type: ignore

        return t_arr, s_arr

    # noinspection PyTypeChecker
    def acceleration_profile(self, n_samples: int = NUM_SAMPLES) -> tuple[NumpyArray, NumpyArray]:
        """
        Calculates the acceleration profile of the motion.

        Returns
        -------
        t_arr :
            Numpy array with values on the time-axis.
        a_arr:
            Numpy array with the corresponding values on the acceleration-axis.
        """
        # initial acceleration phase
        t1_arr, a1_arr = self._acceleration(self.dv_i)

        # constant-velocity phase
        if self.dt_cov > 0.0:
            t1 = float(t1_arr[-1])
            t2 = t1 + self.dt_cov
            t2_arr = np.linspace(t1, t2, n_samples)
            a2_arr = np.full_like(t2_arr, 0.0)
        else:
            t2_arr = None
            a2_arr = None
            t2, a2 = float(t1_arr[-1]), float(a1_arr[-1])

        # final acceleration phase
        if self.dt_f > 0:
            t3_arr, a3_arr = self._acceleration(self.dv_f, t0=t2)
        else:
            t3_arr, a3_arr = None, None

        if t2_arr is None and t3_arr is not None:
            t_arr = np.concatenate((t1_arr, t3_arr))
            a_arr = np.concatenate((a1_arr, a3_arr))
        elif t2_arr is not None and t3_arr is None:
            t_arr = np.concatenate((t1_arr, t2_arr))
            a_arr = np.concatenate((a1_arr, a2_arr))
        elif t2_arr is None and t3_arr is None:
            t_arr = t1_arr
            a_arr = a1_arr
        else:
            t_arr = np.concatenate((t1_arr, t2_arr, t3_arr))
            a_arr = np.concatenate((a1_arr, a2_arr, a3_arr))

        return t_arr, a_arr

    def get_velocity_from_time_fn(self) -> Callable[[float], float]:
        """
        Returns a function that takes a time moment `t` and returns the
        velocity `v` at that time moment (`0 <= t <= dt_tot`).
        """
        t_ax, v_ax = self.velocity_profile()
        interp = interp1d(t_ax, v_ax)

        def f(t: float) -> float:
            try:
                v = interp(t)
            except ValueError:
                v = v_ax[0]
                if t > t_ax[-1]:
                    v = v_ax[-1]
            return v  #type: ignore

        return f

    def get_velocity_from_position_fn(self) -> Callable[[float], float]:
        """
        Returns a function that takes a position `s` and returns the velocity
        `v` at that position (`0 <= s <= ds_tot`).
        """
        t_ax, s_ax = self.position_profile()
        v_ax = np.array(list(map(self.get_velocity_from_time_fn(), t_ax)))
        v_ax = np.clip(v_ax, 1e-12, None)
        interp = interp1d(s_ax, v_ax)

        def f(s: float) -> float:
            try:
                v = interp(s)
            except ValueError:
                v = v_ax[0]
                if s > s_ax[-1]:
                    v = v_ax[-1]
            return v  #type: ignore

        return f

    def get_position_from_time_fn(self) -> Callable[[float], float]:
        """
        Returns a function that takes a time moment `t` and returns the
        position `s` at that time moment in the movement (`0 <= t <= dt_tot`).
        """
        t_ax, s_ax = self.position_profile()
        interp = interp1d(t_ax, s_ax)

        def f(t: float) -> float:
            try:
                s = interp(t)
            except ValueError:
                s = s_ax[0]
                if t > t_ax[-1]:
                    s = s_ax[-1]
            return s  #type: ignore

        return f

    def get_time_from_position_fn(self) -> Callable[[float], float]:
        """
        Returns a function that takes a position `s` and returns the time
        moment `t` this position is reached in the movement (`0 <= s <= ds_tot`).
        """
        t_ax, s_ax = self.position_profile()
        interp = interp1d(s_ax, t_ax)

        def f(s: float) -> float:
            try:
                t = interp(s)
            except ValueError:
                t = t_ax[0]
                if s > s_ax[-1]:
                    t = t_ax[-1]
            return t  #type: ignore

        return f

    def get_acceleration_from_time_fn(self) -> Callable[[float], float]:
        """
        Returns a function that takes a time moment `t` and returns the
        acceleration `a` at that time moment in the movement (`0 <= t <= dt_tot`).
        """
        t_ax, a_ax = self.acceleration_profile()
        interp = interp1d(t_ax, a_ax)

        def f(t: float) -> float:
            try:
                a = interp(t)
            except ValueError:
                a = a_ax[0]
                if t > t_ax[-1]:
                    a = a_ax[-1]
            return a  #type: ignore

        return f

    def get_ini_velocity_from_time_fn(self) -> Callable[[float], float]:
        """
        Returns a function that takes a time moment `t` in seconds and
        returns the velocity `v` at that moment during the initial
        acceleration phase of the movement (`0 <= t <= dt_i`).

        If `t > dt_i`, the velocity at `dt_i` is returned (i.e. the top velocity
        during the constant-velocity phase).
        """
        t_arr, v_arr = self.velocity_acc(self.dv_i)
        t1 = t_arr[-1]
        v_max = v_arr[-1]
        interp = interp1d(t_arr, v_arr)

        def f(t: float) -> float:
            if 0.0 <= t <= t1:
                v = interp(t)
            elif t < 0.0:
                v = 0.0
            else:
                v = v_max
            return v  #type: ignore

        return f

    def get_fin_velocity_from_time_fn(
        self,
        t0: float,
        v0: float
    ) -> Callable[[float], float]:
        """
        Returns a function that takes a time moment `t` in seconds and returns
        the velocity `velocity` at that moment during the final acceleration phase of
        the movement (`t0 <= t <= t0 + dt_f`).

        If `t > t0 + dt_f`, a velocity of zero is returned.
        If `t < t0`, the initial velocity `v0` is returned.

        Parameters
        ----------
        t0 : float
            Time moment the final acceleration phase begins.
        v0: float
            Initial velocity at the start of the final acceleration phase.
        """
        t_arr, v_arr = self.velocity_acc(self.dv_f, t0, v0)
        t1 = t_arr[-1]
        interp = interp1d(t_arr, v_arr)

        def f(t: float) -> float:
            if t0 <= t <= t1:
                v = interp(t)
                if v < 0.0: v = 0.0  #type: ignore
            elif t > t1:
                v = 0.0
            else:
                v = v0
            return v  #type: ignore

        return f

    def get_ini_time_from_position_fn(self) -> Callable[[float], float]:
        """
        Returns a function that takes a position `s` during the initial
        acceleration phase and returns the time moment `t` in seconds when this
        position is reached.

        If `s` is greater than the final acceleration position, the time moment
        is returned when the final acceleration position is reached.
        """
        t_arr, s_arr = self.position_acc(self.dv_i)
        t_min = t_arr[0]
        t_max = t_arr[-1]
        s_max = s_arr[-1]
        interp_s = interp1d(s_arr, t_arr)

        def f(s: float) -> float:
            if 0.0 <= s <= s_max:
                t = interp_s(s)
            elif s > s_max:
                t = t_max
            else:
                t = t_min
            return t  #type: ignore

        return f

    def get_fin_time_from_position_fn(
        self,
        t0: float,
        s0: float,
        v0: float
    ) -> Callable[[float], float]:
        """
        Returns a function that takes a position `s` during the final
        acceleration phase and returns the time moment `t` in seconds when this
        position is reached.

        If `s` is greater than the final acceleration position, the time moment
        is returned when the final acceleration position is reached.
        If `s < s0`, the time moment `t0` is returned.

        Parameters
        ----------
        t0 : float
            Time moment the final acceleration phase begins.
        s0 : float
            Initial position at the start of the final acceleration phase.
        v0: float
            Initial velocity at the start of the final acceleration phase.
        """
        t_arr, v_arr = self.velocity_acc(self.dv_f, t0, v0)
        t1 = t_arr[-1]

        # Between t0 and t1 it is possible for the velocity to become negative
        # depending on the initial conditions.
        if v_arr[-1] < 0.0:
            # Find time `t1` where `velocity = 0`
            t_pos = t_arr[v_arr > 0][-1]
            t_neg = t_arr[v_arr < 0][0]
            interp_t = interp1d(t_arr, v_arr)
            sol = root_scalar(interp_t, bracket=(t_pos, t_neg))
            t1 = sol.root

        t_arr, s_arr = self.position_acc(self.dv_f, t1, s0, v0)
        t_min = t_arr[0]
        t_max = t_arr[-1]
        s_max = s_arr[-1]

        interp_s = interp1d(s_arr, t_arr)

        def f(s: float) -> float:
            if s0 <= s <= s_max:
                t = interp_s(s)
            elif s > s_max:
                t = t_max
            else:
                t = t_min
            return t  #type: ignore

        return f

    def position(self, t: float | NumpyArray) -> float | NumpyArray:
        if isinstance(t, float):
            return self._pos_fn(t)
        elif isinstance(t, NumpyArray):
            return np.array([self._pos_fn(t_) for t_ in t])  #type: ignore
        raise ValueError("Position could not be determined.")

    def velocity(self, t: float | NumpyArray) -> float | NumpyArray:
        if isinstance(t, float):
            return self._vel_fn(t)
        elif isinstance(t, NumpyArray):
            return np.array([self._vel_fn(t_) for t_ in t])  #type: ignore
        raise ValueError("Velocity could not be determined.")

    def acceleration(self, t: float | NumpyArray) -> float | NumpyArray:
        if isinstance(t, float):
            return self._acc_fn(t)
        elif isinstance(t, NumpyArray):
            return np.array([self._acc_fn(t_) for t_ in t])  #type: ignore
        raise ValueError("Acceleration could not be determined.")


class TrapezoidalProfile(TriPhaseMotionProfile):

    def get_dv_min(self) -> float:
        dv_min = self.dt_acc_min * self.a_max
        return dv_min

    def _calc_a_top(self, dv: float) -> tuple[float, float]:
        if dv == 0.0:
            return 0.0, 0.0
        dv = abs(dv)
        dt_acc = dv / self.a_max
        if dt_acc < self.dt_acc_min:
            a_top = dv / self.dt_acc_min
            return a_top, self.dt_acc_min
        return self.a_max, dt_acc

    def _acceleration(
        self,
        dv: float,
        t0: float = 0.0,
        n_samples: int = NUM_SAMPLES
    ) -> tuple[NumpyArray, NumpyArray]:
        a_top, dt_acc = self._calc_a_top(dv)
        if dt_acc > 0.0:
            t_arr = np.linspace(0.0, dt_acc, n_samples)
            t_arr = t0 + t_arr
            a_arr = np.full_like(t_arr, a_top)
            if dv < 0.0:
                a_arr = np.full_like(t_arr, -a_top)
        else:
            t_arr = np.array([t0])
            a_arr = np.array([0.0])
        return t_arr, a_arr  #type: ignore


class SCurvedProfile(TriPhaseMotionProfile):
    """
    Implements the SCurved profile with a variable top acceleration. If the
    acceleration time with acceleration `a_max` becomes smaller than a
    predefined lower limit, the acceleration profile during intial/final
    acceleration of the motion becomes trapezoidal while the acceleration time
    remains fixed to its lower limit.
    """
    def get_dv_min(self) -> float:
        dv_min = self.dt_acc_min * self.a_max / 2
        return dv_min

    def _calc_a_top(self, dv: float) -> tuple[float, float]:
        if dv == 0.0:
            return 0.0, 0.0
        dv = abs(dv)
        dt_acc = 2 * dv / self.a_max
        if dt_acc < self.dt_acc_min:
            theta = np.arctan(2 * self.a_max / self.dt_acc_min)
            a = -np.sin(theta) * np.cos(theta)
            b = self.dt_acc_min * np.sin(theta)
            c = -dv
            a_top = min(
                (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a),
                (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            )
            return a_top, self.dt_acc_min
        return self.a_max, dt_acc

    def _acceleration(
        self,
        dv: float,
        t0: float = 0.0,
        n_samples: int = NUM_SAMPLES
    ) -> tuple[NumpyArray, NumpyArray]:

        a_top, dt_acc = self._calc_a_top(dv)
        if dt_acc > 0.0:
            t_arr = np.linspace(t0, t0 + dt_acc, n_samples)
            dt = t_arr[-1] - t_arr[0]

            if a_top < self.a_max:
                j1 = 2 * self.a_max / self.dt_acc_min
                theta = np.arctan(j1)
                dt_acc = a_top / np.tan(theta)
                dt_cst = dt - 2 * dt_acc
                t1 = t_arr[0] + dt_acc
                t2 = t_arr[0] + dt_acc + dt_cst

                # Add t1 and t2 to t_arr if not already present.
                for t in [t1, t2]:
                    if not np.any(np.isclose(t_arr, t)):
                        t_arr = np.insert(t_arr, np.searchsorted(t_arr, t), t)

                t1_arr = t_arr[t_arr <= t1]
                tc_arr = t_arr[(t_arr > t1) & (t_arr < t2)]
                t2_arr = t_arr[t_arr >= t2]
                a1_arr = _CubicPolynomial.acceleration(t1_arr, t0=t_arr[0], a0=0.0, j0=j1)
                ac_arr = _CubicPolynomial.acceleration(tc_arr, t0=t1, a0=a_top, j0=0.0)
                a2_arr = _CubicPolynomial.acceleration(t2_arr, t0=t2, a0=a_top, j0=-j1)
                a_arr = np.concatenate((a1_arr, ac_arr, a2_arr))
                if dv < 0.0:
                    a_arr = -a_arr
            else:
                j1 = 2 * self.a_max / dt
                dt_acc = dt / 2
                t1 = t_arr[0] + dt_acc

                # Add t1 to t_arr if not already present.
                if not np.any(np.isclose(t_arr, t1)):
                    t_arr = np.insert(t_arr, np.searchsorted(t_arr, t1), t1)

                t1_arr = t_arr[t_arr <= t1]
                t2_arr = t_arr[t_arr > t1]
                a1_arr = _CubicPolynomial.acceleration(t1_arr, t0=t_arr[0], a0=0.0, j0=j1)
                a2_arr = _CubicPolynomial.acceleration(t2_arr, t0=t1, a0=a_top, j0=-j1)
                a_arr = np.concatenate((a1_arr, a2_arr))
                if dv < 0.0:
                    a_arr = -a_arr
        else:
            t_arr = np.array([t0])
            a_arr = np.array([0.0])
        return t_arr, a_arr


class _CubicPolynomial:
    """
    Little helper class to quickly get a result from a cubic kinematic
    polynomial with given boundary conditions.
    """
    @staticmethod
    def acceleration(t: float | NumpyArray, **conds: float) -> float | NumpyArray:
        t0 = conds.get('t0', 0)
        a0 = conds.get('a0', 0)
        j0 = conds.get('j0', 0)
        t = t - t0
        return a0 + j0 * t

    @staticmethod
    def velocity(t: float | NumpyArray, **conds: float) -> float | NumpyArray:
        t0 = conds.get('t0', 0)
        v0 = conds.get('v0', 0)
        a0 = conds.get('a0', 0)
        j0 = conds.get('j0', 0)
        t = t - t0
        return v0 + a0 * t + (1 / 2) * j0 * t ** 2

    @staticmethod
    def position(t: float | NumpyArray, **conds: float) -> float | NumpyArray:
        t0 = conds.get('t0', 0)
        s0 = conds.get('s0', 0)
        v0 = conds.get('v0', 0)
        a0 = conds.get('a0', 0)
        j0 = conds.get('j0', 0)
        t = t - t0
        return s0 + v0 * t + (1 / 2) * a0 * t ** 2 + (1 / 6) * j0 * t ** 3


class _QuinticPolynomial:
    @staticmethod
    def jerk(t: float | NumpyArray, **conds: float) -> float | NumpyArray:
        t0 = conds.get('t0', 0)
        j0 = conds.get('j0', 0)
        jd_0 = conds.get('jd_0', 0)
        jdd_0 = conds.get('jdd_0', 0)
        t = t - t0
        return (
            j0
            + jd_0 * t
            + (1 / 2) * jdd_0 * t ** 2
        )

    @staticmethod
    def acceleration(t: float | NumpyArray, **conds: float) -> float | NumpyArray:
        t0 = conds.get('t0', 0)
        a0 = conds.get('a0', 0)
        j0 = conds.get('j0', 0)
        jd_0 = conds.get('jd_0', 0)
        jdd_0 = conds.get('jdd_0', 0)
        t = t - t0
        return (
            a0
            + j0 * t
            + (1 / 2) * jd_0 * t ** 2
            + (1 / 6) * jdd_0 * t ** 3
        )

    @staticmethod
    def velocity(t: float | NumpyArray, **conds: float) -> float | NumpyArray:
        t0 = conds.get('t0', 0)
        v0 = conds.get('v0', 0)
        a0 = conds.get('a0', 0)
        j0 = conds.get('j0', 0)
        jd_0 = conds.get('jd_0', 0)
        jdd_0 = conds.get('jdd_0', 0)
        t = t - t0
        return (
            v0
            + a0 * t
            + (1 / 2) * j0 * t ** 2
            + (1 / 6) * jd_0 * t ** 3
            + (1 / 24) * jdd_0 * t ** 4
        )

    @staticmethod
    def position(t: float | NumpyArray, **conds: float) -> float | NumpyArray:
        t0 = conds.get('t0', 0)
        s0 = conds.get('s0', 0)
        v0 = conds.get('v0', 0)
        a0 = conds.get('a0', 0)
        j0 = conds.get('j0', 0)
        jd_0 = conds.get('jd_0', 0)
        jdd_0 = conds.get('jdd_0', 0)
        t = t - t0
        return (
            s0
            + v0 * t
            + (1 / 2) * a0 * t ** 2
            + (1 / 6) * j0 * t ** 3
            + (1 / 24) * jd_0 * t ** 4
            + (1 / 120) * jdd_0 * t ** 5
        )
