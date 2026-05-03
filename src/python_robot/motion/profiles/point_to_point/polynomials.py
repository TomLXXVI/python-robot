"""
Polynomials for straight-line (1D) motion profiles.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ....base.types import NumpyArray


__all__ = ["CubicPolynomial", "QuinticPolynomial"]


@dataclass
class Polynomial(ABC):
    """
    Abstract base class for "kinematic polynomials".
    """
    # Total duration of the 1D motion.
    T: float = field(init=False)
    # NumPy array holding the constant coefficients of the polynomial.
    C: NumpyArray = field(init=False)
    
    @classmethod
    @abstractmethod
    def from_constraints(cls, *args, **kwargs) -> Polynomial:
        """
        Determines the values of the polynomial coefficients from the given 
        kinematic boundary conditions or constraints.
        """
        pass
    
    @abstractmethod
    def position(self, t: float | NumpyArray) -> float | NumpyArray:
        """Returns the position of the polynomial at time t."""
        pass

    @abstractmethod
    def velocity(self, t: float | NumpyArray) -> float | NumpyArray:
        """Returns the velocity of the polynomial at time t."""
        pass

    @abstractmethod
    def acceleration(self, t: float | NumpyArray) -> float | NumpyArray:
        """Returns the acceleration of the polynomial at time t."""
        pass

    @abstractmethod
    def jerk(self, t: float | NumpyArray) -> float | NumpyArray:
        """Returns the jerk of the polynomial at time t."""
        pass

    @property
    def s0(self) -> float:
        """Returns the initial position at time t = 0."""
        return float(self.position(0))

    @property
    def sT(self) -> float:
        """Returns the final position at time t = T."""
        return float(self.position(self.T))

    @property
    def v0(self) -> float:
        """Returns the initial velocity at time t = 0."""
        return float(self.velocity(0))

    @property
    def vT(self) -> float:
        """Returns the final velocity at time t = T."""
        return float(self.velocity(self.T))

    @property
    def a0(self) -> float:
        """Returns the initial acceleration at time t = 0."""
        return float(self.acceleration(0))

    @property
    def aT(self) -> float:
        """Returns the final acceleration at time t = T."""
        return float(self.acceleration(self.T))

    @property
    def j0(self) -> float:
        """Returns the initial jerk at time t = 0."""
        return float(self.jerk(0))

    @property
    def jT(self) -> float:
        """Returns the final jerk at time t = T."""
        return float(self.jerk(self.T))


@dataclass
class CubicPolynomial(Polynomial):
    """
    Represents 1D position s as a function of time by a cubic polynomial:
        s(t) = c0 + c1 * t + c2 * t^2 + c3 * t^3.
    
    Position s, velocity v, acceleration a and jerk j are the kinematic 
    variables described the polynomial.
    
    Velocity v is the first derivative of position:
        v(t) = c1 + (2 * c2) * t + (3 * c3) * t^2

    Acceleration a is the first derivative of velocity (second derivative of
    position):
        a(t) = (2 * c2) + (6 * c3) * t

    Jerk j is the first derivative of acceleration:
        j(t) = 6 * c3 (constant for a cubic polynomial)

    Kinematic boundary conditions or constraints will determine the values of 
    the polynomial coefficients c0, c1, c2 and c3.
    """
    T: float
    c0: float
    c1: float
    c2: float
    c3: float
    
    def __post_init__(self):
        self.C = np.array([self.c0, self.c1, self.c2, self.c3])
    
    @classmethod
    def from_constraints(
        cls, 
        T: float,
        s0: float,
        sT: float,
        v0: float,
        vT: float,
    ) -> CubicPolynomial:
        """
        Determines the values of the polynomial coefficients from the given 
        kinematic boundary conditions or constraints.
        
        Parameters
        ----------
        T: float
            Total duration of the straight-line displacement sT - s0.
        s0: float
            Initial position at time t = 0.
        sT: float
            Final position at time t = T.
        v0: float
            Initial velocity at time t = 0.
        vT: float
            Final velocity at time t = T.

        Returns
        -------
        CubicPolynomial
        """
        A = np.array([
            [1, 0, 0, 0],               # constraint 1: initial position s0
            [1, T, T ** 2, T ** 3],     # constraint 2: final position sT
            [0, 1, 0, 0],               # constraint 3: initial velocity v0
            [0, 1, 2 * T, 3 * T ** 2],  # constraint 4: final velocity vT
        ])
        B = np.array([s0, sT, v0, vT])
        C = np.linalg.solve(A, B)
        return cls(T, C[0], C[1], C[2], C[3])
        
    def position(self, t: float | NumpyArray) -> float | NumpyArray:
        return self.C[0] + self.C[1] * t + self.C[2] * t**2 + self.C[3] * t**3

    def velocity(self, t: float | NumpyArray) -> float | NumpyArray:
        return self.C[1] + 2 * self.C[2] * t + 3 * self.C[3] * t**2

    def acceleration(self, t: float | NumpyArray) -> float | NumpyArray:
        return 2 * self.C[2] + 6 * self.C[3] * t

    # noinspection PyUnusedLocal
    def jerk(self, t: float | NumpyArray) -> float | NumpyArray:
        return 6 * self.C[3]


@dataclass
class QuinticPolynomial(Polynomial):
    """
    Represents 1D position s as a function of time by a quintic polynomial:
        s(t) = c0 + c1 * t + c2 * t^2 + c3 * t^3 + c4 * t^4 + c5 * t^5.
    
    Position s, velocity v, acceleration a and jerk j are the kinematic 
    variables described the polynomial.
    
    Velocity v is the first derivative of position:
        v(t) = c1 + (2 * c2) * t + (3 * c3) * t^2 + (4 * c4) * t^3 + (5 * c5) * t^4

    Acceleration a is the first derivative of velocity (second derivative of
    position):
        a(t) = (2 * c2) + (6 * c3) * t + (12 * c4) * t^2 + (20 * c5) * t^3

    Jerk j is the first derivative of acceleration:
        j(t) = 6 * c3 + (24 * c4) * t + (60 * c5) * t^2

    Kinematic boundary conditions or constraints will determine the values of 
    the polynomial coefficients c0, c1, c2, c3, c4, and c5.
    """
    T: float
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float

    def __post_init__(self):
        self.C = np.array([self.c0, self.c1, self.c2, self.c3, self.c4, self.c5])

    @classmethod
    def from_constraints(
        cls,
        T: float,
        s0: float,
        sT: float,
        v0: float,
        vT: float,
        a0: float,
        aT: float,
    ) -> QuinticPolynomial:
        """
        Determines the values of the polynomial coefficients from the given 
        kinematic boundary conditions or constraints.

        Parameters
        ----------
        T: float
            Total duration of the straight-line displacement sT - s0.
        s0: float
            Initial position at time t = 0.
        sT: float
            Final position at time t = T.
        v0: float
            Initial velocity at time t = 0.
        vT: float
            Final velocity at time t = T.
        a0: float
            Initial acceleration at time t = 0.
        aT: float
            Final acceleration at time t = T.
        
        Returns
        -------
        QuinticPolynomial
        """
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3],
        ])
        B = np.array([s0, sT, v0, vT, a0, aT])
        C = np.linalg.solve(A, B)
        return cls(T, C[0], C[1], C[2], C[3], C[4], C[5])

    def position(self, t: float | NumpyArray) -> float | NumpyArray:
        return (
            self.C[0]
            + self.C[1] * t
            + self.C[2] * t**2
            + self.C[3] * t**3
            + self.C[4] * t**4
            + self.C[5] * t**5
        )

    def velocity(self, t: float | NumpyArray) -> float | NumpyArray:
        return (
            self.C[1]
            + 2 * self.C[2] * t
            + 3 * self.C[3] * t**2
            + 4 * self.C[4] * t**3
            + 5 * self.C[5] * t**4
        )

    def acceleration(self, t: float | NumpyArray) -> float | NumpyArray:
        return (
            2 * self.C[2]
            + 6 * self.C[3] * t
            + 12 * self.C[4] * t**2
            + 20 * self.C[5] * t**3
        )

    def jerk(self, t: float | NumpyArray) -> float | NumpyArray:
        return (
            6 * self.C[3]
            + 24 * self.C[4] * t**1
            + 60 * self.C[5] * t**2
        )
