"""
Dynamics of a single rigid body.

Studies the motion of a rigid body caused by the forces and/or torques
that are acting on it.
"""
from typing import Callable
from dataclasses import dataclass, field

import numpy as np
from spatialmath import SO3, SE3

from .types import NumpyArray, ArrayLike3
from .frame import Frame


@dataclass(frozen=True)
class InertiaTensor:
    J_xx: float
    J_yy: float
    J_zz: float
    J_xy: float = 0.0
    J_xz: float = 0.0
    J_yz: float = 0.0

    @property
    def matrix(self) -> NumpyArray:
        return np.array([
            [self.J_xx, self.J_xy, self.J_xz],
            [self.J_xy, self.J_yy, self.J_yz],
            [self.J_xz, self.J_yz, self.J_zz]
        ])


@dataclass(frozen=True)
class RBMSimSolution:
    t_arr: NumpyArray
    p_arr: NumpyArray | None = None
    v_arr: NumpyArray | None = None
    a_arr: NumpyArray | None = None
    omega_arr: NumpyArray | None = None
    omega_dot_arr: NumpyArray | None = None
    frames: list[Frame] = field(default_factory=list)


class TranslationalMotion:
    """
    Implements Newton's equation of translational motion of a rigid body in
    3D space. The translation (position) of the rigid body is described w.r.t.
    the inertial world reference frame.
    """
    def __init__(
        self,
        m: float,
        dt: float,
        frame0: Frame,
        v0: ArrayLike3 | None = None
    ) -> None:
        """
        Initializes the solver.

        Parameters
        ----------
        m: float
            Mass of the rigid body (in kg).
        dt: float
            Simulation time step (in seconds).
        frame0: Frame
            Initial pose of the rigid body w.r.t. the inertial world reference
            frame.
        v0: NumpyArray, optional
            Initial translational velocity vector (in m/s) of the rigid body
            measured in the world reference frame. If not provided, defaults to
            (0, 0, 0).
        """
        self.m = m
        self.dt = dt
        self.frame0 = frame0
        self.v0 = np.asarray(v0, dtype=float) if v0 is not None else np.zeros(3)

        # Initial position of the body frame's origin relative to the origin of
        # the world reference frame:
        self.p0 = np.asarray(frame0.origin, dtype=float)

        # Orientation of the rigid body frame w.r.t. the world reference frame
        # = constant in time:
        self.R0 = self.frame0.orient_mat

    def step(self, v: NumpyArray, F: NumpyArray) -> tuple[NumpyArray, NumpyArray]:
        """
        Solves the Newton equation for the translational velocity of the
        rigid body at the next time moment t + dt.

        Parameters
        ----------
        v: NumpyArray
            Translational velocity vector (in m/s) of the rigid body w.r.t. the
            world reference frame at the current time moment t.
        F: NumpyArray
            External force vector (in N) applied to the rigid body as seen from
            the world reference frame at the current time moment t.

        Returns
        -------
        a: NumpyArray
            Translational acceleration vector between time moments t and t + dt.
        v_next: NumpyArray
            Translational velocity vector at the next time moment t + dt.
        """
        a = F / self.m
        v_next = v + a * self.dt
        return a, v_next

    def inv_dyn(
        self,
        F_fun: Callable[[float], NumpyArray],
        t_end: float = 1.0
    ) -> RBMSimSolution:
        """
        Solves the equation of motion for the translational velocity of the
        rigid body when the net force acting on the rigid body is given as a
        function of time. This is also called inverse dynamics.

        Parameters
        ----------
        F_fun: Callable[[float], NumpyArray]
            Function that takes a time moment t and returns the force vector
            (in N) applied to the rigid body w.r.t. the world reference frame at
            that time moment.
        t_end: float
            End time of the simulation in seconds.

        Returns
        -------
        RBMSimSolution
            t_arr: NumpyArray
                Numpy 1D array with the time moments of the simulation.
            v_arr: NumpyArray
                NumPy (n x 3) array of which the rows contain the translational
                velocity at each time moment of the simulation.
            frames: list[Frame]
                List with the poses of the rigid body w.r.t. the world reference
                frame at each time moment of the simulation.
        """
        n = int(t_end / self.dt) + 1  # number of time steps
        t_arr = np.zeros(n)
        a_arr = np.zeros((n, 3))
        v_arr = np.zeros((n, 3))
        p_arr = np.zeros((n, 3))

        v_arr[0, :] = self.v0
        p_arr[0, :] = self.p0
        frames = [self.frame0]

        for k in range(1, n):
            t = k * self.dt
            t_arr[k] = t

            F = F_fun(t)

            v_prev = v_arr[k - 1, :]
            a, v_next = self.step(v_prev, F)
            a_arr[k, :] = a
            v_arr[k, :] = v_next
            p_arr[k, :] = p_arr[k-1, :] + 0.5 * (v_prev + v_next) * self.dt

            frames.append(Frame.from_matrix(SE3.Rt(self.R0, p_arr[k, :])))

        return RBMSimSolution(t_arr, p_arr, v_arr, a_arr, frames=frames)

    def fwd_dyn(
        self,
        a_fun: Callable[[float], NumpyArray],
        t_end: float = 1.0
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Solves the equation of motion for the force vector acting on the rigid
        body when the translational acceleration of the rigid body is given.
        This is also called forward dynamics.

        Parameters
        ----------
        a_fun: Callable[[float], NumpyArray]
            Function that takes a time moment t and returns the acceleration
            vector (in m/s²) w.r.t. the world reference frame at that time
            moment.
        t_end: float
            End time of the simulation in seconds.

        Returns
        -------
        t_arr: NumpyArray
            Numpy 1D array with the time moments of the simulation.
        F_arr: NumpyArray
            NumPy array (n x 3) of which the rows contain the force vector
            applied to the rigid body w.r.t. the world reference frame at each
            time moment of the simulation.
        """
        t = 0
        t_list = []
        F_list = []
        while t < t_end:
            t_list.append(t)
            a = a_fun(t)
            F = self.m * a
            F_list.append(F)
            t += self.dt
        return np.array(t_list), np.array(F_list)


class RotationalMotion:
    """
    Implements Euler's equation of rotational motion of a rigid body in 3D
    space. Angular velocity, angular acceleration, torque, and the inertia
    tensor are expressed in a body-fixed frame, typically with its origin at
    the body's center of mass. The body's orientation itself is obtained by
    integrating these body-frame angular velocities from the initial pose,
    resulting in a pose expressed with respect to the inertial world frame.
    """
    def __init__(
        self,
        J_body: InertiaTensor,
        dt: float,
        frame0: Frame,
        omega_body_0: ArrayLike3 | None = None
    ) -> None:
        """
        Initializes the solver.

        Parameters
        ----------
        J_body: InertiaTensor
            Inertia tensor describing the rotational inertia of the rigid body
            (in kg.m²) relative to a frame that is fixed to the rigid body (i.e.
            the body frame).
        dt: float
            Simulation time step (in seconds).
        frame0: Frame
            Initial pose of the rigid body w.r.t. the world reference frame.
        omega_body_0: ArrayLike3, optional
            Initial angular velocity vector (in rad/s) of the rigid body
            expressed in the body frame. If not provided, defaults to (0, 0, 0).
        """
        self.J_body = J_body
        self.dt = dt
        self.frame0 = frame0
        self.R0 = frame0.orient_mat
        self.omega_body_0 = np.asarray(omega_body_0, dtype=float) if omega_body_0 is not None else np.zeros(3)

        self._J_body: NumpyArray = J_body.matrix
        self._J_body_inv = np.linalg.inv(self._J_body)

    def step(self, omega_body: NumpyArray, T_body: NumpyArray) -> tuple[NumpyArray, NumpyArray]:
        """
        Solves the Euler equation for the angular velocity of the rigid body
        at the next time moment t + dt.

        Parameters
        ----------
        omega_body: NumpyArray
            Angular velocity vector (in rad/s) w.r.t. the body frame at the
            current time moment t.
        T_body: NumpyArray
            Externally applied torque vector (in Nm) w.r.t. the body frame at
            the current time moment t.

        Returns
        -------
        omega_dot_body: NumpyArray
            Angular acceleration vector w.r.t. body frame between time moments
            t and t + dt
        omega_body: NumpyArray
            Angular velocity vector w.r.t. body frame at the next time moment
            t + dt.
        """
        omega_dot_body = self._J_body_inv @ (T_body - np.cross(omega_body, self._J_body @ omega_body))
        omega_body = omega_body + omega_dot_body * self.dt
        return omega_dot_body, omega_body

    def inv_dyn(
        self,
        T_body_fun: Callable[[float], NumpyArray],
        t_end: float = 1.0
    ) -> RBMSimSolution:
        """
        Solves the equation of motion for the angular velocity and orientation
        of the rigid body when the net torque vector acting about an axis of the
        body frame is given as a function of time. This is also called inverse
        dynamics.

        Parameters
        ----------
        T_body_fun: Callable[[float], NumpyArray]
            Function that takes a time moment t and returns the externally
            applied torque vector (in Nm) w.r.t. the body frame at that time
            moment.
        t_end: float
            End time of the simulation in seconds.

        Returns
        -------
        RBMSimSolution
            t_arr: NumpyArray
                Numpy 1D array with the time moments of the simulation.
            omega_arr: NumpyArray
                Numpy array (n x 3) of which the rows contain the angular velocity
                w.r.t. the body frame at each time moment of the simulation.
            frames: list[Frame]
                List with the pose of the rigid body at each time moment of the
                simulation, expressed with respect to the inertial world reference
                frame. The orientation is obtained by post-multiplying the initial
                orientation by incremental rotations expressed in the current body
                frame.
        """
        n = int(t_end / self.dt) + 1
        t_arr = np.zeros(n)
        omega_dot_arr = np.zeros((n, 3))
        omega_arr = np.zeros((n, 3))
        omega_arr[0, :] = self.omega_body_0
        frames = [self.frame0]
        R = self.R0

        for k in range(1, n):
            t = k * self.dt
            t_arr[k] = t
            T = T_body_fun(t)
            omega_dot_arr[k, :], omega_arr[k, :] = self.step(omega_arr[k - 1, :], T)  # omega at t
            delta_R = SO3.EulerVec(omega_arr[k, :] * self.dt)   # orientation of the body frame at t relative to its orientation at t - dt
            R *= delta_R  # update world-to-body orientation with body-frame increment
            frame = Frame.from_matrix(SE3.Rt(R, self.frame0.origin))  # pose of the rigid body at t relative to its pose at t = 0
            frames.append(frame)

        return RBMSimSolution(
            t_arr,
            omega_arr=omega_arr,
            omega_dot_arr=omega_dot_arr,
            frames=frames
        )

    def fwd_dyn(
        self,
        omega_body_fun: Callable[[float], NumpyArray],
        omega_dot_body_fun: Callable[[float], NumpyArray],
        t_end: float = 1.0
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Solves the equation of motion for the torque vector when the rotational
        kinematics of the rigid body w.r.t. body frame is given. This is also
        called forward dynamics.

        Parameters
        ----------
        omega_body_fun: Callable[[float], NumpyArray]
            Function that takes a time moment t and returns the angular velocity
            vector (in rad/s) w.r.t. the body frame at that time moment.
        omega_dot_body_fun: Callable[[float], NumpyArray]
            Function that takes a time moment t and returns the angular
            acceleration vector (in rad/s²) w.r.t. the body frame at that time
            moment.
        t_end: float
            End time of the simulation in seconds.

        Returns
        -------
        t_arr: NumpyArray
            Numpy 1D array with the time moments of the simulation.
        T_arr: NumpyArray
            NumPy array (n x 3) of which the rows contain the torque vector
            w.r.t. the body frame at each time moment of the simulation.
        """
        t = 0
        t_list = []
        T_list = []
        while t < t_end:
            t_list.append(t)
            omega = omega_body_fun(t)
            omega_dot = omega_dot_body_fun(t)
            T = self._J_body @ omega_dot + np.cross(omega, self._J_body @ omega)
            T_list.append(T)
            t += self.dt
        return np.array(t_list), np.array(T_list)


class RigidBodyMotion:
    """
    Simulates simultaneous translational and rotational motion of a rigid
    body.

    The position and translational velocity of the body's center of mass are
    expressed in the inertial world frame. Angular velocity, torque, and the
    inertia tensor are expressed in the body-fixed frame whose origin is at
    the center of mass.
    
    The simulation is valid for a net force that determines the translation of
    the center of mass, and for a net torque that must be understood as the
    total moment of all external forces around the center of mass. If no net
    torque is introduced, it is implicitly assumed that the line of action of
    the resulting force passes through the center of mass, or that the total
    moment around the center of mass is zero.
    """
    def __init__(
        self,
        m: float,
        J_body: InertiaTensor,
        dt: float,
        frame0: Frame,
        v0: ArrayLike3 | None = None,
        omega_body_0: ArrayLike3 | None = None
    ) -> None:
        """
        Initialize the solver.

        Parameters
        ----------
        m: float
            Mass of the rigid body (in kg).
        J_body: InertiaTensor
            Inertia tensor describing the rotational inertia of the rigid body
            (in kg.m²) relative to the body-fixed frame whose origin is at
            the center of mass.
        dt: float
            Simulation time step (in seconds).
        frame0: Frame
            Initial pose of the body-fixed frame w.r.t. the inertial world
            reference frame.
        v0: NumpyArray, optional
            Initial translational velocity vector (in m/s) of the center of mass
            measured in the world reference frame. If not provided, defaults to
            (0, 0, 0).
        omega_body_0: ArrayLike3, optional
            Initial angular velocity vector (in rad/s) of the rigid body
            expressed in the body-fixed frame. If not provided, defaults to
            (0, 0, 0).
        """
        self.dt = dt
        self._transl = TranslationalMotion(m, dt, frame0, v0)
        self._rot = RotationalMotion(J_body, dt, frame0, omega_body_0)
        self.p0 = self._transl.p0
        self.v0 = self._transl.v0
        self.omega_body_0 = self._rot.omega_body_0
        self.R0 = self._rot.R0

    def inv_dyn(
        self,
        F_fun: Callable[[float], NumpyArray],
        T_body_fun: Callable[[float], NumpyArray],
        t_end=1.0
    ) -> RBMSimSolution:
        """
        Solves the motion of the rigid body when the net force and net torque
        acting on the rigid body are given as functions of time. This is also
        called inverse dynamics.

        Parameters
        ----------
        F_fun: Callable[[float], NumpyArray]
            Function that takes a time moment t and returns the net force vector
            (in N) applied through the center of mass of the rigid body w.r.t.
            the world reference frame at that time moment.
        T_body_fun: Callable[[float], NumpyArray]
            Function that takes a time moment t and returns the externally
            applied torque vector (in Nm) w.r.t. the body-fixed frame at that
            time moment.
        t_end: float
            End time of the simulation in seconds.

        Returns
        -------
        RBMSimSolution
            t_arr: NumpyArray
                Numpy 1D array with the time moments of the simulation.
            v_arr: NumpyArray
                NumPy (n x 3) array of which the rows contain the translational
                velocity at each time moment of the simulation.
            omega_arr: NumpyArray
                Numpy array (n x 3) of which the rows contain the angular velocity
                w.r.t. the body frame at each time moment of the simulation.
            frames: list[Frame]
                List with the poses of the body-fixed frame w.r.t. the world
                reference frame at each time moment of the simulation.
        """
        n = int(t_end / self.dt) + 1
        t_arr = np.zeros(n)
        p_arr = np.zeros((n, 3))
        v_arr = np.zeros((n, 3))
        a_arr = np.zeros((n, 3))
        omega_arr = np.zeros((n, 3))
        omega_dot_arr = np.zeros((n, 3))
        frames = []

        p_arr[0, :] = self.p0
        v_arr[0, :] = self.v0
        omega_arr[0, :] = self.omega_body_0

        R = self.R0
        frames.append(Frame.from_matrix(SE3.Rt(R, p_arr[0, :])))

        for k in range(1, n):
            t = k  * self.dt
            t_arr[k] = t

            F_world = F_fun(t)
            T_body = T_body_fun(t)

            # translation
            v_prev = v_arr[k - 1, :]
            a_arr[k, :], v_next = self._transl.step(v_prev, F_world)
            v_arr[k, :] = v_next
            p_arr[k, :] = p_arr[k - 1, :] + 0.5 * (v_prev + v_next) * self.dt

            # rotation
            omega_dot_arr[k, :], omega_arr[k, :] = self._rot.step(omega_arr[k - 1, :], T_body)
            delta_R = SO3.EulerVec(omega_arr[k, :] * self.dt)
            R *= delta_R

            frames.append(Frame.from_matrix(SE3.Rt(R, p_arr[k, :])))

        return RBMSimSolution(
            t_arr,
            p_arr, v_arr, a_arr,
            omega_arr, omega_dot_arr,
            frames
        )
