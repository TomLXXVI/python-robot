from __future__ import annotations
from typing import Sequence, overload, Literal
from collections.abc import Iterable, Iterator, MutableSequence

from abc import ABC
from copy import deepcopy

import numpy as np
from spatialmath import SE3
from roboticstoolbox import ETS, ERobot

from python_robot.base.types import AngleUnit, NumpyArray, ArrayLike6
from python_robot.base import Frame, WREF_FRAME, SpatialVelocity, Wrench

from .links import AbstractLink
from .exceptions import *

__all__ = [
    "IKSolverSpec",
    "RefFrame",
    "KinematicChain"
]


IKSolverSpec = Literal["LM", "NR", "GN", "QP"]
RefFrame = Literal["world", "end-effector"]


class AbstractKinematicChain(MutableSequence[AbstractLink], ABC):
    """
    Abstract base class that implements a list for holding AbstractLink objects.
    It uses forward 1-based indexing (instead of the conventional forward
    0-based indexing). For backward indexing, the Python-convention is retained.
    So, it behaves like a normal list, but where the first AbstractLink object
    in the list has a forward index of 1 (instead of 0) and the last
    AbstractLink object in the list has a backward index of -1 (just like a
    normal Python list).
    """
    def __init__(
        self,
        links: Iterable[AbstractLink] | None = None,
    ) -> None:
        self._links: list[AbstractLink] = list(links) if links is not None else []

    def __iter__(self) -> Iterator[AbstractLink]:
        return iter(self._links)

    def __len__(self) -> int:
        return len(self._links)

    def iter_indices(self) -> Iterator[int]:
        """
        Returns an iterator over the indices of the links in the chain.
        """
        for i in range(1, len(self) + 1):
            yield i

    def _normalize_index(self, index: int) -> int:
        """
        Convert external 1-based index to internal 0-based index.

        Negative indices are kept Python-like:
        -1 means last element, -2 second-last, etc.
        """
        if index == 0:
            raise IndexError("KinematicChain uses 1-based indexing; index 0 is invalid.")
        if index > 0:
            internal_index = index - 1
        else:
            internal_index = index
        if not (-len(self._links) <= internal_index < len(self._links)):
            raise IndexError("KinematicChain index out of range.")
        return internal_index

    def _normalize_insert_index(self, index: int) -> int:
        """
        Convert external 1-based insertion index to internal 0-based index.

        Examples:
            insert(1, x) inserts at the front
            insert(2, x) inserts before the current second element
            insert(len(chain)+1, x) appends at the end
        """
        n = len(self._links)
        if index == 0:
            raise IndexError("KinematicChain uses 1-based indexing; index 0 is invalid.")
        if index > 0:
            internal_index = index - 1
        else:
            internal_index = max(0, n + index + 1)
        return max(0, min(internal_index, n))

    @overload
    def __getitem__(self, index: int) -> AbstractLink:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[AbstractLink]:
        ...

    def __getitem__(self, index: int | slice) -> AbstractLink | list[AbstractLink]:
        if isinstance(index, slice):
            return self._links[index]
        return self._links[self._normalize_index(index)]

    @overload
    def __setitem__(self, index: int, value: AbstractLink) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[AbstractLink]) -> None:
        ...

    def __setitem__(self, index: int | slice, value: AbstractLink | Iterable[AbstractLink]) -> None:
        if isinstance(index, slice):
            self._links[index] = list(value)  # type: ignore[arg-type]
        else:
            self._links[self._normalize_index(index)] = value  # type: ignore[assignment]

    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, slice):
            del self._links[index]
        else:
            del self._links[self._normalize_index(index)]

    def insert(self, index: int, value: AbstractLink) -> None:
        self._links.insert(self._normalize_insert_index(index), value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._links!r})"


class KinematicChain(AbstractKinematicChain):
    """
    Represents the kinematic chain of a manipulator, implemented as a list
    of links ordered from the base to the tool-end of the manipulator.
    """
    def __init__(
        self,
        links: Sequence[AbstractLink],
        joint_coords: Sequence[float] | None = None,
        base_frame: Frame | None = None,
        tool_frame: Frame | None = None,
    ) -> None:
        """
        Creates a KinematicChain object.

        Parameters
        ----------
        links: Sequence[AbstractLink]
            List of links in the chain, ordered from base to tool.
        joint_coords: Sequence[float], optional
            Initial values for the joint variables of the links. If None, all
            joint variables are set to zero.
            In the case of joint coordinates that are angles, the unit of the
            angles must be the same angle unit that was assigned to the links
            when instantiating these links.

        Notes
        -----
        To avoid any confusion, it is recommended to leave the angle unit at the
        default radian unit when instantiating the links that make up the chain.
        Then, if necessary, convert angles from degrees to radians before
        passing them to the KinematicChain object, and convert angles coming
        from the KinematicChain object back from radians to degrees.
        """
        super().__init__(links)
        self._base_frame: Frame = WREF_FRAME if base_frame is None else base_frame
        self._tool_frame: Frame = WREF_FRAME if tool_frame is None else tool_frame
        self._joint_coords: list[float] = []
        if joint_coords is None:
            self.joint_coords = [0.0] * len(self._links)
        else:
            self.joint_coords = joint_coords

        self._ets: ETS = self._create_ets()
        self._erobot: ERobot = ERobot(
            self._ets,
            base=self.base_frame.matrix,
            tool=self.tool_frame.matrix,
        )

    @property
    def links(self) -> list[AbstractLink]:
        """
        Returns the list of links in the chain (with zero-based indexing).
        """
        return self._links

    @property
    def base_frame(self) -> Frame:
        """
        Returns the pose of the manipulator base frame w.r.t. the world frame.
        """
        return self._base_frame

    @base_frame.setter
    def base_frame(self, frame: Frame) -> None:
        """
        Sets the pose of the manipulator base frame w.r.t. the world frame.
        """
        self._base_frame = frame
        if hasattr(self, "_erobot"):
            self._erobot.base = frame.matrix

    @property
    def tool_frame(self) -> Frame:
        """
        Returns the pose of the tool frame w.r.t. the last link frame.
        """
        return self._tool_frame

    @tool_frame.setter
    def tool_frame(self, frame: Frame) -> None:
        """
        Sets the pose of the tool frame w.r.t. the last link frame.
        """
        self._tool_frame = frame
        if hasattr(self, "_erobot"):
            self._erobot.tool = frame.matrix

    def _tcp_frame_to_link_frame(self, tcp_frame: Frame) -> Frame:
        """
        Given the tool-center point frame w.r.t. the world frame, returns the
        pose of the last link frame (often the wrist frame for serial robot
        arms) w.r.t. the base frame of the kinematic chain.

        This method is used with inverse kinematics (see method `inv_kin()`).

        Parameters
        ----------
        tcp_frame
            Tool-center point (TCP) frame (aka end-effector frame) w.r.t. world
            frame.

        Returns
        -------
        Frame
        """
        return ~self.base_frame * tcp_frame * ~self.tool_frame

    def _create_ets(self) -> ETS:
        et_list = [et for link in self._links for et in link.ets]
        jcounter = 0
        for et in et_list:
            if et.isjoint:
                et.jindex = jcounter
                jcounter += 1
        ets = ETS(et_list)
        return ets

    @property
    def ets(self) -> ETS:
        """
        Returns the underlying Elementary Transform Sequence (ETS) of the chain
        (see roboticstoolbox.robot.ETS.py).
        """
        return self._ets

    @property
    def erobot(self) -> ERobot:
        """
        Returns a ERobot instance of the kinematic chain
        (see roboticstoolbox.robot.ERobot.py).
        """
        return self._erobot

    @property
    def joint_coords(self) -> list[float]:
        """
        Returns the list of joint coordinates.
        """
        return self._joint_coords

    @joint_coords.setter
    def joint_coords(self, v: Sequence[float]) -> None:
        """
        Sets the joint coordinates of the links. The joint coordinates must be
        ordered from base to tool.

        Note that setting the joint coordinates through the setter, modifies the
        current configuration state of this KinematicChain object.

        In the case of joint coordinates that are angles, the unit of the angles
        must be the same angle unit that was assigned to the links when
        instantiating these links.

        Raises
        ------
        JointConfigurationError:
            If the number of joint coordinates does not match the number of
            links.
        """
        if len(v) != len(self._links):
            raise ConfigurationError(
                f"Number of joint coordinates ({len(v)}) does not "
                f"match the number of links in the chain ({len(self._links)})"
            )
        joint_coords_internal = self._convert_to_rad(v)
        self._check_joint_limits(joint_coords_internal)
        self._joint_coords = list(v)
        # Assign the joint variables to their respective links.
        for link, value in zip(self._links, self._joint_coords):
            link.variable = value  # -> an angle in degrees will be converted to radians

    def get_angle_units(self) -> tuple[AngleUnit, ...]:
        """
        Returns the angle units of each link in the chain, which were set when
        instantiating these links.
        """
        return tuple([link.angle_unit for link in self])

    def _convert_back_to_deg(self, joint_coords: Sequence[float]) -> Sequence[float]:
        """
        Given a sequence of joint coordinates, ordered from base to tool,
        converts any joint angle in this sequence, which should be expressed in
        radians, externally to degrees if the angle unit of the link (that is
        associated with this joint angle) is specified as degrees.
        """
        joint_coords_ = list(joint_coords)
        for i in range(len(self._links)):
            link = self._links[i]
            if link.is_revolute and link.angle_unit == "deg":
                joint_coords_[i] = float(np.rad2deg(joint_coords[i]))
            else:
                joint_coords_[i] = joint_coords[i]
        return joint_coords_

    def _convert_to_rad(self, joint_coords: Sequence[float]) -> Sequence[float]:
        """
        Given a sequence of joint coordinates, ordered from base to tool,
        converts any joint angle in this sequence, which should be expressed in
        degrees, internally to radians if the angle unit of the link (that is
        associated with this joint angle) is specified as degrees.
        """
        joint_coords_ = list(joint_coords)
        for i in range(len(self._links)):
            link = self._links[i]
            if link.is_revolute and link.angle_unit == "deg":
                joint_coords_[i] = float(np.deg2rad(joint_coords[i]))
            else:
                joint_coords_[i] = joint_coords[i]
        return joint_coords_

    def _check_number_of_joint_coords(self, joint_coords: Sequence[float]) -> Sequence[float]:
        if len(joint_coords) != len(self._links):
            raise ConfigurationError(
                f"The number of joint coordinates ({len(joint_coords)}) "
                f"does not match the number of links ({len(self._links)})."
            )
        return joint_coords

    def _check_joint_limits(self, joint_coords: Sequence[float]) -> Sequence[float]:
        for i, (link, joint_coord) in enumerate(zip(self._links, joint_coords), start=1):
            q_lim = link.q_lim_internal
            if q_lim is None:
                continue

            lower, upper = q_lim
            if joint_coord < lower or joint_coord > upper:
                raise ConfigurationError(
                    f"Joint coordinate {i} ({joint_coord:.6g}) is outside its "
                    f"mechanical limits [{lower:.6g}, {upper:.6g}]."
                )
        return joint_coords

    def pose(self, index: int) -> Frame:
        """
        Returns the pose (a Frame object) of the given link in the chain. The
        pose describes the position and orientation of the frame of the given
        link w.r.t. the base frame of the chain.

        The returned frame will depend on the current configuration state of
        this KinematicChain object.

        Parameters
        ----------
        index: int
            The index of the link in the chain.
            Note that positive indices are *not* zero-based. So, the first link
            in the chain has index 1 (instead of 0), and so on.
            Negative indices however are kept Python-like. So, the last link in
            the chain (end-effector) has index -1, and so on.

        Returns
        -------
        Frame

        Raises
        ------
        IndexError
            If the given index is out of range.
        """
        matrices = [self.base_frame.matrix]
        if 0 < index <= len(self):
            matrices.extend([self._links[i].frame.matrix for i in range(index)])
        elif -len(self) < index < 0:
            end = len(self) + index + 1
            matrices.extend([self._links[i].frame.matrix for i in range(end)])
        else:
            raise IndexError("KinematicChain index out of range.")
        matrix = np.linalg.multi_dot(matrices)
        return Frame.from_matrix(SE3(matrix), angle_unit="rad")

    def get_joint_configuration(
        self,
        joint_coords: Sequence[float] | None = None
    ) -> KinematicChain:
        """
        Returns a new KinematicChain object with the given joint configuration.

        First, creates a deepcopy of this KinematicChain object (self). Then,
        the joint coordinates are set on this new instance of KinematicChain. So,
        the configuration state of this KinematicChain object (self) remains
        unaltered.

        Parameters
        ----------
        joint_coords: Sequence[float], optional
            Values for the joint variables of the links. If None, all joint
            variables are set to zero.
            In the case of joint coordinates that are angles, the unit of the
            angles must be the same angle unit that was assigned to the links
            when instantiating these links.

        Returns
        -------
        KinematicChain
        """
        new_chain = deepcopy(self)
        if joint_coords is None:
            new_chain.joint_coords = [0.0] * len(self._links)
        else:
            new_chain.joint_coords = joint_coords
        return new_chain

    def fwd_kin(self, joint_coords: Sequence[float] | None = None) -> Frame:
        """
        Returns the pose of the end-effector (the last link frame farthest from
        the base) w.r.t. the fixed base frame of the kinematic chain for the
        given sequence of joint coordinates.

        Parameters
        ----------
        joint_coords: Sequence[float], optional
            Values of the joint variables of the links. If None, the values in
            the current joint configuration of the chain are used. However, if
            joint_coords are given, the current joint configuration of the chain
            is not changed. To actually change the internal joint configuration
            state of the chain, you must use setter joint_coords.
            In the case of joint coordinates that are angles, the unit of the
            angles must be the same angle unit that was assigned to the links
            when instantiating these links.

        Returns
        -------
        Frame
        """
        if joint_coords is not None:
            joint_coords = self._check_number_of_joint_coords(joint_coords)
            joint_coords = self._convert_to_rad(joint_coords)  # type: ignore
            joint_coords = self._check_joint_limits(joint_coords)  # type: ignore
            se3_obj = self.ets.fkine(
                np.asarray(joint_coords, dtype=float),
                base=self.base_frame.matrix,
                tool=self.tool_frame.matrix,
            )
            frame = Frame.from_matrix(se3_obj)
            return frame
        return self.pose(-1) * self.tool_frame

    def inv_kin(
        self,
        ee_frame: Frame,
        ini_guess: Sequence[float] | None = None,
        which_solver: IKSolverSpec = "LM",
        check_joint_limits: bool = True,
        **kwargs
    ) -> NumpyArray:
        """
        Given the desired pose of the end-effector (i.e. the link frame farthest
        from the base) w.r.t. the base frame of the kinematic chain, returns a
        (possible) solution for the joint coordinates using a numeric
        IK-solver .

        Parameters
        ----------
        ee_frame: Frame
            Desired pose of the end-effector.
        ini_guess: Sequence[float], optional
            An initial guess for the joint coordinates.
        which_solver: IKSolverSpec, default = "LM"
            Specificies the numeric IK-solver to use (see the docstrings of
            class ETS in module roboticstoolbox.robot.ETS.py).
        check_joint_limits: bool, default = True
            Indicates whether to check that a solution satisfies any joint
            limits or not.
        kwargs:
            Optional keyword arguments to pass to the underlying IKSolver
            (see the docstrings of the IK-solvers in class ETS in module
            roboticstoolbox.robot.ETS.py).

        Returns
        -------
        NumpyArray:
            The joint coordinates that result in the desired end-effector pose.
            In case of joint coordinates that are angles, the unit of the angle
            corresponds with the angle unit that was assigned when instantiating
            its respective link.

        Raises
        ------
        ConfigurationError:
            If an initial guess is provided whose length does not match the
            number of joints in the chain.
        IKSolverError:
            If the IK-solver could not find a solution.
        NotImplementedError:
            If the specified IK-solver is not implemented.

        Notes
        -----
        Calling method inv_kin(...) does not change the configuration state of
        the underlying KinematicChain object. To update the configuration state,
        pass the returned joint angles to setter joint_coords of this
        KinematicChain object.
        """
        if ini_guess is not None and len(ini_guess) != len(self):
            raise ConfigurationError(
                "'len(ini_guess)' is different from "
                "the number of joints in the chain."
            )

        q0 = np.asarray(self._convert_to_rad(ini_guess)) if ini_guess is not None else None

        kwargs = dict(kwargs) if kwargs is not None else {}
        kwargs.update({"joint_limits": check_joint_limits})

        if which_solver == "LM":
            sol = self.ets.ikine_LM(
                Tep=self._tcp_frame_to_link_frame(ee_frame).matrix,
                q0=q0,
                **kwargs
            )
        else:
            raise NotImplementedError(f"Solver {which_solver} is not implemented (yet).")

        if sol.success:
            return np.asarray(self._convert_back_to_deg(sol.q), dtype=float)  # convert angles in radians to degrees if needed
        raise IKSolverError("A solution was not found.")

    def jacobian(
        self,
        joint_coords: Sequence[float] | None = None,
        ref_frame: RefFrame = "world"
    ) -> NumpyArray:
        """
        Returns the Jacobian matrix of the kinematic chain that maps joint
        velocities to the end-effector spatial velocity.

        Parameters
        ----------
        joint_coords: Sequence[float], optional
            Momentary values of the joint variables of the links.
            The values of the joint variables are always expressed w.r.t. the
            world coordinate frame.
            In the case of joint coordinates that are angles, the unit of the
            angles must be the same angle unit that was assigned to the links
            when instantiating these links.
            If left to None, the joint coordinates in the current configuration
            state of the chain are used to calculate the Jacobian matrix.
        ref_frame: RefFrame, default = "world"
            Specifies the reference frame in which the spatial velocity of the
            end-effector should be expressed. By default, this reference frame
            is the fixed base frame of the chain. To express the end-effector
            spatial velocity in the end-effector frame, set ref_frame to
            "end-effector". Joint velocities are always expressed w.r.t. the
            fixed base frame.

        Returns
        -------
        jacobian_matrix: NumpyArray
            The jacobian matrix of the kinematic chain.

        Raises
        ------
        InvalidArgument:
            If ref_frame is not recognized (neither "world" nor "end-effector").
        """
        if joint_coords is not None:
            joint_coords_ = self._check_number_of_joint_coords(joint_coords)
            joint_coords_ = self._convert_to_rad(joint_coords_)  # any angles in degrees -> radians
            joint_coords_ = self._check_joint_limits(joint_coords_)
        else:
            joint_coords_ = self._convert_to_rad(self.joint_coords)
        if ref_frame == "world":
            jac_base = self.ets.jacob0(
                np.asarray(joint_coords_, dtype=float),
                tool=self.tool_frame.matrix,
            )  # requires angles in radians
            jac = self.base_frame.jacobian() @ jac_base
        elif ref_frame == "end-effector":
            jac = self.ets.jacobe(
                np.asarray(joint_coords_, dtype=float),
                tool=self.tool_frame.matrix,
            )  # requires angles in radians
        else:
            raise InvalidArgument("Unrecognized reference frame.")
        return jac

    def jacobian_pinv(
        self,
        joint_coords: Sequence[float] | None = None,
        ref_frame: RefFrame = "world",
        rcond: float | None = None,
    ) -> NumpyArray:
        """
        Returns the Moore-Penrose pseudo-inverse of the manipulator Jacobian.
        """
        jac = self.jacobian(joint_coords=joint_coords, ref_frame=ref_frame)
        if rcond is None:
            return np.linalg.pinv(jac)
        return np.linalg.pinv(jac, rcond=rcond)

    def jacobian_dot(
        self,
        joint_coords: Sequence[float],
        joint_velocities: Sequence[float],
        representation: Literal["rpy/xyz", "rpy/zyx", "eul", "exp"] | None = None,
    ) -> NumpyArray:
        """
        Returns the time derivative of the world-frame manipulator Jacobian.

        The returned matrix maps joint velocities and accelerations according to:

            A = J(q) qdd + Jdot(q, qd) qd

        where A is the end-effector spatial acceleration in the world frame
        when representation is None.
        """
        q = self._check_number_of_joint_coords(joint_coords)
        qd = self._check_number_of_joint_coords(joint_velocities)

        q = np.asarray(self._convert_to_rad(q), dtype=float)
        q = np.asarray(self._check_joint_limits(q), dtype=float)
        qd = np.asarray(qd, dtype=float)

        j0_base = self.ets.jacob0(q, tool=self.tool_frame.matrix)
        if representation is None:
            h0_base = self.ets.hessian0(q, J0=j0_base, tool=self.tool_frame.matrix)
            jdot_base = np.tensordot(h0_base, qd, (0, 0))
        else:
            jdot_base = self.erobot.jacob0_dot(  # type: ignore
                q, qd,
                J0=j0_base, representation=representation,
            )
        return np.asarray(self.base_frame.jacobian() @ jdot_base, dtype=float)

    def is_singular(
        self,
        joint_coords: Sequence[float] | None = None,
        ref_frame: RefFrame = "world",
        tol: float | None = None,
    ) -> bool:
        """
        Returns True if the kinematic chain is singular at the given joint
        configuration, else False.

        A configuration is singular when the Jacobian loses rank. This test also
        works for non-square Jacobian matrices.

        Parameters
        ----------
        joint_coords: Sequence[float], optional
            Momentary values of the joint variables of the links.
            The values of the joint variables are always expressed w.r.t. the
            world coordinate frame.
            In the case of joint coordinates that are angles, the unit of the
            angles must be the same angle unit that was assigned to the links
            when instantiating these links.
            If left to None, the joint coordinates in the current configuration
            state of the chain are used to calculate the Jacobian matrix.
        ref_frame: RefFrame, default = "world"
            Specifies the reference frame in which the spatial velocity of the
            end-effector should be expressed. By default, this reference frame
            is the fixed base frame of the chain. To express the end-effector
            spatial velocity in the end-effector frame, set ref_frame to
            "end-effector". Joint velocities are always expressed w.r.t. the
            fixed base frame.
        tol: float, optional
            Tolerance used for calculating the rank of the Jacobian matrix.
        """
        J = self.jacobian(joint_coords=joint_coords, ref_frame=ref_frame)

        if tol is None:
            rank = np.linalg.matrix_rank(J)
        else:
            rank = np.linalg.matrix_rank(J, tol=tol)

        return rank < min(J.shape)

    def get_end_effector_velocity(
        self,
        joint_velocities: Sequence[float],
        ref_frame: RefFrame = "world"
    ) -> SpatialVelocity:
        """
        Returns the spatial velocity of the end-effector w.r.t. the fixed base
        frame of the kinematic chain for the given joint velocities and for the
        current joint configuration of the chain.

        Parameters
        ----------
        joint_velocities: Sequence[float]
            The velocities of the joints in the current configuration of the
            chain. Joint velocities are always expressed w.r.t. the fixed base
            frame. Rotational velocities must be expressed in rad/s.
        ref_frame: RefFrame, default = "world"
            Specifies the reference frame in which the spatial velocity of the
            end-effector should be expressed. By default, this reference frame
            is the fixed base frame of the chain. To express the end-effector
            spatial velocity in the end-effector frame, set ref_frame to
            "end-effector".

        Returns
        -------
        SpatialVelocity
        """
        q_seq = self._check_number_of_joint_coords(joint_velocities)
        q_arr = np.asarray(q_seq, dtype=float)
        J = self.jacobian(None, ref_frame=ref_frame)
        V_ee = J @ q_arr
        return SpatialVelocity(V_ee)

    def get_static_joint_torques(
        self,
        W_ee: Wrench | ArrayLike6,
        ref_frame: RefFrame = "world"
    ) -> NumpyArray:
        """
        Calculates the joint torques/forces that must be acting to keep the
        kinematic chain in static equilibrium in the current joint configuration
        of the chain.

        Parameters
        ----------
        W_ee: Wrench | ArrayLike6
            External static load excerted on the end-effector of the chain.
        ref_frame: RefFrame, default = "world"
            Specifies the reference frame in which the wrench of the
            end-effector is expressed. By default, this reference frame
            is the fixed base frame of the chain. If the wrench is expressed in
            the end-effector frame, set ref_frame to "end-effector".

        Returns
        -------
        NumpyArray
            Joint torques/forces needed to apply the force or moment with the
            end-effector.
        """
        W_ee_arr = np.asarray(W_ee, dtype=float)
        J = self.jacobian(None, ref_frame=ref_frame)
        joint_torques = J.T @ W_ee_arr
        return joint_torques

    def __str__(self):
        return f"({''.join(["R" if link.is_revolute else "P" for link in self._links])})"
