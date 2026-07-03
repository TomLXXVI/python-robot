"""
Kinematic-chain containers and serial manipulator kinematics.

The module defines the list-like link container used by serial manipulators and
the concrete :class:`KinematicChain` implementation for forward kinematics,
inverse kinematics, Jacobians, static wrench mapping, plotting, and animation.
"""

from __future__ import annotations
from typing import Sequence, overload, Literal, Any
from collections.abc import Iterable, Iterator, MutableSequence

from abc import ABC
from copy import deepcopy

import numpy as np
from spatialmath import SE3
from roboticstoolbox import ETS, ERobot

from python_robot.base.types import NumpyArray, ArrayLike6
from python_robot.base import Frame, WREF_FRAME, SpatialVelocity, Wrench, Vector

from .exceptions import *
from .links import AbstractLink


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
        """
        Create a list-like container for links.

        Parameters
        ----------
        links : Iterable[AbstractLink], optional
            Initial links ordered from base to tool. Positive external indices
            are one-based; negative indices follow normal Python semantics.
        """
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
        """
        Insert a link at a one-based chain index.

        Parameters
        ----------
        index : int
            One-based insertion index. Negative values follow Python insertion
            semantics relative to the end of the chain.
        value : AbstractLink
            Link to insert.
        """
        self._links.insert(self._normalize_insert_index(index), value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._links!r})"


class KinematicChain(AbstractKinematicChain):
    """
    Represents the kinematic chain of a serial-link, single-arm manipulator,
    implemented as a list of links ordered from the base to the tool-end of the
    manipulator.
    """
    def __init__(
        self,
        links: Sequence[AbstractLink],
        joint_coords: Sequence[float] | None = None,
        base_frame: Frame | None = None,
        tool_frame: Frame | None = None,
        *,
        plot_options: dict[str, Any] | None = None,
        anim_options: dict[str, Any] | None = None,
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
        base_frame: Frame, optional.
            Fixed base frame of the kinematic chain relative to the station
            frame or world frame. If None, the base frame coincides with the
            world frame.
        tool_frame: Frame, optional.
            Tool-center point (TCP) frame or end-effector frame relative to the
            frame of the last link in the kinematic chain. If None, the tool
            frame coincides with the last link frame.
        plot_options: dict[str, Any], optional
            Global plot options used with every call to plot() or plot_async().
        anim_options: dict[str, Any], optional
            Global animation options used with every call to animate() or
            animate_async().
        """
        super().__init__(links)
        self._base_frame: Frame = WREF_FRAME if base_frame is None else base_frame
        self._tool_frame: Frame = WREF_FRAME if tool_frame is None else tool_frame
        self._joint_coords: list[float] = []

        if joint_coords is None:
            self.joint_coords = [0.0] * len(self._links)
        else:
            self.joint_coords = joint_coords

        self._ets: ETS = self._create_ETS()
        self._erobot: ERobot = self._create_ERobot()

        self._create_viewer(plot_options, anim_options)

    @property
    def links(self) -> list[AbstractLink]:
        """
        Returns the list of links in the chain (zero-based index).
        """
        return self._links

    @property
    def base_frame(self) -> Frame:
        """
        Return the pose of the manipulator base frame relative to the world.
        """
        return self._base_frame

    @base_frame.setter
    def base_frame(self, frame: Frame) -> None:
        """
        Sets the manipulator base frame w.r.t. the world frame.
        """
        self._base_frame = frame
        if hasattr(self, "_erobot"):
            self._erobot.base = frame.matrix
            # gravity vector as seen from the manipulator's base frame.
            self._erobot.gravity = (~self._base_frame).transform(Vector(self._erobot.gravity)).array()

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

    def _create_ETS(self) -> ETS:
        ET_list = [et for link in self._links for et in link.ets]
        jcounter = 0
        for et in ET_list:
            if et.isjoint:
                et.jindex = jcounter
                jcounter += 1
        ets = ETS(ET_list)
        return ets

    def _create_ERobot(self) -> ERobot:
        links = [link.rtb_link.copy() for link in self]
        for j, link in enumerate(links):
            link.jindex = j
        erobot = ERobot(
            links,
            base=self.base_frame.matrix,
            tool=self.tool_frame.matrix
        )
        erobot.gravity = (~self._base_frame).transform(Vector(erobot.gravity)).array()
        return erobot

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
        Return the current joint coordinates ordered from base to tool.
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
        self._check_joint_limits(v)
        self._joint_coords = list(v)
        # Assign the joint variables to their respective links.
        for link, value in zip(self._links, self._joint_coords):
            link.variable = value  # -> an angle in degrees will be converted to radians

    def _check_number_of_joint_coords(
        self,
        joint_coords: Sequence[float]
    ) -> Sequence[float]:
        if len(joint_coords) != len(self._links):
            raise ConfigurationError(
                f"The number of joint coordinates ({len(joint_coords)}) "
                f"does not match the number of links ({len(self._links)})."
            )
        return joint_coords

    def _check_joint_limits(
        self,
        joint_coords: Sequence[float]
    ) -> Sequence[float]:
        enumerator = enumerate(zip(self._links, joint_coords), start=1)
        for i, (link, joint_coord) in enumerator:
            q_lim = link.q_lim
            if q_lim is None:
                continue
            lower_limit, upper_limit = q_lim
            if joint_coord < lower_limit or joint_coord > upper_limit:
                raise ConfigurationError(
                    f"Joint coordinate {i} ({joint_coord:.6g}) is outside its "
                    f"mechanical limits [{lower_limit:.6g}, {upper_limit:.6g}]."
                )
        return joint_coords

    def pose(self, index: int) -> Frame:
        """
        Returns the pose (Frame object) of the given link in the chain. The
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
            Specifies the numeric IK-solver to use (see the docstrings of
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

        q0 = np.asarray(ini_guess) if ini_guess is not None else None

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
            return np.asarray(sol.q, dtype=float)
        raise IKSolverError(f"A solution was not found for frame {ee_frame}.")

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
            joint_coords_ = self._check_joint_limits(joint_coords_)
        else:
            joint_coords_ = self.joint_coords
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
        Return the Moore-Penrose pseudo-inverse of the manipulator Jacobian.

        Parameters
        ----------
        joint_coords : Sequence[float], optional
            Joint coordinates at which to evaluate the Jacobian. If omitted,
            the current chain configuration is used.
        ref_frame : RefFrame, default = "world"
            Reference frame used for the spatial velocity represented by the
            Jacobian.
        rcond : float, optional
            Cutoff for small singular values passed to ``numpy.linalg.pinv``.

        Returns
        -------
        NumpyArray
            Pseudo-inverse Jacobian matrix.
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

        q = np.asarray(q, dtype=float)
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
        joint_coords: Sequence[float] | None = None,
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
        joint_coords: Sequence[float], optional
            The coordinates of the joints. If None, the current joint
            configuration of the kinematic chain is taken.
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
        J = self.jacobian(joint_coords, ref_frame=ref_frame)
        V_ee = J @ q_arr
        if len(V_ee) == 3:
            V_ee = [V_ee[0], V_ee[1], 0.0, 0.0, 0.0, V_ee[2]]
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
            External static load exerted on the end-effector of the chain.
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

    def __str__(self) -> str:
        """
        Return the joint-type signature of the kinematic chain.

        Returns
        -------
        str
            Compact representation with ``R`` for revolute links and ``P`` for
            prismatic links, for example ``"(RRP)"``.
        """
        joint_types = "".join(
            "R" if link.is_revolute else "P"
            for link in self._links
        )
        return f"({joint_types})"

    def _create_viewer(
        self,
        plot_options: dict[str, Any] | None,
        anim_options: dict[str, Any] | None
    ) -> None:
        from python_robot.visualisation.kinematic_chain import KinematicChainViewer

        self._viewer = KinematicChainViewer(self)

        if plot_options is not None:
            self._set_plot_options(**plot_options)

        if anim_options is not None:
            self._set_animation_options(**anim_options)

    def _set_plot_options(self, **kwargs) -> None:
        """
        Set default plotting options used by the chain viewer.

        Parameters
        ----------
        **kwargs
            Keyword options forwarded to
            :meth:`KinematicChainViewer.set_plot_options`.
        """
        self._viewer.set_plot_options(**kwargs)

    def plot(self, **kwargs) -> None:
        """
        Plot the current joint configuration of the kinematic chain in 3D.

        Keyword arguments override corresponding defaults supplied through
        ``plot_options`` when the chain was created.

        Parameters
        ----------
        **kwargs
            Optional visualization settings:

            ``extent`` : float, default=4.0
                Half-size of the planar grid.
            ``spacing`` : float, default=1.0
                Distance between adjacent grid lines.
            ``grid_color`` : str, default="lightgray"
                Color of ordinary grid lines.
            ``axis_color`` : str, default="black"
                Color of the principal grid axes.
            ``background_color`` : str, default="white"
                Background color of the render window.
            ``off_screen`` : bool, default=False
                Render without opening an interactive window.
            ``window_size`` : tuple[int, int], default=(800, 600)
                Width and height of the render window in pixels.
            ``world_frame_scale`` : float, default=1.0
                Axis length of the world reference frame.
            ``frame_scale`` : float, default=1.0
                Axis length of each link frame.
            ``line_width`` : float, default=2.0
                Line width of the link-frame axes.
            ``show_label`` : bool, default=True
                Show labels for named link frames.
            ``label_offset`` : float, default=0.1
                Label offset relative to ``frame_scale``.
            ``label_font_size`` : int, default=14
                Font size of link-frame labels.
            ``tool_visual`` : {"auto", "none", "point", "frame", "both"},
                default="auto"
                How the tool-center point is visualized. ``"auto"`` hides an
                identity tool transform and otherwise draws its frame.
            ``tool_frame_scale`` : float, default=1.0
                Axis length of the TCP frame.
            ``tool_frame_line_width`` : float, default=2.0
                Line width of the TCP-frame axes.
            ``tool_point_color`` : str, default="darkorange"
                Color of the TCP marker.
            ``tool_point_size`` : float, default=12.0
                Size of the TCP marker.
            ``tool_link_color`` : str, default="darkorange"
                Color of the segment connecting the final link to the TCP.
            ``tool_link_line_width`` : float, default=3.0
                Line width of the segment connecting the final link to the TCP.
            ``tool_name`` : str or None, default="TCP"
                Optional label for the TCP.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If an unsupported keyword argument is supplied.
        """
        self._viewer.plot(**kwargs)

    async def plot_async(self, **kwargs) -> None:
        """
        Plot the current joint configuration asynchronously in 3D.

        This is the asynchronous counterpart of :meth:`plot`, intended
        primarily for Jupyter notebooks. Call it using ``await``.

        Parameters
        ----------
        **kwargs
            Accepts all keyword arguments documented by :meth:`plot`, plus:

            ``jupyter_backend`` : {"client", "server", "trame"} or None
                PyVista rendering backend used in a Jupyter environment.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If an unsupported keyword argument is supplied.
        """
        await self._viewer.plot_async(**kwargs)

    def _set_animation_options(self, **kwargs) -> None:
        """
        Set default animation options used by the chain viewer.

        Parameters
        ----------
        **kwargs
            Keyword options forwarded to
            :meth:`KinematicChainViewer.set_animation_options`.
        """
        self._viewer.set_animation_options(**kwargs)

    def animate(
        self,
        joint_coords: Sequence[Sequence[float]],
        **kwargs
    ) -> None:
        """
        Animate a sequence of manipulator joint configurations.

        Keyword arguments override corresponding defaults supplied through
        ``anim_options`` when the chain was created.

        Parameters
        ----------
        joint_coords : Sequence[Sequence[float]]
            Sequence of joint-coordinate vectors. Each item is one full
            configuration of the chain, with joints ordered from the base
            toward the tool. The original chain configuration is restored after
            the animation.
        **kwargs
            Optional animation settings:

            ``extent`` : float, default=4.0
                Half-size of the planar grid.
            ``spacing`` : float, default=1.0
                Distance between adjacent grid lines.
            ``grid_color`` : str, default="lightgray"
                Color of ordinary grid lines.
            ``axis_color`` : str, default="black"
                Color of the principal grid axes.
            ``background_color`` : str, default="white"
                Background color of the render window.
            ``off_screen`` : bool, default=False
                Render without opening an interactive window.
            ``window_size`` : tuple[int, int], default=(800, 600)
                Width and height of the render window in pixels.
            ``world_frame_scale`` : float, default=1.0
                Axis length of the world reference frame.
            ``frame_scale`` : float, default=1.0
                Axis length of each link frame.
            ``frame_line_width`` : float, default=2.0
                Line width of the link-frame axes.
            ``link_line_width`` : float, default=5.0
                Line width of the manipulator links.
            ``show_frames`` : bool, default=True
                Draw the local link frames.
            ``frame_names`` : Sequence[str] or None, default=None
                Optional labels for the link frames.
            ``fps`` : int, default=20
                Playback and output frame rate.
            ``step`` : int, default=1
                Animate every ``step``-th configuration.
            ``gif_path`` : str, pathlib.Path or None, default=None
                Optional destination for a GIF recording.
            ``mp4_path`` : str, pathlib.Path or None, default=None
                Optional destination for an MP4 recording.
            ``show`` : bool, default=True
                Show the render window.
            ``interactive_update`` : bool, default=True
                Keep the interactive window responsive during playback.
            ``close_plotter`` : bool, default=False
                Close the plotter after playback or file generation.
            ``show_ee_path`` : bool, default=False
                Draw the path traced by the end effector.
            ``ee_path_color`` : str, default="orange"
                Color of the end-effector path.
            ``ee_path_line_width`` : float, default=3.0
                Line width of the end-effector path.
            ``tool_visual`` : {"auto", "none", "point", "frame", "both"},
                default="auto"
                How the tool-center point is visualized. ``"auto"`` hides an
                identity tool transform and otherwise draws its frame.
            ``tool_frame_scale`` : float or None, default=None
                Axis length of the TCP frame. When omitted, ``0.7 *
                frame_scale`` is used.
            ``tool_frame_line_width`` : float, default=2.0
                Line width of the TCP-frame axes.
            ``tool_point_color`` : str, default="darkorange"
                Color of the TCP marker.
            ``tool_point_size`` : float, default=12.0
                Size of the TCP marker.
            ``tool_link_color`` : str, default="darkorange"
                Color of the segment connecting the final link to the TCP.
            ``tool_link_line_width`` : float, default=3.0
                Line width of the segment connecting the final link to the TCP.
            ``tool_name`` : str or None, default="TCP"
                Optional label for the TCP.
            ``camera_setup`` : Callable[[WorldScene], None] or None, default=None
                Callback that configures the camera before playback starts.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If an unsupported keyword argument is supplied.
        ValueError
            If ``joint_coords`` is empty.
        """
        self._viewer.animate(
            joint_coords=joint_coords,
            **kwargs
        )

    async def animate_async(
        self,
        joint_coords: Sequence[Sequence[float]],
        **kwargs
    ) -> None:
        """
        Animate a sequence of joint configurations asynchronously.

        This is the asynchronous counterpart of :meth:`animate`, intended for
        Jupyter notebooks and other async contexts. Call it using ``await``.

        Parameters
        ----------
        joint_coords : Sequence[Sequence[float]]
            Sequence of complete joint configurations, ordered from base to
            tool. The original chain configuration is restored afterward.
        **kwargs
            Accepts all keyword arguments documented by :meth:`animate`, plus:

            ``jupyter_backend`` : {"client", "server", "trame"} or None
                PyVista rendering backend used in a Jupyter environment.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If an unsupported keyword argument is supplied.
        ValueError
            If ``joint_coords`` is empty.
        """
        await self._viewer.animate_async(
            joint_coords=joint_coords,
            **kwargs
        )
