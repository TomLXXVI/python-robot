"""
Point-to-point motion of a single rigid body.
"""
from dataclasses import dataclass, field, asdict
from abc import ABC

from ...base.types import NumpyArray
from ...base import Frame
from ..profiles_1D.point_to_point import MotionProfile
from ..profiles_1D.point_to_point import (
    TriPhaseMotionProfile,
    PolyMotionProfile,
    TrapezoidalProfile,
    SCurvedProfile,
    CubicMotionProfile,
    QuinticMotionProfile
)

__all__ = [
    "Trapezoidal",
    "SCurved",
    "Cubic",
    "Quintic",
    "CartesianStraightLineMotion"
]


@dataclass(frozen=True)
class MPParams(ABC):
    """
    Abstract base class for holding motion profile parameters.
    """
    pass


@dataclass(frozen=True)
class TriPhaseMPParams(MPParams, ABC):
    """
    Abstract child class for holding tri-phase motion profile parameters.

    Attributes
    ----------
    v_max: float
        Maximum allowable velocity of the motion.
    a_max: float
        Maximum allowable acceleration of the motion.
    v_i: float, default = 0.0
        Initial velocity of the motion.
    v_f: float, default = 0.0
        Final velocity of the motion.
    dt_tot: float, optional
        Total duration of the motion.
    """
    v_max: float
    a_max: float
    v_i: float = 0.0
    v_f: float = 0.0
    dt_tot: float | None = None
    ds_tot: float = field(init=False, default=1.0)


@dataclass(frozen=True)
class Trapezoidal(TriPhaseMPParams):
    """
    Concrete dataclass for holding trapezoidal motion profile parameters.
    """
    pass


@dataclass(frozen=True)
class SCurved(TriPhaseMPParams):
    """
    Concrete dataclass for holding S-curved motion profile parameters.
    """
    pass


@dataclass(frozen=True)
class PolyMPParams(MPParams, ABC):
    """
    Abstract child class for holding polynomial motion profile parameters.

    Attributes
    ----------
    dt_tot: float | None
        Time duration of the motion.
    v_i: float, default = 0.0
        Initial velocity of the motion.
    v_f: float, default = 0.0
        Final velocity of the motion.
    a_i: float, default = 0.0
        Initial acceleration of the motion.
    a_f: float, default = 0.0
        Final acceleration of the motion.
    """
    dt_tot: float
    v_i: float = 0.0
    v_f: float = 0.0
    a_i: float = 0.0
    a_f: float = 0.0
    s_i: float = field(init=False, default=0.0)
    s_f: float = field(init=False, default=1.0)


@dataclass(frozen=True)
class Cubic(PolyMPParams):
    """
    Concrete dataclass for holding cubic motion profile parameters.
    """
    pass


@dataclass(frozen=True)
class Quintic(PolyMPParams):
    """
    Concrete dataclass for holding quintic motion profile parameters.
    """
    pass


@dataclass
class CartesianStraightLineMotion:
    """
    Class for finding a smooth straight-line path between two 3D poses of a 
    frame which involves a change in position as well as in orientation.

    Attributes
    ----------
    pose_ini: Frame
        The initial pose of the frame.
    pose_fin: Frame
        The final pose of the frame.
    time_scaling: MPParams
        Parameters that define the motion profile used for time-scaling the
        straight-line path.
    """
    pose_ini: Frame
    pose_fin: Frame
    time_scaling: MPParams

    def __post_init__(self):
        # Create the MotionProfile object to be used for time-scaling the path.
        if isinstance(self.time_scaling, Trapezoidal):
            self._mp = self._create_triphase_profile(self.time_scaling)
        elif isinstance(self.time_scaling, SCurved):
            self._mp = self._create_triphase_profile(self.time_scaling)
        elif isinstance(self.time_scaling, Cubic):
            self._mp = self._create_poly_profile(self.time_scaling)
        elif isinstance(self.time_scaling, Quintic):
            self._mp = self._create_poly_profile(self.time_scaling)
        else:
            raise NotImplementedError(
                f"Motion_profile type {type(self.time_scaling)} is not implemented."
            )

    @staticmethod
    def _create_triphase_profile(params: TriPhaseMPParams) -> TriPhaseMotionProfile:
        if isinstance(params, Trapezoidal):
            mpt = TrapezoidalProfile
        elif isinstance(params, SCurved):
            mpt = SCurvedProfile
        else:
            raise NotImplementedError(
                f"Motion_profile type {type(params)} is not implemented."
            )
        if params.v_max is None or params.a_max is None:
            raise ValueError(
                f"To create a {mpt.__name__} motion profile, both "
                f"parameters 'v_max' and 'a_max' are required."
            )
        return mpt(**asdict(params))

    @staticmethod
    def _create_poly_profile(params: PolyMPParams) -> PolyMotionProfile:
        if isinstance(params, Cubic):
            mpt = CubicMotionProfile
        elif isinstance(params, Quintic):
            mpt = QuinticMotionProfile
        else:
            raise NotImplementedError(
                f"Motion_profile type {type(params)} is not implemented."
            )
        if params.dt_tot is None:
            raise ValueError(
                f"To create a {mpt.__name__} motion profile, "
                f"paramter 'dt_tot' is required."
            )
        return mpt(**asdict(params))

    def trajectory(self) -> tuple[NumpyArray, list[Frame]]:
        """
        Returns the trajectory between the initial and final pose of the frame
        in 3D space.

        Returns
        -------
        t_arr: NumpyArray
            Numpy array with time moments.
        frames: list[Frame]
            List of frame poses along the trajectory at the time instants in
            `t_arr`.
        """
        t_arr, s_arr = self._mp.position_profile()
        traj = self.pose_ini.matrix.interp(self.pose_fin.matrix, s_arr)  # type: ignore
        frames = [Frame.from_matrix(se3) for se3 in traj]
        return t_arr, frames

    @property
    def motion_profile(self) -> MotionProfile:
        """
        Returns the underlying motion profile that controls the time-scaling
        of the straight-line path.
        """
        return self._mp
