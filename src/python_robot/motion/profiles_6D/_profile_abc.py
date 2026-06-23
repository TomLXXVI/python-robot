"""
Abstract base class for sampled vector-valued multipoint motion profiles.
"""

from abc import ABC, abstractmethod

import numpy as np

from automation_motion.base.types import NumpyArray
from automation_motion.charts import CompositeLineChart

__all__ = ["MultiPointVectorMotionProfile"]


class MultiPointVectorMotionProfile(ABC):
    """
    Abstract interface for vector-valued multipoint motion profiles.

    Subclasses define pose, velocity, acceleration, spatial velocity, and
    spatial acceleration as time-functions. The helper methods sample those
    functions and build charts for inspecting the resulting motion.
    """

    def __init__(self) -> None:
        """
        Initialize the total duration of the motion profile.
        """
        self.dt_tot: float = 0.0

    @abstractmethod
    def pose(self, t: float) -> NumpyArray:
        """
        Return the pose vector at time ``t``.
        """
        pass

    @abstractmethod
    def velocity(self, t: float) -> NumpyArray:
        """
        Return the pose-vector derivative at time ``t``.
        """
        pass

    @abstractmethod
    def acceleration(self, t: float) -> NumpyArray:
        """
        Return the pose-vector second derivative at time ``t``.
        """
        pass

    @abstractmethod
    def spatial_velocity(self, t: float) -> NumpyArray:
        """
        Return the spatial velocity at time ``t``.
        """
        pass

    @abstractmethod
    def spatial_acceleration(self, t: float) -> NumpyArray:
        """
        Return the spatial acceleration at time ``t``.
        """
        pass

    def position_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        """
        Return sampled pose-vector positions over the full motion duration.
        """
        t_arr = np.linspace(0.0, self.dt_tot, n_samples)
        p_arr = np.array([self.pose(t) for t in t_arr])
        return t_arr, p_arr

    def velocity_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        """
        Return sampled pose-vector velocities over the full motion duration.
        """
        t_arr = np.linspace(0.0, self.dt_tot, n_samples)
        v_arr = np.array([self.velocity(t) for t in t_arr])
        return t_arr, v_arr

    def acceleration_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        """
        Return sampled pose-vector accelerations over the full motion duration.
        """
        t_arr = np.linspace(0.0, self.dt_tot, n_samples)
        a_arr = np.array([self.acceleration(t) for t in t_arr])
        return t_arr, a_arr

    def spatial_velocity_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        """
        Returns the sampled spatial velocity profile of the vector-valued path.
        """
        t_arr = np.linspace(0.0, self.dt_tot, n_samples)
        V_arr = np.array([self.spatial_velocity(t) for t in t_arr])
        return t_arr, V_arr

    def spatial_acceleration_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        """
        Returns the sampled spatial acceleration profile of the vector-valued path.
        """
        t_arr = np.linspace(0.0, self.dt_tot, n_samples)
        A_arr = np.array([self.spatial_acceleration(t) for t in t_arr])
        return t_arr, A_arr

    def plot_position_profile(self, n_samples: int = 100):
        """
        Returns a wrapper object with two LineChart objects showing the
        translation and orientation position profiles of the motion path.
        Call method show() on this wrapper object to display the plots.
        """
        parent = self
        class PlotPositionProfile(CompositeLineChart):
            def add_data(self) -> None:
                t_arr, p_arr = parent.position_profile(n_samples)
                self.top_chart.add_xy_data(
                    label="x",
                    x1_values=t_arr,
                    y1_values=p_arr[:, 0]
                )
                self.top_chart.add_xy_data(
                    label="y",
                    x1_values=t_arr,
                    y1_values=p_arr[:, 1]
                )
                self.top_chart.add_xy_data(
                    label="z",
                    x1_values=t_arr,
                    y1_values=p_arr[:, 2]
                )
                self.bottom_chart.add_xy_data(
                    label="rx",
                    x1_values=t_arr,
                    y1_values=p_arr[:, 3]
                )
                self.bottom_chart.add_xy_data(
                    label="ry",
                    x1_values=t_arr,
                    y1_values=p_arr[:, 4]
                )
                self.bottom_chart.add_xy_data(
                    label="rz",
                    x1_values=t_arr,
                    y1_values=p_arr[:, 5]
                )
                self.bottom_chart.x1.add_title("time, s")
                self.top_chart.y1.add_title("translation")
                self.bottom_chart.y1.add_title("orientation")
                self.top_chart.add_legend(anchor="lower center", position=(0.5, 1.05), columns=3)
                self.bottom_chart.add_legend(anchor="upper center", position=(0.5, -0.25), columns=3)

        return PlotPositionProfile()

    def plot_velocity_profile(self, n_samples: int = 100):
        """
        Returns a wrapper object with two LineChart objects showing the linear
        and angular velocity profiles of the motion path.
        Call method show() on this wrapper object to display the plots.
        """
        parent = self
        class PlotVelocityProfile(CompositeLineChart):
            def add_data(self) -> None:
                t_arr, v_arr = parent.velocity_profile(n_samples)
                self.top_chart.add_xy_data(
                    label="x_dot",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 0]
                )
                self.top_chart.add_xy_data(
                    label="y_dot",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 1]
                )
                self.top_chart.add_xy_data(
                    label="z_dot",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 2]
                )
                self.bottom_chart.add_xy_data(
                    label="rx_dot",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 3]
                )
                self.bottom_chart.add_xy_data(
                    label="ry_dot",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 4]
                )
                self.bottom_chart.add_xy_data(
                    label="rz_dot",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 5]
                )
                self.bottom_chart.x1.add_title("time, s")
                self.top_chart.y1.add_title("linear velocity")
                self.bottom_chart.y1.add_title("angular velocity")
                self.top_chart.add_legend(anchor="lower center", position=(0.5, 1.05), columns=3)
                self.bottom_chart.add_legend(anchor="upper center", position=(0.5, -0.25), columns=3)

        return PlotVelocityProfile()

    def plot_acceleration_profile(self, n_samples: int = 100):
        """
        Returns a wrapper object with two LineChart objects showing the linear
        and angular acceleration profiles of the motion path. Call method show()
        on this wrapper object to display the plots.
        """
        parent = self
        class PlotAccelerationProfile(CompositeLineChart):
            def add_data(self) -> None:
                t_arr, a_arr = parent.acceleration_profile(n_samples)
                self.top_chart.add_xy_data(
                    label="x_ddot",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 0]
                )
                self.top_chart.add_xy_data(
                    label="y_ddot",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 1]
                )
                self.top_chart.add_xy_data(
                    label="z_ddot",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 2]
                )
                self.bottom_chart.add_xy_data(
                    label="rx_ddot",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 3]
                )
                self.bottom_chart.add_xy_data(
                    label="ry_ddot",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 4]
                )
                self.bottom_chart.add_xy_data(
                    label="rz_ddot",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 5]
                )
                self.bottom_chart.x1.add_title("time, s")
                self.top_chart.y1.add_title("linear acceleration")
                self.bottom_chart.y1.add_title("angular acceleration")
                self.top_chart.add_legend(anchor="lower center", position=(0.5, 1.05), columns=3)
                self.bottom_chart.add_legend(anchor="upper center", position=(0.5, -0.25), columns=3)

        return PlotAccelerationProfile()

    def plot_spatial_velocity_profile(self, n_samples: int = 100):
        """
        Returns a wrapper object with two LineChart objects showing the linear
        and angular velocity profiles of the motion path.
        Call method show() on this wrapper object to display the plots.
        """
        parent = self
        class PlotSpatialVelocityProfile(CompositeLineChart):
            def add_data(self) -> None:
                t_arr, v_arr = parent.spatial_velocity_profile(n_samples)
                self.top_chart.add_xy_data(
                    label="v_x",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 0]
                )
                self.top_chart.add_xy_data(
                    label="v_y",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 1]
                )
                self.top_chart.add_xy_data(
                    label="v_z",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 2]
                )
                self.bottom_chart.add_xy_data(
                    label="w_x",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 3]
                )
                self.bottom_chart.add_xy_data(
                    label="w_y",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 4]
                )
                self.bottom_chart.add_xy_data(
                    label="w_z",
                    x1_values=t_arr,
                    y1_values=v_arr[:, 5]
                )
                self.bottom_chart.x1.add_title("time, s")
                self.top_chart.y1.add_title("linear velocity")
                self.bottom_chart.y1.add_title("angular velocity")
                self.top_chart.add_legend(anchor="lower center", position=(0.5, 1.05), columns=3)
                self.bottom_chart.add_legend(anchor="upper center", position=(0.5, -0.25), columns=3)

        return PlotSpatialVelocityProfile()

    def plot_spatial_acceleration_profile(self, n_samples: int = 100):
        """
        Returns a wrapper object with two LineChart objects showing the linear
        and angular acceleration profiles of the motion path.
        Call method show() on this wrapper object to display the plots.
        """
        parent = self
        class PlotSpatialAccelerationProfile(CompositeLineChart):
            def add_data(self) -> None:
                t_arr, a_arr = parent.spatial_acceleration_profile(n_samples)
                self.top_chart.add_xy_data(
                    label="a_x",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 0]
                )
                self.top_chart.add_xy_data(
                    label="a_y",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 1]
                )
                self.top_chart.add_xy_data(
                    label="a_z",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 2]
                )
                self.bottom_chart.add_xy_data(
                    label="alpha_x",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 3]
                )
                self.bottom_chart.add_xy_data(
                    label="alpha_y",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 4]
                )
                self.bottom_chart.add_xy_data(
                    label="alpha_z",
                    x1_values=t_arr,
                    y1_values=a_arr[:, 5]
                )
                self.bottom_chart.x1.add_title("time, s")
                self.top_chart.y1.add_title("linear acceleration")
                self.bottom_chart.y1.add_title("angular acceleration")
                self.top_chart.add_legend(anchor="lower center", position=(0.5, 1.05), columns=3)
                self.bottom_chart.add_legend(anchor="upper center", position=(0.5, -0.25), columns=3)

        return PlotSpatialAccelerationProfile()
