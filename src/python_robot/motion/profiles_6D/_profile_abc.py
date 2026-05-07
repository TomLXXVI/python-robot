from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from ...base.types import NumpyArray
from ...charts import LineChart

__all__ = ["MultiPointVectorMotionProfile"]


class MultiPointVectorMotionProfile(ABC):

    def __init__(self) -> None:
        self.dt_tot: float = 0.0

    @abstractmethod
    def position(self, t: float) -> NumpyArray:
        pass

    @abstractmethod
    def velocity(self, t: float) -> NumpyArray:
        pass

    @abstractmethod
    def acceleration(self, t: float) -> NumpyArray:
        pass

    @abstractmethod
    def spatial_velocity(self, t: float) -> NumpyArray:
        pass

    @abstractmethod
    def spatial_acceleration(self, t: float) -> NumpyArray:
        pass

    def position_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        """
        Returns the sampled position profile of the vector-valued path.
        """
        t_arr = np.linspace(0.0, self.dt_tot, n_samples)
        p_arr = np.array([self.position(t) for t in t_arr])
        return t_arr, p_arr

    def velocity_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        """
        Returns the sampled velocity profile of the vector-valued path.
        """
        t_arr = np.linspace(0.0, self.dt_tot, n_samples)
        v_arr = np.array([self.velocity(t) for t in t_arr])
        return t_arr, v_arr

    def acceleration_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        """
        Returns the sampled acceleration profile of the vector-valued path.
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
        class PlotPositionProfile(_CompositeLineChart):
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
        class PlotVelocityProfile(_CompositeLineChart):
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
        class PlotAccelerationProfile(_CompositeLineChart):
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
                    label="ry_dot",
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
        class PlotSpatialVelocityProfile(_CompositeLineChart):
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
        class PlotSpatialAccelerationProfile(_CompositeLineChart):
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


class _CompositeLineChart(ABC):
    
    def __init__(self):
        figure, axes = plt.subplots(
            2,
            1,
            sharex=True,  # type: ignore
            layout="constrained"
        )
        self.top_chart = LineChart(constructs=(figure, axes[0]))
        self.bottom_chart = LineChart(constructs=(figure, axes[1]))
        # Add drawing data to the top and bottom chart
        self.add_data()

    def show(self, with_grid: bool = True) -> None:
        self.top_chart.draw(with_grid)
        self.bottom_chart.draw(with_grid)
        plt.show()

    def save(
        self,
        name: str,
        location: str | None = None,
        fmt: str = 'png',
        with_grid: bool = True
    ) -> None:
        self.top_chart.draw(with_grid)
        self.bottom_chart.save(name, location, fmt, with_grid)
    
    @abstractmethod
    def add_data(self) -> None:
        """
        Implement this method to add data to the top and the bottom line-chart.
        """
        pass
