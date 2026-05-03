from abc import ABC, abstractmethod

import numpy as np

from ....base.types import NumpyArray
from ....charts import LineChart

__all__ = ["MultiPointMotionProfile"]


class MultiPointMotionProfile(ABC):

    def __init__(self) -> None:
        self.dt_tot: float = 0.0

    @abstractmethod
    def position(self, t: float) -> float:
        pass

    @abstractmethod
    def velocity(self, t: float) -> float:
        pass

    @abstractmethod
    def acceleration(self, t: float) -> float:
        pass

    def position_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        t_arr = np.linspace(0, self.dt_tot, n_samples)
        s_arr = np.array([self.position(t) for t in t_arr])
        return t_arr, s_arr

    def velocity_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        t_arr = np.linspace(0, self.dt_tot, n_samples)
        v_arr = np.array([self.velocity(t) for t in t_arr])
        return t_arr, v_arr

    def acceleration_profile(self, n_samples: int = 100) -> tuple[NumpyArray, NumpyArray]:
        t_arr = np.linspace(0, self.dt_tot, n_samples)
        a_arr = np.array([self.acceleration(t) for t in t_arr])
        return t_arr, a_arr

    def plot_position_profile(self, n_samples: int = 100) -> LineChart:
        """
        Returns a LineChart object with the position profile of the motion path.
        Call `show()` on the `LineChart` object to display the plot.
        """
        chart = LineChart()
        t_arr, s_arr = self.position_profile(n_samples)
        chart.add_xy_data(
            label="position",
            x1_values=t_arr,
            y1_values=s_arr
        )
        chart.x1.add_title("time, s")
        chart.y1.add_title("position")
        chart.add_legend()
        return chart

    def plot_velocity_profile(self, n_samples: int = 100) -> LineChart:
        """
        Returns a LineChart object with the velocity profile of the motion path.
        Call `show()` on the `LineChart` object to display the plot.
        """
        chart = LineChart()
        t_arr, v_arr = self.velocity_profile(n_samples)
        chart.add_xy_data(
            label="velocity",
            x1_values=t_arr,
            y1_values=v_arr
        )
        chart.x1.add_title("time, s")
        chart.y1.add_title("velocity")
        chart.add_legend()
        return chart

    def plot_acceleration_profile(self, n_samples: int = 100) -> LineChart:
        """
        Returns a LineChart object with the acceleration profile of the motion
        path. Call `show()` on the `LineChart` object to display the plot.
        """
        chart = LineChart()
        t_arr, a_arr = self.acceleration_profile(n_samples)
        chart.add_xy_data(
            label="acceleration",
            x1_values=t_arr,
            y1_values=a_arr
        )
        chart.x1.add_title("time, s")
        chart.y1.add_title("acceleration")
        chart.add_legend()
        return chart
