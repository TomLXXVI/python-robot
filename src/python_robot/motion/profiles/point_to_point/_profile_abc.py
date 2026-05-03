"""
Definition of an abstract base class to represent a straight-line motion
profile.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ....base.types import NumpyArray
from ....charts import LineChart


class MotionProfile(ABC):

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def position(self, t: float | NumpyArray) -> float | NumpyArray:
        pass

    @abstractmethod
    def velocity(self, t: float | NumpyArray) -> float | NumpyArray:
        pass

    @abstractmethod
    def acceleration(self, t: float | NumpyArray) -> float | NumpyArray:
        pass
    
    @abstractmethod
    def position_profile(self) -> tuple[NumpyArray, NumpyArray]:
        pass

    @abstractmethod
    def velocity_profile(self) -> tuple[NumpyArray, NumpyArray]:
        pass

    @abstractmethod
    def acceleration_profile(self) -> tuple[NumpyArray, NumpyArray]:
        pass

    def plot_position_profile(self) -> LineChart:
        return self.plot_position_profiles(self)

    def plot_velocity_profile(self) -> LineChart:
        return self.plot_velocity_profiles(self)

    def plot_acceleration_profile(self) -> LineChart:
        return self.plot_acceleration_profiles(self)

    @staticmethod
    def plot_position_profiles(*mps: MotionProfile) -> LineChart:
        """
        Returns a `LineChart` with the position profiles of the motion profiles.
        Call `show()` on the `LineChart` object to display the plot.
        """
        chart = LineChart()
        for i, mp in enumerate(mps):
            t_arr, s_arr = mp.position_profile()
            chart.add_xy_data(
                label=f"position, MP{i + 1}",
                x1_values=t_arr,
                y1_values=s_arr
            )
        chart.x1.add_title("time, s")
        chart.y1.add_title("position")
        chart.add_legend()
        return chart

    @staticmethod
    def plot_velocity_profiles(*mps: MotionProfile) -> LineChart:
        """
        Returns a `LineChart` with the velocity profiles of the motion profiles.
        Call `show()` on the `LineChart` object to display the plot.
        """
        chart = LineChart()
        for i, mp in enumerate(mps):
            t_arr, v_arr = mp.velocity_profile()
            chart.add_xy_data(
                label=f"velocity, MP{i + 1}",
                x1_values=t_arr,
                y1_values=v_arr
            )
        chart.x1.add_title("time, s")
        chart.y1.add_title("velocity")
        chart.add_legend()
        return chart

    @staticmethod
    def plot_acceleration_profiles(*mps: MotionProfile) -> LineChart:
        """
        Returns a `LineChart` with the acceleration profiles of the motion
        profiles. Call `show()` on the `LineChart` object to display the plot.
        """
        chart = LineChart()
        for i, mp in enumerate(mps):
            t_arr, a_arr = mp.acceleration_profile()
            chart.add_xy_data(
                label=f"acceleration, MP{i + 1}",
                x1_values=t_arr,
                y1_values=a_arr
            )
        chart.x1.add_title("time, s")
        chart.y1.add_title("acceleration")
        chart.add_legend()
        return chart
