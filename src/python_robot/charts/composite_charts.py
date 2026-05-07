from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

from .matplotlibwrapper import LineChart

__all__ = ["CompositeLineChart"]


class CompositeLineChart(ABC):

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
