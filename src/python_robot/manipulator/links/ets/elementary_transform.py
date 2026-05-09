from dataclasses import dataclass
from roboticstoolbox import ET

from python_robot.base.types import AngleUnit

__all__ = ["LinkETParams", "ETFunc"]


@dataclass(frozen=True)
class LinkETParams:
    """
    Dataclass holding all the parameters for creating any of the ET objects.
    """
    delta_x: float | None = None
    delta_y: float | None = None
    delta_z: float | None = None

    theta_x: float | None = None
    theta_y: float | None = None
    theta_z: float | None = None

    @property
    def rotational(self) -> tuple[float | None, ...]:
        return self.theta_x, self.theta_y, self.theta_z

    @property
    def translational(self) -> tuple[float | None, ...]:
        return self.delta_x, self.delta_y, self.delta_z


class ETFunc:
    """
    Elementary transformation functions.
    """
    @staticmethod
    def translx(delta_x: float | None) -> ET:
        return ET.tx(eta=delta_x)

    @staticmethod
    def transly(delta_y: float | None) -> ET:
        return ET.ty(eta=delta_y)

    @staticmethod
    def translz(delta_z: float | None) -> ET:
        return ET.tz(eta=delta_z)

    @staticmethod
    def rotx(
        theta_x: float | None,
        angle_unit: AngleUnit = "rad"
    ) -> ET:
        return ET.Rx(eta=theta_x, unit=angle_unit)

    @staticmethod
    def roty(
        theta_y: float | None,
        angle_unit: AngleUnit = "rad"
    ) -> ET:
        return ET.Ry(eta=theta_y, unit=angle_unit)

    @staticmethod
    def rotz(
        theta_z: float | None,
        angle_unit: AngleUnit = "rad"
    ) -> ET:
        return ET.Rz(eta=theta_z, unit=angle_unit)
