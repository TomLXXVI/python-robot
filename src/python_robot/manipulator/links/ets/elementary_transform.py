from roboticstoolbox import ET

from python_robot.base.types import AngleUnit

__all__ = ["ElementaryTransform"]


class ElementaryTransform:

    @staticmethod
    def translx(x: float | None) -> ET:
        return ET.tx(eta=x)

    @staticmethod
    def transly(y: float | None) -> ET:
        return ET.ty(eta=y)

    @staticmethod
    def translz(z: float | None) -> ET:
        return ET.tz(eta=z)

    @staticmethod
    def rotx(theta_x: float | None, angle_unit: AngleUnit = "rad") -> ET:
        return ET.Rx(eta=theta_x, unit=angle_unit)

    @staticmethod
    def roty(theta_y: float | None, angle_unit: AngleUnit = "rad") -> ET:
        return ET.Ry(eta=theta_y, unit=angle_unit)

    @staticmethod
    def rotz(theta_z: float | None, angle_unit: AngleUnit = "rad") -> ET:
        return ET.Rz(eta=theta_z, unit=angle_unit)
