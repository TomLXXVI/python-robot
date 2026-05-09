from typing import Literal

from abc import ABC, abstractmethod

import numpy as np

from roboticstoolbox import ET, ETS

from ..link import (
    AbstractLink,
    AbstractRevoluteLink, AbstractPrismaticLink,
    RTBLink, LinkDynamicParams,
)
from .elementary_transform import LinkETParams, ETFunc


__all__ = ["RevoluteETSLink", "PrismaticETSLink"]


class AbstractETSLink(AbstractLink, ABC):

    def __init__(
        self, 
        axis: Literal["x", "y", "z"],
        ET_params: LinkETParams,
        joint_limits: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams = None,
    ) -> None:
        self._axis = axis
        self._ET_params = ET_params
        self._joint_limits = joint_limits

        link_length = self._calculate_link_length()

        ets = self._create_link_ETS()
        kwargs = {} if self._joint_limits is None else {"qlim": joint_limits}
        rtb_link = RTBLink(ets, **kwargs)

        super().__init__(link_length, rtb_link, dynamics)

    def _create_link_ETS(self) -> ETS:
        ET_list = self._ETS_constant(self._axis, self._ET_params)
        ET_var = self._ET_variable(self._axis, self._ET_params)
        ET_list.append(ET_var)
        return ETS(ET_list)

    def _calculate_link_length(self) -> float:
        dxyz = np.asarray(self._ET_params.translational, dtype=float)
        return float(np.sqrt(np.square(dxyz).sum()))

    @staticmethod
    @abstractmethod
    def _ET_variable(
        axis: Literal["x", "y", "z"],
        ET_params: LinkETParams,
    ) -> ET:
        ...
    
    @staticmethod
    @abstractmethod
    def _ETS_constant(
        axis: Literal["x", "y", "z"],
        ET_params: LinkETParams,
    ) -> list[ET]:
        ...


class RevoluteETSLink(AbstractETSLink, AbstractRevoluteLink):
    
    def __init__(
        self,
        rotation_axis: Literal["x", "y", "z"],
        ET_params: LinkETParams,
        limits_joint_angle: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams = None,
    ) -> None:
        super().__init__(rotation_axis, ET_params, limits_joint_angle, dynamics)

    @staticmethod    
    def _ET_variable(
        rotation_axis: Literal["x", "y", "z"],
        ET_params: LinkETParams,
    ) -> ET:
        theta_x, theta_y, theta_z = ET_params.rotational
        match rotation_axis:
            case "x":
                if theta_x is None:
                    return ETFunc.rotx(theta_x)
                raise ValueError("'theta_x' is not None.")
            case "y":
                if theta_y is None:
                    return ETFunc.roty(theta_y)
                raise ValueError("'theta_y' is not None.")
            case "z":
                if theta_z is None:
                    return ETFunc.rotz(theta_z)
                raise ValueError("'theta_z' is not None.")
            case _:
                raise ValueError(f"Wrong joint axis.")
    
    @staticmethod
    def _ETS_constant(
        rotation_axis: Literal["x", "y", "z"],
        ET_params: LinkETParams,
    ) -> list[ET]:
        theta_x, theta_y, theta_z = ET_params.rotational
        delta_x, delta_y, delta_z = ET_params.translational
        ET_list = []
        if rotation_axis != "x" and theta_x is not None:
            ET_list.append(ETFunc.rotx(theta_x))
        if rotation_axis != "y" and theta_y is not None:
            ET_list.append(ETFunc.roty(theta_y))
        if rotation_axis != "z" and theta_z is not None:
            ET_list.append(ETFunc.rotz(theta_z))
        if delta_x is not None:
            ET_list.append(ETFunc.translx(delta_x))
        if delta_y is not None:
            ET_list.append(ETFunc.transly(delta_y))
        if delta_z is not None:
            ET_list.append(ETFunc.translz(delta_z))
        return ET_list


class PrismaticETSLink(AbstractETSLink, AbstractPrismaticLink):
    
    def __init__(
        self,
        translation_axis: Literal["x", "y", "z"],
        ET_params: LinkETParams,
        limits_link_offset: tuple[float, float] | None = None,
        dynamics: LinkDynamicParams = None,
    ) -> None:
        super().__init__(translation_axis, ET_params, limits_link_offset, dynamics)
    
    @staticmethod
    def _ET_variable(
        translation_axis: Literal["x", "y", "z"],
        ET_params: LinkETParams,
    ) -> ET:
        delta_x, delta_y, delta_z = ET_params.translational
        match translation_axis:
            case "x":
                if delta_x is None:
                    return ETFunc.translx(delta_x)
                raise ValueError("'delta_x' is not None.")
            case "y":
                if delta_y is None:
                    return ETFunc.transly(delta_y)
                raise ValueError("'delta_y' is not None.")
            case "z":
                if delta_z is None:
                    return ETFunc.translz(delta_z)
                raise ValueError("'delta_z' is not None.")
            case _:
                raise ValueError(f"Wrong joint axis.")
    
    @staticmethod
    def _ETS_constant(
        translation_axis: Literal["x", "y", "z"],
        ET_params: LinkETParams,
    ) -> list[ET]:
        delta_x, delta_y, delta_z = ET_params.translational
        theta_x, theta_y, theta_z = ET_params.rotational
        ET_list = []
        if translation_axis != "x" and delta_x is not None:
            ET_list.append(ETFunc.translx(delta_x))
        if translation_axis != "y" and delta_y is not None:
            ET_list.append(ETFunc.transly(delta_y))
        if translation_axis != "z" and delta_z is not None:
            ET_list.append(ETFunc.translz(delta_z))
        if theta_x is not None:
            ET_list.append(ETFunc.rotx(theta_x))
        if theta_y is not None:
            ET_list.append(ETFunc.roty(theta_y))
        if theta_z is not None:
            ET_list.append(ETFunc.rotz(theta_z))
        return ET_list
