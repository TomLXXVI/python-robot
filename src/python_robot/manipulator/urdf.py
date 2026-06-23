"""
Import serial manipulators from URDF or xacro robot descriptions.

URDF models are reduced to a single active serial arm chain and adapted to the
``SerialLinkManipulator`` API. Fixed transforms are folded into active links or
the effective tool frame.
"""

from __future__ import annotations
from typing import Sequence, Any

from copy import deepcopy
from pathlib import Path

import numpy as np
from roboticstoolbox import ERobot, ETS
from roboticstoolbox import Link as RTBLink
from roboticstoolbox.tools import URDF
from spatialmath import SE3

from python_robot.base import Frame

from .exceptions import ConfigurationError
from .links.urdf import URDFLink
from .manipulator import SerialLinkManipulator

__all__ = ["URDFManipulator"]


class URDFManipulator(SerialLinkManipulator):
    """
    Serial manipulator imported from a URDF or xacro file with one active arm.

    The imported kinematic graph may contain fixed side branches, such as
    helper frames or alternate tool frames. However, all active joints must
    lie on one linear arm chain. URDFs with multiple active joint branches or
    disconnected link graphs are rejected because this class is a specialized
    ``SerialLinkManipulator``.
    """

    def __init__(
        self,
        file_path: str | Path,
        joint_coords: Sequence[float] | None = None,
        base_frame: Frame | None = None,
        tool_frame: Frame | None = None,
        tld: str | Path | None = None,
        xacro_tld: str | Path | None = None,
        *,
        plot_options: dict[str, Any] | None = None,
        anim_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Read a URDF/xacro model and build a serial manipulator from it.

        Parameters
        ----------
        file_path: str | Path
            URDF or xacro path passed to Robotics Toolbox. If ``tld`` is not
            provided, Robotics Toolbox interprets this path relative to its
            bundled ``rtbdata/xacro`` directory. If ``tld`` is provided,
            ``file_path`` must be relative to that directory instead.
        joint_coords: Sequence[float], optional
            Initial joint configuration for the active joints.
        base_frame: Frame, optional
            Pose of the manipulator base with respect to the world.
        tool_frame: Frame, optional
            Extra TCP/tool transform appended after the URDF tip.
        tld: str | Path, optional
            Top-level directory used as the base path for ``file_path``. Provide
            this when importing URDF/xacro files from a project-local folder
            rather than RTB's bundled data tree.
        xacro_tld: str | Path, optional
            Path relative to the active top-level directory used by Robotics
            Toolbox while expanding xacro includes and related resources.
        plot_options: dict[str, Any], optional
            Global plot options used with every call to ``plot()`` or
            ``plot_async()``. See superclass ``SerialLinkManipulator``.
        anim_options: dict[str, Any], optional
            Global animation options used with every call to ``animate()`` or
            ``animate_async()``.
        """
        rtb_links, name, urdf_string, urdf_path = ERobot.URDF_read(
            str(file_path),
            tld=None if tld is None else str(tld),
            xacro_tld=None if xacro_tld is None else str(xacro_tld),
        )
        raw_source_erobot = ERobot(
            rtb_links,
            name=name,
            urdf_string=urdf_string,
            urdf_filepath=urdf_path,
        )

        ordered_links = self._ordered_active_branch_links(raw_source_erobot.links)
        links, resolved_tool_frame = self._to_active_links(
            ordered_links,
            user_tool_frame=tool_frame,
        )

        if len(links) == 0:
            raise ConfigurationError(
                "URDF does not contain any active revolute or prismatic joints."
            )

        super().__init__(
            links=links,
            joint_coords=joint_coords,
            base_frame=base_frame,
            tool_frame=resolved_tool_frame,
            plot_options=plot_options,
            anim_options=anim_options
        )

        self._name = name
        self._source_erobot = self._create_source_erobot(
            links,
            name=name,
            urdf_string=urdf_string,
            urdf_path=urdf_path,
            base_frame=self.base_frame,
            tool_frame=self.tool_frame,
        )
        self._urdf_string = urdf_string
        self._urdf_path = Path(urdf_path)
        self._urdf_link_names = tuple(link.name for link in ordered_links)
        self._joint_names = tuple(link.rtb_link._joint_name or "" for link in links)

    @property
    def name(self) -> str:
        """
        Return the robot name parsed from the URDF document.
        """
        return self._name

    @property
    def source_erobot(self) -> ERobot:
        """
        Return a Robotics Toolbox robot for the imported active arm chain.

        Robotics Toolbox dynamics expect the robot link list to be ordered by
        active joints. The raw URDF tree may contain fixed links before or
        beside the active chain, so this robot mirrors the reduced serial arm
        while retaining the source URDF metadata.
        """
        return self._source_erobot

    @staticmethod
    def _create_source_erobot(
        links: Sequence[URDFLink],
        *,
        name: str,
        urdf_string: str,
        urdf_path: Any,
        base_frame: Frame,
        tool_frame: Frame,
    ) -> ERobot:
        """
        Build an RTB robot from the reduced active chain for dynamics calls.
        """
        rtb_links = [link.rtb_link.copy() for link in links]

        for j, link in enumerate(rtb_links):
            link._parent = None
            link._parent_name = None
            link._children = []
            link.jindex = j

        robot = ERobot(
            rtb_links,
            name=name,
            base=base_frame.matrix,
            tool=tool_frame.matrix,
        )
        robot._urdf_string = urdf_string
        robot._urdf_filepath = urdf_path
        return robot

    @property
    def urdf_string(self) -> str:
        """
        Return the resolved URDF XML as text.

        For xacro input, this is the expanded URDF document after xacro
        preprocessing.
        """
        return self._urdf_string

    @property
    def urdf_path(self) -> Path:
        """
        Return the concrete URDF/xacro source path resolved by RTB.
        """
        return self._urdf_path

    @property
    def urdf_link_names(self) -> tuple[str, ...]:
        """
        Return the selected serial-chain URDF link names ordered root-to-tip.
        """
        return self._urdf_link_names

    @property
    def joint_names(self) -> tuple[str, ...]:
        """
        Return the active URDF joint names ordered from base to tool.
        """
        return self._joint_names

    @property
    def urdf(self):
        """
        Return the parsed Robotics Toolbox URDF object.

        Returns
        -------
        URDF
            Parsed URDF document loaded from the resolved URDF XML string.
        """
        return URDF.loadstr(self.urdf_string, self.urdf_path, self.urdf_path.parent)

    @staticmethod
    def _ordered_active_branch_links(links: Sequence[RTBLink]) -> list[RTBLink]:
        """
        Select the single active arm chain and return it root-to-tip.

        Fixed-only side branches are allowed and ignored. Raises
        ``ConfigurationError`` for multiple roots, disconnected graphs, or
        branches where more than one child subtree contains active joints.
        """
        roots = [link for link in links if link.parent is None]
        if len(roots) != 1:
            raise ConfigurationError(
                f"URDF must contain exactly one root link; found {len(roots)}."
            )

        root = roots[0]
        reachable_links = URDFManipulator._reachable_links(root)
        if len(reachable_links) != len(links):
            raise ConfigurationError(
                "URDF contains disconnected links or an invalid kinematic graph."
            )

        ordered = []
        current = root
        while current is not None:
            ordered.append(current)
            children = list(current.children or [])  # type: ignore

            active_children = [
                child for child in children
                if URDFManipulator._subtree_contains_active_joint(child)
            ]
            if len(active_children) > 1:
                raise ConfigurationError(
                    f"URDF contains multiple active branches at link "
                    f"{current.name!r}. URDFManipulator only supports one "
                    "serial active joint chain."
                )

            if len(active_children) == 1:
                current = active_children[0]
                continue

            if len(children) == 1:
                current = children[0]
                continue

            current = None

        return ordered

    @staticmethod
    def _reachable_links(root: RTBLink) -> list[RTBLink]:
        """
        Return all links reachable from the root of the RTB link graph.
        """
        visited: list[RTBLink] = []
        stack = [root]

        while stack:
            link = stack.pop()
            if link in visited:
                continue
            visited.append(link)
            stack.extend(list(link.children or []))

        return visited

    @staticmethod
    def _subtree_contains_active_joint(link: RTBLink) -> bool:
        """
        Return True if this link or one of its descendants has an active joint.
        """
        if link.isjoint:
            return True

        return any(
            URDFManipulator._subtree_contains_active_joint(child)
            for child in list(link.children or [])
        )

    @staticmethod
    def _to_active_links(
        ordered_links: Sequence[RTBLink],
        user_tool_frame: Frame | None,
    ) -> tuple[list[URDFLink], Frame]:
        """
        Convert ordered RTB links into active ``URDFLink`` objects.

        Fixed link transforms are folded into adjacent active joints where
        possible. Remaining fixed tip transforms are returned as the effective
        tool frame, optionally followed by ``user_tool_frame``.
        """
        active_links: list[URDFLink] = []
        pending_fixed_ets = ETS()

        for rtb_link in ordered_links:
            if rtb_link.isjoint:
                link = deepcopy(rtb_link)
                link.ets = pending_fixed_ets * link.ets
                active_links.append(URDFLink(link))
                pending_fixed_ets = ETS()
            else:
                pending_fixed_ets = pending_fixed_ets * rtb_link.ets

        fixed_tip_frame = Frame.from_matrix(
            SE3(URDFManipulator._constant_ets_matrix(pending_fixed_ets), check=False)
        )

        if user_tool_frame is None:
            return active_links, fixed_tip_frame

        return active_links, fixed_tip_frame * user_tool_frame

    @staticmethod
    def _constant_ets_matrix(ets: ETS) -> np.ndarray:
        """
        Evaluate a fixed ETS as a homogeneous transform matrix.

        The helper rejects ETS objects that still contain a joint variable.
        """
        matrix = np.eye(4)
        for et in ets:
            if et.isjoint:
                raise ConfigurationError("Expected a fixed ETS.")
            matrix = matrix @ et.A()
        return matrix
