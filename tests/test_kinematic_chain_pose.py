"""Tests for link and tool poses of a kinematic chain."""

import numpy as np

from python_robot.base import Frame
from python_robot.manipulator import KinematicChain
from python_robot.manipulator.links import RevoluteSDHLink


def _create_chain() -> KinematicChain:
    """
    Create a one-link chain with a non-identity tool transformation.

    Returns
    -------
    KinematicChain
        Chain suitable for checking link and tool poses.
    """
    link = RevoluteSDHLink(
        link_length=1.0,
        twist_angle=0.0,
        link_offset=0.0,
    )
    tool_frame = Frame(
        origin=(0.25, 0.0, 0.0),
        rpy_angles=(0.0, 0.0, 0.0),
    )
    return KinematicChain(
        links=[link],
        joint_coords=[0.4],
        base_frame=Frame(
            origin=(0.5, 0.5, 0.0),
            rpy_angles=(0.0, 0.0, 0.5),
        ),
        tool_frame=tool_frame,
    )


def test_link_pose_is_expressed_in_world_frame() -> None:
    """Verify that link_pose includes the fixed base transformation."""
    chain = _create_chain()

    expected = chain.base_frame.matrix.A @ chain[1].frame.matrix.A

    np.testing.assert_allclose(
        chain.link_pose(1).matrix.A,
        expected,
    )


def test_pose_is_alias_for_link_pose() -> None:
    """Verify that the legacy pose method delegates to link_pose."""
    chain = _create_chain()

    np.testing.assert_allclose(
        chain.pose(-1).matrix.A,
        chain.link_pose(-1).matrix.A,
    )


def test_tool_pose_uses_current_joint_configuration() -> None:
    """Verify that tool_pose combines the current last-link and tool frames."""
    chain = _create_chain()

    expected = chain.link_pose(-1) * chain.tool_frame

    np.testing.assert_allclose(
        chain.tool_pose.matrix.A,
        expected.matrix.A,
    )


def test_fwd_kin_without_arguments_returns_tool_pose() -> None:
    """Verify that state-based forward kinematics returns tool_pose."""
    chain = _create_chain()

    np.testing.assert_allclose(
        chain.fwd_kin().matrix.A,
        chain.tool_pose.matrix.A,
    )
