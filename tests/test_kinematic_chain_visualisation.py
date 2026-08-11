"""Tests for kinematic-chain base-frame visualization."""

from typing import Any

import numpy as np
import pytest

from python_robot.base import Frame
from python_robot.manipulator import KinematicChain
from python_robot.manipulator.links import RevoluteSDHLink
from python_robot.visualisation import KinematicChainAnimator
from python_robot.visualisation.kinematic_chain import KinematicChainViewer


class _RecordingScene:
    """Record frames and links added by a kinematic-chain viewer."""

    def __init__(self) -> None:
        """Initialize empty frame and link records."""
        self.frames: list[Frame] = []
        self.links: list[dict[str, Any]] = []

    def add_frame(self, frame: Frame, **kwargs: Any) -> None:
        """
        Record a frame added to the scene.

        Parameters
        ----------
        frame : Frame
            Frame supplied by the viewer.
        **kwargs : Any
            Visualization options accepted but not used by this test double.
        """
        self.frames.append(frame)

    def add_link(self, **kwargs: Any) -> None:
        """
        Record a link added to the scene.

        Parameters
        ----------
        **kwargs : Any
            Link endpoints and visualization options.
        """
        self.links.append(kwargs)


def _create_chain() -> KinematicChain:
    """
    Create a displaced one-link chain for visualization tests.

    Returns
    -------
    KinematicChain
        Chain with an unnamed, non-identity base frame.
    """
    return KinematicChain(
        links=[RevoluteSDHLink(1.0, 0.0, 0.0)],
        joint_coords=[0.25],
        base_frame=Frame(
            origin=(0.5, 0.5, 0.0),
            rpy_angles=(0.0, 0.0, 0.5),
        ),
    )


def test_plot_includes_base_frame_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a static plot contains a named copy of the base frame."""
    chain = _create_chain()
    viewer = KinematicChainViewer(chain)
    scene = _RecordingScene()
    monkeypatch.setattr(viewer, "_create_scene", lambda **kwargs: scene)

    viewer._plot(tool_visual="none")

    assert len(scene.frames) == 2
    assert scene.frames[0].name == "B"
    assert chain.base_frame.name is None
    np.testing.assert_allclose(
        scene.frames[0].matrix.A,
        chain.base_frame.matrix.A,
    )


def test_plot_can_hide_base_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that show_base_frame hides only the fixed base frame."""
    chain = _create_chain()
    viewer = KinematicChainViewer(chain)
    scene = _RecordingScene()
    monkeypatch.setattr(viewer, "_create_scene", lambda **kwargs: scene)

    viewer._plot(show_base_frame=False, tool_visual="none")

    assert len(scene.frames) == 1
    np.testing.assert_allclose(
        scene.frames[0].matrix.A,
        chain.link_pose(1).matrix.A,
    )


def test_animator_base_frame_uses_world_relative_pose() -> None:
    """Verify that animations use a named copy of the chain base frame."""
    chain = _create_chain()

    base_frame = KinematicChainAnimator._get_base_frame(chain)

    assert base_frame.name == "B"
    assert chain.base_frame.name is None
    np.testing.assert_allclose(
        base_frame.matrix.A,
        chain.base_frame.matrix.A,
    )
