"""Tests for joint-space trajectory axis selection and units."""

from types import SimpleNamespace

import numpy as np
import pytest

from python_robot.motion.joint_space.trajectory import JointTrajectory


class _ManipulatorStub:
    """Provide the minimal manipulator interface needed by plot tests.

    Parameters
    ----------
    joint_types : str
        Joint type sequence containing ``R`` and ``P`` characters.
    """

    def __init__(self, joint_types: str) -> None:
        self.links = [
            SimpleNamespace(
                is_revolute=joint_type == "R",
                is_prismatic=joint_type == "P",
            )
            for joint_type in joint_types
        ]

    def __len__(self) -> int:
        """Return the number of joints represented by the stub."""
        return len(self.links)


def _create_trajectory(
    joint_types: str = "RP",
    linear_unit: str = "m",
) -> JointTrajectory:
    """Create a small sampled trajectory for plotting tests.

    Parameters
    ----------
    joint_types : str, default = "RP"
        Joint type sequence containing ``R`` and ``P`` characters.
    linear_unit : str, default = "m"
        Label for the trajectory's linear unit.

    Returns
    -------
    JointTrajectory
        Trajectory whose revolute samples equal pi radians and whose
        prismatic samples remain expressed in metres.
    """
    n_joints = len(joint_types)
    samples = np.zeros((2, n_joints))
    for index, joint_type in enumerate(joint_types):
        samples[:, index] = [0.0, np.pi if joint_type == "R" else 0.25]

    return JointTrajectory(
        t_arr=np.array([0.0, 1.0]),
        q_arr=samples.copy(),
        qd_arr=samples.copy(),
        qdd_arr=samples.copy(),
        target_frames=[],
        dt_segments=[1.0],
        manipulator=_ManipulatorStub(joint_types),  # type: ignore[arg-type]
        q_sets=samples.copy(),
        motion_profiles=[],
        angle_unit="deg",
        linear_unit=linear_unit,
    )


@pytest.mark.parametrize(
    ("plot_method", "revolute_label", "prismatic_label", "y1_title", "y2_title"),
    [
        (
            "plot_positions",
            "q1",
            "q2",
            "revolute joint coordinate, deg",
            "prismatic joint coordinate, m",
        ),
        (
            "plot_velocities",
            "qd1",
            "qd2",
            "revolute joint velocity, deg/s",
            "prismatic joint velocity, m/s",
        ),
        (
            "plot_accelerations",
            "qdd1",
            "qdd2",
            "revolute joint acceleration, deg/s²",
            "prismatic joint acceleration, m/s²",
        ),
    ],
)
def test_mixed_joint_plots_use_separate_axes(
    plot_method: str,
    revolute_label: str,
    prismatic_label: str,
    y1_title: str,
    y2_title: str,
) -> None:
    """Verify that mixed joint types use separate, correctly labelled axes."""
    trajectory = _create_trajectory()

    chart = getattr(trajectory, plot_method)()

    assert chart.y2 is not None
    assert chart.y1.axes.get_ylabel() == y1_title
    assert chart.y2.axes.get_ylabel() == y2_title
    np.testing.assert_allclose(
        chart.datasets[revolute_label]["y1_values"], [0.0, 180.0]
    )
    assert chart.datasets[revolute_label]["y2_values"] is None
    assert chart.datasets[prismatic_label]["y1_values"] is None
    np.testing.assert_allclose(
        chart.datasets[prismatic_label]["y2_values"], [0.0, 0.25]
    )


def test_position_targets_use_the_matching_joint_axes() -> None:
    """Verify that target markers use the same axes as their joint traces."""
    chart = _create_trajectory().plot_positions(show_targets=True)

    assert chart.datasets["q1"]["style_props"]["color"] == "C0"
    assert chart.datasets["q1, targets"]["style_props"]["color"] == "C0"
    assert chart.datasets["q2"]["style_props"]["color"] == "C1"
    assert chart.datasets["q2, targets"]["style_props"]["color"] == "C1"
    np.testing.assert_allclose(
        chart.datasets["q1, targets"]["y1_values"], [0.0, 180.0]
    )
    assert chart.datasets["q1, targets"]["y2_values"] is None
    assert chart.datasets["q2, targets"]["y1_values"] is None
    np.testing.assert_allclose(
        chart.datasets["q2, targets"]["y2_values"], [0.0, 0.25]
    )


def test_homogeneous_prismatic_plot_keeps_a_single_axis() -> None:
    """Verify that a homogeneous prismatic plot does not add an empty y-axis."""
    chart = _create_trajectory("P").plot_positions()

    assert chart.y2 is None
    assert chart.y1.axes.get_ylabel() == "prismatic joint coordinate, m"
    np.testing.assert_allclose(chart.datasets["q1"]["y1_values"], [0.0, 0.25])
    assert chart.datasets["q1"]["y2_values"] is None


def test_prismatic_plot_uses_configured_linear_unit_without_conversion() -> None:
    """Verify that the linear unit changes only the displayed axis label."""
    chart = _create_trajectory("P", linear_unit="mm").plot_positions()

    assert chart.y1.axes.get_ylabel() == "prismatic joint coordinate, mm"
    np.testing.assert_allclose(chart.datasets["q1"]["y1_values"], [0.0, 0.25])
