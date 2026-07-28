"""Tests for continuous blended Cartesian motion."""

import numpy as np

from python_robot.base import Frame
from python_robot.motion.cartesian_space import (
    BlendedCartesianMotion,
    CartesianMultiLineMotion,
    CartesianTrajectory,
)


def _target_frames() -> list[Frame]:
    """
    Create three target frames for Cartesian motion tests.

    Returns
    -------
    list[Frame]
        Frames with translation and orientation changes.
    """
    return [
        Frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        Frame((1.0, 0.0, 0.0), (0.0, 0.0, 0.4)),
        Frame((1.0, 1.0, 0.0), (0.0, 0.0, 0.8)),
    ]


def test_blended_cartesian_motion_is_continuous_not_sampled() -> None:
    """Verify continuous evaluation without a sampling configuration."""
    motion = BlendedCartesianMotion(
        target_frames=_target_frames(),
        segment_durations=(1.0, 2.0),
        blend_durations=0.1,
    )

    assert motion.duration == 3.0
    np.testing.assert_allclose(motion.target_times, (0.0, 1.0, 3.0))
    np.testing.assert_allclose(
        motion.frame_at(0.0).to_pose_vector(),
        motion.pose_vector_at(0.0),
    )
    assert not hasattr(motion, "num_t_samples")


def test_legacy_multi_line_motion_delegates_to_continuous_motion() -> None:
    """Verify that the former sampled API remains available."""
    motion = CartesianMultiLineMotion(
        target_frames=_target_frames(),
        dt_segments=(1.0, 2.0),
        dt_blends=0.1,
        num_t_samples=11,
    )

    times, poses, velocities, accelerations = motion.motion_samples

    assert times.shape == (11,)
    assert poses.shape == velocities.shape == accelerations.shape == (11, 6)
    np.testing.assert_allclose(
        accelerations[5],
        motion.spatial_acceleration_at(times[5]),
    )


def test_cartesian_trajectory_samples_continuous_motion() -> None:
    """Verify that ``from_motion`` samples all Cartesian motion quantities."""
    motion = BlendedCartesianMotion(
        target_frames=_target_frames(),
        segment_durations=(1.0, 2.0),
        blend_durations=0.1,
    )

    trajectory = CartesianTrajectory.from_motion(motion, sample_count=13)

    assert trajectory.time_samples.shape == (13,)
    assert trajectory.poses.shape == (13, 6)
    assert trajectory.spatial_velocities().shape == (13, 6)
    assert trajectory.spatial_accelerations.shape == (13, 6)
    np.testing.assert_allclose(
        trajectory.poses[6],
        motion.pose_vector_at(trajectory.time_samples[6]),
    )
    np.testing.assert_allclose(
        trajectory.spatial_accelerations[6],
        motion.spatial_acceleration_at(trajectory.time_samples[6]),
    )
    np.testing.assert_allclose(
        trajectory._target_A_arr[1],
        motion.spatial_acceleration_at(motion.target_times[1]),
    )


def test_cartesian_trajectory_factories_are_compatible() -> None:
    """Verify that the new and former target-based factories agree."""
    trajectory = CartesianTrajectory.from_targets(
        target_frames=_target_frames(),
        segment_durations=(1.0, 2.0),
        blend_durations=0.1,
        sample_count=11,
    )
    legacy_trajectory = CartesianTrajectory.create(
        target_frames=_target_frames(),
        dt_segments=(1.0, 2.0),
        dt_blends=0.1,
        num_t_samples=11,
    )

    np.testing.assert_allclose(
        trajectory.time_samples,
        legacy_trajectory.time_samples,
    )
    np.testing.assert_allclose(
        trajectory.poses,
        legacy_trajectory.poses,
    )
    np.testing.assert_allclose(
        trajectory.spatial_velocities(),
        legacy_trajectory.spatial_velocities(),
    )
    np.testing.assert_allclose(
        trajectory.spatial_accelerations,
        legacy_trajectory.spatial_accelerations,
    )


def test_cartesian_trajectory_requires_two_samples() -> None:
    """Verify that a trajectory cannot contain fewer than two samples."""
    motion = BlendedCartesianMotion(
        target_frames=_target_frames(),
        segment_durations=(1.0, 2.0),
        blend_durations=0.1,
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "sample_count must be at least 2",
    ):
        CartesianTrajectory.from_motion(motion, sample_count=1)
