"""Regression tests for Matplotlib configuration side effects."""

import matplotlib as mpl


def test_robotics_import_preserves_matplotlib_settings() -> None:
    """Verify that importing the robotics integration preserves Matplotlib.

    Notes
    -----
    The test records the active configuration, imports the compatibility
    module, and verifies that no process-wide Matplotlib style setting changed.
    The backend is excluded because importing pyplot necessarily resolves
    Matplotlib's initially lazy backend selection.
    """
    settings_before_import = mpl.rcParams.copy()

    import python_robot._roboticstoolbox  # noqa: F401

    changed_settings = [
        key
        for key, value in settings_before_import.items()
        if key != "backend" and mpl.rcParams[key] != value
    ]
    assert changed_settings == []
