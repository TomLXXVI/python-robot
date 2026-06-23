# python-robot

`python-robot` is a small robotics package for modelling serial manipulators,
working with 3D frames and transformations, planning robot motion, and
visualising robot configurations and trajectories.

The package is aimed at learning, prototyping, and engineering calculations. It
wraps useful concepts from robotics in a Python API: frames, vectors, links,
kinematic chains, predefined robots, joint-space trajectories, Cartesian
trajectories, dynamics helpers, and PyVista/Swift visualisation.

`python-robot` builds on Peter Corke's robotics ecosystem, in particular
[`petercorke/RVC3-python`](https://github.com/petercorke/RVC3-python), and adds
a package-specific layer for robot modelling, motion schemes, and visualisation.

## What It Offers

- 3D geometry primitives: frames, vectors, axes, rotations, translations,
  screws, spatial velocities, spatial accelerations, forces, torques, and
  wrenches.
- Serial manipulator modelling with Denavit-Hartenberg links, Elementary
  Transform Sequence links, URDF-imported links, and dynamic link parameters.
- Forward kinematics, inverse kinematics, Jacobians, singularity checks, static
  wrench-to-joint-force mapping, and rigid-body dynamics helpers.
- Ready-to-use models such as `Planar3R` and `XYZGantry`.
- Joint-space and Cartesian-space motion planning with cubic profiles, linear
  paths with parabolic blends, and 6D pose-vector trajectories.
- Table and plotting helpers for inspecting motion schemes, joint profiles,
  Cartesian poses, velocities, accelerations, and joint loads.
- PyVista-based 3D plotting and animation for frames, links, kinematic chains,
  tool frames, and end-effector paths.
- Optional Swift simulator integration for interactive robot playback.

## Installation

The package requires Python `>=3.10,<3.13`.

From this repository, install the local dependency package first and then
install `python-robot`:

```bash
pip install -e ../automation-motion
pip install -e .
```

Or, from the repository root:

```bash
pip install -e packages/automation-motion
pip install -e packages/python-robot
```

Main runtime dependencies include `numpy`, `spatialmath`, Peter Corke's
Robotics Toolbox stack through
[`rvc3python`](https://github.com/petercorke/RVC3-python), `pyvista`,
`matplotlib`, `ansitable`, and `automation-motion`.

## Quick Start

Create and transform frames:

```python
from python_robot.base import Frame, Translation, Vector

frame_a = Frame(origin=(0.0, 0.0, 0.0), rpy_angles=(0.0, 0.0, 0.0))
move_x = Translation(Vector((0.25, 0.0, 0.0)))

frame_b = frame_a * move_x
print(frame_b)
```

Create a predefined planar robot and solve kinematics:

```python
from python_robot.base import Frame
from python_robot.models import Planar3R

robot = Planar3R(l1=1.0, l2=1.0)

target = Frame(
    origin=(1.0, 0.5, 0.0),
    rpy_angles=(0.0, 0.0, 0.0),
)

q = robot.inv_kin(target)
ee_frame = robot.fwd_kin(q)

print(q)
print(ee_frame)
```

Plan a joint-space motion through Cartesian target frames:

```python
from python_robot.base import Frame
from python_robot.models import Planar3R
from python_robot.motion import JointSpaceScheme

robot = Planar3R(l1=1.0, l2=1.0)

targets = [
    Frame((0.8, 0.2, 0.0), (0.0, 0.0, 0.0)),
    Frame((0.6, 0.8, 0.0), (0.0, 0.0, 0.0)),
    Frame((1.0, 0.4, 0.0), (0.0, 0.0, 0.0)),
]

scheme = JointSpaceScheme.create(
    targets=targets,
    dt_segments=(2.0, 2.0),
    manipulator=robot,
    num_t_samples=100,
)

print(scheme.tables.target_coordinates)
print(scheme.tables.scheme)
```

Visualise a manipulator:

```python
from python_robot.models import Planar3R

robot = Planar3R(
    l1=1.0,
    l2=1.0,
    plot_options={
        "extent": 2.0,
        "spacing": 0.25,
        "world_frame_scale": 0.25,
        "frame_scale": 0.15,
    },
)

robot.joint_coords = (0.3, 0.8, -0.5)
robot.plot()
```

## Main Concepts

### Frames And Transformations

The `python_robot.base` package contains the geometry layer. `Frame` represents
a pose in 3D space. Transformations such as `Translation`, `Rotation`, and
`Screw` can be composed with frames. Vector-like physical quantities such as
`SpatialVelocity`, `SpatialAcceleration`, `Wrench`, `Force`, and `Torque` are
NumPy-compatible value objects with robotics semantics.

### Manipulators And Links

The `python_robot.manipulator` package models serial-link manipulators.
`KinematicChain` and `SerialLinkManipulator` provide forward kinematics,
inverse kinematics, Jacobians, dynamics helpers, plotting, and animation.

Links can be created with:

- standard or modified Denavit-Hartenberg parameters,
- Elementary Transform Sequences,
- URDF/xacro imports,
- legacy Product-of-Exponentials adapters.

### Motion Planning

The `python_robot.motion` package supports:

- single Cartesian straight-line moves,
- multi-segment Cartesian paths with parabolic blends,
- joint-space motion generated from Cartesian targets,
- conversion between joint-space and Cartesian-space schemes,
- sampled position, velocity, acceleration, and dynamics tables,
- motion plots for joint and Cartesian trajectories.

### Visualisation

The `python_robot.visualisation` package provides PyVista scenes, artists, and
animators. It can draw coordinate frames, links, point markers, tool frames,
end-effector paths, and animated kinematic chains. If Swift is installed, the
optional `SwiftSimulator` can play joint-space schemes in Swift.

## Included Demos

The `demos` folder contains notebooks and scripts that demonstrate:

- frame transformations,
- forward kinematics,
- motion schemes,
- UR3 motion planning from a URDF/xacro model,
- Swift playback of a planned UR3 joint-space trajectory.

Start with:

```text
demos/01_frame_transformations.ipynb
demos/02_forward_kinematics.ipynb
demos/03_motion_schemes.ipynb
demos/04_UR3_motion_planning/ur3_01.ipynb
```

## Package Map

```text
python_robot.base
    Frames, transformations, vectors, spatial quantities, and rigid-body
    dynamics helpers.

python_robot.manipulator
    Kinematic chains, serial manipulators, links, URDF import, and exceptions.

python_robot.models
    Predefined manipulators such as Planar3R and XYZGantry.

python_robot.motion
    Joint-space and Cartesian-space motion profiles and schemes.

python_robot.visualisation
    PyVista and optional Swift visualisation tools.

python_robot.utils
    Table formatting, math helpers, and introspection utilities.
```

## Status

This package is currently version `0.1.0`. The API is useful for experiments,
education, and prototyping, but should still be treated as evolving.
