# KinematicChain visualisation options

This cheat sheet lists the keyword arguments accepted by:

- `KinematicChain.plot(**kwargs)`
- `await KinematicChain.plot_async(**kwargs)`
- `KinematicChain.animate(joint_coords, **kwargs)`
- `await KinematicChain.animate_async(joint_coords, **kwargs)`

The same options can be supplied as persistent defaults when constructing the
chain:

```python
chain = KinematicChain(
    links,
    plot_options={
        "extent": 3.0,
        "frame_scale": 0.25,
        "tool_visual": "frame",
    },
    anim_options={
        "fps": 30,
        "show_frames": False,
        "show_ee_path": True,
    },
)
```

Options passed directly to a method take precedence over constructor defaults:

```python
# Uses extent=3.0 from plot_options, but overrides frame_scale.
chain.plot(frame_scale=0.5)
```

Unknown options raise `TypeError`.

## Reference frames

The viewer draws the following coordinate frames:

- the world frame `W`;
- the fixed manipulator base frame;
- the link frames;
- optionally, the tool/TCP frame.

The base, link, and tool poses are all expressed with respect to the world
frame. If the manipulator base frame has no name, the viewer displays it with
the label `B`. The chain's original `base_frame` object is not modified.

## Scene options

These options are available for plotting and animation.

| Option | Type | Default | Description |
|---|---|---:|---|
| `extent` | `float` | `4.0` | Half-size of the planar grid. |
| `spacing` | `float` | `1.0` | Distance between adjacent grid lines. |
| `grid_color` | `str` | `"lightgray"` | Color of ordinary grid lines. |
| `axis_color` | `str` | `"black"` | Color of the principal grid axes. |
| `background_color` | `str` | `"white"` | Background color of the render window. |
| `off_screen` | `bool` | `False` | Render without opening an interactive window. |
| `window_size` | `tuple[int, int]` | `(800, 600)` | Render-window size in pixels. |
| `world_frame_scale` | `float` | `1.0` | Axis length of the world reference frame. |

## Plot options

These options are accepted by `plot()` and `plot_async()`, in addition to the
scene options above.

### Base- and link-frame options

| Option | Type | Default | Description |
|---|---|---:|---|
| `show_base_frame` | `bool` | `True` | Draw the fixed manipulator base frame. |
| `frame_scale` | `float` | `1.0` | Axis length of the base and link frames. |
| `line_width` | `float` | `2.0` | Line width of the base- and link-frame axes. |
| `show_label` | `bool` | `True` | Show labels for the base and named link frames. |
| `label_offset` | `float` | `0.1` | Label offset relative to `frame_scale`. |
| `label_font_size` | `int` | `14` | Font size of base- and link-frame labels. |

### Tool/TCP options

| Option | Type | Default | Description |
|---|---|---:|---|
| `tool_visual` | `"auto"`, `"none"`, `"point"`, `"frame"`, or `"both"` | `"auto"` | Select how the tool/TCP is drawn. `auto` hides an identity tool transform and otherwise draws its frame. |
| `tool_frame_scale` | `float` | `1.0` | Axis length of the TCP frame. |
| `tool_frame_line_width` | `float` | `2.0` | Line width of the TCP-frame axes. |
| `tool_point_color` | `str` | `"darkorange"` | Color of the TCP marker. |
| `tool_point_size` | `float` | `12.0` | Size of the TCP marker. |
| `tool_link_color` | `str` | `"darkorange"` | Color of the segment from the final link frame to the TCP. |
| `tool_link_line_width` | `float` | `3.0` | Width of the segment from the final link frame to the TCP. |
| `tool_name` | `str \| None` | `"TCP"` | Optional label for the TCP point or frame. |

### Async plot option

| Option | Type | Default | Description |
|---|---|---:|---|
| `jupyter_backend` | `"client"`, `"server"`, `"trame"`, or `None` | `None` | PyVista backend used by `plot_async()` in Jupyter. |

`jupyter_backend` must be passed directly to `plot_async()`. It is not accepted
inside constructor `plot_options`.

## Animation options

These options are accepted by `animate()` and `animate_async()`, in addition to
the scene options above.

### Robot geometry

| Option | Type | Default | Description |
|---|---|---:|---|
| `frame_scale` | `float` | `1.0` | Axis length of the base and link frames. |
| `frame_line_width` | `float` | `2.0` | Line width of the base- and link-frame axes. |
| `link_line_width` | `float` | `5.0` | Line width of the manipulator links. |
| `show_frames` | `bool` | `True` | Draw the local link frames. |
| `show_base_frame` | `bool` | `True` | Draw the fixed manipulator base frame independently of `show_frames`. |
| `frame_names` | `Sequence[str] \| None` | `None` | Optional labels for the link frames, ordered from base to tool. |

### Playback and output

| Option | Type | Default | Description |
|---|---|---:|---|
| `fps` | `int` | `20` | Playback rate and output frame rate. |
| `step` | `int` | `1` | Animate every `step`-th joint configuration. |
| `gif_path` | `str \| Path \| None` | `None` | Optional destination for a GIF recording. |
| `mp4_path` | `str \| Path \| None` | `None` | Optional destination for an MP4 recording. |
| `show` | `bool` | `True` | Show the render window during playback. |
| `interactive_update` | `bool` | `True` | Keep the interactive window responsive during playback. |
| `close_plotter` | `bool` | `False` | Close the plotter after playback or file generation. |

`gif_path` and `mp4_path` are mutually exclusive.

### End-effector path

| Option | Type | Default | Description |
|---|---|---:|---|
| `show_ee_path` | `bool` | `False` | Draw the path traced by the final link-frame origin. |
| `ee_path_color` | `str` | `"orange"` | Color of the path. |
| `ee_path_line_width` | `float` | `3.0` | Line width of the path. |

### Tool/TCP

| Option | Type | Default | Description |
|---|---|---:|---|
| `tool_visual` | `"auto"`, `"none"`, `"point"`, `"frame"`, or `"both"` | `"auto"` | Select how the tool/TCP is drawn. `auto` hides an identity tool transform and otherwise draws its frame. |
| `tool_frame_scale` | `float \| None` | `None` | Axis length of the TCP frame. `None` uses `0.7 * frame_scale`. |
| `tool_frame_line_width` | `float` | `2.0` | Line width of the TCP-frame axes. |
| `tool_point_color` | `str` | `"darkorange"` | Color of the TCP marker. |
| `tool_point_size` | `float` | `12.0` | Size of the TCP marker. |
| `tool_link_color` | `str` | `"darkorange"` | Color of the segment from the final link frame to the TCP. |
| `tool_link_line_width` | `float` | `3.0` | Width of the segment from the final link frame to the TCP. |
| `tool_name` | `str \| None` | `"TCP"` | Optional label for the TCP point or frame. |

### Camera and Jupyter

| Option | Type | Default | Description |
|---|---|---:|---|
| `camera_setup` | `Callable[[WorldScene], None] \| None` | `None` | Callback that configures the camera after creating the robot geometry and before playback starts. |
| `jupyter_backend` | `"client"`, `"server"`, `"trame"`, or `None` | `None` | PyVista backend used in Jupyter. |

Unlike the plot equivalent, animation `jupyter_backend` may also be supplied
through constructor `anim_options`.

## Complete examples

```python
chain.plot(
    extent=2.5,
    spacing=0.25,
    background_color="white",
    world_frame_scale=0.3,
    show_base_frame=True,
    frame_scale=0.2,
    line_width=3.0,
    tool_visual="both",
)
```

```python
chain.animate(
    joint_coords,
    fps=30,
    step=2,
    frame_scale=0.2,
    show_base_frame=True,
    link_line_width=6.0,
    show_ee_path=True,
    ee_path_color="orange",
    tool_visual="frame",
    gif_path="robot_motion.gif",
    show=False,
)
```
