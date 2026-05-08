# TheTeam

Low-latency robot tracking with a Raspberry Pi + RealSense client and an H100
SAM3 server.

## Run

On the H100 machine:

```bash
uv run python server.py
```

On the Raspberry Pi or camera machine, edit `client.server_url` in `config.yaml`
to use the H100 machine's LAN IP:

```yaml
client:
  server_url: "ws://<ip>:8765"
```

For a RealSense camera, also set:

```yaml
client:
  use_realsense: true
```

Then run:

```bash
uv run python client.py
```

If `pyrealsense2` is not installed on the Pi, install Intel RealSense support
there first. The current pipeline only streams RGB frames; depth is not needed
for the top-down floor homography.

## AprilTag Layout

The client automatically detects all AprilTags in the camera image. No manual
pixel coordinates are needed.

Set the four floor boundary tags in clockwise order in `config.yaml`:

```yaml
apriltags:
  boundary_tag_ids: [10, 11, 12, 13]  # top-left, top-right, bottom-right, bottom-left
field:
  width_m: 3.0
  height_m: 2.0
```

Every non-boundary tag is treated as a robot by default. To restrict or name
robots, set:

```yaml
robots:
  tag_ids:
    21: "robot_1"
    22: "robot_2"
```

If a tag is not mounted at the robot center, set a robot-frame offset:

```yaml
robots:
  center_offsets_m:
    21: [0.08, 0.0]  # forward_m, left_m
```

## Windows

The client opens three live views:

- `AprilTag camera`: camera frame with detected boundary and robot tags.
- `Robot map`: top-down field map with robot position and heading.
- `SAM3 contours`: returned SAM3 contours overlaid on the matching camera frame.
