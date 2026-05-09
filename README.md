# TheTeam

Client-centered multirobot tracking for a private robot LAN.

The camera client owns the live feed, computes AprilTag/global poses, sends
frames to the SAM3 server over WebRTC, receives contour-only segmentation
results, renders the dashboard, and publishes per-robot ROS2 topics. The SAM3
server never joins ROS2. Robots never use WebRTC.

## Project Layout

```text
client/
  app.py              # camera loop, WebRTC client, dashboard, ROS2 gateway
  dashboard.py        # browser dashboard and MJPEG stream
  ros2_gateway.py     # per-robot ROS2 publishers
common/
  messages.py         # shared packet/dataclass schemas
  webrtc_transport.py # WebRTC signaling and data channels
robot/
  app.py              # per-robot ROS2 subscriber
server/
  app.py              # WebRTC signaling server and SAM3 session loop
  segmentation.py     # SAM3 loading/inference and mask-to-contour conversion
vision/
  tracking.py         # AprilTags, homography, robot pose estimates
  visualization.py    # annotated camera feed renderer

config.py             # loads config.yaml
config.yaml           # all runtime/network configuration
```

## Install

```bash
uv sync
```

ROS2 Python packages such as `rclpy`, `geometry_msgs`, and `std_msgs` are
expected to come from your ROS2 installation, not PyPI.

## Configure

All runtime and network settings live in `config.yaml`.

Important sections:

- `server.sam3`: model, device, dtype, contour filtering, inference worker count.
- `server.webrtc`: signaling bind address, offer path, ICE servers, data channel names.
- `client.camera`: RealSense/video source settings.
- `client.webrtc`: server offer URL, JPEG quality, target FPS, backpressure limits.
- `client.dashboard`: browser dashboard host/port/routes.
- `apriltags` and `field`: floor calibration and field dimensions.
- `robots`: robot tag IDs, names, ROS namespaces, offsets, global-state policy.
- `ros2`: ROS domain, RMW implementation, QoS, and private LAN binding.
- `robot_runtime`: the single robot identity used by `robot/app.py`.

Set the client to reach the SAM3 server:

```yaml
client:
  webrtc:
    server_offer_url: "http://<server-ip>:8765/offer"
```

Set each robot entry explicitly:

```yaml
robots:
  publish_global_state: false
  entries:
    - tag_id: 4
      name: "robot_4"
      ros_namespace: "/robots/robot_4"
      center_offset_m: [0.0, 0.0]
      allow_global_state: false
```

Set the runtime identity on each robot:

```yaml
robot_runtime:
  name: "robot_4"
  ros_namespace: "/robots/robot_4"
```

## Run

Preferred console scripts:

```bash
uv run theteam-server
uv run theteam-client
uv run theteam-robot
```

Equivalent module commands:

```bash
uv run python -m server.app
uv run python -m client.app
uv run python -m robot.app
```

Open the dashboard:

```text
http://<client-ip>:8080/
```

## Dashboard

The client serves a modern monitoring dashboard with:

- live annotated camera feed
- live field map in its own section
- compact top-line client/server stats
- SAM3 contours
- AprilTag detections
- robot pose labels
- WebRTC state and frame counts
- server inference/end-to-end latency
- client camera FPS and calibration state
- ROS2 gateway state
- per-robot visibility, pose, pose age, and namespace

Dashboard routes are configured under `client.dashboard`.

## WebRTC Flow

WebRTC is used only between the client and SAM3 server.

The client opens two data channels:

- `camera_frames`: chunked JPEG frames from client to server
- `sam3_contours`: contour-only JSON from server to client

Each frame carries:

- `frame_id`
- `capture_unix_ns`
- `capture_monotonic_ns`
- dimensions
- `homography_version`

The server echoes timing metadata in the contour packet. The client matches
contours back to pending frames by `frame_id` and drops stale results.

## ROS2 Topics

ROS2 stays on the private LAN and is published only by the client gateway.

Default per-robot topics:

```text
/robots/<name>/pose              geometry_msgs/msg/PoseStamped
/robots/<name>/location          geometry_msgs/msg/PointStamped
/robots/<name>/tracking_status   std_msgs/msg/String
```

Global state is off by default. If `robots.publish_global_state` is enabled,
only robots with `allow_global_state: true` are included:

```text
/team/tracking_state             std_msgs/msg/String
```

For hard access isolation, add SROS2 permissions at deployment time.

## AprilTag Calibration

Boundary tags must be ordered top-left, top-right, bottom-right, bottom-left:

```yaml
apriltags:
  boundary_tag_ids: [0, 1, 2, 3]
```

The client locks the floor homography once all boundary tags are visible. SAM3
frames are masked to that calibrated field boundary before they are sent to the
server.
