import asyncio
import math
import subprocess
import sys
import time
from collections import OrderedDict, deque
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from client.dashboard import DashboardServer
from client.ros2_gateway import Ros2Gateway
from common.messages import FramePacket
from common.webrtc_transport import TransportStats, WebRTCClient
from config import load_config
from vision.tracking import (
    AprilTagTracker,
    filter_objects_to_boundary,
    mask_frame_to_boundary,
)
from vision.visualization import LiveVisualizationRenderer


CONFIG = load_config()
CLIENT_CONFIG = CONFIG["client"]
CAMERA_CONFIG = CLIENT_CONFIG["camera"]
CLIENT_WEBRTC_CONFIG = CLIENT_CONFIG["webrtc"]
SERVER_WEBRTC_CONFIG = CONFIG["server"]["webrtc"]
APRILTAG_CONFIG = CONFIG["apriltags"]
FIELD_CONFIG = CONFIG["field"]
ROBOT_CONFIG = CONFIG["robots"]
DISPLAY_CONFIG = CONFIG["display"]
ROS2_CONFIG = CONFIG["ros2"]


def probe_realsense_startup(camera_config: dict[str, Any]):
    width = int(camera_config["width"])
    height = int(camera_config["height"])
    fps = int(camera_config["fps"])
    timeout_s = float(camera_config["realsense_probe_timeout_s"])
    probe = (
        "import pyrealsense2 as rs;"
        "pipeline = rs.pipeline();"
        "config = rs.config();"
        f"config.enable_stream(rs.stream.color, {width}, {height}, rs.format.bgr8, {fps});"
        "profile = pipeline.start(config);"
        "pipeline.stop();"
        "print('ok')"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=max(timeout_s, 1.0),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {timeout_s:.1f}s"

    if result.returncode == 0:
        return True, ""

    if result.returncode < 0:
        details = result.stderr.strip() or result.stdout.strip()
        if details:
            return False, f"terminated by signal {-result.returncode}: {details}"
        return False, f"terminated by signal {-result.returncode}"

    details = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    return False, details


class FrameSource:
    def __init__(self, camera_config: dict[str, Any]):
        self.config = camera_config
        self.pipeline = None
        self.rs = None
        self.cap = None
        self.width = int(camera_config["width"])
        self.height = int(camera_config["height"])
        self.fps = int(camera_config["fps"])

        if bool(camera_config["use_realsense"]):
            ok, reason = probe_realsense_startup(camera_config)
            if not ok:
                if not bool(camera_config["realsense_fallback_to_video"]):
                    raise RuntimeError(f"RealSense startup probe failed: {reason}")
                print(
                    "warning: RealSense startup failed; "
                    f"falling back to video_source={camera_config['video_source']}. "
                    f"Details: {reason}"
                )
            else:
                try:
                    import pyrealsense2 as rs
                except ImportError as exc:
                    raise RuntimeError("Install pyrealsense2 on the camera client.") from exc

                self.rs = rs
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(
                    rs.stream.color,
                    self.width,
                    self.height,
                    rs.format.bgr8,
                    self.fps,
                )
                self.pipeline.start(config)

        if self.pipeline is None:
            raw_source = camera_config["video_source"]
            try:
                source = int(raw_source)
            except (TypeError, ValueError):
                source = raw_source

            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {raw_source}")

    def read(self):
        if self.pipeline is not None:
            frames = self.pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                return False, None
            return True, np.asanyarray(color.get_data())

        return self.cap.read()

    def release(self):
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.cap is not None:
            self.cap.release()


class ClientRuntime:
    def __init__(self):
        self.stop = asyncio.Event()
        self.pending = OrderedDict()
        self.max_pending_frames = int(CLIENT_WEBRTC_CONFIG["max_pending_frames"])
        self.jpeg_quality = int(CLIENT_WEBRTC_CONFIG["jpeg_quality"])
        self.target_fps = float(CLIENT_WEBRTC_CONFIG["target_fps"])
        self.reconnect_delay_s = float(CLIENT_WEBRTC_CONFIG["reconnect_delay_s"])
        self.stale_contour_ms = float(CLIENT_WEBRTC_CONFIG["stale_contour_ms"])

        self.tracker = AprilTagTracker(APRILTAG_CONFIG, FIELD_CONFIG, ROBOT_CONFIG)
        self.renderer = LiveVisualizationRenderer(DISPLAY_CONFIG, FIELD_CONFIG, APRILTAG_CONFIG)
        self.dashboard = DashboardServer(CLIENT_CONFIG["dashboard"])
        self.ros2 = Ros2Gateway(
            ROS2_CONFIG,
            self.tracker.robot_specs,
            self.tracker.field_frame_id,
            bool(ROBOT_CONFIG["publish_global_state"]),
        )

        self.transport: WebRTCClient | None = None
        self.last_transport_stats = TransportStats()
        self.latest_contour_packet: dict[str, Any] | None = None
        self.latest_contour_received_monotonic_ns: int | None = None
        self.last_server_stats = {
            "inference_ms": 0.0,
            "e2e_ms": 0.0,
            "objects": 0,
            "contours": 0,
        }
        self.frame_times = deque(maxlen=90)
        self.robot_latest: dict[str, dict[str, Any]] = {}
        self.current_visible_names: set[str] = set()
        self.frame_id = 0
        self.last_state = {"ready": False, "calibrated": False, "missing_boundary": []}

    async def start(self):
        await self.dashboard.start()
        if self.dashboard.enabled:
            print(
                "dashboard listening "
                f"http://{CLIENT_CONFIG['dashboard']['host']}:{CLIENT_CONFIG['dashboard']['port']}"
                f"{CLIENT_CONFIG['dashboard']['path']}"
            )
        self.ros2.start()

        webrtc_task = asyncio.create_task(self._webrtc_reconnect_loop())
        capture_task = asyncio.create_task(self._capture_loop())
        try:
            done, _ = await asyncio.wait(
                [webrtc_task, capture_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
        finally:
            self.stop.set()
            webrtc_task.cancel()
            capture_task.cancel()
            await asyncio.gather(webrtc_task, capture_task, return_exceptions=True)
            if self.transport is not None:
                await self.transport.close()
            self.ros2.stop()
            await self.dashboard.stop()

    async def _webrtc_reconnect_loop(self):
        while not self.stop.is_set():
            client = WebRTCClient(
                CLIENT_WEBRTC_CONFIG,
                SERVER_WEBRTC_CONFIG,
                self._handle_contours,
            )
            self.transport = client
            try:
                await client.connect()
                print("WebRTC segmentation link connected.")
                await client.wait_closed()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                client.stats.last_error = str(exc)
                client.stats.connection_state = "error"
                client.stats.updated_at_unix_ns = time.time_ns()
                print(f"WebRTC segmentation link unavailable: {exc}")
            finally:
                self.last_transport_stats = client.stats
                await client.close()
                if self.transport is client:
                    self.transport = None

            await asyncio.sleep(self.reconnect_delay_s)

    async def _handle_contours(self, packet):
        item = self.pending.pop(packet.frame_id, None)
        e2e_ms = 0.0
        boundary_points = None
        if item is not None:
            sent_monotonic_ns, boundary_points = item
            e2e_ms = (time.monotonic_ns() - sent_monotonic_ns) / 1_000_000.0

        objects = packet.objects
        if boundary_points is not None:
            objects = filter_objects_to_boundary(objects, boundary_points)

        packet_dict = asdict(packet)
        packet_dict["objects"] = objects
        contour_count = sum(len(obj.get("contours_px", obj.get("contours", []))) for obj in objects)
        self.latest_contour_packet = packet_dict
        self.latest_contour_received_monotonic_ns = time.monotonic_ns()
        self.last_server_stats = {
            "inference_ms": float(packet.inference_ms),
            "server_ms": float(packet.server_ms),
            "e2e_ms": float(e2e_ms),
            "objects": len(objects),
            "contours": contour_count,
            "error": packet.error or "",
            **packet.server_stats,
        }

    async def _capture_loop(self):
        source = FrameSource(CAMERA_CONFIG)
        frame_period = 1.0 / self.target_fps if self.target_fps else 0.0

        try:
            while not self.stop.is_set():
                loop_start = time.monotonic_ns()
                ok, frame = source.read()
                capture_unix_ns = time.time_ns()
                capture_monotonic_ns = time.monotonic_ns()
                if not ok:
                    raise RuntimeError("camera frame read failed")

                self.frame_times.append(capture_monotonic_ns)
                tags = self.tracker.detect(frame)
                state = self.tracker.map_state(tags)
                self.last_state = state

                estimates = self.tracker.pose_estimates(self.frame_id, capture_unix_ns, state)
                self._update_robot_latest(estimates, capture_monotonic_ns)
                self.ros2.publish(estimates, self.frame_id, capture_unix_ns)

                contour_age_ms = self._contour_age_ms()
                active_contour = (
                    self.latest_contour_packet
                    if contour_age_ms is not None and contour_age_ms <= self.stale_contour_ms
                    else None
                )
                stats = self._build_stats(state)
                rendered = self.renderer.render(
                    frame,
                    tags,
                    state,
                    active_contour,
                    contour_age_ms,
                    stats,
                )
                map_frame = self.renderer.draw_map(state)
                await self.dashboard.update_frame(rendered)
                await self.dashboard.update_map_frame(map_frame)
                await self.dashboard.update_stats(stats)

                if state.get("ready"):
                    await self._send_segmentation_frame(
                        frame,
                        state,
                        capture_unix_ns,
                        capture_monotonic_ns,
                    )

                self.frame_id += 1
                elapsed_s = (time.monotonic_ns() - loop_start) / 1_000_000_000.0
                sleep_for = frame_period - elapsed_s
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    await asyncio.sleep(0)
        finally:
            source.release()

    async def _send_segmentation_frame(
        self,
        frame,
        state,
        capture_unix_ns: int,
        capture_monotonic_ns: int,
    ):
        transport = self.transport
        if transport is None or not transport.is_ready():
            return

        masked_frame = mask_frame_to_boundary(frame, state["boundary_points_px"])
        ok, jpg = cv2.imencode(
            ".jpg",
            masked_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return

        packet = FramePacket(
            frame_id=self.frame_id,
            capture_unix_ns=capture_unix_ns,
            capture_monotonic_ns=capture_monotonic_ns,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            homography_version=int(state.get("homography_version", 0)),
        )

        self.pending[self.frame_id] = (
            time.monotonic_ns(),
            np.asarray(state["boundary_points_px"], dtype=np.float32).copy(),
        )
        while len(self.pending) > self.max_pending_frames:
            self.pending.popitem(last=False)

        await transport.send_frame(packet, jpg.tobytes())
        self.last_transport_stats = transport.stats

    def _update_robot_latest(self, estimates, capture_monotonic_ns: int):
        self.current_visible_names = {estimate.name for estimate in estimates}
        for estimate in estimates:
            self.robot_latest[estimate.name] = {
                "name": estimate.name,
                "tag_id": estimate.tag_id,
                "ros_namespace": estimate.ros_namespace,
                "x": estimate.x,
                "y": estimate.y,
                "theta": estimate.theta,
                "theta_deg": math.degrees(estimate.theta),
                "frame_id": estimate.frame_id,
                "capture_unix_ns": estimate.capture_unix_ns,
                "last_seen_monotonic_ns": capture_monotonic_ns,
            }

    def _camera_fps(self):
        if len(self.frame_times) < 2:
            return 0.0
        elapsed_s = (self.frame_times[-1] - self.frame_times[0]) / 1_000_000_000.0
        if elapsed_s <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / elapsed_s

    def _contour_age_ms(self):
        if self.latest_contour_received_monotonic_ns is None:
            return None
        return (time.monotonic_ns() - self.latest_contour_received_monotonic_ns) / 1_000_000.0

    def _robot_stats(self):
        now_ns = time.monotonic_ns()
        rows = []
        for spec in self.tracker.robot_specs.values():
            latest = self.robot_latest.get(spec.name)
            if latest is None:
                rows.append(
                    {
                        "name": spec.name,
                        "tag_id": spec.tag_id,
                        "ros_namespace": spec.ros_namespace,
                        "visible": False,
                        "last_seen_age_ms": None,
                    }
                )
                continue

            age_ms = (now_ns - latest["last_seen_monotonic_ns"]) / 1_000_000.0
            rows.append(
                {
                    **latest,
                    "visible": spec.name in self.current_visible_names,
                    "last_seen_age_ms": age_ms,
                }
            )
        return rows

    def _build_stats(self, state):
        transport_stats = self.transport.stats if self.transport is not None else self.last_transport_stats
        return {
            "client": {
                "frame_id": self.frame_id,
                "camera_fps": self._camera_fps(),
                "calibrated": bool(state.get("calibrated", False)),
                "ready": bool(state.get("ready", False)),
                "missing_boundary": [int(tag_id) for tag_id in state.get("missing_boundary", [])],
                "homography_version": int(state.get("homography_version", 0)),
                "pending_frames": len(self.pending),
            },
            "server": self.last_server_stats,
            "webrtc": transport_stats.to_dict(),
            "ros2": self.ros2.stats(),
            "robots": self._robot_stats(),
            "updated_at_unix_ns": time.time_ns(),
        }


async def main():
    runtime = ClientRuntime()
    await runtime.start()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
