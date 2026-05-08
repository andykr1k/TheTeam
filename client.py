import asyncio
import json
import math
import struct
import time
from collections import OrderedDict

import cv2
import numpy as np
import websockets

from config import load_config


if not hasattr(cv2, "aruco"):
    raise RuntimeError("This client needs OpenCV AprilTag support: cv2.aruco is missing.")


CONFIG = load_config()
CLIENT_CONFIG = CONFIG.get("client", {})
APRILTAG_CONFIG = CONFIG.get("apriltags", {})
FIELD_CONFIG = CONFIG.get("field", {})
ROBOT_CONFIG = CONFIG.get("robots", {})
DISPLAY_CONFIG = CONFIG.get("display", {})

SERVER_URL = CLIENT_CONFIG.get("server_url", "ws://127.0.0.1:8765")
USE_REALSENSE = bool(CLIENT_CONFIG.get("use_realsense", False))
VIDEO_SOURCE = CLIENT_CONFIG.get("video_source", 0)
CAMERA_WIDTH = int(CLIENT_CONFIG.get("camera_width", 1280))
CAMERA_HEIGHT = int(CLIENT_CONFIG.get("camera_height", 720))
CAMERA_FPS = int(CLIENT_CONFIG.get("camera_fps", 30))
JPEG_QUALITY = int(CLIENT_CONFIG.get("jpeg_quality", 60))
TARGET_FPS = float(CLIENT_CONFIG.get("target_fps", 30))
MAX_PENDING_FRAMES = int(CLIENT_CONFIG.get("max_pending_frames", 90))

BOUNDARY_TAG_IDS = [int(tag_id) for tag_id in APRILTAG_CONFIG.get("boundary_tag_ids", [10, 11, 12, 13])]
FIELD_WIDTH_M = float(FIELD_CONFIG.get("width_m", 3.0))
FIELD_HEIGHT_M = float(FIELD_CONFIG.get("height_m", 2.0))
ROBOT_TAG_IDS = {int(tag_id): str(name) for tag_id, name in ROBOT_CONFIG.get("tag_ids", {}).items()}
ROBOT_CENTER_OFFSETS_M = {
    int(tag_id): tuple(float(value) for value in offset)
    for tag_id, offset in ROBOT_CONFIG.get("center_offsets_m", {}).items()
}
ROBOT_HEADING_OFFSET_RAD = float(ROBOT_CONFIG.get("heading_offset_rad", 0.0))

APRILTAG_DICT_NAME = APRILTAG_CONFIG.get("dictionary", "DICT_APRILTAG_36h11")
APRILTAG_DICT = getattr(cv2.aruco, APRILTAG_DICT_NAME)
TAG_DETECT_SCALE = float(APRILTAG_CONFIG.get("detect_scale", 1.0))
MAP_SIZE = tuple(int(value) for value in DISPLAY_CONFIG.get("map_size", [900, 650]))
MAP_MARGIN = int(DISPLAY_CONFIG.get("map_margin", 60))

CONTOUR_WINDOW = DISPLAY_CONFIG.get("contour_window", "SAM3 contours")
CAMERA_WINDOW = DISPLAY_CONFIG.get("camera_window", "AprilTag camera")
MAP_WINDOW = DISPLAY_CONFIG.get("map_window", "Robot map")

HEADER = struct.Struct("!Q")


class FrameSource:
    def __init__(self):
        self.pipeline = None
        self.cap = None

        if USE_REALSENSE:
            try:
                import pyrealsense2 as rs
            except ImportError as exc:
                raise RuntimeError("Install pyrealsense2 on the Raspberry Pi.") from exc

            self.rs = rs
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.color,
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                rs.format.bgr8,
                CAMERA_FPS,
            )
            self.pipeline.start(config)
        else:
            self.cap = cv2.VideoCapture(VIDEO_SOURCE)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

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


class AprilTagTracker:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = float(TAG_DETECT_SCALE)
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        corners_list, ids, _ = self.detector.detectMarkers(gray)
        tags = {}
        if ids is None:
            return tags

        for corners, tag_id in zip(corners_list, ids.flatten(), strict=False):
            corners = corners.reshape(4, 2).astype(np.float32)
            if scale != 1.0:
                corners /= scale

            perimeter = cv2.arcLength(corners.astype(np.float32), closed=True)
            if tag_id in tags and tags[tag_id]["perimeter"] >= perimeter:
                continue

            tags[int(tag_id)] = {
                "id": int(tag_id),
                "corners": corners,
                "center": corners.mean(axis=0),
                "perimeter": perimeter,
            }
        return tags

    def map_state(self, tags):
        boundary = [tags.get(tag_id) for tag_id in BOUNDARY_TAG_IDS]
        missing_boundary = [
            tag_id for tag_id, tag in zip(BOUNDARY_TAG_IDS, boundary, strict=False) if tag is None
        ]
        if missing_boundary:
            return {"ready": False, "missing_boundary": missing_boundary, "robots": []}

        image_points = np.float32([tag["center"] for tag in boundary])
        field_points = np.float32(
            [
                [0.0, 0.0],
                [FIELD_WIDTH_M, 0.0],
                [FIELD_WIDTH_M, FIELD_HEIGHT_M],
                [0.0, FIELD_HEIGHT_M],
            ]
        )
        homography = cv2.getPerspectiveTransform(image_points, field_points)

        robot_ids = set(ROBOT_TAG_IDS) if ROBOT_TAG_IDS else None
        robots = []
        for tag_id, tag in tags.items():
            if tag_id in BOUNDARY_TAG_IDS:
                continue
            if robot_ids is not None and tag_id not in robot_ids:
                continue

            floor_corners = transform_points(homography, tag["corners"])
            floor_center = floor_corners.mean(axis=0)
            heading_vec = floor_corners[1] - floor_corners[0]
            theta = math.atan2(float(heading_vec[1]), float(heading_vec[0]))
            theta += ROBOT_HEADING_OFFSET_RAD

            forward_m, left_m = ROBOT_CENTER_OFFSETS_M.get(tag_id, (0.0, 0.0))
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            robot_x = float(floor_center[0] + cos_t * forward_m - sin_t * left_m)
            robot_y = float(floor_center[1] + sin_t * forward_m + cos_t * left_m)

            robots.append(
                {
                    "id": tag_id,
                    "name": ROBOT_TAG_IDS.get(tag_id, f"robot_{tag_id}"),
                    "x": robot_x,
                    "y": robot_y,
                    "theta": normalize_angle(theta),
                    "tag_center_px": tag["center"],
                    "tag_corners_px": tag["corners"],
                }
            )

        return {
            "ready": True,
            "homography": homography,
            "boundary_points_px": image_points,
            "robots": robots,
        }


def transform_points(homography, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def normalize_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def color_for_id(object_id):
    return (
        int(37 * (object_id + 3) % 255),
        int(97 * (object_id + 5) % 255),
        int(173 * (object_id + 7) % 255),
    )


def draw_tag_overlay(frame, tags, state):
    for tag_id, tag in tags.items():
        color = (90, 220, 90) if tag_id in BOUNDARY_TAG_IDS else color_for_id(tag_id)
        corners = tag["corners"].astype(np.int32)
        center = tuple(tag["center"].astype(int))
        cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=2)
        cv2.circle(frame, center, 4, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            str(tag_id),
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if state.get("ready"):
        boundary = state["boundary_points_px"].astype(np.int32)
        cv2.polylines(frame, [boundary], isClosed=True, color=(0, 255, 255), thickness=2)
        for robot in state["robots"]:
            center = tuple(robot["tag_center_px"].astype(int))
            heading = robot["tag_corners_px"][1] - robot["tag_corners_px"][0]
            norm = np.linalg.norm(heading)
            if norm > 0:
                heading = heading / norm
                end = (
                    int(center[0] + heading[0] * 42),
                    int(center[1] + heading[1] * 42),
                )
                cv2.arrowedLine(frame, center, end, color_for_id(robot["id"]), 2, tipLength=0.25)
    else:
        missing = ",".join(str(tag_id) for tag_id in state.get("missing_boundary", []))
        cv2.putText(
            frame,
            f"missing boundary tags: {missing}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def field_to_map(x, y):
    map_w, map_h = MAP_SIZE
    usable_w = map_w - 2 * MAP_MARGIN
    usable_h = map_h - 2 * MAP_MARGIN
    scale = min(usable_w / FIELD_WIDTH_M, usable_h / FIELD_HEIGHT_M)
    offset_x = (map_w - FIELD_WIDTH_M * scale) / 2.0
    offset_y = (map_h - FIELD_HEIGHT_M * scale) / 2.0
    return int(offset_x + x * scale), int(offset_y + y * scale), scale


def draw_map(state):
    map_w, map_h = MAP_SIZE
    canvas = np.full((map_h, map_w, 3), 245, dtype=np.uint8)

    x0, y0, scale = field_to_map(0.0, 0.0)
    x1, y1, _ = field_to_map(FIELD_WIDTH_M, FIELD_HEIGHT_M)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (30, 30, 30), 2)

    for i in range(1, int(FIELD_WIDTH_M) + 1):
        gx, _, _ = field_to_map(float(i), 0.0)
        cv2.line(canvas, (gx, y0), (gx, y1), (210, 210, 210), 1)
    for i in range(1, int(FIELD_HEIGHT_M) + 1):
        _, gy, _ = field_to_map(0.0, float(i))
        cv2.line(canvas, (x0, gy), (x1, gy), (210, 210, 210), 1)

    if not state.get("ready"):
        missing = ", ".join(str(tag_id) for tag_id in state.get("missing_boundary", []))
        cv2.putText(
            canvas,
            f"Waiting for boundary tags: {missing}",
            (35, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 120, 220),
            2,
            cv2.LINE_AA,
        )
        return canvas

    cv2.putText(
        canvas,
        f"{FIELD_WIDTH_M:.2f}m x {FIELD_HEIGHT_M:.2f}m",
        (35, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    robot_radius = max(10, int(0.055 * scale))
    arrow_len = max(24, int(0.18 * scale))
    for robot in state["robots"]:
        px, py, _ = field_to_map(robot["x"], robot["y"])
        color = color_for_id(robot["id"])
        theta = robot["theta"]
        end = (int(px + math.cos(theta) * arrow_len), int(py + math.sin(theta) * arrow_len))

        cv2.circle(canvas, (px, py), robot_radius, color, -1, lineType=cv2.LINE_AA)
        cv2.arrowedLine(canvas, (px, py), end, (20, 20, 20), 2, tipLength=0.35)
        cv2.putText(
            canvas,
            f'{robot["name"]} ({robot["x"]:.2f},{robot["y"]:.2f})',
            (px + 12, py - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    return canvas


def draw_sam_result(frame, result, e2e_ms):
    for obj in result.get("objects", []):
        color = color_for_id(obj["id"])
        for contour in obj["contours"]:
            pts = np.asarray(contour, dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        center = obj.get("center")
        if center is not None:
            x, y = int(center[0]), int(center[1])
            cv2.drawMarker(
                frame,
                (x, y),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(frame, (x, y), 4, color, -1, lineType=cv2.LINE_AA)

    text = f"infer {result.get('inference_ms', 0.0):.1f}ms  e2e {e2e_ms:.1f}ms"
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


async def send_frames(websocket, pending, stop):
    source = FrameSource()
    tracker = AprilTagTracker()
    frame_id = 0
    frame_period = 1.0 / TARGET_FPS if TARGET_FPS else 0.0

    try:
        while not stop.is_set():
            loop_start = time.perf_counter()
            ok, frame = source.read()
            if not ok:
                stop.set()
                break

            tags = tracker.detect(frame)
            state = tracker.map_state(tags)
            display_frame = draw_tag_overlay(frame.copy(), tags, state)
            map_frame = draw_map(state)

            cv2.imshow(CAMERA_WINDOW, display_frame)
            cv2.imshow(MAP_WINDOW, map_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                stop.set()
                break

            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not ok:
                continue

            pending[frame_id] = (display_frame.copy(), time.perf_counter(), state)
            while len(pending) > MAX_PENDING_FRAMES:
                pending.popitem(last=False)

            await websocket.send(HEADER.pack(frame_id) + jpg.tobytes())
            frame_id += 1

            sleep_for = frame_period - (time.perf_counter() - loop_start)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                await asyncio.sleep(0)
    finally:
        source.release()


async def receive_contours(websocket, pending, stop):
    async for message in websocket:
        result = json.loads(message)
        frame_id = result["frame_id"]
        item = pending.pop(frame_id, None)
        if item is None:
            continue

        frame, sent_time, state = item
        e2e_ms = (time.perf_counter() - sent_time) * 1000.0
        frame = draw_sam_result(frame, result, e2e_ms)

        cv2.imshow(CONTOUR_WINDOW, frame)
        print(
            f"frame={frame_id} "
            f"inference_ms={result.get('inference_ms', 0.0):.1f} "
            f"e2e_ms={e2e_ms:.1f} "
            f"robots={len(state.get('robots', []))} "
            f"sam_objects={len(result.get('objects', []))}"
        )


async def main():
    pending = OrderedDict()
    stop = asyncio.Event()

    async with websockets.connect(
        SERVER_URL,
        max_size=8 * 1024 * 1024,
        max_queue=1,
        compression=None,
    ) as websocket:
        send_task = asyncio.create_task(send_frames(websocket, pending, stop))
        recv_task = asyncio.create_task(receive_contours(websocket, pending, stop))
        try:
            await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            stop.set()
            send_task.cancel()
            recv_task.cancel()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
