import math
from typing import Any

import cv2
import numpy as np

from vision.tracking import boundary_polygon


def color_for_id(object_id):
    return (
        int(37 * (object_id + 3) % 255),
        int(97 * (object_id + 5) % 255),
        int(173 * (object_id + 7) % 255),
    )


def put_label(frame, text, origin, color=(245, 245, 245), scale=0.55, thickness=1):
    x, y = origin
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


class LiveVisualizationRenderer:
    def __init__(
        self,
        display_config: dict[str, Any],
        field_config: dict[str, Any],
        apriltag_config: dict[str, Any],
    ):
        self.map_size = tuple(int(value) for value in display_config["map_size"])
        self.map_margin = int(display_config["map_margin"])
        self.field_width_m = float(field_config["width_m"])
        self.field_height_m = float(field_config["height_m"])
        self.boundary_tag_ids = [int(tag_id) for tag_id in apriltag_config["boundary_tag_ids"]]

    def render(
        self,
        frame,
        tags,
        state,
        contour_packet: dict[str, Any] | None,
        contour_age_ms: float | None,
        runtime_stats: dict[str, Any],
    ):
        canvas = frame.copy()
        self._draw_tags(canvas, tags, state)
        self._draw_contours(canvas, contour_packet, contour_age_ms)
        self._draw_pose_labels(canvas, state)
        self._draw_status_strip(canvas, state, runtime_stats, contour_age_ms)
        return canvas

    def _draw_tags(self, frame, tags, state):
        for tag_id, tag in tags.items():
            color = (90, 220, 90) if tag_id in self.boundary_tag_ids else color_for_id(tag_id)
            corners = tag["corners"].astype(np.int32)
            center = tuple(tag["center"].astype(int))
            cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=2)
            cv2.circle(frame, center, 4, color, -1, lineType=cv2.LINE_AA)
            put_label(frame, str(tag_id), (center[0] + 8, center[1] - 8), color=color)

        if state.get("ready"):
            boundary = boundary_polygon(state["boundary_points_px"])
            cv2.polylines(frame, [boundary], isClosed=True, color=(0, 235, 255), thickness=2)
            for robot in state["robots"]:
                center = tuple(robot["tag_center_px"].astype(int))
                heading = robot["tag_corners_px"][1] - robot["tag_corners_px"][0]
                norm = np.linalg.norm(heading)
                if norm > 0:
                    heading = heading / norm
                    end = (
                        int(center[0] + heading[0] * 48),
                        int(center[1] + heading[1] * 48),
                    )
                    cv2.arrowedLine(frame, center, end, color_for_id(robot["id"]), 2, tipLength=0.25)
        else:
            missing = ",".join(str(tag_id) for tag_id in state.get("missing_boundary", []))
            put_label(frame, f"missing boundary tags: {missing}", (20, 78), color=(0, 220, 255), scale=0.75, thickness=2)

    def _draw_contours(self, frame, contour_packet, contour_age_ms):
        if not contour_packet:
            return
        if contour_age_ms is not None and contour_age_ms > 1000.0:
            return

        for obj in contour_packet.get("objects", []):
            object_id = int(obj.get("id", 0))
            color = color_for_id(object_id)
            for contour in obj.get("contours_px", obj.get("contours", [])):
                pts = np.asarray(contour, dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            center = obj.get("center_px", obj.get("center"))
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

    def _draw_pose_labels(self, frame, state):
        if not state.get("ready"):
            return
        for robot in state.get("robots", []):
            center = robot["tag_center_px"].astype(int)
            theta_deg = math.degrees(float(robot["theta"]))
            text = f'{robot["name"]}  x={robot["x"]:.2f} y={robot["y"]:.2f} th={theta_deg:.0f}'
            put_label(
                frame,
                text,
                (int(center[0]) + 12, int(center[1]) + 24),
                color=color_for_id(robot["id"]),
                scale=0.55,
                thickness=1,
            )

    def _draw_status_strip(self, frame, state, runtime_stats, contour_age_ms):
        h, w = frame.shape[:2]
        strip_h = 42
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, strip_h), (14, 18, 24), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        client = runtime_stats.get("client", {})
        webrtc = runtime_stats.get("webrtc", {})
        ros2 = runtime_stats.get("ros2", {})
        contour_text = "contours fresh" if contour_age_ms is not None and contour_age_ms < 500 else "contours waiting"
        text = (
            f'frame {client.get("frame_id", 0)}   '
            f'cam {client.get("camera_fps", 0.0):.1f} fps   '
            f'webrtc {webrtc.get("connection_state", "new")}   '
            f'ros2 {ros2.get("state", "off")}   '
            f'cal {"locked" if state.get("ready") else "waiting"}   '
            f"{contour_text}"
        )
        put_label(frame, text, (18, 28), color=(242, 246, 252), scale=0.62, thickness=1)

    def field_to_map(self, x, y):
        map_w, map_h = self.map_size
        usable_w = map_w - 2 * self.map_margin
        usable_h = map_h - 2 * self.map_margin
        scale = min(usable_w / self.field_width_m, usable_h / self.field_height_m)
        offset_x = (map_w - self.field_width_m * scale) / 2.0
        offset_y = (map_h - self.field_height_m * scale) / 2.0
        return int(offset_x + x * scale), int(offset_y + y * scale), scale

    def draw_map(self, state):
        map_w, map_h = self.map_size
        canvas = np.full((map_h, map_w, 3), 245, dtype=np.uint8)

        x0, y0, scale = self.field_to_map(0.0, 0.0)
        x1, y1, _ = self.field_to_map(self.field_width_m, self.field_height_m)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (30, 30, 30), 2)

        for i in range(1, int(self.field_width_m) + 1):
            gx, _, _ = self.field_to_map(float(i), 0.0)
            cv2.line(canvas, (gx, y0), (gx, y1), (210, 210, 210), 1)
        for i in range(1, int(self.field_height_m) + 1):
            _, gy, _ = self.field_to_map(0.0, float(i))
            cv2.line(canvas, (x0, gy), (x1, gy), (210, 210, 210), 1)

        if not state.get("ready"):
            missing = ", ".join(str(tag_id) for tag_id in state.get("missing_boundary", []))
            put_label(canvas, f"Waiting: {missing}", (24, 36), color=(0, 100, 220), scale=0.58, thickness=1)
            return canvas

        robot_radius = max(8, int(0.05 * scale))
        arrow_len = max(20, int(0.16 * scale))
        for robot in state["robots"]:
            px, py, _ = self.field_to_map(robot["x"], robot["y"])
            color = color_for_id(robot["id"])
            theta = robot["theta"]
            end = (int(px + math.cos(theta) * arrow_len), int(py + math.sin(theta) * arrow_len))

            cv2.circle(canvas, (px, py), robot_radius, color, -1, lineType=cv2.LINE_AA)
            cv2.arrowedLine(canvas, (px, py), end, (20, 20, 20), 2, tipLength=0.35)
            put_label(canvas, robot["name"], (px + 10, py - 10), color=(20, 20, 20), scale=0.48, thickness=1)

        return canvas
