import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from common.messages import RobotPoseEstimate


if not hasattr(cv2, "aruco"):
    raise RuntimeError("This client needs OpenCV AprilTag support: cv2.aruco is missing.")


@dataclass(frozen=True, slots=True)
class RobotSpec:
    tag_id: int
    name: str
    ros_namespace: str
    center_offset_m: tuple[float, float] = (0.0, 0.0)
    allow_global_state: bool = False


def parse_robot_specs(robot_config: dict[str, Any]) -> dict[int, RobotSpec]:
    entries = robot_config.get("entries")
    if isinstance(entries, list):
        parsed = {}
        for entry in entries:
            tag_id = int(entry["tag_id"])
            name = str(entry.get("name") or f"robot_{tag_id}")
            ros_namespace = str(entry.get("ros_namespace") or f"/robots/{name}")
            offset = entry.get("center_offset_m", [0.0, 0.0])
            parsed[tag_id] = RobotSpec(
                tag_id=tag_id,
                name=name,
                ros_namespace=ros_namespace.rstrip("/") or f"/robots/{name}",
                center_offset_m=(float(offset[0]), float(offset[1])),
                allow_global_state=bool(entry.get("allow_global_state", False)),
            )
        return parsed

    # Backward-compatible parser for the older config shape.
    raw_tag_ids = robot_config.get("tag_ids", {})
    raw_offsets = robot_config.get("center_offsets_m", {})
    if isinstance(raw_tag_ids, dict):
        tag_names = {
            int(raw_tag_id): f"robot_{int(raw_tag_id)}" if raw_name in (None, "") else str(raw_name)
            for raw_tag_id, raw_name in raw_tag_ids.items()
        }
    elif isinstance(raw_tag_ids, (list, tuple, set)):
        tag_names = {int(tag_id): f"robot_{int(tag_id)}" for tag_id in raw_tag_ids}
    else:
        tag_names = {}

    parsed = {}
    for tag_id, name in tag_names.items():
        offset = raw_offsets.get(tag_id, raw_offsets.get(str(tag_id), [0.0, 0.0]))
        parsed[tag_id] = RobotSpec(
            tag_id=tag_id,
            name=name,
            ros_namespace=f"/robots/{name}",
            center_offset_m=(float(offset[0]), float(offset[1])),
        )
    return parsed


def transform_points(homography, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def normalize_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def boundary_polygon(boundary_points):
    return np.round(np.asarray(boundary_points, dtype=np.float32)).astype(np.int32)


def mask_frame_to_boundary(frame, boundary_points):
    polygon = boundary_polygon(boundary_points)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def filter_objects_to_boundary(objects, boundary_points):
    if boundary_points is None:
        return list(objects)

    polygon = np.asarray(boundary_points, dtype=np.float32).reshape(-1, 1, 2)
    filtered = []
    for obj in objects:
        center = obj.get("center_px", obj.get("center"))
        if center is None:
            continue
        if cv2.pointPolygonTest(polygon, (float(center[0]), float(center[1])), False) >= 0:
            filtered.append(obj)
    return filtered


class AprilTagTracker:
    def __init__(
        self,
        apriltag_config: dict[str, Any],
        field_config: dict[str, Any],
        robot_config: dict[str, Any],
    ):
        dictionary_name = str(apriltag_config["dictionary"])
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        self.detect_scale = float(apriltag_config["detect_scale"])
        self.boundary_tag_ids = [int(tag_id) for tag_id in apriltag_config["boundary_tag_ids"]]
        self.field_width_m = float(field_config["width_m"])
        self.field_height_m = float(field_config["height_m"])
        self.field_frame_id = str(field_config.get("frame_id", "field"))
        self.robot_specs = parse_robot_specs(robot_config)
        self.heading_offset_rad = float(robot_config["heading_offset_rad"])

        self.calibrated_homography = None
        self.calibrated_boundary_points_px = None
        self.homography_version = 0

    def reset_calibration(self):
        self.calibrated_homography = None
        self.calibrated_boundary_points_px = None
        self.homography_version += 1

    def is_calibrated(self):
        return (
            self.calibrated_homography is not None
            and self.calibrated_boundary_points_px is not None
        )

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = self.detect_scale
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
            if int(tag_id) in tags and tags[int(tag_id)]["perimeter"] >= perimeter:
                continue

            tags[int(tag_id)] = {
                "id": int(tag_id),
                "corners": corners,
                "center": corners.mean(axis=0),
                "perimeter": perimeter,
            }
        return tags

    def _boundary_tags_from(self, tags):
        boundary = [tags.get(tag_id) for tag_id in self.boundary_tag_ids]
        missing_boundary = [
            tag_id
            for tag_id, tag in zip(self.boundary_tag_ids, boundary, strict=False)
            if tag is None
        ]
        return boundary, missing_boundary

    def _calibrate_from_tags(self, tags):
        boundary, missing_boundary = self._boundary_tags_from(tags)
        if missing_boundary:
            return False, missing_boundary

        image_points = np.float32([tag["center"] for tag in boundary])
        field_points = np.float32(
            [
                [0.0, 0.0],
                [self.field_width_m, 0.0],
                [self.field_width_m, self.field_height_m],
                [0.0, self.field_height_m],
            ]
        )
        self.calibrated_homography = cv2.getPerspectiveTransform(image_points, field_points)
        self.calibrated_boundary_points_px = image_points
        self.homography_version += 1
        return True, []

    def uncalibrated_state(self, tags):
        _, missing_boundary = self._boundary_tags_from(tags)
        return {
            "ready": False,
            "calibrated": False,
            "homography_version": self.homography_version,
            "missing_boundary": missing_boundary,
            "robots": [],
        }

    def map_state(self, tags, allow_calibration=True):
        _, missing_boundary = self._boundary_tags_from(tags)
        if not self.is_calibrated():
            if not allow_calibration:
                return self.uncalibrated_state(tags)
            calibrated, missing_boundary = self._calibrate_from_tags(tags)
            if not calibrated:
                return {
                    "ready": False,
                    "calibrated": False,
                    "homography_version": self.homography_version,
                    "missing_boundary": missing_boundary,
                    "robots": [],
                }

        homography = self.calibrated_homography
        robot_ids = set(self.robot_specs) if self.robot_specs else None
        robots = []
        for tag_id, tag in tags.items():
            if tag_id in self.boundary_tag_ids:
                continue
            if robot_ids is not None and tag_id not in robot_ids:
                continue

            spec = self.robot_specs.get(
                tag_id,
                RobotSpec(
                    tag_id=tag_id,
                    name=f"robot_{tag_id}",
                    ros_namespace=f"/robots/robot_{tag_id}",
                ),
            )

            floor_corners = transform_points(homography, tag["corners"])
            floor_center = floor_corners.mean(axis=0)
            heading_vec = floor_corners[1] - floor_corners[0]
            theta = math.atan2(float(heading_vec[1]), float(heading_vec[0]))
            theta += self.heading_offset_rad

            forward_m, left_m = spec.center_offset_m
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            robot_x = float(floor_center[0] + cos_t * forward_m - sin_t * left_m)
            robot_y = float(floor_center[1] + sin_t * forward_m + cos_t * left_m)

            robots.append(
                {
                    "id": tag_id,
                    "name": spec.name,
                    "ros_namespace": spec.ros_namespace,
                    "allow_global_state": spec.allow_global_state,
                    "x": robot_x,
                    "y": robot_y,
                    "theta": normalize_angle(theta),
                    "tag_center_px": tag["center"],
                    "tag_corners_px": tag["corners"],
                }
            )

        return {
            "ready": True,
            "calibrated": True,
            "homography_version": self.homography_version,
            "homography": homography,
            "boundary_points_px": self.calibrated_boundary_points_px,
            "missing_boundary": missing_boundary,
            "robots": robots,
        }

    def pose_estimates(self, frame_id: int, capture_unix_ns: int, state):
        return [
            RobotPoseEstimate(
                tag_id=int(robot["id"]),
                name=str(robot["name"]),
                ros_namespace=str(robot["ros_namespace"]),
                x=float(robot["x"]),
                y=float(robot["y"]),
                theta=float(robot["theta"]),
                frame_id=int(frame_id),
                capture_unix_ns=int(capture_unix_ns),
                visible=True,
            )
            for robot in state.get("robots", [])
        ]
