import json
import math
import os
import socket
import threading
import time
from typing import Any

from common.messages import RobotPoseEstimate
from vision.tracking import RobotSpec


def _stamp_from_unix_ns(stamp, unix_ns: int):
    stamp.sec = int(unix_ns // 1_000_000_000)
    stamp.nanosec = int(unix_ns % 1_000_000_000)


def _configure_private_lan_environment(ros2_config: dict[str, Any]):
    os.environ["ROS_DOMAIN_ID"] = str(int(ros2_config["domain_id"]))
    rmw = str(ros2_config["rmw_implementation"])
    if rmw:
        os.environ["RMW_IMPLEMENTATION"] = rmw

    lan_config = ros2_config.get("private_lan", {})
    interface = str(lan_config.get("interface", ""))
    bind_address = str(lan_config.get("bind_address", ""))
    enforce_interface = bool(lan_config.get("enforce_interface", False))

    if interface:
        interfaces = {name for _, name in socket.if_nameindex()}
        if enforce_interface and interface not in interfaces:
            raise RuntimeError(
                f"Configured ROS2 private LAN interface {interface!r} was not found. "
                f"Available interfaces: {sorted(interfaces)}"
            )

    if rmw == "rmw_cyclonedds_cpp" and "CYCLONEDDS_URI" not in os.environ:
        network_target = bind_address or interface
        if network_target:
            os.environ["CYCLONEDDS_URI"] = (
                "<CycloneDDS><Domain><General>"
                f"<NetworkInterfaceAddress>{network_target}</NetworkInterfaceAddress>"
                "</General></Domain></CycloneDDS>"
            )


def _qos_profile(rclpy_qos, qos_config: dict[str, Any]):
    reliability_name = str(qos_config["pose_reliability"]).lower()
    reliability = (
        rclpy_qos.ReliabilityPolicy.BEST_EFFORT
        if reliability_name == "best_effort"
        else rclpy_qos.ReliabilityPolicy.RELIABLE
    )
    return rclpy_qos.QoSProfile(
        reliability=reliability,
        history=rclpy_qos.HistoryPolicy.KEEP_LAST,
        depth=int(qos_config["pose_history_depth"]),
    )


class Ros2Gateway:
    def __init__(
        self,
        ros2_config: dict[str, Any],
        robot_specs: dict[int, RobotSpec],
        field_frame_id: str,
        publish_global_state: bool,
    ):
        self.config = ros2_config
        self.robot_specs = robot_specs
        self.field_frame_id = field_frame_id
        self.publish_global_state = bool(publish_global_state)
        self.enabled = bool(ros2_config["enabled"])
        self.state = "disabled" if not self.enabled else "new"
        self.last_error = ""
        self.node = None
        self.rclpy = None
        self.thread: threading.Thread | None = None
        self.publish_counts = {spec.name: 0 for spec in robot_specs.values()}
        self.status_counts = {spec.name: 0 for spec in robot_specs.values()}

    def start(self):
        if not self.enabled:
            return

        try:
            _configure_private_lan_environment(self.config)
            import rclpy
            from geometry_msgs.msg import PointStamped, PoseStamped
            from rclpy import qos as rclpy_qos
            from rclpy.node import Node
            from std_msgs.msg import String

            qos_profile = _qos_profile(rclpy_qos, self.config["qos"])
            field_frame_id = self.field_frame_id
            robot_specs = self.robot_specs
            publish_global_state = self.publish_global_state

            class GatewayNode(Node):
                def __init__(self):
                    super().__init__("theteam_client_gateway")
                    self.pose_publishers = {}
                    self.location_publishers = {}
                    self.status_publishers = {}
                    for spec in robot_specs.values():
                        namespace = spec.ros_namespace.rstrip("/")
                        self.pose_publishers[spec.name] = self.create_publisher(
                            PoseStamped,
                            f"{namespace}/pose",
                            qos_profile,
                        )
                        self.location_publishers[spec.name] = self.create_publisher(
                            PointStamped,
                            f"{namespace}/location",
                            qos_profile,
                        )
                        self.status_publishers[spec.name] = self.create_publisher(
                            String,
                            f"{namespace}/tracking_status",
                            qos_profile,
                        )
                    self.global_publisher = None
                    if publish_global_state:
                        self.global_publisher = self.create_publisher(
                            String,
                            "/team/tracking_state",
                            qos_profile,
                        )

                def publish_pose(self, estimate: RobotPoseEstimate):
                    pose = PoseStamped()
                    _stamp_from_unix_ns(pose.header.stamp, estimate.capture_unix_ns)
                    pose.header.frame_id = field_frame_id
                    pose.pose.position.x = float(estimate.x)
                    pose.pose.position.y = float(estimate.y)
                    pose.pose.position.z = 0.0
                    half_theta = float(estimate.theta) / 2.0
                    pose.pose.orientation.z = math.sin(half_theta)
                    pose.pose.orientation.w = math.cos(half_theta)
                    self.pose_publishers[estimate.name].publish(pose)

                    point = PointStamped()
                    _stamp_from_unix_ns(point.header.stamp, estimate.capture_unix_ns)
                    point.header.frame_id = field_frame_id
                    point.point.x = float(estimate.x)
                    point.point.y = float(estimate.y)
                    point.point.z = 0.0
                    self.location_publishers[estimate.name].publish(point)

                def publish_status(self, spec: RobotSpec, status: dict[str, Any]):
                    message = String()
                    message.data = json.dumps(status, separators=(",", ":"))
                    self.status_publishers[spec.name].publish(message)

                def publish_global(self, payload: dict[str, Any]):
                    if self.global_publisher is None:
                        return
                    message = String()
                    message.data = json.dumps(payload, separators=(",", ":"))
                    self.global_publisher.publish(message)

            if not rclpy.ok():
                rclpy.init(args=None)
            self.rclpy = rclpy
            self.node = GatewayNode()
            self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            self.thread.start()
            self.state = "running"
        except Exception as exc:
            self.state = "error"
            self.last_error = str(exc)
            if bool(self.config.get("private_lan", {}).get("enforce_interface", False)):
                raise

    def publish(self, estimates: list[RobotPoseEstimate], frame_id: int, capture_unix_ns: int):
        if self.state != "running" or self.node is None:
            return

        visible_names = set()
        for estimate in estimates:
            visible_names.add(estimate.name)
            self.node.publish_pose(estimate)
            self.publish_counts[estimate.name] = self.publish_counts.get(estimate.name, 0) + 1
            spec = self.robot_specs.get(estimate.tag_id)
            if spec is not None:
                self.node.publish_status(
                    spec,
                    {
                        "visible": True,
                        "frame_id": int(frame_id),
                        "capture_unix_ns": int(capture_unix_ns),
                        "updated_unix_ns": time.time_ns(),
                    },
                )
                self.status_counts[spec.name] = self.status_counts.get(spec.name, 0) + 1

        for spec in self.robot_specs.values():
            if spec.name in visible_names:
                continue
            self.node.publish_status(
                spec,
                {
                    "visible": False,
                    "frame_id": int(frame_id),
                    "capture_unix_ns": int(capture_unix_ns),
                    "updated_unix_ns": time.time_ns(),
                },
            )
            self.status_counts[spec.name] = self.status_counts.get(spec.name, 0) + 1

        if self.publish_global_state:
            allowed_estimates = [
                estimate
                for estimate in estimates
                if self.robot_specs.get(estimate.tag_id) is not None
                and self.robot_specs[estimate.tag_id].allow_global_state
            ]
            self.node.publish_global(
                {
                    "frame_id": int(frame_id),
                    "capture_unix_ns": int(capture_unix_ns),
                    "robots": [estimate.to_json_dict() for estimate in allowed_estimates],
                }
            )

    def stats(self):
        return {
            "enabled": self.enabled,
            "state": self.state,
            "last_error": self.last_error,
            "domain_id": int(self.config["domain_id"]),
            "rmw_implementation": str(self.config["rmw_implementation"]),
            "publish_counts": self.publish_counts,
            "status_counts": self.status_counts,
        }

    def stop(self):
        if self.node is not None:
            self.node.destroy_node()
        if self.rclpy is not None and self.rclpy.ok():
            self.rclpy.shutdown()
        self.state = "stopped" if self.enabled else "disabled"
