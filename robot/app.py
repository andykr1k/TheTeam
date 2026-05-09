import json
import math
import os
import socket
import sys
import time

from config import load_config


CONFIG = load_config()
ROS2_CONFIG = CONFIG["ros2"]
ROBOT_RUNTIME = CONFIG["robot_runtime"]


def configure_ros2_environment():
    os.environ["ROS_DOMAIN_ID"] = str(int(ROS2_CONFIG["domain_id"]))
    rmw = str(ROS2_CONFIG["rmw_implementation"])
    if rmw:
        os.environ["RMW_IMPLEMENTATION"] = rmw

    lan_config = ROS2_CONFIG.get("private_lan", {})
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


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_unix_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def main():
    configure_ros2_environment()

    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped
    from rclpy import qos as rclpy_qos
    from rclpy.node import Node
    from std_msgs.msg import String

    qos_config = ROS2_CONFIG["qos"]
    reliability = (
        rclpy_qos.ReliabilityPolicy.BEST_EFFORT
        if str(qos_config["pose_reliability"]).lower() == "best_effort"
        else rclpy_qos.ReliabilityPolicy.RELIABLE
    )
    qos_profile = rclpy_qos.QoSProfile(
        reliability=reliability,
        history=rclpy_qos.HistoryPolicy.KEEP_LAST,
        depth=int(qos_config["pose_history_depth"]),
    )

    robot_name = str(ROBOT_RUNTIME["name"])
    namespace = str(ROBOT_RUNTIME["ros_namespace"]).rstrip("/")

    class RobotPoseSubscriber(Node):
        def __init__(self):
            node_name = f"{robot_name}_pose_listener".replace("/", "_")
            super().__init__(node_name)
            self.create_subscription(PoseStamped, f"{namespace}/pose", self.pose_callback, qos_profile)
            self.create_subscription(
                PointStamped,
                f"{namespace}/location",
                self.location_callback,
                qos_profile,
            )
            self.create_subscription(
                String,
                f"{namespace}/tracking_status",
                self.status_callback,
                qos_profile,
            )
            print(
                f"listening for {robot_name} on {namespace}/pose and {namespace}/location",
                file=sys.stderr,
            )

        def pose_callback(self, msg):
            payload = {
                "type": "pose",
                "robot": robot_name,
                "frame_id": msg.header.frame_id,
                "timestamp_ns": stamp_to_unix_ns(msg.header.stamp),
                "received_unix_ns": time.time_ns(),
                "x": float(msg.pose.position.x),
                "y": float(msg.pose.position.y),
                "theta": yaw_from_quaternion(msg.pose.orientation),
            }
            print(json.dumps(payload, separators=(",", ":")), flush=True)

        def location_callback(self, msg):
            payload = {
                "type": "location",
                "robot": robot_name,
                "frame_id": msg.header.frame_id,
                "timestamp_ns": stamp_to_unix_ns(msg.header.stamp),
                "received_unix_ns": time.time_ns(),
                "x": float(msg.point.x),
                "y": float(msg.point.y),
                "z": float(msg.point.z),
            }
            print(json.dumps(payload, separators=(",", ":")), flush=True)

        def status_callback(self, msg):
            print(msg.data, file=sys.stderr)

    rclpy.init(args=None)
    node = RobotPoseSubscriber()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
