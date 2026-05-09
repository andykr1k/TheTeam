import json
from dataclasses import asdict, dataclass, field
from typing import Any


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def json_loads(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise ValueError("protocol message must be a JSON object")
    return payload


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    capture_unix_ns: int
    capture_monotonic_ns: int
    width: int
    height: int
    homography_version: int
    encoding: str = "jpeg"
    payload_size: int = 0
    chunks: int = 1

    def to_start_message(self) -> str:
        payload = asdict(self)
        payload["type"] = "frame_start"
        return json_dumps(payload)

    @classmethod
    def from_start_message(cls, payload: dict[str, Any]) -> "FramePacket":
        return cls(
            frame_id=int(payload["frame_id"]),
            capture_unix_ns=int(payload["capture_unix_ns"]),
            capture_monotonic_ns=int(payload["capture_monotonic_ns"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            homography_version=int(payload.get("homography_version", 0)),
            encoding=str(payload.get("encoding", "jpeg")),
            payload_size=int(payload.get("payload_size", 0)),
            chunks=int(payload.get("chunks", 1)),
        )


@dataclass(slots=True)
class ContourObject:
    id: int
    score: float
    center_px: list[int] | None
    contours_px: list[list[list[int]]]


@dataclass(slots=True)
class ContourPacket:
    frame_id: int
    capture_unix_ns: int
    capture_monotonic_ns: int
    server_recv_unix_ns: int
    server_done_unix_ns: int
    inference_ms: float
    server_ms: float
    objects: list[dict[str, Any]] = field(default_factory=list)
    server_stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_message(self) -> str:
        payload = asdict(self)
        payload["type"] = "contours"
        return json_dumps(payload)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ContourPacket":
        return cls(
            frame_id=int(payload["frame_id"]),
            capture_unix_ns=int(payload.get("capture_unix_ns", 0)),
            capture_monotonic_ns=int(payload.get("capture_monotonic_ns", 0)),
            server_recv_unix_ns=int(payload.get("server_recv_unix_ns", 0)),
            server_done_unix_ns=int(payload.get("server_done_unix_ns", 0)),
            inference_ms=float(payload.get("inference_ms", 0.0)),
            server_ms=float(payload.get("server_ms", 0.0)),
            objects=list(payload.get("objects", [])),
            server_stats=dict(payload.get("server_stats", {})),
            error=payload.get("error"),
        )


@dataclass(slots=True)
class RobotPoseEstimate:
    tag_id: int
    name: str
    ros_namespace: str
    x: float
    y: float
    theta: float
    frame_id: int
    capture_unix_ns: int
    visible: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
