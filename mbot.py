import asyncio
import json
import sys

import websockets
from websockets.exceptions import ConnectionClosed

from config import load_config


CONFIG = load_config()
MBOT_CONFIG = CONFIG.get("mbot", {})

POSE_STREAM_URL = str(MBOT_CONFIG.get("pose_stream_url", "ws://127.0.0.1:8766"))
ROBOT_TAG_ID = MBOT_CONFIG.get("robot_tag_id")
RECONNECT_DELAY_S = float(MBOT_CONFIG.get("reconnect_delay_s", 1.0))


def selected_robot_pose(payload, robot_tag_id):
    for robot in payload.get("robots", []):
        if int(robot.get("id", -1)) != int(robot_tag_id):
            continue
        return {
            "tag_id": int(robot["id"]),
            "name": str(robot["name"]),
            "x": float(robot["x"]),
            "y": float(robot["y"]),
            "theta": float(robot["theta"]),
            "frame_id": int(payload.get("frame_id", 0)),
            "timestamp": float(payload.get("timestamp", 0.0)),
        }
    return None


async def stream_selected_robot_pose():
    if ROBOT_TAG_ID is None:
        raise RuntimeError("Set mbot.robot_tag_id in config.yaml before running mbot.py.")

    last_status = None
    while True:
        try:
            async with websockets.connect(
                POSE_STREAM_URL,
                max_queue=1,
                compression=None,
            ) as websocket:
                print(
                    f"listening for robot tag {int(ROBOT_TAG_ID)} on {POSE_STREAM_URL}",
                    file=sys.stderr,
                )
                async for message in websocket:
                    payload = json.loads(message)

                    if not payload.get("ready", False):
                        status = (
                            "waiting for floor calibration; "
                            f"missing boundary tags: {payload.get('missing_boundary', [])}"
                        )
                        if status != last_status:
                            print(status, file=sys.stderr)
                            last_status = status
                        continue

                    pose = selected_robot_pose(payload, ROBOT_TAG_ID)
                    if pose is None:
                        status = f"robot tag {int(ROBOT_TAG_ID)} is not currently visible"
                        if status != last_status:
                            print(status, file=sys.stderr)
                            last_status = status
                        continue

                    last_status = "tracking"
                    print(json.dumps(pose, separators=(",", ":")))
        except (OSError, ConnectionClosed) as exc:
            print(
                f"pose stream disconnected: {exc}. "
                f"retrying in {RECONNECT_DELAY_S:.1f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(RECONNECT_DELAY_S)


if __name__ == "__main__":
    asyncio.run(stream_selected_robot_pose())
