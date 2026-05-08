import asyncio
import json
import struct
import time
from collections import OrderedDict

import cv2
import numpy as np
import websockets


SERVER_URL = "ws://127.0.0.1:8765"
VIDEO_SOURCE = 0

JPEG_QUALITY = 60
TARGET_FPS = 30
MAX_PENDING_FRAMES = 90
WINDOW_NAME = "SAM3 contours"

HEADER = struct.Struct("!Q")


def color_for_id(object_id):
    return (
        int(37 * (object_id + 3) % 255),
        int(97 * (object_id + 5) % 255),
        int(173 * (object_id + 7) % 255),
    )


def draw_result(frame, result, e2e_ms):
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
            cv2.putText(
                frame,
                f"({x},{y})",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

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
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    frame_id = 0
    frame_period = 1.0 / TARGET_FPS if TARGET_FPS else 0.0

    try:
        while not stop.is_set():
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                stop.set()
                break

            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not ok:
                continue

            pending[frame_id] = (frame, time.perf_counter())
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
        cap.release()


async def receive_contours(websocket, pending, stop):
    async for message in websocket:
        result = json.loads(message)
        frame_id = result["frame_id"]
        item = pending.pop(frame_id, None)
        if item is None:
            continue

        frame, sent_time = item
        e2e_ms = (time.perf_counter() - sent_time) * 1000.0
        frame = draw_result(frame, result, e2e_ms)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        print(
            f"frame={frame_id} "
            f"inference_ms={result.get('inference_ms', 0.0):.1f} "
            f"e2e_ms={e2e_ms:.1f} "
            f"objects={len(result.get('objects', []))}"
        )
        if key in (27, ord("q")):
            stop.set()
            break


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
