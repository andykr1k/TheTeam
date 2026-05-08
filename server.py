import asyncio
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import websockets
from websockets.exceptions import ConnectionClosed
from transformers import Sam3VideoModel, Sam3VideoProcessor


PROMPT = "robot"
HOST = "0.0.0.0"
PORT = 8765

MODEL_ID = "facebook/sam3"
DEVICE = "cuda:7"
DTYPE = torch.bfloat16

MIN_CONTOUR_AREA = 80
APPROX_EPSILON = 1.5
KEEP_PROCESSED_FRAMES = 4

HEADER = struct.Struct("!Q")


model = None
processor = None


def masks_to_objects(outputs):
    masks = outputs["masks"].detach().cpu().numpy()
    object_ids = outputs["object_ids"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()

    objects = []
    for mask, object_id, score in zip(masks, object_ids, scores, strict=False):
        mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        packed_contours = []
        total_area = 0.0
        total_m10 = 0.0
        total_m01 = 0.0
        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue
            moments = cv2.moments(contour)
            if moments["m00"]:
                total_area += moments["m00"]
                total_m10 += moments["m10"]
                total_m01 += moments["m01"]
            contour = cv2.approxPolyDP(contour, APPROX_EPSILON, closed=True)
            points = contour.reshape(-1, 2).astype(int)
            packed_contours.append(points.tolist())

        if packed_contours and total_area:
            center_x = total_m10 / total_area
            center_y = total_m01 / total_area
            objects.append(
                {
                    "id": int(object_id),
                    "score": float(score),
                    "center": [int(center_x), int(center_y)],
                    "contours": packed_contours,
                }
            )
    return objects


def drop_old_session_frames(session, current_frame_idx):
    frames = getattr(session, "processed_frames", None)
    if not isinstance(frames, dict):
        return
    min_keep_idx = current_frame_idx - KEEP_PROCESSED_FRAMES + 1
    for frame_idx in list(frames):
        if frame_idx < min_keep_idx:
            del frames[frame_idx]


def infer_frame(session, client_frame_id, session_frame_id, jpg_bytes, recv_time):
    jpg = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return {"frame_id": client_frame_id, "error": "could not decode JPEG"}

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    torch.cuda.synchronize(DEVICE)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        inputs = processor(images=frame_rgb, device=DEVICE, return_tensors="pt").to(DEVICE)
        model_outputs = model(
            inference_session=session,
            frame_idx=session_frame_id,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        outputs = processor.postprocess_outputs(
            session,
            model_outputs,
            original_sizes=inputs.original_sizes,
        )
    torch.cuda.synchronize(DEVICE)

    inference_ms = (time.perf_counter() - start) * 1000.0
    objects = masks_to_objects(outputs)
    drop_old_session_frames(session, session_frame_id)

    return {
        "frame_id": client_frame_id,
        "inference_ms": inference_ms,
        "server_ms": (time.perf_counter() - recv_time) * 1000.0,
        "objects": objects,
    }


async def handle_client(websocket):
    latest = None
    latest_lock = asyncio.Lock()
    stop = asyncio.Event()
    executor = ThreadPoolExecutor(max_workers=1)

    session = processor.init_video_session(
        inference_device=DEVICE,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=DTYPE,
        max_vision_features_cache_size=1,
    )
    session = processor.add_text_prompt(session, PROMPT)

    async def receiver():
        nonlocal latest
        try:
            async for message in websocket:
                if not isinstance(message, bytes) or len(message) <= HEADER.size:
                    continue
                frame_id = HEADER.unpack_from(message)[0]
                jpg_bytes = bytes(memoryview(message)[HEADER.size :])
                async with latest_lock:
                    latest = (frame_id, jpg_bytes, time.perf_counter())
        finally:
            stop.set()

    async def inference_sender():
        nonlocal latest
        loop = asyncio.get_running_loop()
        session_frame_id = 0

        while not stop.is_set():
            async with latest_lock:
                item = latest
                latest = None

            if item is None:
                await asyncio.sleep(0.001)
                continue

            client_frame_id, jpg_bytes, recv_time = item
            result = await loop.run_in_executor(
                executor,
                infer_frame,
                session,
                client_frame_id,
                session_frame_id,
                jpg_bytes,
                recv_time,
            )
            session_frame_id += 1

            try:
                await websocket.send(json.dumps(result, separators=(",", ":")))
            except ConnectionClosed:
                stop.set()
                break

    tasks = [asyncio.create_task(receiver()), asyncio.create_task(inference_sender())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        executor.shutdown(wait=False, cancel_futures=True)


async def main():
    global model, processor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(DEVICE)
    model = Sam3VideoModel.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    processor = Sam3VideoProcessor.from_pretrained(MODEL_ID)

    async with websockets.serve(
        handle_client,
        HOST,
        PORT,
        max_size=8 * 1024 * 1024,
        max_queue=1,
        compression=None,
    ):
        print(f"listening ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
