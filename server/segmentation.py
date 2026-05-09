import time
from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from common.messages import ContourPacket, FramePacket


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def masks_to_contour_objects(outputs: dict[str, Any], min_contour_area: float, approx_epsilon: float):
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
            if cv2.contourArea(contour) < min_contour_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"]:
                total_area += moments["m00"]
                total_m10 += moments["m10"]
                total_m01 += moments["m01"]
            contour = cv2.approxPolyDP(contour, approx_epsilon, closed=True)
            points = contour.reshape(-1, 2).astype(int)
            packed_contours.append(points.tolist())

        if packed_contours and total_area:
            center_x = total_m10 / total_area
            center_y = total_m01 / total_area
            objects.append(
                {
                    "id": int(object_id),
                    "score": float(score),
                    "center_px": [int(center_x), int(center_y)],
                    "contours_px": packed_contours,
                }
            )
    return objects


def drop_old_session_frames(session, current_frame_idx: int, keep_processed_frames: int):
    frames = getattr(session, "processed_frames", None)
    if not isinstance(frames, dict):
        return
    min_keep_idx = current_frame_idx - keep_processed_frames + 1
    for frame_idx in list(frames):
        if frame_idx < min_keep_idx:
            del frames[frame_idx]


class Sam3Segmenter:
    def __init__(self, config: dict[str, Any]):
        self.prompt = str(config["prompt"])
        self.model_id = str(config["model_id"])
        self.device = str(config["device"])
        self.dtype_name = str(config["dtype"])
        self.dtype = DTYPES[self.dtype_name]
        self.min_contour_area = float(config["min_contour_area"])
        self.approx_epsilon = float(config["approx_epsilon"])
        self.keep_processed_frames = int(config["keep_processed_frames"])
        self.model = None
        self.processor = None

    def load(self):
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            torch.cuda.set_device(self.device)

        self.model = Sam3VideoModel.from_pretrained(self.model_id, dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(self.model_id)

    def create_session(self):
        if self.processor is None:
            raise RuntimeError("SAM3 processor is not loaded")
        session = self.processor.init_video_session(
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
            max_vision_features_cache_size=1,
        )
        return self.processor.add_text_prompt(session, self.prompt)

    def _sync_device(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def _autocast_context(self):
        if self.device.startswith("cuda") and self.dtype is not torch.float32:
            return torch.autocast("cuda", dtype=self.dtype)
        return nullcontext()

    def infer_jpeg(
        self,
        session,
        packet: FramePacket,
        session_frame_id: int,
        jpg_bytes: bytes,
        server_recv_unix_ns: int,
        server_recv_monotonic_ns: int,
        server_stats: dict[str, Any] | None = None,
    ) -> ContourPacket:
        jpg = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            now_ns = time.time_ns()
            return ContourPacket(
                frame_id=packet.frame_id,
                capture_unix_ns=packet.capture_unix_ns,
                capture_monotonic_ns=packet.capture_monotonic_ns,
                server_recv_unix_ns=server_recv_unix_ns,
                server_done_unix_ns=now_ns,
                inference_ms=0.0,
                server_ms=(time.monotonic_ns() - server_recv_monotonic_ns) / 1_000_000.0,
                objects=[],
                server_stats=server_stats or {},
                error="could not decode JPEG",
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        self._sync_device()
        start_ns = time.monotonic_ns()
        with torch.inference_mode(), self._autocast_context():
            inputs = self.processor(images=frame_rgb, device=self.device, return_tensors="pt")
            inputs = inputs.to(self.device)
            model_outputs = self.model(
                inference_session=session,
                frame_idx=session_frame_id,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
            outputs = self.processor.postprocess_outputs(
                session,
                model_outputs,
                original_sizes=inputs.original_sizes,
            )
        self._sync_device()

        inference_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        objects = masks_to_contour_objects(
            outputs,
            self.min_contour_area,
            self.approx_epsilon,
        )
        drop_old_session_frames(session, session_frame_id, self.keep_processed_frames)
        now_mono_ns = time.monotonic_ns()

        return ContourPacket(
            frame_id=packet.frame_id,
            capture_unix_ns=packet.capture_unix_ns,
            capture_monotonic_ns=packet.capture_monotonic_ns,
            server_recv_unix_ns=server_recv_unix_ns,
            server_done_unix_ns=time.time_ns(),
            inference_ms=inference_ms,
            server_ms=(now_mono_ns - server_recv_monotonic_ns) / 1_000_000.0,
            objects=objects,
            server_stats=server_stats or {},
        )
