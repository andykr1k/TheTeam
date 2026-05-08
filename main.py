import time

import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor


PROMPT = "robot"
VIDEO_PATH = "data/video1.mp4"
OUTPUT_VIDEO_PATH = "data/output_video1.mp4"

MODEL_ID = "facebook/sam3"
DEVICE = "cuda:7"
DTYPE = torch.bfloat16


def draw_contours(frame, masks, inference_seconds):
    for i, mask in enumerate(masks):
        mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (
            int(37 * (i + 3) % 255),
            int(97 * (i + 5) % 255),
            int(173 * (i + 7) % 255),
        )
        cv2.drawContours(frame, contours, -1, color, 2)

    cv2.putText(
        frame,
        f"{inference_seconds:.3f}s/frame",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    if not PROMPT:
        raise ValueError('Set PROMPT = "..."')
    if not VIDEO_PATH:
        raise ValueError('Set VIDEO_PATH = "..."')
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your PyTorch/CUDA install.")

    torch.cuda.set_device(DEVICE)
    model = Sam3VideoModel.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    processor = Sam3VideoProcessor.from_pretrained(MODEL_ID)
    session = processor.init_video_session(
        inference_device=DEVICE,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=DTYPE,
    )
    session = processor.add_text_prompt(session, PROMPT)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        torch.cuda.synchronize(DEVICE)
        start = time.perf_counter()

        with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
            inputs = processor(images=frame_rgb, device=DEVICE, return_tensors="pt").to(
                DEVICE
            )
            model_outputs = model(session, frame=inputs.pixel_values[0])
            outputs = processor.postprocess_outputs(
                session,
                model_outputs,
                original_sizes=inputs.original_sizes,
            )

        torch.cuda.synchronize(DEVICE)
        inference_seconds = time.perf_counter() - start

        masks = outputs["masks"].detach().cpu().numpy()
        writer.write(draw_contours(frame_bgr, masks, inference_seconds))
        print(f"frame={frame_idx} inference_seconds={inference_seconds:.4f}")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"output_video={OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
