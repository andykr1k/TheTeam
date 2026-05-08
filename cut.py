import cv2

input_path = "data/video2.mp4"
output_path = "data/output_38_52.mp4"

start_sec = 38
end_sec = 52

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'X','2','6','4' if supported
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

# Clamp to valid range
start_frame = max(0, min(start_frame, total_frames - 1))
end_frame = max(0, min(end_frame, total_frames))

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

current_frame = start_frame
while current_frame < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    current_frame += 1

cap.release()
out.release()
