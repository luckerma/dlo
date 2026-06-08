from pathlib import Path
from time import time

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Webcam or video?
CAMERA: int | None = 0  # Set to None to use video
VIDEO_PATH: Path = Path(__file__).resolve(strict=True).parent / "data" / "video.MOV"

# YOLO model
# https://docs.ultralytics.com/models/yolo11/
model: YOLO = YOLO("yolo11n.pt")


def detect_objects(frame: np.ndarray) -> np.ndarray:
    """Detect objects in a frame using YOLOv8 and return the annotated frame.

    Args:
        frame: Input image frame as a NumPy array.

    Returns:
        Annotated frame with detected objects as a NumPy array.
    """

    results: list[Results] = model(frame)
    annotated_frame: np.ndarray = results[0].plot()

    return annotated_frame


def process_video(video_path: str | Path) -> None:
    """Process a video file for object detection and display results.

    Args:
        video_path: Path to the video file, can be a string or Path object.
    """

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Skip frames

    for _ in range(640):
        cap.read()

    prev_time = time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        annotated_frame = detect_objects(frame)

        # FPS
        current_time = time()
        delta_time = current_time - prev_time
        fps = 1 / delta_time if delta_time > 0 else 0
        prev_time = current_time
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO Object Detection - Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_webcam(camera_id: int = 0) -> None:
    """Process real-time webcam footage for object detection and display results."""

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return

    prev_time = time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = detect_objects(frame)

        # FPS
        current_time = time()
        delta_time = current_time - prev_time
        fps = 1 / delta_time if delta_time > 0 else 0
        prev_time = current_time
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO Object Detection - Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main function to run the script
if __name__ == "__main__":
    process_webcam(CAMERA) if CAMERA is not None else process_video(VIDEO_PATH)
