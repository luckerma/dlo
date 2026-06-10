import argparse
from pathlib import Path
from time import perf_counter
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

Frame = NDArray[np.uint8]

BASE_DIR: Final[Path] = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH: Final[Path] = (
    BASE_DIR / "yolo26n.pt"
)  # https://docs.ultralytics.com/models/yolo26#supported-tasks-and-modes
DEFAULT_VIDEO_PATH: Final[Path] = BASE_DIR / "data" / "video.MOV"
DEFAULT_START_FRAME: Final[int] = 0
QUIT_KEY: Final[str] = "q"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> YOLO:
    """Load a YOLO model without doing work at import time."""

    return YOLO(str(model_path))


def detect_objects(frame: Frame, model: YOLO) -> Frame:
    """Detect objects in a frame using YOLO and return the annotated frame.

    Args:
        frame: Input image frame as a NumPy array.
        model: Loaded YOLO model.

    Returns:
        Annotated frame with detected objects as a NumPy array.
    """

    results: list[Results] = model.predict(frame, verbose=False)
    annotated_frame: Frame = frame.copy()
    for result in results:
        annotated_frame = result.plot()

    return annotated_frame


def draw_fps(frame: Frame, fps: float) -> None:
    """Draw frames per second on the frame in-place."""

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )


def open_capture(source: int | str | Path) -> cv2.VideoCapture:
    """Open a webcam or video file."""

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        msg = f"Could not open video source: {source}"
        raise RuntimeError(msg)

    return cap


def process_capture(
    cap: cv2.VideoCapture,
    model: YOLO,
    window_name: str,
) -> None:
    """Read frames, run detection, and display annotated output."""

    previous_time = perf_counter()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = detect_objects(frame, model)

            current_time = perf_counter()
            delta_time = current_time - previous_time
            previous_time = current_time
            draw_fps(annotated_frame, 1 / delta_time if delta_time > 0 else 0)

            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def process_video(
    video_path: str | Path,
    model: YOLO,
    *,
    start_frame: int = DEFAULT_START_FRAME,
) -> None:
    """Process a video file for object detection and display results.

    Args:
        video_path: Path to the video file, can be a string or Path object.
        model: Loaded YOLO model.
        start_frame: First video frame to process.
    """

    cap = open_capture(Path(video_path))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    process_capture(
        cap,
        model,
        "YOLO Object Detection -- Video",
    )


def process_webcam(camera_id: int, model: YOLO) -> None:
    """Process real-time webcam footage for object detection and display results."""

    cap = open_capture(camera_id)
    process_capture(cap, model, "YOLO Object Detection -- Webcam")


def non_negative_int(value: str) -> int:
    """Parse a command-line integer that must be zero or greater."""

    parsed_value = int(value)
    if parsed_value < 0:
        msg = "value must be zero or greater"
        raise argparse.ArgumentTypeError(msg)

    return parsed_value


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the demo."""

    parser = argparse.ArgumentParser(description="Run YOLO object detection.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--camera",
        type=int,
        help="webcam index to use instead of the default video",
    )
    source_group.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help=f"video path to process (default: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"YOLO model path or name (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--start-frame",
        type=non_negative_int,
        default=DEFAULT_START_FRAME,
        help=f"first video frame to process (default: {DEFAULT_START_FRAME})",
    )

    return parser.parse_args()


def main() -> None:
    """Load the model and run detection on the chosen source."""

    args = parse_args()
    model = load_model(args.model)

    try:
        if args.camera is not None:
            process_webcam(args.camera, model)
        else:
            process_video(
                args.video,
                model,
                start_frame=args.start_frame,
            )
    except RuntimeError as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
