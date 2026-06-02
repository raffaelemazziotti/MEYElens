"""
Record raw camera video with synchronized keyboard triggers.

This script opens a MEYELens camera preview and lets you record:

    1. raw camera video
    2. frame-level timestamps
    3. keyboard trigger pulses

No online pupil prediction is performed during acquisition. This makes the
recording loop lightweight and useful when you want to run pupil detection
offline later.

Keyboard controls
-----------------
s       start recording in a new timestamped folder
e       stop recording and finalize files
1-9     send a one-frame trigger pulse
q       quit

Output
------
Each recording creates a folder like:

    recordings/20260602_153012/

containing:

    pupillometry.avi
    expinfo.csv
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2

from meyelens.camera import Camera
from meyelens.fileio import FileWriter
from meyelens.offline import FrameRateManager
from meyelens.utils import CountdownTimer


# ============================================================
# User settings
# ============================================================

CAMERA_INDEX = 0

OUTPUT_DIR = Path("recordings")

TARGET_FPS = 30.0
MAX_DURATION_SEC = None  # example: 60.0 for automatic stop after 60 seconds

FLIP_VERTICAL = False

VIDEO_FILENAME = "pupillometry.avi"
TRIGGER_FILENAME = "expinfo.csv"

MOVIE_CODEC = "MJPG"  # alternatives: "XVID", "mp4v"


# ============================================================
# Drawing
# ============================================================

def draw_hud(frame, is_recording: bool) -> None:
    """
    Draw a simple heads-up display on the live preview.

    The HUD is drawn after the raw frame has been written to video, so the saved
    video remains clean.
    """
    lines = [
        f"Recording: {'ON' if is_recording else 'OFF'}",
        "s: start  e: stop  q: quit",
        "1-9: trigger pulse",
    ]

    y = 25

    for line in lines:
        # Black outline for readability.
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        # White foreground text.
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 24


# ============================================================
# Recording helpers
# ============================================================

def make_session_dir(output_dir: Path) -> Path:
    """
    Create and return a timestamped recording folder.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def open_video_writer(
    session_dir: Path,
    frame_shape: Tuple[int, int, int],
    fps: float,
) -> cv2.VideoWriter:
    """
    Create the OpenCV video writer for the raw camera stream.
    """
    height, width = frame_shape[:2]
    video_path = session_dir / VIDEO_FILENAME

    fourcc = cv2.VideoWriter_fourcc(*MOVIE_CODEC)

    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        float(fps),
        (int(width), int(height)),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    return writer


def open_trigger_writer(session_dir: Path) -> FileWriter:
    """
    Create the trigger CSV writer.

    The file is first created using the FileWriter internal naming convention.
    At the end of the session it is renamed to expinfo.csv.
    """
    headers = [
        "frame_index",
        "timestamp",
        "signal",
        "trial",
        "trg1",
        "trg2",
        "trg3",
        "trg4",
        "trg5",
        "trg6",
        "trg7",
        "trg8",
        "trg9",
    ]

    writer = FileWriter(
        session_dir,
        filename="expinfo",
        sep=",",
        headers=headers,
    )

    writer.write(f"# fps: {TARGET_FPS}")
    writer.write(f"# video_file: {VIDEO_FILENAME}")

    return writer


def start_session(
    frame_shape: Tuple[int, int, int],
    output_dir: Path,
) -> Tuple[Path, FileWriter, cv2.VideoWriter]:
    """
    Start a new recording session.
    """
    session_dir = make_session_dir(output_dir)

    trigger_writer = open_trigger_writer(session_dir)
    video_writer = open_video_writer(
        session_dir=session_dir,
        frame_shape=frame_shape,
        fps=TARGET_FPS,
    )

    print(f"### RECORDER ### Started recording: {session_dir}")

    return session_dir, trigger_writer, video_writer


def stop_session(
    session_dir: Path,
    trigger_writer: FileWriter,
    video_writer: cv2.VideoWriter,
) -> None:
    """
    Stop recording and finalize the output files.
    """
    video_writer.release()
    trigger_writer.close()

    final_csv_path = session_dir / TRIGGER_FILENAME

    if trigger_writer.path.exists():
        if final_csv_path.exists():
            final_csv_path.unlink()

        trigger_writer.path.rename(final_csv_path)

    print(f"### RECORDER ### Stopped recording: {session_dir}")


def trigger_signal(trigger_pulse) -> int:
    """
    Return the compact trigger code.

    0 means no trigger.
    1-9 indicates the first active trigger channel.
    """
    for index, value in enumerate(trigger_pulse):
        if value:
            return index + 1

    return 0


# ============================================================
# Main loop
# ============================================================

def main() -> None:
    """
    Run the preview/recording loop.
    """
    cam = Camera(camera_index=CAMERA_INDEX)

    frame_manager = FrameRateManager(
        fps=TARGET_FPS,
        duration=1e9,
    )
    frame_manager.start()

    is_recording = False

    session_dir: Optional[Path] = None
    trigger_writer: Optional[FileWriter] = None
    video_writer: Optional[cv2.VideoWriter] = None

    frame_index = 0
    record_start_time = 0.0
    record_timer = None

    trigger_pulse = [0] * 9

    print("")
    print("### RECORDER ### Offline video + trigger recorder")
    print("### RECORDER ### s=start | e=stop | 1-9=trigger | q=quit")
    print("")

    try:
        while True:
            if not frame_manager.is_ready():
                cv2.waitKey(1)
                continue

            frame = cam.get_frame(flip_vertical=FLIP_VERTICAL)

            if frame is None:
                frame_manager.set_frame_time()
                continue

            if is_recording:
                timestamp = time.time() - record_start_time
                signal = trigger_signal(trigger_pulse)

                trigger_writer.write_sv(
                    [frame_index, f"{timestamp:.6f}", signal, 0] + trigger_pulse
                )

                video_writer.write(frame)

                frame_index += 1
                trigger_pulse = [0] * 9

                if record_timer is not None and record_timer.is_finished():
                    stop_session(session_dir, trigger_writer, video_writer)

                    is_recording = False
                    session_dir = None
                    trigger_writer = None
                    video_writer = None
                    record_timer = None

            preview = frame.copy()
            draw_hud(preview, is_recording)
            cam.show(preview, name="MEYELens Offline Recorder")

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s") and not is_recording:
                session_dir, trigger_writer, video_writer = start_session(
                    frame_shape=frame.shape,
                    output_dir=OUTPUT_DIR,
                )

                record_start_time = time.time()
                frame_index = 0
                trigger_pulse = [0] * 9
                is_recording = True

                if MAX_DURATION_SEC is not None:
                    record_timer = CountdownTimer(MAX_DURATION_SEC)
                    record_timer.start()

            elif key == ord("e") and is_recording:
                stop_session(session_dir, trigger_writer, video_writer)

                is_recording = False
                session_dir = None
                trigger_writer = None
                video_writer = None
                record_timer = None

            elif is_recording and ord("1") <= key <= ord("9"):
                trigger_index = key - ord("1")
                trigger_pulse[trigger_index] = 1

            frame_manager.set_frame_time()

    finally:
        if is_recording and session_dir and trigger_writer and video_writer:
            stop_session(session_dir, trigger_writer, video_writer)

        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()