#!/usr/bin/env python3
"""
Record a camera stream to video + trigger CSV with keyboard control.

Keys:
  s - start recording (new timestamped files)
  e - stop recording
  1-9 - send trigger pulse to that channel (recorded on the next frame)
  q - quit
"""

import time
from pathlib import Path

import cv2

from meyelens.camera import Camera
from meyelens.fileio import FileWriter
from meyelens.offline import FrameRateManager
from meyelens.utils import CountdownTimer

OUTPUT_DIR = Path("recordings")
TARGET_FPS = 30.0
MAX_DURATION_SEC = None  # set to a number to auto-stop after N seconds
FLIP_VERTICAL = False  # set True to invert the camera upside down


def _draw_hud(frame, is_recording: bool) -> None:
    lines = [
        f"Recording: {'ON' if is_recording else 'OFF'}",
        "s: start  e: stop  q: quit",
        "1-9: trigger pulse",
    ]

    y = 25
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 22


def _start_session(frame_shape, output_dir: Path):
    session_stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / session_stamp
    session_dir.mkdir(parents=True, exist_ok=True)

    trigger_writer = FileWriter(session_dir, filename="expinfo", sep=",")
    trigger_writer.write(f"# fps: {TARGET_FPS}")
    trigger_writer.write_sv(
        [
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
    )

    video_path = session_dir / "pupillometry.avi"

    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, TARGET_FPS, (width, height), isColor=True)

    return session_dir, trigger_writer, video_writer, video_path


def _stop_session(session_dir: Path, trigger_writer, video_writer):
    video_writer.release()
    trigger_writer.close()

    csv_path = session_dir / "expinfo.csv"
    if trigger_writer.path.exists():
        if csv_path.exists():
            csv_path.unlink()
        trigger_writer.path.rename(csv_path)


def main():
    cam = Camera(camera_index=0)
    frame_manager = FrameRateManager(fps=TARGET_FPS, duration=1e9)
    frame_manager.start()

    is_recording = False
    frame_index = 0
    trigger_pulse = [0] * 9
    session_dir = None
    trigger_writer = None
    video_writer = None
    record_start = 0.0
    record_timer = None

    print("### RECORDER ### Press 's' to start, 'e' to stop, 'q' to quit.")

    while True:
        if not frame_manager.is_ready():
            cv2.waitKey(1)
            continue

        frame = cam.get_frame(flip_vertical=FLIP_VERTICAL)
        if frame is None:
            frame_manager.set_frame_time()
            continue

        if is_recording:
            timestamp = time.time() - record_start
            signal = 0
            for idx, val in enumerate(trigger_pulse):
                if val:
                    signal = idx + 1
                    break
            trigger_writer.write_sv([frame_index, f"{timestamp:.6f}", signal, 0] + trigger_pulse)
            video_writer.write(frame)
            frame_index += 1
            trigger_pulse = [0] * 9

            if record_timer is not None and record_timer.is_finished():
                _stop_session(session_dir, trigger_writer, video_writer)
                is_recording = False
                record_timer = None

        _draw_hud(frame, is_recording)
        cam.show(frame, name="Recorder")
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s") and not is_recording:
            session_dir, trigger_writer, video_writer, _ = _start_session(frame.shape, OUTPUT_DIR)
            record_start = time.time()
            frame_index = 0
            is_recording = True
            if MAX_DURATION_SEC:
                record_timer = CountdownTimer(MAX_DURATION_SEC)
                record_timer.start()
        elif key == ord("e") and is_recording:
            _stop_session(session_dir, trigger_writer, video_writer)
            is_recording = False
            record_timer = None
        elif is_recording and ord("1") <= key <= ord("9"):
            trigger_pulse[key - ord("1")] = 1

        frame_manager.set_frame_time()

    if is_recording:
        _stop_session(session_dir, trigger_writer, video_writer)

    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
