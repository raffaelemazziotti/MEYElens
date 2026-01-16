#!/usr/bin/env python3
"""
Record online pupil predictions to CSV with keyboard control.

Keys:
  s - start recording (new timestamped file)
  e - stop recording
  1-9 - send trigger pulse to that channel (recorded on the next frame)
  q - quit
"""

import time
from pathlib import Path

import cv2

from meyelens.online import MeyeAsyncRecorder
from meyelens.utils import CountdownTimer

OUTPUT_DIR = Path("recordings")
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


def _stop_and_rename(recorder):
    csv_path = recorder.writer.path
    recorder.stop()
    recorder.writer = None
    recorder.time_start = None

    csv_target = csv_path.parent / "expinfo.csv"
    if csv_path.exists():
        if csv_target.exists():
            csv_target.unlink()
        csv_path.rename(csv_target)


def _start_session(recorder, output_dir: Path) -> None:
    session_stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / session_stamp
    recorder.path_to_file = session_dir
    recorder.filename = "expinfo"


def _wrap_cam_flip(cam, flip_vertical: bool) -> None:
    original_get_frame = cam.get_frame

    def get_frame_wrapped(*args, **kwargs):
        kwargs["flip_vertical"] = flip_vertical
        return original_get_frame(*args, **kwargs)

    cam.get_frame = get_frame_wrapped


def main():
    recorder = MeyeAsyncRecorder(cam_ind=0, path_to_file=OUTPUT_DIR)
    _wrap_cam_flip(recorder.cam, FLIP_VERTICAL)
    is_recording = False
    trigger_pulse = [0] * 9
    record_timer = None

    print("### ONLINE ### Press 's' to start, 'e' to stop, 'q' to quit.")

    while True:
        if is_recording:
            recorder.save_frame(*trigger_pulse)
            frame = recorder.frame
            mask = recorder.predicted
            trigger_pulse = [0] * 9

            if frame is None:
                continue
            overlay = recorder.meye.overlay_roi(mask, frame)

            if record_timer is not None and record_timer.is_finished():
                _stop_and_rename(recorder)
                is_recording = False
                record_timer = None
        else:
            frame = recorder.cam.get_frame()
            if frame is None:
                continue
            mask, _ = recorder.meye.predict(frame)
            overlay = recorder.meye.overlay_roi(mask, frame)

        _draw_hud(overlay, is_recording)
        recorder.cam.show(overlay, name="Online Prediction")
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s") and not is_recording:
            _start_session(recorder, OUTPUT_DIR)
            recorder.start()
            is_recording = True
            if MAX_DURATION_SEC:
                record_timer = CountdownTimer(MAX_DURATION_SEC)
                record_timer.start()
        elif key == ord("e") and is_recording:
            _stop_and_rename(recorder)
            is_recording = False
            record_timer = None
        elif is_recording and ord("1") <= key <= ord("9"):
            trigger_pulse[key - ord("1")] = 1

    if is_recording:
        _stop_and_rename(recorder)

    recorder.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
