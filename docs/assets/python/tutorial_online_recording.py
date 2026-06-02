"""
Record online pupil predictions to CSV with keyboard control.

This script opens the camera, runs online pupil prediction, displays a live
preview with the predicted pupil overlay, and writes frame-by-frame results
to a CSV file while recording is active.

Keys:
  s   start recording in a new timestamped folder
  e   stop recording and rename the CSV file to expinfo.csv
  1-9 send a one-frame trigger pulse to the selected trigger channel
  q   quit the script
"""

import time
from pathlib import Path

import cv2

from meyelens.online import MeyeAsyncRecorder
from meyelens.utils import CountdownTimer


# Folder where all recording sessions will be saved.
# Each recording will create a timestamped subfolder inside this directory.
OUTPUT_DIR = Path("recordings")

# Optional maximum recording duration.
# Keep this as None for manual stop with the "e" key.
# Set to a number, for example 60, to stop automatically after 60 seconds.
MAX_DURATION_SEC = None

# Set to True if the camera image appears upside down.
FLIP_VERTICAL = False


def _draw_hud(frame, is_recording: bool) -> None:
    """
    Draw simple on-screen instructions over the live preview.

    Parameters
    ----------
    frame:
        The image shown in the OpenCV preview window.
    is_recording:
        Whether data are currently being saved to CSV.
    """
    lines = [
        f"Recording: {'ON' if is_recording else 'OFF'}",
        "s: start  e: stop  q: quit",
        "1-9: trigger pulse",
    ]

    y = 25
    for line in lines:
        # Draw black thick text first to create contrast.
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )

        # Draw thin white text on top.
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        y += 22


def _stop_and_rename(recorder: MeyeAsyncRecorder) -> None:
    """
    Stop the current recording and rename the output CSV.

    MeyeAsyncRecorder creates the CSV using the recorder filename.
    Here we rename the final file to expinfo.csv so that every session folder
    has a consistent output filename.
    """
    csv_path = recorder.writer.path

    # Stop the recorder and reset its internal writer state.
    recorder.stop()
    recorder.writer = None
    recorder.time_start = None

    csv_target = csv_path.parent / "expinfo.csv"

    # Replace an existing expinfo.csv only if needed.
    if csv_path.exists():
        if csv_target.exists():
            csv_target.unlink()

        csv_path.rename(csv_target)


def _start_session(recorder: MeyeAsyncRecorder, output_dir: Path) -> None:
    """
    Prepare a new timestamped output folder before starting acquisition.

    Example output folder:
      recordings/20260511_153012/
    """
    session_stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / session_stamp

    # Tell the recorder where to save the next CSV file.
    recorder.path_to_file = session_dir

    # Temporary filename used by the recorder before we rename it to expinfo.csv.
    recorder.filename = "expinfo"


def _wrap_cam_flip(cam, flip_vertical: bool) -> None:
    """
    Patch the camera get_frame method to consistently apply vertical flipping.

    This keeps the rest of the acquisition loop unchanged. Every time the
    recorder asks the camera for a frame, the flip_vertical option is passed.
    """
    original_get_frame = cam.get_frame

    def get_frame_wrapped(*args, **kwargs):
        kwargs["flip_vertical"] = flip_vertical
        return original_get_frame(*args, **kwargs)

    cam.get_frame = get_frame_wrapped


def main() -> None:
    """
    Main online acquisition loop.

    The loop has two states:
      1. Preview mode: camera frames are shown but not saved.
      2. Recording mode: predictions and trigger values are saved to CSV.
    """
    # Create the asynchronous online recorder.
    # cam_ind=0 usually selects the first available camera.
    # Change this to 1, 2, ... if the wrong camera opens.
    recorder = MeyeAsyncRecorder(cam_ind=0, path_to_file=OUTPUT_DIR)

    # Apply optional vertical flipping to the camera frames.
    _wrap_cam_flip(recorder.cam, FLIP_VERTICAL)

    # Recording state.
    is_recording = False

    # Trigger channels.
    # This creates 9 trigger columns. Pressing keys 1-9 sets one channel to 1
    # for exactly one saved frame, then the vector is reset to zeros.
    trigger_pulse = [0] * 9

    # Optional timer used only when MAX_DURATION_SEC is not None.
    record_timer = None

    print("### ONLINE ### Press 's' to start, 'e' to stop, 'q' to quit.")

    while True:
        if is_recording:
            # Save the current frame prediction and trigger state to CSV.
            recorder.save_frame(*trigger_pulse)

            # Retrieve the last camera frame and predicted pupil mask.
            frame = recorder.frame
            mask = recorder.predicted

            # Reset triggers immediately after saving.
            # This makes each keypress a one-frame pulse.
            trigger_pulse = [0] * 9

            if frame is None:
                continue

            # Overlay the predicted pupil mask on the eye image.
            overlay = recorder.meye.overlay_roi(mask, frame)

            # Auto-stop when the optional timer reaches the requested duration.
            if record_timer is not None and record_timer.is_finished():
                _stop_and_rename(recorder)
                is_recording = False
                record_timer = None

        else:
            # Preview mode: get a frame and run prediction, but do not save.
            frame = recorder.cam.get_frame()

            if frame is None:
                continue

            mask, _ = recorder.meye.predict(frame)
            overlay = recorder.meye.overlay_roi(mask, frame)

        # Draw text instructions on the preview image.
        _draw_hud(overlay, is_recording)

        # Show the live preview.
        recorder.cam.show(overlay, name="Online Prediction")

        # Read keyboard input from the OpenCV window.
        # waitKey(1) keeps the preview responsive.
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            # Quit the loop.
            break

        if key == ord("s") and not is_recording:
            # Start a new recording session.
            _start_session(recorder, OUTPUT_DIR)
            recorder.start()
            is_recording = True

            # Start the optional auto-stop timer.
            if MAX_DURATION_SEC:
                record_timer = CountdownTimer(MAX_DURATION_SEC)
                record_timer.start()

        elif key == ord("e") and is_recording:
            # Stop recording manually.
            _stop_and_rename(recorder)
            is_recording = False
            record_timer = None

        elif is_recording and ord("1") <= key <= ord("9"):
            # Convert the pressed key into a trigger-channel index.
            # "1" -> index 0, "2" -> index 1, ..., "9" -> index 8.
            trigger_index = key - ord("1")
            trigger_pulse[trigger_index] = 1

    # If the user quits while recording is active, close the file safely.
    if is_recording:
        _stop_and_rename(recorder)

    # Release camera resources and close OpenCV windows.
    recorder.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()