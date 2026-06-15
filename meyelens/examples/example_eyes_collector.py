from pathlib import Path
from datetime import datetime
import csv
import time

import cv2
import numpy as np

from meyelens import Camera, Meye

"""
Collect eye-image datasets with MEYELens.

This script opens a camera stream, runs MEYELens prediction on each frame,
and displays a live preview with pupil and eye masks overlaid. It allows the
user to save raw camera snapshots, prediction-overlay snapshots, raw movies,
and prediction-overlay movies using keyboard commands.

All saved files are organized into a timestamped session folder, together with
a metadata CSV file containing acquisition settings, thresholds, mask options,
and recording duration when applicable.
"""

# ============================================================
# Dataset collector settings
# ============================================================

CAMERA_INDEX = 0
TARGET_FPS = 30
RESOLUTION = (640, 480)
USE_ROI = False

DATASET_ROOT = Path.home() / "Documents" / "meyeDATASET"

WINDOW_NAME = "MEYELens dataset collector"

MOVIE_CODEC = "mp4v"
MOVIE_EXTENSION = ".mp4"

MASK_ALPHA = 0.45


# ============================================================
# Folders and metadata
# ============================================================

def make_session_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = DATASET_ROOT / f"session_{timestamp}"

    folders = {
        "raw_snapshots": session_dir / "raw_snapshots",
        "overlay_snapshots": session_dir / "overlay_snapshots",
        "raw_movies": session_dir / "raw_movies",
        "overlay_movies": session_dir / "overlay_movies",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return session_dir, folders


def write_metadata_header(metadata_path):
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "type",
            "filename",
            "width",
            "height",
            "camera_index",
            "fps",
            "crop",
            "pupil_threshold",
            "eye_threshold",
            "show_pupil_mask",
            "show_eye_mask",
            "recording_duration_sec",
        ])


def append_metadata(
    metadata_path,
    file_type,
    filename,
    frame,
    camera_index,
    fps,
    crop,
    meye,
    recording_duration_sec="",
):
    h, w = frame.shape[:2]

    with metadata_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            file_type,
            str(filename),
            w,
            h,
            camera_index,
            fps,
            crop,
            meye.get_threshold("pupil") if hasattr(meye, "get_threshold") else "",
            meye.get_threshold("eye") if hasattr(meye, "get_threshold") else "",
            getattr(meye, "show_pupil_mask", ""),
            getattr(meye, "show_eye_mask", ""),
            recording_duration_sec,
        ])


# ============================================================
# Mask-only prediction overlay
# ============================================================

def apply_mask_overlay(frame, result, show_pupil=True, show_eye=True, alpha=0.45):
    """
    Create a prediction overlay with masks only.

    No ellipses.
    No centroids.
    No feature points.
    """
    out = frame.copy()

    if result is None or not hasattr(result, "masks") or result.masks is None:
        return out

    masks_to_draw = []

    if show_pupil and "pupil" in result.masks:
        masks_to_draw.append(("pupil", result.masks["pupil"], (0, 0, 255)))

    if show_eye and "eye" in result.masks:
        masks_to_draw.append(("eye", result.masks["eye"], (0, 255, 0)))

    for _, mask, color in masks_to_draw:
        if mask is None:
            continue

        mask_bool = mask > 0

        if mask_bool.shape[:2] != out.shape[:2]:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (out.shape[1], out.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

        color_array = np.array(color, dtype=np.float32)

        out[mask_bool] = (
            (1.0 - alpha) * out[mask_bool].astype(np.float32)
            + alpha * color_array
        ).astype(np.uint8)

    return out


def draw_meye_status(frame, meye, result):
    """
    Draw only useful prediction status.
    """
    out = frame.copy()

    fps = getattr(result, "inference_fps", np.nan)
    ms = getattr(result, "inference_time_ms", np.nan)

    lines = [
        f"Meye: {fps:.1f} FPS / {ms:.1f} ms",
        (
            f"pupil_thr={meye.get_threshold('pupil'):.2f} "
            f"eye_thr={meye.get_threshold('eye'):.2f}"
        ),
        (
            f"pupil_mask={int(getattr(meye, 'show_pupil_mask', True))} "
            f"eye_mask={int(getattr(meye, 'show_eye_mask', True))}"
        ),
    ]

    y = 25
    for line in lines:
        cv2.putText(
            out,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 25

    return out


def make_prediction_view(frame, result, meye, draw_status=True):
    """
    Image with only prediction masks, according to the current toggles.

    If only pupil is enabled, only pupil mask is drawn.
    If only eye is enabled, only eye mask is drawn.
    If both are enabled, both masks are drawn.
    """
    out = apply_mask_overlay(
        frame,
        result,
        show_pupil=getattr(meye, "show_pupil_mask", True),
        show_eye=getattr(meye, "show_eye_mask", True),
        alpha=MASK_ALPHA,
    )

    if draw_status:
        out = draw_meye_status(out, meye, result)

    return out


# ============================================================
# Collector overlay and controls
# ============================================================

def draw_collector_controls(frame, recording_raw=False, recording_overlay=False):
    out = frame.copy()

    lines = [
        "s: save RAW snapshot",
        "S: save PREDICTION snapshot",
        "r: start/stop RAW movie",
        "R: start/stop PREDICTION movie",
        "+/-: pupil threshold",
        "[/]: eye threshold",
        "p: toggle pupil mask",
        "o: toggle eye mask",
        "h: print commands",
        "q or ESC: quit",
    ]

    if recording_raw:
        lines.append("RECORDING RAW MOVIE")

    if recording_overlay:
        lines.append("RECORDING PREDICTION MOVIE")

    y = 105
    for line in lines:
        cv2.putText(
            out,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 22

    return out


def print_commands():
    print("")
    print("Controls:")
    print("  s       save raw PNG snapshot")
    print("  S       save prediction-overlay PNG snapshot")
    print("  r       start/stop raw movie")
    print("  R       start/stop prediction-overlay movie")
    print("  + / =   increase pupil threshold")
    print("  - / _   decrease pupil threshold")
    print("  ]       increase eye threshold")
    print("  [       decrease eye threshold")
    print("  p       toggle pupil mask")
    print("  o       toggle eye mask")
    print("  h       print commands")
    print("  q/ESC   quit")
    print("")


def handle_key(key, meye):
    if key in (ord("q"), 27):
        return False

    if key in (ord("+"), ord("=")):
        meye.set_threshold("pupil", meye.get_threshold("pupil") + meye.threshold_step)
        print(f"pupil threshold = {meye.get_threshold('pupil'):.2f}")

    elif key in (ord("-"), ord("_")):
        meye.set_threshold("pupil", meye.get_threshold("pupil") - meye.threshold_step)
        print(f"pupil threshold = {meye.get_threshold('pupil'):.2f}")

    elif key == ord("]"):
        meye.set_threshold("eye", meye.get_threshold("eye") + meye.threshold_step)
        print(f"eye threshold = {meye.get_threshold('eye'):.2f}")

    elif key == ord("["):
        meye.set_threshold("eye", meye.get_threshold("eye") - meye.threshold_step)
        print(f"eye threshold = {meye.get_threshold('eye'):.2f}")

    elif key == ord("p"):
        meye.show_pupil_mask = not meye.show_pupil_mask
        print(f"show_pupil_mask = {meye.show_pupil_mask}")

    elif key == ord("o"):
        meye.show_eye_mask = not meye.show_eye_mask
        print(f"show_eye_mask = {meye.show_eye_mask}")

    elif key == ord("h"):
        print_commands()

    return True


# ============================================================
# Movie helpers
# ============================================================

def start_movie(path, frame, fps):
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*MOVIE_CODEC)

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        (w, h),
    )

    if not writer.isOpened():
        writer.release()
        return None

    return writer


# ============================================================
# Main
# ============================================================

session_dir, folders = make_session_folder()
metadata_path = session_dir / "metadata.csv"
write_metadata_header(metadata_path)

print(f"Dataset folder: {session_dir}")
print_commands()

meye = Meye()

# Mask overlay toggles.
meye.show_pupil_mask = True
meye.show_eye_mask = True

with Camera(
    camera_index=CAMERA_INDEX,
    framerate=TARGET_FPS,
    resolution=RESOLUTION,
) as cam:

    if USE_ROI:
        cam.select_roi(window_name="Select eye ROI")

    raw_writer = None
    overlay_writer = None

    raw_movie_path = None
    overlay_movie_path = None

    raw_movie_start = None
    overlay_movie_start = None

    raw_movie_count = 0
    overlay_movie_count = 0
    raw_snapshot_count = 0
    overlay_snapshot_count = 0

    try:
        while True:
            frame = cam.get_frame()

            if frame is None:
                print("Could not read frame.")
                continue

            # Prediction is used only for live preview and optional overlay saves.
            result = meye.predict(frame)

            prediction_view = make_prediction_view(
                frame,
                result,
                meye,
                draw_status=True,
            )

            preview = draw_collector_controls(
                prediction_view,
                recording_raw=raw_writer is not None,
                recording_overlay=overlay_writer is not None,
            )

            cv2.imshow(WINDOW_NAME, preview)

            # Raw movie saves clean camera frames.
            if raw_writer is not None:
                raw_writer.write(frame)

            # Overlay movie saves masks only, without collector command text.
            if overlay_writer is not None:
                overlay_writer.write(prediction_view)

            key = cv2.waitKey(1) & 0xFF

            if key == 255:
                continue

            if not handle_key(key, meye):
                break

            # ----------------------------------------------------
            # Save raw snapshot
            # ----------------------------------------------------
            if key == ord("s"):
                raw_snapshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                snapshot_path = (
                    folders["raw_snapshots"]
                    / f"raw_eye_{timestamp}_{raw_snapshot_count:04d}.png"
                )

                cv2.imwrite(str(snapshot_path), frame)

                append_metadata(
                    metadata_path=metadata_path,
                    file_type="raw_snapshot",
                    filename=snapshot_path,
                    frame=frame,
                    camera_index=CAMERA_INDEX,
                    fps=TARGET_FPS,
                    crop=cam.crop,
                    meye=meye,
                )

                print(f"Saved raw snapshot: {snapshot_path}")

            # ----------------------------------------------------
            # Save prediction-overlay snapshot
            # ----------------------------------------------------
            elif key == ord("S"):
                overlay_snapshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                snapshot_path = (
                    folders["overlay_snapshots"]
                    / f"prediction_eye_{timestamp}_{overlay_snapshot_count:04d}.png"
                )

                cv2.imwrite(str(snapshot_path), prediction_view)

                append_metadata(
                    metadata_path=metadata_path,
                    file_type="prediction_snapshot",
                    filename=snapshot_path,
                    frame=prediction_view,
                    camera_index=CAMERA_INDEX,
                    fps=TARGET_FPS,
                    crop=cam.crop,
                    meye=meye,
                )

                print(f"Saved prediction snapshot: {snapshot_path}")

            # ----------------------------------------------------
            # Start / stop raw movie
            # ----------------------------------------------------
            elif key == ord("r"):
                if raw_writer is None:
                    raw_movie_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    raw_movie_path = (
                        folders["raw_movies"]
                        / f"raw_eye_movie_{timestamp}_{raw_movie_count:04d}{MOVIE_EXTENSION}"
                    )

                    raw_writer = start_movie(raw_movie_path, frame, TARGET_FPS)

                    if raw_writer is None:
                        print("Could not start raw movie.")
                        raw_movie_path = None
                        continue

                    raw_movie_start = time.time()
                    print(f"Started raw movie: {raw_movie_path}")

                else:
                    raw_writer.release()
                    raw_writer = None

                    duration = time.time() - raw_movie_start

                    append_metadata(
                        metadata_path=metadata_path,
                        file_type="raw_movie",
                        filename=raw_movie_path,
                        frame=frame,
                        camera_index=CAMERA_INDEX,
                        fps=TARGET_FPS,
                        crop=cam.crop,
                        meye=meye,
                        recording_duration_sec=f"{duration:.3f}",
                    )

                    print(f"Stopped raw movie: {raw_movie_path} ({duration:.2f} s)")

                    raw_movie_path = None
                    raw_movie_start = None

            # ----------------------------------------------------
            # Start / stop prediction-overlay movie
            # ----------------------------------------------------
            elif key == ord("R"):
                if overlay_writer is None:
                    overlay_movie_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    overlay_movie_path = (
                        folders["overlay_movies"]
                        / f"prediction_eye_movie_{timestamp}_{overlay_movie_count:04d}{MOVIE_EXTENSION}"
                    )

                    overlay_writer = start_movie(
                        overlay_movie_path,
                        prediction_view,
                        TARGET_FPS,
                    )

                    if overlay_writer is None:
                        print("Could not start prediction movie.")
                        overlay_movie_path = None
                        continue

                    overlay_movie_start = time.time()
                    print(f"Started prediction movie: {overlay_movie_path}")

                else:
                    overlay_writer.release()
                    overlay_writer = None

                    duration = time.time() - overlay_movie_start

                    append_metadata(
                        metadata_path=metadata_path,
                        file_type="prediction_movie",
                        filename=overlay_movie_path,
                        frame=prediction_view,
                        camera_index=CAMERA_INDEX,
                        fps=TARGET_FPS,
                        crop=cam.crop,
                        meye=meye,
                        recording_duration_sec=f"{duration:.3f}",
                    )

                    print(
                        f"Stopped prediction movie: "
                        f"{overlay_movie_path} ({duration:.2f} s)"
                    )

                    overlay_movie_path = None
                    overlay_movie_start = None

    finally:
        if raw_writer is not None:
            raw_writer.release()

        if overlay_writer is not None:
            overlay_writer.release()

        cv2.destroyAllWindows()

print(f"Done. Files saved in: {session_dir}")
