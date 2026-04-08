#!/usr/bin/env python3
"""
Live gaze calibration + prediction demo.

Workflow:
1) Show a sequence of calibration targets.
2) Train a polynomial gaze model from pupil centroids.
3) Show a gray screen with predicted gaze as a red dot.

Keys:
  q - quit
"""

import time
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np

from meyelens.camera import Camera
from meyelens.gaze import GazeModelPoly
from meyelens.meye import Meye

CALIBRATION_WINDOW_NAME = "Gaze Calibration"
LIVE_WINDOW_NAME = "Gaze Prediction"
DEFAULT_SCREEN_WIDTH = 1280
DEFAULT_SCREEN_HEIGHT = 720
BACKGROUND_GRAY = 160
DOT_RADIUS = 10
TARGET_MARGIN_RATIO = 0.05
TARGET_MARGIN_MIN_PX = 30

TARGET_PREVIEW_SEC = 2.0
CALIBRATION_RECORD_SEC = 5.0
RANDOM_POINTS = False
NUM_POINTS = 5
FLIP_VERTICAL = True  # set True to invert the camera upside down

MODEL_OUTPUT = Path("gaze_models/gazemodel_poly.pkl")


def get_screen_size() -> tuple[int, int]:
    try:
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    return DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT


def create_blank_screen(screen_width: int, screen_height: int) -> np.ndarray:
    return np.full((screen_height, screen_width, 3), BACKGROUND_GRAY, dtype=np.uint8)


def screen_to_pixel(pos, screen_width: int, screen_height: int) -> tuple[int, int]:
    x, y = pos
    x_pix = int(screen_width / 2 + x)
    y_pix = int(screen_height / 2 - y)
    x_pix = max(0, min(screen_width - 1, x_pix))
    y_pix = max(0, min(screen_height - 1, y_pix))
    return x_pix, y_pix


def draw_dot(frame: np.ndarray, pos, color, screen_width: int, screen_height: int) -> None:
    x_pix, y_pix = screen_to_pixel(pos, screen_width, screen_height)
    cv2.circle(frame, (x_pix, y_pix), DOT_RADIUS + 4, (0, 0, 0), thickness=-1)
    cv2.circle(frame, (x_pix, y_pix), DOT_RADIUS, color, thickness=-1)


def draw_text(frame: np.ndarray, text: str, y: int) -> None:
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def is_valid_centroid(centroid) -> bool:
    return centroid is not None and not np.isnan(centroid).any()


def predict_pupil(meye: Meye, frame: np.ndarray):
    # `morph=False` avoids OpenCV ellipse fitting on tiny contours; this demo only needs the mask and centroid.
    return meye.predict(frame, morph=False)


def set_fullscreen(window_name: str, width: int, height: int) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def make_calibration_positions(screen_width: int, screen_height: int) -> list[tuple[float, float]]:
    margin_x = max(TARGET_MARGIN_MIN_PX, int(screen_width * TARGET_MARGIN_RATIO))
    margin_y = max(TARGET_MARGIN_MIN_PX, int(screen_height * TARGET_MARGIN_RATIO))
    x_limit = screen_width / 2 - margin_x
    y_limit = screen_height / 2 - margin_y

    fixed_points = [
        (-x_limit, y_limit),
        (x_limit, y_limit),
        (-x_limit, -y_limit),
        (x_limit, -y_limit),
        (0, 0),
    ]

    if not RANDOM_POINTS or NUM_POINTS <= len(fixed_points):
        return fixed_points

    random_points = [
        (
            np.random.uniform(-x_limit, x_limit),
            np.random.uniform(-y_limit, y_limit),
        )
        for _ in range(NUM_POINTS - len(fixed_points))
    ]
    positions = fixed_points + random_points
    np.random.shuffle(positions)
    return positions


def run_calibration(cam: Camera, meye: Meye, screen_width: int, screen_height: int):
    gaze_points = []
    screen_positions = []

    positions = make_calibration_positions(screen_width, screen_height)
    total_points = len(positions)

    set_fullscreen(CALIBRATION_WINDOW_NAME, screen_width, screen_height)

    for idx, target in enumerate(positions):
        preview_start = time.time()
        while time.time() - preview_start < TARGET_PREVIEW_SEC:
            display = create_blank_screen(screen_width, screen_height)
            draw_dot(display, target, (255, 255, 255), screen_width, screen_height)
            draw_text(display, f"Calibration {idx + 1}/{total_points}", 30)
            draw_text(display, "Look at the white dot", 55)
            cv2.imshow(CALIBRATION_WINDOW_NAME, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None, None

        record_start = time.time()
        while time.time() - record_start < CALIBRATION_RECORD_SEC:
            frame = cam.get_frame(flip=FLIP_VERTICAL)
            if frame is not None:
                _, centroid = predict_pupil(meye, frame)
                if is_valid_centroid(centroid):
                    gaze_points.append([centroid[1], centroid[0]])
                    screen_positions.append([target[0], target[1]])

            display = create_blank_screen(screen_width, screen_height)
            draw_dot(display, target, (0, 0, 255), screen_width, screen_height)
            draw_text(display, f"Calibration {idx + 1}/{total_points}", 30)
            draw_text(display, "Keep looking at the red dot", 55)
            cv2.imshow(CALIBRATION_WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None, None

    if not gaze_points:
        print("### GAZE ### No calibration samples collected.")
        return None, None

    return np.array(gaze_points, dtype=float), np.array(screen_positions, dtype=float)


def run_live_prediction(cam: Camera, meye: Meye, model: GazeModelPoly, screen_width: int, screen_height: int) -> None:
    print("### GAZE ### Calibration done. Showing live prediction.")
    cv2.destroyWindow(CALIBRATION_WINDOW_NAME)
    set_fullscreen(LIVE_WINDOW_NAME, screen_width, screen_height)

    while True:
        frame = cam.get_frame(flip=FLIP_VERTICAL)
        if frame is None:
            continue

        _, centroid = predict_pupil(meye, frame)
        display = create_blank_screen(screen_width, screen_height)

        if is_valid_centroid(centroid):
            prediction = model.predict(np.array([[centroid[1], centroid[0]]], dtype=float))[0]
            draw_dot(display, prediction, (0, 0, 255), screen_width, screen_height)

        draw_text(display, "Live prediction (q to quit)", 30)
        cv2.imshow(LIVE_WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    screen_width, screen_height = get_screen_size()
    cam = Camera(camera_ind=0)
    meye = Meye()

    print('### MEYE ### Close the preview window to start calibration.')
    meye.preview(cam)

    gaze_points, screen_positions = run_calibration(cam, meye, screen_width, screen_height)
    if gaze_points is None or screen_positions is None:
        cam.close()
        cv2.destroyAllWindows()
        return

    model = GazeModelPoly()
    model.train(gaze_points, screen_positions, degree=2)
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_OUTPUT))

    run_live_prediction(cam, meye, model, screen_width, screen_height)

    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
