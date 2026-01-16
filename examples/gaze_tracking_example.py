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

import cv2
import numpy as np

from meyelens.camera import Camera
from meyelens.gaze import GazeModelPoly, ScreenPositions
from meyelens.meye import Meye

WINDOW_NAME = "Gaze Calibration"
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BACKGROUND_GRAY = 160
DOT_RADIUS = 10

CALIBRATION_POINT_SEC = 1.5
RANDOM_POINTS = False
NUM_POINTS = 5
FLIP_VERTICAL = False  # set True to invert the camera upside down

MODEL_OUTPUT = Path("gaze_models/gazemodel_poly.pkl")


def _blank_screen() -> np.ndarray:
    return np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), BACKGROUND_GRAY, dtype=np.uint8)


def _screen_to_pixel(pos) -> tuple[int, int]:
    x, y = pos
    x_pix = int(SCREEN_WIDTH / 2 + x)
    y_pix = int(SCREEN_HEIGHT / 2 - y)
    x_pix = max(0, min(SCREEN_WIDTH - 1, x_pix))
    y_pix = max(0, min(SCREEN_HEIGHT - 1, y_pix))
    return x_pix, y_pix


def _draw_dot(frame: np.ndarray, pos, color) -> None:
    x_pix, y_pix = _screen_to_pixel(pos)
    cv2.circle(frame, (x_pix, y_pix), DOT_RADIUS, color, thickness=-1)


def _draw_text(frame: np.ndarray, text: str, y: int) -> None:
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def _calibrate(cam: Camera, meye: Meye):
    gaze_points = []
    screen_positions = []

    positions = ScreenPositions(
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        random_points=RANDOM_POINTS,
        num_points=NUM_POINTS,
    )

    total_points = len(positions.positions)
    for idx in range(total_points):
        target = positions.next()
        if target is None:
            break

        start = time.time()
        while time.time() - start < CALIBRATION_POINT_SEC:
            frame = cam.get_frame(flip_vertical=FLIP_VERTICAL)
            if frame is not None:
                _, centroid = meye.predict(frame)
                if not np.isnan(centroid[0]):
                    gaze_points.append([centroid[1], centroid[0]])
                    screen_positions.append([target[0], target[1]])

            display = _blank_screen()
            _draw_dot(display, target, (255, 255, 255))
            _draw_text(display, f"Calibration {idx + 1}/{total_points}", 30)
            _draw_text(display, "Look at the dot", 55)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None, None

    if not gaze_points:
        print("### GAZE ### No calibration samples collected.")
        return None, None

    return np.array(gaze_points, dtype=float), np.array(screen_positions, dtype=float)


def main():
    cam = Camera(camera_index=0)
    meye = Meye()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_WIDTH, SCREEN_HEIGHT)

    gaze_points, screen_positions = _calibrate(cam, meye)
    if gaze_points is None or screen_positions is None:
        cam.close()
        cv2.destroyAllWindows()
        return

    model = GazeModelPoly()
    model.train(gaze_points, screen_positions, degree=2)
    model.save(str(MODEL_OUTPUT))

    print("### GAZE ### Calibration done. Showing live prediction.")

    while True:
        frame = cam.get_frame(flip_vertical=FLIP_VERTICAL)
        if frame is None:
            continue

        _, centroid = meye.predict(frame)
        display = _blank_screen()

        if not np.isnan(centroid[0]):
            prediction = model.predict(np.array([[centroid[1], centroid[0]]], dtype=float))[0]
            _draw_dot(display, prediction, (0, 0, 255))

        _draw_text(display, "Live prediction (q to quit)", 30)
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
