import math
from pathlib import Path

import cv2
import numpy as np

from meyelens.gui import (
    BatchVideoWorker,
    feature_to_row,
    prepare_frame,
    result_to_row,
)
from meyelens.meye import MeyeResult


def test_dual_output_csv_schema():
    feature = {
        "valid": True,
        "area": 10,
        "centroid": (4.0, 5.0),
        "ellipse": {
            "valid": True,
            "major_diameter": 8.0,
            "minor_diameter": 6.0,
            "orientation_deg": 20.0,
            "ovality": 0.25,
            "eccentricity": 0.66,
            "major_axis": {"p1": (1.0, 2.0), "p2": (3.0, 4.0)},
            "minor_axis": {"p1": (5.0, 6.0), "p2": (7.0, 8.0)},
        },
    }
    result = MeyeResult(
        probabilities={},
        masks={},
        features={"pupil": feature, "eye": feature},
        inference_time_ms=4.0,
        inference_fps=250.0,
    )

    row = result_to_row(result, frame_index=3, source_time_ms=100.0)

    assert row["pupil_x"] == 5.0
    assert row["pupil_major_p1_x"] == 2.0
    assert row["eye_minor_p2_y"] == 7.0
    assert row["source_time_ms"] == 100.0


def test_empty_feature_schema_uses_nan():
    row = feature_to_row("pupil", {})

    assert row["pupil_valid"] is False
    assert math.isnan(row["pupil_x"])


class _FakeMeye:
    def predict(self, frame):
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        return MeyeResult(
            probabilities={},
            masks={"pupil": mask, "eye": mask},
            features={"pupil": {}, "eye": {}},
            inference_time_ms=1.0,
            inference_fps=1000.0,
        )

    def overlay(self, frame, result, alpha=0.45, draw_text=False):
        return frame.copy()


def test_batch_worker_respects_frame_range_and_output_name(tmp_path):
    video_path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (16, 16),
    )
    assert writer.isOpened()
    for value in range(4):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()

    settings = {
        "flip_vertical": False,
        "invert": False,
        "crop_enabled": False,
        "crop_x": 0,
        "crop_y": 0,
        "crop_size": 16,
        "save_video": False,
        "mask_opacity": 0.45,
        "start_frame": 2,
        "end_frame": 3,
        "output_dir": str(tmp_path),
        "output_name": "partial",
    }
    worker = BatchVideoWorker(_FakeMeye(), [video_path], settings)
    completed = []
    worker.finished.connect(
        lambda outputs, count, cancelled: completed.append(
            (outputs, count, cancelled)
        )
    )

    worker.run()

    outputs, count, cancelled = completed[0]
    assert count == 2
    assert cancelled is False
    csv_path = Path(outputs[0]["csv"])
    assert csv_path.name == "partial.csv"
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert rows[1].startswith("1,")
    assert rows[2].startswith("2,")
