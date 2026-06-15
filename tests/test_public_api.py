from pathlib import Path

import numpy as np

import meyelens
from meyelens import Meye


def test_public_api_exports():
    assert meyelens.__version__ == "2.0.1"
    assert meyelens.Camera is not None
    assert meyelens.MeyeRecorder is not None
    assert meyelens.MeyeGazeCalibrator is not None


def test_bundled_model_exists():
    assert Path(Meye.default_model_path()).is_file()


def test_default_model_cpu_prediction():
    model = Meye(gpu_device="cpu", verbose=False)
    result = model.predict(np.zeros((64, 64, 3), dtype=np.uint8))

    assert set(result.masks) == {"pupil", "eye"}
    assert set(result.features) == {"pupil", "eye"}
    assert result.masks["pupil"].shape == (64, 64)
    assert result.masks["eye"].shape == (64, 64)
