# Changelog

## 2.0.1

- Require Python 3.11+ and SciPy 1.16+ so pip selects current scientific
  Python wheels on modern macOS instead of the incompatible SciPy 1.15
  wheel limited to Python 3.10.

## 2.0.0

- Replaced TensorFlow/Keras inference with PyTorch.
- Added dual pupil and eye segmentation.
- Added CUDA and Apple MPS automatic device selection.
- Added complete pupil and eye geometric measurements.
- Added buffered recording and gaze-calibration APIs.
- Rebuilt the offline PyQt6 GUI around `Meye.predict()` and `MeyeResult`.
- Added video timeline navigation, play/pause preview, and frame-range analysis.
- Added batch processing, selectable output paths, TOML GUI settings, speed,
  and estimated-time reporting.
- Added independent pupil/eye mask visibility and overlay opacity controls.
- Added complete dual-output CSV export.
- Bundled the default PyTorch checkpoint.
- Made PyTorch a separately installed prerequisite so users can select their
  CPU, CUDA, or Apple MPS build without pip replacing it.
- Added the `pt` extra for users who prefer `pip install "meyelens[pt]"`.
- Retired the separate `meyelens-headless` distribution for 2.0.
- Rebuilt the Sphinx API reference for the 2.0 public modules.
- Removed automated GitHub publishing workflows in favor of manual releases.

## 1.x

The final 1.x release preserves the implementation associated with the
published workflow. See the corresponding Git tag and `legacy/1.x` branch.
