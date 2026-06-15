# Migrating From 1.x to 2.0

MEYElens 2.0 replaces the TensorFlow/Keras inference implementation with
PyTorch and changes the model output from the legacy pupil/blink interface to
dual pupil and eye segmentation.

## Installation

Remove the old installation before installing 2.0:

```bash
python -m pip uninstall -y meyelens meyelens-headless
# Install a PyTorch 2.2+ build appropriate for CPU, CUDA, or Apple MPS first.
python -m pip install "meyelens>=2,<3"
```

Version 2.0 intentionally does not install PyTorch as a package dependency.
This preserves the PyTorch build selected for the target hardware.

For users who want pip to install the standard PyPI PyTorch build:

```bash
python -m pip install "meyelens[pt]>=2,<3"
```

For a specific CUDA build, install PyTorch from the official PyTorch package
index first, then install plain `meyelens`.

The legacy `meyelens-headless` distribution is not continued in 2.0. The
separate distribution installed the same `meyelens` import package and could
not safely coexist with the normal distribution.

## Imports

Use public package imports:

```python
from meyelens import Camera, Meye, MeyeRecorder
```

## Model Construction

Old code may have passed a Keras model with `Meye(model=...)`. Version 2.0
accepts a PyTorch checkpoint path:

```python
meye = Meye(model_path="model.pt", gpu_device="auto")
```

With no path, the bundled `meye_default.pt` checkpoint is used.

## Prediction Results

Version 2.0 returns a `MeyeResult`:

```python
result = meye.predict(frame)
```

The main fields are:

```python
result.masks["pupil"]
result.masks["eye"]
result.features["pupil"]
result.features["eye"]
result.inference_time_ms
result.inference_fps
```

The legacy `mask, info = model(...)` TensorFlow pattern is no longer
supported.

## GPU Selection

Automatic selection uses CUDA, then Apple MPS, then CPU:

```python
meye = Meye(gpu_device="auto")
```

Explicit selections include `"cpu"`, `"mps"`, `"cuda"`, and `"cuda:0"`.

## GUI Outputs

The 2.0 GUI saves complete pupil and eye geometry. CSV column names therefore
differ from the legacy pupil-only GUI output. The GUI also supports video
batches, start/end frame ranges, custom output names, optional overlay videos,
mask visibility and opacity controls, and TOML settings files.

## Reproducibility

Use the final 1.x release or the `legacy/1.x` branch when reproducing analyses
that require the published TensorFlow implementation.
