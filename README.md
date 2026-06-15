# MEYElens

**MEYElens: 3D-Printable Wearable System for Pupillometry and Gaze Tracking**

![MEYElens](assets/intro_picture.png)

MEYElens 2.0 is the PyTorch implementation of the MEYElens acquisition and
analysis software. It provides:

- dual-output pupil and eye segmentation;
- CPU, NVIDIA CUDA, and Apple Metal (MPS) inference;
- pupil and eye geometry, including centroids, areas, fitted ellipses, and axes;
- live camera preview and recording;
- gaze calibration;
- offline video analysis through a PyQt6 GUI;
- pupil-signal cleaning, filtering, and trial extraction tools.

The hardware, assembly instructions, model information, and tutorials are
available at [meyelens.com](https://www.meyelens.com).

## Install

Use a dedicated environment:

```bash
conda create -n meyelens python=3.11 -y
conda activate meyelens
```

Install PyTorch first using the command for your operating system and compute
platform from the [official PyTorch selector](https://pytorch.org/get-started/locally/).
MEYElens supports PyTorch 2.2 or newer and is tested with PyTorch 2.12.

Then install MEYElens:

```bash
pip install meyelens
```

`pip install meyelens` installs the MEYElens runtime dependencies but
intentionally does not install PyTorch. This prevents pip from replacing a
user-selected CPU, CUDA, or Apple MPS build.

For development from a cloned repository:

```bash
python -m pip install torch
python -m pip install -e ".[dev]"
```

The tested SegFormer dependency range is:

```text
transformers>=4.57,<=5.8.1
```

The 2.0 loader includes compatibility handling for the checkpoint's original
SegFormer parameter names. Version 5.8.1 is pinned in `environment.yml` for a
reproducible conda installation.

## GPU Support

`Meye(gpu_device="auto")` selects devices in this order:

1. NVIDIA CUDA;
2. Apple MPS;
3. CPU.

On Windows and Linux, select the appropriate NVIDIA CUDA wheel when GPU
inference is required. Apple Silicon uses the MPS backend included in supported
macOS PyTorch builds. CPU-only installations should select the CPU build.

## Quick Start

```python
from meyelens import Camera, Meye

meye = Meye()

with Camera(camera_index=0) as camera:
    camera.select_roi()
    meye.preview(camera)
```

Single-frame prediction:

```python
result = meye.predict(frame)

pupil_mask = result.masks["pupil"]
eye_mask = result.masks["eye"]
pupil_features = result.features["pupil"]
eye_features = result.features["eye"]
```

## Offline GUI

Launch the GUI after installation:

```bash
meyelens-gui
```

Or from a source checkout:

```bash
python -m meyelens
```

The GUI supports draggable ROI selection, frame scrubbing and playback,
independent pupil and eye mask controls, mask opacity, separate thresholds,
morphology controls, MPS/CUDA/CPU selection, start/end frame ranges, custom
output locations, batch processing, TOML settings, progress speed and ETA,
and optional overlay-video output.

The generated CSV contains both pupil and eye outputs:

- mask and ellipse validity;
- centroid coordinates;
- area;
- major and minor ellipse diameters;
- orientation, ovality, and eccentricity;
- major and minor axis endpoints;
- source-video time;
- inference time and FPS.

## Examples

Examples are under [`meyelens/examples`](meyelens/examples):

- `example_simple.py`: camera preview and feature extraction;
- `example_recording.py`: synchronized recording and triggers;
- `example_plr.py`: pupillary light reflex experiment;
- `example_oddball.py`: visual oddball experiment;
- `example_gaze.py`: gaze calibration;
- `example_eyes_collector.py`: dataset acquisition.

PsychoPy examples require:

```bash
pip install "meyelens[experiments]"
```

## Version History

The published TensorFlow implementation remains available through the final
1.x Git tag and `legacy/1.x` branch. Version 2.0 replaces the runtime with the
dual-output PyTorch pipeline. See [MIGRATION.md](MIGRATION.md).

## References

If you use MEYElens, cite:

**MEYElens: 3D-Printable Wearable System for Pupillometry and Gaze Tracking**<br>
G. Vecchieschi, L. Ingenito, A. Benedetto, C. Luciani, F. Carrara, G. Cioni,
A. Guzzetta, T. Pizzorusso, L. Baroncelli, R. M. Mazziotti.

**MEYE: Web App for Translational and Real-Time Pupillometry**<br>
R. Mazziotti et al., eNeuro (2021), 8(5): ENEURO.0122-21.2021.<br>
[https://doi.org/10.1523/ENEURO.0122-21.2021](https://doi.org/10.1523/ENEURO.0122-21.2021)

## License

- Software: GPL-3.0, see [LICENSE-SOFTWARE](LICENSE-SOFTWARE).
- Hardware: CERN-OHL-P-2.0, see [LICENSE-HARDWARE](LICENSE-HARDWARE).
