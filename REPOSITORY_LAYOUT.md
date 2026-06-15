# Repository Layout For 2.0

## Preserved

- `docs/` top-level HTML pages and website assets, maintained separately.
- `3d_print_files/` printable hardware files.
- `assets/` repository images.
- `c++_scripts/dual_cam_record.cpp` camera recording source.
- `LICENSE-SOFTWARE` and `LICENSE-HARDWARE`.
- `.gitattributes`.

## Updated For 2.0

- `meyelens/`: PyTorch inference, camera, recording, gaze, analysis, file I/O,
  examples, GUI, and bundled dual-output model.
- `tests/`: public API, model, CSV, and GUI helper tests.
- `pyproject.toml`: PyPI metadata, dependencies, model package data, and the
  `meyelens-gui` entry point.
- `requirements.txt`: supported runtime dependency ranges.
- `environment.yml`: exact direct dependency versions tested for release.
- `README.md`, `MIGRATION.md`, `CHANGELOG.md`, and `RELEASING.md`.
- `docs/docs/`: generated Sphinx API reference only.

## Removed

The following remain available through Git history and tag `v1.1.0`:

- TensorFlow/Keras runtime modules and `.h5` model;
- the old wxPython GUI;
- the separate headless distribution builder;
- obsolete examples and `README_old.md`;
- compiled platform-specific camera binaries;
- GitHub Actions publishing and documentation-push workflows.

## Software Tree

```text
meyelens/
    __init__.py
    __main__.py
    meye.py
    analysis.py
    fileio.py
    gui.py
    examples/
    models/
        __init__.py
        meye_default.pt
tests/
```

Camera, inference, recording, result features, and gaze calibration remain in
`meye.py`; signal analysis is in `analysis.py`; asynchronous text output is in
`fileio.py`; and offline batch analysis is in `gui.py`.

PyTorch is present in `environment.yml` for the tested development environment
but is intentionally absent from `pyproject.toml` and `requirements.txt`.
End users install the hardware-appropriate PyTorch build before MEYElens.
The optional `pt` extra provides a convenience installation of the standard
PyPI PyTorch build.

The legacy `meyelens-headless` distribution is not rebuilt for 2.0 because it
used the same `meyelens` import namespace as the normal distribution. Keeping
one canonical package avoids conflicting ownership of installed files.
