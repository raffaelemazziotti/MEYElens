# Release Checklist

All releases are built and uploaded manually. This repository intentionally
contains no GitHub Actions publishing workflow.

## Preserved 1.x Release

The published TensorFlow implementation is already preserved by the annotated
tag `v1.1.0`. Keep that tag permanently. After the 2.0 branch is pushed, also
create a maintenance branch from the tag:

```bash
git branch legacy/1.x v1.1.0
git push origin legacy/1.x
```

Do not move or recreate `v1.1.0`.

## Prepare 2.0

Create the release environment and install the project:

```bash
conda env create -f environment.yml
conda activate meyelens
python -m pip install -e ".[dev,docs]"
```

Run the tests:

```bash
pytest
```

Build fresh distributions:

```bash
rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
```

Inspect the archive contents before uploading:

```bash
python -m zipfile -l dist/meyelens-2.0.0-py3-none-any.whl
tar -tf dist/meyelens-2.0.0.tar.gz
```

Verify the wheel in a separate environment:

```bash
conda create -n meyelens-wheel-test python=3.11 -y
conda activate meyelens-wheel-test
python -m pip install torch
python -m pip install dist/meyelens-2.0.0-py3-none-any.whl
python -c "import meyelens; print(meyelens.__version__)"
python -c "from meyelens import Meye; print(Meye(gpu_device='cpu', verbose=False).device)"
meyelens-gui
```

## Merge And Tag

Commit the reviewed migration branch and push it:

```bash
git add -A
git commit -m "Release PyTorch-based MEYElens 2.0"
git push -u origin pytorch-2.0
```

Open a pull request from `pytorch-2.0` to `main`. The top-level website files
under `docs/` must not be changed by this release; only `docs/docs/` contains
the regenerated Python API reference.

After review and merge:

```bash
git switch main
git pull --ff-only
git tag -a v2.0.0 -m "MEYElens 2.0.0"
git push origin v2.0.0
```

Create a GitHub release from `v2.0.0` and attach the wheel and source archive
from `dist/`.

## Publish To PyPI

Upload manually with a PyPI API token:

```bash
python -m twine upload dist/*
```

For a dry run, upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Verify the public package in a clean environment:

```bash
conda create -n meyelens-pypi-test python=3.11 -y
conda activate meyelens-pypi-test
python -m pip install torch
python -m pip install "meyelens==2.0.0"
python -c "import meyelens; print(meyelens.__version__)"
meyelens-gui
```
