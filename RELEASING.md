# Release Checklist

Releases are published to PyPI by `.github/workflows/publish.yml` when a
GitHub release is published. PyPI Trusted Publishing is configured for that
workflow, so no API token is required.

## Preserved 1.x Release

The published TensorFlow implementation is already preserved by the annotated
tag `v1.1.0`. Keep that tag permanently. After the 2.0 branch is pushed, also
create a maintenance branch from the tag:

```bash
git branch legacy/1.x v1.1.0
git push origin legacy/1.x
```

Do not move or recreate `v1.1.0`.

## Prepare A Release

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
VERSION=2.0.1
python -m zipfile -l "dist/meyelens-${VERSION}-py3-none-any.whl"
tar -tf "dist/meyelens-${VERSION}.tar.gz"
```

Verify the wheel in a separate environment:

```bash
conda create -n meyelens-wheel-test python=3.11 -y
conda activate meyelens-wheel-test
VERSION=2.0.1
python -m pip install torch
python -m pip install "dist/meyelens-${VERSION}-py3-none-any.whl"
python -c "import meyelens; print(meyelens.__version__)"
python -c "from meyelens import Meye; print(Meye(gpu_device='cpu', verbose=False).device)"
meyelens-gui
```

## Merge And Tag

Commit and push the reviewed release changes:

```bash
git add -A
git commit -m "Prepare MEYElens 2.0.1"
git push origin main
```

After the commit is on `main`:

```bash
git tag -a v2.0.1 -m "MEYElens 2.0.1"
git push origin v2.0.1
```

Do not modify the top-level website files under `docs/` as part of a package
release.

## Publish To PyPI

Create and publish a GitHub release from `v2.0.1`. The release event runs the
Trusted Publishing workflow, which builds fresh distributions and uploads them
to PyPI. Check the result under GitHub Actions.

Verify the public package in a clean environment:

```bash
conda create -n meyelens-pypi-test python=3.11 -y
conda activate meyelens-pypi-test
python -m pip install "meyelens[pt]==2.0.1"
python -c "import meyelens; print(meyelens.__version__)"
meyelens-gui
```

Also verify that plain installation leaves PyTorch user-managed:

```bash
python -m pip install "meyelens==2.0.1"
```

Do not publish `meyelens-pt`, `meyelens-headless-pt`, or a 2.0 update of
`meyelens-headless`. Version 2.0 uses one canonical PyPI project and provides
PyTorch through the `pt` extra.
