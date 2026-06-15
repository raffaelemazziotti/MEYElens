# Configuration file for the Sphinx documentation builder.

import os
import re
import sys
from datetime import datetime, timezone
from importlib.util import find_spec

# Make sure Sphinx can import your package from the repo root.
sys.path.insert(0, os.path.abspath("../../.."))

# If you use a src/ layout (MEYElens/src/meyelens/...), use this instead:
# sys.path.insert(0, os.path.abspath("../../src"))

project = "MEYELens"
copyright = (
    f"{datetime.now(timezone.utc).year}, "
    "Giacomo Vecchieschi, Raffaele Mario Mazziotti"
)
author = "Giacomo Vecchieschi, Raffaele Mario Mazziotti"


def _read_project_version() -> str:
    pyproject_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "pyproject.toml"))
    try:
        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return "0.0.0"

    match = re.search(r'(?m)^version\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else "0.0.0"


release = _read_project_version()

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

if find_spec("sphinx_copybutton") is not None:
    extensions.append("sphinx_copybutton")

root_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Generate autosummary stubs
autosummary_generate = True

# This is what makes autodoc actually list your functions/classes
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autoclass_content = "init"

# Docstring style support (Google / NumPy)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock optional experiment and hardware dependencies only. Runtime dependencies
# are imported normally so broken public API imports fail the documentation build.
autodoc_mock_imports = [
    "joblib",
    "pyo",
    "psychopy",
    "sounddevice",
    "soundfile",
    "pygame",
    "pyobjc",
    "skimage",
    "timm",
]
