# Configuration file for the Sphinx documentation builder.

import os
import re
import sys
from datetime import datetime
from importlib.util import find_spec

# Make sure Sphinx can import your package from the repo root.
sys.path.insert(0, os.path.abspath("../../.."))

# If you use a src/ layout (MEYElens/src/meyelens/...), use this instead:
# sys.path.insert(0, os.path.abspath("../../src"))

project = "MEYELens"
copyright = f"{datetime.utcnow().year}, Giacomo Vecchieschi, Raffaele Mario Mazziotti"
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
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Docstring style support (Google / NumPy)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Avoid importing heavy dependencies during docs build
autodoc_mock_imports = [
    "cv2",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "wx",
    "wxPython",
    "PyQt6",
    "pyo",
    "sounddevice",
    "soundfile",
    "pygame",
    "pyobjc",
    "skimage",
    "sklearn",
    "tensorflow",
    "toml",
]
