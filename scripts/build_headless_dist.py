from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any


HEADLESS_MAIN = """\
def main() -> int:
    print(
        "MEYElens headless is installed. The PyQt GUI is not included in this "
        "distribution. Install the 'meyelens' package if you need the offline GUI."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_array(values: list[str]) -> str:
    if not values:
        return "[]"
    inner = ",\n".join(f"  {quote(value)}" for value in values)
    return f"[\n{inner},\n]"


def render_authors(authors: list[dict[str, str]]) -> str:
    rows = [
        f'  {{ name = {quote(author["name"])}, email = {quote(author["email"])} }}'
        for author in authors
    ]
    return "[\n" + ",\n".join(rows) + "\n]"


def render_optional_dependencies(optional_dependencies: dict[str, list[str]]) -> str:
    lines: list[str] = ["[project.optional-dependencies]"]
    for extra, dependencies in optional_dependencies.items():
        lines.append(f"{extra} = {render_array(dependencies)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_urls(urls: dict[str, str]) -> str:
    lines = ["[project.urls]"]
    for key, value in urls.items():
        lines.append(f"{key} = {quote(value)}")
    return "\n".join(lines) + "\n"


def build_headless_pyproject(project: dict[str, Any], build_system: dict[str, Any], package_data: dict[str, list[str]]) -> str:
    dependencies = [
        dependency
        for dependency in project["dependencies"]
        if not dependency.startswith("PyQt6")
    ]

    parts = [
        "[build-system]",
        f'requires = {render_array(build_system["requires"])}',
        f'build-backend = {quote(build_system["build-backend"])}',
        "",
        "[project]",
        'name = "meyelens-headless"',
        f'version = {quote(project["version"])}',
        'description = "Headless MEYElens package without the PyQt offline GUI"',
        'readme = "README.md"',
        f'requires-python = {quote(project["requires-python"])}',
        f'license = {quote(project["license"])}',
        f'license-files = {render_array(project.get("license-files", []))}',
        f"authors = {render_authors(project['authors'])}",
        f"dependencies = {render_array(dependencies)}",
        "",
        render_urls(project["urls"]).rstrip(),
        "",
        render_optional_dependencies(project.get("optional-dependencies", {})).rstrip(),
        "",
        "[tool.setuptools.packages.find]",
        'where = ["."]',
        'include = ["meyelens*"]',
        "",
        "[tool.setuptools.package-data]",
    ]

    for package_name, patterns in package_data.items():
        parts.append(f"{quote(package_name)} = {render_array(patterns)}")

    return "\n".join(parts) + "\n"


def build_headless_readme(version: str) -> str:
    return textwrap.dedent(
        f"""\
        # MEYElens Headless

        This is the headless PyPI distribution for MEYElens. It installs the
        core `meyelens` Python package without the PyQt offline GUI.

        ## Install

        ```bash
        pip install meyelens-headless
        ```

        With the pinned PyTorch dependency:

        ```bash
        pip install "meyelens-headless[pt]"
        ```

        ## Notes

        - Version: `{version}`
        - Import path remains `meyelens`
        - `PyQt6`, `meyelens.gui`, and `meyelens-gui` are intentionally omitted
        - Do not install `meyelens` and `meyelens-headless` into the same environment
        - Install the `meyelens` package instead if you need the GUI
        """
    )


def copy_package_tree(source_root: Path, staging_root: Path) -> None:
    shutil.copytree(
        source_root / "meyelens",
        staging_root / "meyelens",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "gui.py"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the meyelens-headless distribution.")
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=Path("build/headless-package"),
        help="Temporary staging directory used to assemble the headless package.",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist-headless"),
        help="Output directory for built distributions.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Prepare the staging directory without invoking python -m build.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    stage_dir = (repo_root / args.stage_dir).resolve()
    dist_dir = (repo_root / args.dist_dir).resolve()

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    copy_package_tree(repo_root, stage_dir)
    (stage_dir / "meyelens" / "__main__.py").write_text(HEADLESS_MAIN, encoding="utf-8")
    (stage_dir / "README.md").write_text(
        build_headless_readme(pyproject["project"]["version"]),
        encoding="utf-8",
    )
    (stage_dir / "pyproject.toml").write_text(
        build_headless_pyproject(
            pyproject["project"],
            pyproject["build-system"],
            pyproject["tool"]["setuptools"]["package-data"],
        ),
        encoding="utf-8",
    )
    shutil.copy2(repo_root / "LICENSE-SOFTWARE", stage_dir / "LICENSE-SOFTWARE")
    shutil.copy2(repo_root / "LICENSE-HARDWARE", stage_dir / "LICENSE-HARDWARE")

    if args.skip_build:
        print(stage_dir)
        return 0

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True)

    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(dist_dir)],
        cwd=stage_dir,
        check=True,
    )
    print(dist_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
