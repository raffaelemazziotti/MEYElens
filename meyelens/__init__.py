# meyelens/__init__.py
from importlib.metadata import version, PackageNotFoundError

for _dist_name in ("meyelens", "meyelens-headless"):
    try:
        __version__ = version(_dist_name)
        break
    except PackageNotFoundError:
        continue
else:
    __version__ = "0.0.0"
