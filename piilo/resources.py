"""Locate data files bundled inside the installed ``piilo`` package."""

from importlib.resources import files
from pathlib import Path


def package_path(*parts: str) -> Path:
    """Return the filesystem path of a file shipped with the piilo package."""
    return Path(str(files("piilo").joinpath(*parts)))
