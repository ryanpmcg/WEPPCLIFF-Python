"""Path and working-directory helpers — port of WEPPCLIFF.R lines 84-114."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from Source.config import WeppcliffConfig


def find_this_file() -> Path:
    """
    Locate the WEPPCLIFF entry script path.

    Credit: Katie Barnhart @kbarnhart — R equivalent uses ``commandArgs`` /
    ``sys.frames``; Python uses ``sys.argv[0]`` with a module fallback.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()

    if sys.argv:
        candidate = Path(sys.argv[0]).resolve()
        if candidate.exists():
            return candidate

    main_mod = sys.modules.get("__main__")
    if main_mod and getattr(main_mod, "__file__", None):
        return Path(main_mod.__file__).resolve()

    return Path(__file__).resolve()


def set_wds(config: WeppcliffConfig) -> None:
    """Set home and runtime directory paths on *config*."""
    config.home_dir = str(find_this_file().parent)

    config.in_dir = config.d or config.in_dir
    config.out_dir = config.o or config.out_dir
    config.ex_dir = config.e or config.ex_dir
    config.plot_dir = config.p or config.plot_dir


def create_directory(path: str | os.PathLike[str]) -> None:
    """Create *path* if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def create_base_directory(config: WeppcliffConfig) -> None:
    """Create input, output, export, and plot directories."""
    for directory in (config.d, config.o, config.e, config.p):
        if directory:
            create_directory(directory)