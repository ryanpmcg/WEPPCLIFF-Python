"""Main orchestration — port of WEPPCLIFF.R main() and program entry."""

from __future__ import annotations

import os
import subprocess
import sys
import time

from Source.args import (
    assign_args_to_vars,
    assign_empty_args,
    check_args,
    early_license_agreement,
    parse_args,
)
from Source.branding import print_license_agreement, print_weppcliff_logo, prompt_license_agreement
from Source.config import WeppcliffConfig, set_config
from Source.io import load_input_directory
from Source.paths import create_base_directory, find_this_file, set_wds
from Source.utils import stop_quietly
from Source.workflow import parallel_execution, series_execution


def install_dependencies() -> None:
    root = str(find_this_file().parent)
    req = os.path.join(root, "requirements.txt")
    print("Installing Python package dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", root])
    print("\nDone.")


def load_packages() -> None:
    """No-op in Python (R loads CRAN packages here)."""
    print("Loading Python package dependencies... Completed in 0.00 seconds.")


def main(config: WeppcliffConfig | None = None) -> None:
    """Port of WEPPCLIFF.R main() lines 2689-2734."""
    if config is None:
        config = parse_args()
    early_license_agreement(config, config.args)
    set_config(config)

    print_weppcliff_logo()
    if str(config.la or "").upper() == "Y":
        print_license_agreement()
    else:
        prompt_license_agreement()

    assign_empty_args(config)
    assign_args_to_vars(config, config.args)
    create_base_directory(config)
    set_wds(config)

    if config.tz:
        os.environ.setdefault("TZ", str(config.tz))

    if config.is_true("fr"):
        install_dependencies()
        load_packages()
        print("Installation successful. WEPPCLIFF is now ready to run.")
        stop_quietly()

    t0 = time.perf_counter()
    check_args(config)
    if not config.is_true("fr"):
        load_packages()

    if config.f:
        s = 1
        fpath = config.f
        if not os.path.isabs(fpath):
            fpath = os.path.join(config.in_dir or config.d or "Input", fpath)
        files = [fpath]
    else:
        files = load_input_directory(config)
        s = len(files)

    mp_val = str(config.mp or "").strip()
    use_parallel = s > 1 and (config.is_true("mp") or mp_val.isdigit())
    if not use_parallel:
        series_execution(files, s, config)
    else:
        parallel_execution(files, s, config)

    elapsed = time.perf_counter() - t0
    print(f"Program executed successfully in {elapsed:.2f} seconds... now exiting WEPPCLIFF script.",)
    print()
    print("*" * 80)
    print()


def cli_entry() -> None:
    """CLI entry point invoked by weppcliff.py and the weppcliff console script."""
    try:
        main(parse_args())
    except SystemExit:
        raise
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)