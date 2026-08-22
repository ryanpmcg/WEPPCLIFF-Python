"""CLI argument parsing and validation — port of WEPPCLIFF.R argument functions."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

from Source.config import BOOLEAN_FLAGS, FLAG_NAMES, WeppcliffConfig
from Source.paths import find_this_file

# Human-readable names from WEPPCLIFF.R lines 2760-2770.
FLAG_DISPLAY_NAMES: tuple[str, ...] = (
    "First Run",
    "License Agreement",
    "Input Directory",
    "Output Directory",
    "Export Directory",
    "Plot Directory",
    "Input Filename",
    "Output Filename",
    "File Delimiter",
    "File Search Pattern",
    "Ignore Search Case",
    "Recursive Search",
    "Unit Conversion",
    "Quality Check",
    "Impute Missing Data",
    "Plot Data",
    "Export Data",
    "Export Format",
    "Use Alternative Data",
    "Preserve Missing Data",
    "Cumulative Precipitation",
    "Precipitation Interval",
    "Aggregation Interval",
    "Calculate Erosion Indices",
    "Energy Equation(s)",
    "Storm Identifier",
    "Storm Separation Time Threshhold",
    "Storm Separation Depth Threshhold",
    "Omit Storms",
    "Storm Omission Depth",
    "Quality Checking Option",
    "Checking Threshold",
    "Spiking Threshold",
    "Streaking Threshold",
    "Sticking Threshold",
    "Return Period",
    "Impute Method",
    "Iteration Override",
    "Quick Impute",
    "Impute Verbosity",
    "PyMICE Imputation",
    "Nonstationary Imputation",
    "Koppen-Geiger Zone",
    "Storm Template Sub-Daily",
    "JOMO Multivariate Block",
    "Gap-Fill Engine",
    "Tau QC Threshold",
    "Imputation Report",
    "Time Zone",
    "Start Datetime",
    "End Datetime",
    "Round Time Bounds",
    "Ignore Precipitation Bounds",
    "Precipitation Datetime Format",
    "Alternative Data Datetime Format",
    "Non-Precipitation Datetime Format",
    "CLIGEN Version",
    "Simulation Mode",
    "Breakpoint Format",
    "Wind Information",
    "CLI Trust Groups",
    "CLI Overwrite Groups",
    "Multiprocessing",
    "Profile Code",
    "Profile Interval",
    "Profile Memory",
    "Profile Garbage Collection",
    "Profile Lines",
    "Show Warnings",
    "Verbosity",
)

FLAG_TYPES: tuple[str, ...] = (
    "INSTALLATION ARGUMENTS",
    "INPUT/OUTPUT ARGUMENTS",
    "FUNCTIONALITY ARGUMENTS",
    "PRECIPITATION ARGUMENTS",
    "STORM CONTROL ARGUMENTS",
    "QUALITY CONTROL ARGUMENTS",
    "IMPUTE CONTROL ARGUMENTS",
    "DATETIME CONTROL ARGUMENTS",
    "CLIGEN FILE FORMAT ARGUMENTS",
    "OPTIMIZATION ARGUMENTS",
    "DEVELOPER ARGUMENTS",
)

# 1-based flag indices that begin a new section (WEPPCLIFF.R lines 2784-2794).
FLAG_SECTION_STARTS: tuple[int, ...] = (1, 3, 13, 20, 26, 31, 37, 49, 57, 63, 64)


def parse_args(argv: Optional[list[str]] = None) -> WeppcliffConfig:
    """
    Parse CLI tokens from *argv* (defaults to ``sys.argv[1:]``).

    Value flags use ``-flag value`` pairs. Pure boolean flags (see
    ``BOOLEAN_FLAGS`` in ``Source.config``) are presence-based: the flag
    alone is true; absence is false. A following bare ``t``/``f`` after a
    boolean flag is ignored so legacy ``-qc t`` still enables the flag
    (``-qc f`` is treated as present and therefore true — omit the flag
    for false).
    """
    if argv is None:
        argv = sys.argv[1:]

    raw: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("-") and len(token) > 1:
            flag = token[1:]
            if flag in FLAG_NAMES:
                if flag in BOOLEAN_FLAGS:
                    # Presence => true; do not consume a following value.
                    raw[flag] = "t"
                    # Skip a legacy bare t/f token so it is not mistaken for another arg.
                    if (
                        i + 1 < len(argv)
                        and not argv[i + 1].startswith("-")
                        and argv[i + 1].lower() in ("t", "f", "true", "false")
                    ):
                        i += 2
                        continue
                    i += 1
                    continue
                if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    raw[flag] = argv[i + 1]
                    i += 2
                    continue
                raw[flag] = ""
        i += 1

    config = WeppcliffConfig()
    config.args = list(argv)
    for flag, value in raw.items():
        config.set_flag(flag, value)
    locations = parse_arg_locations(argv)
    config.arg_locations = locations
    if "u" in locations:
        config.u_loc = locations["u"]
    return config


def parse_arg_locations(argv: Optional[list[str]] = None) -> dict[str, int]:
    """
    Locate each flag's position in *argv*.

    Returns a mapping of ``flag -> index`` (0-based) for parity with R ``*.loc`` globals.
    """
    if argv is None:
        argv = sys.argv[1:]

    locations: dict[str, int] = {}
    for i, token in enumerate(argv):
        if token.startswith("-") and len(token) > 1:
            flag = token[1:]
            if flag in FLAG_NAMES:
                locations[flag] = i
    return locations


def early_license_agreement(config: WeppcliffConfig, argv: Optional[list[str]] = None) -> None:
    """
    Assign an early license-agreement value before full argument processing.

    Mirrors ``early_license_agreement`` in WEPPCLIFF.R (lines 131-136).
    """
    if argv is None:
        argv = sys.argv[1:]

    locations = config.arg_locations if config.arg_locations is not None else parse_arg_locations(argv)
    if "la" in locations:
        idx = locations["la"] + 1
        if idx < len(argv):
            config.la = argv[idx]


def _format_leader(display_name: str, flag: str) -> str:
    """Build the dashed leader used in ``assign_args_to_vars`` output."""
    dashes = max(1, 40 - len(display_name) - len(flag))
    return "-" * dashes


def _display_value(config: WeppcliffConfig, flag: str) -> str:
    """Effective value shown in the argument summary (defaults included)."""
    value = config.get_flag(flag)
    if value is None:
        return ""
    return str(value)


def assign_args_to_vars(
    config: WeppcliffConfig,
    argv: Optional[list[str]] = None,
) -> WeppcliffConfig:
    """
    Print formatted argument summary and ensure flag values are on *config*.

    Mirrors ``assign_args_to_vars`` in WEPPCLIFF.R (lines 140-179).
    """
    if argv is None:
        argv = sys.argv[1:]

    locations = config.arg_locations or parse_arg_locations(argv)

    specified = sum(1 for token in argv if token.startswith("-"))
    if specified > 0:
        specified -= 1  # R subtracts one from grep("-", args) count

    print()
    print("*" * 80)
    print(
        f" User Specified Arguments: {specified} of {len(FLAG_NAMES)} possible arguments.",
    )
    print("*" * 80)
    print()

    section_idx = 0
    for counter, (flag, display_name) in enumerate(zip(FLAG_NAMES, FLAG_DISPLAY_NAMES), start=1):
        if counter in FLAG_SECTION_STARTS:
            if counter != 1:
                print()
                print(" ", "*" * 44, sep="")
                print()
            spaces = (45 - len(FLAG_TYPES[section_idx])) // 2
            print(f"{' ' * spaces}{FLAG_TYPES[section_idx]}")
            print()
            section_idx += 1

        value = _display_value(config, flag)
        leader = _format_leader(display_name, flag)
        print(f" {display_name} {leader} {flag}:\t{value}")

    print()
    print("*" * 80)
    print()
    return config


def assign_empty_args(config: WeppcliffConfig) -> WeppcliffConfig:
    """
    Fill unspecified arguments with defaults.

    Mirrors ``assign_empty_args`` in WEPPCLIFF.R (lines 183-243).
    Uses ``Input``, ``Output``, ``Export``, ``Plots`` (not ALL CAPS).
    """
    if config.home_dir is None:
        config.home_dir = str(find_this_file().parent)

    home = Path(config.home_dir)

    if not config.d:
        config.d = str(home / "Input")
    if not config.o:
        config.o = str(home / "Output")
    if not config.e:
        config.e = str(home / "Export")
    if not config.p:
        config.p = str(home / "Plots")

    if not config.u:
        config.u = "m"
    if not config.fn:
        config.fn = "station"
    if not config.fr:
        config.fr = "f"
    if config.la is None:
        config.la = None
    if not config.mp:
        config.mp = "t"
    if not config.qc:
        config.qc = "f"
    if config.rp in (None, ""):
        config.rp = 1000
    if not config.id:
        config.id = "f"
    if not config.pd:
        config.pd = "f"
    if config.ed in (None, ""):
        config.ed = 0
    if not config.ef:
        config.ef = "npz"
    if not config.cp:
        config.cp = "f"
    if not config.pi:
        config.pi = "f"
    if not config.ai:
        config.ai = "f"
    if not config.cv:
        config.cv = "0.0"
    if config.sm in (None, ""):
        config.sm = 1
    if config.bf in (None, ""):
        config.bf = 1
    if config.wi in (None, ""):
        config.wi = 1
    if not config.ctr:
        config.ctr = "all"
    if not config.cow:
        config.cow = "none"
    if not config.ei:
        config.ei = "f"
    if not config.ee:
        config.ee = "R2"
    if not config.im:
        config.im = "DEFAULT"
    if config.io in (None, ""):
        config.io = 100
    if not config.qi:
        config.qi = "f"
    if not config.iv:
        config.iv = "f"
    if not config.tz:
        config.tz = "GMT"
    if not config.rs:
        config.rs = "f"
    if config.fsp is None:
        config.fsp = None
    if not config.isc:
        config.isc = "f"
    if not config.alt:
        config.alt = "f"
    if not config.pmd:
        config.pmd = "f"
    if config.sid in (None, ""):
        config.sid = 1
    if not config.sdt:
        config.sdt = "start"
    if not config.edt:
        config.edt = "end"
    if not config.ipb:
        config.ipb = "f"
    if not config.rtb:
        config.rtb = "m"
    if config.tth in (None, ""):
        config.tth = 6.0
    if config.dth in (None, ""):
        config.dth = 1.27
    if not config.os:
        config.os = "f"
    if config.sod in (None, ""):
        config.sod = 0.5
    if not config.qcop:
        config.qcop = "b"
    if not config.dtf1:
        config.dtf1 = "%Y-%m-%d %H:%M:%S"
    if not config.dtf2:
        config.dtf2 = config.dtf1
    if not config.dtf3:
        config.dtf3 = config.dtf1
    if not config.prof:
        config.prof = "f"
    if config.pint in (None, ""):
        config.pint = 0.02
    if not config.mepr:
        config.mepr = "f"
    if not config.gcpr:
        config.gcpr = "f"
    if not config.lnpr:
        config.lnpr = "f"
    if not config.warn:
        config.warn = "f"
    if not config.verb:
        config.verb = "f"
    if config.chkth in (None, ""):
        config.chkth = 0.9
    if config.spkth in (None, ""):
        config.spkth = 0.9
    if config.strth in (None, ""):
        config.strth = 2.0
    if config.stkth in (None, ""):
        config.stkth = 4.0
    if not config.delim:
        config.delim = " "
    if not config.pm:
        config.pm = "f"
    if not config.ns:
        config.ns = "f"
    if not config.kg:
        config.kg = "AUTO"
    if not config.st:
        config.st = "f"
    if not config.jm:
        config.jm = "f"
    if not config.gf:
        config.gf = "legacy"
    if not config.tauqc:
        config.tauqc = "auto"
    if not config.ir:
        config.ir = "f"

    return config


_ED = re.compile(r"^(?:1[01]|[0-9])$")
_SM = re.compile(r"^[12]$")
_BF = re.compile(r"^[01]$")
_WI = re.compile(r"^[01]$")
_MP = re.compile(r"^(?:t|f|\d{1,5})$", re.IGNORECASE)
_PI = re.compile(r"^(?:f|\d{1,4})$", re.IGNORECASE)
_CV = re.compile(r"^(?:f|0\.0|4\.3|5\.3)$", re.IGNORECASE)
_RTB = re.compile(r"^[mdhf]$", re.IGNORECASE)
_U = re.compile(r"^[me].*", re.IGNORECASE)
_SOD = re.compile(r"^\d+(?:\.\d+)?$")
_GF = re.compile(r"^(?:legacy|phigi)$", re.IGNORECASE)
_EF = re.compile(r"^(?:npz|parquet)$", re.IGNORECASE)
_TAUQC = re.compile(r"^(?:auto|\d+(?:\.\d+)?)$", re.IGNORECASE)

_ARG_CHECKS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("u", _U, "Invalid Argument: [-u] must be specified as 'm' (metric) or 'e' (english) followed by any exceptions."),
    ("mp", _MP, "Invalid Argument: [-mp] must be specified as 't' (true), 'f' (false), or an integer number of cores."),
    (
        "ed",
        _ED,
        "Invalid Argument: [-ed] must be specified as '0' (off), '1' (all), '2' (binary), "
        "'3-6' (precip, storm, breakpoint, daily csv), or '7-11' "
        "(monthly timeseries, annual timeseries, biweekly averages, monthly averages, "
        "annual average csv).",
    ),
    (
        "ef",
        _EF,
        "Invalid Argument: [-ef] must be 'npz' (compressed numpy archive, default) "
        "or 'parquet' (zip of parquet slot tables; requires PyArrow).",
    ),
    ("pi", _PI, "Invalid Argument: [-pi] must be specified as 'f' (false) or an integer up to 1440 (in minutes)."),
    ("ai", _PI, "Invalid Argument: [-ai] must be specified as 'f' (false) or an integer up to 1440 (in minutes)."),
    ("sm", _SM, "Invalid Argument: [-sm] must be specified as '1' (continuous) or '2' (event)."),
    ("bf", _BF, "Invalid Argument: [-bf] must be specified as '0' (daily) or '1' (breakpoint)."),
    ("wi", _WI, "Invalid Argument: [-wi] must be specified as '0' (exclude) or '1' (include)."),
    ("cv", _CV, "Invalid Argument: [-cv] must be specified as 'f' (false) or one of the supported CLIGEN version control numbers (0.0, 4.3, or 5.3)."),
    ("rtb", _RTB, "Invalid Argument: [-rtb] must be specified as 'm' (month), 'd' (day), 'h' (hour), or 'f' (false)."),
    (
        "sod",
        _SOD,
        "Invalid Argument: [-sod] must be specified as a non-negative number of inches.",
    ),
    (
        "gf",
        _GF,
        "Invalid Argument: [-gf] must be 'legacy' (staged MICE) or 'phigi' (physics-informed gap fill).",
    ),
    (
        "tauqc",
        _TAUQC,
        "Invalid Argument: [-tauqc] must be 'auto' or a positive transmittance upper bound.",
    ),
)


def check_args(config: WeppcliffConfig) -> None:
    """
    Halt execution for invalid arguments.

    Mirrors ``check_args`` in WEPPCLIFF.R (lines 247-283).
    """
    for attr, pattern, message in _ARG_CHECKS:
        if pattern.fullmatch(str(getattr(config, attr))) is None:
            raise SystemExit(message)

    if str(config.ai).upper() != "F":
        ai_val = int(config.ai)
        if 1440 % ai_val != 0:
            raise SystemExit(
                "Invalid Argument: [-ai]; 1440 must be perfectly divisible by the specified integer."
            )

    from Source.cli_input import parse_cli_trust_groups

    try:
        parse_cli_trust_groups(config.ctr, default_all=True)
        parse_cli_trust_groups(config.cow, default_all=False)
    except SystemExit as exc:
        raise SystemExit(str(exc).replace("CLI group", "[-ctr/-cow] CLI group")) from exc


def assign_final_args(config: WeppcliffConfig) -> WeppcliffConfig:
    """
    Assign final normalized values to arguments.

    Mirrors ``assign_final_args`` in WEPPCLIFF.R (lines 287-291).
    """
    if str(config.pmd).upper() == "T":
        config.pmd = True
    elif str(config.pmd).upper() == "F":
        config.pmd = False

    config.ed = int(config.ed)
    config.ef = str(config.ef).lower()
    config.sm = int(config.sm)
    config.bf = int(config.bf)
    config.wi = int(config.wi)
    config.io = int(config.io)
    config.rp = int(config.rp)
    config.sid = int(config.sid)
    config.tth = float(config.tth)
    config.dth = float(config.dth)
    config.sod = float(config.sod)
    config.chkth = float(config.chkth)
    config.spkth = float(config.spkth)
    config.strth = float(config.strth)
    config.stkth = float(config.stkth)
    config.pint = float(config.pint)
    config.apply_runtime_flags()

    return config