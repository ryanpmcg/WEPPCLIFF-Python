"""Input loading and data export — port of WEPPCLIFF.R I/O functions."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from typing import TYPE_CHECKING, Any, Union

import pandas as pd

from Source.branding import LR
from Source.data_model import WeppcliffData
from Source.paths import create_directory

if TYPE_CHECKING:
    from Source.config import WeppcliffConfig

DataLike = Union[WeppcliffData, dict[int, Any]]

# Human-readable export basename suffixes: {station}-{suffix}.csv
EXPORT_FILE_SUFFIXES: dict[str, str] = {
    "precip": "Precipitation",
    "daily": "Daily-Time-Series",
    "daily_ts": "Daily-Time-Series",
    "storm": "Storm-Characteristics",
    "breakpoints": "Precipitation-Breakpoints",
    "monthly_ts": "Monthly-Time-Series",
    "annual_ts": "Annual-Time-Series",
    "monthly_mean": "Monthly-Averages",
    "annual_mean": "Annual-Average",
    "biweekly_mean": "Biweekly-Averages",
}


def export_csv_path(export_dir: Path, station: str, suffix_key: str) -> Path:
    """Build ``{station}-{Suffix}.csv`` using a key from ``EXPORT_FILE_SUFFIXES``."""
    try:
        suffix = EXPORT_FILE_SUFFIXES[suffix_key]
    except KeyError as exc:
        raise KeyError(f"Unknown export suffix key: {suffix_key}") from exc
    return export_dir / f"{station}-{suffix}.csv"


def _as_store(data: DataLike) -> dict[int, Any]:
    if isinstance(data, WeppcliffData):
        return data.store
    return data


def _resolve_input_path(f: str | Path, config: "WeppcliffConfig") -> Path:
    path = Path(f)
    if not path.is_absolute():
        base = config.in_dir or config.d or "."
        path = Path(base) / path
    return path.resolve()


def load_input_file(f: str | Path, config: "WeppcliffConfig") -> pd.DataFrame:
    """
    Load a single input file (.txt, .tsv, .csv, or CLIGEN .cli).

    Mirrors ``load_input_file`` in WEPPCLIFF.R (lines 303-323).
    """
    path = _resolve_input_path(f, config)
    name_lower = path.name.lower()

    if config.is_true("verb"):
        print(LR, "Loading input file... ", end="", sep="")

    start = time.perf_counter()

    if name_lower.endswith(".cli"):
        from Source.cli_input import load_cligen_input

        frame = load_cligen_input(path, config)
        _ = time.perf_counter() - start
        return frame

    txt = ".txt" in name_lower
    tsv = ".tsv" in name_lower
    csv = ".csv" in name_lower
    matches = sum((txt, tsv, csv))

    if matches == 0:
        raise SystemExit(
            "The input file does not contain an acceptable extension "
            "(.txt, .csv, or CLIGEN .cli)."
        )
    if matches > 1:
        raise SystemExit("The input file contains more than one acceptable extension.")

    dtype = {"DT_1": str, "DT_2": str, "DT_3": str}

    if tsv:
        frame = pd.read_csv(path, sep="\t", dtype=dtype, low_memory=False)
    elif csv:
        frame = pd.read_csv(path, dtype=dtype, low_memory=False)
    else:
        sep = config.delim if config.delim else " "
        read_kwargs = dict(sep=sep, dtype=dtype, low_memory=False)
        if len(sep) == 1:
            frame = pd.read_csv(path, **read_kwargs)
        else:
            frame = pd.read_csv(path, engine="python", **read_kwargs)

    _ = time.perf_counter() - start
    return frame


def check_input_format(file: pd.DataFrame, config: "WeppcliffConfig") -> pd.DataFrame:
    """
    Validate required columns in the input file.

    Mirrors ``check_input_format`` in WEPPCLIFF.R (lines 327-342).
    """
    frame = pd.DataFrame(file)
    old_names = list(frame.columns)

    if not isinstance(frame, pd.DataFrame):
        raise SystemExit("The input file could not be stored as a data frame.")

    if config.is_true("alt"):
        required = [
            "DT_1",
            "PRECIP",
            "DT_2",
            "AIR_TEMP",
            "REL_HUM",
            "DT_3",
            "SO_RAD",
            "W_VEL",
            "W_DIR",
        ]
    elif str(config.cv).upper() == "F":
        required = ["DT_1", "PRECIP"]
    else:
        required = [
            "DT_1",
            "PRECIP",
            "DT_3",
            "MAX_TEMP",
            "MIN_TEMP",
            "SO_RAD",
            "W_VEL",
            "W_DIR",
            "DP_TEMP",
        ]

    present = [var for var in required if var in old_names]
    missing = [var for var in required if var not in present]
    if missing:
        print(f"Some expected variables are missing: {missing} ")
        raise SystemExit(1)

    return frame


def load_input_directory(config: "WeppcliffConfig") -> list[str]:
    """
    Discover input files in the input directory.

    Mirrors ``load_input_directory`` in WEPPCLIFF.R (lines 346-368).
    """
    start = time.perf_counter()
    print("Parsing input files... ", end="", sep="")

    in_dir = Path(config.in_dir or config.d or ".")
    recursive = config.is_true("rs")
    ignore_case = config.is_true("isc")
    pattern = config.fsp if config.fsp else "*"

    candidates: list[Path] = []
    globber = in_dir.rglob if recursive else in_dir.glob
    for path in globber(pattern):
        if path.is_file():
            candidates.append(path)

    extensions = (".txt", ".tsv", ".csv", ".cli")
    files: list[Path] = []
    for path in candidates:
        name = path.name.lower() if ignore_case else path.name
        check_name = name.lower() if not ignore_case else name
        if any(ext in check_name for ext in extensions):
            files.append(path)

    seen: set[Path] = set()
    unique_files: list[Path] = []
    for path in files:
        if path not in seen:
            seen.add(path)
            unique_files.append(path)

    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.3f} seconds.")
    return [str(p) for p in unique_files]


def _resolve_output_name(config: "WeppcliffConfig", data: DataLike) -> str:
    if str(config.fn).upper() == "STATION":
        store = _as_store(data)
        meta = store.get(4, {})
        if isinstance(meta, dict):
            station = meta.get("STATION", "station")
        else:
            station = "station"
        return str(station)
    return str(config.fn)


def export_data(data: DataLike, config: "WeppcliffConfig") -> None:
    """
    Export the primary data structure to file.

    Uses pickle (``.pkl``) instead of R ``.rds`` for ``ed`` values 1 and 2.
    """
    if config.is_true("verb"):
        print(LR, "Exporting data... ", end="", sep="")

    export_dir = Path(config.ex_dir or config.e or ".")
    create_directory(export_dir)

    ed = int(config.ed)
    filename = _resolve_output_name(config, data)
    store = _as_store(data)

    if ed in (1, 2):
        out_path = export_dir / f"{filename}.pkl"
        with out_path.open("wb") as handle:
            pickle.dump(store, handle)

    if ed in (1, 3):
        export_to_csv(data, config, og_df=5, gf_df=15, suffix_key="precip")
    if ed in (1, 4):
        export_to_csv(data, config, og_df=9, gf_df=17, suffix_key="storm")
    if ed in (1, 5):
        export_pcp_bps(data, config, og_df=5, gf_df=15, suffix_key="breakpoints")
    if ed in (1, 6):
        export_weather_summary(data, config, kind="daily_ts")
    if ed in (1, 7):
        export_weather_summary(data, config, kind="monthly_ts")
    if ed in (1, 8):
        export_weather_summary(data, config, kind="annual_ts")
    if ed in (1, 9):
        export_weather_summary(data, config, kind="biweekly_mean")
    if ed in (1, 10):
        export_weather_summary(data, config, kind="monthly_mean")
    if ed in (1, 11):
        export_weather_summary(data, config, kind="annual_mean")


def export_weather_summary(
    data: DataLike,
    config: "WeppcliffConfig",
    *,
    kind: str,
) -> None:
    """Export monthly/annual weather timeseries or climatological averages."""
    from Source.export_weather import build_weather_export_frame

    export_dir = Path(config.ex_dir or config.e or ".")
    create_directory(export_dir)

    if isinstance(data, WeppcliffData):
        frame = build_weather_export_frame(data, config, kind)  # type: ignore[arg-type]
    else:
        frame = build_weather_export_frame(WeppcliffData(data), config, kind)  # type: ignore[arg-type]

    filename = _resolve_output_name(config, data)
    out_path = export_csv_path(export_dir, filename, kind)
    frame.to_csv(out_path, index=False)


def export_to_csv(
    data: DataLike,
    config: "WeppcliffConfig",
    og_df: int,
    gf_df: int,
    suffix_key: str,
) -> None:
    """Write a selected dataframe slot to CSV."""
    export_dir = Path(config.ex_dir or config.e or ".")
    create_directory(export_dir)

    df_loc = gf_df if config.is_true("id") else og_df
    store = _as_store(data)
    frame = store[df_loc]
    if suffix_key == "storm":
        from Source.export_weather import annotate_storm_export_frame

        frame = annotate_storm_export_frame(frame, config)
    filename = _resolve_output_name(config, data)
    out_path = export_csv_path(export_dir, filename, suffix_key)
    frame.to_csv(out_path, index=False)


def export_pcp_bps(
    data: DataLike,
    config: "WeppcliffConfig",
    og_df: int,
    gf_df: int,
    suffix_key: str,
) -> None:
    """
    Export precipitation breakpoint data with zero-padding rows.

    Mirrors ``export_pcp_bps`` in WEPPCLIFF.R (lines 2239-2263).
    """
    export_dir = Path(config.ex_dir or config.e or ".")
    create_directory(export_dir)

    df_loc = gf_df if config.is_true("id") else og_df
    store = _as_store(data)
    plot_df = store[df_loc]
    filename = _resolve_output_name(config, data)

    dt_parsed = pd.to_datetime(plot_df["DT_1"])
    precip_col = plot_df["PRECIP"].to_numpy()
    dur_col = plot_df["DUR"].to_numpy(dtype=float)

    dt_start = dt_parsed - pd.to_timedelta(dur_col, unit="m")
    existing = set(dt_parsed)
    missing_mask = np.array([t not in existing for t in dt_start])
    if not missing_mask.any():
        df_out = pd.DataFrame({"DT": dt_parsed, "PRECIP": precip_col})
    else:
        df_out = pd.DataFrame({
            "DT": pd.concat([dt_parsed, dt_start[missing_mask]]).to_numpy(),
            "PRECIP": np.concatenate([precip_col, np.zeros(missing_mask.sum())]),
        })
    df_out = df_out.sort_values("DT", kind="mergesort").reset_index(drop=True)

    out_path = export_csv_path(export_dir, filename, suffix_key)
    df_out.to_csv(out_path, index=False)