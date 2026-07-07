"""CLIGEN .cli file input — parse daily and breakpoint formats for WEPPCLIFF."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pandas as pd

from Source.cli_validate import BP_LINE_RE, Breakpoint, date_key_from_parts

if TYPE_CHECKING:
    from Source.config import WeppcliffConfig

_NUM = r"(-?\d+\.\d+)"
_NUM_OR_NA = r"(-?(?:\d+\.\d*|\d+)|NA)"

DAILY_BF1_RE = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d{{1,4}})\s+(\d+)\s+"
    rf"{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s*$"
)
DAILY_BF0_RE = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d{{1,4}})\s+"
    rf"{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+"
    rf"{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s+{_NUM_OR_NA}\s*$"
)

CLI_TRUST_GROUPS: frozenset[str] = frozenset({
    "all", "none", "meta", "monthly", "daily", "precip", "storm", "bps",
})


@dataclass
class CligenHeader:
    cv: str
    sm: int
    bf: int
    wi: int
    station: str
    lat: float
    lon: float
    elev: float
    obs_yrs: float
    b_yr: int
    yrs_sim: int
    monthly_max_temp: list[float] = field(default_factory=list)
    monthly_min_temp: list[float] = field(default_factory=list)
    monthly_rad: list[float] = field(default_factory=list)
    monthly_precip: list[float] = field(default_factory=list)


@dataclass
class CligenDayRecord:
    day: int
    month: int
    year: int
    tmax: float | None
    tmin: float | None
    rad: float | None
    wvel: float | None
    wdir: float | None
    tdew: float | None
    prcp: float | None = None
    dur_h: float | None = None
    tp: float | None = None
    ip: float | None = None
    breaks_declared: int | None = None
    breakpoints: list[Breakpoint] = field(default_factory=list)

    @property
    def ymd(self) -> str:
        return f"{self.year:04d}{self.month:02d}{self.day:02d}"

    @property
    def date_key(self) -> str:
        return date_key_from_parts(self.day, self.month, self.year)


@dataclass
class CligenInput:
    header: CligenHeader
    days: list[CligenDayRecord]
    is_breakpoint: bool


def _parse_float(token: str) -> float | None:
    token = token.strip()
    if token.upper() == "NA" or token == "":
        return None
    return float(token)


def _parse_monthly_values(line: str) -> list[float]:
    return [float(v) for v in line.split()]


def _resolve_calendar_year(year: int, b_yr: int) -> int:
    """Map CLIGEN simulation-year indices to pandas-safe calendar years."""
    if year >= 1000:
        return year
    if b_yr >= 1000:
        return b_yr + year - 1
    return 1900 + year


def _find_body_start(lines: Sequence[str]) -> tuple[int, bool]:
    for idx, line in enumerate(lines):
        lower = line.lower()
        if ("day month year" in lower or "da mo year" in lower) and "breaks" in lower:
            return idx, True
        if ("day month year" in lower or "da mo year" in lower) and "prcp" in lower:
            return idx, False
    raise SystemExit("The CLIGEN input file does not contain a recognizable daily header row.")


def parse_cligen_text(text: str) -> CligenInput:
    """Parse a CLIGEN .cli file into structured header and daily records."""
    lines = text.splitlines()
    if len(lines) < 14:
        raise SystemExit("The CLIGEN input file is too short to contain a valid header.")

    cv = lines[0].strip()
    mode_parts = lines[1].split()
    sm = int(mode_parts[0])
    bf = int(mode_parts[1])
    wi = int(mode_parts[2]) if len(mode_parts) > 2 else 1

    station = "STATION"
    if "Station:" in lines[2]:
        station = lines[2].split("Station:", 1)[1].strip() or station

    meta_parts = lines[4].split()
    if len(meta_parts) < 6:
        raise SystemExit("The CLIGEN input file header is missing station metadata.")
    lat, lon, elev = float(meta_parts[0]), float(meta_parts[1]), float(meta_parts[2])
    obs_yrs = float(meta_parts[3])
    b_yr = int(meta_parts[4])
    yrs_sim = int(meta_parts[5])

    monthly_max = _parse_monthly_values(lines[6])
    monthly_min = _parse_monthly_values(lines[8])
    monthly_rad = _parse_monthly_values(lines[10])
    monthly_precip = _parse_monthly_values(lines[12])

    header = CligenHeader(
        cv=cv,
        sm=sm,
        bf=bf,
        wi=wi,
        station=station,
        lat=lat,
        lon=lon,
        elev=elev,
        obs_yrs=obs_yrs,
        b_yr=b_yr,
        yrs_sim=yrs_sim,
        monthly_max_temp=monthly_max,
        monthly_min_temp=monthly_min,
        monthly_rad=monthly_rad,
        monthly_precip=monthly_precip,
    )

    body_start, is_breakpoint = _find_body_start(lines)
    days = _parse_body_lines(lines[body_start + 2 :], is_breakpoint=is_breakpoint)
    for day in days:
        day.year = _resolve_calendar_year(day.year, header.b_yr)
    if not days:
        raise SystemExit("The CLIGEN input file does not contain any daily data rows.")

    return CligenInput(header=header, days=days, is_breakpoint=is_breakpoint)


def _parse_body_lines(lines: Sequence[str], *, is_breakpoint: bool) -> list[CligenDayRecord]:
    days: list[CligenDayRecord] = []
    i = 0
    daily_re = DAILY_BF1_RE if is_breakpoint else DAILY_BF0_RE

    while i < len(lines):
        m = daily_re.match(lines[i])
        if not m:
            i += 1
            continue

        if is_breakpoint:
            day = CligenDayRecord(
                day=int(m.group(1)),
                month=int(m.group(2)),
                year=int(m.group(3)),
                breaks_declared=int(m.group(4)),
                tmax=_parse_float(m.group(5)),
                tmin=_parse_float(m.group(6)),
                rad=_parse_float(m.group(7)),
                wvel=_parse_float(m.group(8)),
                wdir=_parse_float(m.group(9)),
                tdew=_parse_float(m.group(10)),
            )
            i += 1
            while i < len(lines):
                bm = BP_LINE_RE.match(lines[i])
                if bm:
                    day.breakpoints.append(
                        Breakpoint(time_h=float(bm.group(1)), depth_mm=float(bm.group(2)))
                    )
                    i += 1
                    continue
                if daily_re.match(lines[i]):
                    break
                i += 1
        else:
            day = CligenDayRecord(
                day=int(m.group(1)),
                month=int(m.group(2)),
                year=int(m.group(3)),
                prcp=_parse_float(m.group(4)),
                dur_h=_parse_float(m.group(5)),
                tp=_parse_float(m.group(6)),
                ip=_parse_float(m.group(7)),
                tmax=_parse_float(m.group(8)),
                tmin=_parse_float(m.group(9)),
                rad=_parse_float(m.group(10)),
                wvel=_parse_float(m.group(11)),
                wdir=_parse_float(m.group(12)),
                tdew=_parse_float(m.group(13)),
            )
            i += 1

        days.append(day)

    return days


def parse_cligen_file(path: Path | str) -> CligenInput:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_cligen_text(text)


def parse_cli_trust_groups(spec: str | None, *, default_all: bool = True) -> set[str]:
    """Parse a comma-separated CLI trust/overwrite group specification."""
    if spec is None or str(spec).strip() == "":
        return set(CLI_TRUST_GROUPS - {"all", "none"}) if default_all else set()
    token = str(spec).strip().lower()
    if token == "all":
        return set(CLI_TRUST_GROUPS - {"all", "none"})
    if token == "none":
        return set()
    groups = {part.strip().lower() for part in token.split(",") if part.strip()}
    unknown = groups - CLI_TRUST_GROUPS
    if unknown:
        raise SystemExit(
            f"Invalid CLI group(s): {sorted(unknown)}. "
            f"Valid groups: {', '.join(sorted(CLI_TRUST_GROUPS - {'all', 'none'}))}, all, none."
        )
    return groups


def group_is_trusted(group: str, trust: set[str], overwrite: set[str]) -> bool:
    if group in overwrite:
        return False
    return group in trust


def apply_config_from_cligen(config: "WeppcliffConfig", cligen: CligenInput) -> None:
    """Apply CLIGEN header controls to runtime config when reading a .cli input."""
    header = cligen.header
    config.cv = header.cv
    config.sm = header.sm
    config.bf = header.bf
    config.wi = header.wi
    if cligen.is_breakpoint:
        config.pi = "f"


def _hours_to_datetime(year: int, month: int, day: int, hours: float) -> str:
    base = pd.Timestamp(year=year, month=month, day=day)
    if hours >= 24.0:
        carry_days = int(hours // 24.0)
        hours = hours - carry_days * 24.0
        base = base + pd.Timedelta(days=carry_days)
    whole_h = int(hours)
    minutes = int(round((hours - whole_h) * 60.0))
    if minutes == 60:
        whole_h += 1
        minutes = 0
    dt = base + pd.Timedelta(hours=whole_h, minutes=minutes)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _breakpoints_to_rows(day: CligenDayRecord) -> list[tuple[str, float]]:
    if not day.breakpoints:
        return []

    rows: list[tuple[str, float]] = []
    for idx, bp in enumerate(day.breakpoints):
        if idx == 0:
            increment = 0.0
        else:
            increment = max(0.0, bp.depth_mm - day.breakpoints[idx - 1].depth_mm)
        if increment > 0.0 or idx == 0:
            rows.append((_hours_to_datetime(day.year, day.month, day.day, bp.time_h), increment))
    return rows


def _daily_event_to_rows(day: CligenDayRecord) -> list[tuple[str, float]]:
    prcp = day.prcp or 0.0
    dur_h = day.dur_h or 0.0
    tp = day.tp if day.tp is not None else 0.5
    if prcp <= 0.0 or dur_h <= 0.0:
        return []

    start_h = max(0.0, 12.0 - dur_h * (1.0 - tp))
    end_h = min(23.99, start_h + dur_h)
    return [
        (_hours_to_datetime(day.year, day.month, day.day, start_h), 0.0),
        (_hours_to_datetime(day.year, day.month, day.day, end_h), prcp),
    ]


def _daily_weather_row(day: CligenDayRecord) -> dict[str, object]:
    dt3 = pd.Timestamp(year=day.year, month=day.month, day=day.day).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "DT_3": dt3,
        "MAX_TEMP": day.tmax,
        "MIN_TEMP": day.tmin,
        "SO_RAD": day.rad,
        "W_VEL": day.wvel,
        "W_DIR": day.wdir,
        "DP_TEMP": day.tdew,
        "YYYYMMDD": day.ymd,
    }


def cligen_to_input_frame(
    cligen: CligenInput,
    config: "WeppcliffConfig",
) -> pd.DataFrame:
    """Convert parsed CLIGEN content into a WEPPCLIFF input dataframe."""
    trust = parse_cli_trust_groups(getattr(config, "ctr", None), default_all=True)
    overwrite = parse_cli_trust_groups(getattr(config, "cow", None), default_all=False)

    precip_rows: list[dict[str, object]] = []
    daily_by_ymd: dict[str, dict[str, object]] = {}

    for day in cligen.days:
        daily_by_ymd[day.ymd] = _daily_weather_row(day)
        if cligen.is_breakpoint:
            if group_is_trusted("bps", trust, overwrite) and group_is_trusted("storm", trust, overwrite):
                storm_rows = _breakpoints_to_rows(day)
            elif group_is_trusted("precip", trust, overwrite):
                total = day.breakpoints[-1].depth_mm if day.breakpoints else 0.0
                storm_rows = [
                    (_hours_to_datetime(day.year, day.month, day.day, 12.0), total),
                ] if total > 0 else []
            else:
                storm_rows = []
        else:
            if group_is_trusted("storm", trust, overwrite) or group_is_trusted("precip", trust, overwrite):
                storm_rows = _daily_event_to_rows(day)
            else:
                storm_rows = []

        if storm_rows:
            for dt_1, amount in storm_rows:
                precip_rows.append({"DT_1": dt_1, "PRECIP": amount, "YYYYMMDD": day.ymd})
        else:
            precip_rows.append({
                "DT_1": _hours_to_datetime(day.year, day.month, day.day, 12.0),
                "PRECIP": 0.0,
                "YYYYMMDD": day.ymd,
            })

    precip_df = pd.DataFrame(precip_rows)
    if not group_is_trusted("precip", trust, overwrite):
        precip_df["PRECIP"] = np.nan
    if not group_is_trusted("storm", trust, overwrite):
        precip_df["DT_1"] = np.nan

    weather_df = pd.DataFrame(daily_by_ymd.values())
    if not weather_df.empty:
        precip_df = precip_df.merge(
            weather_df,
            on="YYYYMMDD",
            how="left",
            suffixes=("_drop", ""),
        )
        drop_cols = [c for c in precip_df.columns if c.endswith("_drop")]
        precip_df = precip_df.drop(columns=drop_cols, errors="ignore")
    weather_cols = ["DT_3", "MAX_TEMP", "MIN_TEMP", "SO_RAD", "W_VEL", "W_DIR", "DP_TEMP"]
    for col in weather_cols:
        if col not in precip_df.columns:
            precip_df[col] = np.nan
    if not group_is_trusted("daily", trust, overwrite):
        for col in weather_cols:
            precip_df[col] = np.nan

    header = cligen.header
    meta = {
        "STATION": header.station,
        "LAT": header.lat,
        "LON": header.lon,
        "ELEV": header.elev,
        "OBS_YRS": header.obs_yrs,
        "B_YR": header.b_yr,
        "YRS_SIM": header.yrs_sim,
    }
    if not group_is_trusted("meta", trust, overwrite):
        for key in meta:
            meta[key] = np.nan

    if group_is_trusted("monthly", trust, overwrite):
        config.cligen_monthly = {
            "MAX_TEMP": header.monthly_max_temp,
            "MIN_TEMP": header.monthly_min_temp,
            "SO_RAD": header.monthly_rad,
            "PRECIP": header.monthly_precip,
        }
    else:
        config.cligen_monthly = None

    for key, value in meta.items():
        precip_df[key] = value

    precip_df = precip_df.drop(columns=["YYYYMMDD"])
    ordered = [
        "DT_1", "PRECIP", "DT_3", "MAX_TEMP", "MIN_TEMP", "SO_RAD", "W_VEL", "W_DIR", "DP_TEMP",
        "STATION", "LAT", "LON", "ELEV", "OBS_YRS", "B_YR", "YRS_SIM",
    ]
    for col in ordered:
        if col not in precip_df.columns:
            precip_df[col] = np.nan
    extra = [c for c in precip_df.columns if c not in ordered]
    return precip_df[ordered + extra]


def load_cligen_input(path: Path | str, config: "WeppcliffConfig") -> pd.DataFrame:
    """Load a CLIGEN .cli file and return a WEPPCLIFF-compatible input dataframe."""
    cligen = parse_cligen_file(path)
    apply_config_from_cligen(config, cligen)
    config.apply_runtime_flags()
    return cligen_to_input_frame(cligen, config)


def is_cligen_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() == ".cli"