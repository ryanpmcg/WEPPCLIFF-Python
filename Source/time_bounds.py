"""Datetime bounds — port of R time_bounds and start/end helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence, Union

import pandas as pd

from .config import LR, WeppcliffConfig


def _to_timestamp(value: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    return pd.Timestamp(value)


def parse_user_datetime(value: str, fmt: str) -> pd.Timestamp:
    """Parse CLI datetime strings; CLIGEN ``24:00:00`` rolls to next-day midnight (R parity)."""
    text = str(value).strip()
    date_part, _, time_part = text.rpartition(" ")
    if time_part == "24:00:00":
        midnight = pd.to_datetime(f"{date_part} 00:00:00", format=fmt)
        return midnight + pd.Timedelta(days=1)
    return pd.to_datetime(text, format=fmt)


def start_of_year(
    date: Union[str, datetime, pd.Timestamp],
    config: WeppcliffConfig,
) -> pd.Timestamp:
    """Find beginning of next full calendar year (R ``start_of_year``)."""
    real_start = _to_timestamp(date)
    good_start = real_start.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    month_start = real_start.replace(month=2, day=1, hour=0, minute=0, second=0, microsecond=0)
    day_start = real_start.replace(hour=0, minute=0, second=0, microsecond=0)
    hour_start = real_start.replace(minute=0, second=0, microsecond=0)

    rtb = config.rtb.upper()
    if rtb == "M" and real_start > month_start:
        good_start = good_start.replace(year=good_start.year + 1)
    elif rtb == "D" and day_start > good_start:
        good_start = good_start.replace(year=good_start.year + 1)
    elif rtb == "H" and hour_start > good_start:
        good_start = good_start.replace(year=good_start.year + 1)
    elif rtb == "F" and real_start > good_start:
        good_start = good_start.replace(year=good_start.year + 1)
    return good_start


def start_of_day(
    date: Union[str, datetime, pd.Timestamp],
    config: WeppcliffConfig,
) -> pd.Timestamp:
    """Find beginning of calendar day (R ``start_of_day``)."""
    real_start = _to_timestamp(date)
    return real_start.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_year(
    date: Union[str, datetime, pd.Timestamp],
    config: WeppcliffConfig,
) -> pd.Timestamp:
    """Find end of last full calendar year (R ``end_of_year``)."""
    real_end = _to_timestamp(date)
    good_end = real_end.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=0)
    month_end = real_end.replace(month=11, day=30, hour=23, minute=59, second=59, microsecond=0)
    day_end = real_end.replace(hour=23, minute=59, second=59, microsecond=0)
    hour_end = real_end.replace(minute=59, second=59, microsecond=0)

    rtb = config.rtb.upper()
    if rtb == "M" and real_end < month_end:
        good_end = good_end.replace(year=good_end.year - 1)
    elif rtb == "D" and day_end < good_end:
        good_end = good_end.replace(year=good_end.year - 1)
    elif rtb == "H" and hour_end < good_end:
        good_end = good_end.replace(year=good_end.year - 1)
    elif rtb == "F" and real_end < good_end:
        good_end = good_end.replace(year=good_end.year - 1)
    return good_end


def end_of_day(
    date: Union[str, datetime, pd.Timestamp],
    config: WeppcliffConfig,
) -> pd.Timestamp:
    """Find end of calendar day (R ``end_of_day``)."""
    real_end = _to_timestamp(date)
    return real_end.replace(hour=23, minute=59, second=59, microsecond=0)


def time_bounds(
    start: Union[str, datetime, pd.Timestamp],
    end: Union[str, datetime, pd.Timestamp],
    config: WeppcliffConfig,
) -> Sequence[pd.Timestamp]:
    """Determine acceptable simulation start/end datetimes (R ``time_bounds``)."""
    start_ts = _to_timestamp(start)
    end_ts = _to_timestamp(end)

    if config.sdt.upper() == "START":
        first_dt = start_ts
    else:
        first_dt = parse_user_datetime(config.sdt, config.dtf1)

    if config.edt.upper() == "END":
        last_dt = end_ts
    else:
        last_dt = parse_user_datetime(config.edt, config.dtf1)

    if config.sm == 1:
        good_start = start_of_year(first_dt, config)
        good_end = end_of_year(last_dt, config)
        if good_start > good_end:
            raise ValueError(
                LR + LR
                + "Insufficient Data: At least one full calendar year must be "
                "available with your DATETIME CONTROL ARGUMENTS."
                + LR + LR
            )
        return [good_start, good_end, first_dt, last_dt]

    if config.sm == 2:
        good_start = start_of_day(first_dt, config)
        good_end = end_of_day(last_dt, config)
        if good_start > good_end:
            raise ValueError(
                LR + LR
                + "Insufficient Data: At least one precipitation event and daily "
                "data for the same day must be available with your DATETIME "
                "CONTROL ARGUMENTS."
                + LR + LR
            )
        return [good_start, good_end, first_dt, last_dt]

    return [first_dt, last_dt, first_dt, last_dt]