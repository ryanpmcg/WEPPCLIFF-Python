"""CLIGEN breakpoint time helpers — match R WEPPCLIFF.R create_cli_file formulas."""

from __future__ import annotations

MIDNIGHT_END = 24.0
MIDNIGHT_TOLERANCE = 1e-6


def interval_crosses_midnight(clock_hour: str, clock_min: str, dur_min: float) -> bool:
    """
    True when a precip interval ends after calendar midnight but started before it.

    Uses wall-clock hour/minute (00 at midnight), not the CLIGEN HOUR=24 substitution.
    """
    end_min = int(clock_hour) * 60 + int(clock_min)
    return end_min < float(dur_min)


def storm_start_time_cligen(hour: str, minute: str, dur_min: float) -> float:
    """
    Storm-interval start using CLIGEN HOUR/MIN (HOUR may be 24 at exact midnight).

    Matches R: ``as.numeric(HOUR) + as.numeric(MIN)/60 - as.numeric(DUR)/60``.
    Does not clamp; R can emit values slightly above 24 or below 0 on edge cases.
    """
    return float(hour) + float(minute) / 60.0 - float(dur_min) / 60.0


def first_breakpoint_time(
    clock_hour: str,
    clock_min: str,
    cligen_hour: str,
    cligen_min: str,
    dur_min: float,
) -> float:
    """
    Time for the depth-0.00 starter line of a calendar day.

    R uses the raw CLIGEN start formula only (no midnight-continuation rewrite
    and no floor at 0). ``clock_hour`` / ``clock_min`` are unused but kept for
    call-site compatibility.
    """
    _ = (clock_hour, clock_min)
    return storm_start_time_cligen(cligen_hour, cligen_min, dur_min)


def gap_breakpoint_time(cligen_hour: str, cligen_min: str, dur_min: float) -> float:
    """Non-precip gap line time within a day (R raw formula, no floor)."""
    return storm_start_time_cligen(cligen_hour, cligen_min, dur_min)


def should_cap_last_breakpoint_at_midnight(
    last_bp_time: float,
    last_clock_hour: str,
    last_clock_min: str,
    last_dur_min: float,
    next_day_opens_with_midnight_continuation: bool,
) -> bool:
    """
    R does not rewrite the last breakpoint of a day to 24.000.

    Always False for R .cli parity. Signature retained for call sites.
    """
    _ = (
        last_bp_time,
        last_clock_hour,
        last_clock_min,
        last_dur_min,
        next_day_opens_with_midnight_continuation,
    )
    return False