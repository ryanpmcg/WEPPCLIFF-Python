"""CLIGEN breakpoint time helpers — midnight crossing and day boundaries."""

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
    """Storm-interval start using CLIGEN HOUR/MIN (HOUR may be 24 at exact midnight)."""
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

    Post-midnight continuation (interval crosses midnight): 0.000.
    Same-day start: end_time - duration, floored at 0.000.
    """
    if interval_crosses_midnight(clock_hour, clock_min, dur_min):
        return 0.0
    start = storm_start_time_cligen(cligen_hour, cligen_min, dur_min)
    return max(0.0, start)


def gap_breakpoint_time(cligen_hour: str, cligen_min: str, dur_min: float) -> float:
    """Non-precip gap line time within a day (floored at 0.000)."""
    return max(0.0, storm_start_time_cligen(cligen_hour, cligen_min, dur_min))


def should_cap_last_breakpoint_at_midnight(
    last_bp_time: float,
    last_clock_hour: str,
    last_clock_min: str,
    last_dur_min: float,
    next_day_opens_with_midnight_continuation: bool,
) -> bool:
    """
    Day 1 of a midnight-crossing storm should end at 24.000 when the storm continues.
    """
    if last_bp_time >= MIDNIGHT_END - MIDNIGHT_TOLERANCE:
        return False
    if interval_crosses_midnight(last_clock_hour, last_clock_min, last_dur_min):
        return True
    return next_day_opens_with_midnight_continuation