"""Shared breakpoint emission logic for CLI write and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from Source.cli_breakpoint_time import (
    first_breakpoint_time,
    gap_breakpoint_time,
    should_cap_last_breakpoint_at_midnight,
)
from Source.cli_validate import Breakpoint


@dataclass(frozen=True)
class PcpEmitContext:
    """Pre-extracted precip arrays for breakpoint emission."""

    clock_h: np.ndarray
    clock_m: np.ndarray
    cligen_h: np.ndarray
    cligen_m: np.ndarray
    btw: np.ndarray
    dur: np.ndarray
    precip: np.ndarray
    n: int


def build_pcp_emit_context(
    clock_h: np.ndarray,
    clock_m: np.ndarray,
    cligen_h: np.ndarray,
    cligen_m: np.ndarray,
    btw: np.ndarray,
    dur: np.ndarray,
    precip: np.ndarray,
) -> PcpEmitContext:
    return PcpEmitContext(
        clock_h=clock_h,
        clock_m=clock_m,
        cligen_h=cligen_h,
        cligen_m=cligen_m,
        btw=btw,
        dur=dur,
        precip=precip,
        n=len(clock_h),
    )


def _count_breakpoint_lines(
    loop_bps: int,
    current_pcp: int,
    ctx: PcpEmitContext,
) -> Tuple[int, int, int, bool]:
    """Return (last_pcp, expected_lines, next_current_pcp, overflow)."""
    last_pcp = current_pcp - 1
    expected_lines = 0
    overflow = False

    if loop_bps <= 0:
        return last_pcp, expected_lines, last_pcp + 1, overflow

    last_pcp = current_pcp + loop_bps - 2
    if last_pcp > ctx.n:
        return last_pcp, 0, last_pcp + 1, True

    expected_lines = 1
    for j in range(current_pcp, last_pcp + 1):
        row = j - 1
        if row >= ctx.n:
            return last_pcp, expected_lines, last_pcp + 1, True
        if j > current_pcp and ctx.btw[row] > 0:
            expected_lines += 1
        expected_lines += 1

    return last_pcp, expected_lines, last_pcp + 1, overflow


def emit_day_breakpoints(
    loop_bps: int,
    current_pcp: int,
    ctx: PcpEmitContext,
    *,
    fmt_time,
    fmt_depth,
    zero_depth_str: str,
    next_day_opens_continuation: bool = False,
) -> Tuple[List[Breakpoint], List[str], int]:
    """
    Build breakpoint records and formatted line suffixes for one day.

    *fmt_time* / *fmt_depth* are callables matching ``_fmt_num`` width rules.
    Returns (breakpoints, line_strings, last_pcp).
    """
    breakpoints: List[Breakpoint] = []
    lines: List[str] = []

    if loop_bps <= 0:
        return breakpoints, lines, current_pcp - 1

    cur = current_pcp - 1
    bp_time = first_breakpoint_time(
        ctx.clock_h[cur],
        ctx.clock_m[cur],
        ctx.cligen_h[cur],
        ctx.cligen_m[cur],
        ctx.dur[cur],
    )
    # R: paste(lr, time.out, " ", zero.out, sep="") — no trailing space after depth.
    lines.append(f"{fmt_time(bp_time)} {zero_depth_str}")
    breakpoints.append(Breakpoint(time_h=bp_time, depth_mm=0.0))
    bp_depth = 0.0
    new_day = True
    last_pcp = current_pcp + loop_bps - 2
    _ = next_day_opens_continuation  # R has no next-day midnight cap

    for j in range(current_pcp, last_pcp + 1):
        row = j - 1
        if row >= ctx.n:
            break
        if not new_day and ctx.btw[row] > 0:
            gap_t = gap_breakpoint_time(ctx.cligen_h[row], ctx.cligen_m[row], ctx.dur[row])
            lines.append(f"{fmt_time(gap_t)} {fmt_depth(bp_depth)}")
            breakpoints.append(Breakpoint(time_h=gap_t, depth_mm=bp_depth))
        bp_time = float(ctx.cligen_h[row]) + float(ctx.cligen_m[row]) / 60
        bp_depth = ctx.precip[row] + bp_depth
        lines.append(f"{fmt_time(bp_time)} {fmt_depth(bp_depth)}")
        breakpoints.append(Breakpoint(time_h=bp_time, depth_mm=bp_depth))
        if j == last_pcp or row + 1 >= ctx.n:
            break
        new_day = False

    if breakpoints:
        last_idx = last_pcp - 1
        if 0 <= last_idx < ctx.n:
            last_bp = breakpoints[-1]
            if should_cap_last_breakpoint_at_midnight(
                last_bp.time_h,
                ctx.clock_h[last_idx],
                ctx.clock_m[last_idx],
                ctx.dur[last_idx],
                next_day_opens_continuation,
            ):
                breakpoints[-1] = Breakpoint(time_h=24.0, depth_mm=last_bp.depth_mm)
                lines[-1] = f"{fmt_time(24.0)} {fmt_depth(last_bp.depth_mm)}"

    return breakpoints, lines, last_pcp