"""Named data-slot indices for the WEPPCLIFF pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from Source.config import WeppcliffConfig


@dataclass(frozen=True)
class PipelineSlots:
    """Resolve original vs gap-filled slot numbers."""

    precip_in: int = 5
    precip_qc: int = 5
    daily_in: int = 7
    alt_in: int = 6
    daily_ts: int = 10
    monthly_ts: int = 11
    annual_ts: int = 12
    monthly_mean: int = 13
    storm_list: int = 8
    storm_df: int = 9
    annual_summary: int = 14

    @classmethod
    def for_config(cls, config: WeppcliffConfig) -> "PipelineSlots":
        if config.do_id:
            return cls(
                precip_in=15,
                precip_qc=15,
                daily_in=18,
                alt_in=6,
                daily_ts=18,
                monthly_ts=19,
                annual_ts=20,
                monthly_mean=21,
                storm_list=16,
                storm_df=17,
                annual_summary=22,
            )
        return cls()