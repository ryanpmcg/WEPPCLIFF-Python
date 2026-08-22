"""Unit tests for midnight-crossing duration split (R parity)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.duration import modify_midnight_breakpoints  # noqa: E402


def test_post_midnight_row_keeps_tfm_duration() -> None:
    """
    R sets the post-midnight row DUR to TFM (minutes past midnight) and the
    new midnight row to (original DUR - TFM). Python previously assigned the
    pre-midnight duration to both, which broke BTW and gap emission.
    """
    # Interval ends at 00:18 with 30 min duration → crosses midnight.
    # Pre-midnight: 12 min ending at 00:00; post-midnight: 18 min (TFM).
    df = pd.DataFrame(
        {
            "DT_1": ["2000-02-26 00:18:00"],
            "PRECIP": [0.30],
            "DUR": [30.0],
        }
    )
    out = modify_midnight_breakpoints(df, "%Y-%m-%d %H:%M:%S", SimpleNamespace())
    assert len(out) == 2
    mid = out.loc[out["DT_1"] == "2000-02-26 00:00:00"].iloc[0]
    post = out.loc[out["DT_1"] == "2000-02-26 00:18:00"].iloc[0]
    assert mid["DUR"] == 12.0
    assert post["DUR"] == 18.0
    assert abs(float(mid["PRECIP"]) + float(post["PRECIP"]) - 0.30) < 1e-12
