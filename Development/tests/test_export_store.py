"""Round-trip tests for live-store binary export (npz / parquet)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.args import assign_empty_args, check_args, parse_args  # noqa: E402
from Source.config import FLAG_NAMES, WeppcliffConfig  # noqa: E402
from Source.data_model import WeppcliffData  # noqa: E402
from Source.export_store import (  # noqa: E402
    load_exported_store,
    read_npz_store,
    write_binary_store,
    write_npz_store,
)
from Source.io import export_data  # noqa: E402


def _sample_store() -> dict[int, object]:
    precip = pd.DataFrame(
        {
            "DT_1": ["2000-01-02 11:20:00", "2000-01-02 11:35:00"],
            "PRECIP": [1.5, np.nan],
            "DUR": [15.0, 10.0],
            "PRECIP_QC": ["PASS", "PASS"],
        }
    )
    daily = pd.DataFrame(
        {
            "YYYYMMDD": ["20000101", "20000102"],
            "MIN_TEMP": [-3.0, 2.0],
            "PRECIP": [np.nan, 1.5],
            "BPS": [0, 2],
        }
    )
    storms = [
        precip.iloc[[0]].copy(),
        precip.iloc[[1]].copy(),
    ]
    empty_alt = pd.DataFrame({"DT_2": pd.Series(dtype=float), "AIR_TEMP": pd.Series(dtype=float)})
    return {
        1: precip,
        2: empty_alt,
        4: {"STATION": "KMQE", "LAT": 42.2, "LON": -71.1, "ELEV": 199.0},
        5: precip.copy(),
        8: storms,
        10: daily,
    }


def test_ef_flag_registered() -> None:
    assert "ef" in FLAG_NAMES
    cfg = parse_args(["-f", "x.csv", "-la", "y"])
    assign_empty_args(cfg)
    assert cfg.ef == "npz"


def test_ef_parquet_parses() -> None:
    cfg = parse_args(["-f", "x.csv", "-la", "y", "-ef", "parquet"])
    assign_empty_args(cfg)
    check_args(cfg)
    assert cfg.ef.lower() == "parquet"


def test_ef_rejects_pickle_name() -> None:
    cfg = parse_args(["-f", "x.csv", "-la", "y", "-ef", "pkl"])
    assign_empty_args(cfg)
    with pytest.raises(SystemExit, match=r"\[-ef\]"):
        check_args(cfg)


def test_npz_roundtrip_preserves_values(tmp_path: Path) -> None:
    store = _sample_store()
    path = write_npz_store(tmp_path / "station.npz", store)
    loaded = read_npz_store(path)
    assert set(loaded) == set(store)
    assert loaded[4]["STATION"] == "KMQE"
    assert float(loaded[4]["LAT"]) == pytest.approx(42.2)
    orig = pd.to_numeric(store[5]["PRECIP"], errors="coerce")
    new = pd.to_numeric(loaded[5]["PRECIP"], errors="coerce")
    assert new.sum(skipna=True) == pytest.approx(orig.sum(skipna=True))
    assert list(loaded[10]["YYYYMMDD"].astype(str)) == ["20000101", "20000102"]
    assert len(loaded[8]) == 2
    assert len(loaded[8][0]) == 1
    assert len(loaded[2]) == 0
    dt = pd.to_datetime(loaded[5]["DT_1"])
    assert dt.iloc[0] == pd.Timestamp("2000-01-02 11:20:00")


def test_export_data_ed2_writes_npz(tmp_path: Path) -> None:
    from Source.config import set_config

    cfg = WeppcliffConfig(ed=2, ef="npz", e=str(tmp_path), ex_dir=str(tmp_path), fn="Sample")
    set_config(cfg)
    data = WeppcliffData(_sample_store())
    export_data(data, cfg)
    out = tmp_path / "Sample.npz"
    assert out.exists()
    loaded = load_exported_store(out)
    assert loaded[4]["STATION"] == "KMQE"


def test_legacy_pkl_still_loads(tmp_path: Path) -> None:
    import pickle

    store = _sample_store()
    pkl = tmp_path / "legacy.pkl"
    with pkl.open("wb") as handle:
        pickle.dump(store, handle)
    loaded = load_exported_store(pkl)
    assert loaded[4]["STATION"] == "KMQE"


def test_npz_roundtrip_autotest1_pickle_if_present(tmp_path: Path) -> None:
    import pickle

    pkl = ROOT / "Autotest" / "Output" / "AutoTest.1.pkl"
    if not pkl.exists():
        pytest.skip("AutoTest.1.pkl missing")
    with pkl.open("rb") as handle:
        store = pickle.load(handle)
    path = write_npz_store(tmp_path / "AutoTest.1.npz", store)
    loaded = read_npz_store(path)
    assert set(loaded) == set(store)
    orig = pd.to_numeric(store[5]["PRECIP"], errors="coerce")
    new = pd.to_numeric(loaded[5]["PRECIP"], errors="coerce")
    assert new.sum(skipna=True) == pytest.approx(float(orig.sum(skipna=True)))
    assert len(loaded[8]) == len(store[8])


def test_parquet_roundtrip_if_pyarrow(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = _sample_store()
    path = write_binary_store(tmp_path, "Sample", store, "parquet")
    assert path.name.endswith(".parquet.zip")
    loaded = load_exported_store(path)
    assert loaded[4]["STATION"] == "KMQE"
    orig = pd.to_numeric(store[5]["PRECIP"], errors="coerce")
    new = pd.to_numeric(loaded[5]["PRECIP"], errors="coerce")
    assert new.sum(skipna=True) == pytest.approx(orig.sum(skipna=True))
    assert len(loaded[8]) == 2
