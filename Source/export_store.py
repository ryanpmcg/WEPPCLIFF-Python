"""Binary export of the live WEPPCLIFF store (memory → .npz / parquet zip).

``-ed 1`` and ``-ed 2`` write the numbered slot dict from RAM. This module does
not pickle DataFrames. ``-ef npz`` (default) uses ``np.savez_compressed``.
``-ef parquet`` writes a zip of per-slot parquet tables (PyArrow).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Source.data_model import WeppcliffData

STORE_FORMAT_VERSION = 1
_DATETIME_COLUMNS = frozenset({"DT", "DT_1", "DT_2", "DT_3"})
_SCHEMA_KEY = "_schema"
_META_KEY = "_meta_json"


def _is_str_series(s: pd.Series) -> bool:
    dt = str(s.dtype)
    return s.dtype == object or dt in {"str", "string"} or dt.startswith("str") or "StringDtype" in dt


def _classify_series(name: str, s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if name in _DATETIME_COLUMNS and _is_str_series(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s) and int(s.isna().sum()) == 0:
        return "bool"
    if pd.api.types.is_integer_dtype(s) and int(s.isna().sum()) == 0:
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if _is_str_series(s):
        return "str"
    return "float"


def _series_to_array(name: str, s: pd.Series, role: str) -> np.ndarray:
    if role == "datetime":
        parsed = s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")
        return np.ascontiguousarray(parsed.to_numpy(dtype="datetime64[ns]"))
    if role == "bool":
        return np.ascontiguousarray(s.to_numpy(dtype=np.bool_))
    if role == "int":
        return np.ascontiguousarray(s.to_numpy())
    if role == "float":
        return np.ascontiguousarray(pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64))
    filled = s.where(s.notna(), "").astype(str)
    return np.asarray(filled.to_list(), dtype="U")


def _array_to_series(values: np.ndarray, role: str, index: pd.Index) -> pd.Series:
    if role == "datetime":
        return pd.Series(pd.to_datetime(values), index=index)
    if role == "bool":
        return pd.Series(values.astype(bool), index=index)
    if role == "int":
        return pd.Series(values, index=index)
    if role == "float":
        return pd.Series(values.astype(np.float64), index=index)
    text = pd.Series(values.astype(str), index=index)
    return text.replace("", np.nan)


def _normalize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    cols: list[dict[str, str]] = []
    out = pd.DataFrame(index=df.index)
    for name in df.columns:
        series = df[name]
        role = _classify_series(str(name), series)
        cols.append({"name": str(name), "role": role})
        arr = _series_to_array(str(name), series, role)
        out[str(name)] = _array_to_series(arr, role, df.index)
    return out, cols


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _prepare_store(store: dict[int, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    prepared: dict[int, dict[str, Any]] = {}
    schema_slots: list[dict[str, Any]] = []
    for index in sorted(store):
        value = store[index]
        if isinstance(value, pd.DataFrame):
            frame, cols = _normalize_frame(value)
            prepared[index] = {"kind": "frame", "frame": frame, "columns": cols}
            schema_slots.append({"index": int(index), "kind": "frame", "columns": cols})
        elif isinstance(value, list):
            frames = [x for x in value if isinstance(x, pd.DataFrame)]
            lengths = np.array([len(x) for x in frames], dtype=np.int32)
            if frames:
                concat = pd.concat(frames, ignore_index=True)
                frame, cols = _normalize_frame(concat)
            else:
                frame = pd.DataFrame()
                cols = []
            prepared[index] = {
                "kind": "frames",
                "frame": frame,
                "columns": cols,
                "lengths": lengths,
            }
            schema_slots.append(
                {
                    "index": int(index),
                    "kind": "frames",
                    "columns": cols,
                    "n_frames": int(len(frames)),
                }
            )
        elif isinstance(value, dict):
            prepared[index] = {"kind": "meta", "value": _json_safe(value)}
            schema_slots.append({"index": int(index), "kind": "meta"})
        else:
            raise TypeError(f"slot {index} has unsupported type {type(value).__name__}")
    schema = {"version": STORE_FORMAT_VERSION, "slots": schema_slots}
    return prepared, schema


def _frame_from_columns(
    columns: list[dict[str, str]],
    getter,
    n_rows: int | None = None,
) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    index = None
    for spec in columns:
        name = spec["name"]
        role = spec["role"]
        arr = np.asarray(getter(name))
        if index is None:
            index = pd.RangeIndex(len(arr) if n_rows is None else n_rows)
        data[name] = _array_to_series(arr, role, index)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def _split_storms(frame: pd.DataFrame, lengths: np.ndarray) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    start = 0
    for n in lengths.tolist():
        end = start + int(n)
        out.append(frame.iloc[start:end].reset_index(drop=True))
        start = end
    return out


def _rebuild_store(prepared: dict[int, dict[str, Any]]) -> dict[int, Any]:
    store: dict[int, Any] = {}
    for index, item in prepared.items():
        kind = item["kind"]
        if kind == "frame":
            store[index] = item["frame"].reset_index(drop=True)
        elif kind == "frames":
            store[index] = _split_storms(item["frame"].reset_index(drop=True), item["lengths"])
        elif kind == "meta":
            store[index] = dict(item["value"])
        else:
            raise ValueError(f"unknown slot kind {kind}")
    return store


def write_npz_store(path: Path, store: dict[int, Any]) -> Path:
    """Write compressed ``.npz`` from the live slot dict."""
    prepared, schema = _prepare_store(store)
    arrays: dict[str, np.ndarray] = {
        _SCHEMA_KEY: np.frombuffer(json.dumps(schema).encode("utf-8"), dtype=np.uint8),
    }
    for index, item in prepared.items():
        if item["kind"] == "meta":
            arrays[_META_KEY] = np.frombuffer(json.dumps(item["value"]).encode("utf-8"), dtype=np.uint8)
            continue
        frame: pd.DataFrame = item["frame"]
        for i, spec in enumerate(item["columns"]):
            name = spec["name"]
            arrays[f"s{index}__c{i}"] = _series_to_array(name, frame[name], spec["role"])
        if item["kind"] == "frames":
            arrays[f"s{index}__storm_len"] = np.ascontiguousarray(item["lengths"], dtype=np.int32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def read_npz_store(path: Path) -> dict[int, Any]:
    loaded = np.load(Path(path), allow_pickle=False)
    schema = json.loads(bytes(loaded[_SCHEMA_KEY]).decode("utf-8"))
    prepared: dict[int, dict[str, Any]] = {}
    for slot in schema["slots"]:
        index = int(slot["index"])
        kind = slot["kind"]
        if kind == "meta":
            prepared[index] = {
                "kind": "meta",
                "value": json.loads(bytes(loaded[_META_KEY]).decode("utf-8")),
            }
            continue
        columns = slot["columns"]

        def getter(name: str, *, _index: int = index, _columns: list = columns) -> np.ndarray:
            col_i = next(i for i, spec in enumerate(_columns) if spec["name"] == name)
            return loaded[f"s{_index}__c{col_i}"]

        frame = _frame_from_columns(columns, getter)
        if kind == "frame":
            prepared[index] = {"kind": "frame", "frame": frame, "columns": columns}
        elif kind == "frames":
            lengths = np.asarray(loaded[f"s{index}__storm_len"], dtype=np.int32)
            prepared[index] = {
                "kind": "frames",
                "frame": frame,
                "columns": columns,
                "lengths": lengths,
            }
        else:
            raise ValueError(f"unknown slot kind {kind}")
    return _rebuild_store(prepared)


def read_npz_slots(path: Path, indices: set[int]) -> dict[int, Any]:
    """Load selected frame slots from ``.npz`` without storm-list expansion."""
    loaded = np.load(Path(path), allow_pickle=False)
    schema = json.loads(bytes(loaded[_SCHEMA_KEY]).decode("utf-8"))
    out: dict[int, Any] = {}
    want = {int(i) for i in indices}
    for slot in schema["slots"]:
        index = int(slot["index"])
        if index not in want:
            continue
        kind = slot["kind"]
        if kind == "meta":
            out[index] = json.loads(bytes(loaded[_META_KEY]).decode("utf-8"))
            continue
        columns = slot["columns"]

        def getter(name: str, *, _index: int = index, _columns: list = columns) -> np.ndarray:
            col_i = next(i for i, spec in enumerate(_columns) if spec["name"] == name)
            return loaded[f"s{_index}__c{col_i}"]

        frame = _frame_from_columns(columns, getter)
        if kind == "frames":
            lengths = np.asarray(loaded[f"s{index}__storm_len"], dtype=np.int32)
            out[index] = _split_storms(frame.reset_index(drop=True), lengths)
        else:
            out[index] = frame.reset_index(drop=True)
    return out


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Invalid Argument: [-ef parquet] requires PyArrow. "
            "Install with: pip install 'weppcliff[parquet]' or pip install pyarrow"
        ) from exc
    return pa, pq


def write_parquet_store(path: Path, store: dict[int, Any]) -> Path:
    """Write a zip of per-slot parquet tables from the live slot dict."""
    pa, pq = _require_pyarrow()
    prepared, schema = _prepare_store(store)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("schema.json", json.dumps(schema))
        for index, item in prepared.items():
            if item["kind"] == "meta":
                zf.writestr("meta.json", json.dumps(item["value"]))
                continue
            table = pa.Table.from_pandas(item["frame"], preserve_index=False)
            raw = io.BytesIO()
            pq.write_table(table, raw, compression="zstd")
            zf.writestr(f"slot_{index:02d}.parquet", raw.getvalue())
            if item["kind"] == "frames":
                len_table = pa.table({"storm_len": item["lengths"]})
                len_raw = io.BytesIO()
                pq.write_table(len_table, len_raw, compression="zstd")
                zf.writestr(f"slot_{index:02d}_storm_len.parquet", len_raw.getvalue())
    return path


def read_parquet_store(path: Path) -> dict[int, Any]:
    _pa, pq = _require_pyarrow()
    prepared: dict[int, dict[str, Any]] = {}
    with zipfile.ZipFile(Path(path), "r") as zf:
        schema = json.loads(zf.read("schema.json").decode("utf-8"))
        for slot in schema["slots"]:
            index = int(slot["index"])
            kind = slot["kind"]
            if kind == "meta":
                prepared[index] = {
                    "kind": "meta",
                    "value": json.loads(zf.read("meta.json").decode("utf-8")),
                }
                continue
            table = pq.read_table(io.BytesIO(zf.read(f"slot_{index:02d}.parquet")))
            frame = table.to_pandas()
            if kind == "frame":
                prepared[index] = {"kind": "frame", "frame": frame, "columns": slot["columns"]}
            elif kind == "frames":
                lengths = (
                    pq.read_table(io.BytesIO(zf.read(f"slot_{index:02d}_storm_len.parquet")))
                    .column("storm_len")
                    .to_numpy()
                    .astype(np.int32)
                )
                prepared[index] = {
                    "kind": "frames",
                    "frame": frame,
                    "columns": slot["columns"],
                    "lengths": lengths,
                }
            else:
                raise ValueError(f"unknown slot kind {kind}")
    return _rebuild_store(prepared)


def resolve_binary_export_path(export_dir: Path, filename: str, fmt: str) -> Path:
    fmt = str(fmt).lower()
    if fmt == "npz":
        return Path(export_dir) / f"{filename}.npz"
    if fmt == "parquet":
        return Path(export_dir) / f"{filename}.parquet.zip"
    raise ValueError(f"unsupported export format {fmt!r}")


def write_binary_store(export_dir: Path, filename: str, store: dict[int, Any], fmt: str) -> Path:
    """Encode the live store to ``-ef npz`` or ``-ef parquet``."""
    fmt = str(fmt).lower()
    out_path = resolve_binary_export_path(export_dir, filename, fmt)
    if fmt == "npz":
        return write_npz_store(out_path, store)
    if fmt == "parquet":
        return write_parquet_store(out_path, store)
    raise ValueError(f"unsupported export format {fmt!r}")


def load_exported_store(path: Path | str) -> dict[int, Any]:
    """Load a WEPPCLIFF binary export (``.npz``, ``.parquet.zip``, or legacy ``.pkl``)."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".parquet.zip") or name.endswith(".zip"):
        return read_parquet_store(path)
    if path.suffix.lower() == ".npz":
        return read_npz_store(path)
    if path.suffix.lower() == ".pkl":
        import pickle

        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, WeppcliffData):
            return obj.store
        if isinstance(obj, dict):
            return obj
        raise TypeError(f"{path} did not contain a WEPPCLIFF store")
    raise ValueError(f"unrecognized binary export: {path}")
