"""General-purpose utilities — port of WEPPCLIFF.R vec/df helpers and dev tools."""

from __future__ import annotations

import sys
from typing import Any, NoReturn, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, np.ndarray, list[Any]]


def groupby_mean(
    frame: pd.DataFrame,
    grouper: Any,
    *,
    skipna: bool = True,
) -> pd.DataFrame:
    """Group-wise column means with explicit ``skipna`` (pandas-version safe)."""
    return frame.groupby(grouper).agg(lambda s: s.mean(skipna=skipna)).reset_index()


def vec_find_replace(
    vec: ArrayLike,
    find: bool = False,
    inf: bool = False,
    null: bool = False,
    nan: bool = False,
    na: bool = False,
    vals: Any = -99999,
    replace: Any = np.nan,
) -> ArrayLike:
    """
    Find and replace values in a vector.

    Mirrors ``vec_find_replace`` in WEPPCLIFF.R (lines 471-487).
    """
    if isinstance(vec, pd.Series):
        result = vec.copy()
        arr = result.to_numpy(copy=True)
    else:
        arr = np.asarray(vec, dtype=object).copy()

    replace_val = np.nan if replace is None or (isinstance(replace, float) and np.isnan(replace)) else replace

    if null:
        arr[arr == None] = replace_val  # noqa: E711

    if find:
        if not isinstance(vals, (list, tuple, np.ndarray)):
            match_vals = [vals]
        else:
            match_vals = list(vals)
        for val in match_vals:
            arr[arr == val] = replace_val

    numeric = pd.to_numeric(arr, errors="coerce")
    if inf:
        arr[np.isinf(numeric)] = replace_val
    if nan:
        arr[np.isnan(numeric)] = replace_val
    if na:
        arr[pd.isna(arr)] = replace_val

    if isinstance(vec, pd.Series):
        result[:] = arr
        return result
    return arr


def _resolve_cols(df: pd.DataFrame, cols: Union[int, slice, list[int], range]) -> list:
    if isinstance(cols, int):
        return [cols]
    if isinstance(cols, slice):
        start = (cols.start or 1) - 1
        stop = cols.stop
        if stop is None:
            stop = len(df.columns)
        return list(range(start + 1, stop + 1))
    return list(cols)


def replace_sentinels_columns(
    df: pd.DataFrame,
    cols: Union[int, slice, list[int], range],
    vals: Any = -99999,
    replace: Any = np.nan,
) -> pd.DataFrame:
    """In-place sentinel/null/inf replacement on numeric columns (no full-frame copy)."""
    col_list = _resolve_cols(df, cols)
    replace_val = (
        np.nan
        if replace is None or (isinstance(replace, float) and np.isnan(replace))
        else replace
    )
    match_vals = [vals] if not isinstance(vals, (list, tuple, np.ndarray)) else list(vals)

    for col in col_list:
        col_label = df.columns[col - 1] if isinstance(col, int) else col
        series = df[col_label]
        arr = series.to_numpy(copy=False)
        if arr.dtype == object:
            for val in match_vals:
                arr[arr == val] = replace_val
            arr[arr == None] = replace_val  # noqa: E711
            numeric = pd.to_numeric(arr, errors="coerce")
            arr[np.isinf(numeric)] = replace_val
            arr[np.isnan(numeric)] = replace_val
            arr[pd.isna(arr)] = replace_val
            df[col_label] = arr
        else:
            mask = np.zeros(len(arr), dtype=bool)
            for val in match_vals:
                mask |= arr == val
            mask |= np.isinf(arr)
            mask |= np.isnan(arr)
            if mask.any():
                arr = arr.copy()
                arr[mask] = replace_val
                df[col_label] = arr
    return df


def df_find_replace(
    df: pd.DataFrame,
    cols: Union[int, slice, list[int], range],
    find: bool = False,
    inf: bool = False,
    null: bool = False,
    nan: bool = False,
    na: bool = False,
    vals: Any = -99999,
    replace: Any = np.nan,
) -> pd.DataFrame:
    """
    Find and replace values in selected dataframe columns.

    Mirrors ``df_find_replace`` in WEPPCLIFF.R (lines 465-467).
    """
    result = df.copy()
    col_list = _resolve_cols(result, cols)

    for col in col_list:
        col_label = result.columns[col - 1] if isinstance(col, int) else col
        result[col_label] = vec_find_replace(
            result[col_label],
            find=find,
            inf=inf,
            null=null,
            nan=nan,
            na=na,
            vals=vals,
            replace=replace,
        )
    return result


def rle(values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Run-length encoding matching R ``rle()``."""
    arr = np.asarray(values)
    if len(arr) == 0:
        return np.array([], dtype=int), np.array([])
    change = np.concatenate([[True], arr[1:] != arr[:-1]])
    lengths = np.diff(np.concatenate([np.where(change)[0], [len(arr)]]))
    values_out = arr[np.where(change)[0]]
    return lengths, values_out


def convert_df_type(
    df: pd.DataFrame,
    cols: Union[int, slice, list[int], range],
    type_code: str,
) -> pd.DataFrame:
    """
    Change dataframe column types.

    ``type_code`` values:
    - ``c``: character / string
    - ``n``: numeric
    - ``ftn``: factor-to-numeric (character then numeric), used in ``convert_data``
    """
    result = df.copy()
    col_list = _resolve_cols(result, cols)

    for col in col_list:
        col_label = result.columns[col - 1] if isinstance(col, int) else col
        if type_code in ("c", "ftn"):
            result[col_label] = result[col_label].astype(str)
        if type_code in ("n", "ftn"):
            result[col_label] = pd.to_numeric(result[col_label], errors="coerce")
    return result


def view_data(data: Any) -> None:
    """
    Print ``data`` object contents (development utility).

    Mirrors ``view_data`` in WEPPCLIFF.R (lines 2553-2558).
    """
    if hasattr(data, "store"):
        indices = sorted(data.store)
        items = [(i, data[i]) for i in indices]
    elif isinstance(data, dict):
        items = sorted(data.items(), key=lambda kv: kv[0])
    else:
        items = list(enumerate(data, start=1))

    for index, value in items:
        print(type(value))
        if isinstance(value, pd.DataFrame):
            print(value.head(5))


def stop_quietly() -> NoReturn:
    """Stop program execution quietly (first-run / installation exit)."""
    sys.exit(0)