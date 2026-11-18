"""WEPPCLIFF primary data container — mirrors R's numbered ``data`` list."""

from __future__ import annotations

from typing import Any, Iterator, MutableMapping, Optional

import pandas as pd

# Metadata keys stored at index 4 in the R ``data`` list.
METADATA_KEYS: tuple[str, ...] = (
    "STATION",
    "LAT",
    "LON",
    "ELEV",
    "OBS_YRS",
    "B_YR",
    "YRS_SIM",
)


class WeppcliffData:
    """
    Wrapper around a 1-based indexed store matching R ``data[[n]]`` semantics.

    Index 4 holds station metadata (STATION, LAT, LON, ELEV, OBS_YRS, B_YR, YRS_SIM).
    """

    def __init__(self, store: Optional[MutableMapping[int, Any]] = None) -> None:
        self._store: dict[int, Any] = dict(store) if store is not None else {}
        self.observed_precip_ymd: Optional[set[str]] = None

    def __getitem__(self, index: int) -> Any:
        if index < 1:
            raise IndexError("WeppcliffData uses 1-based indexing like R")
        return self._store[index]

    def __setitem__(self, index: int, value: Any) -> None:
        if index < 1:
            raise IndexError("WeppcliffData uses 1-based indexing like R")
        self._store[index] = value

    def __contains__(self, index: object) -> bool:
        if not isinstance(index, int):
            return False
        return index in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._store))

    def get(self, index: int, default: Any = None) -> Any:
        """Return slot *index* or *default* if absent."""
        return self._store.get(index, default)

    def set(self, index: int, value: Any) -> None:
        """Assign *value* to slot *index*."""
        self[index] = value

    @property
    def store(self) -> dict[int, Any]:
        """Underlying 1-based index dictionary."""
        return self._store

    def keys(self) -> list[int]:
        """Sorted list of populated 1-based indices."""
        return sorted(self._store)

    def copy(self) -> "WeppcliffData":
        """
        Shallow copy of the slot dictionary.

        DataFrame slots are **not** copied — mutations visible through both handles.
        Use ``deep_copy()`` when branching pipelines need isolated frames.
        """
        out = WeppcliffData(self._store.copy())
        out.observed_precip_ymd = (
            set(self.observed_precip_ymd) if self.observed_precip_ymd else None
        )
        return out

    def deep_copy(self) -> "WeppcliffData":
        """Copy slot dict and duplicate every DataFrame slot."""
        store: dict[int, Any] = {}
        for key, value in self._store.items():
            store[key] = value.copy() if isinstance(value, pd.DataFrame) else value
        out = WeppcliffData(store)
        out.observed_precip_ymd = (
            set(self.observed_precip_ymd) if self.observed_precip_ymd else None
        )
        return out

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata dict at index 4.

        Creates an empty metadata dict at index 4 if not yet present.
        Returns a live dict reference so in-place updates persist.
        """
        if 4 not in self._store:
            self._store[4] = {key: None for key in METADATA_KEYS}
        meta = self._store[4]
        if isinstance(meta, pd.Series):
            converted = meta.to_dict()
            self._store[4] = converted
            return converted
        if isinstance(meta, dict):
            return meta
        if isinstance(meta, (list, tuple)):
            converted = dict(zip(METADATA_KEYS, meta))
            self._store[4] = converted
            return converted
        converted = {"value": meta}
        self._store[4] = converted
        return converted

    def set_metadata(self, metadata: MutableMapping[str, Any]) -> None:
        """Replace metadata at index 4."""
        self._store[4] = dict(metadata)

    def update_metadata(self, **kwargs: Any) -> None:
        """Update one or more metadata fields at index 4."""
        meta = self.get_metadata()
        meta.update(kwargs)
        self.set_metadata(meta)

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata dict at index 4 (R ``data[[4]]``)."""
        return self.get_metadata()

    @metadata.setter
    def metadata(self, value: MutableMapping[str, Any]) -> None:
        self.set_metadata(value)

    def slot_df(self, index: int) -> pd.DataFrame:
        """Return slot *index* as a DataFrame, or empty frame if absent."""
        val = self.get(index, pd.DataFrame())
        if isinstance(val, pd.DataFrame):
            return val
        raise TypeError(f"Slot {index} is not a DataFrame.")