"""L1 Köppen–Geiger prior store (v2 schema) with multivariate sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from Source.phigi.l1_koppen import resolve_koppen_zone
from Source.phigi.l3_tau import meteorological_season

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PRIORS = _REPO_ROOT / "priors" / "koppen_priors_v2.json"


@dataclass(frozen=True)
class VarPrior:
    mean: float
    std: float


class KoppenPriorStore:
    """Zone × season priors with optional correlation matrices (PHIGI v2)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.version = int(data.get("version", 2))
        self.variables: tuple[str, ...] = tuple(data.get("variables", ("TAU",)))

    @classmethod
    def load(cls, path: str | Path | None = None) -> KoppenPriorStore:
        p = Path(path) if path is not None else _DEFAULT_PRIORS
        with p.open(encoding="utf-8") as fh:
            return cls(json.load(fh))

    @property
    def zones(self) -> list[str]:
        return sorted(self._data.get("zones", {}).keys())

    def resolve_zone(
        self,
        kg: str,
        lat: float,
        lon: float = 0.0,
        *,
        grid_path: str | Path | None = None,
    ) -> str:
        zone = resolve_koppen_zone(lat, lon, kg=kg, grid_path=grid_path)
        if zone in self._data.get("zones", {}):
            return zone
        # Fallback to nearest zone in DB by latitude band
        if abs(lat) < 23.5 and "Am" in self.zones:
            return "Am"
        if abs(lat) >= 50.0 and "Dfb" in self.zones:
            return "Dfb"
        return "Cfa" if "Cfa" in self.zones else self.zones[0]

    def _season_entry(self, zone: str, season: int) -> dict[str, Any] | None:
        seasons = self._data.get("zones", {}).get(zone, {}).get("seasons", {})
        return seasons.get(str(int(season)))

    def variable_prior(self, zone: str, season: int, variable: str) -> VarPrior | None:
        entry = self._season_entry(zone, season)
        if not entry:
            return None
        means = entry.get("means", {})
        stds = entry.get("stds", {})
        if variable not in means:
            return None
        return VarPrior(
            mean=float(means[variable]),
            std=float(stds.get(variable, 0.1)),
        )

    def tau_prior(self, zone: str, season: int) -> VarPrior | None:
        return self.variable_prior(zone, season, "TAU")

    def correlation_matrix(self, zone: str, season: int) -> np.ndarray | None:
        entry = self._season_entry(zone, season)
        if not entry:
            return None
        corr = entry.get("correlation")
        if corr is None:
            return None
        return np.asarray(corr, dtype=np.float64)

    def sample_multivariate(
        self,
        zone: str,
        season: int,
        *,
        variables: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """Draw one correlated vector from zone × season prior."""
        gen = rng or np.random.default_rng()
        vars_ = list(variables or self.variables)
        entry = self._season_entry(zone, season)
        if not entry:
            return {}

        means = np.array([float(entry["means"][v]) for v in vars_], dtype=np.float64)
        stds = np.array([float(entry["stds"].get(v, 0.1)) for v in vars_], dtype=np.float64)
        corr = self.correlation_matrix(zone, season)
        if corr is None or corr.shape != (len(vars_), len(vars_)):
            draws = gen.normal(means, np.maximum(stds, 1e-6))
        else:
            cov = np.outer(stds, stds) * corr
            cov = _nearest_psd(cov)
            draws = gen.multivariate_normal(means, cov)

        out: dict[str, float] = {}
        for name, val in zip(vars_, draws):
            if name == "TAU":
                out[name] = float(np.clip(val, 0.02, 1.15))
            else:
                out[name] = float(val)
        if "MIN_TEMP" in out and "MAX_TEMP" in out:
            lo = min(out["MIN_TEMP"], out["MAX_TEMP"])
            hi = max(out["MIN_TEMP"], out["MAX_TEMP"])
            out["MIN_TEMP"], out["MAX_TEMP"] = lo, hi
        return out

    def tau_prior_series(
        self,
        yyyymmdd: list[str] | np.ndarray,
        *,
        zone: str,
        lat_deg: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Per-day τ prior mean for a date list (no local borrowing)."""
        dates = np.asarray(yyyymmdd, dtype=str)
        months = np.array([int(d[4:6]) for d in dates], dtype=int)
        seasons = np.array([meteorological_season(m, lat_deg) for m in months], dtype=int)
        out = np.empty(len(dates), dtype=np.float64)
        gen = rng or np.random.default_rng()
        for i, season in enumerate(seasons):
            prior = self.tau_prior(zone, int(season))
            if prior is None:
                out[i] = 0.5
                continue
            out[i] = float(np.clip(gen.normal(prior.mean, prior.std * 0.25), 0.02, 1.15))
        return out


def _nearest_psd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semi-definite form."""
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


@lru_cache(maxsize=1)
def load_default_priors() -> KoppenPriorStore:
    return KoppenPriorStore.load(_DEFAULT_PRIORS)