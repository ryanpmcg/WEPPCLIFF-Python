"""L2 local weather-type Gaussian mixtures (station-first, no Köppen required)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from Source.phigi.l0_astronomy import clear_sky_series_langleys
from Source.phigi.l3_tau import compute_tau, meteorological_season

FEATURES: tuple[str, ...] = ("TAU", "MIN_TEMP", "MAX_TEMP", "W_VEL", "DP_TEMP")
DEFAULT_MIN_N = 30
DEFAULT_MAX_COMPONENTS = 4


@dataclass
class SeasonMixture:
    """Fitted mixture for one meteorological season."""

    season: int
    n_components: int
    n_fit: int
    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    means_scaled: NDArray[np.float64]
    covariances_scaled: NDArray[np.float64]
    feature_names: tuple[str, ...]
    scaler_mean: NDArray[np.float64]
    scaler_scale: NDArray[np.float64]
    bic: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "n_components": self.n_components,
            "n_fit": self.n_fit,
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "features": list(self.feature_names),
            "bic": self.bic,
        }

    def predict_component(self, row: NDArray[np.float64]) -> int:
        """MAP component for one raw feature row."""
        if not np.isfinite(row).all():
            return -1
        scaled = (row - self.scaler_mean) / np.maximum(self.scaler_scale, 1e-8)
        gm = GaussianMixture(n_components=self.n_components, covariance_type="full")
        gm.weights_ = self.weights
        gm.means_ = self.means_scaled
        gm.covariances_ = self.covariances_scaled
        gm.precisions_cholesky_ = np.stack(
            [np.linalg.cholesky(np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))) for cov in gm.covariances_]
        )
        return int(gm.predict(scaled.reshape(1, -1))[0])


@dataclass
class WeatherTypeModel:
    """Per-season mixture models fit from local observed days."""

    seasons: dict[int, SeasonMixture] = field(default_factory=dict)
    min_n: int = DEFAULT_MIN_N
    max_components: int = DEFAULT_MAX_COMPONENTS

    @property
    def n_seasons_fitted(self) -> int:
        return len(self.seasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_n": self.min_n,
            "max_components": self.max_components,
            "n_seasons_fitted": self.n_seasons_fitted,
            "seasons": {str(k): v.to_dict() for k, v in sorted(self.seasons.items())},
        }


def _feature_matrix(
    panel: pd.DataFrame,
    dates: NDArray[np.str_] | pd.Series,
    *,
    lat_deg: float,
    elev_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Build τ + weather feature matrix; mask True where all features are usable."""
    df = panel.copy()
    ymd = np.asarray(dates.astype(str), dtype=str)
    r_cs = clear_sky_series_langleys(ymd.tolist(), lat_deg=lat_deg, elev_m=elev_m)
    so = pd.to_numeric(df.get("SO_RAD"), errors="coerce").to_numpy(dtype=np.float64)
    tau = compute_tau(so, r_cs)

    cols: dict[str, NDArray[np.float64]] = {"TAU": tau}
    for name in FEATURES:
        if name == "TAU":
            continue
        cols[name] = pd.to_numeric(df.get(name), errors="coerce").to_numpy(dtype=np.float64)

    mat = np.column_stack([cols[n] for n in FEATURES])
    mask = np.isfinite(mat).all(axis=1)
    return mat, mask


def _select_component_count(
    scaled: NDArray[np.float64],
    *,
    max_components: int,
    min_n: int,
    rng: np.random.Generator,
) -> tuple[int, float | None]:
    """Choose mixture order via BIC on standardized features."""
    n = scaled.shape[0]
    upper = min(max_components, max(1, n // max(min_n, 10)))
    if upper <= 1 or n < min_n:
        return 1, None

    best_k = 1
    best_bic = np.inf
    for k in range(1, upper + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=int(rng.integers(0, 2**31 - 1)),
            max_iter=200,
            n_init=3,
        )
        gm.fit(scaled)
        bic = gm.bic(scaled)
        if bic < best_bic:
            best_bic = bic
            best_k = k
    return best_k, float(best_bic)


def fit_season_mixture(
    features: NDArray[np.float64],
    *,
    season: int,
    min_n: int = DEFAULT_MIN_N,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    rng: np.random.Generator | None = None,
) -> SeasonMixture | None:
    """Fit one season's mixture; return None when n < min_n."""
    gen = rng or np.random.default_rng()
    clean = features[np.isfinite(features).all(axis=1)]
    if len(clean) < min_n:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)
    n_comp, bic = _select_component_count(
        scaled,
        max_components=max_components,
        min_n=min_n,
        rng=gen,
    )
    gm = GaussianMixture(
        n_components=n_comp,
        covariance_type="full",
        random_state=int(gen.integers(0, 2**31 - 1)),
        max_iter=200,
        n_init=5,
    )
    gm.fit(scaled)
    means = scaler.inverse_transform(gm.means_)
    return SeasonMixture(
        season=int(season),
        n_components=int(n_comp),
        n_fit=int(len(clean)),
        weights=np.asarray(gm.weights_, dtype=np.float64),
        means=np.asarray(means, dtype=np.float64),
        means_scaled=np.asarray(gm.means_, dtype=np.float64),
        covariances_scaled=np.asarray(gm.covariances_, dtype=np.float64),
        feature_names=FEATURES,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        bic=bic,
    )


def fit_local_weather_types(
    panel: pd.DataFrame,
    dates: pd.Series | NDArray[np.str_],
    *,
    lat_deg: float,
    elev_m: float = 0.0,
    min_n: int = DEFAULT_MIN_N,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    rng: np.random.Generator | None = None,
) -> WeatherTypeModel:
    """
    Fit 1–4 component mixtures per meteorological season from local observations.

    Does not use Köppen zones; spatial pooling (optional L1) is a later fallback.
    """
    gen = rng or np.random.default_rng()
    ymd = np.asarray(dates.astype(str), dtype=str)
    mat, mask = _feature_matrix(panel, ymd, lat_deg=lat_deg, elev_m=elev_m)
    months = np.array([int(d[4:6]) for d in ymd], dtype=int)
    seasons = np.array([meteorological_season(m, lat_deg) for m in months], dtype=int)

    model = WeatherTypeModel(min_n=min_n, max_components=max_components)
    for season in (1, 2, 3, 4):
        loc = mask & (seasons == season)
        if not loc.any():
            continue
        fit = fit_season_mixture(
            mat[loc],
            season=season,
            min_n=min_n,
            max_components=max_components,
            rng=gen,
        )
        if fit is not None:
            model.seasons[season] = fit
    return model


def assign_weather_types(
    panel: pd.DataFrame,
    dates: pd.Series | NDArray[np.str_],
    model: WeatherTypeModel,
    *,
    lat_deg: float,
    elev_m: float = 0.0,
) -> NDArray[np.int_]:
    """
    Assign each day a weather-type component id (0-based) or -1 when unassigned.

    Days with missing features receive the seasonal mixture's highest-weight component.
    """
    ymd = np.asarray(dates.astype(str), dtype=str)
    mat, _ = _feature_matrix(panel, ymd, lat_deg=lat_deg, elev_m=elev_m)
    months = np.array([int(d[4:6]) for d in ymd], dtype=int)
    seasons = np.array([meteorological_season(m, lat_deg) for m in months], dtype=int)
    out = np.full(len(ymd), -1, dtype=np.int_)

    for i, season in enumerate(seasons):
        mixture = model.seasons.get(int(season))
        if mixture is None:
            out[i] = 0
            continue
        cid = mixture.predict_component(mat[i])
        if cid < 0:
            cid = int(np.argmax(mixture.weights))
        out[i] = cid
    return out


def weather_type_cluster_labels(
    dates: pd.Series | NDArray[np.str_],
    components: NDArray[np.int_],
    *,
    lat_deg: float,
) -> NDArray[np.int_]:
    """Encode season × component as a single CLUSTER-like integer (season * 10 + component)."""
    ymd = np.asarray(dates.astype(str), dtype=str)
    months = np.array([int(d[4:6]) for d in ymd], dtype=int)
    seasons = np.array([meteorological_season(m, lat_deg) for m in months], dtype=int)
    comp = np.asarray(components, dtype=np.int_)
    out = seasons * 10 + np.maximum(comp, 0)
    out[comp < 0] = seasons[comp < 0]
    return out.astype(np.int_)