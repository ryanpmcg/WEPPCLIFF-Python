"""Imputation diagnostics for PHIGI gap filling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImputationReport:
    """Per-run summary of PHIGI methods, tiers, and variable-level actions."""

    engine: str = "phigi"
    gf: str = "phigi"
    layers_used: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    confidence_tier: str = "high"
    n_sorad_imputed: int = 0
    n_sorad_invalidated: int = 0
    n_tau_imputed: int = 0
    n_temp_imputed: int = 0
    n_moisture_wind_imputed: int = 0
    n_event_imputed: int = 0
    n_wet_event_imputed: int = 0
    n_dry_days_enforced: int = 0
    n_subdaily_days_imputed: int = 0
    n_subdaily_cells_imputed: int = 0
    tau_qc_lo: float = 0.0
    tau_qc_hi: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    elev_m: float = 0.0
    koppen_zone: str = ""
    weather_type_seasons_fitted: int = 0
    weather_type_components: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def mark_low_confidence(self, reason: str) -> None:
        """L6: flag degraded imputation (no silent mean-fill without tier)."""
        self.confidence_tier = "low"
        if reason not in self.notes:
            self.notes.append(reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "gf": self.gf,
            "layers_used": list(self.layers_used),
            "strategies": list(self.strategies),
            "confidence_tier": self.confidence_tier,
            "n_sorad_imputed": self.n_sorad_imputed,
            "n_sorad_invalidated": self.n_sorad_invalidated,
            "n_tau_imputed": self.n_tau_imputed,
            "n_temp_imputed": self.n_temp_imputed,
            "n_moisture_wind_imputed": self.n_moisture_wind_imputed,
            "n_event_imputed": self.n_event_imputed,
            "n_wet_event_imputed": self.n_wet_event_imputed,
            "n_dry_days_enforced": self.n_dry_days_enforced,
            "n_subdaily_days_imputed": self.n_subdaily_days_imputed,
            "n_subdaily_cells_imputed": self.n_subdaily_cells_imputed,
            "tau_qc_lo": self.tau_qc_lo,
            "tau_qc_hi": self.tau_qc_hi,
            "lat": self.lat,
            "lon": self.lon,
            "elev_m": self.elev_m,
            "koppen_zone": self.koppen_zone,
            "weather_type_seasons_fitted": self.weather_type_seasons_fitted,
            "weather_type_components": dict(self.weather_type_components),
            "notes": list(self.notes),
        }