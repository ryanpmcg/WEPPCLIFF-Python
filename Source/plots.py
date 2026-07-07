"""Matplotlib plotting — dirs: Plots/Input, Plots/Output, Plots/Other."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Source.config import LR, get_config
from Source.data_model import WeppcliffData
from Source.paths import create_directory

# Display labels for data columns / variables (title case).
_VARIABLE_LABELS: dict[str, str] = {
    "PRECIP": "Precipitation",
    "BPS": "Breakpoints",
    "ELEN": "Storms",
    "AIR_TEMP": "Air Temperature",
    "REL_HUM": "Relative Humidity",
    "MIN_TEMP": "Minimum Temperature",
    "MAX_TEMP": "Maximum Temperature",
    "SO_RAD": "Solar Radiation",
    "W_VEL": "Wind Velocity",
    "W_DIR": "Wind Direction",
    "DP_TEMP": "Dew Point Temperature",
    "INT": "Intensity",
}

_PLOT_WIDTH_IN = 21.6
_TS_FIGSIZE = (_PLOT_WIDTH_IN, _PLOT_WIDTH_IN / 2)  # 2:1 width:height
_HIST_FIGSIZE = (_PLOT_WIDTH_IN, _PLOT_WIDTH_IN / 2)  # 2:1 width:height
_PLOT_DPI = 100

_STORM_LABELS: dict[str, str] = {
    "DT": "Datetime",
    "DEPTH": "Depth",
    "DUR": "Duration",
    "PDUR": "Peak Duration",
    "PRATIO": "Peak Ratio",
    "PEAKINT": "Peak Intensity",
    "PEAKTIME": "Peak Time",
    "TIME_WM_INT": "Time Weighted Mean Intensity",
    "DEPTH_WM_INT": "Depth Weighted Mean Intensity",
    "I30": "I30",
    "ANT": "Antecedent Time",
}


def _plot_path(*parts: str) -> Path:
    return Path(*parts)


def _path_label(label: str) -> str:
    """Normalize a label for plot paths and filenames (spaces → hyphens)."""
    return label.replace(" ", "-")


def _plot_filename(*parts: str) -> str:
    """Build a plot PNG basename: hyphens only, no spaces."""
    return "-".join(_path_label(p) for p in parts if p) + ".png"


def _title_label(name: str) -> str:
    """Convert a column or variable token to a human-readable title-case label."""
    if name in _VARIABLE_LABELS:
        return _VARIABLE_LABELS[name]
    if name in _STORM_LABELS:
        return _STORM_LABELS[name]
    if "_" in name:
        head, tail = name.split("_", 1)
        if head.upper() in {"R2", "AH537", "AH703", "USER"}:
            tail_labels = {"KE": "Kinetic Energy", "ED": "Erosion Density", "EI": "Erosion Index"}
            tail_label = tail_labels.get(tail.upper(), _title_label(tail))
            return f"{head} {tail_label}"
    return name.replace("_", " ").title()


def _save_figure(fig, path: Path) -> None:
    create_directory(path.parent)
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def _font_sizes(figsize: tuple[float, float]) -> dict[str, float]:
    """Scale text for WEPPCLIFF's large export figures (~12 pt at 10\" width)."""
    width_in, _ = figsize
    base = max(10.0, 12.0 * (width_in / 10.0))
    return {
        "title": base * 1.35,
        "label": base * 1.1,
        "tick": base * 0.95,
        "legend": base * 0.9,
    }


def _new_figure(figsize: tuple[float, float]):
    fonts = _font_sizes(figsize)
    fig, ax = plt.subplots(figsize=figsize, dpi=_PLOT_DPI)
    return fig, ax, fonts


def _style_axes(
    ax,
    fonts: dict[str, float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xrotation: int = 0,
) -> None:
    ax.set_title(title, fontsize=fonts["title"])
    ax.set_xlabel(xlabel, fontsize=fonts["label"])
    ax.set_ylabel(ylabel, fontsize=fonts["label"])
    ax.tick_params(axis="both", labelsize=fonts["tick"])
    if xrotation:
        for tick in ax.get_xticklabels():
            tick.set_rotation(xrotation)


def _valid_datetime_mask(dt: pd.Series | np.ndarray) -> np.ndarray:
    dt_parsed = pd.to_datetime(dt, errors="coerce")
    return ~pd.isna(dt_parsed)


def _finite_mask(dt: pd.Series | np.ndarray, values: np.ndarray) -> np.ndarray:
    dt_parsed = pd.to_datetime(dt, errors="coerce")
    return ~(np.isnan(values) | pd.isna(dt_parsed))


def _observation_interval_stats_minutes(dt) -> dict[str, float] | None:
    """Min/median/mean/max spacing between consecutive observation times (minutes)."""
    dt_s = pd.to_datetime(dt, errors="coerce").dropna().sort_values()
    if len(dt_s) < 2:
        return None
    deltas = dt_s.to_series().diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return None
    return {
        "min": float(deltas.min()),
        "median": float(deltas.median()),
        "mean": float(deltas.mean()),
        "max": float(deltas.max()),
    }


def _format_interval_minutes(minutes: float) -> str:
    if minutes >= 1440:
        return f"{minutes / 1440:.3g} d"
    if minutes >= 60:
        return f"{minutes / 60:.3g} h"
    return f"{minutes:.3g} min"


def _interval_legend_entries(stats: dict[str, float]) -> list[tuple[Line2D, str]]:
    entries: list[tuple[Line2D, str]] = []
    for key in ("min", "median", "mean", "max"):
        label = f"Interval {key}: {_format_interval_minutes(stats[key])}"
        entries.append((Line2D([], [], linestyle="none"), label))
    return entries


def _plot_timeseries(
    ax,
    dt,
    values,
    style: str,
    *,
    color: str,
    label: str | None = None,
    linewidth_scale: float = 1.0,
) -> None:
    """Plot a time series; ``style`` ``h`` = bars, ``l`` = line."""
    dt_parsed = pd.to_datetime(dt, errors="coerce")
    vec = np.asarray(values, dtype=float)

    if style == "h":
        keep = _finite_mask(dt_parsed, vec)
        dt_parsed = dt_parsed[keep]
        vec = vec[keep]
        if len(vec) == 0:
            return
        x = mdates.date2num(dt_parsed)
        if len(x) > 1:
            width = float(np.median(np.diff(x))) * 0.8
        else:
            width = 1.0
        ax.bar(x, vec, width=width, color=color, align="center", label=label)
        ax.xaxis_date()
    else:
        keep = _valid_datetime_mask(dt_parsed)
        dt_parsed = dt_parsed[keep]
        vec = vec[keep]
        if len(dt_parsed) == 0:
            return
        lw = max(0.5, linewidth_scale * (ax.figure.get_figwidth() / 10.0))
        # NaN values break the line so gaps in missing data are not bridged.
        ax.plot(dt_parsed, vec, color=color, linewidth=lw, label=label)


def create_input_plots(var_nm, pty_nm, var_u, var_vec, dt_vec, var_qc, dt_qc, stat_nm, plot_type):
    config = get_config()
    outdir = _plot_path(config.p, "Input", _path_label(pty_nm))
    create_directory(outdir)

    vec = np.asarray(var_vec, dtype=float)
    dt = pd.to_datetime(dt_vec, errors="coerce")

    vqc = dqc = None
    if config.do_qc and var_qc is not None and len(var_qc):
        vqc = np.asarray(var_qc, dtype=float)
        dqc = pd.to_datetime(dt_qc, errors="coerce")

    if np.nansum(np.isfinite(vec)) > 0:
        fig, ax, fonts = _new_figure(_TS_FIGSIZE)
        _plot_timeseries(ax, dt, vec, plot_type, color="black", label="Original")
        if vqc is not None and config.do_qc and np.isfinite(vqc).any():
            _plot_timeseries(ax, dqc, vqc, plot_type, color="red", label="Quality-Checked")
        _style_axes(
            ax,
            fonts,
            title=f"Input {pty_nm} Time Series",
            xlabel="Datetime",
            ylabel=f"{pty_nm} ({var_u})",
        )
        handles, labels = ax.get_legend_handles_labels()
        if var_nm == "PRECIP":
            stats = _observation_interval_stats_minutes(dt)
            if stats:
                for handle, label in _interval_legend_entries(stats):
                    handles.append(handle)
                    labels.append(label)
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=fonts["legend"])
        _save_figure(fig, outdir / _plot_filename(stat_nm, pty_nm, "Input", "Time Series"))

    hist_vec = vec[_finite_mask(dt, vec)]
    hist_vqc = None
    if vqc is not None and dqc is not None:
        hist_vqc = vqc[_finite_mask(dqc, vqc)]

    if len(hist_vec) > 0 and int(config.sm) == 1:
        ul, ll = np.nanmax(hist_vec), np.nanmin(hist_vec)
        bins = np.linspace(ll, ul, 101)
        fig, ax, fonts = _new_figure(_HIST_FIGSIZE)
        if hist_vqc is not None and config.do_qc and len(hist_vqc):
            ax.hist(hist_vqc, bins=bins, density=True, color="red", alpha=0.5, label="Quality-Checked")
            ax.hist(hist_vec, bins=bins, density=True, histtype="bar", fill=False, edgecolor="black", label="Original")
        else:
            ax.hist(hist_vec, bins=bins, density=True, histtype="bar", fill=False, edgecolor="black", label="Original")
        _style_axes(
            ax,
            fonts,
            title=f"Input {pty_nm} Histogram",
            xlabel=f"{pty_nm} ({var_u})",
            ylabel="Density",
        )
        ax.legend(loc="upper right", fontsize=fonts["legend"])
        _save_figure(fig, outdir / _plot_filename(stat_nm, pty_nm, "Input", "Histogram"))


def _period_bar_plot(
    data: WeppcliffData,
    og_df: int,
    gf_df: int,
    var_nm: str,
    pty_nm: str,
    var_u: str,
    stat_nm: str,
    *,
    period: str,
    dt_col: str,
    title_prefix: str,
    xlabel: str,
    rotate_labels: bool,
    filename_suffix: str,
) -> None:
    config = get_config()
    outdir = _plot_path(config.p, "Output", period)
    create_directory(outdir)

    var_vec = data.slot_df(og_df)[var_nm].values
    dt_vec = data.slot_df(og_df)[dt_col].astype(str).values
    var_gf = None
    if config.do_id:
        var_gf = data.slot_df(gf_df)[var_nm].values

    if np.nansum(~np.isnan(var_vec.astype(float))) <= 0:
        return

    fig, ax, fonts = _new_figure(_TS_FIGSIZE)
    x = np.arange(len(dt_vec))
    if var_gf is not None:
        ax.bar(x, var_gf, color="red", label="Gap-Filled")
        ax.bar(x, var_vec, fill=False, edgecolor="black", label="Original")
    else:
        ax.bar(x, var_vec, fill=False, edgecolor="black", label="Original")
    ax.set_xticks(x)
    ax.set_xticklabels(dt_vec, rotation=90 if rotate_labels else 0)
    _style_axes(
        ax,
        fonts,
        title=f"{title_prefix} {pty_nm} Time Series",
        xlabel=xlabel,
        ylabel=f"{pty_nm} ({var_u})",
        xrotation=90 if rotate_labels else 0,
    )
    ax.legend(loc="upper right", fontsize=fonts["legend"])
    _save_figure(fig, outdir / _plot_filename(stat_nm, pty_nm, filename_suffix))


def annual_plots(data, og_df, gf_df, var_nm, pty_nm, var_u, stat_nm):
    _period_bar_plot(
        data, og_df, gf_df, var_nm, pty_nm, var_u, stat_nm,
        period="Annual", dt_col="YYYY", title_prefix="Annual",
        xlabel="Year", rotate_labels=True, filename_suffix="Annual Time Series",
    )


def monthly_plots(data, og_df, gf_df, var_nm, pty_nm, var_u, stat_nm):
    _period_bar_plot(
        data, og_df, gf_df, var_nm, pty_nm, var_u, stat_nm,
        period="Monthly", dt_col="MM", title_prefix="Monthly Mean",
        xlabel="Month", rotate_labels=False, filename_suffix="Monthly Mean Time Series",
    )


def daily_plots(data, og_df, gf_df, var_nm, pty_nm, var_u, stat_nm, plot_type):
    config = get_config()
    outdir = _plot_path(config.p, "Output", "Daily")
    create_directory(outdir)

    var_vec = data.slot_df(og_df)[var_nm].values
    dt_vec = pd.to_datetime(data.slot_df(og_df)["YYYYMMDD"].astype(str), format="%Y%m%d")
    var_gf = dt_gf = None
    if config.do_id:
        var_gf = data.slot_df(gf_df)[var_nm].values.copy()
        dt_gf = pd.to_datetime(data.slot_df(gf_df)["YYYYMMDD"].astype(str), format="%Y%m%d")
        dups = ~np.isnan(var_vec)
        var_gf[dups] = np.nan
    if np.nansum(np.isfinite(var_vec.astype(float))) > 0:
        fig, ax, fonts = _new_figure(_TS_FIGSIZE)
        _plot_timeseries(ax, dt_vec, var_vec, plot_type, color="black", label="Original")
        if var_gf is not None:
            _plot_timeseries(ax, dt_gf, var_gf, plot_type, color="red", label="Gap-Filled")
        _style_axes(
            ax,
            fonts,
            title=f"Daily {pty_nm} Time Series",
            xlabel="Date",
            ylabel=f"{pty_nm} ({var_u})",
        )
        ax.legend(loc="upper right", fontsize=fonts["legend"])
        _save_figure(fig, outdir / _plot_filename(stat_nm, pty_nm, "Daily", "Time Series"))


def storm_plots(data, og_df, gf_df, stat_nm):
    config = get_config()
    outdir = _plot_path(config.p, "Output", "Storms")
    create_directory(outdir)
    og = data.slot_df(og_df)
    col_list = list(og.columns[1:])
    eqn_count = max(0, (len(col_list) - 10) // 2)
    unit_list = (
        ["mm", "min", "min", "fraction", "mm/hr", "fraction", "mm/hr", "mm/hr", "mm/hr", "min"]
        + ["MJ/ha"] * eqn_count
        + ["MJ-mm/ha-hr"] * eqn_count
    )
    dt_vec = og["DT"]
    dup_mask = None
    gf = None
    if config.do_id:
        gf = data.slot_df(gf_df)
        dt_gf = gf["DT"]
        dup_mask = dt_gf.isin(dt_vec.dropna()).to_numpy()

    for k, col in enumerate(col_list):
        var_vec = og[col].astype(float).values
        var_gf = None
        if config.do_id and gf is not None:
            var_gf = gf[col].astype(float).values
            var_gf = var_gf[~dup_mask]
        if np.nansum(~np.isnan(var_vec)) > 0:
            ul, ll = np.nanmax(var_vec), np.nanmin(var_vec)
            bins = np.linspace(ll, ul, 101)
            fig, ax, fonts = _new_figure(_HIST_FIGSIZE)
            unit = unit_list[k] if k < len(unit_list) else ""
            label = _title_label(col)
            if var_gf is not None and len(var_gf):
                ax.hist(var_gf, bins=bins, density=True, color="red", alpha=0.5, label="Gap-Filled")
                ax.hist(var_vec, bins=bins, density=True, histtype="bar", fill=False, edgecolor="black", label="Original")
            else:
                ax.hist(var_vec, bins=bins, density=True, histtype="bar", fill=False, edgecolor="black", label="Original")
            _style_axes(
                ax,
                fonts,
                title=f"Storm {label} Histogram",
                xlabel=f"{label} ({unit})" if unit else label,
                ylabel="Density",
            )
            ax.legend(loc="upper right", fontsize=fonts["legend"])
            _save_figure(fig, outdir / _plot_filename(stat_nm, label, "Storm", "Histogram"))


def create_other_plots(data, stat_nm):
    config = get_config()
    other_dir = _plot_path(config.p, "Other")
    create_directory(other_dir)
    if int(config.sm) == 2:
        plot_df = data.slot_df(5)
        df_now = pd.DataFrame({"DT": pd.to_datetime(plot_df["DT_1"]), "INT": plot_df["INT"]})
        dt_new = df_now["DT"] - pd.to_timedelta(plot_df["DUR"], unit="m")
        keep = ~dt_new.isin(df_now["DT"])
        df_new = pd.DataFrame({"DT": dt_new[keep], "INT": 0.0})
        df_out = pd.concat([df_now, df_new]).sort_values("DT")
        if df_out["DT"].notna().sum() > 0:
            fig, ax, fonts = _new_figure(_TS_FIGSIZE)
            intensity = df_out["INT"].to_numpy(dtype=float)
            lw = max(0.5, ax.figure.get_figwidth() / 10.0)
            ax.plot(df_out["DT"], intensity, drawstyle="steps-post", linewidth=lw)
            _style_axes(
                ax,
                fonts,
                title="Input Intensity Time Series",
                xlabel="Datetime",
                ylabel="Intensity (mm/hr)",
            )
            _save_figure(fig, other_dir / _plot_filename(stat_nm, "Input", "Intensity", "Time Series"))


_OUTPUT_PLOT_REGISTRY = (
    ("Annual", annual_plots, 12, 20),
    ("Monthly", monthly_plots, 13, 21),
)


def _slot_dataframe(data: WeppcliffData, slot: int) -> pd.DataFrame | None:
    try:
        return data.slot_df(slot)
    except (KeyError, AttributeError, TypeError):
        return None


def _plot_skip_reason(data: WeppcliffData, og_slot: int, var_nm: str) -> str | None:
    """Return a human-readable skip reason, or None when plotting may proceed."""
    df = _slot_dataframe(data, og_slot)
    pretty = _title_label(var_nm)
    label = f"{pretty} ({var_nm})" if pretty != var_nm else var_nm

    if df is None:
        return f"processed data slot {og_slot} is not available"

    if var_nm not in df.columns:
        return f"{label} was not provided in the input file"

    vec = np.asarray(df[var_nm].values, dtype=float)
    if np.nansum(np.isfinite(vec)) <= 0:
        return f"{label} has no plottable values in the processed data"

    return None


def _run_plot_step(
    step,
    *args,
    step_name: str | None = None,
    skip_reason: str | None = None,
    var_nm: str | None = None,
) -> None:
    """Run one plot helper; report skips and surface unexpected errors when verbose."""
    config = get_config()
    name = step_name or step.__name__

    if skip_reason:
        if config.do_verb:
            detail = f" ({var_nm})" if var_nm else ""
            print(f"{LR}Plot skipped — {name}{detail}: {skip_reason}", end="")
        return

    try:
        step(*args)
    except Exception as exc:
        if config.do_verb:
            detail = f" ({var_nm})" if var_nm else ""
            print(f"{LR}Plot step {name}{detail} failed: {exc}", end="")


def graph_data(data: WeppcliffData) -> None:
    config = get_config()
    if config.do_verb:
        print(LR, "Plotting data...", end="")
    for sub in ("Input", "Output", "Other"):
        create_directory(_plot_path(config.p, sub))

    var_nm = ["PRECIP", "AIR_TEMP", "REL_HUM", "MIN_TEMP", "MAX_TEMP", "SO_RAD", "W_VEL", "W_DIR", "DP_TEMP"]
    pretty_nm = [_title_label(v) for v in var_nm]
    var_u = ["mm", "C", "%", "C", "C", "Langleys/Day", "m/s", "Degrees from North (CW)", "C"]
    og_l = [1, 2, 2, 3, 3, 3, 3, 3, 3]
    qc_l = [5, 6, 6, 7, 7, 7, 7, 7, 7]
    plot_type = ["h", "l", "l", "l", "l", "l", "l", "l", "l"]
    stat_nm = config.fn
    if str(stat_nm).upper() == "STATION":
        stat_nm = data.metadata.get("STATION", "station")

    for j in range(len(var_nm)):
        og_df = data.slot_df(og_l[j])
        qc_df = data.slot_df(qc_l[j])
        dt_col = og_df.columns[0]
        if var_nm[j] not in og_df.columns:
            if config.do_verb:
                reason = _plot_skip_reason(data, og_l[j], var_nm[j])
                print(
                    f"{LR}Plot skipped — create_input_plots ({var_nm[j]}): {reason}",
                    end="",
                )
            continue
        _run_plot_step(
            create_input_plots,
            var_nm[j], pretty_nm[j], var_u[j],
            og_df[var_nm[j]].values, og_df[dt_col].values,
            qc_df.get(var_nm[j], pd.Series(dtype=float)).values,
            qc_df[dt_col].values if dt_col in qc_df.columns else pd.Series(dtype=object),
            stat_nm, plot_type[j],
            step_name="create_input_plots",
            var_nm=var_nm[j],
        )

    if int(config.sm) == 1:
        out_vars = [
            "PRECIP", "BPS", "ELEN", "AIR_TEMP", "REL_HUM", "MIN_TEMP", "MAX_TEMP",
            "SO_RAD", "W_VEL", "W_DIR", "DP_TEMP",
        ]
        out_pretty = [_title_label(v) for v in out_vars]
        out_units = ["mm", "#", "#", "C", "%", "C", "C", "Langleys/Day", "m/s", "Degrees from North (CW)", "C"]
        out_plot_type = ["h", "h", "h", "l", "l", "l", "l", "l", "l", "l", "l"]

        for j in range(len(out_vars)):
            var = out_vars[j]
            for _kind, plot_fn, og_slot, gf_slot in _OUTPUT_PLOT_REGISTRY:
                _run_plot_step(
                    plot_fn,
                    data, og_slot, gf_slot, var, out_pretty[j], out_units[j], stat_nm,
                    step_name=plot_fn.__name__,
                    skip_reason=_plot_skip_reason(data, og_slot, var),
                    var_nm=var,
                )
            _run_plot_step(
                daily_plots,
                data, 10, 18, var, out_pretty[j], out_units[j], stat_nm, out_plot_type[j],
                step_name="daily_plots",
                skip_reason=_plot_skip_reason(data, 10, var),
                var_nm=var,
            )

        if config.do_ei:
            _run_plot_step(
                storm_plots, data, 9, 17, stat_nm,
                step_name="storm_plots",
            )

    _run_plot_step(create_other_plots, data, stat_nm, step_name="create_other_plots")