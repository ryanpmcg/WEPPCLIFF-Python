"""Core workflow, gap-fill workflow, and parallel/series execution."""

from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from Source.args import assign_final_args
from Source.cli_output import create_cli_file, generate_export_data, generate_header_data
from Source.cli_validate import validate_raw_bps
from Source.config import LR, WeppcliffConfig, set_config
from Source.daily import annual_summary, process_daily_data
from Source.data_model import WeppcliffData
from Source.duration import create_duration_data
from Source.erosion import calculate_erosion_indices
from Source.impute import fill_data
from Source.io import check_input_format, export_data, load_input_file
from Source.plots import graph_data
from Source.precip import aggregate_precip_data, process_precip_data
from Source.preprocess import (
    convert_data,
    preserve_missing_data,
    preprocess_file,
    remove_missing_datetime_rows,
    store_data,
)
from Source.quality import initialize_qc_data, quality_check_inputs, quality_check_outputs
from Source.slots import PipelineSlots
from Source.storms import break_storms, trim_data
from Source.units import convert_units


def create_cluster(s: int, mp: str) -> int:
    """Return max worker count (R ``create_cluster`` equivalent)."""
    import multiprocessing

    print("Creating multiprocessing virtual cluster... ", end="")
    cores = multiprocessing.cpu_count()
    cores_90 = int(cores * 0.90)
    cores_4b = cores - 4
    max_cores = max(1, cores_90, cores_4b)
    if str(mp).isdigit():
        max_cores = int(mp)
    workers = min(s, max_cores)
    print(f"Done. Cores deployed: {workers} ")
    return workers


def kill_cluster() -> None:
    """No-op — ProcessPoolExecutor manages worker lifecycle."""
    print("Killing multiprocessing virtual cluster... Done.")


def fill_workflow(data: WeppcliffData, config: WeppcliffConfig) -> WeppcliffData:
    data = fill_data(data, 5, 10, 18, 15)
    config.alt = "f"
    data = break_storms(data, 15, 16, config)
    data = process_daily_data(data, 18, 6, 18, 19, 20, 21, config)
    data = process_precip_data(data, 15, 18, 19, 20, 21, config)
    data = calculate_erosion_indices(data, 16, 17, config)
    data = annual_summary(data, 20, 22, config)
    meta = data.metadata
    dly = data.slot_df(18)
    meta["B_YR"] = str(dly["YYYYMMDD"].iloc[0])[:4]
    end = pd.to_datetime(dly["YYYYMMDD"].iloc[-1], format="%Y%m%d")
    start = pd.to_datetime(dly["YYYYMMDD"].iloc[0], format="%Y%m%d")
    meta["YRS_SIM"] = int((end - start).days / 365.2425)
    data[4] = meta
    return data


def _slots(config: WeppcliffConfig) -> PipelineSlots:
    return PipelineSlots.for_config(config)


def _load_and_preprocess(file_path: str, config: WeppcliffConfig):
    """Load input file, validate format, and preprocess columns."""
    file = load_input_file(file_path, config)
    check_input_format(file, config)
    return preprocess_file(file, config)


def core_workflow(file, config: WeppcliffConfig) -> None:
    set_config(config)
    assign_final_args(config)

    x = [1, 3]
    if config.cv_off:
        x = [1]
        config.id = "f"
        config.qc = "f"
        config.apply_runtime_flags()

    data = store_data(file, config)
    data = convert_data(data, config)
    data = remove_missing_datetime_rows(data, config)
    data = preserve_missing_data(data, x, config)
    data = create_duration_data(data, x, config)
    if config.do_ai:
        data = aggregate_precip_data(data, config)
    data = convert_units(data, config)
    data = initialize_qc_data(data, config)
    if config.do_qc:
        data = quality_check_inputs(data, config)
    data = trim_data(data, 5, 6, 7, 8, config)
    if config.do_ei:
        data = calculate_erosion_indices(data, 8, 9, config)
    if not config.cv_off:
        data = process_daily_data(data, 7, 6, 10, 11, 12, 13, config)
    data = process_precip_data(data, 5, 10, 11, 12, 13, config)
    if not config.cv_off and int(config.bf) == 1:
        raw_violations = validate_raw_bps(data.slot_df(10), data.slot_df(5))
        if raw_violations and config.do_verb:
            print(
                LR,
                f"Warning: {len(raw_violations)} raw BPS inconsistencies detected.",
                end="",
            )
    if config.do_qc:
        data = quality_check_outputs(data, 10, config)
    data = annual_summary(data, 12, 14, config)
    data = preserve_missing_data(data, [13], config)
    if config.do_id:
        data = fill_workflow(data, config)
    if config.do_pd:
        graph_data(data)
    if int(config.ed) > 0:
        export_data(data, config)
    if not config.cv_off:
        header = generate_header_data(data)
        slots = _slots(config)
        export = generate_export_data(data, slots.daily_ts, slots.precip_qc)
        create_cli_file(data, header, export)


def _process_station(file_path: str, config_blob: bytes) -> float:
    config = pickle.loads(config_blob)
    os.chdir(config.home_dir or ".")
    t0 = time.perf_counter()
    file = _load_and_preprocess(file_path, config)
    core_workflow(file, config)
    return time.perf_counter() - t0


def parallel_execution(files: list[str], s: int, config: WeppcliffConfig) -> None:
    print(f"Detected {s} stations. Parallel processing has been enabled.")
    workers = create_cluster(s, str(config.mp))
    print("Processing... see the log file (log.txt) for progress.")

    log_path = os.path.join(config.out_dir or config.o or ".", "log.txt")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    config_blob = pickle.dumps(config.for_worker())

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_station, files[i], config_blob): i + 1
            for i in range(s)
        }
        for future in as_completed(futures):
            iter_n = futures[future]
            with open(log_path, "w", encoding="utf-8") as log_fh:
                log_fh.write(f"Processing {iter_n} of {s} stations... ")
            future.result()
    kill_cluster()


def series_execution(files: list[str], s: int, config: WeppcliffConfig) -> None:
    if s > 1:
        print(f"Detected {s} stations. Parallel processing has been disabled.")
    else:
        print("Detected 1 station. Parallel processing has been disabled.")

    for iter_n in range(1, s + 1):
        t0 = time.perf_counter()
        if s > 1:
            print(f"Processing {iter_n} of {s} stations... ", end="")
        else:
            print("Processing station... ", end="")
        file = _load_and_preprocess(files[iter_n - 1], config)
        core_workflow(file, config)
        elapsed = time.perf_counter() - t0
        print(f"Completed in {elapsed:.2f} seconds.")