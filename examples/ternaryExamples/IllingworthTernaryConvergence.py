"""
Notebook-friendly convergence helpers for the ternary Illingworth example.

The functions in this module wrap ``IllingworthTernaryExamples`` without making
command-line usage the primary interface. A typical interactive workflow is::

    from examples.ternaryExamples import IllingworthTernaryConvergence as conv

    cfg = conv.default_convergence_config()
    cfg["nodes"] = [31, 61]
    cfg["semi_log_dt"] = [0.1, 0.05]
    results = conv.run_convergence_sweep(cfg)
    summary = conv.summarize_convergence(results)
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_EXAMPLE_MODULE_NAME = "examples.ternaryExamples.IllingworthTernaryExamples"


def _example_module():
    """
    Returns the ternary example module without importing it at convergence-module load.

    This lazy lookup avoids circular imports when users import this convergence
    helper from inside ``IllingworthTernaryExamples.py`` or a notebook cell that
    has already executed the example definitions.
    """
    for module_name in ("__main__", _EXAMPLE_MODULE_NAME):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "build_case_context") and hasattr(module, "run_case"):
            return module
    try:
        return importlib.import_module(_EXAMPLE_MODULE_NAME)
    except ImportError:  # pragma: no cover - supports direct notebook execution from this directory.
        return importlib.import_module("IllingworthTernaryExamples")


def default_convergence_config(dt_mode="semi_log"):
    """
    Returns a mutable default convergence-sweep configuration dictionary.

    ``dt_mode='semi_log'`` sweeps ``semi_log_dt`` values, while
    ``dt_mode='fixed'`` sweeps ``fixed_time_step`` values. The reference run is
    always chosen later as the run with the largest node count and smallest
    timestep control value.
    """
    dt_mode = str(dt_mode)
    if dt_mode not in {"semi_log", "fixed"}:
        raise ValueError("dt_mode must be 'semi_log' or 'fixed'.")
    example = _example_module()
    config = {
        "dt_mode": dt_mode,
        "nodes": [31, 61, 121],
        "solve_time": example.SOLVE_TIME,
        "analysis_time_count": 64,
        "keep_models": True,
        "verbose": False,
        "base_overrides": {},
    }
    if dt_mode == "semi_log":
        config["semi_log_dt"] = [0.1, 0.05, 0.025]
    else:
        config["fixed_time_step"] = [4.0, 2.0, 1.0]
    return config


def run_convergence_sweep(config=None, context=None, progress=True):
    """
    Runs a mesh/timestep convergence sweep and returns structured results.

    The returned dictionary contains the normalized config, per-run records, the
    selected reference run, common scalar analysis times, error dictionaries,
    and a summary table. Expensive thermodynamics setup is reused through
    ``context`` when supplied, or built once when omitted.
    """
    config = _normalize_config(config)
    example = _example_module()
    if context is None:
        context = example.build_case_context(
            overrides=config["base_overrides"],
            print_matrices=bool(config.get("print_matrices", progress)),
        )

    runs = []
    sweep_specs = _sweep_specs(config)
    total = len(sweep_specs)
    for index, overrides in enumerate(sweep_specs, start=1):
        label = _run_label(overrides)
        if progress:
            print(f"[{index}/{total}] running {label}")
        start_time = time.perf_counter()
        run = example.run_case(overrides=overrides, context=context, make_plots=False)
        runtime_s = time.perf_counter() - start_time
        runs.append(_extract_run_record(run["model"], overrides, runtime_s=runtime_s, keep_model=config["keep_models"]))

    reference_run = _select_reference_run(runs)
    analysis_times = _common_analysis_times(runs, count=config["analysis_time_count"], include_zero=True)
    errors = _compute_errors(runs, reference_run, analysis_times)
    summary = summarize_convergence({"runs": runs, "reference": reference_run, "errors": errors})
    return {
        "config": config,
        "context": context,
        "runs": runs,
        "reference": reference_run,
        "analysis_times": analysis_times,
        "errors": errors,
        "summary": summary,
    }


def summarize_convergence(results):
    """
    Returns a tabular convergence summary.

    A ``pandas.DataFrame`` is returned when pandas is importable; otherwise the
    fallback is a list of dictionaries with the same row content.
    """
    runs = list(results["runs"])
    errors = results.get("errors") or {}
    rows = []
    for run in runs:
        key = run["key"]
        run_errors = errors.get(key, {})
        inventory_drift = np.asarray(run.get("inventory_drift", np.full(2, np.nan)), dtype=np.float64)
        initial_concentration = np.asarray(run.get("initial_concentration", np.full(2, np.nan)), dtype=np.float64)
        idealized_concentration = np.asarray(run.get("idealized_concentration", np.full(2, np.nan)), dtype=np.float64)
        initial_minus_idealized = np.asarray(run.get("initial_minus_idealized_concentration", np.full(2, np.nan)), dtype=np.float64)
        rows.append(
            {
                "key": key,
                "nodes": run["parameters"]["nodes"],
                "phase_a_nodes": run["parameters"]["phase_a_nodes"],
                "phase_b_nodes": run["parameters"]["phase_b_nodes"],
                "dt_mode": run["parameters"]["dt_mode"],
                "semi_log_dt": run["parameters"].get("semi_log_dt"),
                "fixed_time_step": run["parameters"].get("fixed_time_step"),
                "step_count": run["step_count"],
                "runtime_s": run["runtime_s"],
                "final_time": run["final_time"],
                "final_interface_position": run["final_interface_position"],
                "initial_eta": run["initial_eta"],
                "final_eta": run["final_eta"],
                "initial_concentration_cr": initial_concentration[0],
                "initial_concentration_ni": initial_concentration[1],
                "idealized_concentration_cr": idealized_concentration[0],
                "idealized_concentration_ni": idealized_concentration[1],
                "initial_minus_idealized_concentration_cr": initial_minus_idealized[0],
                "initial_minus_idealized_concentration_ni": initial_minus_idealized[1],
                "inventory_drift_cr": inventory_drift[0],
                "inventory_drift_ni": inventory_drift[1],
                "final_interface_error": run_errors.get("final_interface_error", np.nan),
                "final_normalized_interface_error": run_errors.get("final_normalized_interface_error", np.nan),
                "max_interface_error": run_errors.get("max_interface_error", np.nan),
                "final_eta_error": run_errors.get("final_eta_error", np.nan),
                "max_eta_error": run_errors.get("max_eta_error", np.nan),
                "profile_linf_error": run_errors.get("profile_linf_error", np.nan),
                "profile_rms_error": run_errors.get("profile_rms_error", np.nan),
            }
        )
    _add_observed_orders(rows)
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


def plot_interface_convergence(results, ax=None):
    """
    Plots normalized interface position histories for all convergence runs.

    The function is notebook-friendly: it returns ``(fig, ax)`` and never calls
    ``plt.show()``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    for run in results["runs"]:
        times = np.asarray(run["times"], dtype=np.float64)
        positive = times > 0.0
        if not np.any(positive):
            continue
        normalized = np.asarray(run["interface_position"], dtype=np.float64) / float(run["initial_interface_position"])
        ax.plot(times[positive], normalized[positive], marker="o", linewidth=1.2, label=run["label"])
    ax.set_xscale("log")
    ax.set_xlabel("time / s")
    ax.set_ylabel("normalized interface position")
    ax.set_title("Ternary Illingworth convergence sweep")
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig, ax


def plot_profile_convergence(results, component=None, ax=None):
    """
    Plots final transformed-profile error versus run controls.

    ``component`` may be ``None`` for the combined profile error, ``0`` for CR,
    or ``1`` for NI. The current summary-level implementation plots ``Linf``
    profile error for each non-reference run.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    errors = results["errors"]
    xs = []
    ys = []
    labels = []
    for run in results["runs"]:
        run_errors = errors.get(run["key"], {})
        error = run_errors.get("profile_linf_error", np.nan)
        if component is not None:
            component_errors = run_errors.get("profile_linf_error_by_component")
            if component_errors is not None:
                error = np.asarray(component_errors, dtype=np.float64)[int(component)]
        xs.append(_timestep_control(run))
        ys.append(error)
        labels.append(f"N={run['parameters']['nodes']}")
    ax.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        if np.isfinite(y):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize="small")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("timestep control")
    ax.set_ylabel("final transformed-profile Linf error")
    ax.set_title("Final profile convergence")
    fig.tight_layout()
    return fig, ax


def save_convergence_results(results, output_dir):
    """
    Saves convergence summary, numeric arrays, metadata, and quick-look plots.

    The output directory is created if needed. The return value maps artifact
    names to paths for notebook inspection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _summary_rows(results["summary"])
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0])) if summary_rows else None
        if writer is not None:
            writer.writeheader()
            writer.writerows(summary_rows)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "config": _jsonable(results["config"]),
                "reference_key": results["reference"]["key"],
                "run_keys": [run["key"] for run in results["runs"]],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    arrays = {"analysis_times": np.asarray(results["analysis_times"], dtype=np.float64)}
    for run in results["runs"]:
        prefix = _safe_key(run["key"])
        arrays[f"{prefix}_times"] = np.asarray(run["times"], dtype=np.float64)
        arrays[f"{prefix}_interface_position"] = np.asarray(run["interface_position"], dtype=np.float64)
        arrays[f"{prefix}_eta"] = np.asarray(run["eta"], dtype=np.float64)
        arrays[f"{prefix}_inventory"] = np.asarray(run["inventory"], dtype=np.float64)
    npz_path = output_dir / "results.npz"
    np.savez_compressed(npz_path, **arrays)

    interface_fig, _ = plot_interface_convergence(results)
    interface_path = output_dir / "interface_error.svg"
    interface_fig.savefig(interface_path)
    plt.close(interface_fig)

    profile_fig, _ = plot_profile_convergence(results)
    profile_path = output_dir / "profile_error.svg"
    profile_fig.savefig(profile_path)
    plt.close(profile_fig)

    return {
        "summary_csv": summary_path,
        "metadata_json": metadata_path,
        "results_npz": npz_path,
        "interface_svg": interface_path,
        "profile_svg": profile_path,
    }


def _normalize_config(config):
    config = default_convergence_config() if config is None else dict(config)
    dt_mode = str(config.get("dt_mode", "semi_log"))
    if dt_mode not in {"semi_log", "fixed"}:
        raise ValueError("config['dt_mode'] must be 'semi_log' or 'fixed'.")
    defaults = default_convergence_config(dt_mode)
    defaults.update(config)
    defaults["dt_mode"] = dt_mode
    defaults["nodes"] = [int(value) for value in defaults["nodes"]]
    defaults["analysis_time_count"] = int(defaults["analysis_time_count"])
    defaults["keep_models"] = bool(defaults.get("keep_models", True))
    defaults["verbose"] = bool(defaults.get("verbose", False))
    defaults["base_overrides"] = dict(defaults.get("base_overrides", {}))
    if dt_mode == "semi_log":
        defaults["semi_log_dt"] = [float(value) for value in defaults["semi_log_dt"]]
    else:
        defaults["fixed_time_step"] = [float(value) for value in defaults["fixed_time_step"]]
    return defaults


def _sweep_specs(config):
    specs = []
    for nodes in config["nodes"]:
        controls = config["semi_log_dt"] if config["dt_mode"] == "semi_log" else config["fixed_time_step"]
        for control in controls:
            overrides = dict(config["base_overrides"])
            overrides.update(
                {
                    "NODES": int(nodes),
                    "DT_MODE": config["dt_mode"],
                    "SOLVE_TIME": float(config["solve_time"]),
                    "VERBOSE": bool(config["verbose"]),
                }
            )
            if config["dt_mode"] == "semi_log":
                overrides["SEMI_LOG_DT"] = float(control)
            else:
                overrides["FIXED_TIME_STEP"] = float(control)
            specs.append(overrides)
    return specs


def _run_label(overrides):
    dt_mode = overrides["DT_MODE"]
    if dt_mode == "semi_log":
        return f"N={overrides['NODES']}, semiLog_dt={overrides['SEMI_LOG_DT']:g}"
    return f"N={overrides['NODES']}, fixed_dt={overrides['FIXED_TIME_STEP']:g}"


def _extract_run_record(model, overrides, *, runtime_s=0.0, keep_model=True):
    n_interface = model.interfaceData.N + 1
    n_eta = model.etaData.N + 1
    n_inventory = model.inventoryData.N + 1
    parameters = {
        "nodes": int(overrides["NODES"]),
        "phase_a_nodes": int(len(model._u_grid)),
        "phase_b_nodes": int(len(model._v_grid)),
        "dt_mode": str(overrides["DT_MODE"]),
    }
    if parameters["dt_mode"] == "semi_log":
        parameters["semi_log_dt"] = float(overrides["SEMI_LOG_DT"])
    else:
        parameters["fixed_time_step"] = float(overrides["FIXED_TIME_STEP"])
    inventory = np.asarray(model.inventoryData._y[:n_inventory], dtype=np.float64)
    initial_concentration = np.asarray(model.inventoryData._y[0], dtype=np.float64) / float(model._R)
    idealized_concentration = _idealized_concentration(overrides)
    record = {
        "key": _run_label(overrides),
        "label": _run_label(overrides),
        "parameters": parameters,
        "runtime_s": float(runtime_s),
        "step_count": int(max(n_interface, 1) - 1),
        "times": np.asarray(model.interfaceData._time[:n_interface], dtype=np.float64),
        "eta_times": np.asarray(model.etaData._time[:n_eta], dtype=np.float64),
        "inventory_times": np.asarray(model.inventoryData._time[:n_inventory], dtype=np.float64),
        "interface_position": np.asarray(model.interfaceData._y[:n_interface], dtype=np.float64),
        "eta": np.asarray(model.etaData._y[:n_eta], dtype=np.float64),
        "inventory": inventory,
        "inventory_drift": np.max(np.abs(inventory - inventory[0]), axis=0) if len(inventory) else np.full(2, np.nan),
        "initial_concentration": initial_concentration,
        "idealized_concentration": idealized_concentration,
        "initial_minus_idealized_concentration": initial_concentration - idealized_concentration,
        "initial_interface_position": float(model.interfaceData._y[0]),
        "final_interface_position": float(model.interfaceData._y[n_interface - 1]),
        "initial_eta": float(model.etaData._y[0]),
        "final_eta": float(model.etaData._y[n_eta - 1]),
        "final_time": float(model.currentTime),
        "u_grid": np.asarray(model._u_grid, dtype=np.float64).copy(),
        "v_grid": np.asarray(model._v_grid, dtype=np.float64).copy(),
        "p_final": np.asarray(model.pData._y[model.pData.N], dtype=np.float64).copy() if model.pData is not None else None,
        "q_final": np.asarray(model.qData._y[model.qData.N], dtype=np.float64).copy() if model.qData is not None else None,
    }
    if keep_model:
        record["model"] = model
    return record


def _idealized_concentration(overrides):
    """Returns the continuum two-bulk idealized composition for one run setup."""
    example = _example_module()
    if hasattr(example, "_temporary_config"):
        with example._temporary_config(overrides):
            return np.asarray(example.idealized_comp, dtype=np.float64).copy()
    left_bulk = np.asarray(getattr(example, "LEFT_BULK"), dtype=np.float64)
    right_bulk = np.asarray(getattr(example, "RIGHT_BULK"), dtype=np.float64)
    interface_position = float(getattr(example, "INTERFACE_POSITION"))
    length = float(getattr(example, "LENGTH"))
    return left_bulk * (interface_position / length) + right_bulk * (1.0 - interface_position / length)


def _select_reference_run(runs):
    """Returns the finest-node, smallest-timestep-control run."""
    if not runs:
        raise ValueError("At least one run is required to select a reference.")
    return min(runs, key=lambda run: (-int(run["parameters"]["nodes"]), _timestep_control(run)))


def _timestep_control(run):
    parameters = run["parameters"]
    if parameters["dt_mode"] == "semi_log":
        return float(parameters["semi_log_dt"])
    return float(parameters["fixed_time_step"])


def _common_analysis_times(runs, count=64, include_zero=True):
    """Returns common scalar comparison times lying inside every run history."""
    if not runs:
        return np.asarray([], dtype=np.float64)
    count = max(2, int(count))
    final_time = min(float(np.max(run["times"])) for run in runs)
    positive_starts = []
    for run in runs:
        times = np.asarray(run["times"], dtype=np.float64)
        positives = times[times > 0.0]
        if positives.size:
            positive_starts.append(float(positives[0]))
    if not positive_starts or final_time <= 0.0:
        return np.asarray([0.0], dtype=np.float64) if include_zero else np.asarray([], dtype=np.float64)
    start = max(positive_starts)
    if start >= final_time:
        values = np.asarray([final_time], dtype=np.float64)
    else:
        values = np.geomspace(start, final_time, count)
    if include_zero:
        values = np.concatenate(([0.0], values))
    return np.unique(values)


def _compute_errors(runs, reference_run, analysis_times):
    errors = {}
    ref_s = _interp_scalar(reference_run["times"], reference_run["interface_position"], analysis_times)
    ref_eta = _interp_scalar(reference_run["eta_times"], reference_run["eta"], analysis_times)
    ref_s0 = float(reference_run["initial_interface_position"])
    for run in runs:
        s = _interp_scalar(run["times"], run["interface_position"], analysis_times)
        eta = _interp_scalar(run["eta_times"], run["eta"], analysis_times)
        profile_errors = _final_profile_errors(run, reference_run)
        errors[run["key"]] = {
            "final_interface_error": abs(float(run["final_interface_position"]) - float(reference_run["final_interface_position"])),
            "final_normalized_interface_error": abs(
                float(run["final_interface_position"]) / float(run["initial_interface_position"])
                - float(reference_run["final_interface_position"]) / ref_s0
            ),
            "max_interface_error": float(np.max(np.abs(s - ref_s))),
            "final_eta_error": abs(float(run["final_eta"]) - float(reference_run["final_eta"])),
            "max_eta_error": float(np.max(np.abs(eta - ref_eta))),
            **profile_errors,
        }
    return errors


def _interp_scalar(times, values, query_times):
    """Linearly interpolates scalar history values on requested times."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    if times.size == 0:
        return np.full_like(query_times, np.nan, dtype=np.float64)
    return np.interp(query_times, times, values)


def _final_profile_errors(run, reference_run):
    """Compares final transformed profiles after interpolation to reference grids."""
    if run.get("p_final") is None or reference_run.get("p_final") is None:
        return {
            "profile_linf_error": np.nan,
            "profile_rms_error": np.nan,
            "profile_linf_error_by_component": np.full(2, np.nan),
        }
    p_interp = _interp_profile(run["u_grid"], run["p_final"], reference_run["u_grid"])
    q_interp = _interp_profile(run["v_grid"], run["q_final"], reference_run["v_grid"])
    p_delta = p_interp - np.asarray(reference_run["p_final"], dtype=np.float64)
    q_delta = q_interp - np.asarray(reference_run["q_final"], dtype=np.float64)
    combined = np.concatenate((p_delta, q_delta), axis=0)
    return {
        "profile_linf_error": float(np.max(np.abs(combined))),
        "profile_rms_error": float(np.sqrt(np.mean(combined * combined))),
        "profile_linf_error_by_component": np.max(np.abs(combined), axis=0),
    }


def _interp_profile(source_grid, source_profile, target_grid):
    """
    Interpolates a transformed two-component profile onto another monotone grid.

    This is intentionally componentwise because the Illingworth transformed
    profiles store independent ternary components on phase-specific grids.
    """
    source_grid = np.asarray(source_grid, dtype=np.float64)
    source_profile = np.asarray(source_profile, dtype=np.float64)
    target_grid = np.asarray(target_grid, dtype=np.float64)
    values = np.empty((target_grid.size, source_profile.shape[1]), dtype=np.float64)
    for component in range(source_profile.shape[1]):
        values[:, component] = np.interp(target_grid, source_grid, source_profile[:, component])
    return values


def _add_observed_orders(rows):
    previous_by_family = {}
    for row in sorted(rows, key=lambda item: (item["dt_mode"], item.get("semi_log_dt") or item.get("fixed_time_step"), item["nodes"])):
        family = (row["dt_mode"], row.get("semi_log_dt"), row.get("fixed_time_step"))
        previous = previous_by_family.get(family)
        row["observed_node_order"] = np.nan
        if previous is not None:
            previous_error = previous.get("final_interface_error", np.nan)
            current_error = row.get("final_interface_error", np.nan)
            if previous_error > 0.0 and current_error > 0.0 and row["nodes"] != previous["nodes"]:
                refinement = float(row["nodes"] - 1) / float(previous["nodes"] - 1)
                row["observed_node_order"] = np.log(previous_error / current_error) / np.log(refinement)
        previous_by_family[family] = row


def _summary_rows(summary):
    if hasattr(summary, "to_dict"):
        return summary.to_dict(orient="records")
    return list(summary)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _safe_key(key):
    return "".join(char if char.isalnum() else "_" for char in str(key)).strip("_")
