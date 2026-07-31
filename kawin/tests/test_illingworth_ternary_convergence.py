import numpy as np

from examples.ternaryExamples import IllingworthTernaryConvergence as convergence


def _synthetic_run(nodes, semi_log_dt, *, shift=0.0, key=None):
    times = np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    interface = np.asarray([1.0, 1.1, 1.2, 1.4], dtype=np.float64) + float(shift)
    eta = np.asarray([0.2, 0.25, 0.30, 0.40], dtype=np.float64) + 0.5 * float(shift)
    inventory = np.asarray(
        [
            [0.10, 0.20],
            [0.10 + shift, 0.20 - shift],
            [0.10, 0.20],
            [0.10 - shift, 0.20 + shift],
        ],
        dtype=np.float64,
    )
    u_grid = np.linspace(0.0, 1.0, 3 if nodes < 60 else 5)
    v_grid = np.linspace(0.0, 1.0, 4 if nodes < 60 else 6)
    p_final = np.column_stack((u_grid, 2.0 * u_grid)) + shift
    q_final = np.column_stack((1.0 - v_grid, v_grid)) - shift
    initial_concentration = np.asarray([0.10, 0.20], dtype=np.float64)
    idealized_concentration = np.asarray([0.11, 0.19], dtype=np.float64)
    return {
        "key": key or f"N={nodes}, semiLog_dt={semi_log_dt:g}",
        "label": key or f"N={nodes}, semiLog_dt={semi_log_dt:g}",
        "parameters": {
            "nodes": nodes,
            "phase_a_nodes": int(len(u_grid)),
            "phase_b_nodes": int(len(v_grid)),
            "dt_mode": "semi_log",
            "semi_log_dt": semi_log_dt,
        },
        "runtime_s": 0.0,
        "step_count": len(times) - 1,
        "times": times,
        "eta_times": times,
        "inventory_times": times,
        "interface_position": interface,
        "eta": eta,
        "inventory": inventory,
        "inventory_drift": np.max(np.abs(inventory - inventory[0]), axis=0),
        "initial_concentration": initial_concentration,
        "idealized_concentration": idealized_concentration,
        "initial_minus_idealized_concentration": initial_concentration - idealized_concentration,
        "initial_interface_position": float(interface[0]),
        "final_interface_position": float(interface[-1]),
        "initial_eta": float(eta[0]),
        "final_eta": float(eta[-1]),
        "final_time": float(times[-1]),
        "u_grid": u_grid,
        "v_grid": v_grid,
        "p_final": p_final,
        "q_final": q_final,
    }


def test_convergence_reference_selection_uses_finest_nodes_and_smallest_timestep():
    coarse = _synthetic_run(31, 0.025)
    fine_large_dt = _synthetic_run(61, 0.1)
    fine_small_dt = _synthetic_run(61, 0.025)

    reference = convergence._select_reference_run([coarse, fine_large_dt, fine_small_dt])

    assert reference is fine_small_dt


def test_convergence_common_analysis_times_stay_inside_all_histories():
    early = _synthetic_run(31, 0.1)
    late = _synthetic_run(61, 0.05)
    early["times"] = np.asarray([0.0, 0.25, 1.0, 2.0], dtype=np.float64)
    late["times"] = np.asarray([0.0, 0.5, 1.5, 3.0], dtype=np.float64)

    times = convergence._common_analysis_times([early, late], count=5, include_zero=True)

    assert times[0] == 0.0
    assert np.min(times[times > 0.0]) >= 0.5
    assert np.max(times) <= 2.0
    assert np.all(np.diff(times) > 0.0)


def test_convergence_scalar_errors_compare_against_reference():
    reference = _synthetic_run(61, 0.025, shift=0.0, key="reference")
    candidate = _synthetic_run(31, 0.1, shift=0.1, key="candidate")
    analysis_times = np.asarray([0.0, 1.0, 4.0], dtype=np.float64)

    errors = convergence._compute_errors([candidate, reference], reference, analysis_times)

    assert errors["reference"]["final_interface_error"] == 0.0
    assert np.isclose(errors["candidate"]["final_interface_error"], 0.1)
    assert np.isclose(errors["candidate"]["max_interface_error"], 0.1)
    assert np.isclose(errors["candidate"]["final_eta_error"], 0.05)
    assert np.isclose(errors["candidate"]["max_eta_error"], 0.05)


def test_convergence_profile_interpolation_is_componentwise():
    source_grid = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    source_profile = np.column_stack((source_grid, source_grid * source_grid))
    target_grid = np.asarray([0.0, 0.25, 0.75, 1.0], dtype=np.float64)

    interpolated = convergence._interp_profile(source_grid, source_profile, target_grid)

    assert np.allclose(interpolated[:, 0], target_grid)
    assert np.allclose(interpolated[:, 1], np.asarray([0.0, 0.125, 0.625, 1.0]))


def test_convergence_summary_falls_back_to_rows_or_dataframe():
    reference = _synthetic_run(61, 0.025, key="reference")
    candidate = _synthetic_run(31, 0.1, shift=0.1, key="candidate")
    errors = convergence._compute_errors([candidate, reference], reference, np.asarray([0.0, 1.0, 4.0]))

    summary = convergence.summarize_convergence({"runs": [candidate, reference], "reference": reference, "errors": errors})
    rows = convergence._summary_rows(summary)

    assert {row["key"] for row in rows} == {"candidate", "reference"}
    candidate_row = next(row for row in rows if row["key"] == "candidate")
    assert candidate_row["phase_a_nodes"] == 3
    assert candidate_row["phase_b_nodes"] == 4
    assert np.isclose(candidate_row["initial_eta"], 0.25)
    assert np.isclose(candidate_row["profile_linf_error"], errors["candidate"]["profile_linf_error"])
    assert np.isclose(candidate_row["initial_concentration_cr"], 0.10)
    assert np.isclose(candidate_row["idealized_concentration_ni"], 0.19)
    assert np.isclose(candidate_row["initial_minus_idealized_concentration_cr"], -0.01)
