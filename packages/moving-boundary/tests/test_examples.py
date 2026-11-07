"""Run the shipped examples through their installed-API code paths and check behaviour.

No private fixtures are used. Tests needing a database read ``MB_EXAMPLE_TDB`` and skip
with a reason without one; the local acceptance gate always supplies it. Limits come from
each example, which documents the configuration, normalization and window they apply to.
"""
import ast
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
NEW_EXAMPLES = ["binary_planar_growth", "ternary_interstitial_similarity", "surrogate_lifecycle",
                "ternary_fecrni_lee1996", "three_phase_similarity"]
TERNARY_MODELS = {"MovingBoundaryIllingworthTernaryFD1DModel", "MovingBoundaryIllingworthTernaryThreePhaseFD1DModel"}
requires_database = pytest.mark.skipif(not os.environ.get("MB_EXAMPLE_TDB"), reason="set MB_EXAMPLE_TDB to a Fe-Cr-Ni TDB; none is distributed yet")


def load(name):
    """Load an example by path; it isn't an importable package after installation."""
    module_name = "mb_example_" + name
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLES / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NEW_EXAMPLES)
def test_example_uses_only_the_installed_api_and_is_batch_safe(name):
    tree = ast.parse((EXAMPLES / (name + ".py")).read_text(encoding="utf-8"))
    allowed = {"moving_boundary", "kawin", "numpy", "scipy", "matplotlib", "argparse", "contextlib", "dataclasses",
               "hashlib", "importlib", "json", "os", "pathlib", "shutil", "sys", "tempfile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            roots = []
        assert set(roots) <= allowed, (name, roots)
        if isinstance(node, ast.Call):
            callee = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            assert callee not in {"input", "breakpoint", "debugInPlace"}, name
            if callee in TERNARY_MODELS:
                keywords = {kw.arg for kw in node.keywords}
                assert "terminal_thin_phase_policy" in keywords, (name, callee)
    assert not name.startswith("pycalphad"), "Running such a script makes pycalphad import it as a plugin"


@pytest.mark.parametrize("name", NEW_EXAMPLES)
def test_example_has_the_documented_entry_points(name):
    module = load(name)
    assert callable(module.run) and callable(module.build_model)
    assert hasattr(module, "ExampleResult") and {"config", "model", "exact", "metrics"} <= set(module.ExampleResult.__dataclass_fields__)


def within(metrics, limits):
    """Require each named metric to be below its limit; lists apply per interface."""
    for name, limit in limits.items():
        value = np.max(np.asarray(metrics[name], dtype=np.float64))
        assert value < limit, (name, value, limit)


# ---- 1. binary -------------------------------------------------------------

def test_binary_demo_meets_its_accuracy_limits():
    example = load("binary_planar_growth")
    result = example.run(example.DEMO)
    within(result.metrics, example.DEMO_LIMITS)
    assert result.metrics["validity_time_estimate"] > example.DEMO.t_end


def test_binary_fast_runs_and_a_coarse_grid_is_visibly_worse():
    example = load("binary_planar_growth")
    assert example.run(example.FAST).model.currentTime == pytest.approx(example.FAST.duration)
    assert example.run(example.COARSE_LONG).metrics["interface_rel"] > 3.0 * example.run(example.DEMO).metrics["interface_rel"]


# ---- 2. interstitial --------------------------------------------------------

def test_interstitial_demo_meets_its_accuracy_limits_in_u():
    example = load("ternary_interstitial_similarity")
    result = example.run(example.DEMO)
    within(result.metrics, example.DEMO_LIMITS)
    assert result.metrics["validity_time_alpha"] > example.DEMO.duration
    assert result.metrics["validity_time_beta"] > example.DEMO.duration


def test_interstitial_fast_has_no_accuracy_window_and_stays_within_validity():
    example = load("ternary_interstitial_similarity")
    result = example.run(example.FAST)
    assert "k_rel" not in result.metrics
    assert result.metrics["validity_time_alpha"] > example.FAST.duration


def _extended_grid(distances, new_width, growth=1.15):
    """Keep node distances from the interface, then grow spacing out to a farther wall."""
    d = list(distances)
    spacing = d[-1] - d[-2]
    while d[-1] < new_width:
        spacing *= growth
        d.append(d[-1] + spacing)
    d[-1] = new_width
    if d[-1] - d[-2] < 0.5 * (d[-2] - d[-3]):
        d.pop(-2)
    return np.asarray(d)


# Regression thresholds for this enlargement study, at the refined settings below.
# Each metric must move by under 10% of its DEMO limit; direct differences are in u.
DOMAIN_FRACTION_OF_LIMIT = 0.10
DIRECT_INTERFACE_REL = 1.0e-3   # max|Delta(s-s0)| / max|s-s0|, t >= 0.1
DIRECT_CARBON_LINF = 1.0e-4     # max|Delta u_C| in the shared region at t = 0.1, 0.25, 0.5, 1
DIRECT_MANGANESE_LINF = 1.0e-3  # max|Delta u_Mn| likewise


def test_interstitial_domain_enlargement_changes_results_little():
    """Doubling the domain, with matching spacing in the shared region, barely matters.

    The validity time is only an estimate, so this compares a larger domain directly: both
    error metrics and physical trajectories and profiles at common times and coordinates.
    Runs at twice DEMO's resolution, because a Landau grid's spacing changes as the interface
    moves, which is a discretization effect that shrinks with refinement.
    """
    example = load("ternary_interstitial_similarity")
    base = example.Config(left_nodes=161, right_nodes=161, semi_log_dt=0.025)
    wide = example.Config(left_width=1.6, right_width=0.8, left_nodes=161, right_nodes=161, semi_log_dt=0.025, mesh_nodes=2395)
    u, v = example.transformed_grids(base)
    left = _extended_grid(base.left_width * (1.0 - u[::-1]), wide.left_width)
    right = _extended_grid(base.right_width * v, wide.right_width)
    wide_u = (1.0 - left / wide.left_width)[::-1]
    wide_u[0], wide_u[-1] = 0.0, 1.0
    wide_v = right / wide.right_width
    wide_v[0], wide_v[-1] = 0.0, 1.0

    def solve(config, grids=None):
        model = example.build_model(config, grids=grids)
        model.solve(config.duration, minDtFrac=1e-14)
        return model, example.evaluate(model, config, example.exact_solution(config))

    model_a, metrics_a = solve(base)
    model_b, metrics_b = solve(wide, (wide_u, wide_v))
    for name in ("k_rel", "interface_rel", "eta_abs", "carbon_profile_linf"):
        assert abs(metrics_a[name] - metrics_b[name]) < DOMAIN_FRACTION_OF_LIMIT * example.DEMO_LIMITS[name], name

    from moving_boundary.analytic import recorded_history
    times, s_a = recorded_history(model_a.interfaceData)
    times_b, s_b = recorded_history(model_b.interfaceData)
    assert np.array_equal(times, times_b)
    window = times >= 0.1
    displacement_a, displacement_b = s_a[window] - base.s0, s_b[window] - wide.s0
    assert np.max(np.abs(displacement_a - displacement_b)) / np.max(np.abs(displacement_a)) < DIRECT_INTERFACE_REL

    def sample(z, x, state, s0):
        return np.column_stack([np.interp(z, x - s0, state[:, i]) for i in range(2)])

    for target in (0.1, 0.25, 0.5, 1.0):
        time_a = float(times[np.argmin(np.abs(times - target))])
        states_a, states_b = model_a.getTransformedState(time_a), model_b.getTransformedState(time_a)
        coords_a, coords_b = model_a.getPhysicalPhaseProfiles(time_a), model_b.getPhysicalPhaseProfiles(time_a)
        top = min(coords_a[0][0][-1] - base.s0, coords_b[0][0][-1] - wide.s0)
        bottom = max(coords_a[1][0][0] - base.s0, coords_b[1][0][0] - wide.s0)
        shared = (np.linspace(-0.99 * base.left_width, top, 800), np.linspace(bottom, 0.99 * base.right_width, 800))
        for phase, z in enumerate(shared):
            difference = np.abs(sample(z, coords_a[phase][0], states_a[phase], base.s0) - sample(z, coords_b[phase][0], states_b[phase], wide.s0))
            assert np.max(difference[:, 0]) < DIRECT_CARBON_LINF, (target, phase)
            assert np.max(difference[:, 1]) < DIRECT_MANGANESE_LINF, (target, phase)


# ---- 3. lifecycle -----------------------------------------------------------

def test_surrogate_lifecycle_builds_reloads_rebuilds_and_detects_damage(tmp_path):
    example = load("surrogate_lifecycle")
    first = example.run(output_dir=tmp_path)
    assert first.metrics["action"] == "built"
    assert first.metrics["fresh_vs_reloaded_interface_abs"] == 0.0
    assert first.metrics["changed_specification_refused"] and first.metrics["damaged_member_refused"]
    assert example.run(output_dir=tmp_path).metrics["action"] == "reloaded"
    assert example.run(output_dir=tmp_path, rebuild=True).metrics["action"] == "built"


def test_surrogate_lifecycle_refuses_an_incompatible_bundle_without_rebuilding(tmp_path, monkeypatch):
    example = load("surrogate_lifecycle")
    example.run(output_dir=tmp_path)
    monkeypatch.setitem(example.BUILD_SPEC, "revision", 99)
    monkeypatch.setattr(example, "build_surrogate", lambda: pytest.fail("A reload must not build"))
    with pytest.raises(example.SurrogateArtifactCompatibilityError, match="--rebuild"):
        example.prepare_bundle(tmp_path / "bundle")


# ---- 4. Fe-Cr-Ni ------------------------------------------------------------

def test_fecrni_without_a_database_explains_what_is_needed(monkeypatch, tmp_path):
    example = load("ternary_fecrni_lee1996")
    monkeypatch.delenv("MB_EXAMPLE_TDB", raising=False)
    with pytest.raises(FileNotFoundError, match="--tdb"):
        example.prepare_surrogate(example.LAGGED_FAST, None, output_dir=tmp_path)


def test_database_digest_ignores_line_endings(tmp_path):
    example = load("ternary_fecrni_lee1996")
    (tmp_path / "lf.tdb").write_bytes(b"ELEMENT A\nELEMENT B\n")
    (tmp_path / "crlf.tdb").write_bytes(b"ELEMENT A\r\nELEMENT B\r\n")
    assert example.database_digest(tmp_path / "lf.tdb") == example.database_digest(tmp_path / "crlf.tdb")


def test_simplex_points_stay_inside_the_simplex():
    example = load("ternary_fecrni_lee1996")
    points = example.simplex_bulk_points(0.1)
    assert len(points) == 66
    # Rescaling a corner back onto the simplex leaves it just under the floor, which is harmless.
    assert np.all(points >= 0.99e-5) and np.all(points.sum(axis=1) <= 1.0 - 1.0e-5 + 1.0e-12)


@requires_database
def test_live_lagged_workflow_uses_its_data(tmp_path):
    """Build the simplex surrogate live and complete the real lagged workflow.

    These are workflow properties, not accuracy. Conservation and reload can both pass
    without the bulk path being used, so this also checks hull coverage from the retained
    samples, real matrix variation over accepted states, and interpolation evidence.
    """
    example = load("ternary_fecrni_lee1996")
    config = example.LAGGED_FAST
    result = example.run(config, output_dir=tmp_path)
    limits = example.LAGGED_THRESHOLDS
    metrics = result.metrics
    assert metrics["surrogate_action"] == "built"
    assert result.model.currentTime == pytest.approx(config.duration)
    assert metrics["inventory_rel_max"] < limits["inventory_rel_max"]
    workflow = metrics["workflow"]
    assert workflow["interpolation"] == "simplex_positive_2x2"
    for phase in example.PHASES:
        report = workflow[phase]
        assert report["retained_samples"] > len(example.simplex_bulk_points(config.bulk_spacing)) * 0.9
        assert report["accepted_outside_hull"] == 0
        assert report["attempted_outside_hull"] == 0
        assert report["attempted_queries"] > 0
        assert report["matrix_spread"] > limits["matrix_spread_min"]
        assert report["simplices_crossed"] >= limits["simplices_crossed_min"]
        assert report["interpolation_vs_nearest_median"] > limits["interpolation_vs_nearest_min"]
        assert report["centroids_within_vertex_range"] > 0.9
        assert report["training_point_relative_error"] < 1.0e-8
    assert metrics["lagged_vs_frozen"] > limits["lagged_vs_frozen_min"]


@requires_database
def test_fresh_and_reloaded_lagged_solves_match(tmp_path):
    example = load("ternary_fecrni_lee1996")
    config = example.LAGGED_FAST
    bundle, action = example.prepare_surrogate(config, None, output_dir=tmp_path)
    assert action == "built"
    reloaded, action = example.prepare_surrogate(config, None, output_dir=tmp_path)
    assert action == "reloaded"
    assert reloaded.manifest["build_fingerprint"] == bundle.manifest["build_fingerprint"]
    runs = []
    for surrogate in (bundle.surrogate, reloaded.surrogate):
        model = example.build_model(config, surrogate)
        model.solve(config.duration, minDtFrac=1e-14)
        runs.append(np.array(example.recorded_history(model.interfaceData)[1]))
    assert np.max(np.abs(runs[0] - runs[1])) == 0.0


@requires_database
def test_fecrni_frozen_matrices_meet_their_analytic_limits(tmp_path):
    example = load("ternary_fecrni_lee1996")
    result = example.run(example.FROZEN_ANALYTIC, output_dir=tmp_path)
    within(result.metrics, example.FROZEN_ANALYTIC_LIMITS)


# ---- 5. three-phase ---------------------------------------------------------

@pytest.mark.parametrize("config_name", ["DEMO", "DEMO_DRIFT"])
def test_three_phase_demo_meets_its_accuracy_limits(config_name):
    example = load("three_phase_similarity")
    result = example.run(getattr(example, config_name))
    within(result.metrics, example.DEMO_LIMITS)
    assert result.metrics["validity_time_a"] > 1.0 and result.metrics["validity_time_c"] > 1.0


def test_three_phase_fast_runs_without_an_accuracy_claim():
    example = load("three_phase_similarity")
    result = example.run(example.FAST)
    assert "k_rel" not in result.metrics


@pytest.mark.slow
def test_three_phase_errors_converge_with_refinement():
    """Source: test_three_phase_model_matches_asymmetric_similarity_solution, same settings."""
    example = load("three_phase_similarity")
    errors = [np.asarray(example.run(example.Config(nodes=nodes)).metrics["k_rel"]) for nodes in (21, 41, 81)]
    assert np.all(errors[1] < errors[0]) and np.all(errors[2] < errors[1])
    orders = np.log2(errors[0] / errors[1]), np.log2(errors[1] / errors[2])
    assert all(np.all((order > 0.6) & (order < 1.5)) for order in orders)


# ---- acceptance tooling ------------------------------------------------------

def test_settings_record_catches_a_post_construction_change():
    """Constructor arguments alone would miss this: the record is of effective settings."""
    spec = importlib.util.spec_from_file_location("example_settings", Path(__file__).parents[1] / "tools/example_settings.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    example = load("ternary_interstitial_similarity")
    arguments = dict(minDtFrac=1e-14)
    baseline = settings.effective_settings(example.build_model(example.FAST), example.FAST.duration, arguments)
    assert settings.differences(baseline, settings.effective_settings(example.build_model(example.FAST), example.FAST.duration, arguments)) == []
    changed = example.build_model(example.FAST)
    changed.dtMode, changed.timeStep = "fixed", 600.0
    found = settings.differences(baseline, settings.effective_settings(changed, example.FAST.duration, arguments))
    assert {"attributes.dtMode", "attributes.timeStep"} <= set(found)
    assert settings.differences(baseline, settings.effective_settings(changed, example.FAST.duration, arguments), {"dtMode": "why", "timeStep": "why"}) == []


# ---- notebook use ------------------------------------------------------------

@pytest.mark.parametrize("name", ["binary_planar_growth", "ternary_interstitial_similarity", "surrogate_lifecycle", "three_phase_similarity"])
def test_example_runs_as_a_notebook_kernel_would(name, tmp_path, monkeypatch):
    """Run every cell in order as a Jupyter kernel does, including its own argv and plots.

    A kernel passes ``--f=<connection file>``, which argparse would read as ``--fast`` and
    reject; the examples must ignore it. Outputs go to a temporary home directory.
    """
    import runpy
    import types
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher.py", f"--f={tmp_path / 'kernel.json'}"])
    namespace = runpy.run_path(str(EXAMPLES / (name + ".py")), run_name="__main__")
    if name == "surrogate_lifecycle":
        assert Path(namespace["bundle"].path).is_relative_to(tmp_path)
        assert namespace["interface_difference"](namespace["model"], namespace["fresh"]) == 0.0
    else:
        assert namespace["result"].config == namespace["CONFIG"]


@pytest.mark.parametrize("name", ["binary_planar_growth", "ternary_interstitial_similarity", "three_phase_similarity"])
def test_example_command_line_still_parses_options(name, monkeypatch):
    example = load(name)
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    monkeypatch.setattr(sys, "argv", [name + ".py", "--fast"])
    config, plot_figures = example.settings_from_command_line(example.DEMO, True)
    assert config == example.FAST and plot_figures is False
    monkeypatch.setattr(sys, "argv", [name + ".py", "--f=kernel.json"])
    with pytest.raises(SystemExit):
        example.settings_from_command_line(example.DEMO, True)
