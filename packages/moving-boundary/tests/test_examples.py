"""Run the shipped examples through their installed-API code paths and check behaviour.

No private fixtures are used. Tests needing a database read ``MB_EXAMPLE_TDB`` and skip
with a reason without one; the local acceptance gate always supplies it. The open Fe-Cr-Ni
workflow uses its bundled open database and needs neither ``MB_EXAMPLE_TDB`` nor Thermo-Calc:
its Thermo-Calc paths are tested with a relabelled synthetic bundle and ``tc_python`` blocked.
Limits come from each example, which documents the configuration, normalization and window
they apply to.
"""
import ast
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
NEW_EXAMPLES = ["binary_planar_growth", "binary_tlp_bonding", "ternary_interstitial_similarity", "surrogate_lifecycle",
                "ternary_fecrni_lee1996", "three_phase_similarity", "surrogate_workflow_fecrni_open"]
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
               "hashlib", "importlib", "json", "os", "pathlib", "shutil", "sys", "tempfile", "time"}
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


def test_tlp_bonding_demo_meets_its_limits_and_widens_then_solidifies():
    example = load("binary_tlp_bonding")
    result = example.run(example.DEMO)
    within(result.metrics, example.DEMO_LIMITS)
    metrics = result.metrics
    assert metrics["peak_width"] > example.PARAMS["s0"] and metrics["final_width"] < 0.25 * metrics["peak_width"]
    assert metrics["peak_time"] < 10.0 < metrics["final_time"]


def test_tlp_bonding_growth_constant_satisfies_the_flux_balance():
    """Check k against the interface balance written with plain erfc, independently of erfcx."""
    from scipy.special import erfc

    example = load("binary_tlp_bonding")
    p, k = example.PARAMS, example.similarity_growth_constant()
    liquid = -(p["c_liquid"] - p["c_liquid0"]) * np.sqrt(p["D_liquid"] / np.pi) * np.exp(-k**2 / (4 * p["D_liquid"])) / erfc(-k / (2 * np.sqrt(p["D_liquid"])))
    solid = -(p["c_solid"] - p["c_solid0"]) * np.sqrt(p["D_solid"] / np.pi) * np.exp(-k**2 / (4 * p["D_solid"])) / erfc(k / (2 * np.sqrt(p["D_solid"])))
    assert (p["c_liquid"] - p["c_solid"]) * k / 2 == pytest.approx(liquid + solid, rel=1.0e-12)
    assert example.width_bound() == pytest.approx(12.5 * 0.19 / 0.10223)


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
    differences = first.metrics["fresh_vs_reloaded_max_abs"]
    assert set(differences) == {"times", "interface", "eta", "inventory", "profiles"}
    assert all(value == 0.0 for value in differences.values()), differences
    assert first.metrics["changed_specification_refused"] and first.metrics["damaged_member_refused"]
    assert example.run(output_dir=tmp_path).metrics["action"] == "reloaded"
    assert example.run(output_dir=tmp_path, rebuild=True).metrics["action"] == "built"


def test_surrogate_lifecycle_run_differences_see_more_than_the_interface():
    example = load("surrogate_lifecycle")
    surrogate = example.build_surrogate()
    model = example.solve(example.DEMO, surrogate)
    other = example.solve(example.DEMO, surrogate)
    accessor = other.getTransformedState
    other.getTransformedState = lambda time: [state + 1.0e-3 for state in accessor(time)]  # profiles only
    differences = example.run_differences(model, other)
    assert differences["interface"] == 0.0 and differences["profiles"] == pytest.approx(1.0e-3)


def test_surrogate_lifecycle_refuses_a_bundle_without_its_manifest(tmp_path, monkeypatch):
    example = load("surrogate_lifecycle")
    bundle_path = tmp_path / "bundle"
    example.prepare_bundle(bundle_path)
    (bundle_path / "manifest.json").unlink()
    before = {path.name: path.read_bytes() for path in bundle_path.iterdir()}
    monkeypatch.setattr(example, "build_surrogate", lambda: pytest.fail("A folder without a manifest must not be rebuilt"))
    with pytest.raises(example.SurrogateArtifactIntegrityError, match="no manifest.json(.|\n)*REBUILD"):
        example.prepare_bundle(bundle_path)
    assert {path.name: path.read_bytes() for path in bundle_path.iterdir()} == before
    monkeypatch.undo()
    assert example.prepare_bundle(bundle_path, rebuild=True)[1] == "built"


def test_surrogate_lifecycle_damaged_bundle_says_how_to_rebuild(tmp_path):
    example = load("surrogate_lifecycle")
    bundle, _ = example.prepare_bundle(tmp_path / "bundle")
    model_file = bundle.path / bundle.manifest["members"]["model"]["path"]
    data = bytearray(model_file.read_bytes())
    data[len(data) // 2] ^= 0xFF
    model_file.write_bytes(bytes(data))
    with pytest.raises(example.SurrogateArtifactIntegrityError, match="REBUILD"):
        example.prepare_bundle(tmp_path / "bundle")


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
    full = np.column_stack((points, 1.0 - points.sum(axis=1)))
    assert np.all(full >= 1.0e-5 - 1.0e-12)
    # All three components are treated alike: any swap of components maps the set onto itself.
    rows = {tuple(np.round(row, 9)) for row in full}
    for order in ((1, 0, 2), (2, 1, 0), (0, 2, 1)):
        assert {tuple(np.round(row, 9)) for row in full[:, order]} == rows
    # An edge point gives up the floor equally from its other two components.
    assert np.allclose(full[1], [1.0e-5, 0.1 - 0.5e-5, 0.9 - 0.5e-5], rtol=0.0, atol=1.0e-12)


def test_fecrni_build_spec_records_package_version_and_production_settings(tmp_path):
    example = load("ternary_fecrni_lee1996")
    (tmp_path / "db.tdb").write_bytes(b"ELEMENT A\n")
    spec = example.build_spec(example.LAGGED_FAST, tmp_path / "db.tdb")
    assert spec["package"]["moving_boundary"] == importlib.metadata.version("moving-boundary")
    production = example.CONFIGS["PRODUCTION_B"]
    assert (production.terminal_thin_phase_policy, production.compare_frozen) == ("continue", False)
    assert example.LAGGED_FAST.terminal_thin_phase_policy == "raise"


def test_fecrni_check_limits_reports_pass_and_fail():
    import dataclasses

    example = load("ternary_fecrni_lee1996")
    limits = example.LAGGED_THRESHOLDS
    requested = len(example.simplex_bulk_points(example.LAGGED_FAST.bulk_spacing))
    phase = {"retained_samples": requested, "accepted_outside_hull": 0, "attempted_queries": 5,
             "attempted_outside_hull": 1, "matrix_spread": 2 * limits["matrix_spread_min"],
             "simplices_crossed": limits["simplices_crossed_min"], "interpolation_vs_nearest_median": 0.0,
             "centroids_within_vertex_range": 1.0, "training_point_relative_error": 0.0}
    metrics = {"inventory_rel_max": 0.0, "workflow": {"interpolation": "nearest", **{name: dict(phase) for name in example.PHASES}}}
    rows = {name: passed for name, _, _, passed in
            example.check_limits(example.ExampleResult(example.LAGGED_FAST, None, None, metrics))}
    assert "lagged_vs_frozen" not in rows
    assert rows["inventory_rel_max"] and rows["BCC_A2 simplices_crossed"] and rows["FCC_A1 matrix_spread"]
    assert rows["BCC_A2 retained_fraction"] and rows["FCC_A1 accepted_outside_hull"]
    assert not rows["FCC_A1 interpolation_vs_nearest_median"]
    assert not rows["BCC_A2 attempted_outside_hull"] and not rows["interpolation"]
    frozen = {name: 2 * limit for name, limit in example.FROZEN_ANALYTIC_LIMITS.items()}
    result = example.ExampleResult(example.FROZEN_ANALYTIC, None, None, frozen)
    assert not any(passed for *_, passed in example.check_limits(result))
    # No validated limits: other frozen runs, the untested production settings and modified copies.
    for config in (example.FROZEN_FAST, example.PRODUCTION_B, dataclasses.replace(example.LAGGED_FAST, duration=1.0)):
        assert example.check_limits(example.ExampleResult(config, None, None, metrics)) == []


def test_fecrni_refuses_a_bundle_without_its_manifest(tmp_path, monkeypatch):
    example = load("ternary_fecrni_lee1996")
    (tmp_path / "db.tdb").write_bytes(b"ELEMENT A\n")
    folder = tmp_path / "out" / example.LAGGED_FAST.name
    folder.mkdir(parents=True)
    (folder / "model.npz").write_bytes(b"partial")
    monkeypatch.setattr(example, "sample_surrogate", lambda *_: pytest.fail("A folder without a manifest must not be rebuilt"))
    with pytest.raises(example.SurrogateArtifactIntegrityError, match="no manifest.json(.|\n)*REBUILD"):
        example.prepare_surrogate(example.LAGGED_FAST, tmp_path / "db.tdb", output_dir=tmp_path / "out")
    assert (folder / "model.npz").read_bytes() == b"partial"


@requires_database
def test_live_lagged_workflow_uses_its_data(tmp_path):
    """Build the simplex surrogate live and complete the real lagged workflow.

    These are workflow properties, not accuracy. Conservation and reload can both pass
    without the bulk path being used, so this also checks hull coverage from the retained
    samples, real matrix variation over accepted states, and interpolation evidence. The
    checks are the example's own ``check_limits``, so the console report and this test agree.
    """
    example = load("ternary_fecrni_lee1996")
    config = example.LAGGED_FAST
    result = example.run(config, output_dir=tmp_path)
    assert result.metrics["surrogate_action"] == "built"
    assert result.model.currentTime == pytest.approx(config.duration)
    rows = example.check_limits(result)
    names = {name for name, *_ in rows}
    assert {"interpolation", "lagged_vs_frozen", "BCC_A2 attempted_outside_hull", "FCC_A1 retained_fraction"} <= names
    failed = [row for row in rows if not row[3]]
    assert not failed, failed


@requires_database
def test_fresh_and_reloaded_lagged_solves_match(tmp_path):
    example = load("ternary_fecrni_lee1996")
    config = example.LAGGED_FAST
    bundle, action = example.prepare_surrogate(config, None, output_dir=tmp_path)
    assert action == "built"
    reloaded, action = example.prepare_surrogate(config, None, output_dir=tmp_path)
    assert action == "reloaded"
    assert reloaded.manifest["build_fingerprint"] == bundle.manifest["build_fingerprint"]
    models = []
    for surrogate in (bundle.surrogate, reloaded.surrogate):
        model = example.build_model(config, surrogate)
        model.solve(config.duration, minDtFrac=config.min_dt_frac)
        models.append(model)
    # Times, interface, eta, inventory and both phases' states, via the lifecycle example's helper.
    differences = load("surrogate_lifecycle").run_differences(*models)
    assert all(value == 0.0 for value in differences.values()), differences


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


# ---- 6. open Fe-Cr-Ni surrogate workflow ---------------------------------------

OPEN = "surrogate_workflow_fecrni_open"
# Runs the example in a child that refuses tc_python, then reports what it saw as JSON.
BLOCKED_TC_DRIVER = r"""
import importlib.abc, json, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "tc_python" or name.startswith("tc_python."):
            raise ModuleNotFoundError("tc_python blocked for this test", name=name)
        return None
sys.meta_path.insert(0, Block())
sys.argv = [sys.argv[1]] + sys.argv[2:]
namespace = runpy.run_path(sys.argv[0], run_name="__main__")
metrics = namespace["result"].metrics
print("RESULT" + json.dumps({"tc_python": "tc_python" in sys.modules,
                             "tc_provider": "moving_boundary.providers.thermocalc" in sys.modules,
                             "keys": sorted(metrics), "pycalphad": sorted(metrics["pycalphad"]),
                             "thermocalc": sorted(metrics["thermocalc"]), "comparison": metrics["comparison"],
                             "thermocalc_action": metrics["thermocalc"]["action"],
                             "simulations": [metrics[k]["simulation"]["final_time"] for k in ("pycalphad", "thermocalc")]},
                            default=float))
"""


def _run_blocked(arguments, cwd):
    """Run the example's command line in a child with tc_python blocked and this process's import path."""
    import subprocess
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p))
    return subprocess.run([sys.executable, "-c", BLOCKED_TC_DRIVER, str(EXAMPLES / (OPEN + ".py")), *map(str, arguments)],
                          capture_output=True, text=True, timeout=600, cwd=cwd, env=env)


@pytest.fixture(scope="module")
def open_workflow(tmp_path_factory):
    """One real pycalphad run of the open workflow on the bundled database, shared by the tests below."""
    example = load(OPEN)
    output = tmp_path_factory.mktemp("open_fecrni")
    return example, output, example.run(example.WORKFLOW, output_dir=output)


def _pycalphad_surrogate(example, output):
    """The pycalphad surrogate of the shared run, strictly reloaded from its bundle."""
    return example.prepare_pycalphad_bundle(example.WORKFLOW, None, output_dir=output)[0].surrogate


def _relabelled_thermocalc_bundle(example, surrogate, folder, **spec_changes):
    """Publish pycalphad data relabelled with Thermo-Calc composition-set names.

    Synthetic: the data are not Thermo-Calc-derived. The spec carries the fields a saved
    Thermo-Calc bundle is checked against offline.
    """
    saved = folder / "relabel.npz"
    surrogate.save(saved)
    with np.load(saved, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    labels = dict(zip(example.PHASES, example.TC_PHASES))
    arrays["phases"] = np.asarray([labels[p] for p in arrays["phases"].tolist()])
    arrays["tieline_phases"] = np.asarray([labels[p] for p in arrays["tieline_phases"].tolist()])
    metadata = json.loads(str(arrays["metadata_json"].tolist()))
    if metadata.get("precipitate_phase") in labels:
        metadata["precipitate_phase"] = labels[metadata["precipitate_phase"]]
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(saved, **arrays)
    relabelled = type(surrogate).load(saved)
    saved.unlink()
    config = dataclasses.replace(example.WORKFLOW, **spec_changes)
    spec = {**example.shared_spec(config, example.DATABASE, example.TC_PHASES),
            "source": {"engine": "thermocalc", "synthetic_test_bundle": "relabelled pycalphad data"}}
    return example.SurrogateArtifactBundle.publish(folder / "thermocalc", relabelled, spec)


def test_open_fecrni_bundled_database_is_the_approved_excerpt():
    example = load(OPEN)
    assert hashlib.sha256(example.DATABASE.read_bytes()).hexdigest() == example.DATABASE_SHA256
    assert example.resolve_database() == example.DATABASE


def test_open_fecrni_nonexistent_database_is_actionable(tmp_path):
    example = load(OPEN)
    with pytest.raises(FileNotFoundError, match="--tdb PATH or set MB_OPEN_FECRNI_TDB"):
        example.resolve_database(tmp_path / "missing.tdb")


def test_open_fecrni_workflow_runs_on_the_bundled_database(open_workflow):
    """The real pycalphad path at the reviewed settings. These are workflow properties, not accuracy.

    Training queries must reproduce the stored data; held-out queries are only required to
    exist and be evaluated, since no accuracy limit has been validated for them.
    """
    example, output, result = open_workflow
    metrics = result.metrics["pycalphad"]
    assert metrics["action"] == "built" and metrics["saved_bundle"]["tie_lines"] == 17
    for phase in example.PHASES:
        loo = metrics["saved_bundle"]["leave_one_out"][phase]
        assert loo["sample_count"] == 20 and loo["failure_count"] == 0 and np.isfinite(loo["max_matrix_relative_error"])
        tie = metrics["fresh_queries"]["tie_lines"]["endpoint_error"][phase]
        assert (tie["training"]["count"], tie["held_out"]["count"]) == (2, 6)
        assert tie["training"]["max"] <= 1.0e-12 and tie["held_out"]["max"] > 0.0
        interface = metrics["fresh_queries"]["interface_matrices"][phase]
        assert (interface["training"]["count"], interface["held_out"]["count"]) == (2, 4)
        assert interface["training"]["max"] <= 1.0e-12 and interface["held_out"]["max"] > 0.0
        bulk = metrics["fresh_queries"]["bulk_matrices"][phase]
        assert (bulk["training"]["count"], bulk["held_out"]["count"], bulk["invalid_predictions"]) == (0, 9, 0)
    assert metrics["fresh_queries"]["failures"] == 0 and metrics["fresh_queries"]["tie_lines"]["failures"] == 0
    simulation = metrics["simulation"]
    assert simulation["final_time"] == pytest.approx(example.WORKFLOW.duration)
    assert simulation["inventory_rel_max"] < 1.0e-8
    assert simulation["final_state_outside_hull"] == {phase: 0 for phase in example.PHASES}
    assert "comparison" not in result.metrics and "thermocalc" not in result.metrics
    figures = example.figures(result)
    assert {"pycalphad leave-one-out BCC_A2", "pycalphad leave-one-out FCC_A1", "pycalphad tie lines"} <= set(figures)
    assert all(figure.data for figure in figures.values())
    bundle, action = example.prepare_pycalphad_bundle(example.WORKFLOW, None, output_dir=output)
    assert action == "reloaded" and bundle.manifest["build_fingerprint"] == metrics["build_fingerprint"]


def test_open_fecrni_saved_thermocalc_bundle_is_reused_without_tc_python(open_workflow):
    """Saved-bundle reuse with tc_python blocked: diagnostics, mapped comparison and simulation.

    The default fresh pycalphad queries run; Thermo-Calc is never imported or queried. The
    'Thermo-Calc' bundle is relabelled pycalphad data, so the mapped comparison must find no
    difference at all.
    """
    example, output, result = open_workflow
    _relabelled_thermocalc_bundle(example, _pycalphad_surrogate(example, output), output)
    completed = _run_blocked(["--output-dir", output, "--thermocalc", "saved"], output)
    assert completed.returncode == 0, completed.stderr[-4000:]
    seen = json.loads(completed.stdout.split("RESULT", 1)[1])
    assert seen["tc_python"] is False and seen["tc_provider"] is False
    assert "fresh_queries" in seen["pycalphad"] and "fresh_queries" not in seen["thermocalc"]
    assert seen["thermocalc_action"] == "loaded saved" and seen["simulations"] == [3600.0, 3600.0]
    comparison = seen["comparison"]
    assert comparison["phase_labels"] == {"BCC_A2": "BCC_A2#1", "FCC_A1": "FCC_A1#1"}
    assert comparison["max_endpoint_displacement"] == 0.0 and comparison["interface_cross_shared_fraction"] == 1.0
    for values in ("interface_own_paths_max", "interface_at_reference_compositions_max", "bulk_shared_max"):
        assert all(value == 0.0 for value in comparison[values].values()), comparison
    assert comparison["bulk_coverage"]["reference only"] == comparison["bulk_coverage"]["candidate only"] == 0


@pytest.mark.parametrize("arguments", [["--thermocalc", "build"], ["--thermocalc", "saved", "--tc-fresh-queries"],
                                       ["--tc-fresh-queries"]])
def test_open_fecrni_live_thermocalc_requests_stop_before_any_work(tmp_path, arguments):
    output = tmp_path / "out"
    completed = _run_blocked(["--output-dir", output, *arguments], tmp_path)
    assert completed.returncode == 2
    assert "TC-Python" in completed.stderr and "licence" in completed.stderr
    assert not output.exists()


def test_open_fecrni_missing_or_incompatible_saved_thermocalc_bundle(open_workflow, tmp_path):
    example, output, _ = open_workflow
    with pytest.raises(FileNotFoundError, match="--thermocalc build"):
        example.load_saved_thermocalc_bundle(tmp_path / "none", example.WORKFLOW, None)
    _relabelled_thermocalc_bundle(example, _pycalphad_surrogate(example, output), tmp_path, samples_per_side=4)
    with pytest.raises(example.SurrogateArtifactCompatibilityError, match="tie_lines.*--thermocalc build"):
        example.load_saved_thermocalc_bundle(tmp_path / "thermocalc", example.WORKFLOW, None)


def test_open_fecrni_thermocalc_cache_must_be_short_and_separate():
    """Only paths are resolved here; nothing is created."""
    import tempfile

    example = load(OPEN)
    base = Path(tempfile.gettempdir())
    assert example.resolve_tc_cache(base / "mbtc", base / "mbout") == (base / "mbtc").resolve()
    with pytest.raises(ValueError, match="at most 150 characters"):
        example.resolve_tc_cache(base / ("x" * 160), base / "mbout")
    with pytest.raises(ValueError, match="outside the bundle folder"):
        example.resolve_tc_cache(base / "mbout" / "tc", base / "mbout")


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


def _settings_tool():
    spec = importlib.util.spec_from_file_location("example_settings", Path(__file__).parents[1] / "tools/example_settings.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


def test_bookkeeping_attributes_added_after_capture_are_ignored():
    """A reference captured before the R07/R08 bookkeeping fields existed still matches, whatever their values."""
    settings = _settings_tool()
    example = load("binary_planar_growth")
    arguments = dict(minDtFrac=1e-14)
    model = example.build_model(example.FAST)
    package = settings.effective_settings(model, example.FAST.duration, arguments)
    assert set(settings.BOOKKEEPING) <= set(package["attributes"])
    older = dict(package, attributes={k: v for k, v in package["attributes"].items() if k not in settings.BOOKKEEPING})
    assert settings.differences(older, package) == []
    model._outerMaxTimeStep, model._solveMaxDtFrac, model._awaitingFreshCandidate = 3.0, 0.5, True
    model._lastImplicitResidual, model._lastImplicitResidualScale = 1.0, 2.0
    assert settings.differences(package, settings.effective_settings(model, example.FAST.duration, arguments)) == []


@pytest.mark.parametrize(("attribute", "value"), [
    ("tolerance", 1.0e-8), ("residualTolerance", 1.0e-10), ("interfaceResidualCompositionScale", 1.0e-6),
    ("timeStep", 1.0e-5), ("dtMode", "semi_log"), ("semiLog_dt", 0.1), ("semiLogT0", 1.0e-6),
    ("maxIterations", 7), ("bulkPicardRtol", 1.0e-8), ("_unlistedPrivateSetting", 1.0),
])
def test_genuine_settings_are_still_compared(attribute, value):
    """Tolerance, residual and timestep settings, and any private attribute not named as bookkeeping, still fail."""
    settings = _settings_tool()
    example = load("binary_planar_growth")
    arguments = dict(minDtFrac=1e-14)
    baseline = settings.effective_settings(example.build_model(example.FAST), example.FAST.duration, arguments)
    changed = example.build_model(example.FAST)
    setattr(changed, attribute, value)
    assert settings.differences(baseline, settings.effective_settings(changed, example.FAST.duration, arguments)) == [
        "attributes." + attribute]


def test_changed_solve_arguments_are_still_compared():
    """The outer timestep fractions are compared through the recorded solve arguments, not their copies."""
    settings = _settings_tool()
    example = load("binary_planar_growth")
    model = example.build_model(example.FAST)
    baseline = settings.effective_settings(model, example.FAST.duration, dict(minDtFrac=1e-14))
    assert settings.differences(baseline, settings.effective_settings(model, example.FAST.duration, dict(minDtFrac=1e-14, maxDtFrac=0.1))) == ["solve"]
    assert settings.differences(baseline, settings.effective_settings(model, 2 * example.FAST.duration, dict(minDtFrac=1e-14))) == ["duration"]


# ---- notebook use ------------------------------------------------------------

@pytest.mark.parametrize("name", ["binary_planar_growth", "binary_tlp_bonding", "ternary_interstitial_similarity", "surrogate_lifecycle", "three_phase_similarity"])
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
        differences = namespace["run_differences"](namespace["model"], namespace["fresh"])
        assert all(value == 0.0 for value in differences.values()), differences
    else:
        assert namespace["result"].config == namespace["CONFIG"]


@pytest.mark.parametrize("name", ["binary_planar_growth", "binary_tlp_bonding", "ternary_interstitial_similarity", "three_phase_similarity"])
def test_example_command_line_still_parses_options(name, monkeypatch):
    example = load(name)
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    monkeypatch.setattr(sys, "argv", [name + ".py", "--fast"])
    config, plot_figures = example.settings_from_command_line(example.DEMO, True)
    assert config == example.FAST and plot_figures is False
    monkeypatch.setattr(sys, "argv", [name + ".py", "--f=kernel.json"])
    with pytest.raises(SystemExit):
        example.settings_from_command_line(example.DEMO, True)
