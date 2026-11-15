"""Approval correctness criteria: analytic error against exact solutions, and conservation (M7, plan §5).

These are *correctness gates*, separate from reference comparisons: every case they cover must
meet them whether or not its references are being approved. The limits are fixed numbers in the
committed manifest section ``approval_criteria``, set once from the frozen references (user
approval, 2026-11-10); they are never recomputed from a previous output.

Metrics (``measure``), each computed from one run's arrays (a reference or package ``.npz``):

- ``max_abs``: largest |value| over the named arrays (analytic interface, eta and three-phase
  per-phase profile errors; two-phase C33/F33 per-component profile and fitted-k errors).
- ``max_abs_outside_nodes``: largest |value| of a binary planar profile error outside a *frozen*
  node range around the exact final interface (centre = exact final interface of the reference,
  half-width = approved interface-error limit + 2h on the uniform mesh). Inside it the exact and
  numerical solutions can sit on opposite sides of the composition jump, which the interface gate
  covers; the excluded nodes are stored, so a candidate's own interface never moves the mask.
- ``relative_drift``: max over time and components of |Q(t) - Q(0)| / max |Q(0)|, for the
  physically conserved quantity Q (binary transformed inventory ``concData``; composition
  inventory where molar volumes are equal; component moles and total moles for explicit volumes).
- ``closure_relative``: max |dependent-mole closure error| / |total moles(0)|.

A measurement that cannot be computed fails; it never passes: a missing array, an empty array or
mask, a non-finite value, fewer than two history points for a drift, a zero reference magnitude,
or a profile whose length differs from the frozen mesh each raise ``CriteriaError``.

Limit rules (``propose``): analytic = 2 x current, rounded up to two significant figures;
conservation = 10 x current, rounded up to two significant figures, never below 1e-13.

Inactive cases (``PENDING_ANALYTIC``; C33/F33 until their activation, now empty) are in no registry,
so ``propose`` and ``check`` never require their baselines; ``preview`` measures one fixture pair by
the same rules (``proposal_for``) and writes nothing. Activated package-owned cases (C33/F33) are
measured by ``propose``/``check`` from their hash-validated ``reference/`` baselines like every
other case.

    python tools/acceptance_criteria.py check     # evaluate the frozen references against the manifest
    python tools/acceptance_criteria.py propose OUT.json   # regenerate the proposal (historical record)
    python tools/acceptance_criteria.py preview --case CXX --json CXX.json --npz CXX.npz   # pending cases only
"""
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "tools/acceptance_manifest.json"
REFERENCE = PROJECT / ".verification/acceptance/reference"
CONSERVATION_FLOOR = 1e-13

ANALYTIC_CASES = {"C04": "binary", "C31": "binary", "F04": "binary",
                  "C11": "three_phase", "C12": "three_phase", "F11": "three_phase", "F12": "three_phase",
                  "C33": "two_phase", "F33": "two_phase"}   # C33/F33: package-captured baselines (package_baselines)
# Registered but inactive analytic cases: in no registry ``propose``/``check`` iterate, so their
# missing baselines affect nothing; ``preview`` alone measures them. Empty since C33/F33 were
# activated (moved into ANALYTIC_CASES, kind "two_phase").
PENDING_ANALYTIC = {}
TWO_PHASE_INVENTORY = ("composition inventory, relative normalized drift: max over t and k of |I_k(t) - I_k(0)| / "
                       "max_k |I_k(0)| (dimensionless); distinct from the absolute source bound "
                       "|I_k(t_end) - I_k(0)| <= 1e-9 m checked at baseline capture")
BINARY = ("C01", "C02", "C03", "C04", "C13", "C14", "C15", "C30", "C31", "F04", "O07")
TERNARY = ("C05", "C06", "C07", "C08", "C16", "C17", "C18", "C19", "C24", "O01", "O03a", "O04", "O05", "O06")
THREE_PHASE_IMPLICIT_EQUAL = ("C11", "C12", "F11", "F12", "O02", "O03b")   # no explicit volumes (Vm = 1)
THREE_PHASE_UNEQUAL = ("C09", "C10", "C20", "C21", "C22", "C32")           # phase_molar_volumes (2.0, 3.0, 1.5)
THREE_PHASE_EXPLICIT_EQUAL = ("C23",)                                     # (7.2e-6,) * 3
LEGACY_MODES = ("phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit")


class CriteriaError(ValueError):
    """A criterion could not be measured; it counts as a failure, never as a pass."""


def _finite(name, value):
    import numpy as np
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise CriteriaError(f"{name}: empty array")
    if not np.all(np.isfinite(array)):
        raise CriteriaError(f"{name}: {int((~np.isfinite(array)).sum())} non-finite values")
    return array


def _get(arrays, name):
    if name not in arrays:
        raise CriteriaError(f"{name}: array missing")
    return _finite(name, arrays[name])


def measure(arrays, spec):
    """Compute one criterion's value from ``arrays`` (mapping name -> array); see module docstring."""
    import numpy as np
    metric = spec["metric"]
    if metric == "max_abs":
        return max(float(np.max(np.abs(_get(arrays, name)))) for name in spec["arrays"])
    if metric == "max_abs_outside_nodes":
        (name,) = spec["arrays"]
        values = np.ravel(_get(arrays, name))
        if values.size != spec["nodes"]:
            raise CriteriaError(f"{name}: {values.size} nodes, frozen mask expects {spec['nodes']}")
        first, last = spec["excluded_nodes"]
        keep = np.ones(values.size, dtype=bool)
        keep[first:last + 1] = False
        if not keep.any():
            raise CriteriaError(f"{name}: empty profile mask")
        return float(np.max(np.abs(values[keep])))
    if metric == "relative_drift":
        drifts = []
        for name in spec["arrays"]:
            values = _get(arrays, name)
            values = values.reshape(values.shape[0], -1) if values.ndim else values.reshape(1, 1)
            if values.shape[0] < 2:
                raise CriteriaError(f"{name}: fewer than two history points")
            base = float(np.max(np.abs(values[0])))
            if base == 0:
                raise CriteriaError(f"{name}: zero initial magnitude")
            drifts.append(float(np.max(np.abs(values - values[0]))) / base)
        return max(drifts)
    if metric == "closure_relative":
        error, total = spec["arrays"]
        total0 = abs(float(np.ravel(_get(arrays, total))[0]))
        if total0 == 0:
            raise CriteriaError(f"{total}: zero initial total moles")
        return float(np.max(np.abs(_get(arrays, error)))) / total0
    raise CriteriaError("unknown metric: " + str(metric))


def evaluate(case, arrays, criteria):
    """All criteria registered for ``case``: list of dicts with value, limit and passed.

    A criterion that cannot be measured is reported with ``passed=False`` and its error.
    """
    results = []
    for name, spec in criteria_for(case, criteria):
        try:
            value = measure(arrays, spec)
            results.append(dict(case=case, criterion=name, value=value, limit=spec["limit"], passed=value <= spec["limit"]))
        except CriteriaError as exc:
            results.append(dict(case=case, criterion=name, value=None, limit=spec["limit"], passed=False, error=str(exc)))
    return results


def criteria_for(case, criteria):
    """``(name, spec)`` pairs registered for ``case`` (analytic first, then conservation)."""
    pairs = [(f"analytic/{key}", spec) for key, spec in criteria["analytic"].get(case, {}).items()]
    pairs += [(f"conservation/{key}", spec) for key, spec in criteria["conservation"].get(case, {}).items()]
    return pairs


def round_up(value, factor, floor=0.0):
    """``factor * value`` rounded up to two significant figures (exact decimal), at least ``floor``."""
    scaled = Decimal(repr(float(value))) * Decimal(repr(float(factor)))
    if scaled <= 0:
        return floor
    exponent = scaled.adjusted() - 1
    quantum = Decimal(1).scaleb(exponent)
    rounded = (scaled / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum
    return max(floor, float(rounded))


def _specs(case, report, kind=None):
    """Unlimited criterion specs for ``case`` (metric definitions only; limits added by ``propose``).

    ``kind`` defaults to the registered analytic kind (``ANALYTIC_CASES``); ``preview`` passes the
    pending kind of an inactive case. Two-phase (C33/F33): interface, eta, per-component profile
    and fitted-k errors, and the composition inventory labelled as a relative normalized drift.
    """
    analytic, conservation = {}, {}
    kind = ANALYTIC_CASES.get(case) if kind is None else kind
    if kind == "binary":
        analytic["interface"] = dict(metric="max_abs", arrays=["exact/interface_error"])
        analytic["profile"] = dict(metric="max_abs_outside_nodes", arrays=["exact/profile_error"])
    elif kind == "three_phase":
        analytic["interface"] = dict(metric="max_abs", arrays=["exact/interface_error"])
        analytic["eta"] = dict(metric="max_abs", arrays=["exact/eta_error"])
        analytic["profile"] = dict(metric="max_abs", arrays=[f"exact/profile_error{i}" for i in range(3)])
    elif kind == "two_phase":
        analytic["interface"] = dict(metric="max_abs", arrays=["exact/interface_error"])
        analytic["eta"] = dict(metric="max_abs", arrays=["exact/eta_error"])
        analytic["profile_X"] = dict(metric="max_abs", arrays=["exact/profile_error_X"])
        analytic["profile_Y"] = dict(metric="max_abs", arrays=["exact/profile_error_Y"])
        analytic["k"] = dict(metric="max_abs", arrays=["exact/k_rel_error"])
        conservation["composition_inventory"] = dict(metric="relative_drift", arrays=["inventoryData/values"],
                                                     quantity=TWO_PHASE_INVENTORY)
        return analytic, conservation
    if case == "LEGACY":
        for mode in LEGACY_MODES:
            conservation[f"inventory_{mode}"] = dict(metric="relative_drift", arrays=[f"binary_{mode}__concData_y"],
                                                     quantity="binary transformed inventory (concData = inventory / L)")
    elif case in BINARY:
        conservation["inventory"] = dict(metric="relative_drift", arrays=["concData/values"],
                                         quantity="binary transformed inventory (concData = inventory / L)")
    elif case in TERNARY or case in THREE_PHASE_IMPLICIT_EQUAL:
        conservation["composition_inventory"] = dict(metric="relative_drift", arrays=["inventoryData/values"],
                                                     quantity="composition inventory (equal molar volumes)")
    elif case in THREE_PHASE_UNEQUAL + THREE_PHASE_EXPLICIT_EQUAL:
        conservation["component_moles"] = dict(metric="relative_drift", arrays=["molarInventoryData/values"],
                                               quantity="component moles (explicit molar volumes)")
        conservation["total_moles"] = dict(metric="relative_drift", arrays=["totalMolesData/values"],
                                           quantity="total moles (explicit molar volumes)")
        conservation["closure"] = dict(metric="closure_relative",
                                       arrays=["dependentMoleClosureErrorData/values", "totalMolesData/values"],
                                       quantity="dependent-mole closure error / total moles(0)")
        if case in THREE_PHASE_EXPLICIT_EQUAL:
            conservation["composition_inventory"] = dict(metric="relative_drift", arrays=["inventoryData/values"],
                                                         quantity="composition inventory (equal explicit volumes)")
    return analytic, conservation


def load_reference(case, expected):
    """A frozen reference pair, hash-validated against ``expected_artifacts`` first."""
    import numpy as np
    for suffix in ("json", "npz"):
        path = REFERENCE / f"{case}.{suffix}"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[case][suffix]:
            raise CriteriaError(f"{case}: reference/{path.name} does not match expected_artifacts")
    report = json.loads((REFERENCE / f"{case}.json").read_text(encoding="utf-8"))
    with np.load(REFERENCE / f"{case}.npz", allow_pickle=False) as data:
        return report, {key: data[key] for key in data.files}


def propose(expected):
    """Measure today's references and derive every limit by the approved rules (deterministic)."""
    criteria = dict(analytic={}, conservation={}, reported_only={})
    cases = sorted(set(ANALYTIC_CASES) | set(BINARY) | set(TERNARY) | set(THREE_PHASE_IMPLICIT_EQUAL)
                   | set(THREE_PHASE_UNEQUAL) | set(THREE_PHASE_EXPLICIT_EQUAL) | {"LEGACY"})
    for case in cases:
        report, arrays = load_reference(case, expected)
        analytic, conservation, reported = proposal_for(case, report, arrays)
        if reported is not None:
            criteria["reported_only"][case] = reported
        if analytic:
            criteria["analytic"][case] = analytic
        if conservation:
            criteria["conservation"][case] = conservation
    return criteria


def proposal_for(case, report, arrays, kind=None):
    """One case's criteria with measured ``current`` values and rule-derived limits.

    Returns ``(analytic, conservation, reported_only)`` (the last ``None`` unless the case has a
    reported-only quantity). ``propose`` uses it for every registered case; ``preview`` uses it
    with an explicit pending ``kind`` for an inactive case, without touching the manifest.
    """
    import numpy as np
    analytic, conservation = _specs(case, report, kind)
    if "interface" in analytic:
        current = measure(arrays, analytic["interface"])
        analytic["interface"].update(current=current, limit=round_up(current, 2))
    if analytic.get("profile", {}).get("metric") == "max_abs_outside_nodes":
        # Freeze the mask: centre on the exact final interface, half-width = interface limit + 2h.
        profile = np.ravel(_get(arrays, "exact/profile_error"))
        length = float(report["length_scale"])
        z = np.linspace(0.0, length, profile.size)    # CartesianFD1D(["CR"], [0, R], 401): uniform
        h = float(z[1] - z[0])
        exact_interface = float(np.ravel(_get(arrays, "interfaceData/values"))[-1]
                                - np.ravel(_get(arrays, "exact/interface_error"))[-1])
        half_width = analytic["interface"]["limit"] + 2 * h
        excluded = np.flatnonzero(np.abs(z - exact_interface) <= half_width)
        analytic["profile"].update(nodes=int(profile.size), excluded_nodes=[int(excluded[0]), int(excluded[-1])],
                                   mask=dict(centre="exact final interface", exact_interface=exact_interface,
                                             half_width=half_width, h=h, mesh=f"uniform {profile.size} nodes on [0, {length}]"))
    for key in ("profile", "eta", "profile_X", "profile_Y", "k"):
        if key in analytic:
            current = measure(arrays, analytic[key])
            analytic[key].update(current=current, limit=round_up(current, 2))
    for spec in conservation.values():
        current = measure(arrays, spec)
        spec.update(current=current, limit=round_up(current, 10, CONSERVATION_FLOOR))
    reported = None
    if case in THREE_PHASE_UNEQUAL:
        reported = dict(
            quantity="unweighted composition inventory (unequal molar volumes; not conserved)",
            relative_drift=measure(arrays, dict(metric="relative_drift", arrays=["inventoryData/values"])))
    return analytic, conservation, reported


def preview(case, json_path, npz_path):
    """Proposed criteria for an inactive (``PENDING_ANALYTIC``) case from one fixture pair; writes nothing.

    No hash check and no manifest access: the output is a proposal for review. After activation
    ``propose`` must reproduce it exactly from the hash-validated fixture.
    """
    import numpy as np
    if case not in PENDING_ANALYTIC:
        raise CriteriaError(f"{case}: not a pending (inactive) analytic case; registered cases use propose")
    report = json.loads(Path(json_path).read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    analytic, conservation, _ = proposal_for(case, report, arrays, kind=PENDING_ANALYTIC[case])
    return dict(case=case, kind=PENDING_ANALYTIC[case], analytic=analytic, conservation=conservation)


def main(argv=None):
    """``check``: evaluate the frozen references against the manifest; ``propose OUT``: regenerate."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("check", "propose", "preview"))
    parser.add_argument("out", nargs="?", type=Path)
    parser.add_argument("--case", help="preview only: an inactive PENDING_ANALYTIC case (none since C33/F33 activation)")
    parser.add_argument("--json", type=Path, help="preview only: the case's fixture report")
    parser.add_argument("--npz", type=Path, help="preview only: the case's fixture arrays")
    args = parser.parse_args(argv)
    if args.action == "preview":
        if args.out is not None or None in (args.case, args.json, args.npz):
            parser.error("preview takes --case, --json and --npz and writes nothing")
        print(json.dumps(preview(args.case, args.json, args.npz), indent=1, sort_keys=True))
        return 0
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["expected_artifacts"]["cases"]
    if args.action == "propose":
        if args.out is None:
            parser.error("propose needs an output path")
        args.out.write_text(json.dumps(propose(expected), indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return 0
    criteria = manifest["approval_criteria"]
    failed = []
    for case in sorted(set(criteria["analytic"]) | set(criteria["conservation"])):
        _, arrays = load_reference(case, expected)
        for result in evaluate(case, arrays, criteria):
            mark = "PASS" if result["passed"] else "FAIL"
            value = "n/a" if result["value"] is None else f"{result['value']:.4g}"
            print(f"{case} {result['criterion']}: {value} <= {result['limit']:.4g} {mark} {result.get('error', '')}".rstrip())
            if not result["passed"]:
                failed.append((case, result["criterion"]))
    print(f"{len(failed)} criteria failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
