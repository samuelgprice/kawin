"""Package-owned baselines for registered new cases (C33/F33): capture, install, verify (pure parts).

``acceptance.py baseline`` orchestrates; this module holds everything that does not run Git or
workers, so it can be tested on synthetic files.

Contract (plan "Baseline capture", D2 = Option S):

- Only cases listed in the manifest's ``package_baselines`` are accepted, and capture refuses a
  case that already has an ``expected_artifacts`` entry (``check_registration``). Research cases
  are never captured here and their ``reference/`` files are never written.
- A capture writes only a new staging folder ``baseline-staging/<capture-id>.partial/``: the two
  fixture files per case, re-hashed and reloaded, then ``capture.json`` last, then one rename to
  ``<capture-id>/`` (``write_capture``). An id is never reused; a ``.partial`` folder or a complete
  folder no manifest entry names is inert (``leftovers``).
- Fixture content separates deterministic scientific content from run records: the worker report
  minus ``STRIPPED_FIELDS`` (timing and the local import path) plus the worker's npz as written.
  The stripped values go into the capture record. ``scientific_sha256`` digests the npz arrays and
  the retained report fields canonically, so a regenerated run can be compared even where the
  original files are absent.
- Installation into ``reference/`` is the documented exclusive, hash-checked copy
  (``install_capture``) from one named completed staging folder and its original ``capture.json``:
  each file goes to a capture-specific temporary sibling, is verified against that record, and is
  published without any possibility of overwriting a destination. Recovery after an interruption
  at any point repeats it against the same folder. Original-file hashes are always validated by
  ``acceptance.validate_expected``; regeneration (``verify_identity``) never replaces a fixture.

Documented install (from the package directory, after the capture's review):

    python -c "import runpy; m = runpy.run_path('tools/acceptance_baseline.py'); print(m['install_capture'](
        '.verification/acceptance/baseline-staging/<capture-id>', '.verification/acceptance/reference'))"
"""
from pathlib import Path
import hashlib
import json
import os

STRIPPED_FIELDS = ("elapsed_seconds", "kawin_origin", "wall_seconds")
STAGING = "baseline-staging"
CAPTURE_RECORD = "capture.json"


class BaselineError(RuntimeError):
    """A baseline request or file failed a check; nothing was written for it."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered(manifest):
    """The ``package_baselines`` case entries (empty when the section is absent)."""
    return (manifest.get("package_baselines") or {}).get("cases") or {}


def check_registration(manifest, cases, *, verify=False):
    """Refuse unregistered cases; capture refuses existing entries, verify requires package entries."""
    if not cases:
        raise BaselineError("baseline needs --cases")
    entries, expected = registered(manifest), manifest["expected_artifacts"]["cases"]
    for case in cases:
        if case not in entries:
            raise BaselineError(f"{case}: not a registered package baseline (package_baselines); "
                                "research cases are never captured here")
        entry = expected.get(case)
        if not verify and entry is not None:
            raise BaselineError(f"{case}: already has an expected_artifacts entry; a baseline is never "
                                "recaptured (use --verify)")
        if verify and (entry is None or entry.get("captured_role") != "package" or "scientific_sha256" not in entry):
            raise BaselineError(f"{case}: no package-captured expected_artifacts entry to verify")


def fixture_report(report):
    """The deterministic part of a worker report: PASS, unseeded, without ``STRIPPED_FIELDS``."""
    if report.get("status") != "PASS":
        raise BaselineError(f"{report.get('case')}: worker status {report.get('status')}")
    if report.get("hash_seed") is not None or report.get("numpy_seed_override") is not False:
        raise BaselineError(f"{report.get('case')}: baseline runs must be unseeded")
    return {key: value for key, value in report.items() if key not in STRIPPED_FIELDS}


def fixture_json_bytes(stripped):
    """The fixture report as written (worker formatting: ``json.dumps(indent=2)``, UTF-8)."""
    return json.dumps(stripped, indent=2).encode("utf-8")


def scientific_sha256(stripped, arrays):
    """Canonical digest of retained report fields plus every array's key, dtype, shape and bytes."""
    import numpy as np
    digest = hashlib.sha256()
    digest.update(json.dumps({k: v for k, v in stripped.items() if k not in STRIPPED_FIELDS},
                             sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)]).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def independent_checks(arrays, bounds, origin):
    """Fixed source bounds, independent of any limit derived from the candidate itself.

    Analytic: max |eta error|, max |profile error| (X and Y), |k_rel error|, and max |interface
    error| / max exact displacement from the similarity ``origin``. Conservation (the absolute
    source form): |I_k(t_end) - I_k(0)| per component, in m.
    """
    import numpy as np
    def get(name):
        if name not in arrays:
            raise BaselineError(f"{name}: array missing")
        value = np.asarray(arrays[name], dtype=float)
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise BaselineError(f"{name}: empty or non-finite")
        return value
    interface = get("exact/interface_error")
    exact_positions = get("interfaceData/values") - interface
    displacement = float(np.max(np.abs(exact_positions - float(origin))))
    inventory = get("inventoryData/values")
    values = dict(eta_max_abs=float(np.max(np.abs(get("exact/eta_error")))),
                  profile_max_abs=float(max(np.max(np.abs(get("exact/profile_error_X"))),
                                            np.max(np.abs(get("exact/profile_error_Y"))))),
                  k_rel_abs=float(np.max(np.abs(get("exact/k_rel_error")))),
                  interface_max_abs_over_displacement=float(np.max(np.abs(interface))) / displacement,
                  final_inventory_drift_abs=float(np.max(np.abs(inventory[-1] - inventory[0]))))
    failures = sorted(key for key, value in values.items() if not value <= float(bounds[key]))
    return dict(values=values, bounds={key: float(bounds[key]) for key in values}, failures=failures,
                passed=not failures, displacement=displacement)


RESIDUAL_BUDGET_LIMITATIONS = (
    "diagnostic only: not a rigorous conservation bound and not a correctness gate",
    "uses the recorded accepted inventories, not the solver's internal per-iteration values",
    "assumes every accepted step met the interface residual criterion and that the residual equals "
    "the step's inventory change exactly (the solver states 'up to round-off'); no floating-point term",
    "worst case: assumes every step's permitted defect has the same sign and accumulates",
)


def residual_budget_diagnostic(arrays, inputs):
    """Diagnostic D: approximate accumulated residual-tolerance budget against the observed drift.

    ``D_k = sum_{n=1..N} tol * s_k(n)`` in m (composition x length, the inventory unit), with
    ``tol`` the effective residual tolerance and ``s_k(n)`` the solver's per-step scale evaluated
    post hoc from the accepted inventory at the *start* of step ``n`` (recorded value ``n-1``):
    ``max(|I_k|, R * minComposition)`` for the default (R13) scale, or ``max(|I_k|, R * s_k,
    1e-300)`` for an explicit composition scale. It is compared with the observed history drift
    ``max_t |I_k(t) - I_k(0)|`` and final drift ``|I_k(t_end) - I_k(0)|``. ``inputs`` are the
    worker's ``residual_budget_inputs``. Raises ``BaselineError`` when it cannot be computed as
    specified (inputs missing, or not every step recorded); see ``RESIDUAL_BUDGET_LIMITATIONS``.
    """
    import numpy as np
    if not inputs:
        raise BaselineError("residual_budget_inputs missing; diagnostic D cannot be computed as specified")
    if int(inputs["inventory_record_interval"]) != 1:
        raise BaselineError("inventory not recorded every step; diagnostic D cannot be computed as specified")
    inventory = np.asarray(arrays["inventoryData/values"], dtype=float).reshape(-1, 2)
    if inventory.shape[0] < 2:
        raise BaselineError("fewer than two inventory records; diagnostic D cannot be computed")
    old, length = np.abs(inventory[:-1]), float(inputs["domain_length"])
    if inputs["explicit_composition_scale"] is None:
        scale, expression = np.maximum(old, length * float(inputs["min_composition"])), "max(|I_k|, R * minComposition)"
    else:
        explicit = np.broadcast_to(np.asarray(inputs["explicit_composition_scale"], dtype=float), (2,))
        scale, expression = np.maximum(np.maximum(old, length * explicit), 1e-300), "max(|I_k|, R * s_k, 1e-300)"
    allowance = float(inputs["residual_tolerance"]) * scale.sum(axis=0)
    observed_max = np.max(np.abs(inventory - inventory[0]), axis=0)
    observed_final = np.abs(inventory[-1] - inventory[0])
    return dict(name="D (approximate accumulated residual-tolerance budget)", gate=False,
                formula=f"D_k = sum over accepted steps n of tol * s_k(n), s_k = {expression} at the step-start inventory",
                units="m (composition x length; inventory unit)", steps=int(inventory.shape[0] - 1),
                residual_tolerance=float(inputs["residual_tolerance"]), allowance=allowance.tolist(),
                observed_max_drift=observed_max.tolist(), observed_final_drift=observed_final.tolist(),
                observed_max_over_allowance=(observed_max / allowance).tolist(),
                limitations=list(RESIDUAL_BUDGET_LIMITATIONS))


def new_capture_id(root, head, now):
    """``<UTC>-<sha7>-<n>`` never used before by a complete or partial folder under ``root``."""
    base = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{head[:7]}"
    n = 1
    while (Path(root) / f"{base}-{n}").exists() or (Path(root) / f"{base}-{n}.partial").exists():
        n += 1
    return f"{base}-{n}"


def _write_new(path, data):
    """Write ``data`` to a path that must not exist (``"xb"``), flushed and fsynced (also used by ``acceptance_transfer``)."""
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_capture(root, capture_id, files, record, fault=None):
    """Write ``files`` (``{case: {"json": bytes, "npz": bytes}}``) as capture ``capture_id``.

    Steps: create ``<id>.partial`` (refusing any existing folder of either name), write every file
    exclusively, re-hash and reload them, write ``capture.json`` (``record`` plus file hashes)
    last, rename to ``<id>``. ``fault(step)`` is a test hook called after each step. Returns
    ``(folder, hashes)``.
    """
    import numpy as np
    fault = fault or (lambda step: None)
    root = Path(root)
    final, partial = root / capture_id, root / (capture_id + ".partial")
    if final.exists() or partial.exists():
        raise BaselineError(f"capture {capture_id} already exists; ids are never reused")
    root.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    hashes = {}
    for case, pair in sorted(files.items()):
        for suffix in ("json", "npz"):
            data = pair[suffix]
            _write_new(partial / f"{case}.{suffix}", data)
            hashes.setdefault(case, {})[suffix] = hashlib.sha256(data).hexdigest()
    fault("written")
    for case, pair in hashes.items():
        for suffix, digest in pair.items():
            path = partial / f"{case}.{suffix}"
            if _sha(path) != digest:
                raise BaselineError(f"{path.name}: re-hash mismatch")
        json.loads((partial / f"{case}.json").read_text(encoding="utf-8"))
        with np.load(partial / f"{case}.npz", allow_pickle=False) as data:
            [data[key] for key in data.files]
    fault("validated")
    document = dict(record, capture_id=capture_id, files=hashes)
    _write_new(partial / CAPTURE_RECORD, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    fault("recorded")
    if final.exists():
        raise BaselineError(f"capture {capture_id} appeared during the write; not replaced")
    os.rename(partial, final)
    fault("renamed")
    return final, hashes


def referenced_captures(manifest):
    """Capture ids named by ``expected_artifacts`` entries (the only captures any gate may use)."""
    return {entry["capture_id"] for entry in manifest["expected_artifacts"]["cases"].values() if "capture_id" in entry}


def leftovers(root, manifest):
    """Inert staging folders: ``.partial`` ones and complete ones no manifest entry names."""
    root = Path(root)
    if not root.is_dir():
        return dict(partial=[], unreferenced=[])
    folders = sorted(path.name for path in root.iterdir() if path.is_dir())
    named = referenced_captures(manifest)
    return dict(partial=[name for name in folders if name.endswith(".partial")],
                unreferenced=[name for name in folders if not name.endswith(".partial") and name not in named])


def _temporary_name(name, capture_id):
    """The one temporary sibling an install of ``name`` from ``capture_id`` may use (never read by gates)."""
    return f".{name}.{capture_id}.install-tmp"


def _publish(temporary, target):
    """Make ``temporary`` appear at ``target`` without ever overwriting an existing ``target``.

    A hard link fails with ``FileExistsError`` when ``target`` exists (no replace on any
    platform). Where hard links are unavailable, Windows ``os.rename`` is used, which also refuses
    an existing destination; elsewhere the error is raised rather than risking a replace.
    ``acceptance_transfer`` publishes restored files with this function too.
    """
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise
    except OSError:
        if os.name != "nt":
            raise
        os.rename(temporary, target)
        return
    os.unlink(temporary)


def install_capture(capture_dir, reference_dir, cases=None):
    """The documented install: exclusive, hash-checked, no-overwrite copy from one completed capture.

    Every staged file must match the folder's original ``capture.json`` before anything is copied.
    Each file is then written to its own temporary sibling (``_temporary_name``), re-hashed
    against that record, and published with ``_publish``, which never overwrites. A destination
    that already exists is left untouched if it equals the recorded hash (an earlier, interrupted
    install) and is an error otherwise. Recovery repeats this call against the same completed
    capture and record; it never needs a new capture. A leftover temporary of *this* capture and
    file (an interruption during its write or before its publication) is discarded and rewritten;
    no other file is touched. Returns ``{file name: "installed" | "already installed"}``.
    """
    capture_dir, reference_dir = Path(capture_dir), Path(reference_dir)
    if capture_dir.name.endswith(".partial") or not (capture_dir / CAPTURE_RECORD).is_file():
        raise BaselineError(f"{capture_dir.name}: not a completed capture")
    record = json.loads((capture_dir / CAPTURE_RECORD).read_text(encoding="utf-8"))
    if record.get("capture_id") != capture_dir.name:
        raise BaselineError(f"{capture_dir.name}: capture.json names {record.get('capture_id')}")
    selected = sorted(record["files"]) if cases is None else list(cases)
    for case in selected:                       # check every source before copying anything
        for suffix, digest in record["files"][case].items():
            if _sha(capture_dir / f"{case}.{suffix}") != digest:
                raise BaselineError(f"{case}.{suffix}: staged file does not match capture.json")
    status = {}
    for case in selected:
        for suffix in ("json", "npz"):
            digest, name = record["files"][case][suffix], f"{case}.{suffix}"
            target = reference_dir / name
            temporary = reference_dir / _temporary_name(name, capture_dir.name)
            if target.exists():
                if _sha(target) != digest:
                    raise BaselineError(f"reference/{name} exists and differs from capture {capture_dir.name}; not overwritten")
                if temporary.exists():
                    temporary.unlink()              # interrupted after publication: the copy is complete
                status[name] = "already installed"
                continue
            if temporary.exists():
                temporary.unlink()                  # interrupted during the write or before publication
            _write_new(temporary, (capture_dir / name).read_bytes())
            if _sha(temporary) != digest:
                raise BaselineError(f"{temporary.name}: temporary bytes do not match capture.json; not published")
            try:
                _publish(temporary, target)
            except FileExistsError:
                raise BaselineError(f"reference/{name} appeared during the install; not overwritten") from None
            if _sha(target) != digest:
                raise BaselineError(f"reference/{name}: published bytes do not match capture.json")
            status[name] = "installed"
    return status


def verify_identity(regenerated_report, regenerated_arrays, reference_report, reference_arrays, expected_digest=None):
    """Regenerated scientific identity: same array keys and bytes, equal retained report fields.

    ``STRIPPED_FIELDS`` are ignored on both sides. ``expected_digest`` (the committed
    ``scientific_sha256``) is checked when given. Byte identity of the files is not required here.
    """
    stripped = fixture_report(regenerated_report)
    reference = {k: v for k, v in reference_report.items() if k not in STRIPPED_FIELDS}
    keys = set(regenerated_arrays) == set(reference_arrays)
    differing = sorted(key for key in set(regenerated_arrays) & set(reference_arrays)
                       if not (regenerated_arrays[key].shape == reference_arrays[key].shape
                               and regenerated_arrays[key].dtype == reference_arrays[key].dtype
                               and regenerated_arrays[key].tobytes() == reference_arrays[key].tobytes()))
    fields = sorted(key for key in set(stripped) | set(reference) if stripped.get(key) != reference.get(key))
    digest = scientific_sha256(stripped, regenerated_arrays)
    digest_ok = expected_digest is None or digest == expected_digest
    return dict(keys_equal=keys, differing_arrays=differing, differing_fields=fields, scientific_sha256=digest,
                digest_equal=digest_ok if expected_digest is not None else None,
                passed=keys and not differing and not fields and digest_ok)
