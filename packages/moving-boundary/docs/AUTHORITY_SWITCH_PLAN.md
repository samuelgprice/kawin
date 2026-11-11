# Plan: package-owned verification, research oracle optional (items 7, 14-18)

Status: **plan; the six design decisions were approved 2026-11-10 (see Decisions). Implementation
has not started and needs the user's go-ahead.** Written 2026-11-10 at extraction `518162b`
(research main `7af24f4`, oracle worktree `mb-ref-7af24f4`). It replaces the approval-path part of
[AUTHORITY_SWITCH_PROPOSAL.md](AUTHORITY_SWITCH_PROPOSAL.md). Branch ledger:
`branch_status/refactor-package-extraction.md`, items 7 and 14-18.

## Progress

| Batch | Content | Status |
|---|---|---|
| M0 | Classification, materializer, `expected_artifacts` hashes | done 2026-11-10 (see log) |
| M1 | Definitions for C01-C23, C30-C32, LEGACY | done 2026-11-10 (see log) |
| M2 | C25-C29 assertions + item 7 per-model scales | done 2026-11-10 (scales approved by the user; see log) |
| M3 | C24 + `prepare` | done 2026-11-10 (see log) |
| M4 | F04/F11/F12, O01-O08 | done 2026-11-10 (see log) |
| M5 | Example worker + hashed settings | done 2026-11-10 (see log) |
| M5b | Regeneration verification of adopted inputs | done 2026-11-10 (12/12 verified; see log) |
| M6 | Normal/historical split, package-role repeatability | done 2026-11-10 (see log) |
| M7 | Approval path | done 2026-11-10: M7a `approval_criteria` adopted (user-approved); M7b approval path (see log) |
| M8 | Test port (item 16), 4 batches | done 2026-11-10: 301 of 314 research tests ported, 13 deferred (see log) |
| M9 | Independence proof + missing-resource matrix | done 2026-11-11: all green from Git-blocked exports (empty and active approved sets); E05 table row corrected (user-approved); editable install removed (see log) |
| M10 | Docs + final recorded gates | done 2026-11-11: all gates green and recorded (see log) |

### Implementation log

- **M0 (2026-11-10, uncommitted).**
  - *Historical repeat before hashing:* research role at `7af24f4`, scratch `--artifacts` copy,
    43/43 cases bit-identical with `reference/` (arrays, key sets, status/stop status, scales;
    268 s). O06/O08 were not repeated (hash-frozen). The research-role O05 resample reproduced
    `pycalphad-reference.npz` byte for byte in scratch.
  - *`expected_artifacts`:* 45 cases (`json` + `npz` sha256, capture role and oracle, how
    verified); O06/O08 entries equal their frozen sections. `recorded_at_extraction` is `518162b`.
  - *Materializer:* `tools/materialize_definitions.py` records today's `CaseInputs.load` AST and
    emits verbatim segments. Dry run: 36 definitions (24 top-level + 12 nested research-symbol
    loads), all AST-equal with **no normalization**. Emitted bodies keep `_input_symbol(...)` calls
    (resolved at run time by `_bindings`, without a root) instead of static imports, so today's
    fallback resolution is reproduced exactly. Unit tests: `tests/test_materialize_definitions.py` (2).
  - *Flag for M1:* the `illingworth2005_planar` closure computes `THIS_FILE`/`SCRIPT_DIR` from
    `__file__` at load (only plot/figure-data path strings). The loader sets each definitions
    module's `__file__` to its research-relative source key (never opened), so no fallback search
    reaches a checkout; exact C04/C31/F04 and E01/E02 settings equality guard it.
  - *Classification:* `tools/verify.py` is historical apart from `main()` (an alias for
    `acceptance.py quick`); `check_baseline` is used only by `tests/test_verification.py` on
    synthetic repositories. Research `test_surrogate.py` (9) tests kawin-core `kawin.thermo`
    surrogates, not solvers: out of scope. `test_surrogate_artifacts.py` (12) is M8 batch 1; its
    TC-gated `test_cuniti_tc_python_bundle_smoke` imports `examples.ThermoCalc.tc_python_adapter`
    (to resolve at M8). `test_fecrni_surrogate_artifact_recipe.py` (11) tests research study
    scripts through their source: deferred with FUTURE_PORTS entry 6 (recorded there).
  - *Unreferenced input:* `inputs/pycalphad.npz` (2026-11-05 18:51, 19 621 bytes, differs from
    `pycalphad-reference.npz`) is not adopted and was left untouched.
- **M1 (2026-11-10, uncommitted).**
  - 20 definition files (11 top-level entries, `ternary_two_phase_similarity_matrix` because M1
    definitions load it, and 8 nested), all AST-equal; `docs/definitions-provenance.json`.
    Stems and headers come from recording the whole table, so they don't depend on the batch.
  - `tools/definitions/_bindings.py` (`Context`, `SourceKey`); worker `PACKAGE_DEFINITIONS`
    covers C01-C23, C30-C32 and LEGACY in the package role and asserts that `acceptance_inputs`
    was not imported.
  - F5: worker's research-npz assertion is reference-role only. One-time proof: `LEGACY.npz`
    `binary_*` (33 arrays, key sets equal) bit-identical with research
    `kawin/tests/data/illingworth_substitutional_baseline.npz` (blob `faa85c8`, sha256
    `4d51e6b8…65ad`) at clean `7af24f4`.
  - Exact gate: `quick --tolerance exact --cases <27>`: 27/27 bit-identical; 27 reference
    repeat pairs, 0 differences; 358 s. Tests: `test_definitions_bindings.py` (4) +
    materializer (2) pass in `mb-public-050`.
- **M1 commits:** M0 `7f7755f`, M1 `e3b9bfd`, then fix `8f74746`
  (`tools/definitions/.gitattributes` `eol=lf`). Under `core.autocrlf=true` a `git archive`
  export of `e3b9bfd` had CRLF definitions, failing their sha256 check (2 of 6 tests from the
  export); with the pin the export has 0 CR bytes and 6/6 pass. Committed files that get
  hash-checked later (M7 approval reports) need the same pin.
- **M2 (2026-11-10).**
  - 10 contract definition files (4 top-level, 3 new nested; shared files unchanged), AST-equal;
    C25-C29 in `PACKAGE_DEFINITIONS`. F6: completed-assertion lists identical (package report,
    `reference/C2x.json`, fresh reference-role run) for all five.
  - Item 7: contracts report `model_scales` (`acceptance_contracts.model_scales`, both roles);
    reference-role scales equal package scales. **The user approved the scale table on
    2026-11-10**: no model is shorter than 1 m; the mole-inventory allowance tightens ×0.667 for
    the Vm = 1.5 models (C27 model0-4, C29 model2-3); C28 loosens to 1.47e-7 (covers the old
    1.46e-11 ulp failure); C25/C29 binary models (L = 5) loosen ×5 for interface and composition
    inventories. Manifest `contract_model_scales` (21 models); `compare()` →
    `contract_scales()` requires exact equality, exact `model{i}/` prefix sets and finite
    positive values, then scales each prefix like a run case. `reference/C25-C29.json` untouched.
  - Gates: exact contracts 5/5 bit-identical (52 s); public contracts 5/5 bit-identical with the
    new allowances applied (C28 `model1/dependentMoleClosureErrorData` atol 1.47e-7; 42 s).
    `tests/test_contract_scales.py` (9) + reporting + bindings: 26 passed.
- **M2 test-fixture fix (`3b4926c`).** M3's full-suite run showed that M2's `compare()` reads
  `manifest["contracts"]`, which the synthetic manifests in `test_thermocalc_acceptance.py` (5
  tests) and `test_default_phase_acceptance.py` (3) lacked (`KeyError`). The real manifest always
  has the list, so no acceptance gate was affected; M2 had run only its targeted test files. Both
  fixtures now carry `contracts=[]`; `compare()` stays strict. From M3 on, every batch runs the
  full `tests/` suite.
- **M3 (2026-11-10).**
  - `tools/definitions/surrogate_artifacts.py` (`_surrogate`, 1 import transform), AST-equal;
    C24 and `prepare` in `PACKAGE_DEFINITIONS`. Materializer `--check --entries <all 17
    materialized stems>`: OK. (An unrestricted `--check` lists the 8 M4 stems not yet written.)
  - `--prepare` loads through `inputs.root` and saves to the worker's own `--output` npz in both
    roles, never to `--artifacts`; historical `capture` passes a scratch `prepare-*/` output, so
    it no longer rewrites `inputs/open-surrogate.npz` (§3).
  - F7: `acceptance.regenerate_surrogate` runs a package-role `prepare` into scratch and
    `npz_identity` compares it with the adopted file (key sets; dtype, shape and bytes per key;
    ZIP members reported separately). Result: 35/35 arrays bit-identical, key sets equal, and the
    files are also byte-identical (container included; `savez_compressed` writes fixed member
    timestamps). The adopted sha256 `c224e5f2…693f` is unchanged before/after. M5b records this
    formally with the metadata payload. Unit tests: `tests/test_npz_identity.py` (3).
  - Gates: `quick --tolerance exact --cases C24`: 1/1 bit-identical, package report
    `case_definitions=tools/definitions`, reference repeat 0 differences (25 s). Full `tests/`:
    329 passed, 5 skipped, 1 deselected (120 s). Artifact hashes: 207 files, 0 changed.
- **M4 (2026-11-10).**
  - 8 definition files (7 top-level: `exb5_run_kawin2/3`, `fe_c_exb1a`, `ferrite_growth_le`,
    `fe_mn_c_ferrite_case`, `three_phase_symmetric_closure`, `fecrni_analytic_comparison`; 1
    nested), all AST-equal (F1 constants, F2 constructor cut-offs, F4 module-global overrides).
    The unrestricted materializer `--check` is now OK for the whole table.
  - F04, F11, F12 and O01-O08 in `PACKAGE_DEFINITIONS`. O05's pycalphad sample is saved to
    `<output>.pycalphad.npz` beside the worker's output in both roles, so no run rewrites
    `inputs/pycalphad-<role>.npz` (§3); the worker reads its `provider/*` arrays from there.
  - Gate: `full --tolerance exact --cases F04 F11 F12 O01 O02 O03a O03b O04 O05 O06 O07 O08`:
    **12/12 bit-identical** (O06 live TC, 6 accepted steps; O05/O08 under `PYTHONHASHSEED=0`, F8),
    all reports `case_definitions=tools/definitions`, O05/O06/O08 generation audits guarded with
    no `examples` import or read; 164 s. Artifact hashes: 207 files, 0 changed.
  - Full `tests/`: 329 passed, 5 skipped, 1 deselected (212 s). Skipped: the 5
    `test_acceptance_contracts.py::test_main_vs_package_contract[C25-C29]`, which need
    `MB_ACCEPTANCE_STATE` and run only inside `acceptance.py quick|full` (by design; covered by
    the M6/M10 gates). Deselected: `test_examples.py::test_three_phase_errors_converge_with_refinement`
    (`slow` marker; `pyproject.toml` `addopts = -m 'not slow'`).
- **M5 (2026-11-10).**
  - `tools/example_manifest.json` committed unchanged first (`chore` commit; sha256
    `c2d77779…8951`, last modified 2026-11-06; it was hidden by the shared `*.json` exclude), then
    edited in M5.
  - Hashed settings: `example_manifest.json` `settings_sha256` covers the 8 non-private adopted
    settings files (C04, C11, C12, C16, C24, F04, F11, F12). The O04/O05 settings (TC/TDB-derived)
    and the adopted E05 surrogate `pycalphad-reference.npz` are hashed in the local
    `inputs/private-inputs.json` (new keys `example_settings`, `pycalphad-reference.npz`; existing
    entries unchanged, asserted). That file's sha256 therefore changes, by design, from
    `19f26a29…` to `b623016b…`; it is the only local artifact change, and the new artifact baseline.
  - `acceptance_examples.py run` checks each adopted settings file (unhashed or changed → case
    fails) and each case's private surrogate input against `private-inputs.json` before running.
  - Historical writes go to scratch: `capture-settings` re-records all 10 into a fresh
    `settings-capture-*` folder and reports byte equality with the adopted files, never writing
    `example_settings/`; `example_worker.py`'s reference-role O05 build saves its pycalphad sample
    beside `--output`. (Since M4, `sample_pycalphad` needs `inputs.pycalphad_sample`; before M4 a
    historical capture rewrote `inputs/pycalphad-reference.npz`.)
  - Gates: `acceptance_examples.py run --skip-tests`: **E01-E10 10/10 PASS** (settings equal,
    parity arrays bit-identical, stop status; 87 s). `capture-settings` (reference role at
    `7af24f4`): **10/10 identical** with the adopted files (57 s). Unit tests
    `tests/test_example_settings_hashes.py` (3). Full `tests/`: 332 passed, 5 skipped, 1
    deselected (same as M4; 115 s). Artifact hashes: 207 files, only `private-inputs.json` changed.
  - Export of `9c37a44`: example acceptance E01-E10 10/10 PASS with the wheel built from the
    export (the committed `example_manifest.json` exercised); full `tests/` 332 passed.
- **M5b (2026-11-10).** New `tools/acceptance_regeneration.py run` (package generation paths only,
  scratch `regeneration-*` output) and `example_worker.py --settings-only` (package role: the
  accepted case built from `tools/definitions`, settings recorded, no solve).
  - Backends: this environment (`mb-public-050`) and the generation environment (research `kawin`
    env, **probed at M5b, not recorded when the adopted files were made**) match: pycalphad
    0.11.1, numpy 2.4.3, scipy 1.17.1, symengine 0.13.0 (kawin 0.5.0 vs fork 0.4.0, by design).
  - **12/12 checks `verified`, 0 failed, 0 unavailable, 0 not comparable; no regeneration
    differences** (allowed-differences list stays empty):
    - `open-surrogate.npz` (package `prepare`): 35/35 arrays bit-identical, key sets equal,
      `metadata_json` empty in both (0 fields), container byte-identical.
    - `pycalphad-reference.npz` (package O05 sampling, `PYTHONHASHSEED=0`): 35/35 arrays
      bit-identical including non-numeric, 30 metadata fields equal, container byte-identical;
      adopted numeric arrays = `reference/O05.npz` `provider/*` (21/21, key sets equal).
    - 10 settings files (C04, C11, C12, C16, C24, F04, F11, F12, O04, O05): byte-identical and
      JSON-equal (62-111 fields each) after hash checks of the adopted files and their inputs.
  - Not checked, reported explicitly (`deferred`, do not count as verified): `exb5.json`,
    `femnc.json`, `fec.json`, `fecrni.surrogate` — no package generation path
    ([FUTURE_PORTS.md § Cache rebuilding gaps](FUTURE_PORTS.md#cache-rebuilding-gaps)).
  - Adopted files' sha256 unchanged before/after (asserted by the tool); artifact hashes 207/207
    unchanged against the M5 baseline. Redacted summary committed as
    `docs/regeneration-verification.json` (outcomes, counts, container notes; no hashes, values or
    local paths); full local report `.verification/acceptance/regeneration-9fc018a59afe.json`.
  - Unit tests `tests/test_acceptance_regeneration.py` (4); full `tests/`: 336 passed, 5 skipped,
    1 deselected (120 s). Run time 87-110 s.
- **M6 (2026-11-10).**
  - `acceptance.py`: normal actions (`quick`, `full`, `setup`) take no research checkout and run no
    Git. `validate_expected(cases)` runs before and after: `expected_artifacts` hashes of every
    case compared (contracts and O06 included), the new committed `adopted_inputs` section
    (`open-surrogate.npz` sha256, `required_by` C24: a missing file fails C24 with "required input
    missing: inputs/open-surrogate.npz (needed by C24; never regenerated)"), and every definitions
    file against its provenance; `private-inputs.json` checks are unchanged. Historical actions
    are `historical capture|repeat --research PATH` with today's checks renamed
    `validate_research_checkout`; `capture` now refuses to overwrite any existing `reference/`
    pair (previously only O06/O08); `repeat` runs the reference role twice per case and requires
    both runs to equal each other and `reference/` (O05 once, seeded; O06/O08 not repeated).
  - Unseeded repeatability runs two package-role processes per case (normal runs and contract
    pytest). Package workers (`acceptance_worker.py`, `example_worker.py`) get no `--root`; the
    package role accepts only `PACKAGE_DEFINITIONS` cases. `acceptance_examples.py run` uses
    `validate_expected`; `capture-settings` needs `--research`. `REFERENCE` (the default oracle
    path) is gone.
  - `tooling_sha256` and `tooling_revision_note` removed from the manifest (nothing read them);
    their supersession is recorded in the package docs in M10.
  - Follow-up 2 done: `_bindings.py` has no research-solver fallback. `model()`, the
    `CartesianFD1D` wrapper, `imported_module()` and `symbol()` for research-only (`MODULES`)
    modules raise `LookupError` naming the missing package solver/module; ordinary imports
    (stock kawin, third-party) are unchanged.
  - Gates: **exact quick 30/30 bit-identical, contracts 5/5, contract pytest 93 passed** (89 + 4
    new parser-rejection cases), package repeatability 33 cases × 2, 0 differences (464 s of 600);
    **exact full 40/40, contracts 5/5, pytest 93, repeatability 34 cases (O06 live), 0
    differences** (578 s of 1200; nothing skipped). `historical repeat --cases C01 C25`: both
    reproduce `reference/` bit for bit; `historical capture --cases C01` refuses ("fixture is
    frozen; refusing overwrite"). Examples E01-E10 10/10; regeneration 12/12 `verified` (summary
    byte-identical to the committed one). Materializer `--check` OK. Full `tests/`: 347 passed,
    5 skipped, 1 deselected (new `tests/test_validate_expected.py`, 7). Artifact hashes 207/207
    unchanged.
- **M7a (2026-11-10): `approval_criteria` adopted, user-approved.** The proposal shown at the M7 gate
  was revised on review: the binary profile mask is frozen (centred on the reference's exact final
  interface, half-width = approved interface limit + 2h; excluded nodes stored: C04 2-8, C31 3-7,
  F04 27-42), and conservation limits are 10x current at two significant figures (floor 1e-13)
  instead of the next decade. `tools/acceptance_criteria.py` is the single implementation:
  `propose` measures the hash-validated references and applies the approved rules, `check`
  evaluates them, and the manifest section equals a fresh proposal (tested). 74 limits: 18
  analytic (C04, C31, F04 interface + masked profile; C11, C12, F11, F12 interface + eta + profile;
  e.g. F04 profile 0.00277 → 0.0056) and 56 conservation across all three families (binary
  transformed inventory; composition inventory for ternary and equal-volume three-phase cases;
  component moles, total moles and closure for explicit volumes). Unequal-volume composition
  inventories are reported only; ternary accuracy, contracts and O08 are recorded as not covered.
  Any criterion that cannot be measured (missing/empty array or mask, NaN/inf, fewer than two
  points, zero reference, mesh mismatch) fails. `check` on today's references: all pass. Unit
  tests `tests/test_acceptance_criteria.py` (22); full `tests/`: 374 passed, 5 skipped, 1
  deselected.
- **M7b (2026-11-10): approval path.** New `tools/acceptance_approval.py` (no Git):
  `expected_reference` (one selection helper, used by `compare()`, `contract_assertions()` and
  `acceptance_examples.compare_case()`; with no `approved_references` section it returns today's
  `reference/<case>` paths and the O06/O08 frozen hashes), Git-free `validate` (called by
  `validate_expected`), approval IDs `<UTC>-<sha7>-<counter>` that skip IDs used by records,
  folders or reports, crash-safe `promote` (`.partial` → re-hash + reload → rename → report →
  temp manifest + `os.replace` with OneDrive-lock retry, newline style preserved), `rollback`, and
  `leftovers`. `acceptance.py approve --cases … --reason …` / `approve --rollback ID --reason …`.
  Implementation choices to note:
  - *Gates:* every run case at public tolerance; repeat pairs for the normal repeat set plus every
    listed case (whose candidate must also equal its repeats) plus O05/O08 as seeded pairs;
    `approval_criteria` on the candidate arrays (unmeasurable fails); the installed-target pytest
    set, in which contracts gate on their assertion groups (`contract_assertions`: worker PASS,
    `contract_assertions_passed`, same group names as the reference, matching count) and only a
    *listed* contract's array comparison is reported instead of failed. Normal runs always clear
    the candidate list, so they gate every comparison as before.
  - *Fail fast:* approve stops at the first failed gate and writes nothing; the adoption procedure
    already lists failing cases with a public quick first.
  - *Budget:* approve has its own limit, 900 s (quick) / 1500 s (full), because it repeats more
    cases than a normal gate. Measured: quick approve 420 s in the clone.
  - *Rollback* undoes one approval (its cases return to what was in effect before it) and is
    refused while a later approval of the same case is in effect (LIFO per case).
  - *Not approvable:* O04/O05, whose example settings are hashed only in the private record.
  - *Clean tree* = nothing under the package directory in `git status --porcelain
    --untracked-files=all`. A real approval in this worktree therefore needs the user's
    uncommitted package files (`NEXT_SESSIONS.md`, the example Settings lines,
    `AUTHORITY_SWITCH_PROPOSAL.md`) set aside first.
  - `docs/approvals/.gitattributes` pins `*.json text eol=lf`; reports must be force-added.
  - `test_acceptance_approval.py` (41 tests, `tmp_path` projects) joined the installed-target
    pytest set (97 → 138 in exact quick).

  Validation:
  - Unit and interruption tests: fault after every step and during the replace (persistent and
    transient locks), after the commit point, rollback interrupted. Each time the old (or new)
    state validates and a retry succeeds with the next counter.
  - Worktree: exact quick 30/30 runs and 5/5 contracts bit-identical, 0 repeat differences,
    pytest 138, 481 s of 600; full `tests/` 415 passed, 5 skipped, 1 deselected.
  - Isolated clone (base `3ca6911`, test-only snapshot commit `0fb2ca7` with the 5 changed or new
    files byte for byte; acceptance inputs, references, settings, pilot, numba cache, wheel and
    environment file copied; `install-*` and run output not copied). It had to move from the
    session scratchpad to `%TEMP%\claude\m7b`: there a pytest throwaway Git repo exceeded MAX_PATH
    (270 > 260) and the first approve failed closed at its pytest gate, writing nothing.
    Results:
    - Synthetic change (test-only commit `d18ffdd`: C01 duration 0.5 → 0.45).
      `approve --cases C01 C04` passed: C01 differs (12 arrays), C04 bit-identical, 33 unlisted
      cases and contracts passed, 30 repeat pairs, pytest 138. C04's package-captured settings
      have the adopted file's sha256; the CRLF manifest kept its newlines (50 lines inserted).
    - Against the mixed set (exact C01/C04 approved, C02 reference): 3/3 bit-identical; examples
      (quick tier) 6/6 with E01 against the approved C04.
    - `git archive` export (`core.autocrlf=true`): `validate_expected` passes with an LF report.
      With the pin removed, the report gets 812 CR bytes and validation fails ("approval report
      changed").
    - Rollback: no approved file moved; exact C01 then fails against `reference/` as expected and
      C04 passes.
    - Leak check: `approve --cases C04` with the changed C01 unlisted stopped at C01 in 13 s and
      wrote nothing.
  - Real paths (manifest, `docs/approvals/`, `docs/acceptance*.json`, `example_settings/`,
    `reference/`, `reference-approved/`): sha256 and mtime unchanged (193 files). Local artifacts
    equal the M5 baseline (207 files).
- **M7b follow-up fix (user review of `e86ff94`).**
  - *Rollback:* a rollback whose restored selection does not validate, e.g. an earlier approval's
    file is missing, used to replace the manifest and only then fail, leaving acceptance pointed
    at a missing artifact. `rollback` now validates the proposed section (`_check_proposed`)
    before the replace and refuses with the manifest byte-for-byte unchanged. `promote` runs the
    same check before its replace.
  - *Build provenance:* `approve --reuse-targets` accepted a saved `state.json` target that could
    come from another commit. `approve` now always builds fresh targets from the clean tree at
    HEAD; the CLI rejects `--reuse-targets` for approve, which stays available for diagnostic runs.
  - Regression tests, which fail on `e86ff94`: rejected rollback leaves the manifest unchanged,
    through both the module and the CLI path; approve never reads a saved target; the CLI
    rejects the flag. 44 approval tests.
  - The Git dependency is follow-up 7, on the M9 checklist.
- **M7b gate scope (user decision after the review; follow-up 8).**
  - `approve` gates on the whole `tests/` suite and repeats every run case in every tier.
  - Probe in the worktree, fresh targets: whole suite on the installed wheel 429 passed, 1
    failed, 0 skipped. The failure was a new test that assumed `MB_EXAMPLE_TDB` absent from the
    parent environment; the test was corrected.
  - Clone (test-only commit `27fd12c` on top of the synthetic change):
    `approve --tier full --cases C01` passed in 793 s of the 1500 s budget. Repeat pairs for all
    40 run cases, including F04/F11/F12, O01-O04, O07 and live O06. Whole suite 430 passed, 0
    skipped. 44 unlisted cases and contracts passed; nothing skipped.
  - Quick approve was not re-measured. Estimate: 420 s measured with the allowlist, plus about
    110-200 s for the whole suite, under its 900 s budget.
  - Real paths unchanged (193 files); artifacts equal the M5 baseline.
- **M8 batch 1 (2026-11-10).**
  - *Port tool:* `tools/port_tests.py` (historical: reads research tests by path, never imports
    them). It rewrites imports name by name with `_bindings.Context.symbol`'s resolution, renames
    research module paths inside strings, copies `kawin.tests.*` helpers verbatim into
    `tests/ported_helpers/`, removes deferred tests and the imports only they used, and prepends a
    provenance docstring. Every transform and the LF-normalized source and output sha256 are
    recorded in `docs/test-port-provenance.json`. `--check` regenerates and compares; any research
    module path left in the output is an error.
  - *Helpers:* reusing `tools/definitions/*` was rejected for the test helpers. Those files carry
    load-time transforms (e.g. the injected terminal thin-phase policy) and only selected names,
    so the research tests would no longer check what they check. The helper is therefore a
    verbatim copy with provenance, the plan's fallback. `tests/conftest.py` puts `tests/` on
    `sys.path`, so `ported_helpers` imports also work under `--import-mode=importlib` (approve's
    gate).
  - Per file, ported / deferred / total:
    - `test_illingworth_setup_and_diagnostics.py` 6/0/6;
    - `test_illingworth_ternary_three_phase.py` 62/0/62;
    - `test_illingworth_interstitials_three_phase.py` 2/0/2;
    - `test_surrogate_artifacts.py` 11/1/12;
    - helper `illingworth_baseline_cases.py`.
  - *Deferred:* `test_cuniti_tc_python_bundle_smoke`, an opt-in licence smoke test
    (`KAWIN_RUN_TC_PYTHON_SMOKE=1`, TCHEA5/MOBHEA4). Ported as is, it would skip in every normal
    run and so block approve's no-skip whole-suite gate. Its adapter is already ported. Listed in
    FUTURE_PORTS.md § Deferred tests; where it should run is a gate decision for the user.
  - Results:
    - 141 ported test items (81 functions) pass from `src/` (11 s) and under
      `--import-mode=importlib`;
    - `--check` reports the 5 files identical to a fresh port;
    - `tests/test_port_tests.py` (7);
    - full `tests/` 573 passed, 5 skipped, 1 deselected.
- **M8 batch 2 (2026-11-10).**
  - Per file, ported / deferred / total:
    - `test_illingworth_binary_variable_diffusivity.py` 14/0/14;
    - `test_illingworth_interstitials.py` 13/0/13;
    - `test_transformed_grids.py` 16/0/16 (the plan's "transformed_grids").
  - *Subset helpers:* two helpers are themselves research test modules, so the tool gained a
    subset mode (`HELPER_SUBSETS`). It extracts only the named definitions, plus the top-level
    code and import aliases they need, as verbatim segments with research line markers, into a
    non-test file name:
    - `test_diffusion.py` (4338 lines, 131 tests) → `ported_helpers/diffusion_helpers.py`
      (`ConstantBinaryThermodynamics`, `VariableBinaryThermodynamics`,
      `_build_illingworth_default_model`);
    - `test_illingworth_ternary.py` → `ported_helpers/ternary_helpers.py`
      (`_IdentityTernaryThermodynamics`, `_LinearInterfaceEquilibrium`).
    The extract keeps research's double import of `TemperatureParameters` in order, so the stock
    binding still wins, as in research. A star-imported name or an undefined name is an error.
  - *Tool fixes:* batch 2's research file names corrected (`test_transformed_grids.py`;
    batch 3's `test_illingworth_ternary_physical_scale.py`, `test_similarity_comparison_basis.py`).
    Helpers are shared across batches, so their header and record carry no batch; this
    regenerated batch 1's helper header and record line only.
  - Results:
    - 129 ported items pass (6 s; also under importlib mode);
    - `--check` identical for batches 1 and 2;
    - `test_port_tests.py` 8;
    - full `tests/` 703 passed, 5 skipped, 1 deselected.
- **M8 batch 3 (2026-11-10).**
  - Per file, ported / deferred / total:
    - `test_illingworth_ternary_three_phase_similarity.py` 8/0/8;
    - `test_illingworth_interstitials_ternary.py` 8/0/8;
    - `test_illingworth_ternary_physical_scale.py` 6/0/6;
    - `test_similarity_comparison_basis.py` 2/0/2;
    - `test_illingworth_ternary_similarity.py` 2/0/2;
    - deferred whole, recorded under `deferred_files`: `test_illingworth_ternary_convergence.py`
      0/5/5 and `test_three_phase_examples.py` 0/3/3.
    Total 26 of 34, as planned.
  - *Relocated example helpers:* the tests import the similarity helpers from the research
    example scripts `ternaryTwoPhaseSimilarity`/`ternaryThreePhaseSimilarity`. They now bind
    `moving_boundary.analytic`, where §6 and FUTURE_PORTS record the helpers as ported; all 10
    names exist there. The example paths remain only in docstring prose. No test checks the
    example scripts themselves, so none was flagged.
  - *Patches across the module split (new transform):*
    - `test_ternary_initial_eta_componentwise_scaling...` in `interstitials_ternary` failed first.
      It patches `_COMPONENTWISE_ROUNDOFF_FACTOR` on the research ternary module, where both
      that constant and `estimate_initial_eta_from_instantaneous_balance` lived. The package
      moved both into `illingworth/shared.py`, and `ternary.py` re-imports them, so patching
      `ternary` no longer reached the function. This is a relocation effect, not a numerical
      difference.
    - New hand-written `tests/ported_helpers/relocation.py`: `setattr_owners` patches the name
      on the module and on every package module along its `from … import` chain, with the same
      value.
    - The tool uses it only where the chain is longer than one, or the name is not a literal,
      and records each use. Batch 4 has 10 module-attribute patches that this covers.
  - Results:
    - 36 ported items pass (75 s; also under importlib mode);
    - `--check` identical for batches 1-3;
    - `test_port_tests.py` 9;
    - full `tests/` 740 passed, 5 skipped, 1 deselected, in 204 s (up from 140 s).
- **M8 batch 4 (2026-11-10) and M8 totals.**
  - `test_illingworth_ternary.py` 151/4/155. Deferred:
    - the 3 surrogate-construction diagnostics tests (planned), whose imports
      `evaluate_/plot_surrogate_construction_diagnostics` were dropped;
    - **one more, not in the plan:** `test_ternary_illingworth_quasi_binary_cu_zn_dummy_matches_binary_brass_case`.
      It builds its binary reference with `build_fig3_present_work_model` and
      `build_fig6_illingworth_case_params` from the research example
      `examples/Illingworth2005/compare_illingworth2005_planar.py`, which is not ported
      (FUTURE_PORTS entry 6). Added to FUTURE_PORTS § Deferred tests; flagged for the user.
  - *Patch rule corrected before committing:* the batch 3 rule followed every package re-import.
    For a name the research module only *imported*, research changed that module's binding
    alone, so following the chain would patch more code than research did.
    - The examples here were `solve_illingworth_block_tridiagonal` (research TernaryFDM imports
      it from the mesh module) and `MulticomponentThermodynamics` (research Surrogates imports it
      from `kawin.thermo`).
    - Now the chain is followed only for names the research module *defines* (read by path at
      port time). A patch whose name is known only at run time (`helper_name`) passes a per-file
      frozenset of the research module's definitions to `setattr_owners`.
    - Result in batch 4: 5 patches follow the chain (`_validate_ternary_diffusivity_matrix` ×3,
      `_select_interface_motion_branch`, and `helper_name` checked at run time); the other 5
      stay plain. Batch 3's single use is unchanged (identical regeneration).
  - Results:
    - 263 ported items pass (27 s);
    - `--check` identical for batches 1-3;
    - `test_port_tests.py` 10;
    - full `tests/` 1004 passed, 5 skipped, 1 deselected (227 s).
  - *Approve's gate:* the whole suite on freshly built installed targets, contracts configured,
    passes 1009, 0 skipped, in 298 s. Estimated quick approve about 640 s of 900.
  - **M8 totals:** 301 of 314 research tests ported, in 13 files plus 3 generated helpers and
    `relocation.py`. 13 deferred, all listed in FUTURE_PORTS.md § Deferred tests:
    - the TC licence smoke test;
    - 8 deferred whole (convergence 5, three_phase_examples 3);
    - the 3 diagnostics tests;
    - the quasi-binary brass test.
- **M9a (2026-11-10): guards and fixes needed before the independence runs.**
  - *Guard* `tools/batch_safety.guard_independence`, opt-in, installed by both workers, the test
    `conftest` and the `acceptance.py` / `acceptance_examples.py` `main`s; child processes
    inherit the environment variables:
    - `MB_FORBID_RESEARCH=1`: a first meta-path finder plus the `import` audit event refuse the
      §7 research modules; the `open`, `os.listdir` and `os.scandir` events refuse paths under
      `MB_FORBID_ROOTS` or with an `mb-ref-*` component (checked before and after resolving
      links); a forbidden root on `sys.path` is refused at every import. M9 forbids the whole
      extraction worktree, not only its `kawin/` and `examples/`, because `mb-public-050` has an
      editable `.pth` pointing at the worktree's `src`. Workers run with `-S`, so it is never
      processed there.
    - `MB_FORBID_GIT=1`: Git launched through `subprocess`, `os.system`, `os.exec*`, `os.spawn*`,
      `os.posix_spawn` or `os.startfile` is refused. That includes Git started through a shell.
    - `MB_GUARD_LOG`: every violation is also logged, so one swallowed by a broad `except` still
      fails the check of the log.
    - Violations raise `IndependenceViolation`, a `RuntimeError` rather than `ImportError` or
      `OSError`, so fallback handlers cannot hide them.
    - `tests/test_acceptance_independence.py` (28) proves each case in its own process: research
      imports, reads, listing, a swallowed read, `sys.path`, a junction into a forbidden root,
      Git direct, by full path and through the shell, and the guard off by default.
  - *Follow-up 7:* `test_frozen_content_guard` is marked `git` (a new marker). The normal gate's
    `run_pytest` passes `-m "not slow and not git"`; approve (which uses Git) passes `-m "not slow"`.
  - *Follow-up 6:* that test's disposable repository now lives under a short system temp root
    (`tempfile`, prefix `mbgit-`), not `tmp_path`, so `--basetemp` depth no longer matters.
  - *Follow-up 5:* `validate_expected(cases, required_inputs_for=None)`. Example acceptance calls
    `validate_references` (`required_inputs_for=()`). It hash-checks the reference artifacts it
    compares with and any present adopted input, but no longer requires C24's
    `open-surrogate.npz`. Tested on the synthetic project in `test_validate_expected.py`.
  - Results: full `tests/` 1033 passed, 5 skipped, 1 deselected (205 s).
- **M9a fix `4d64aa0` (found by the first guarded export run of `0f71c19`).**
  - On Windows, `subprocess` audits the joined command line. `git_invocation` had matched "git"
    inside a Python `-c` script. It now decides by the program token, and searches the rest only
    for a shell; `os.system` strings are searched whole.
  - The guard tests' fixture no longer creates an `mb-ref-*` file in the (possibly guarded) test
    process.
  - `test_pycalphad_provider::test_generation_guard_blocks_research_imports_and_source_reads`
    imports `examples` on purpose, to test the M4 generation guard, and the M9 guard refused it
    first. It now runs in its own process without the M9 variables, with the same assertions.
- **M9b (2026-11-11): independence proof and missing-resource matrix (scratch driver
  `m9_proof.py`, not committed).**
  - *Exports and environment:*
    - Exports are `git archive` of the package at the commit, under
      `%TEMP%\claude\m9\<name>`, with no checkout above them. `.verification` is copied (226
      files: inputs, references, example settings, pilot, stock wheel, environment file), not
      linked.
    - Every run had `MB_FORBID_RESEARCH=1` and `MB_FORBID_GIT=1`. `MB_FORBID_ROOTS` covered the
      whole extraction worktree, the research checkout, the research env and both scratch
      clones, plus any `mb-ref-*` path.
    - There was one `MB_GUARD_LOG` per run, PYTHON* and CONDA* variables were removed, and
      conda was activated afresh for `mb-public-050`.
    - `PATH` had every directory holding `git.exe`/`git.cmd`/`git.bat` removed and a failing
      `git` shim first. Inside the activated env, `where git` finds only the shim.
    - After each setup, the installed target's `moving_boundary` was checked equal to the
      export's `src` (32 files), so the unguarded `pip wheel` built from the export.
  - *Finding (environment, not changed):* `mb-public-050` has an editable `.pth` pointing at
    the worktree's `src`. The guard refused the first guarded `quick`, because a normally
    started acceptance parent has that path on `sys.path`. The parent never imports
    `moving_boundary`; workers and test runs use `-S`. The M9 runs start tool parents the same
    way (`python -S` with that env's site-packages on `PYTHONPATH`). Follow-up 11.
  - *E1, empty approved set (export of `4d64aa0`), every run with 0 guard violations:*
    - whole `tests/`: 1036 passed, 5 skipped (C25-C29 contract comparisons, which need
      `MB_ACCEPTANCE_STATE`), 4 deselected (1 `slow`, 3 `git`), in 218 s;
    - ported files on their own: 570 items passed, 0 skipped, in 98 s. Per file:
      - binary_variable_diffusivity 31;
      - interstitials 19, interstitials_ternary 13, interstitials_three_phase 3;
      - setup_and_diagnostics 16;
      - ternary 264, ternary_physical_scale 6, ternary_similarity 2;
      - ternary_three_phase 104, ternary_three_phase_similarity 13;
      - similarity_comparison_basis 2, surrogate_artifacts 18, transformed_grids 79;
    - exact quick: 30/30 runs and 5 contracts bit-identical, 33 repeat pairs with 0
      differences, gate pytest 145 passed with the 3 `git` tests deselected; 433 s of 600;
    - exact full: 40/40 runs (O06 live TC) and 5 contracts bit-identical, 34 repeat pairs;
      549 s of 1200;
    - examples `--skip-tests`: 10/10, 75 s; with tests: 10/10 plus 56 tests, 142 s.
  - *E2, active synthetic approval:*
    - Made in the scratch clone `%TEMP%\claude\m9c`: base `4d64aa0`, then the M7b synthetic
      C01 change (duration 0.5 to 0.45, `394cb71`).
    - `approve --cases C01 C25` gave `20261111T022430Z-394cb71-1`: C01 differs, C25 is
      unchanged; the whole-suite gate passed 1044 tests, `git` tests included; 603 s of 900.
    - The approval was committed in the clone (`c77815e`) and exported. The real worktree's
      approval files were unchanged (193 files, 0 changed).
    - Git-blocked exact quick: 30/30 runs and 5 contracts bit-identical, 0 violations, 430 s.
      The run record shows C01 and C25 compared with `20261111T022430Z-394cb71-1` and every
      other case with `reference`. The changed C01 can match only the approved arrays.
    - Ported files: 570 passed, 0 violations.
  - *Missing-resource matrix (each row a fresh export, guard on, Git blocked, 0 violations):*

    | Removed / changed | Observed |
    |---|---|
    | `FeCrNi.tdb` | quick: O05, O08 SKIPPED "database input missing"; 28/28 and 5 contracts bit-identical. **E05 PASSED** in example acceptance (see below) |
    | `exb5.json` | full: O01, O02 SKIPPED "cached input missing"; 38/38 and 5 contracts bit-identical |
    | `inputs/O06.json` (TC config) | full: O06 SKIPPED "local O06 input missing" (the probe reason); 39/39 and 5 contracts bit-identical |
    | `open-surrogate.npz` | quick fails at once: "required input missing: inputs/open-surrogate.npz (needed by C24; never regenerated)". Example acceptance: **E04 PASS**, 10/10 |
    | one bit of `reference/C04.npz` | quick fails: "C04: expected artifact changed: reference/C04.npz" |
    | one bit of the approved `C01.npz` (E2) | quick fails: "C01: approved artifact changed: reference-approved/20261111T022430Z-394cb71-1/C01.npz" |
    | extra: `pycalphad-reference.npz` | example acceptance: E05 SKIPPED "local input missing: pycalphad-reference.npz"; the other 9 pass |

  - **Discrepancy with the §7 table (awaiting the user):** the table expects E05 to skip when
    `FeCrNi.tdb` is removed. E05 solves from the frozen `inputs/pycalphad-reference.npz` (as
    `ACCEPTANCE_CANDIDATES.md` states) and never reads the TDB, so it ran and passed. Its input
    is `pycalphad-reference.npz` (extra row). Proposed correction to the table: `FeCrNi.tdb`
    row "O05, O08 skipped"; new row `pycalphad-reference.npz`, "E05 skipped (local input
    missing)". The table is not edited until the user decides.
  - Artifact hashes vs `hashes-M5-after`: 207 files, 0 changed. The worktree's full `tests/`
    for `4d64aa0` is included in E1 (same tree).
  - Gate time: about 76 min of runs (E1 26, approval 10, E2 9, matrix 28) plus the M9a suite
    and fix runs, against the §9 estimate of 44-46 min. The excess comes from checks added
    after the estimate: the separate ported runs and the active-approval repeats (user review
    after M8), a fresh approval under the whole-suite approve gate (M7b scope decision), and
    the skip rows each needing a real quick or full gate.
- **M9c (2026-11-11): user decisions applied; M9 done.**
  - The user approved the E05 correction. The §7 table now says that removing `FeCrNi.tdb`
    skips O05 and O08, and that removing `pycalphad-reference.npz` skips E05. Both were
    observed in M9b.
  - Follow-up 11 is resolved. At the user's request, only the editable `moving-boundary`
    install was removed from `mb-public-050` (`pip uninstall moving-boundary`).
    - It removed `__editable__.moving_boundary-0.0.1.dev0.pth` and the `dist-info`.
    - `pip freeze` differs only by that line. Stock `kawin` is unchanged, and the source tree is
      untouched (32 files). The research `kawin` env was only read.
    - `setup_public_environment.py` never installed the package, so it is unchanged.
  - Normal start from a fresh export of `d879153`, with the guard on and Git blocked (driver
    kind `tool`: no `-S`, no injected `PYTHONPATH`); 0 violations in every run:
    - `acceptance.py quick --tolerance exact`: 30/30 runs and 5 contracts bit-identical, in
      432 s. The target matches the export's `src`.
    - `acceptance_examples.py run` with tests: 10/10 plus the tests, in 150 s.
    - Worker and test isolation (`-P -S` workers, `-I -S` gate pytest) is unchanged.
  - *Developer test command corrected (README "Local acceptance").*
    - `PYTHONPATH=<package>/src python -m pytest tests` now fails 5 FeCrNi example tests:
      `PackageNotFoundError: moving-boundary`. The example records
      `importlib.metadata.version("moving-boundary")` in its bundle provenance, and the editable
      install used to supply that metadata.
    - The documented command is now a private non-editable install
      (`pip install <package> --no-deps --no-build-isolation --no-index --target <scratch>`) on
      `PYTHONPATH`. From the export, guarded, that gives 1036 passed, 5 skipped, 4 deselected,
      0 violations.
    - Correction (user review after M10): installing into an existing target folder can
      silently keep the old module. The user reproduced this: pip exits successfully with
      warnings, and modules deleted since are never removed. The README now requires a new,
      empty target folder per revision (`mkdir` first, so a reused name fails).
    - The example is unchanged: it is meant to run against an installed package.
  - Environment instructions are updated in README "Local acceptance" and `docs/HANDOFF.md`.
- **M10 (2026-11-11): docs and final recorded gates.**
  - Docs: `a58aa09`:
    - LEDGERS "Authority switch (2026-11-11)": ownership, normal vs historical actions,
      approvals, item 7, item 14 supersession, migration log;
    - NEXT_SESSIONS: definition of done, the policy lines the switch supersedes, and the
      deferred-ports backlog link;
    - AUTHORITY_SWITCH_PROPOSAL.md rewritten as a record of what was built.
  - Environment instructions: M9c `4c6761f`.
  - Final gates, worktree at `a58aa09`, documented commands:

    | Gate | Result |
    |---|---|
    | exact quick | 30/30 runs and 5 contracts bit-identical; 33 repeat pairs, 0 differences; gate pytest 145 passed, 3 `git` deselected; 385 s of 600 |
    | exact full | 40/40 runs (O06 live TC) and 5 contracts bit-identical; 34 repeat pairs; 502 s of 1200 |
    | public quick `--record` | 30 runs and 5 contracts bit-identical, none within 10× of tolerance; 381 s |
    | public full `--record` | 40 runs and 5 contracts bit-identical, none within 10× of tolerance; 492 s |
    | examples with tests | 10/10 and 56 tests; 143 s |
    | full `tests/` (private non-editable install on `PYTHONPATH`) | 1039 passed, 5 skipped (C25-C29 comparisons outside an acceptance run), 1 deselected (`slow`); 210 s |

  - `--record` updated `docs/acceptance.json`: quick and full elapsed times, and the gate pytest
    count 89 → 145.
  - It also updated `docs/acceptance-deviations.json`: the contract entries now list per-model
    effective tolerances from the item 7 scales adopted in M2, e.g. C25 `atol` 5e-12, C28 mole
    inventory 1.47e-07. The previous record predated M2. Every deviation is 0.0.
  - Artifact hashes vs `hashes-M5-after`: 207 files, 0 changed.
  - Ported and deferred counts per file: M8 log and M9b (570 ported items, 13 deferred tests in
    FUTURE_PORTS.md).
- **Follow-ups recorded 2026-11-10 (user review):**
  1. *`docs/definitions-provenance.json` is git-ignored.* It matches `*.json` in the shared
     `.git/info/exclude` (research checkout, line 13); the other `docs/*.json` are tracked only
     because they were force-added. The loader needs it, so: stage it explicitly (`git add -f`) in
     the M1/M2 commits; build the M7/M9 scratch snapshot from the owned-file list plus
     `git status --porcelain --ignored`, not plain `--porcelain`; and the M9 `git archive` export
     must contain it (an export without it fails every package-definition case, which the M9 runs
     would expose). Also check any other new JSON (e.g. `docs/approvals/`) the same way.
  2. *Remove research-solver fallbacks from `_bindings.py` (M6, before M9 independence).* **Done in M6.**
     `model()` falls back to `kawin.diffusion.MovingBoundaryIllingworth*FDM`; `symbol()`'s
     `CartesianFD1D` wrapper falls back to `kawin.diffusion.mesh` when the solver or
     `moving_boundary.mesh` isn't found; `imported_module()` falls back to the research module
     name. None was exercised in the M1/M2 runs (all providers `moving_boundary.*`). They become
     clear errors naming the missing package solver; the final `importlib.import_module(module)`
     for stock-kawin names stays. Guarded by a unit test (missing package solver raises) and the
     M6 exact quick/full gates.
  3. *Pre-existing ignored files found while checking `git status --ignored` (never committed):*
     `tools/example_manifest.json` (shared `*.json` exclude; needed by `acceptance_examples.py`
     and `example_worker.py`) is force-added in the M5 commit, and M5's checks must run example
     acceptance from a `git archive` export of that commit, so the committed file is exercised.
     `examples/requirements-tested.txt` (package `.gitignore` `*.txt`; referenced by
     `examples/README.md`) was committed on its own in `51414d8` at the user's request, after
     checking each pin against `mb-public-050`: numpy was pinned 2.4.5 but 2.4.3 is installed and
     recorded by every run, so it was corrected to 2.4.3 (all other pins matched).
     Both must be in place before the migration is declared complete.
  4. *Private-input records must never pass silently (user review after M6; fixed).* Before the
     fix, `acceptance.py` skipped hash checks when `private-inputs.json` or an input's entry was
     missing (pre-M6 behaviour), and the M5 example helper accepted an empty hash map. Now one
     helper, `acceptance.private_input_problems`, used by `acceptance.py`, `acceptance_examples.py`
     and `acceptance_regeneration.py`, fails an *available* input of an enabled case for a missing
     record file, a missing entry, an empty or malformed `sha256` map, a malformed digest, a record
     that doesn't name exactly the file (or every file of a directory), and a missing or changed
     member. Absent optional inputs keep their documented skips.
  5. *Examples over-require C24's input (for correction and verification in M9).*
     `acceptance_examples.py run` calls `validate_expected` with every mapped reference, so a
     missing `inputs/open-surrogate.npz` (`required_by` C24) fails example acceptance, although E04
     builds its own surrogate (`lifecycle`) and only compares against C24's frozen *output*. The
     examples should validate the reference artifacts they compare with, not the original inputs of
     those reference cases. Correct it and verify it in the M9 missing-resource matrix (missing
     `open-surrogate.npz`: C24 fails acceptance; E04 still runs).
  6. *Path-length headroom (found in M7b; not changed).* `test_verification.py::
     test_frozen_content_guard` makes a throwaway Git repo under pytest's `--basetemp`
     (`.verification/acceptance/pytest/…`). Its longest Git object path is 257 characters in this
     worktree, 3 below Windows' MAX_PATH (`core.longpaths` is unset), and 270 in a clone under the
     session scratchpad, where the test fails. M9's export and any other deeper checkout need a
     short root, or the test needs a shorter temp location; decide in M9.
  7. *Normal acceptance invokes Git indirectly (user review after M7b; on the M9 checklist, §7).*
     The test behind follow-up 6 runs `git init` and disposable commits inside the installed-target
     pytest set. M9 separates it from the portable gate and verifies acceptance with Git invocation
     blocked, not merely from an export without `.git`.
  8. *Which tests and repeat pairs gate an approval (user review after M7b; decided by the user
     2026-11-10: align with §5).* `e86ff94` enforced the installed-target pytest allowlist (138
     tests) instead of the whole `tests/` suite, and a full-tier approval repeated only the normal
     repeat set, the listed cases and O05/O08. Now `approve` runs the **whole `tests/` suite** on
     the freshly built targets (`run_pytest(full_suite=True)`: contracts configured,
     `MB_EXAMPLE_TDB` set, no failures, errors or skips allowed; `slow` stays deselected), so the
     tests ported in M8 gate approvals automatically. It also repeats **every run case in every
     tier** (O05/O08 as seeded pairs; per-run caps 60 s, F 180 s, O06 300 s, as in a gate run).
     Normal quick/full gates keep their allowlist and repeat set unchanged.
  9. *Deferral decisions (user review after M8, 2026-11-10).*
     - `test_ternary_illingworth_quasi_binary_cu_zn_dummy_matches_binary_brass_case`: deferring
       it in M8 is accepted, but it is a **priority backlog item**. It checks binary/ternary
       agreement over a long run: accepted times, interface motion, profiles, conservation.
       Preferred route: port only the two builders it needs (`build_fig3_present_work_model`,
       `build_fig6_illingworth_case_params`), or reproduce that configuration as a small package
       test fixture, without porting the plotting example.
     - `test_cuniti_tc_python_bundle_smoke`: keep it as an **explicit opt-in check outside the
       mandatory approval gate**. It exercises a distinct provider workflow, but its database
       licences (TCHEA5/MOBHEA4) must not become prerequisites for ordinary solver changes.
     - Recorded in FUTURE_PORTS.md § Deferred tests.
  10. *Reviewer's check of M8 (user, 2026-11-10).* On an export of `42f32b1`:
      - the provenance source and output hashes matched;
      - the 301 ported / 13 deferred count was correct;
      - no ported assertion changed;
      - 62 focused tests passed under `--import-mode=importlib`.
      The full suite was not rerun. The M7b follow-ups were confirmed to address the earlier
      findings.
  11. *Editable worktree install in `mb-public-050` (found in M9b; environment not changed).* Its
      `__editable__.moving_boundary-0.0.1.dev0.pth` puts the worktree's `src` on `sys.path` for
      any process started with normal site processing. Acceptance workers and test runs use
      `-S`, and the acceptance parent never imports `moving_boundary`, so no run used it. But a
      parent started normally from an export would still have the worktree on its path, which the
      guard refuses. Options for the user: remove the editable install (M10 docs would then say
      `pip install -e` is for development only), or document that independence runs start the
      parent with `-S`, as M9 did.
  - *Status after M9:* follow-ups 5, 6 and 7 are resolved (M9a; verified in M9b). Follow-up 11
    is resolved (M9c): the editable install was removed at the user's request.

## Decisions

Confirmed by the user (2026-11-10):

- [x] Future Illingworth solver changes happen only in the package.
- [x] Research main stays frozen at `7af24f4`.
- [x] Approved reference artifacts stay local; their hashes, provenance and reports are committed.
- [x] The frozen research checkout may support optional historical comparisons, but normal package
      tests and acceptance must not require it.
- [x] Research solver modules stay frozen at `7af24f4` behaviour (item 17).
- [x] Item 7: contracts get the molar-volume scale, as run cases do.
- [x] Approval granularity: one approval per change by default (`approve --cases` takes any list).
- [x] Order: item 7, then the approval path (15), then the test port (16), then docs.

Constraints that still apply: never widen tolerances or regenerate references to make tests pass;
don't edit research main; leave historical references untouched; don't capture E11 or change tiers
without approval; don't start R07/R08 until this plan is done.

## Decisions on the former open questions (approved by the user 2026-11-10)

1. **LEGACY (F5):** the package owns its baseline and validation. `reference/LEGACY.npz` is the
   baseline, hash-recorded in `expected_artifacts`, and normal acceptance compares against it.
   The research `illingworth_substitutional_baseline.npz` is used once, in the one-time bitwise
   proof that `LEGACY.npz`'s `binary_*` arrays equal it, and is never needed by normal acceptance.
2. **F1-F4, F6-F9:** relocation with proof, with no numerical approval entries, provided the
   stated equivalence checks pass.
3. **Definitions:** `tools/definitions/`, outside the wheel, with the materializer committed.
4. **Research-generated local inputs:** adopt the existing ones with validated provenance and
   hashes, **then verify that package-owned generation paths recreate them independently** (§3a,
   batch M5b).
5. **`tooling_sha256`:** remove the unused section (in M6).
6. **Item 14:** record supersession in package documentation only; research main stays frozen.

## Review log

- **2026-11-10, Codex review, all points adopted** (sections 2, 5, 7 and 9 revised):
  1. No Git in normal acceptance: ancestry is checked only when an approval is created; M9 exports
     a nonempty approved set.
  2. Crash-safe promotion: immutable versioned approval directories, unique approval IDs, atomic
     manifest replace, interruption tests.
  3. Scratch isolation covers every project-relative write: approvals are tested in a temporary
     project clone.
  4. Correctness gates are defined precisely: limits registered beforehand, conservation of the
     physically conserved quantities, correctness assertions separated from reference comparisons.
  - Clarifications: the O05/O08 hash-seed exceptions are kept; "absent approval section" preserves
    reference selection, but the manifest does change during the migration.
- **Claude's refinements on top of the review:** unlisted cases must still pass their reference
  comparison during `approve`; analytic limits are against the exact solution, not the previous
  reference; the approval scratch project is a `git clone` (since `approve` needs Git); the atomic
  manifest replace handles OneDrive locks.

## 1. Dependency inventory

### Source code read from the research tree in package role

Through `acceptance_inputs.CaseInputs.load()` (AST definition closure), unless noted.

| Consumer | Research file (at `7af24f4`) | Names | Cases |
|---|---|---|---|
| `acceptance_cases.py` | `kawin/tests/illingworth_baseline_cases.py` | `_binary_model`, `_ternary_model`, `_three_phase_model`, `_Coupled…`, `_SmoothBulk…Thermodynamics` | C01, C05-C10 |
| | `kawin/tests/test_illingworth_binary_variable_diffusivity.py` | `_model`, `StrongVariable…`, `CountingVariable…` | C02, C03, C30 |
| | `examples/Illingworth2005/replicate_illingworth2005_fig4_5_6_planar.py` | `PlanarValidationCase`, builder, `_clustered_v_grid`, exact solutions, `VALIDATION_PARAMS` | C04, C31, F04 |
| | `kawin/tests/test_illingworth_ternary_three_phase_similarity.py` | `_solution`, `_build_model` (constant 6001→401) | C11, C12, F11, F12 |
| | `kawin/tests/test_illingworth_interstitials.py` | `_binary_model`, `_VariableBinary…`, `StepProfile1D` | C13-C15 |
| | `kawin/tests/test_illingworth_interstitials_ternary.py` | `_model`, `_closure`, `_diffusivities`, `_SpyTernaryDiffusivity` | C16-C19 |
| | `kawin/tests/test_illingworth_interstitials_three_phase.py` | `_model` | C20, C21 |
| | `kawin/tests/test_illingworth_ternary_three_phase.py` | `_make_nonstationary_unequal_volume_model`, `_make_eta_varying_three_phase_model` | C22, C23, C32 |
| | `kawin/tests/test_surrogate_artifacts.py` | `_surrogate` (only through `prepare`, next row) | C24 |
| `acceptance_worker.py` `--prepare` | `kawin/tests/test_surrogate_artifacts.py` | `_surrogate` → writes `inputs/open-surrogate.npz` | C24 input |
| `acceptance_worker.py` LEGACY | `illingworth_baseline_cases.py` | `_binary_model`, `_collect` | LEGACY |
| `acceptance_contracts.py` | `test_illingworth_setup_and_diagnostics.py`, `test_illingworth_ternary_three_phase.py`, `test_illingworth_binary_variable_diffusivity.py` | 3 + 4 + 2 research **test functions**, run as the contract assertions | C25-C29 |
| `acceptance_optional.py` | `examples/ternaryExamples/exb5_gamma_alpha_gamma.py` | `run_kawin2`, `run_kawin3` (constants changed, constructor cut short) | O01, O02 |
| | `examples/Interstitials/fe_c_exb1a.py` | `run_kawin` (same treatment) | O07 |
| | `examples/Interstitials/ternary_ferrite_growth_le.py`, `fe_mn_c_ferrite.py`, `three_phase_symmetric_check.py` | `run_kawin`, `tie_line_closure`, `TabulatedTernaryDiffusivity`, `CASE`, `_AusteniteFerriteClosure` | O03a, O03b |
| | `examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_AnalyticComparison.py`, `ternaryTwoPhaseSimilarity.py` | comparison builders (module globals overridden), `FixedMatrixTernaryDiffusivity` | O04, O05 |
| `acceptance_thermocalc.py` | the same two FeCrNi files | model assembly after generation | O06 |
| `CaseInputs.symbol()` | `kawin/tests/*`, `examples/*` | names those closures import from other research files, resolved at run time | all of the above |

Not research dependencies in package role:

- O05/O08/O06 *generation* already uses `moving_boundary.providers.*`.
- `binary_cases.py` uses the package's own `examples/binary_constant.py`.
- `collect()` is package code.
- The package-role example worker already blocks `examples.*` imports.

### Research needed only for structural reasons

- `acceptance.py main()` always calls `validate_reference()`: detached, clean HEAD, 82 `source_sha256` hashes.
- The worker and example worker require `--root`.
- `REFERENCE` defaults to `PROJECT.parents[2] / "mb-ref-7af24f4"`.
- `check_repeatability()` and the contract repeatability check run the **reference** role.
- `capture-settings` builds research cases in the reference role.
- `tools/verify.py` (`FORK`, `EXPECTED = ae9fb29`, `check_baseline`) is the older fork verifier.
  Classify in M0; expected to be historical-only.

**The extraction worktree is itself a research-layout checkout** (`kawin/` and `examples/` at its
root), so independence can only be proven from an exported copy of the package, not in place.

### Local artifacts (not source code)

| Artifact | Origin | Hashed today? |
|---|---|---|
| `FeCrNi.tdb` | supplied separately (Lee TDB, not redistributed) | in `private-inputs.json` when present |
| TC-Python installation, licence, `inputs/O06.json` | supplied separately (licensed) | O06 config checked by `load_configuration` |
| `exb5.json`, `femnc.json`, `fec.json`, `fecrni.surrogate/` | TC-derived private caches | yes, `private-inputs.json` |
| `open-surrogate.npz` | **made by research** (`_surrogate`, reference-role `prepare`) | **no** |
| `pycalphad-reference.npz` (E05 surrogate) | **made by research** (reference-role O05) | **no** |
| `reference/*.json,npz` (all cases) | research reference role at `7af24f4` | only O06/O08 |
| `example_settings/*.settings.json` | research reference role | **no** |
| `kawin-0.5.0` wheel | PyPI | yes (`setup_targets`) |

The manifest's `tooling_sha256` section is stale and nothing reads it.

## 2. Target design

**Normal acceptance:** `acceptance.py quick|full`, `acceptance_examples.py run`, and the package `tests/`.

- Installed wheel on stock kawin 0.5.0, with package-owned definitions in `tools/definitions/`
  (not shipped in the wheel).
- Expected artifacts are local and hash-validated.
- No research root, no Git metadata of any checkout, no `acceptance_inputs` import in package role.
  The worker asserts that last point.

**Historical comparison** (explicit opt-in, e.g. `acceptance.py historical capture|repeat|compare --research <mb-ref-…>`):

- Keeps today's reference-role code unchanged: `CaseInputs`, AST loading, research `source_sha256`,
  detached-clean checks.
- `acceptance_inputs.py` becomes historical-only and is not edited.
- Research references are never rewritten (`capture` keeps refusing to overwrite).

**Validation is split into two functions:**

- `validate_expected()` runs on every normal run. It checks:
  - a new manifest section `expected_artifacts` (sha256 + provenance for each `reference/<case>` pair);
  - the existing O06/O08 frozen hashes;
  - `approved_references`: the committed approval record (ID, package commit, reason, cases,
    versioned directory, artifact hashes) and the presence and hash of its approval report.
    **No Git:** whether `package_commit` is an ancestor of HEAD is checked only when `approve`
    creates the approval, in the development repository. An optional `acceptance.py audit` re-runs
    the Git checks where a repository exists, but it is not part of normal acceptance;
  - input specs: `private-inputs.json` plus new hashes for the adopted research-generated inputs.
    `open-surrogate.npz` (synthetic) and the settings of non-private cases are hashed in the
    committed manifest. `pycalphad-reference.npz` (Lee TDB-derived) and the O04/O05 settings are
    hashed in the local `private-inputs.json`, following the existing rule that private hashes
    never enter committed evidence;
  - `definitions-provenance.json`: each definition file's hash, source path, source sha256 at
    `7af24f4`, and the transforms applied;
  - `contract_model_scales`.
- `validate_research_checkout()` runs only for historical actions: today's detached/clean/HEAD/82-hash checks.

**Repeatability:** normal runs repeat in the **package** role (two unseeded package processes,
bit-identical). The O05/O08 exceptions stay as they are: both run with `PYTHONHASHSEED=0` and are
excluded from unseeded repeatability, as `check_repeatability()` already asserts. Repeating the
research reference moves to `historical repeat`.

**Manifest changes:** the migration adds `expected_artifacts`, `contract_model_scales` and the
definitions provenance, so the manifest *does* change. What stays fixed: with `approved_references`
absent, reference selection is exactly today's (every case reads `reference/<case>`).

## 3. How definitions migrate

- **Generate, don't hand-copy.** A new historical tool, `tools/materialize_definitions.py`, needs
  `--research`. For each of the ~26 distinct `inputs.load(...)` calls, it emits the selected
  definitions as **verbatim source segments** (comments kept) into `tools/definitions/<group>.py`.
  It then applies today's transforms as small, recorded edits: constant substitution, constructor
  cut-off, policy keyword injection, module-global overrides, and statically resolved package
  imports in place of `_input_symbol`.
- **Proof of equivalence:** for each call, the AST of the emitted module (imports normalized) must
  equal the AST that `BindInputs` produces today. Any mismatch stops the batch.
- **Reusing package equivalents:** only where the source is AST-identical, e.g.
  `moving_boundary.analytic.FixedMatrixTernaryDiffusivity` against the research
  `ternaryTwoPhaseSimilarity` class. Otherwise the copy stays. Package examples
  (`binary_planar_growth`, `three_phase_similarity`, ...) are **not** used as substitutes, because
  their construction differs from the reference cases (see E05/E06).
- **A small runtime binding module, `definitions/_bindings.py`** (no AST, no root), provides model
  classes (with the model tracking the contracts need), `terminal_thin_phase_policy`, artifacts
  and family.
- **Provenance:** the header of each file, plus `docs/definitions-provenance.json`, records the
  source path, `7af24f4`, source sha256 and transforms. The materializer and its AST-equality
  report are committed.
- **Accepting a batch:** that batch's cases must be bit-identical against the existing, now
  hash-validated, references in exact mode, with no new references captured. For the E-mapped
  cases, `effective_settings` from the new definitions must also equal the recorded settings files.

### Historical runs must not overwrite adopted inputs

Today `capture` (any run except O06-only) rewrites `inputs/open-surrogate.npz` through
`prepare`, and every O05 worker rewrites `inputs/pycalphad-<role>.npz`. So after adoption:

- historical `prepare` and reference-role O05 sampling write to a scratch folder;
- package-role O05 writes its transient `pycalphad-package.npz` to the worker's scratch folder,
  not to `inputs/`;
- `validate_expected()` rejects any change to an adopted input.

`inputs/pycalphad.npz` (2026-11-05) is referenced by no tool, test, example or doc. It is not
adopted, it is left untouched, and it is reported in M0.

## 3a. Regeneration verification of adopted inputs (batch M5b)

**Purpose:** show both that the historical inputs were preserved (hashes unchanged) and that the
package can recreate them without research source files. This is verification only: the adopted
artifacts and their hashes never change, and no reference or tolerance is refreshed.

**When:** after the builders and generation paths have migrated (M1-M5), before M6 and well
before R07/R08.

### Inventory

| Artifact | Adopted from | Package generation path | Already covered by the plan? | Check added in M5b | Needs |
|---|---|---|---|---|---|
| `inputs/open-surrogate.npz` | research `_surrogate` via reference-role `prepare` | `tools/definitions` copy of `_surrogate`, package-role `prepare` (M3) | **Yes:** F7 array identity (M3) | Extend F7: key set, dtype/shape/bytes per array, and the surrogate's metadata payload, each recorded; ZIP container compared separately | none |
| `inputs/pycalphad-reference.npz` (E05 surrogate) | research reference-role O05 sampling (research `pycalphad_default_phase_adapter`) | `moving_boundary.providers.pycalphad` via the package O05 sampling path, `PYTHONHASHSEED=0` | **Partly.** The O05 exact gate compares the package sample's numeric `provider/*` arrays with `reference/O05.npz`. It never checks the adopted file itself or its non-numeric keys. | Generate into scratch and compare with the adopted file: every key, non-numeric included, plus metadata. Also confirm the adopted file's numeric arrays equal `reference/O05.npz` `provider/*`. | `FeCrNi.tdb` |
| `example_settings/*.settings.json` (10: C04, C11, C12, C16, C24, F04, F11, F12, O04, O05) | research `reference_settings()` (reference role, build only, no solve) | `effective_settings()` on the case built from package definitions (no solve) | **Partly.** §3 checks the E-mapped cases per migration batch, but not as one recorded check of all 10 against the adopted files. | Regenerate all 10 into scratch; exact JSON equality with the adopted files (`sort_keys`, same serializer); record per-file results | O04: `fecrni.surrogate`; O05: `FeCrNi.tdb` |
| `reference/*` (expected outputs), O06/O08 fixtures | research reference role; O06 package-verified | normal acceptance in package role | **Yes:** exact/byte gates every run (O06 with live TC, O08 byte-only) | none | O06: TC; O08: `FeCrNi.tdb` |
| `exb5.json`, `femnc.json`, `fec.json`, `fecrni.surrogate/` | research Thermo-Calc/pycalphad study scripts (sources in `private-inputs.json`) | **none:** the generators are research study workflows, tracked in [FUTURE_PORTS.md § Cache rebuilding gaps](FUTURE_PORTS.md#cache-rebuilding-gaps) | n/a | **Not checked; reported explicitly** as "no package generation path (deferred)", which does not count as verified. Hashes stay validated. | — |
| `FeCrNi.tdb`, `O06.json`, TC installation and licence | supplied separately | n/a (inputs, not generated) | n/a | none | — |

### Method

- **Same inputs:** the adopted `FeCrNi.tdb`, private inputs and manifest configuration,
  hash-validated before generation.
- **Same backends:** record pycalphad, numpy, scipy, symengine and kawin versions in the report.
  On 2026-11-10 the research `kawin` env and `mb-public-050` match (pycalphad 0.11.1, numpy
  2.4.3, scipy 1.17.1, symengine 0.13.0); kawin differs by design (fork 0.4.0 vs stock 0.5.0). If
  a backend version differs at run time, report the check as not comparable instead of passing it.
- **Hash-seed rules:** O05 sampling under `PYTHONHASHSEED=0` (the existing exception); everything
  else unseeded.
- **Scratch only:** generate under the M5b scratch folder; nothing is written to `inputs/` or
  `example_settings/`.
- **NPZ comparison:**
  1. **Array identity:** same key set, and per key the same dtype, shape and bytes
     (`bit_identical`), including non-numeric arrays.
  2. **Container metadata** (member order, ZIP timestamps, compression), reported separately as
     metadata-only differences, never as numerical ones.

  Where an existing gate is exact or byte-level (O06, O08), that requirement is kept unchanged.
- **Embedded metadata** (e.g. surrogate metadata JSON): compared decoded, field by field. The list
  of allowed differences starts **empty**; adding to it needs the user's approval, with a reason.
- **Unexpected differences are investigated and reported.** They never trigger replacing the
  adopted artifact.
- **Optional resources:** follow the existing policy. A missing `FeCrNi.tdb` or
  `fecrni.surrogate` gives an explicit "unavailable: <reason>" entry, never a silent pass.
- **"Unavailable" and "not comparable" never count as verified.** Each check has exactly one
  outcome: `verified` (ran and matched), `failed`, `unavailable` (optional resource missing) or
  `not comparable` (e.g. a backend version differs). Only `verified` satisfies the completion
  criterion; anything else stays open and is listed.
- **This only affects regeneration checks, not normal acceptance.** A missing *optional* resource
  can prevent a regeneration check, but a missing *required* acceptance input still fails normal
  acceptance as documented in §7. For example, a missing `open-surrogate.npz` fails C24; it is
  never skipped, and it never falls back to regeneration.
- **Report:** numerical details go in a local `.verification/acceptance/regeneration-<sha>.json`.
  The committed summary in `docs/` records per-artifact pass / fail / unavailable and container
  notes, with TDB/TC-derived values redacted as in `record_deviations`.

### Cases flagged as non-mechanical

None of these changes inputs or numerical behaviour by design, but none is a plain verbatim move.
Approved 2026-11-10 as relocation with proof (Decisions 1-2), provided each listed guard passes.
If a guard fails, stop and report; don't fall back to an approval.

| # | Case(s) | Why it isn't verbatim | Guard |
|---|---|---|---|
| F1 | C11, C12, F11, F12 (6001→401); O01, O02 (4001/8001/3.6e6); O07 (4001); O03a, O03b (2001) | constant substitution becomes explicit code | AST equality + exact gate |
| F2 | O01, O02, O07, O03a, O03b | constructor cut-off (`return model` after the assignment) becomes explicit code | same |
| F3 | all ternary and three-phase constructions | `terminal_thin_phase_policy` injection becomes an explicit keyword | same |
| F4 | O04, O05, O06 | module-global overrides (`surrogate`, `PHASE_A/B`, `PHYSICAL_NODES=401`) become function parameters | same; O06 exact needs live TC |
| F5 | **LEGACY** | **Gate semantics change.** Today the package worker asserts exact equality with the research `illingworth_substitutional_baseline.npz` *inside the worker*. Proposal: drop that in normal mode, because `compare()` against the hashed `reference/LEGACY.npz` covers the same `binary_*` arrays. A one-time migration check proves `LEGACY.npz` `binary_*` equals the baseline bitwise. Historical mode keeps the assertion. | one-time bitwise proof, logged |
| F6 | C25-C29 | The assertions are research **test functions** and are copied as such. Any edit needed to run them without research is flagged per function; no silent edits to assertions. | completed-assertion lists identical + exact gate |
| F7 | C24 / `prepare` | `open-surrogate.npz` is research-generated. Port `_surrogate`; a package-role `prepare` writes to scratch and must match the existing file **array-wise** (npz zip timestamps differ by bytes). The existing file stays the hashed input. | array identity |
| F8 | O05, O08 | Hash-seed-sensitive generation. The research assembly that follows it is replaced, and the generation guard becomes strict throughout. Must stay bit-identical (O08 is a byte gate). If not, stop and report; nothing gets approved. | exact/byte gate |
| F9 | E05 | consumes the research-made `pycalphad-reference.npz`; it becomes a hashed local input, unchanged | hash |

## 4. Item 7 without touching the historical references

- **No edits to `reference/C25-C29.json`.** Molar volume and length are *inputs*, so the per-model
  scales go into a committed manifest section,
  `contract_model_scales: {C25: {model0: {length_scale, minimum_molar_volume}, ...}}`.
- **Capturing them:** the package-role worker records `model_scales` from the package definitions.
  Historical mode cross-checks them against a reference-role run.
- **Explicit validation in `compare()`:**
  - the package report's `model_scales` must equal the manifest exactly;
  - the keys must match the npz's `model{i}/` prefixes exactly, none missing and none extra;
  - values must be finite and positive;
  - the run fails otherwise.
- **Before adoption:** show the user each contract's resulting allowance against today's,
  flagging any that **tighten** (length < 1 m).

## 5. Approval path (item 15)

### Isolation of approval tests

- Synthetic approvals run in a **temporary project clone**: a local `git clone` of the extraction
  branch into the scratchpad, plus a copy of `.verification/` without the `install-*` folders.
  `approve` needs Git (clean tree, HEAD, ancestry), so a plain file copy is not enough. Every
  project-relative write therefore lands in the clone: manifest, `docs/approvals/`, reports,
  `example_settings`, `reference-approved/`.
- The tests record sha256 and mtime of the real manifest, `docs/approvals/`,
  `docs/acceptance*.json`, `example_settings/`, `reference/` and `reference-approved/` before and
  after, and fail on any change.
- Unit tests (`tests/test_acceptance_approval.py`) use `tmp_path` project trees only.
- **The clone must contain the complete candidate implementation.** A clone only sees committed
  content, and extraction-branch commits need the user's approval. So either use user-approved
  checkpoint commits, or copy only the implementation-owned changes (modified and new files)
  byte for byte into the scratch clone and make **test-only commits there**. Exclude the user's
  personal Settings edits, `AGENTS.md` and `branch_status/`. Scratch commits don't authorize
  extraction-branch commits. Record the snapshot tested: base commit, scratch SHA, copied paths
  with sha256. The same applies to the M9 export.

### Correctness gates vs reference comparisons

These are two separate responsibilities. An intended change must fail the comparison for the
cases it changes, so requiring every comparison to pass would be circular.

- **Correctness assertions** must pass for every case, approved or not:
  - the package `tests/`;
  - the contract *assertion groups* (C25-C29 test functions). These are separated from the
    contracts' array comparison: today `contract_case()` does both;
  - two package-role runs bit-identical (O05/O08 under their hash-seed rule);
  - analytic error **against the exact solution** for C04, C11, C12, F04, F11 and F12 within
    limits registered in the manifest (`approval_criteria`). The limits are set from the current
    references and committed **before any change is made**. They are not computed relative to the
    previous output;
  - conservation of the **physically conserved quantity**: molar inventories (`molarInventoryData`,
    `totalMolesData`, `dependentMoleClosureErrorData`) for explicit-volume and unequal-volume cases
    (C09-C12, C20-C23, C32, F11, F12, O02, O03b), and composition inventories only where they are
    conserved (equal molar volumes). Unweighted composition integrals in unequal-volume cases are
    reported, not gated. The thresholds are registered in `approval_criteria` alongside the
    analytic limits and are not borrowed from reference-comparison tolerances.
- **Reference comparisons:**
  - the **listed** cases are *reported* (per-array deviation report against the previous expected
    artifact, referenced by the LEDGERS entry) and are allowed to differ;
  - **every unlisted case must still pass** at public tolerance, so a change can't leak into
    families that weren't approved.
- The `approval_criteria` values are shown to the user and committed at M7, before any real approval.

### Crash-safe, recoverable promotion

- **Unique approval ID** per approval: `<UTC timestamp>-<short package sha>-<counter>`, so several
  approvals at one commit can't collide.
- **Immutable versioned directories:** `reference-approved/<approval-id>/<case>.{json,npz}`.
  These are written once and never overwritten or deleted by the tool.
- **Steps:**
  1. Write the files into `reference-approved/<approval-id>.partial/`.
  2. Validate hashes and reload the arrays.
  3. Rename to `reference-approved/<approval-id>/`.
  4. Write the new manifest to a temp file in the same folder and `os.replace` it over the
     manifest. This is the single commit point. OneDrive sync locks can make the replace fail; the
     writer retries briefly, then fails loudly with the old manifest still valid.
- The manifest maps each case to the approval ID currently in effect and keeps the full history
  of earlier approvals.
- `approve --rollback <approval-id>` is just another atomic manifest replace pointing back at an
  earlier version. No files move.
- A crash before step 4 leaves the old manifest pointing at unchanged, hash-valid files; leftover
  `.partial` or unreferenced folders are reported and can be cleaned up.
- **Interruption tests:** inject a failure after each of steps 1-4 (and during the replace) and
  assert that `validate_expected()` passes on the old state and that a retry succeeds.

### Other rules

- **Untouched historical references:** nothing ever writes under `reference/`.
  `validate_expected()` checks `expected_artifacts` on every run.
- **Reference selection:** one helper, `expected_reference()`, used by `compare()`,
  `contract_case()` and `acceptance_examples.compare_case()`. It routes the O06/O08 hashes and
  LEGACY, and selects package-captured settings for approved E-mapped cases. With
  `approved_references` absent, selection is exactly today's.

Adoption procedure for a deliberate change: change + tests on the extraction branch → public quick
to list failing cases and magnitudes → commit → `approve --cases … --reason "LEDGERS Rxx"` from a
clean tree (correctness gates + unlisted-case comparisons enforced) → exact quick/full bit-identical
against the mixed reference set → public quick/full `--record` → examples → LEDGERS entry →
separate "approve references" commit.

## 6. Item 16 test port

- **Shared helpers:** reuse `tools/definitions/*` where a ported test needs the same helper
  (loaded by path); otherwise copy verbatim with provenance. The R06 AST-copy generator was not
  found and will be rewritten.
- **Batches:**
  - [x] 1: setup_and_diagnostics (6), three_phase (62), interstitials_three_phase (2),
        **surrogate_artifacts (12; added, covers the C24 builder)** (done 2026-11-10: 81 of 82;
        the TC licence smoke test deferred, see log)
  - [x] 2: binary_variable_diffusivity (14, plus `test_diffusion` helpers already reproduced in
        `test_binary_deviation_form.py`), interstitials (13), transformed_grids (16) (done
        2026-11-10: 43 of 43, see log)
  - [x] 3: similarity group (34 across 7 files: three_phase_similarity 8, interstitials_ternary 8,
        physical_scale 6, convergence 5, three_phase_examples 3, similarity_comparison_basis 2,
        ternary_similarity 2). Tests that only check research example scripts are flagged, not forced.
        (done 2026-11-10: 26 of 34, the 8 deferred whole as planned; see log)
  - [x] 4: ternary (155) (done 2026-11-10: 151 of 155; one deferral beyond the planned 3, see log)
- **Before batch 1 (M0):** check the other research `test_surrogate*.py` files and
  `test_fecrni_surrogate_artifact_recipe.py` for solver-relevant coverage and report them, without
  porting them by default.
- **Tests that need unported functionality are deferred, not skipped.** They are listed with their
  feature in [FUTURE_PORTS.md § Deferred tests](FUTURE_PORTS.md#deferred-tests) and are not counted
  as ported:
  - the 3 surrogate-construction diagnostic tests in `test_illingworth_ternary.py` (batch 4 ports
    152 of 155);
  - `test_illingworth_ternary_convergence.py` (5) and `test_three_phase_examples.py` (3), deferred
    whole. Batch 3 therefore ports 26 of 34.

  The generator also drops the deferred tests' module-level imports. Each ported file's docstring
  names its exclusions, and the batch report gives ported / deferred / total per file. The rest of
  the similarity group is portable: its helpers already exist in `moving_boundary.analytic`.
- Deferred features (diagnostics, comparison, benchmarking, plotting/convergence, training data,
  study scripts) are **not** part of M0-M10. None of them is needed by normal acceptance
  (FUTURE_PORTS.md, "Does current acceptance need any of this?").

## 7. Independence proof (M9)

- **Export:** `git archive` of `packages/moving-boundary` at the commit under test (a user-approved
  checkpoint commit, or the test-only commit of the §5 scratch clone holding the complete
  candidate implementation; record which), into a scratch
  folder with no kawin checkout above it. Then copy only `.verification/` (local artifacts, wheel,
  `public-environment.json`).
- **Guard (`MB_FORBID_RESEARCH=1`):** a worker and conftest guard that blocks:
  - imports of `examples.*`, `kawin.tests.*`, `acceptance_inputs`, and the research-only
    `kawin.diffusion.MovingBoundary*` / `kawin.diffusion._illingworth_*` modules;
  - via `sys.addaudithook`, every `open` of a path under the research checkout, the extraction
    root's `kawin/` and `examples/`, or any `mb-ref-*`.
  - `REFERENCE` is unset. Unit tests prove the guard catches a deliberate read and a deliberate import.
- **The export has no `.git`,** so any Git call in normal acceptance fails the run.
- **Git invocation blocked, not only absent (user review after M7b, follow-up 7):** an export
  without `.git` does not catch a test that runs `git init` itself. The installed-target pytest
  set includes `test_verification.py::test_frozen_content_guard`, which makes disposable
  repositories and commits, so normal acceptance still depends on Git. M9 moves that historical
  test out of the portable gate, and runs normal acceptance with every `git` invocation blocked
  (e.g. a `git` shim first on `PATH` that fails loudly, plus an audit hook on `subprocess`).
  Any Git call outside `approve`, `historical` and the scratch tools fails the run.
- **Runs:** package `tests/`, normal `quick` and `full` (exact), and
  `acceptance_examples.py run --skip-tests`, then with tests.
- **Ported tests run on their own, from the export with Git blocked (user review after M8):** normal
  `quick`/`full` run only their smaller pytest allowlist, so passing them does not show that the
  M8 port (`tests/test_illingworth_*.py`, `test_transformed_grids.py`,
  `test_surrogate_artifacts.py`, `test_similarity_comparison_basis.py`, `tests/ported_helpers/`)
  works without research or Git. Run the whole `tests/` suite, and the ported files as their own
  reported run, from the `git archive` export with the research guard on and every `git`
  invocation blocked. Report ported pass/fail counts separately from the normal quick/full
  results.
- **The same checks with an active synthetic approval:** with the nonempty approved set below in
  effect, repeat the Git-blocked export runs (normal quick and the ported tests). Approved cases
  must read their hash-validated versioned files and every other case `reference/`, without Git.
- **Nonempty approved set:** a second export carries a synthetic approval (made in the M7 scratch
  clone, e.g. one quick case and one contract). Quick must select and hash-validate the approved
  versioned files without Git, and the other cases must still read `reference/`. This exposes any
  hidden Git or ancestry dependency before the first real approval.
- **Missing-resource matrix** (each on its own scratch copy):

| Removed | Expected outcome |
|---|---|
| `FeCrNi.tdb` | O05, O08 skipped "database input missing" (E05 does not read the TDB; corrected after M9b, user-approved 2026-11-11) |
| `pycalphad-reference.npz` | E05 skipped "local input missing" (its frozen surrogate input) |
| `exb5.json` | O01, O02 skipped |
| TC licence or config | O06 skipped, with the probe reason |
| `open-surrogate.npz` | C24 **fails** with a clear, documented message (quick-required); **E04 still runs and passes** in example acceptance (it builds its own surrogate and compares with C24's frozen output; follow-up 5) |
| one `reference/*.npz` byte | hash failure |
| `reference-approved` file | hash failure |

No case may fall back to research files; the guard turns any attempt into a failure.

## 8. Affected files

- **tools:**
  - `acceptance.py`: split validation; `historical`/`approve` actions; package-role repeatability;
    reference selection; overrides.
  - `acceptance_worker.py`: package role without `--root` or `acceptance_inputs`; LEGACY F5;
    model scales; prepare.
  - Also: `acceptance_cases.py`, `acceptance_contracts.py`, `acceptance_optional.py`,
    `acceptance_thermocalc.py` and `acceptance_default_phase.py` (role-dispatch to definitions),
    `acceptance_examples.py`, `example_worker.py`, `batch_safety.py` (guard),
    `acceptance_manifest.json`, `example_manifest.json`.
  - New: `tools/definitions/*.py`, `tools/materialize_definitions.py`.
  - `acceptance_inputs.py` stays unchanged.
- **tests:**
  - `test_acceptance_contracts.py`: reference optional.
  - `test_thermocalc_acceptance.py`, `test_default_phase_acceptance.py`: they call
    `tool.worker('O06','reference')`, so the signature changes.
  - New: `test_acceptance_approval.py`, `test_acceptance_independence.py`, plus the ported test files.
- **docs:** LEDGERS (Ownership and provenance, an item 7 entry, migration log), NEXT_SESSIONS
  (definition of done: "matches the frozen research oracle, except cases with logged approved
  references"), the proposal (rewritten to match what was built), new
  `definitions-provenance.json`, new `docs/approvals/`, `acceptance.json` and
  `acceptance-deviations.json` (final record only), and the branch ledger via `branch-status`.

## 9. Batches, validation, runtime

Every step stays under 30 min; long runs go to the background.

Revised 2026-11-10 (user): from M3 on, every batch also runs the full package `tests/` suite in
the working tree and from a `git archive` export of its commit (about 2 + 2.5 min). "Est." is the
batch's own gates; "Est. incl. suites" adds roughly 4-6 min and is the figure the overrun stop
condition is measured against (stop and report on a substantial unexpected overrun beyond it).

| Batch | Content | Validation | Est. | Est. incl. suites |
|---|---|---|---|---|
| M0 | Classification (`verify.py`, surrogate tests), materializer, `expected_artifacts` hashes of current files | historical `repeat` confirms the research role still reproduces `reference/` before hashing | 15 min | done |
| M1 | Definitions for C01-C23, C30-C32, LEGACY (F1-F3, F5) | AST equality; `quick --cases … --tolerance exact` | 8 min | done |
| M2 | C25-C29 assertions + item 7 scales (F6) | assertion lists; exact contracts; scale table for the user | 5 min | done |
| M3 | C24, prepare (F7) | array identity; exact C24 | 3 min | done (~12 min measured) |
| M4 | F04/F11/F12, O01-O04, O07, O05/O08 (F8), O06 (F4, live TC) | exact full subset | 12 min | 16-18 min |
| M5 | Example worker + hashed settings; historical writes redirected to scratch | `acceptance_examples run --skip-tests`, E01-E10 | 2 min | 6-8 min |
| M5b | Regeneration verification (§3a): `open-surrogate.npz`, `pycalphad-reference.npz`, 10 settings files | adopted hashes unchanged; array identity + metadata per artifact; unavailable checks listed | 10 min | 14-16 min |
| M6 | Normal/historical split, package-role repeatability; remove `tooling_sha256` | exact quick + exact full (30/40 runs, 5 contracts, contract-pytest 89+N) | 20 min | 24-26 min |
| M7 | Approval path, `approval_criteria` (shown to the user before commit), contract assertion/comparison split | unit tests; interruption tests; synthetic approval and rollback in a temporary project clone; real-path hashes unchanged | 25 min | 29-31 min |
| M8 | Test port, 4 batches | each batch green; full `tests/` (311 + ported) | 4 × 3-8 min | 4 × 7-14 min |
| M9 | Independence proof (empty and nonempty approved sets) + missing-resource matrix | as §7 | 40 min, split into ≤12 min runs | 44-46 min |
| M10 | Docs; **final recorded gates after all changes**: exact quick/full, public quick/full `--record`, examples with tests, full `tests/` | all green; report | 45 min | 49-51 min |

About 3.5-4 h of gate time in total (M5b adds about 10 min), plus about 45-65 min of full-suite
runs from M3 on.

## Completion criteria

The authority-switch work is complete only when all of these hold, with results recorded:

- [x] Exact quick and full are bit-identical (30/40 runs, 5 contracts) using package-owned
      definitions; E01-E10 pass; the full package `tests/` suite is green, with ported/deferred
      counts reported per file. (Met 2026-11-11, M10.)
- [x] LEGACY's one-time bitwise proof is recorded; normal acceptance uses the hashed `LEGACY.npz`.
      (Met in M1: 33 `binary_*` keys bit-identical; see log.)
- [x] Every adopted input and expected artifact keeps its original hash (validated before and
      after every batch). (207 files unchanged through M10.)
- [x] **Regeneration verification (M5b; met 2026-11-10, 12/12 `verified`, see log):** `open-surrogate.npz`, `pycalphad-reference.npz` and the
      10 example settings files are regenerated by package-owned paths without research source
      files and match the adopted originals (array identity plus metadata; container differences
      reported separately), each with outcome `verified`. Unexpected differences have been
      investigated and reported, and no artifact was replaced.
      - `unavailable` and `not comparable` outcomes **do not meet this criterion.** They are listed
        with reasons as open items, and the check is rerun once the resource is available.
      - The four caches without package generators are reported as known gaps and tracked in
        FUTURE_PORTS.md. They are outside this criterion, and they are not verified.
- [x] A missing required acceptance input (e.g. `open-surrogate.npz` for C24) fails normal
      acceptance (shown in the M9 missing-resource matrix); it is not skipped. (Met 2026-11-11, M9b.)
- [x] The independence proof (M9) passes with empty and nonempty approved sets, and the
      missing-resource matrix behaves as documented. (Met 2026-11-11, M9b/M9c; E05 row corrected
      with the user's approval.)
- [x] The approval path (M7) passes its unit, interruption and isolated-clone tests (met
      2026-11-10, M7b; see log).
- [x] `tooling_sha256` is removed; supersession is recorded in package docs only (LEDGERS,
      NEXT_SESSIONS definition of done, branch ledger). (Removed in M6; recorded in M10.)
- [x] Final recorded gates after all changes (M10): exact quick/full, public quick/full
      `--record`, examples with tests. (Met 2026-11-11.)

**Budget risk:** `full` has an internal 1200 s limit. If package-role repeatability plus the guard
push it over, report the measured time and ask; don't raise the budget unilaterally.

## Out of scope

R07, R08, R09-R12 (items 1-3); items 4-6; E11, E-case tiers, Lee TDB, Linux checks (items 8-11);
research issues R01-R04 (item 12); moving the oracle again; editing research main; pushing; every
entry in [FUTURE_PORTS.md](FUTURE_PORTS.md) (deferred until after R07/R08).
