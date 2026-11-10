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
| M5 | Example worker + hashed settings | not started |
| M5b | Regeneration verification of adopted inputs | not started |
| M6 | Normal/historical split, package-role repeatability | not started |
| M7 | Approval path | not started |
| M8 | Test port (item 16), 4 batches | not started |
| M9 | Independence proof + missing-resource matrix | not started |
| M10 | Docs + final recorded gates | not started |

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
- **Follow-ups recorded 2026-11-10 (user review):**
  1. *`docs/definitions-provenance.json` is git-ignored.* It matches `*.json` in the shared
     `.git/info/exclude` (research checkout, line 13); the other `docs/*.json` are tracked only
     because they were force-added. The loader needs it, so: stage it explicitly (`git add -f`) in
     the M1/M2 commits; build the M7/M9 scratch snapshot from the owned-file list plus
     `git status --porcelain --ignored`, not plain `--porcelain`; and the M9 `git archive` export
     must contain it (an export without it fails every package-definition case, which the M9 runs
     would expose). Also check any other new JSON (e.g. `docs/approvals/`) the same way.
  2. *Remove research-solver fallbacks from `_bindings.py` (M6, before M9 independence).*
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
  - [ ] 1: setup_and_diagnostics (6), three_phase (62), interstitials_three_phase (2),
        **surrogate_artifacts (12; added, covers the C24 builder)**
  - [ ] 2: binary_variable_diffusivity (14, plus `test_diffusion` helpers already reproduced in
        `test_binary_deviation_form.py`), interstitials (13), transformed_grids (16)
  - [ ] 3: similarity group (34 across 7 files: three_phase_similarity 8, interstitials_ternary 8,
        physical_scale 6, convergence 5, three_phase_examples 3, similarity_comparison_basis 2,
        ternary_similarity 2). Tests that only check research example scripts are flagged, not forced.
  - [ ] 4: ternary (155)
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
- **Runs:** package `tests/`, normal `quick` and `full` (exact), and
  `acceptance_examples.py run --skip-tests`, then with tests.
- **Nonempty approved set:** a second export carries a synthetic approval (made in the M7 scratch
  clone, e.g. one quick case and one contract). Quick must select and hash-validate the approved
  versioned files without Git, and the other cases must still read `reference/`. This exposes any
  hidden Git or ancestry dependency before the first real approval.
- **Missing-resource matrix** (each on its own scratch copy):

| Removed | Expected outcome |
|---|---|
| `FeCrNi.tdb` | O05, O08, E05 skipped "database input missing" |
| `exb5.json` | O01, O02 skipped |
| TC licence or config | O06 skipped, with the probe reason |
| `open-surrogate.npz` | C24 **fails** with a documented message (quick-required) |
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

- [ ] Exact quick and full are bit-identical (30/40 runs, 5 contracts) using package-owned
      definitions; E01-E10 pass; the full package `tests/` suite is green, with ported/deferred
      counts reported per file.
- [ ] LEGACY's one-time bitwise proof is recorded; normal acceptance uses the hashed `LEGACY.npz`.
- [ ] Every adopted input and expected artifact keeps its original hash (validated before and
      after every batch).
- [ ] **Regeneration verification (M5b):** `open-surrogate.npz`, `pycalphad-reference.npz` and the
      10 example settings files are regenerated by package-owned paths without research source
      files and match the adopted originals (array identity plus metadata; container differences
      reported separately), each with outcome `verified`. Unexpected differences have been
      investigated and reported, and no artifact was replaced.
      - `unavailable` and `not comparable` outcomes **do not meet this criterion.** They are listed
        with reasons as open items, and the check is rerun once the resource is available.
      - The four caches without package generators are reported as known gaps and tracked in
        FUTURE_PORTS.md. They are outside this criterion, and they are not verified.
- [ ] A missing required acceptance input (e.g. `open-surrogate.npz` for C24) fails normal
      acceptance (shown in the M9 missing-resource matrix); it is not skipped.
- [ ] The independence proof (M9) passes with empty and nonempty approved sets, and the
      missing-resource matrix behaves as documented.
- [ ] The approval path (M7) passes its unit, interruption and isolated-clone tests.
- [ ] `tooling_sha256` is removed; supersession is recorded in package docs only (LEDGERS,
      NEXT_SESSIONS definition of done, branch ledger).
- [ ] Final recorded gates after all changes (M10): exact quick/full, public quick/full
      `--record`, examples with tests.

**Budget risk:** `full` has an internal 1200 s limit. If package-role repeatability plus the guard
push it over, report the measured time and ask; don't raise the budget unilaterally.

## Out of scope

R07, R08, R09-R12 (items 1-3); items 4-6; E11, E-case tiers, Lee TDB, Linux checks (items 8-11);
research issues R01-R04 (item 12); moving the oracle again; editing research main; pushing; every
entry in [FUTURE_PORTS.md](FUTURE_PORTS.md) (deferred until after R07/R08).
