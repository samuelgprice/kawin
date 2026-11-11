# Authority switch: what was built (2026-11-11)

Status: **implemented.** This replaces the 2026-11-10 draft proposal. The approved plan, with
per-batch results, is [AUTHORITY_SWITCH_PLAN.md](AUTHORITY_SWITCH_PLAN.md); the ownership
record and migration log are in [LEDGERS.md "Authority switch"](LEDGERS.md#authority-switch-2026-11-11).
This note compares the draft with what was built, for readers who remember the draft.

## Decisions taken (user, 2026-11-10)

- Future Illingworth solver changes (R07, R08, ...) happen only in this package. Research main
  stays frozen at `7af24f4`; its solver modules keep that behaviour and are not made into
  wrappers.
- `mb-ref-7af24f4` is the last research oracle. It is optional: only the explicit historical
  actions use it.
- Approved reference artifacts stay local. Their hashes, provenance and reports are committed.
- One approval per change by default; `approve --cases` accepts any list.
- Supersession (item 14) is recorded in package docs only.

## Draft versus built

| Draft proposal | Built |
|---|---|
| The research worktree stays as the read-only *input source* for package-role case builders | The package owns its inputs: `tools/definitions/` (materialized once, hash-checked against `docs/definitions-provenance.json`), adopted local inputs with recorded hashes, regenerated independently by package paths (M5b). Package workers never read the research tree |
| `approved_references` per case with `package_commit`, hashes and `previous` | Per approval, with immutable versioned folders `reference-approved/<approval id>/`, an `in_effect` map, a committed hash-checked report in `docs/approvals/`, and rollback as another approval-history entry. Crash-safe promotion (partial folder, validate, rename, report, atomic manifest replace) |
| `approve` runs each listed case twice and records deviations | `approve` gates on correctness first: the whole `tests/` suite, a repeat pair for every case, the registered `approval_criteria` (analytic limits against exact solutions; conservation of conserved quantities), contract assertion groups, and a public-tolerance pass of every unlisted case. Only then does it report deviations and promote |
| Move the previous expected files to a backup and write new ones | `reference/` is never rewritten; selection happens in `expected_reference()`, used by every comparison including example acceptance |
| LEGACY needs a package-owned baseline copy | `reference/LEGACY.npz` is the hashed baseline after a one-time bitwise proof against the research file |
| Contracts: resolve mole-array scaling first (item 7) | Done in M2: per-model molar-volume scales, as run cases use |
| Port the missing research solver tests | 301 of 314 ported with a provenance-recording tool; 13 deferred (FUTURE_PORTS.md) |
| Normal acceptance still validates the research checkout | Normal `quick`, `full` and example acceptance use no research file and no Git; proved from `git archive` exports with research reads and Git blocked, with the approved set both empty and active (M9) |

## Procedure for a deliberate numerical change

1. Make the change in the package with tests; commit it.
2. From a clean package tree: `acceptance.py approve --cases ... --reason "LEDGERS Rxx"`. It
   builds a fresh wheel from HEAD and refuses unless every gate above passes.
3. Commit `tools/acceptance_manifest.json` and force-add the report it names under
   `docs/approvals/`, separately from the solver change. Log the change in LEDGERS.
4. Rerun exact quick and full; they must be bit-identical against the mixed reference set.

## Still open

- Whether a research-only workflow (for example DICTRA comparisons) needs to move to the
  package; none is needed by acceptance.
- The deferred ports in FUTURE_PORTS.md, with the quasi-binary brass test as a priority item.
