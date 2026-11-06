# Definition of done

The end state: my Illingworth solvers (binary, ternary two-phase, ternary three-phase,
with interstitial and molar-volume support) live in this package. The package depends
on UNMODIFIED public kawin (a chosen release, currently 0.5.0) only for what it
really needs: the model/solver loop, mesh and recording, TemperatureParameters, and
the pycalphad/thermo side. Fork-only host behavior lives in the package's own
compatibility layer. A single command shows that the runs I actually use still give
the same results as main.

## Current checkpoint

Illingworth solvers, shared/interstitial helpers and numerics moved under `solvers/illingworth/`, leaving room for sibling solver families.
Commit: `refactor: group Illingworth modules under solvers/illingworth`; permitted docstring changes: `solvers/__init__.py`, `solvers/illingworth/__init__.py` and the approved `solvers/illingworth/numerics/__init__.py`.
Fresh installed-wheel full `--record`: 40 runs and 5 contracts bit-identical, 88 tests, 34 repeat pairs, 578.02 s; solver origins `moving_boundary.solvers.illingworth.*`.

Whole-module relocation is complete: binary 8233ad2, ternary e08eed0, three-phase
693595d. Each separate installed-wheel quick gate passed 29 exact run comparisons
and 22 package tests. The method-by-method subclass strategy is replaced.

Stage 1B is complete against unmodified public kawin 0.5.0 in its separate local
environment, with bounded package policy adapters and no fork-only runtime imports.
Quick: 30 run comparisons and 88 pytest tests passed in 427.34 s. Full: 40 run comparisons and 88 pytest tests passed in 578.02 s.
See PUBLIC_KAWIN_MILESTONE.md, acceptance.json and README for scope and command.

The final quick gate used 427.34 s of its unchanged 600 s
budget, below the 540 s review threshold. It changed zero tracked files. Full
used `--record` and completed in 578.02 s of its unchanged
1200 s cap. All 33 quick/contract reference pairs remain; full adds O06's
unseeded pair for 34. No case was skipped on this configured machine. Only
the new ignored O06 fixture pair was captured; all 89 older files are unchanged.

## Installed solving and research-owned generation

The installed package with stock kawin alone supports solving with user-supplied
analytic providers, saved surrogates, or cached tables supplied through the solver's
provider interface. Those solves do not require the research checkout once their
provider inputs are assembled using installed APIs.

Generating data with pycalphad now works from the installed package through
`moving_boundary.providers.pycalphad.create_pycalphad_thermodynamics_source`.
The engine is imported lazily; stock kawin 0.5.0 already requires pycalphad>=0.11,
so no separate extra is needed. Stage 2B adapter relocation is complete for the
accepted O05 restricted-phase generation and O08 default-phase endpoint/equilibrium
sampling. Both use the installed adapter under the examples-import/source-read
guard; reference workers use the original adapter from clean detached 06dc7b4.
O08 has two valid BCC/FCC probes and one invalid FCC-only probe at 1373 K, using
the same FeCrNi database. It compares all arrays bitwise and phase labels in order.
Only O05/O08 keep PYTHONHASHSEED=0; R01 remains open on the research branch.

Anything importing or reading examples/ still needs the research checkout. This
includes research builders/assembly scripts and TC training_data.py. Those
orchestration scripts are not installed APIs; the installed pycalphad generation
step does not import or read them. The batch-safe TC adapter is now installed
in providers/thermocalc.py; its real O06 generation acceptance is described below.

surrogates.py, surrogate_artifacts.py, _diffusivity_domain.py and equilibrium.py
were copied whole and unchanged in implementation, apart from required import
relocations and documented whitespace-only edits. Their presence in the package
does not complete the broader Stage 2 provider/surrogate separation. Stage 2B
adapter relocation is complete within the O05/O08 scope; R01 remains a separate
research-branch scientific issue.

## Completed implementation order

Approved list/tier changes were recorded in ACCEPTANCE_CANDIDATES.md, followed by
item 5 batch guards, item 6 documentation/test-stub cleanup, frozen inputs and
quantity tolerances, detached 06dc7b4 reference capture, then the three separate
item 2 relocations and item 3 public-host compatibility. No approval is outstanding
for that completed work. All TC-derived fixtures remain ignored; O06 is enabled in full when available.

## Research reproducibility issue R01 (open)

O05 restricted and O08 default-phase generation are sensitive to Python hash order
in third-party pycalphad
0.11.1: model.py:893/895/903 builds the ideal-mixing expression from an unordered
Species set (variables.py:215), then sympydiff_utils.py:137 compiles a hash-order-
sensitive Hessian with SymEngine CSE. A diagnostic ordered-iteration control alone
eliminated all O05 hash-0/hash-1 differences. NumPy seeding was unnecessary.
This affects rebuilt pycalphad-backed surrogate data and subsequent histories;
the 42 other cases repeat bitwise unseeded. O08 hash 0/1 reference workers differ
in five of seven arrays (endpoints, phase amounts/compositions and chemical
potentials), while a fresh hash-0 repeat is bit-identical. See LEDGERS.md R01 for
controls and main-06dc7b4 call sites. Only O05/O08 retain PYTHONHASHSEED=0; no
NumPy seed is imposed. Every unaffected quick case/contract is checked twice unseeded.

Propose a research-branch provider regression across hash seeds, followed by
validated deterministic symbolic ordering or a dependency/compiler fix. Do not
implement that scientific fix independently in the package, on main, or by
refreshing references. Numerical code, references and tolerances remain unchanged.

## Stage 2B: adapter relocation complete; acceptance bounded to O05/O08

The source-mapped provider facade and phase-universe/g_offset/endpoint-ordering
helpers now live in providers/pycalphad.py. Their arithmetic, iteration order,
component conventions and R01 hash sensitivity are unchanged. The only closure
outside the source adapter was an unused conditional import of examples.debugInPlace;
its call was commented out. That import is omitted; no debugger helper or larger
examples tree was copied. See PUBLIC_KAWIN_MILESTONE.md for the full source map.

O05 samples restricted-phase surrogate data and O08 samples default-phase
endpoints/equilibrium through the installed adapter while rejecting examples/debugger
imports and direct examples source reads. The original reference adapter remains
at detached 06dc7b4. O08 exercises its invalid-endpoint branch, including the unused
research debugger import (LEDGERS.md R02); the package copy omits that import.
O08's newly captured fixture is frozen by JSON/NPZ hashes in the manifest; every
older reference, input hash and preregistered tolerance remains unchanged. O08 is
bitwise-only even in public mode. No cached/fake provider replaces real generation.
This acceptance does not cover arbitrary databases/phase universes, default-phase
surrogate training, or every delegated kinetics/diagnostics method.
Research builders/assembly remain available only from the checkout after the
guarded generation step. Databases, trained artifacts and TC outputs stay local.

The broader Stage 2 separation of generation, surrogate evaluation, persistence/
domain validation and equilibrium interfaces is still open. The four whole copied
modules above have not been refactored into that split. R01 is still open on the
research branch; Stage 2B makes no species/term sorting or CSE change.

See [LEDGERS.md R04](LEDGERS.md#research-phase-name-issue-r04-open-research-cleanup-o06-convention-resolved) for the passing suffixed-name probe and remaining research-branch work.

## Stage 2C complete within O06

providers/thermocalc.py is a source-mapped, batch-safe copy of the frozen adapter.
Installed generation uses it and installed surrogate APIs under the examples
import/source-read guard; the reference uses the original 06dc7b4 adapter and
training_data.py. Every availability/generation/repeat worker gets its own fresh,
empty ignored TC cache. TC-Python is lazy, separately installed and licensed;
it is not a pip dependency. TC inputs, caches, outputs and numerical metrics
remain ignored; committed evidence contains only sanitized counts/hashes.

O06 accepts only the approved suffixed FeCrNi two-phase input: three real tie
lines, two bulk points and the same six-step solve. Its two unseeded pilot runs
and hash-1 control were bit-identical. It runs unseeded in full only, with a fresh
unseeded reference repeat pair inside full's unchanged 1200 s cap. Every numerical
array requires identical shape, dtype and bytes. All structural arrays are also
compared; only metadata_json.thermocalc_config.cache_dir is replaced with null,
retaining every other JSON character. This exception is recorded in the manifest.
The new detached-06dc7b4 JSON/NPZ fixture refuses overwrite and is hash-checked
before comparison. All 89 older reference files and existing inputs/tolerances
are unchanged. A missing runtime/licence/local input produces an explicit skip;
a present input hash mismatch or unrelated generation failure fails the gate.

The research builders/assembly, training_data.py dataset export/resume and plotting
scripts remain outside the installed package. NiTiNb/other TC databases or phase
universes, arbitrary three-phase generation, default-phase invalid probes with TC,
broader diagnostics/retry settings and physical convergence are not accepted by
O06. The broader Stage 2 generation/evaluation/persistence/domain/equilibrium split
is still open. Scientific fixes remain research-branch work.

See [LEDGERS.md R03](LEDGERS.md#research-dependency-issue-r03-tc-debugger-importcall-open) for main's remaining TC debugger dependency.

## Next scoped work

- Stage 2C is complete within O06. Expand TC acceptance only with a separately
  approved scope and inputs; retain the frozen adapter behavior and exact gate.
- Broaden fallback/Numba, plotting/result workflows, restart/coupling and scientific
  convergence coverage before claiming those capabilities. Parity is not proof of
  mathematical convergence or support for untested combinations.
- Preserve one numerical authority. Scientific fixes originate on the designated
  research branch and are ported deliberately with source SHA, affected functions,
  rationale and expected changes. A reference refresh is a separate reviewed change.

## Open decisions reserved for the user

- Final package name and intended audience/distribution/maintenance ownership.
- Pushing branches, publication, MAP licensing outreach and upstream contribution.
- Changing either ternary solver's thin-phase prompt default.

## Baseline and research policy

The extraction's kawin/, examples/ and root packaging remain frozen at ae9fb29.
The approved numerical oracle is the clean detached 06dc7b4 worktree; local main
5f4ad57 additionally removes obsolete test stubs. Do not merge research HEAD into
the extraction or develop scientific repairs independently in both locations.
Preserve floating-point order, x/u and volume conventions, R=8.3145 where used,
and temperature-array hours. Never regenerate references to hide a difference.
The shared Conda environment is unchanged. No push, research-branch merge,
upstream contact, MAP-author contact or blanket redistribution permission is given.
