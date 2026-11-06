# Definition of done

The end state: my Illingworth solvers (binary, ternary two-phase, ternary three-phase,
with interstitial and molar-volume support) live in this package. The package depends
on UNMODIFIED public kawin (a chosen release, currently 0.5.0) only for what it
really needs: the model/solver loop, mesh and recording, TemperatureParameters, and
the pycalphad/thermo side. Fork-only host behavior lives in the package's own
compatibility layer. A single command shows that the runs I actually use still give
the same results as main.

## Current checkpoint

Whole-module relocation is complete: binary 8233ad2, ternary e08eed0, three-phase
693595d. Each separate installed-wheel quick gate passed 29 exact run comparisons
and 22 package tests. The method-by-method subclass strategy is replaced.

Stage 1B is complete against unmodified public kawin 0.5.0 in its separate local
environment, with bounded package policy adapters and no fork-only runtime imports.
Quick: 30 run comparisons and 72 pytest tests passed in 436.25 s. Full: 39 run comparisons and 72 pytest tests passed in 508.95 s.
See PUBLIC_KAWIN_MILESTONE.md, acceptance.json and README for scope and command.

The final quick gate including O08 used 436.25 s of its unchanged
600 s budget, 103.75 s below the 540 s review threshold. O08 was
explicitly approved for quick and full after its 3.83 s worker pilot. All 33
unaffected quick/contract reference pairs remain in both gates. The default
quick run changed zero tracked files. Full used `--record`; its compact export
has one record per case. Only O06 was skipped under the existing TC-adapter
batch-safety restriction. The new O08 fixture is local/git-ignored and frozen
by manifest hashes; no existing reference or input hash was changed.

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
in providers/thermocalc.py, but O06 generation remains blocked on R04.

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
for that completed work. All TC-derived fixtures remain ignored; O06 stays disabled.

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

## Research phase-name issue R04 (open)

In examples/ThermoCalc/tc_python_adapter.py, calculate_equilibrium:452 stopped
applying base_phase_name in cd87f88 (2026-08-22), preserving composition-set
names such as FCC_A1#1. With the O06 FeCrNi base-phase configuration, the exact
phase-key match at :1754 fails, every tie line is invalid, and
kawin/diffusion/MovingBoundarySurrogates.py:873 rejects the endpoints. The
examples/ThermoCalc tree is identical at frozen 06dc7b4 and main 5f4ad57, so
main's FeCrNi TC workflow is affected too.

The duplicate-base-phase check at :453 can no longer fire for separately numbered
composition sets, while the docstring at :441-446 still describes base-name
merging. The cd87f88 commit explicitly preserves composition-set numbers; that
choice may be deliberate for systems where several composition sets are
physically distinct phases, including the NiTiNb three-phase TC work. A fix
requires a deliberate name-mapping rule on the research branch, validated on
both FeCrNi and NiTiNb; unconditional suffix stripping is not an approved fix.
Do not repair this behavior in the package or on main, change accepted inputs,
or refresh references to hide it. O06 remains disabled pending R04.

## Next scoped work

- Stage 2C: batch-safe adapter relocated and unit-tested; O06 blocked on R04.
  b688ecf retains the frozen numerical behavior; 77 package tests passed. Keep
  O06 disabled until the research name-mapping rule and acceptance scope are reviewed.
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
