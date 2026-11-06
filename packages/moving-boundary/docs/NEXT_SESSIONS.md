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
Quick: 29 run comparisons and 41 pytest tests passed in 435.73 s. Full: 38 run comparisons and 41 pytest tests passed in 495.45 s.
See PUBLIC_KAWIN_MILESTONE.md, acceptance.json and README for scope and command.

## Installed solving and research-owned generation

The installed package with stock kawin alone supports solving with user-supplied
analytic providers, saved surrogates, or cached tables supplied through the solver's
provider interface. Those solves do not require the research checkout once their
provider inputs are assembled using installed APIs.

Generating data with pycalphad now works from the installed package through
`moving_boundary.providers.pycalphad.create_pycalphad_thermodynamics_source`.
The engine is imported lazily; stock kawin 0.5.0 already requires pycalphad>=0.11,
so no separate extra is needed. Stage 2B is complete. O05 performs real sampling
through this installed adapter under an examples-import/source-read guard. Its
reference worker still uses the original adapter from clean detached 06dc7b4.
Only O05 keeps PYTHONHASHSEED=0; R01 remains open on the research branch.

Anything importing or reading examples/ still needs the research checkout. This
includes the research builders/assembly scripts used after sampling by acceptance
and the unpromoted live TC adapter. Those orchestration scripts are not installed
APIs; the installed pycalphad generation step does not import or read them.

surrogates.py, surrogate_artifacts.py, _diffusivity_domain.py and equilibrium.py
were copied whole and unchanged in implementation, apart from required import
relocations and documented whitespace-only edits. Their presence in the package
does not complete the broader Stage 2 provider/surrogate separation. Stage 2B
is complete; R01 remains a separate research-branch scientific issue.

## Completed implementation order

Approved list/tier changes were recorded in ACCEPTANCE_CANDIDATES.md, followed by
item 5 batch guards, item 6 documentation/test-stub cleanup, frozen inputs and
quantity tolerances, detached 06dc7b4 reference capture, then the three separate
item 2 relocations and item 3 public-host compatibility. No approval is outstanding
for that completed work. All TC-derived fixtures remain ignored; O06 stays disabled.

## Research reproducibility issue R01 (open)

O05 data generation is sensitive to Python hash order in third-party pycalphad
0.11.1: model.py:893/895/903 builds the ideal-mixing expression from an unordered
Species set (variables.py:215), then sympydiff_utils.py:137 compiles a hash-order-
sensitive Hessian with SymEngine CSE. A diagnostic ordered-iteration control alone
eliminated all O05 hash-0/hash-1 differences. NumPy seeding was unnecessary.
This affects rebuilt pycalphad-backed surrogate data and subsequent histories;
the 42 other cases repeat bitwise unseeded. Default-phase generation shares the
dependency but has not been diagnosed separately. See LEDGERS.md R01 for controls
and the main-06dc7b4 call sites. Only O05 retains PYTHONHASHSEED=0; no NumPy seed
is imposed. Every unaffected quick case/contract is checked twice unseeded.

Propose a research-branch provider regression across hash seeds, followed by
validated deterministic symbolic ordering or a dependency/compiler fix. Do not
implement that scientific fix independently in the package, on main, or by
refreshing references. Numerical code, references and tolerances remain unchanged.

## Stage 2B: pycalphad generation adapter (complete)

The source-mapped provider facade and phase-universe/g_offset/endpoint-ordering
helpers now live in providers/pycalphad.py. Their arithmetic, iteration order,
component conventions and R01 hash sensitivity are unchanged. The only closure
outside the source adapter was an unused conditional import of examples.debugInPlace;
its call was commented out. That import is omitted; no debugger helper or larger
examples tree was copied. See PUBLIC_KAWIN_MILESTONE.md for the full source map.

O05 samples real pycalphad data through the installed adapter while rejecting
examples/debugger imports and direct examples source reads. Reference generation
keeps the original detached-06dc7b4 adapter. Sampling inputs, references and
preregistered tolerances remain unchanged; no cached/fake provider replaces it.
Research builders/assembly remain available only from the checkout after the
guarded generation step. Databases, trained artifacts and TC outputs stay local.

The broader Stage 2 separation of generation, surrogate evaluation, persistence/
domain validation and equilibrium interfaces is still open. The four whole copied
modules above have not been refactored into that split. R01 is still open on the
research branch; Stage 2B makes no species/term sorting or CSE change.

## Next scoped work

- Stage 2C: remove debugInPlace from the TC adapter before promoting it; establish
  live licensed availability and its own acceptance scope.
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
