# DICTRA exb5: γ/α/γ Fe-Cr-Ni diffusion couple vs kawin

Driver: [`exb5_gamma_alpha_gamma.py`](exb5_gamma_alpha_gamma.py). Every stage
is cached in `outputs/exb5/`.

## Setup

**DICTRA example exb5.** This is Kajihara, Lim and Kikuchi (ISIJ Int. 33,
1993), couple A:
- a thin α (BCC) slice of Fe-38Cr-0Ni (wt%), l0 = 186.9 µm, clamped
  between two 500 µm γ (FCC) slices;
- the γ slices are Fe-27Cr-28Ni. The macro's header says 20 % Ni, but it
  enters 0.28, which is what DICTRA runs, so 28 % is used here;
- annealed at 1373 K for 1000 h, with FEDEMO + MFEDEMO data.

**How DICTRA runs it.** It simulates half of the symmetric couple: 93.45 µm
of α beside 500 µm of γ. Both regions use `AUTO` grids and the default
time-step control.

**kawin inputs.** These come from the same Thermo-Calc databases:
- 404 BCC/FCC tie-lines, parameterised by η = BCC x_Cr. BCC x_Ni is not
  monotonic along the α+γ field, peaking near x_Cr = 0.7. The tie-lines were
  checked for crossing.
- DC matrices tabulated on (x_Cr, x_Ni) grids and interpolated bilinearly.
  The tables are accurate to within 1–2 % of max|D|. A table 4× denser
  changes the results by 0.2 % or less.

**The two kawin models:**
- **two-phase:** the same half couple as DICTRA;
- **three-phase:** the full γ|α|γ couple.

## Symmetric couple (exb5 as written)

Normalised α thickness 2s/l0:

| | 1 h | 10 h | 100 h | 1000 h | Runtime |
|---|---|---|---|---|---|
| Experiment | 1.11 | 1.33 | 1.44 | 1.15 | |
| DICTRA exb5 (1000 h run) | 1.064 | 1.222 | 1.255 | 0.919 | 77 s |
| DICTRA stopped at 10 h / 100 h | | 1.207 | 1.260 | | 90 s each |
| kawin two-phase (half couple) | 1.066 | 1.209 | 1.264 | 0.925 | 8 s |
| kawin three-phase (full couple) | 1.066 | 1.209 | 1.264 | 0.925 | 30 s |

- **The two kawin models agree with each other** to about 2e-4. They use
  different α grids, so this is the level expected.
- **DICTRA's answer depends on the simulation horizon.** Its default time
  steps scale with the simulated time, so the 1000 h run's value at 10 h
  (1.222) is 1.2 % above that of a run stopped at 10 h (1.207).
- **Against DICTRA at the end of each run**, the α displacement (2s/l0 − 1)
  agrees with kawin to 1.0 % at 10 h and 1.5 % at 100 h.
- **Tight `TimestepControl`** did not finish in 50 min to 1000 h, or in
  30 min to 10 h.
- **Experiment.** Both codes fall below the experiment by the same amount,
  which points to the FEDEMO/MFEDEMO data rather than the solvers.

## Asymmetric variant (three-region DICTRA)

The right γ slice is changed to Fe-25Cr-20Ni. This is not a DICTRA example.
DICTRA runs it as three regions γ_L | α | γ_R, and kawin uses the
three-phase model. The two interfaces behave differently:
- the right one moves about twice as far;
- the left one moves out at first, then reverses late in the run.

Interface displacements (µm), with DICTRA values taken at the end of each
run:

| | Left, 10 h | Right, 10 h | Left, 100 h | Right, 100 h | Left, 1000 h | Right, 1000 h |
|---|---|---|---|---|---|---|
| DICTRA, `AUTO` grid | −20.73 | 30.28 | −26.42 | 52.27 | 12.48 | 42.24 |
| DICTRA, 100-point linear grids | −19.71 | 30.47 | | | | |
| DICTRA, 200-point linear grids | −19.68 | 30.67 | | | | |
| kawin three-phase | −19.64 | 30.27 | −25.61 | 52.09 | 12.95 | 42.46 |

- **DICTRA's grid.** With the `AUTO` grid DICTRA's left interface is 5 %
  away from kawin at 10 h. Refining DICTRA's grid brings it to within 0.2–0.4
  %, so the gap was DICTRA's coarse automatic grid at that interface.
- **Agreement.** Both interfaces agree within about 1 % once DICTRA is
  grid-refined. At 100 h and 1000 h only the `AUTO` runs are available; there
  the right interface agrees to 0.5 % and the left to 3–4 %, the same size as
  the `AUTO`-grid gap at 10 h.
- **kawin is converged.** A coarse/base/fine ladder (nodes ×2 and Δln t ÷2
  per level) changes kawin's displacements by at most 0.3 %.
- **Runtime:** kawin 24 s; DICTRA 2–4 min per run.

## Reproducing

From the repo root, through the conda wrapper in AGENTS.md (needs a
Thermo-Calc licence for the DICTRA and tc-inputs stages, about 15 min in
total):

```
python -u -m examples.ternaryExamples.exb5_gamma_alpha_gamma --stage all
```

The DICTRA grid-refinement runs call
`run_dictra(ASYM_REGIONS, tight=False, sim_time=HORIZONS[0], grid_points=n)`
directly. They are not cached stages.
