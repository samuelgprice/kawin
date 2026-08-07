# Thermo-Calc TC-Python example

This folder contains an example-local adapter for sampling Fe-Cr-Ni thermodynamic and kinetic data from Thermo-Calc through TC-Python, then using that data with kawin's moving-boundary surrogate workflow.

The integration is intentionally not part of the production `kawin` package API. TC-Python requires a local Thermo-Calc installation, licensed databases, and an environment where `tc_python` can be imported.

## Default workflow

The default configuration targets:

- Elements: `FE, CR, NI`
- Independent composition input: `[x_CR, x_NI]`
- Phases: `BCC_A2` matrix and `FCC_A1` product
- Databases: `TCFE9` and `MOBFE4`
- Temperature: `1373 K`

Generated datasets, plots, surrogates, and Thermo-Calc cache files should be written under `examples/ThermoCalc/outputs/`, which is ignored by git.

The notebook startup cell adds the repository root to `sys.path`, so it can be launched either from the repo root or directly from `examples/ThermoCalc`.

## Files

- `tc_python_adapter.py`: kawin-style facade backed by TC-Python.
- `training_data.py`: grid creation, sampling, checkpoint/resume, and NPZ/JSON export.
- `FeCrNi_TC_Python.ipynb`: notebook workflow for smoke checking, sampling, plotting, building, checking, and saving a moving-boundary surrogate.

## Data conventions

- Public composition inputs use independent mole fractions `[x_CR, x_NI]`.
- Full compositions and tracer diffusivities use `[FE, CR, NI]`.
- Chemical interdiffusivity matrices use rows/columns `[CR, NI]` with Fe as the reference element.
- Temperatures are K, driving force is J/mol, and diffusivities are m^2/s.
- TC-Python's dimensionless `DGM` is exported as `DGM * R * T`.

## User TDB preflight

The adapter also accepts a TC-compatible user TDB through `ThermoCalcConfig(user_database_path=..., kinetic_database=None)`. Run `TCPythonThermodynamics(config).preflight(...)` before sampling.

The checked-in `examples/FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb` is documented as pycalphad-oriented. In local testing it failed TC-Python preflight with:

```text
QPFIND : NO SUCH INTENSIVE VARIABLE
```

That database should be repaired or converted as a separate task; this example only detects and reports the incompatibility clearly.

## Live tests

Ordinary tests use a fake backend and do not need a Thermo-Calc license. Live TC-Python tests should remain opt-in because they require proprietary databases and can be slow:

```powershell
cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && set KAWIN_TC_PYTHON_LIVE=1 && python -m pytest kawin\tests\test_thermocalc_examples.py -k live'
```
