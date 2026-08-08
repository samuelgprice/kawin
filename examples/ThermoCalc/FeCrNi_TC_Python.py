"""Fe-Cr-Ni TC-Python sampling workflow for Jupyter Interactive.

Open this file in VS Code or another editor that supports ``# %%`` cells and
run the cells sequentially.  The workflow requires a Thermo-Calc installation,
TC-Python, and access to TCFE9/MOBFE4.
"""

# %% [markdown]
# # Fe-Cr-Ni TC-Python sampling for kawin
#
# This example samples a small Fe-Cr-Ni grid from Thermo-Calc through TC-Python
# and builds kawin's `TernaryMovingBoundaryThermodynamicsSurrogate` from
# `kawin.diffusion.MovingBoundarySurrogates`.

# %%
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "kawin").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from examples.ThermoCalc.tc_python_adapter import TCPythonThermodynamics, ThermoCalcConfig
from examples.ThermoCalc.training_data import (
    build_moving_boundary_surrogate,
    make_fecrni_demo_grid,
    sample_training_data,
)

OUTPUTS = REPO_ROOT / "examples" / "ThermoCalc" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Configuration and smoke check

# %%
config = ThermoCalcConfig(
    thermodynamic_database="TCFE9",
    kinetic_database="MOBFE4",
    elements=("FE", "CR", "NI"),
    phases=("BCC_A2", "FCC_A1"),
    reference_element="FE",
    cache_dir=OUTPUTS / "tc_cache",
)

therm = TCPythonThermodynamics(config)
smoke = therm.preflight(x=(0.38, 0.001), T=1373.0)
smoke

# %% [markdown]
# ## Sample a fast demo grid

# %%
T = 1373.0
x_grid = make_fecrni_demo_grid()
dataset = sample_training_data(
    therm,
    x_grid,
    T,
    output_prefix=OUTPUTS / "fecrni_1373_fast",
    checkpoint_every=10,
    resume=True,
)
dataset["manifest"]["complete"], len(dataset["manifest"]["failures"])

# %% [markdown]
# ## Quick plots

# %%
arrays = dataset["arrays"]
valid = arrays["driving_force_valid"]

fig, ax = plt.subplots(figsize=(5, 4))
sc = ax.scatter(
    arrays["compositions"][valid, 0],
    arrays["compositions"][valid, 1],
    c=arrays["driving_force"][valid],
)
ax.set_xlabel("x_CR")
ax.set_ylabel("x_NI")
ax.set_title("FCC_A1 driving force at 1373 K")
fig.colorbar(sc, ax=ax, label="J/mol")
fig.tight_layout()
fig.savefig(OUTPUTS / "driving_force_fast.png", dpi=200)
plt.show()

# %% [markdown]
# ## Build and save a moving-boundary surrogate
#
# The probe path below stays in a BCC/FCC two-phase region at 1373 K for the
# default TCFE9/MOBFE4 setup.  The surrogate stores phase-ordered tie-line
# endpoints and phase-specific interdiffusivity matrices.

# %%
eta_samples = np.linspace(0.0, 1.0, 5)
probe_start = np.array([0.25, 0.068])
probe_end = np.array([0.45, 0.242])

mb_surrogate = build_moving_boundary_surrogate(
    therm,
    temperature=T,
    probe_start=probe_start,
    probe_end=probe_end,
    eta_samples=eta_samples,
    diffusivity_bulk_points=x_grid,
    output_path=OUTPUTS / "fecrni_tc_python_moving_boundary_surrogate.npz",
)

left, right = mb_surrogate.interface_compositions(0.5)
bcc_D = mb_surrogate.getInterdiffusivity(left, T, phase="BCC_A2")
fcc_D = mb_surrogate.getInterdiffusivity(right, T, phase="FCC_A1")
print("BCC/FCC midpoint endpoints:", left, right)
print("BCC_A2 midpoint diffusivity:\n", bcc_D)
print("FCC_A1 midpoint diffusivity:\n", fcc_D)

# %% [markdown]
# ## Optional: local TDB preflight
#
# The checked-in Lee TDB is pycalphad-oriented and is expected to fail
# TC-Python preflight with `QPFIND : NO SUCH INTENSIVE VARIABLE`.  Keep this
# cell commented unless you are actively checking user-TDB compatibility.

# %%
# from examples.ThermoCalc.tc_python_adapter import ThermoCalcDatabaseError
#
# lee_config = ThermoCalcConfig(
#     thermodynamic_database="user",
#     kinetic_database=None,
#     user_database_path=REPO_ROOT / "examples" / "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb",
#     elements=("FE", "CR", "NI"),
#     phases=("BCC_A2", "FCC_A1"),
#     reference_element="FE",
#     cache_dir=OUTPUTS / "tc_cache",
# )
# try:
#     TCPythonThermodynamics(lee_config).preflight(x=(0.38, 0.001), T=1373.0)
# except ThermoCalcDatabaseError as exc:
#     print(exc)

