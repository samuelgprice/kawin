"""Deterministic open benchmark inputs; no research, tests or acceptance imports.

The binary and curved two-phase inputs follow the public substitutional fixtures
in tests/ported_helpers/illingworth_baseline_cases.py and the research ternary
microbenchmark. Three-phase uses its linear closures with equal explicit molar
volumes. These independent copies are versioned performance workloads, not
acceptance references. All boundaries are closed, planar and substitutional.
"""
import numpy as np

from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from moving_boundary.solvers import (
    MovingBoundaryIllingworthFD1DModel,
    MovingBoundaryIllingworthTernaryFD1DModel,
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
)


class Diffusivity:
    """Synthetic positive scalar/matrix diffusivity, with optional smooth D(c)."""

    def __init__(self, family, variable):
        self.family, self.variable = family, variable

    def clearCache(self):
        """No provider cache is used by these analytic inputs."""

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        """Accept scalar or vectorized queries in the solvers' mole-fraction basis."""
        c = np.asarray(composition, dtype=float)
        if self.family == "binary":
            base = {"ALPHA": 1e-3, "BETA": 1e-2}[phase]
            return np.squeeze(base * (1 + .5*c) if self.variable else
                              np.ones(np.atleast_1d(temperature).shape)*base)
        if self.family == "ternary":
            base = np.array([[1e-3, 2e-4], [1e-4, 8e-4]]) if phase == "ALPHA" else np.array([[7e-4, -1e-4], [2e-4, 1.1e-3]])
        else:
            base = {"A": 1., "B": .7, "C": 1.3}[phase]*np.array([[1., .1], [.05, .8]])
        scale = 1 + .1*np.sum(c, axis=-1) if self.variable else np.ones(c.shape[:-1])
        return np.asarray(scale)[..., None, None]*base


class CurvedEquilibrium:
    """Noncrossing curved two-phase tie lines from the public ternary fixture."""
    eta_bounds = (0., 1.)

    def interface_compositions(self, eta):
        """Return two independent substitutional mole fractions on each side."""
        bend = eta*(1-eta)
        center = np.array([.24+.25*eta, .15-.015*eta+.22*bend])
        tangent = np.array([.25, -.015+.22*(1-2*eta)])
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        tilt = np.deg2rad(55-30*eta)
        direction = np.sin(tilt)*tangent + np.cos(tilt)*normal
        return (center-.042*(1+.12*(2*eta-1))*direction,
                center+.055*(1-.10*(2*eta-1))*direction)


class LinearEquilibrium:
    """Parallel linear tie lines for one of the three-phase interfaces."""
    eta_bounds = (0., 1.)

    def __init__(self, left, right):
        self.left, self.right = np.array(left), np.array(right)

    def interface_compositions(self, eta):
        """Return the A|B or B|C endpoint pair at eta."""
        shift = np.array([.1*eta, 0.])
        return self.left+shift, self.right+shift


def build_model(case):
    """Construct a fresh uninitialized model using explicit versioned controls.

    Larger transformed grids exercise linear algebra rather than just fixed-mesh
    output. Equal explicit volumes make both physical and legacy three-phase
    conservation checks applicable. Terminal phase stopping always raises.
    """
    family, nodes = case["family"], case["phase_nodes"]
    binary = family == "binary"
    components = ["CR"] if binary else ["X", "Y"]
    mesh = CartesianFD1D(components, [0., 5. if binary else 1.], case["physical_nodes"])
    if binary:
        profile = StepProfile1D(1., .8, .4)
    elif family == "ternary":
        profile = StepProfile1D(.45, [.16, .06], [.3261538461538461, .193])
    else:
        def profile(z):
            x = np.asarray(z).reshape(-1)
            return np.where((x < .35)[:, None], [.21, .09],
                            np.where((x < .7)[:, None], [.31, .16], [.20, .21]))
    mesh.setResponseProfile(ProfileBuilder([(profile, components)]),
                            boundaryConditions=MixedBoundary1D(len(components)))
    common = dict(mesh=mesh, elements=["NI", "CR"] if binary else ["Z", "X", "Y"],
                  thermodynamics=Diffusivity(family, case["variable"]), temperature=1000.,
                  bulk_diffusivity_mode=case["mode"], time_step=case["dt"],
                  tolerance=1e-10, residual_tolerance=1e-10, max_iterations=100,
                  bulk_picard_rtol=1e-10, bulk_picard_atol=1e-12,
                  bulk_picard_max_iterations=25, bulk_picard_relaxation=1.,
                  record=case["record"], record_pq_data=True,
                  max_step_retries=8, retry_factor=.5)
    if binary:
        return MovingBoundaryIllingworthFD1DModel(
            **common, phases=["ALPHA", "BETA"], interfacePosition=1.,
            interface_compositions=(.6, 0.), phase_a_nodes=nodes[0], phase_b_nodes=nodes[1])
    common.update(terminal_thin_phase_policy="raise", initial_eta_root_rtol=1e-10,
                  initial_eta_root_xtol=None, initial_eta_componentwise_scaling=False)
    if family == "ternary":
        return MovingBoundaryIllingworthTernaryFD1DModel(
            **common, phases=["ALPHA", "BETA"], interfacePosition=.45,
            interface_equilibrium=CurvedEquilibrium(), initial_eta_bracket=(0., 1.),
            phase_a_nodes=nodes[0], phase_b_nodes=nodes[1], interface_newton_start="extrapolated")
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        **common, phases=["A", "B", "C"], interfacePositions=(.35, .7),
        interface_equilibria=(LinearEquilibrium([.20, .10], [.30, .15]),
                              LinearEquilibrium([.24, .15], [.18, .20])),
        initial_eta_guess=(.5, .5), phase_molar_volumes=(1., 1., 1.), phase_nodes=nodes)


def validate_model(model, case):
    """Reject incomplete solves, invalid simplex/geometry, drift and broken history.

    Reuses solver conservation diagnostics; the fixed absolute bound is suitable
    for these nondimensional closed synthetic domains, not a general error test.
    Checks and output summaries run outside every timed region.
    """
    duration = case["dt"]*case["steps"]
    if not np.isclose(model.currentTime, duration, rtol=1e-12, atol=1e-15):
        raise ValueError(f"Incomplete solve: {model.currentTime} != {duration}")
    if case["family"] == "three_phase":
        interfaces = np.asarray(model.getInterfacePositions())
        right = model.getRightBoundary()
        drift = model.checkMolarConservation(case["conservation_atol"])
    else:
        interfaces = np.atleast_1d(model.getInterfacePosition())
        right = 5. if case["family"] == "binary" else 1.
        drift = model.checkConservation(case["conservation_atol"])
    if not np.all(np.diff(np.r_[0., interfaces, right]) > 0):
        raise ValueError("Invalid interface geometry")
    if case["family"] == "binary":
        p, q = model.getTransformedState()
        s = float(interfaces[0])
        profiles = ((s*model._u_grid, p), (s+(right-s)*model._v_grid, q))
    else:
        profiles = model.getPhysicalPhaseProfiles()
    for coords, profile in profiles:
        c = np.asarray(profile)
        if (not np.all(np.isfinite(coords)) or not np.all(np.diff(np.ravel(coords)) >= 0)
                or not np.all(np.isfinite(c)) or np.min(c) < -1e-12
                or np.max(c) > 1+1e-12
                or (case["family"] != "binary" and np.any(np.sum(c, axis=-1) > 1+1e-12))):
            raise ValueError("Invalid phase profile")
    drift = float(np.max(np.abs(drift)))
    if not np.isfinite(drift) or drift > case["conservation_atol"]:
        raise ValueError(f"Inventory drift {drift} exceeds {case['conservation_atol']}")
    if case["record"]:
        history = model.interfaceData
        times = history._time[:history.N+1]
        if not np.all(np.diff(times) > 0) or not np.isclose(times[-1], duration):
            raise ValueError("Invalid recorded times")
    return dict(final_time=float(model.currentTime), inventory_drift=drift,
                interfaces=interfaces.tolist(), profile_sums=[float(np.sum(p)) for _, p in profiles])
