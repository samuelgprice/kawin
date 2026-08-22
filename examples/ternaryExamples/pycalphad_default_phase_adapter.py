"""Pycalphad facade for default-phase moving-boundary surrogate sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pycalphad import Database
from pycalphad.core.utils import filter_phases

from kawin.thermo import MulticomponentThermodynamics


def resolve_pycalphad_equilibrium_phases(database, elements, equilibrium_phases=None):
    """Return pycalphad phases to consider for equilibrium calculations.

    If ``equilibrium_phases`` is supplied, it is used directly. Otherwise the
    phase list is resolved from the database using pycalphad's component filter.
    """
    if equilibrium_phases is not None:
        return tuple(str(phase) for phase in equilibrium_phases)
    dbf = database if isinstance(database, Database) else Database(str(database))
    comps = [str(element) for element in elements]
    if "VA" not in {comp.upper() for comp in comps}:
        comps.append("VA")
    return tuple(str(phase) for phase in filter_phases(dbf, comps))


def order_pycalphad_phase_universe(interface_phases, equilibrium_phases):
    """Return a phase-order-independent equilibrium universe for interface sampling.

    The requested interface phases must be present, but they do not define the
    order used for full-phase equilibrium. Preserving the resolved database (or
    user-supplied) order keeps adjacent interface facades from asking pycalphad
    slightly different equilibrium problems.
    """
    interface = tuple(str(phase) for phase in interface_phases)
    if len(interface) != 2:
        raise ValueError("interface_phases must contain exactly two phases.")
    universe = tuple(dict.fromkeys(str(phase) for phase in equilibrium_phases))
    missing = [phase for phase in interface if phase not in universe]
    if missing:
        raise ValueError(
            "interface phases must be present in the pycalphad equilibrium phase universe; "
            f"missing {tuple(missing)} from {universe}."
        )
    return universe


def create_pycalphad_thermodynamics_source(
    database,
    elements,
    interface_phases,
    *,
    use_default_phases=True,
    equilibrium_phases=None,
    **thermodynamics_kwargs,
):
    """Create either restricted or default-phase pycalphad thermodynamics."""
    if use_default_phases:
        return PycalphadDefaultPhaseThermodynamics(
            database,
            elements,
            interface_phases,
            equilibrium_phases=equilibrium_phases,
            **thermodynamics_kwargs,
        )
    database_source = str(database) if isinstance(database, Path) else database
    return MulticomponentThermodynamics(
        database_source,
        list(elements),
        list(interface_phases),
        **thermodynamics_kwargs,
    )


class PycalphadDefaultPhaseThermodynamics:
    """Kawin-style thermodynamics facade with full-phase pycalphad equilibrium.

    The facade lets equilibrium and tie-line validity checks see all compatible
    pycalphad phases, while diffusivity queries remain phase-specific. A probe
    is treated as a valid interface point only when the active composition sets
    are exactly the requested matrix/precipitate pair.
    """

    def __init__(
        self,
        database,
        elements,
        interface_phases,
        *,
        equilibrium_phases=None,
        phase_amount_tolerance=1.0e-12,
        **thermodynamics_kwargs,
    ):
        self.database = str(database) if isinstance(database, (str, Path)) else database
        self.interface_phases = tuple(str(phase) for phase in interface_phases)
        resolved_phases = resolve_pycalphad_equilibrium_phases(database, elements, equilibrium_phases)
        self.equilibrium_phases = order_pycalphad_phase_universe(self.interface_phases, resolved_phases)
        self.thermodynamics = MulticomponentThermodynamics(
            self.database,
            list(elements),
            list(self.equilibrium_phases),
            **thermodynamics_kwargs,
        )
        self.thermodynamics.gOffset = 0.0
        self.elements = list(self.thermodynamics.elements)
        self.phases = list(self.equilibrium_phases)
        self.phase_amount_tolerance = float(phase_amount_tolerance)
        if not np.isfinite(self.phase_amount_tolerance) or self.phase_amount_tolerance < 0.0:
            raise ValueError("phase_amount_tolerance must be finite and nonnegative.")
        print(f"Using default-phase pycalphad facade with the following phases: {self.equilibrium_phases}")

    def __enter__(self):
        """Return this facade for symmetry with TC-Python-backed examples."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """No-op context-manager exit."""
        return False

    def clearCache(self):
        """Clear the wrapped pycalphad thermodynamics cache."""
        self.thermodynamics.clearCache()

    def getEquilibriumData(self, x, T, removeCache=True):
        """Return active phases, amounts, compositions, and chemical potentials."""
        wks = self.thermodynamics.getEq(x, T, 0, list(self.equilibrium_phases))
        composition_sets = self._active_composition_sets(wks.get_composition_sets())
        phase_amounts = {}
        phase_compositions = {}
        for composition_set in composition_sets:
            phase = composition_set.phase_record.phase_name
            amount = float(getattr(composition_set, "NP", 1.0))
            phase_amounts[phase] = phase_amounts.get(phase, 0.0) + amount
            phase_compositions.setdefault(phase, self._full_composition_from_composition_set(composition_set))
        return {
            "stable_phases": [composition_set.phase_record.phase_name for composition_set in composition_sets],
            "phase_amounts": phase_amounts,
            "phase_compositions": phase_compositions,
            "chemical_potentials": np.squeeze(wks.eq.MU),
        }

    def getInterfacialComposition(self, x, T, gExtra=0, precPhase=None, returnMeta=False):
        """Return interface endpoints only for the requested two-phase field.

        Nonzero ``gExtra`` is intentionally unsupported because applying one
        Gibbs-energy offset while all default phases are active is ambiguous in
        kawin's current pycalphad wrapper.
        """
        if np.any(np.asarray(gExtra, dtype=np.float64) != 0.0):
            raise NotImplementedError("Default-phase pycalphad facade supports only planar gExtra=0 endpoints.")
        precipitate_phase = self.interface_phases[1] if precPhase is None else str(precPhase)
        expected = (self.interface_phases[0], precipitate_phase)
        wks = self.thermodynamics.getEq(x, T, 0, list(self.equilibrium_phases))
        composition_sets = self._active_composition_sets(wks.get_composition_sets())
        metadata = self._metadata_from_composition_sets(composition_sets, expected)
        invalid = -1.0 * np.ones(len(self._nonvacant_elements()) - 1, dtype=np.float64)
        if tuple(metadata.get("endpoint_phases", ())) == expected and len(metadata.get("endpoints", ())) == 2:
            endpoints_by_phase = {endpoint["phase"]: endpoint["composition"] for endpoint in metadata["endpoints"]}
            x_alpha = endpoints_by_phase[expected[0]]
            x_beta = endpoints_by_phase[expected[1]]
        else:
            # if expected==('LIQUID', 'BCC_A2'):
            from examples.debugInPlace import debugInPlace
            print(x)
            # debugInPlace()
            x_alpha = invalid.copy()
            x_beta = invalid.copy()
        if returnMeta:
            return np.squeeze(x_alpha), np.squeeze(x_beta), metadata
        return np.squeeze(x_alpha), np.squeeze(x_beta)

    def getInterdiffusivity(self, x, T, phase=None, **kwargs):
        """Delegate phase-specific metastable diffusivity to kawin pycalphad."""
        return self.thermodynamics.getInterdiffusivity(x, T, phase=phase, **kwargs)

    def _active_composition_sets(self, composition_sets):
        return [
            composition_set
            for composition_set in composition_sets
            if float(getattr(composition_set, "NP", 1.0)) > self.phase_amount_tolerance
        ]

    def _metadata_from_composition_sets(self, composition_sets, expected_phases):
        endpoints = tuple(
            {
                "phase": composition_set.phase_record.phase_name,
                "composition": self._independent_composition_from_composition_set(composition_set),
            }
            for composition_set in composition_sets
        )
        observed = tuple(endpoint["phase"] for endpoint in endpoints)
        if len(endpoints) == 2 and set(observed) == set(expected_phases) and len(set(observed)) == 2:
            endpoints_by_phase = {endpoint["phase"]: endpoint["composition"] for endpoint in endpoints}
            endpoints = tuple(
                {"phase": phase, "composition": endpoints_by_phase[phase]}
                for phase in expected_phases
            )
            observed = tuple(expected_phases)
        return {
            "endpoint_phases": observed,
            "endpoints": endpoints,
        }

    def _full_composition_from_composition_set(self, composition_set):
        values = np.asarray(composition_set.X, dtype=np.float64).reshape(-1)
        nonvacant = self._nonvacant_elements()
        sort_indices = np.argsort(nonvacant)
        unsort_indices = np.argsort(sort_indices)
        if values.size != len(nonvacant):
            raise ValueError(
                f"Composition set for {composition_set.phase_record.phase_name} has "
                f"{values.size} components, expected {len(nonvacant)}."
            )
        return values[unsort_indices]

    def _independent_composition_from_composition_set(self, composition_set):
        return self._full_composition_from_composition_set(composition_set)[1:]

    def _nonvacant_elements(self):
        return [element for element in self.elements if str(element).upper() != "VA"]
