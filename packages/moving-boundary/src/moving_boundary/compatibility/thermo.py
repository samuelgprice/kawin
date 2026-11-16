"""Endpoint metadata for local providers; the thermodynamic engine stays public.

Metadata/endpoint and tracer wrapper source: 06dc7b4 MultiTherm.py and
Thermodynamics.py. Equilibrium and uncaptured interdiffusivity remain inherited.
Opt-in kinetics capture reuses the host's local equilibrium and matrix arithmetic;
record extraction follows the research capture at 7af24f4 (Thermodynamics.py).
No shared kawin code is patched and no extra equilibrium is performed.
"""
from contextlib import contextmanager

import numpy as np
from kawin.thermo import MulticomponentThermodynamics as _HostThermodynamics
from kawin.thermo.utils import _getPrecipitatePhase, _getMatrixPhase
from kawin.thermo.Mobility import (
    inverseMobility, inverseMobility_from_diffusivity,
    mobility_from_composition_set, tracer_diffusivity_from_diff,
)
from pycalphad import variables as v

GAS_CONSTANT = 8.3145


def tracer_diffusivity(composition_set, mobility_callables=None,
                       mobility_correction=None, parameters=None):
    """Evaluate D=MRT with the explicit research R; do not alter host constants."""
    if mobility_callables is None:
        raise ValueError('mobility_callables is required')
    temperature = composition_set.dof[composition_set.phase_record.state_variables.index(v.T)]
    mobility = mobility_from_composition_set(composition_set, mobility_callables,
        mobility_correction, {} if parameters is None else parameters)
    return GAS_CONSTANT * temperature * mobility


class MulticomponentThermodynamics(_HostThermodynamics):
    """Attach phase-labelled endpoints and explicit-R tracer queries to stock thermo.

    Local providers use equilibrium, interdiffusivity and tracer queries only;
    unrelated precipitation curvature APIs retain the host's conventions.
    """
    @contextmanager
    def captureKineticsDiagnostics(self, callback):
        """Capture one JSON-compatible record per successful scalar diffusivity query.

        Records describe the same local equilibrium and matrix returned to the caller,
        including cache-seeded solves. Optional metadata failures are recorded, not
        substituted with surrogate predictions. No extra equilibrium is performed.
        Nested scopes restore the outer callback and index even on interruption.
        Instances are not safe for concurrent capture/query calls.
        """
        if not callable(callback):
            raise TypeError("callback must be callable.")
        names = ("_kinetics_diagnostic_callback", "_kinetics_diagnostic_index")
        missing = object()
        previous = {name: vars(self).get(name, missing) for name in names}
        self._kinetics_diagnostic_callback = callback
        self._kinetics_diagnostic_index = 0
        try:
            yield self
        finally:
            for name, value in previous.items():
                if value is missing:
                    vars(self).pop(name, None)
                else:
                    setattr(self, name, value)

    def _interdiffusivitySingle(self, x, T, removeCache=True, phase=None):
        """Use stock matrix calculation normally; observe its existing state during capture.

        The capture branch repeats stock kawin 0.5.0's calculation in the same order,
        retaining the thermodynamic factors it otherwise discards. Diagnostic extraction
        follows research 7af24f4; it does not solve equilibrium again or alter the cache.
        """
        callback = getattr(self, "_kinetics_diagnostic_callback", None)
        if callback is None:
            return super()._interdiffusivitySingle(x, T, removeCache, phase)
        phase = _getMatrixPhase(self.phases, phase)
        comp_sets = self._diffusivity_cache.get(phase, None)
        cache_seeded = comp_sets is not None
        result, comp_sets = self.getLocalEq(x, T, 0, [phase], composition_sets=comp_sets)
        cs_matrix = comp_sets[0]
        if self.mobCallables[phase] is None:
            matrix, factors, _ = inverseMobility_from_diffusivity(
                result.chemical_potentials, cs_matrix, self.elements[0], self.diffCallables[phase],
                diffusivity_correction=self.mobility_correction, parameters=self._parameters)
        else:
            matrix, factors, _ = inverseMobility(
                result.chemical_potentials, cs_matrix, self.elements[0], self.mobCallables[phase],
                mobility_correction=self.mobility_correction,
                vacancy_poor_interstitial_sublattice=self.vacancyPoorInterstitialSublattice.get(phase, False),
                parameters=self._parameters)
        if self.numElements != 2:
            order = np.argsort(np.argsort(self.elements[1:-1]))
            matrix = matrix[order, :][:, order]
            factors = factors[order, :][:, order]
        self._diffusivity_cache[phase] = None if removeCache else comp_sets
        matrix = np.squeeze(matrix)
        details = self._pycalphad_kinetics_diagnostics(phase, cs_matrix, factors, cache_seeded)
        record = {"record_type": "kinetics", "backend": "pycalphad",
                  "query_index": self._kinetics_diagnostic_index, "temperature": float(T),
                  "input_composition": np.asarray(x, dtype=np.float64).reshape(-1).tolist(),
                  "interdiffusivity": np.asarray(matrix, dtype=np.float64).tolist(), **details}
        callback(record)
        self._kinetics_diagnostic_index += 1
        return matrix

    def _pycalphad_kinetics_diagnostics(
        self, phase, composition_set, thermodynamic_factors, cache_hit
    ):
        """Extract optional kinetics provenance from an existing composition set.

        PyCalphad orders non-vacant elements alphabetically. Returned vectors
        and matrices are reordered to ``self.elements`` so they align with the
        public interdiffusivity API and the surrogate diagnostics schema.
        ``cache_hit`` here denotes a cached result (always false); ``cache_seeded``
        records reuse of the starting composition set for the equilibrium solve.
        """
        errors = {}
        nonvacant_input = [str(element) for element in self.elements if str(element) != "VA"]
        phase_elements = [str(element) for element in composition_set.phase_record.nonvacant_elements]
        try:
            input_order = np.asarray(
                [phase_elements.index(element) for element in nonvacant_input], dtype=np.int64
            )
        except ValueError as exc:
            input_order = None
            errors["element_order"] = str(exc)

        def optional(label, calculation):
            try:
                values = np.asarray(calculation(), dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError("quantity contains non-finite values")
                return values
            except Exception as exc:
                errors[label] = f"{type(exc).__name__}: {exc}"
                return None

        phase_composition = optional(
            "phase_composition",
            lambda: np.asarray(composition_set.X, dtype=np.float64)[input_order],
        ) if input_order is not None else None
        if self.mobCallables[phase] is None:
            tracer_values = optional(
                "tracer_diffusivities",
                lambda: tracer_diffusivity_from_diff(
                    composition_set, self.diffCallables[phase],
                    diffusivity_correction=self.mobility_correction,
                    parameters=self._parameters,
                ),
            )
            mobility_values = optional(
                "mobilities",
                lambda: tracer_values / (
                    GAS_CONSTANT * float(composition_set.dof[
                        composition_set.phase_record.state_variables.index(v.T)
                    ])
                ),
            ) if tracer_values is not None else None
        else:
            tracer_values = optional(
                "tracer_diffusivities",
                lambda: tracer_diffusivity(
                    composition_set, self.mobCallables[phase],
                    mobility_correction=self.mobility_correction,
                    parameters=self._parameters,
                ),
            )
            mobility_values = optional(
                "mobilities",
                lambda: mobility_from_composition_set(
                    composition_set, self.mobCallables[phase],
                    mobility_correction=self.mobility_correction,
                    parameters=self._parameters,
                ),
            )

        def ordered(values):
            if values is None or input_order is None:
                return None
            return np.asarray(values, dtype=np.float64)[input_order].tolist()

        site_fractions = None
        try:
            offset = len(composition_set.phase_record.state_variables)
            grouped = {}
            for variable, value in zip(
                composition_set.phase_record.variables,
                composition_set.dof[offset:offset + len(composition_set.phase_record.variables)],
            ):
                if not np.isfinite(value):
                    raise ValueError("site fractions contain non-finite values")
                sublattice = int(variable.sublattice_index) + 1
                grouped.setdefault(sublattice, {})[str(variable.species.name)] = float(value)
            site_fractions = [
                {"sublattice": index, "constituents": grouped[index]}
                for index in sorted(grouped)
            ]
        except Exception as exc:
            errors["site_fractions"] = f"{type(exc).__name__}: {exc}"

        factors = optional(
            "thermodynamic_factors", lambda: thermodynamic_factors
        )
        stable_phase = str(getattr(composition_set.phase_record, "phase_name", phase))
        return {
            "cache_hit": False,
            "cache_seeded": bool(cache_hit),
            "requested_phase": str(phase),
            "tracer_diffusivities": ordered(tracer_values),
            "thermodynamic_factors": None if factors is None else factors.tolist(),
            "mobilities": ordered(mobility_values),
            "stable_composition_sets": [stable_phase],
            "phase_composition": None if phase_composition is None else phase_composition.tolist(),
            "site_fractions": site_fractions,
            "kinetics_strategy": "pycalphad_local_single_phase",
            "scout_seed_count": None,
            "unique_seed_count": None,
            "converged_candidate_count": 1,
            "selected_seed_index": None,
            "selected_source_composition_set": stable_phase,
            "selected_gibbs_energy": None,
            "selected_gibbs_energy_units": None,
            "multistart_candidates": None,
            "diagnostic_errors": errors,
        }

    def getInterfacialComposition(self, x, T, gExtra = 0, precPhase = None, returnMeta = False):
        '''
        Gets interfacial composition by calculating equilibrum with Gibbs-Thomson effect

        Parameters
        ----------
        T : float or array
            Temperature in K
        gExtra : float or array (optional)
            Extra contributions to the precipitate Gibbs free energy
            Gibbs Thomson contribution defined as Vm * (2*gamma/R + g_Elastic)
            Defaults to 0
        precPhase : str
            Precipitate phase to consider (default is first precipitate in list)

        Note: for multiple conditions, only gExtra has to be an array
            This will calculate compositions for multiple gExtra at the input Temperature

            If T is also an array, then T and gExtra must be the same length
            where each index will pertain to a single condition

        returnMeta : bool (optional)
            Defaults to False.
            If True, returns endpoint metadata including phase labels aligned with
            the returned interface compositions.

        Returns
        -------
        If returnMeta is False:
            (parent composition, precipitate composition)
            Both will be either float or array based off shape of gExtra
            Will return (None, None) if precipitate is unstable
        If returnMeta is True:
            (parent composition, precipitate composition, metadata)
            metadata contains:
                - endpoint_phases: tuple(str, str)
                - endpoints: tuple(dict, dict) with keys {"phase", "composition"}
        '''
        gExtra = np.atleast_1d(gExtra)
        T = np.atleast_1d(T)
        precPhase = _getPrecipitatePhase(self.phases, precPhase)
        if len(T) == 1:
            T = T*np.ones(gExtra.shape, dtype=np.float64)

        if returnMeta:
            outputs = [self._interfacialComposition(x, T[i], gExtra[i], precPhase, returnMeta=True) for i in range(len(gExtra))]
            caArray, cbArray, metaArray = zip(*outputs)
            ca = np.squeeze(caArray)
            cb = np.squeeze(cbArray)
            if len(metaArray) == 1:
                return ca, cb, metaArray[0]
            metadata = {
                "endpoint_phases": tuple(metaArray[0]["endpoint_phases"]),
                "endpoints": (
                    {"phase": metaArray[0]["endpoints"][0]["phase"], "composition": ca},
                    {"phase": metaArray[0]["endpoints"][1]["phase"], "composition": cb},
                ),
            }
            return ca, cb, metadata

        caArray, cbArray = zip(*[self._interfacialComposition(x, T[i], gExtra[i], precPhase) for i in range(len(gExtra))])
        return np.squeeze(caArray), np.squeeze(cbArray)

    def _interfacialComposition(self, x, T, gExtra, precPhase, returnMeta=False):
        '''
        Gets interfacial composition, will return None, None if composition is in single phase region

        Parameters
        ----------
        T : float
            Temperature in K
        gExtra : float (optional)
            Extra contributions to the precipitate Gibbs free energy
            Gibbs Thomson contribution defined as Vm * (2*gamma/R + g_Elastic)
            Defaults to 0
        precPhase : str
            Precipitate phase to consider (default is first precipitate in list)

        returnMeta : bool (optional)
            If True, also returns endpoint phase metadata.

        Returns
        -------
        If returnMeta is False:
            (parent composition, precipitate composition)
            Both will be either float or array based off shape of gExtra
            Will return (None, None) if precipitate is unstable
        If returnMeta is True:
            (parent composition, precipitate composition, metadata)
        '''
        wks = self.getEq(x, T, gExtra, precPhase)
        mu = np.squeeze(wks.eq.MU)

        #Check for convergence, return None if not converged
        if np.any(np.isnan(mu)):
            ca = -1*np.ones(len(self.elements[:-1]), dtype=np.float64)
            cb = -1*np.ones(len(self.elements[:-1]), dtype=np.float64)
            if returnMeta:
                return ca, cb, {"endpoint_phases": (None, None), "endpoints": ({"phase": None, "composition": ca}, {"phase": None, "composition": cb})}
            return ca, cb

        cs_list = wks.get_composition_sets()
        ph = [cs.phase_record.phase_name for cs in cs_list]

        #Check if matrix and precipitate phase are stable, and check if there's no miscibility gaps
        if len(ph) == 2 and self.phases[0] in ph and precPhase in ph:
            cs_matrix = cs_list[ph.index(self.phases[0])]
            cs_precip = cs_list[ph.index(precPhase)]

            xM = np.array(cs_matrix.X, dtype=np.float64)
            xP = np.array(cs_precip.X, dtype=np.float64)

            sortIndices = np.argsort(self.elements[:-1])
            unsortIndices = np.argsort(sortIndices)
            ca = xM[unsortIndices]
            cb = xP[unsortIndices]
            if returnMeta:
                return ca, cb, {
                    "endpoint_phases": (self.phases[0], precPhase),
                    "endpoints": (
                        {"phase": self.phases[0], "composition": ca},
                        {"phase": precPhase, "composition": cb},
                    ),
                }
            return ca, cb

        ca = -1*np.ones(len(self.elements[:-1]), dtype=np.float64)
        cb = -1*np.ones(len(self.elements[:-1]), dtype=np.float64)
        if returnMeta:
            return ca, cb, {"endpoint_phases": (None, None), "endpoints": ({"phase": None, "composition": ca}, {"phase": None, "composition": cb})}
        return ca, cb

    def _tracerDiffusivitySingle(self, x, T, removeCache = True, phase = None):
        '''
        Gets tracer diffusivity at unique composition and temperature

        Parameters
        ----------
        x : float or array
            Composition
        T : float
            Temperature
        el : str
            Element to calculate diffusivity

        Returns
        -------
        Tracer diffusivity as a float
        '''
        # Compute local equilibrium on phase
        #phase = self.phases[0] if phase is None else phase
        phase = _getMatrixPhase(self.phases, phase)
        comp_sets = self._diffusivity_cache.get(phase, None)
        result, comp_sets = self.getLocalEq(x, T, 0, [phase], composition_sets=comp_sets)
        cs_matrix = comp_sets[0]

        # Get interdiffusivity from mobility or diffusivity models whichever is available
        # If both mobility and diffusivity models exist, then favor the mobility model
        if self.mobCallables[phase] is None:
            #NOTE: This is not tested yet
            Dtrace = tracer_diffusivity_from_diff(cs_matrix, self.diffCallables[phase],
                                                  diffusivity_correction=self.mobility_correction,
                                                  parameters=self._parameters)
        else:
            Dtrace = tracer_diffusivity(cs_matrix, self.mobCallables[phase],
                                        mobility_correction=self.mobility_correction,
                                        parameters=self._parameters)

        # Sort Dnkj from alphabetical to the input order of the elements
        sortIndices = np.argsort(self.elements[:-1])
        unsortIndices = np.argsort(sortIndices)
        Dtrace = Dtrace[unsortIndices]

        self._diffusivity_cache[phase] = None if removeCache else comp_sets
        return Dtrace
