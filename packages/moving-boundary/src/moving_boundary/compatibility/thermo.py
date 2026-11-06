"""Endpoint metadata for local providers; the thermodynamic engine stays public.

Metadata/endpoint and tracer wrapper source: 06dc7b4 MultiTherm.py and
Thermodynamics.py. Interdiffusivity and equilibrium remain inherited.
"""
import numpy as np
from kawin.thermo import MulticomponentThermodynamics as _HostThermodynamics
from kawin.thermo.utils import _getPrecipitatePhase, _getMatrixPhase
from kawin.thermo.Mobility import mobility_from_composition_set, tracer_diffusivity_from_diff
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
