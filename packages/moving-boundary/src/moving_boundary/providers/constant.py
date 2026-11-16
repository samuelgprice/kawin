"""Constant scalar diffusivities for binary moving-boundary problems."""

import numpy as np


class ConstantDiffusivity:
    """Supply a positive scalar per phase in any consistent solver units.

    Composition, temperature and query context do not change the diffusivity.
    Interface and bulk values are identical; this is not an equilibrium closure.
    """

    def __init__(self, diffusivities):
        self.diffusivities = {phase: float(value) for phase, value in diffusivities.items()}
        if not self.diffusivities or any(
            not np.isfinite(value) or value <= 0.0 for value in self.diffusivities.values()
        ):
            raise ValueError("Supply finite positive scalar diffusivities for at least one phase.")

    def clearCache(self):
        """No calculation cache is retained."""

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        """Return the phase value, squeezed to the shape of the temperature input.

        Scalar and one-entry temperatures return a scalar; longer temperature
        arrays preserve their non-singleton dimensions. Composition is ignored.
        """
        if phase not in self.diffusivities:
            raise ValueError(f"Unknown phase {phase!r}; expected one of {list(self.diffusivities)}.")
        return np.squeeze(np.ones(np.atleast_1d(T).shape) * self.diffusivities[phase])
