from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class MovingBoundaryEquilibriumClosure(Protocol):
    """
    Protocol for ternary moving-boundary interface equilibrium closures.

    A closure maps a scalar tie-line coordinate ``eta`` to the two independent
    compositions on the phase-A and phase-B sides of the sharp interface. The
    dependent component is not returned because the diffusion models already
    enforce the substitutional sum constraint.
    """

    eta_bounds: tuple[float, float]

    def interface_compositions(self, eta: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(c_left, c_right)`` independent composition vectors."""
        ...


def _validate_interface_vector(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (2,):
        raise ValueError(f"{name} must contain exactly two independent ternary compositions.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


@dataclass(frozen=True)
class FixedTernaryInterfaceEquilibrium:
    """
    Fixed interface compositions for a ternary moving-boundary calculation.

    This closure is useful for benchmark problems with prescribed phase-end
    compositions. Because the tie-line coordinate is fixed, a valid moving
    interface step must satisfy both component balances at the same interface
    position.
    """

    left: np.ndarray
    right: np.ndarray
    eta: float = 0.0

    @property
    def eta_bounds(self) -> tuple[float, float]:
        return (float(self.eta), float(self.eta))

    def interface_compositions(self, eta: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        if eta is not None and not np.isclose(float(eta), float(self.eta), rtol=0.0, atol=1e-12):
            raise ValueError("Fixed interface equilibrium was queried away from its fixed eta.")
        return (
            _validate_interface_vector(self.left, "left").copy(),
            _validate_interface_vector(self.right, "right").copy(),
        )


@dataclass(frozen=True)
class CallableTernaryInterfaceEquilibrium:
    """
    Callable adapter for ternary interface tie-line closures.

    The callable must accept ``eta`` and return ``(c_left, c_right)`` where each
    composition vector contains the two independent substitutional components.
    ``eta_bounds`` gives the bounded interval used by the nonlinear interface
    solve.
    """

    func: Callable[[float], tuple[np.ndarray, np.ndarray]]
    eta_bounds: tuple[float, float] = (0.0, 1.0)

    def interface_compositions(self, eta: float) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = self.eta_bounds
        eta = float(np.clip(float(eta), float(lower), float(upper)))
        left, right = self.func(eta)
        return (
            _validate_interface_vector(left, "left").copy(),
            _validate_interface_vector(right, "right").copy(),
        )


class ThermodynamicTernaryInterfaceEquilibrium:
    """
    Adapter for thermodynamics objects that expose interfacial compositions.

    The wrapped object must implement ``getInterfacialComposition(eta, phases=...)``
    or ``getInterfacialComposition(eta)`` and return a pair of independent
    composition vectors. This class keeps the ternary Illingworth model decoupled
    from any one thermodynamics API shape.
    """

    def __init__(self, thermodynamics, phases=None, eta_bounds=(0.0, 1.0)):
        self.thermodynamics = thermodynamics
        self.phases = phases
        self.eta_bounds = tuple(float(v) for v in eta_bounds)

    def interface_compositions(self, eta: float) -> tuple[np.ndarray, np.ndarray]:
        eta = float(np.clip(float(eta), self.eta_bounds[0], self.eta_bounds[1]))
        getter = getattr(self.thermodynamics, "getInterfacialComposition", None)
        if getter is None:
            raise TypeError("Thermodynamics object must implement getInterfacialComposition for this adapter.")
        try:
            result = getter(eta, phases=self.phases)
        except TypeError:
            result = getter(eta)
        if len(result) != 2:
            raise ValueError("getInterfacialComposition must return (left, right).")
        left, right = result
        return (
            _validate_interface_vector(left, "left").copy(),
            _validate_interface_vector(right, "right").copy(),
        )
