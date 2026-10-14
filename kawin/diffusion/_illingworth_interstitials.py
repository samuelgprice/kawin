"""u-fraction support shared by the Illingworth moving-boundary solvers.

Convention
----------
With interstitial elements (``kawin.thermo.Mobility.interstitials``) present, the
Illingworth solvers store independent compositions as u-fractions,
``u_k = x_k / sum_{j substitutional} x_j``. Diffusion is written in the
volume-fixed frame of DICTRA: only substitutional atoms carry volume, the molar
volume per mole of substitutional atoms is constant and equal in every phase,
and the conserved quantity is ``u`` (so ``du/dt = -dJ/dz`` with
``J_k = -sum_j D_kj du_j/dz``). ``kawin.thermo.Mobility.interdiffusivity`` and
Thermo-Calc's ``DC(phase,k,j,ref)`` both return ``D`` in this basis (the
Thermo-Calc frame was checked by ``examples/Interstitials/check_tc_interstitial_frame.py``).

Everything outside the solvers (thermodynamics objects, the Thermo-Calc
adapter, surrogate tables and tie-line closures) keeps working in mole
fractions ``x``. The wrappers here convert at that boundary:

* :class:`UFractionDiffusivityProvider` converts solver ``u`` to ``x`` before a
  ``getInterdiffusivity`` query (the returned matrix is already the u-basis
  matrix and is not modified).
* :class:`UFractionInterfaceEquilibrium` converts closure endpoints from ``x``
  to ``u`` unless the closure declares ``composition_basis = "u"``.

For substitutional-only systems ``u == x``; :func:`resolve_diffusivity_provider`
and :func:`resolve_interface_equilibrium` then return the original objects so
results stay bit-identical.
"""

import numpy as np

from kawin.thermo.Mobility import interstitials

_SUPPORTED_COMPOSITION_BASES = ("x", "u")


def independent_interstitial_mask(all_elements):
    """Returns a boolean mask over the independent elements (``all_elements[1:]``)."""
    return np.asarray([str(e).upper() in interstitials for e in list(all_elements)[1:]], dtype=bool)


def has_interstitials(all_elements):
    """Returns whether any element of the system is an interstitial."""
    return any(str(e).upper() in interstitials for e in all_elements)


def validate_composition_basis(basis, name="composition_basis"):
    """Normalizes a composition-basis label to ``'x'`` or ``'u'``."""
    value = str(basis).lower()
    if value not in _SUPPORTED_COMPOSITION_BASES:
        raise ValueError(f"{name} must be 'x' or 'u'; received {basis!r}.")
    return value


def validate_interstitial_element_order(all_elements):
    """
    Rejects element orderings that the u-fraction conversion cannot represent.

    The dependent element (``all_elements[0]``) must be substitutional because
    ``u_dependent = 1 - sum(substitutional u)``. Independent elements may be in
    any order: kawin's ``Thermodynamics.getInterdiffusivity`` and the
    Thermo-Calc adapter both return matrices in the caller's element order.
    """
    elements = [str(e).upper() for e in all_elements]
    if elements[0] in interstitials:
        raise ValueError(
            f"The dependent (first) element must be substitutional for u-fractions; received {elements[0]!r}."
        )


def _independent_array(values, n_independent):
    """Returns ``values`` as a float array whose last axis is the independent axis.

    Binary inputs (one independent component) may be scalars, 1-D arrays of
    nodes, or ``(n, 1)`` arrays; the first two are given a trailing axis of
    length one. The second return value restores the caller's shape.
    """
    array = np.asarray(values, dtype=np.float64)
    if n_independent == 1:
        original_shape = array.shape
        if array.ndim >= 2 and array.shape[-1] == 1:
            return array, original_shape
        return array.reshape(original_shape + (1,)), original_shape
    if array.shape[-1:] != (n_independent,):
        raise ValueError(f"Expected compositions with a trailing axis of length {n_independent}; received {array.shape}.")
    return array, array.shape


def u_to_x_independent(u, all_elements):
    """
    Converts independent u-fractions to independent mole fractions.

    Uses ``x_k = u_k / (1 + sum_I u_I)`` (the total moles per mole of
    substitutional atoms), which equals ``kawin.thermo.Mobility.u_to_x_frac``
    without a per-point matrix inverse. Shapes: binary ``()``/``(n,)``,
    multicomponent ``(e-1,)``/``(n, e-1)``.
    """
    mask = independent_interstitial_mask(all_elements)
    array, shape = _independent_array(u, mask.size)
    total = 1.0 + np.sum(array[..., mask], axis=-1, keepdims=True)
    return (array / total).reshape(shape)


def x_to_u_independent(x, all_elements):
    """
    Converts independent mole fractions to independent u-fractions.

    Uses ``u_k = x_k / (1 - sum_I x_I)``, i.e. division by the substitutional
    mole-fraction sum, matching ``kawin.thermo.Mobility.x_to_u_frac``.
    """
    mask = independent_interstitial_mask(all_elements)
    array, shape = _independent_array(x, mask.size)
    substitutional_sum = 1.0 - np.sum(array[..., mask], axis=-1, keepdims=True)
    return (array / substitutional_sum).reshape(shape)


def full_mole_fractions(u_independent, all_elements):
    """
    Returns full ``(n, e)`` mole fractions (dependent element first) from independent u-fractions.

    The dependent u-fraction is ``1 - sum(substitutional u)``; all u-fractions
    are then divided by ``1 + sum(interstitial u)``.
    """
    mask = independent_interstitial_mask(all_elements)
    u = np.atleast_2d(np.asarray(u_independent, dtype=np.float64))
    if mask.size == 1 and u.shape[-1] != 1:
        u = u.reshape(-1, 1)
    dependent = 1.0 - np.sum(u[:, ~mask], axis=1)
    total = 1.0 + np.sum(u[:, mask], axis=1)
    return np.column_stack((dependent, u)) / total[:, np.newaxis]


def record_initial_u_fractions(model):
    """
    Re-records the model's initial mesh profile as independent u-fractions.

    ``DiffusionModel.__init__`` converts the mesh profile from mole fractions to
    u-fractions in ``_validateMeshComposition``, but ``DiffusionModel.reset``
    re-records the raw mole fractions. Illingworth models call this after
    ``reset`` when interstitials are present so ``setup`` always starts from
    u-fractions. The mesh profile was already clamped in place by the first
    ``_validateMeshComposition`` call, so it is converted here without clamping
    again.
    """
    x = np.asarray(model.mesh.flattenResponse(model.mesh.y), dtype=np.float64)
    u = x_to_u_independent(x, model.allElements)
    model.data.setResponseAtN(u)
    model.data.currentY = u


def composition_bound_violation(profile, mask, min_composition):
    """
    Returns a short description of the first composition-bound violation, or ``None``.

    ``profile`` has shape ``(n_nodes, n_independent)`` in u-fractions. Each
    independent u must be at least ``min_composition`` and so must the
    dependent substitutional u-fraction ``1 - sum(substitutional u)``.
    Interstitial u-fractions have no upper bound (no sublattice site-ratio cap).
    With no interstitials this is exactly the ternary simplex test used before
    u-fraction support, including its arithmetic.
    """
    profile = np.asarray(profile, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    min_comp = float(min_composition)
    if np.any(mask):
        dependent = 1.0 - np.sum(profile[:, ~mask], axis=1)
    else:
        dependent = 1.0 - np.sum(profile, axis=1)
    component_bad = np.argwhere(profile < min_comp)
    if component_bad.size:
        node, component = component_bad[0]
        return f"node {int(node)} component {int(component)} = {profile[node, component]:.6g}"
    dependent_bad = np.flatnonzero(dependent < min_comp)
    if dependent_bad.size:
        node = int(dependent_bad[0])
        return f"node {node} dependent component = {dependent[node]:.6g}"
    return None


class UFractionDiffusivityProvider:
    """
    Thermodynamics wrapper that accepts u-fractions in ``getInterdiffusivity``.

    The composition is converted to mole fractions and every other argument,
    including ``query_context``, is forwarded unchanged, so a provider that
    rejects ``query_context`` still raises ``TypeError`` and the solvers'
    fallback paths keep working. The returned matrix is not modified: kawin
    and Thermo-Calc already return the volume-fixed u-basis diffusivity.
    Other attributes are forwarded to the wrapped object.
    """

    def __init__(self, thermodynamics, all_elements):
        self._thermodynamics = thermodynamics
        self._allElements = tuple(all_elements)

    @property
    def wrapped(self):
        return self._thermodynamics

    def getInterdiffusivity(self, u, T, *args, **kwargs):
        x = u_to_x_independent(u, self._allElements)
        return self._thermodynamics.getInterdiffusivity(x, T, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._thermodynamics, name)


class UFractionInterfaceEquilibrium:
    """
    Tie-line closure wrapper returning endpoint compositions as u-fractions.

    Endpoints are converted from mole fractions unless the wrapped closure
    declares ``composition_basis = "u"``. ``eta_bounds`` and other attributes
    are forwarded.
    """

    def __init__(self, closure, all_elements):
        self._closure = closure
        self._allElements = tuple(all_elements)
        self._basis = validate_composition_basis(getattr(closure, "composition_basis", "x"))

    @property
    def wrapped(self):
        return self._closure

    @property
    def eta_bounds(self):
        return self._closure.eta_bounds

    def interface_compositions(self, eta):
        left, right = self._closure.interface_compositions(eta)
        if self._basis == "u":
            return left, right
        return (
            x_to_u_independent(left, self._allElements),
            x_to_u_independent(right, self._allElements),
        )

    def __getattr__(self, name):
        return getattr(self._closure, name)


def resolve_diffusivity_provider(thermodynamics, all_elements):
    """Returns ``thermodynamics`` unchanged without interstitials, else a u-fraction wrapper."""
    if thermodynamics is None or not has_interstitials(all_elements):
        return thermodynamics
    return UFractionDiffusivityProvider(thermodynamics, all_elements)


def resolve_interface_equilibrium(closure, all_elements):
    """Returns ``closure`` unchanged without interstitials, else a u-fraction wrapper."""
    if closure is None or not has_interstitials(all_elements):
        return closure
    return UFractionInterfaceEquilibrium(closure, all_elements)
