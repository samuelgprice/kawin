"""Shared diffusivity-mode helpers from the ternary module at 06dc7b4ab56f67226ceabfde41ebf2690c088841."""

_BULK_DIFFUSIVITY_PHASE_UNIFORM = "phase_uniform"


_BULK_DIFFUSIVITY_LAGGED = "composition_dependent_lagged"


_BULK_DIFFUSIVITY_IMPLICIT = "composition_dependent_implicit"


_SUPPORTED_BULK_DIFFUSIVITY_MODES = {
    _BULK_DIFFUSIVITY_PHASE_UNIFORM,
    _BULK_DIFFUSIVITY_LAGGED,
    _BULK_DIFFUSIVITY_IMPLICIT,
}


def _coerce_bulk_diffusivity_mode(mode):
    """Normalizes and validates the ternary Illingworth bulk diffusivity mode."""
    value = str(mode)
    if value not in _SUPPORTED_BULK_DIFFUSIVITY_MODES:
        raise ValueError(
            "bulk_diffusivity_mode must be 'phase_uniform', "
            "'composition_dependent_lagged', or 'composition_dependent_implicit'."
        )
    return value
