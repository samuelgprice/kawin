# Relocated from samuelgprice/kawin 06dc7b4ab56f67226ceabfde41ebf2690c088841
# Source: kawin/diffusion/_spectral_validation.py; import changes only. See NOTICE and docs/LEDGERS.md.
"""Shared scale-invariant spectral acceptance constants for ternary diffusion."""

# A conjugate pair whose scaled imaginary component is at most this value is
# treated as numerically real by the production ternary diffusion path.
TERNARY_DIFFUSIVITY_REAL_SPECTRUM_TOL = 1.0e-12

# The existing Illingworth solver accepts only scaled eigenvalues above this
# numerical usability threshold. This is distinct from mathematical positive
# representability, which only requires an eigenvalue to be strictly positive.
TERNARY_DIFFUSIVITY_POSITIVE_EIGENVALUE_TOL = 1.0e-14
