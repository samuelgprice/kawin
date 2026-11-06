"""Numerical contracts for the extracted scalar kernel and optional acceleration."""

import subprocess
import sys

import numpy as np
import pytest

from moving_boundary.solvers.illingworth.numerics import tridiagonal as kernel


def test_dense_solution_and_input_preservation():
    """Check the sign convention against an independent dense solve."""
    lo = np.array([0., -1., -2., -1.])
    diag = np.array([4., 5., 6., 3.])
    up = np.array([-2., -1., -.5, 0.])
    rhs = np.array([1., 2., -1., .5])
    inputs = [value.copy() for value in (lo, diag, up, rhs)]
    matrix = np.diag(diag) + np.diag(lo[1:], -1) + np.diag(up[:-1], 1)
    expected = np.linalg.solve(matrix, rhs)
    fallback = kernel._solve_illingworth_tridiagonal_python(*inputs)
    result = kernel.solve_illingworth_tridiagonal(*inputs)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(result, fallback)
    for original, after in zip((lo, diag, up, rhs), inputs):
        np.testing.assert_array_equal(original, after)


def test_pivot_and_length_errors():
    """Preserve existing failure types without adding pivoting or regularization."""
    with pytest.raises(ValueError, match="same length"):
        kernel.solve_illingworth_tridiagonal([0.], [1., 1.], [0.], [1.])
    for solve in (kernel.solve_illingworth_tridiagonal,
                  kernel._solve_illingworth_tridiagonal_python):
        with pytest.raises(ZeroDivisionError, match="zero pivot"):
            solve(np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1))


def test_kernel_import_and_fallback_without_host_or_numba():
    """Use a fresh process with host and accelerator imports actively excluded."""
    code = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'kawin', 'numba'}:
            raise ModuleNotFoundError('excluded for fallback test', name=fullname)
sys.meta_path.insert(0, Block())
from moving_boundary.solvers.illingworth.numerics import tridiagonal as kernel
assert kernel._solve_illingworth_tridiagonal_numba is None
assert kernel.solve_illingworth_tridiagonal([0.], [-2.], [0.], [-1.])[0] == .5
assert 'kawin' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
