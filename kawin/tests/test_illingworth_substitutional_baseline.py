"""Bitwise regression for substitutional-only Illingworth runs.

The arrays in ``data/illingworth_substitutional_baseline.npz`` were produced by
``make_illingworth_substitutional_baseline.py`` before interstitial (u-fraction)
support was added. Substitutional systems have u == x, and the interstitial
conversion layer returns the original thermodynamics and closure objects when
no interstitial is present, so every recorded history must match exactly.
"""

import numpy as np
import pytest

from kawin.tests.illingworth_baseline_cases import BASELINE_PATH, run_all_cases


@pytest.fixture(scope="module")
def fresh_results():
    return run_all_cases()


def test_illingworth_substitutional_baseline_unchanged(fresh_results):
    if not BASELINE_PATH.exists():
        pytest.skip("Substitutional baseline file is missing.")
    with np.load(BASELINE_PATH) as stored:
        stored_keys = set(stored.files)
        assert stored_keys == set(fresh_results), sorted(stored_keys ^ set(fresh_results))
        mismatched = []
        for key in sorted(stored_keys):
            expected = stored[key]
            actual = np.asarray(fresh_results[key])
            if expected.shape != actual.shape or not np.array_equal(expected, actual, equal_nan=True):
                mismatched.append(key)
        assert not mismatched, f"Substitutional baseline changed for: {mismatched}"
