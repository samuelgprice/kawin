"""Regenerates the bitwise substitutional Illingworth baseline.

Run this only on code whose substitutional behaviour is known-good (it was
first run on commit 6daa5af, before interstitial support was added, and
regenerated when the ternary interface Newton began evaluating every trial on
its own motion branch; that changed only ``ternary_uniform_left``, whose
receding interface previously relied on downwind trial evaluations, by about
1e-10 relative, the Newton tolerance level; and again when the ternary model's
``interface_newton_start`` default became ``'extrapolated'``, which moved the
four ternary cases by at most 2e-10 relative while leaving the binary and
three-phase cases bitwise unchanged). The
regression test ``test_illingworth_substitutional_baseline.py`` compares
fresh runs against the saved arrays with ``np.array_equal`` (NaN-aware: the three-phase
physical profile is NaN outside the moving domain).
"""

import numpy as np

from kawin.tests.illingworth_baseline_cases import BASELINE_PATH, run_all_cases


def main():
    results = run_all_cases()
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BASELINE_PATH, **results)
    print(f"Saved {len(results)} arrays to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
