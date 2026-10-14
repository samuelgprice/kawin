"""Regenerates the bitwise substitutional Illingworth baseline.

Run this only on code whose substitutional behaviour is known-good (it was
first run on commit 6daa5af, before interstitial support was added). The
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
