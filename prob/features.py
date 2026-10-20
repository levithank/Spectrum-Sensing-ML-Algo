"""
Hand-crafted, interpretable features for PSK vs ASK.

The idea, in constellation terms:
  * ASK puts every cluster on ONE straight line through the origin (phases 0 / 180 deg).
    Squaring each sample doubles its phase, so 0 and 180 both land on 0 deg: the squared
    samples all point the same way and their average is large.
  * M-PSK spreads clusters evenly around the circle (360/M deg apart). Squaring QPSK
    gives points at +90 and -90 deg, which cancel: the average of x^2 is ~0.
    Raising to the 4th power collapses QPSK instead (4 x 90 = 360).
  * PSK has constant amplitude; ASK amplitude jumps between levels.

All features are normalised by signal+noise power and use magnitudes of averages,
so they don't care about the receive gain or the (unknown) carrier phase.
Note: with AWGN, E[n^2] = E[n^4] = 0, so the noise does not bias the numerators,
it only adds variance and inflates the denominators.
"""

import numpy as np

FEATURE_NAMES = ["circ2", "circ4", "amp_cv", "kurt"]

FEATURE_HELP = {
    "circ2": "|mean(x^2)| / mean(|x|^2)  - 'are all clusters on one line?'  ASK high, PSK ~0",
    "circ4": "|mean(x^4)| / mean(|x|^2)^2 - 4-fold symmetry, high for QPSK and ASK",
    "amp_cv": "std(|x|) / mean(|x|)         - amplitude spread, PSK low, ASK higher",
    "kurt": "mean(|x|^4) / mean(|x|^2)^2  - amplitude kurtosis (2.0 for pure noise)",
}


def extract(X):
    """X: complex array [num_examples, n] -> float array [num_examples, len(FEATURE_NAMES)]."""
    X = np.atleast_2d(np.asarray(X, dtype=np.complex128))
    p = np.mean(np.abs(X) ** 2, axis=1)              # signal + noise power
    a = np.abs(X)

    circ2 = np.abs(np.mean(X ** 2, axis=1)) / p
    circ4 = np.abs(np.mean(X ** 4, axis=1)) / p ** 2
    amp_cv = a.std(axis=1) / a.mean(axis=1)
    kurt = np.mean(a ** 4, axis=1) / p ** 2

    return np.stack([circ2, circ4, amp_cv, kurt], axis=1)
