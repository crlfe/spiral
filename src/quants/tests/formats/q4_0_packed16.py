"""Q4_0 variant with host-packed 16-bit scale codes.

The canonical source format and dequantized reference are identical to Q4_0;
only the packed scale representation differs. The packed scale keeps a 9-bit
significand, so weights are near-exact rather than bit-exact.
"""

import numpy as np

from formats.q4_0 import *  # noqa: F401,F403


def output_tolerance(expected_outputs):
    """Near-exact mmul policy: 5% of the per-column output scale."""
    column_scale = np.abs(expected_outputs).max(axis=0)
    return np.maximum(1e-5, 0.05 * column_scale)


def verify(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return False, f"shape mismatch: {actual.shape} != {expected.shape}"
    if np.array_equal(actual, expected):
        return True, "exact bf16 match"

    actual_f = actual.astype(np.float32)
    expected_f = expected.astype(np.float32)
    error = np.abs(actual_f - expected_f)
    zero = expected_f == 0
    if np.any(actual_f[zero] != 0):
        return False, "non-zero output for a zero weight"
    nonzero = ~zero
    if np.any(nonzero):
        relative = error[nonzero] / np.abs(expected_f[nonzero])
        if relative.max() > 0.02:
            return False, (
                f"max relative error {relative.max():.6f} "
                f"(max abs error {error.max():.6f})"
            )
        return True, (
            f"within packed16 tolerance (max abs error {error.max():.6f}, "
            f"max relative error {relative.max():.6f})"
        )
    return True, "zero weights exact"
