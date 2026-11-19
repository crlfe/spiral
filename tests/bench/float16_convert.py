#!/usr/bin/env python3
"""float16 bits -> f32 conversion probes.

XDNA2 has no float16 type, so the only useful conversion is from the raw
16-bit pattern to f32. These probes compare the mask/rebias decode with the
shift-based decode. The scale experiments start from these primitives.
"""

import sys
from pathlib import Path

import numpy as np

import bench_common
from bench_common import ITERS, IN_BYTES, Probe

BENCH_ROOT = Path(__file__).resolve().parent
SOURCE = BENCH_ROOT / "float16_convert.aie.cc"


def make_float16_bits(seed):
    rng = np.random.default_rng(seed)
    values = rng.uniform(-8.0, 8.0, size=ITERS * 8).astype(np.float16)
    payload = values.view(np.uint16)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: payload.nbytes] = payload.view(np.uint8)
    return data, values.astype(np.float32)


def check_f32(out, expected):
    used = expected.nbytes
    actual = out[:used].view(np.float32)
    error = np.abs(actual - expected)
    return float(error.max()), "emulated f32" if error.max() else "exact"


PROBES = [
    Probe(
        "float16_bits_f32_x8",
        "FLOAT16_BITS_F32_X8",
        True,
        make_float16_bits,
        check_f32,
    ),
    Probe(
        "float16_shift_f32_x8",
        "FLOAT16_SHIFT_F32_X8",
        True,
        make_float16_bits,
        check_f32,
    ),
]


if __name__ == "__main__":
    sys.exit(bench_common.run(__doc__, SOURCE, PROBES))
