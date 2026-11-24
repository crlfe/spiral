#!/usr/bin/env python3
"""Capability probes for float16 on the AMD XDNA2 NPU.

XDNA2 is arch 21 (`aie2p`). The AIE API headers only define `float16` for
arch 22 (`aie2ps`), so these probes check which float16 operations the
compiler accepts. The u16 and bf16 load/store probes are timing and capability
controls.

Expected failures are results too: each unsupported probe is compiled in
isolation so it fails only its own row. The durable findings belong in
`src/quants/DESIGN.md`; this script is the evidence.
"""

import sys
from pathlib import Path

import numpy as np

import bench_common
from bench_common import ITERS, IN_BYTES, Probe

BENCH_ROOT = Path(__file__).resolve().parent
SOURCE = BENCH_ROOT / "float16_support.aie.cc"


def _seed_input(seed):
    return np.random.default_rng(seed)


def make_u16(seed):
    rng = _seed_input(seed)
    payload = rng.integers(0, 65536, size=ITERS * 64, dtype=np.uint16)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: payload.nbytes] = payload.view(np.uint8)
    return data, data[: payload.nbytes].copy()


def check_bytes(out, expected):
    used = expected.size
    if np.array_equal(out[:used], expected):
        return 0.0, "exact"
    diff = np.abs(out[:used].astype(np.int16) - expected.astype(np.int16))
    return float(diff.max()), "byte mismatch"


def make_bf16(seed):
    from ml_dtypes import bfloat16

    rng = _seed_input(seed)
    values = rng.standard_normal(ITERS * 64).astype(np.float32).astype(bfloat16)
    payload = values.view(np.uint16)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: payload.nbytes] = payload.view(np.uint8)
    return data, data[: payload.nbytes].copy()


PROBES = [
    Probe(
        "u16_load_store",
        "U16_LOAD_STORE",
        True,
        make_u16,
        check_bytes,
        values_per_iter=64,
    ),
    Probe(
        "bf16_load_store",
        "BF16_LOAD_STORE",
        True,
        make_bf16,
        check_bytes,
        values_per_iter=64,
    ),
    Probe(
        "float16_scalar_add",
        "FLOAT16_SCALAR_ADD",
        note="expect failure: scalar float16 arithmetic",
    ),
    Probe(
        "float16_vector_decl",
        "FLOAT16_VECTOR_DECL",
        note="expect failure: vector<float16>",
    ),
    Probe(
        "float16_vector_load_store",
        "FLOAT16_VECTOR_LOAD_STORE",
        note="expect failure: float16 load/store",
    ),
    Probe(
        "float16_size",
        "FLOAT16_SIZE",
        note="expect failure: sizeof(float16) != 2",
    ),
    Probe(
        "float16_int8_convert",
        "FLOAT16_INT8_CONVERT",
        note="expect failure: int8 -> float16",
    ),
    Probe(
        "float16_acc_convert",
        "FLOAT16_ACC_CONVERT",
        note="expect failure: accfloat -> float16",
    ),
    Probe("float16_add", "FLOAT16_ADD", note="expect failure: float16 add"),
    Probe("float16_mul", "FLOAT16_MUL", note="expect failure: float16 mul"),
    Probe("float16_mmul", "FLOAT16_MMUL", note="expect failure: float16 mmul"),
]


if __name__ == "__main__":
    sys.exit(bench_common.run(__doc__, SOURCE, PROBES))
