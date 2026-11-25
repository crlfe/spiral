"""Standalone RMS norm test for `src/funcs/rmsnorm`.

The device kernel processes the program's input-group layout: two groups for
K=1536 (four real row steps, then two real plus two padded row steps).  The
reference normalizes the same bf16 input in f32, so the residual is the device
bf16 rounding plus the r/w hi/lo split.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import common


def parse_args():
    parser = argparse.ArgumentParser(description="src/funcs RMS norm test")
    parser.add_argument("--rows-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def float_bits(value):
    return int(np.array([value], dtype=np.float32).view(np.int32)[0])


def pack_groups(inputs, rows_steps, const):
    groups = (rows_steps + const["kRowsPerGroup"] - 1) // const["kRowsPerGroup"]
    packed = np.zeros(
        groups * const["kGroupInputElems"], dtype=bfloat16
    )
    k_chunk = const["kChunk"]
    vecs = const["kVecs"]
    row_step = const["kRowStep"]
    chunks = const["kChunksPerGroup"]
    for g in range(groups):
        base = g * const["kGroupInputElems"]
        for within in range(const["kRowsPerGroup"]):
            row = g * const["kRowsPerGroup"] + within
            if row >= rows_steps:
                continue
            for chunk in range(chunks):
                for v in range(vecs):
                    src = inputs[
                        v,
                        row * row_step + chunk * k_chunk :
                        row * row_step + (chunk + 1) * k_chunk,
                    ]
                    dst = base + (
                        (within * chunks + chunk) * vecs + v
                    ) * k_chunk
                    packed[dst : dst + k_chunk] = src
    return packed


def unpack_groups(actual, rows_steps, const):
    groups = (rows_steps + const["kRowsPerGroup"] - 1) // const["kRowsPerGroup"]
    out = np.zeros(
        (const["kVecs"], rows_steps * const["kRowStep"]), dtype=np.float32
    )
    k_chunk = const["kChunk"]
    vecs = const["kVecs"]
    row_step = const["kRowStep"]
    chunks = const["kChunksPerGroup"]
    values = np.asarray(actual, dtype=np.float32)
    for g in range(groups):
        base = g * const["kGroupInputElems"]
        for within in range(const["kRowsPerGroup"]):
            row = g * const["kRowsPerGroup"] + within
            if row >= rows_steps:
                continue
            for chunk in range(chunks):
                for v in range(vecs):
                    src = base + (
                        (within * chunks + chunk) * vecs + v
                    ) * k_chunk
                    out[
                        v,
                        row * row_step + chunk * k_chunk :
                        row * row_step + (chunk + 1) * k_chunk,
                    ] = values[src : src + k_chunk]
    return out


def main():
    args = parse_args()
    if args.rows_steps <= 0:
        raise SystemExit("--rows-steps must be positive")
    const = common.layout_constants()
    rows_steps = args.rows_steps
    k = rows_steps * const["kRowStep"]
    groups = (rows_steps + const["kRowsPerGroup"] - 1) // const["kRowsPerGroup"]
    input_elems = groups * const["kGroupInputElems"]

    rng = np.random.default_rng(args.seed)
    inputs = rng.normal(0.0, 1.0, (const["kVecs"], k)).astype(np.float32)
    inputs_bf16 = inputs.astype(bfloat16)
    weight = rng.normal(0.0, 1.0, (k,)).astype(np.float32)
    weight_hi = weight.astype(bfloat16)
    weight_lo = (weight - weight_hi.astype(np.float32)).astype(bfloat16)
    # Device layout: per 8-value K chunk, [hi8][lo8].
    weight_packed = np.stack(
        [weight_hi.reshape(-1, 8), weight_lo.reshape(-1, 8)], axis=1
    ).reshape(-1)

    packed = pack_groups(inputs_bf16, rows_steps, const)
    if packed.size != input_elems:
        raise AssertionError("packed input size mismatch")

    reference_input = inputs_bf16.astype(np.float32)
    mean_sq = np.mean(reference_input * reference_input, axis=1)
    rms = np.sqrt(mean_sq + args.eps)
    reference = (
        reference_input / rms[:, None] * weight[None, :]
    ).astype(np.float32)

    print(f"rows_steps: {rows_steps}, K: {k}, groups: {groups}")
    print(
        f"input elems: {input_elems}, weight elems: {weight_packed.size} "
        f"({k} hi + {k} lo)"
    )

    if not common.npu_available():
        print("ERROR: no NPU device available")
        return 1

    try:
        from aie import iron

        program = common.build_rmsnorm_program(
            rows_steps, k, float_bits(args.eps), float_bits(1.0 / k)
        )
        inputs_tensor = iron.tensor(packed, dtype=bfloat16)
        weights_tensor = iron.tensor(weight_packed, dtype=bfloat16)
        outputs = iron.zeros(input_elems, dtype=bfloat16)
        intervals, kernel_dir = common.run_traced(
            program,
            [inputs_tensor, weights_tensor, outputs],
            f"funcs_rmsnorm_{rows_steps}x{k}",
        )
    except common.NpuUnavailable as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        if common.is_device_error(exc):
            print(f"ERROR: {exc}")
            return 1
        raise

    actual = unpack_groups(outputs, rows_steps, const)
    error = np.abs(actual - reference)
    worst_ratio = 0.0
    for v in range(const["kVecs"]):
        scale = float(np.max(np.abs(reference[v])))
        tolerance = max(1e-3, 6e-3 * scale)
        worst_ratio = max(worst_ratio, float(np.max(error[v])) / tolerance)

    flat = common.top_level_intervals(intervals)
    cycles = common.total_cycles(flat)
    print(f"cycles: {cycles}")
    for index, interval in enumerate(flat):
        print(f"  interval {index}: {interval['cycles']} cycles")
    print(f"cycles/row-step: {cycles / rows_steps:.3f}")
    print(f"max abs error: {error.max():.6f}")
    print(f"worst tolerance ratio: {worst_ratio:.3f}")
    common.print_program_size(kernel_dir)
    if worst_ratio <= 1.0:
        print("verified rmsnorm")
        return 0
    print("FAILED rmsnorm")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
