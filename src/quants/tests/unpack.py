"""Repack and NPU-unpack test for `src/quants` formats."""

import argparse
import importlib
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import common
import numpy as np
from ml_dtypes import bfloat16


def parse_args():
    parser = argparse.ArgumentParser(description="quants unpack test")
    parser.add_argument("--format", required=True, choices=["q4_0", "q6_k"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weights-rows", type=int)
    parser.add_argument("--weights-cols", type=int)
    return parser.parse_args()


def resolve_shape(args, fmt):
    rows = (
        args.weights_rows
        if args.weights_rows is not None
        else fmt.DEFAULT_ROWS
    )
    cols = (
        args.weights_cols
        if args.weights_cols is not None
        else fmt.DEFAULT_COLS
    )
    constants = common.layout_constants(args.format)
    if rows <= 0 or cols <= 0:
        raise ValueError("weights_rows and weights_cols must be positive")
    if rows % constants["kRecordRows"] != 0:
        raise ValueError(
            f"weights_rows must be a multiple of "
            f"{constants['kRecordRows']}"
        )
    if cols % constants["kTileCols"] != 0:
        raise ValueError(
            f"weights_cols must be a multiple of {constants['kTileCols']}"
        )
    return rows, cols, constants


def main():
    args = parse_args()
    try:
        fmt = importlib.import_module(f"formats.{args.format}")
    except ModuleNotFoundError as exc:
        raise SystemExit(f"unknown format: {args.format}") from exc

    try:
        weights_rows, weights_cols, constants = resolve_shape(args, fmt)
    except ValueError as exc:
        raise SystemExit(f"invalid shape: {exc}") from exc

    rng = np.random.default_rng(args.seed)
    raw = fmt.make_raw(weights_rows, weights_cols, rng)
    src_col_stride_bytes = (
        weights_rows // constants["kBlockRows"]
    ) * constants["kBlockBytes"]
    packed = common.repack_host(
        args.format,
        raw,
        weights_rows,
        weights_cols,
        src_col_stride_bytes,
    )
    expected = fmt.reference_weights(raw, weights_rows, weights_cols)
    tiles = expected.size // (
        constants["kUnpackTileRows"] * constants["kTileCols"]
    )

    print(f"format: {args.format}")
    print(f"weights: {weights_rows} rows x {weights_cols} cols")
    print(f"packed bytes: {packed.size}")

    if not common.npu_available():
        print("SKIP: no NPU device available")
        return 0

    try:
        from aie import iron

        program = common.build_unpack_program(
            args.format,
            packed.size,
            expected.size,
            weights_rows,
            weights_cols,
        )
        packed_tensor = iron.tensor(packed, dtype=np.uint8)
        actual = iron.zeros((expected.size,), dtype=bfloat16)
        intervals = common.run_traced(
            program,
            [packed_tensor, actual],
            f"quants_unpack_{args.format}_{weights_rows}x{weights_cols}",
        )
    except common.SkipTest as exc:
        print(f"SKIP: {exc}")
        return 0
    except Exception as exc:
        if common.is_device_error(exc):
            print(f"SKIP: {exc}")
            return 0
        raise

    actual_view = np.asarray(actual)
    top = [
        interval
        for interval in common.flatten_intervals(intervals)
        if interval["depth"] == 0
    ]
    cycles = common.total_cycles(top)
    print(f"cycles: {cycles}")
    print(f"cycles/tile: {cycles / tiles:.3f}")

    ok, message = fmt.verify(actual_view, expected)
    print(message if ok else f"FAILED: {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
