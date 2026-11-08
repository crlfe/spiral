"""Quant format/layout conformance test.

Verifies host repack and exact bf16 unpack against the Python reference.
The fused unpack + mmul path is covered by `mmul.py`.
"""

import argparse
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
    common.add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    fmt = common.load_format(args.format)

    try:
        weights_rows, weights_cols, _ = common.resolve_shape(
            fmt, args.format, args.rows, args.cols
        )
    except ValueError as exc:
        raise SystemExit(f"invalid shape: {exc}") from exc

    case = common.prepare_case(
        fmt, args.format, args.seed, weights_rows, weights_cols
    )
    common.print_case_header(case)

    if not common.npu_available():
        print("ERROR: no NPU device available")
        return 1

    try:
        from aie import iron

        program = common.build_unpack_program(
            case.format_name,
            case.packed.size,
            case.expected_weights.size,
            case.weights_rows,
            case.weights_cols,
        )
        packed_tensor = iron.tensor(case.packed, dtype=np.uint8)
        actual = iron.zeros(
            (case.expected_weights.size,), dtype=bfloat16
        )
        intervals = common.run_traced(
            program,
            [packed_tensor, actual],
            f"quants_unpack_{case.format_name}_"
            f"{case.weights_rows}x{case.weights_cols}",
        )
    except common.NpuUnavailable as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        if common.is_device_error(exc):
            print(f"ERROR: {exc}")
            return 1
        raise

    actual_view = np.asarray(actual)
    top = common.top_level_intervals(intervals)
    cycles = common.total_cycles(top)
    print(f"cycles: {cycles}")
    print(f"cycles/tile: {cycles / case.tiles:.3f}")
    print(f"weights/cycle: {case.expected_weights.size / cycles:.3f}")

    ok, message = case.fmt.verify(actual_view, case.expected_weights)
    print(message if ok else f"FAILED: {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
