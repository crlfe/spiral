"""End-to-end fused unpack + bf16 mmul test for `src/quants` formats."""

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
    parser = argparse.ArgumentParser(description="quants mmul test")
    common.add_common_args(parser)
    return parser.parse_args()


def make_inputs(weights_rows, seed, inputs_vecs, inputs_len_tile):
    rng = np.random.default_rng(seed + 1000)
    inputs_rows = rng.integers(
        -4, 5, (inputs_vecs, weights_rows)
    ).astype(np.float32)
    packed = np.concatenate(
        [
            inputs_rows[
                :, tile * inputs_len_tile : (tile + 1) * inputs_len_tile
            ].reshape(-1)
            for tile in range(weights_rows // inputs_len_tile)
        ]
    )
    return packed.astype(bfloat16), inputs_rows


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
    constants = case.constants
    inputs_vecs = constants["kMmulInputsVecs"]
    inputs_len_tile = constants["kMmulInputsLen"]
    outputs_len_tile = constants["kMmulOutputsLen"]
    packed_inputs, inputs_rows = make_inputs(
        case.weights_rows, args.seed, inputs_vecs, inputs_len_tile
    )
    cols_tiles = case.weights_cols // constants["kTileCols"]
    outputs_elems = cols_tiles * inputs_vecs * outputs_len_tile

    common.print_case_header(case)

    if not common.npu_available():
        print("ERROR: no NPU device available")
        return 1

    try:
        from aie import iron

        program = common.build_mmul_program(
            case.format_name,
            case.packed.size,
            packed_inputs.size,
            outputs_elems,
            case.weights_rows,
            case.weights_cols,
        )
        packed_tensor = iron.tensor(case.packed, dtype=np.uint8)
        inputs_tensor = iron.tensor(packed_inputs, dtype=bfloat16)
        outputs = iron.zeros((outputs_elems,), dtype=np.float32)
        intervals, kernel_dir = common.run_traced(
            program,
            [packed_tensor, inputs_tensor, outputs],
            f"quants_mmul_{case.format_name}_"
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

    flat = common.top_level_intervals(intervals)
    cycles = common.total_cycles(flat)
    print(f"cycles: {cycles}")
    print(f"cycles/tile: {cycles / case.tiles:.3f}")
    print(f"weights/cycle: {case.expected_weights.size / cycles:.3f}")
    print(f"trace intervals: {len(flat)}")
    common.print_program_size(kernel_dir)

    weights_matrix = (
        case.expected_weights.reshape(
            cols_tiles, case.weights_rows, constants["kTileCols"]
        )
        .transpose(1, 0, 2)
        .reshape(case.weights_rows, case.weights_cols)
        .astype(np.float32)
    )
    expected_outputs = inputs_rows @ weights_matrix
    actual = np.asarray(outputs, dtype=np.float32).reshape(
        cols_tiles, inputs_vecs, outputs_len_tile
    )
    actual_outputs = actual.transpose(1, 0, 2).reshape(
        inputs_vecs, case.weights_cols
    )
    error = np.abs(actual_outputs - expected_outputs)

    if np.allclose(actual_outputs, expected_outputs, rtol=1e-3, atol=1e-3):
        print(f"verified mmul (max abs error {error.max():.6f})")
        return 0

    print(f"FAILED mmul: max abs error {error.max():.6f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
