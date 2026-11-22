"""Python test runner for the config-driven Spiral NPU program.

Run with ``python src/program/tests.py`` (or ``python src/program/run.py``).
The device program itself lives in ``run.py``; this module only builds host
buffers and checks the returned outputs.
"""

import sys

import numpy as np
from aie import iron
from ml_dtypes import bfloat16

import design
import pack
from run import program

Q4_0_TILE_BYTES = 2304
Q6_K_TILE_BYTES = 3392


def _native_block(g, c, band, half):
    """Block index in the production [N/8][vec][8] output layout."""
    half_cols = design.COLS_TILE // 2
    blocks_per_band = design.BAND_COLS // half_cols
    blocks_per_compute = design.COLS_TILE // half_cols
    blocks_per_group = design.COMPUTES_PER_GROUP * blocks_per_compute
    return (
        band * blocks_per_band
        + g * blocks_per_group
        + c * blocks_per_compute
        + half
    )


def _native_tile(actual, g, c, band):
    """Gather one compute's native blocks into float [vec][16]."""
    half_cols = design.COLS_TILE // 2
    block_elems = design.INPUTS_VECS * half_cols
    tile = np.zeros((design.INPUTS_VECS, design.COLS_TILE), dtype=np.float32)
    for half in range(2):
        start = _native_block(g, c, band, half) * block_elems
        for vec in range(design.INPUTS_VECS):
            tile[vec, half * half_cols : (half + 1) * half_cols] = (
                np.asarray(actual[start + vec * half_cols :
                                  start + (vec + 1) * half_cols],
                           dtype=np.float32)
            )
    return tile


def _native_compute_bits(bits, g, c):
    """Invert the native group drain for one compute's flat output object."""
    group_offset = (
        g * design.COMPUTES_PER_GROUP * design.COLS_TILE *
        design.INPUTS_VECS
    )
    compute_offset = c * design.COLS_TILE * design.INPUTS_VECS
    band_stride = design.BAND_COLS * design.INPUTS_VECS
    obj = np.zeros(design.OUTPUT_ELEMS, dtype=np.uint16)
    for band in range(design.MAX_BANDS_PER_CHUNK):
        src = group_offset + compute_offset + band * band_stride
        dst = band * design.OUTPUT_TILE_ELEMS
        obj[dst : dst + design.OUTPUT_TILE_ELEMS] = bits[
            src : src + design.OUTPUT_TILE_ELEMS
        ]
    return obj


def _real_error_metrics(actual, expected, cols_bands, rel=6e-3):
    """Return (max abs error, worst error / per-column tolerance).

    The device accumulates K terms in fp32, so the absolute error grows with
    K and with the column's output scale.  The device also stores the epilogue
    result as bf16 for direct chaining, which dominates the error.  The
    pass/fail quantity is the element's error against
    ``max(1e-3, 6e-3 * column_scale)``, where the column scale is the largest
    expected magnitude in that output column over the four input vectors.
    This mirrors host/test.cc.
    """
    max_error = 0.0
    max_ratio = 0.0
    for g in range(design.GROUPS):
        for c in range(design.COMPUTES_PER_GROUP):
            for band in range(cols_bands):
                tile = _native_tile(actual, g, c, band)
                reference = expected[g, c, band]
                error = np.abs(tile - reference)
                column_scale = np.abs(reference).max(axis=0)
                tolerance = np.maximum(1e-3, rel * column_scale)
                max_error = max(max_error, float(error.max()))
                max_ratio = max(max_ratio, float((error / tolerance).max()))
    return max_error, max_ratio


def _noop_mismatches(actual, expected, total_tiles, cols_bands):
    mismatches = 0
    first = None
    bits = np.asarray(actual).view(np.uint16)
    for g in range(design.GROUPS):
        for c in range(design.COMPUTES_PER_GROUP):
            obj = _native_compute_bits(bits, g, c).view(np.uint32)
            for tile_index in range(total_tiles):
                got = obj[tile_index]
                want = expected[g, c, tile_index]
                if got != want:
                    mismatches += 1
                    if first is None:
                        first = (g, c, tile_index, int(got), int(want))
    return mismatches, first


def run_case(rows_steps, cols_bands, tile_bytes, seed=1,
             format=design.MODE_TILE_CHECK, soft_cap=0.0, rel=6e-3):
    if format == design.MODE_TILE_CHECK:
        case = pack.build_case(rows_steps, cols_bands, tile_bytes, seed)
    elif format == design.MODE_RAW_CHECK:
        case = pack.build_case(
            rows_steps, cols_bands, tile_bytes, seed, format_id=format
        )
    elif format == design.FORMAT_Q4_0:
        case = pack.build_q4_0_case(
            rows_steps, cols_bands, seed, soft_cap=soft_cap
        )
    elif format == design.FORMAT_Q6_K:
        case = pack.build_q6_k_case(rows_steps, cols_bands, seed)
    else:
        raise ValueError(f"unknown format {format}")
    weights_blocks = case["weights_blocks"]
    total_tiles = case["total_tiles"]
    if weights_blocks > design.MAX_WEIGHT_BLOCKS:
        raise ValueError(
            f"case needs {weights_blocks} blocks, max is "
            f"{design.MAX_WEIGHT_BLOCKS}"
        )

    weights_host = np.zeros(design.MAX_WEIGHT_WORDS, dtype=np.uint32)
    actual_group_bytes = weights_blocks * design.L2_STREAM_BYTES
    max_group_bytes = design.MAX_WEIGHT_BLOCKS * design.L2_STREAM_BYTES
    weights_bytes = weights_host.view(np.uint8).reshape(
        design.GROUPS, max_group_bytes
    )
    weights_bytes[:, :actual_group_bytes] = case["weights"].reshape(
        design.GROUPS, actual_group_bytes
    )

    inputs_host = np.zeros(design.MAX_INPUT_WORDS, dtype=np.uint32)
    inputs_words = inputs_host.view(np.uint16)
    input_groups = (
        rows_steps + design.INPUT_GROUP_ROWS - 1
    ) // design.INPUT_GROUP_ROWS
    input_objects = input_groups
    inputs_words[: input_objects * design.INPUT_GROUP_ELEMS] = case[
        "inputs"
    ].reshape(-1)

    config_host = np.zeros(design.INPUT_GROUP_ELEMS // 2, dtype=np.uint32)
    config_host[: design.CONFIG_LEN] = case["config"].view(np.uint32)

    weights_tensor = iron.tensor(weights_host, dtype=np.uint32)
    inputs_tensor = iron.tensor(inputs_host, dtype=np.uint32)
    config_tensor = iron.tensor(config_host, dtype=np.uint32)
    outputs_tensor = iron.zeros(
        design.MAX_OUTPUT_ELEMS, dtype=bfloat16
    )

    program(weights_tensor, inputs_tensor, config_tensor, outputs_tensor)

    if format == design.MODE_RAW_CHECK:
        blocks = weights_blocks
        expected = (
            case["weights"]
            .reshape(
                design.GROUPS,
                blocks,
                design.COMPUTES_PER_GROUP,
                design.L1_STREAM_BYTES,
            )
            .view(np.uint32)
            .sum(axis=3, dtype=np.uint64)
            .astype(np.uint32)
            .transpose(0, 2, 1)
        )
        bits = np.asarray(outputs_tensor).view(np.uint16)
        actual = np.stack(
            [
                [
                    _native_compute_bits(bits, g, c)
                    for c in range(design.COMPUTES_PER_GROUP)
                ]
                for g in range(design.GROUPS)
            ],
            axis=0,
        ).view(np.uint32)[:, :, :blocks]
        mismatches = int(np.count_nonzero(actual != expected))
        print(
            f"raw rows_steps={rows_steps} cols_bands={cols_bands} "
            f"blocks={blocks} mismatches={mismatches}"
        )
        return mismatches == 0

    if format != design.MODE_TILE_CHECK:
        max_error, max_ratio = _real_error_metrics(
            np.asarray(outputs_tensor), case["expected"], cols_bands, rel
        )
        label = "q4_0" if format == design.FORMAT_Q4_0 else "q6_k"
        print(
            f"{label} rows_steps={rows_steps} cols_bands={cols_bands} "
            f"tile={tile_bytes} blocks={weights_blocks} "
            f"tiles={total_tiles} max_error={max_error:.6f} "
            f"worst_tolerance={max_ratio:.3f}"
        )
        return max_ratio <= 1.0

    mismatches, first = _noop_mismatches(
        np.asarray(outputs_tensor),
        case["expected"],
        total_tiles,
        cols_bands,
    )
    print(
        f"rows_steps={rows_steps} cols_bands={cols_bands} "
        f"tile={tile_bytes} blocks={weights_blocks} "
        f"tiles={total_tiles} mismatches={mismatches}"
    )
    if first:
        print(f"  first mismatch g,c,t got want: {first}")
    return mismatches == 0


def run_head_chunks(
    rows_steps=6, total_bands=512, chunk_bands=design.MAX_BANDS_PER_CHUNK
):
    """Run the Q6_K output-head shape as disjoint band chunks."""
    remaining = total_bands
    chunk = 0
    status = True
    while remaining > 0:
        bands = min(chunk_bands, remaining)
        print(
            f"head chunk {chunk}: bands {total_bands - remaining}"
            f"..{total_bands - remaining + bands}"
        )
        status &= run_case(
            rows_steps,
            bands,
            Q6_K_TILE_BYTES,
            seed=100 + chunk,
            format=design.FORMAT_Q6_K,
        )
        remaining -= bands
        chunk += 1
    print(
        f"head: {chunk} chunks, {total_bands} bands, "
        f"{total_bands * design.BAND_COLS} columns"
    )
    return status


def main():
    if "--head" in sys.argv:
        return 0 if run_head_chunks() else 1

    noop_cases = [
        (1, 1, Q4_0_TILE_BYTES),
        (4, 1, Q4_0_TILE_BYTES),
        (4, 4, Q4_0_TILE_BYTES),
        (1, 1, Q6_K_TILE_BYTES),
        (6, 12, Q6_K_TILE_BYTES),
        (5, 13, Q6_K_TILE_BYTES),
        (48, 2, Q4_0_TILE_BYTES),
        (48, 2, Q6_K_TILE_BYTES),
        (16, 5, Q6_K_TILE_BYTES),
    ]
    q4_0_cases = [
        (1, 1),
        (4, 1),
        (4, 4),
        (48, 3),
        (6, 12),
    ]
    q6_k_cases = [
        (1, 1),
        (6, 12),
        (6, 24),
    ]
    raw_cases = [
        (1, 1, Q4_0_TILE_BYTES),
        (4, 4, Q4_0_TILE_BYTES),
        (6, 12, Q6_K_TILE_BYTES),
    ]

    status = True
    for rows_steps, cols_bands, tile_bytes in noop_cases:
        status &= run_case(rows_steps, cols_bands, tile_bytes)
    for rows_steps, cols_bands in q4_0_cases:
        status &= run_case(
            rows_steps, cols_bands, Q4_0_TILE_BYTES, format=design.FORMAT_Q4_0
        )
    for rows_steps, cols_bands in q6_k_cases:
        status &= run_case(
            rows_steps, cols_bands, Q6_K_TILE_BYTES, format=design.FORMAT_Q6_K
        )
    for rows_steps, cols_bands, tile_bytes in raw_cases:
        status &= run_case(
            rows_steps,
            cols_bands,
            tile_bytes,
            format=design.MODE_RAW_CHECK,
        )
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
