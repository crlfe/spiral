"""Case builders for the NPU program tests and C++ golden data."""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from design import (
    BAND_COLS,
    CFG_COLS_BANDS,
    CFG_FORMAT,
    CFG_ROWS_STEPS,
    CFG_SOFT_CAP_HI_LO,
    CFG_SOFT_INV_HI_LO,
    CFG_TILE_BYTES,
    CFG_TOTAL_TILES,
    FORMAT_Q4_0,
    FORMAT_Q6_K,
    MODE_TILE_CHECK,
    CFG_WEIGHTS_BLOCKS,
    COLS_TILE,
    COMPUTES_PER_GROUP,
    CONFIG_LEN,
    GROUPS,
    INPUT_ELEMS,
    INPUT_GROUP_ELEMS,
    INPUT_GROUP_ROWS,
    INPUTS_VECS,
    L1_STREAM_BYTES,
    MAX_BANDS_PER_CHUNK,
    MAX_ROWS_STEPS,
    OUTPUT_ELEMS,
    ROWS_STEP,
    TILES_PER_BAND,
)


def _validate_shape(rows_steps, cols_bands, tile_bytes=None):
    if rows_steps <= 0 or cols_bands <= 0:
        raise ValueError("rows_steps and cols_bands must be positive")
    if rows_steps > MAX_ROWS_STEPS:
        raise ValueError("rows_steps exceeds the design maximum")
    if cols_bands > MAX_BANDS_PER_CHUNK:
        raise ValueError("cols_bands exceeds the chunk maximum")
    if tile_bytes is not None and (
        tile_bytes <= 0 or tile_bytes > L1_STREAM_BYTES
    ):
        raise ValueError("tile_bytes must fit one L1 block")


def _group_count(rows_steps):
    return (rows_steps + INPUT_GROUP_ROWS - 1) // INPUT_GROUP_ROWS


def _blocks_for(rows_steps, cols_bands, tile_bytes):
    return (
        rows_steps * cols_bands * tile_bytes + L1_STREAM_BYTES - 1
    ) // L1_STREAM_BYTES


def _bf16_bits(value):
    return int(np.array([value], dtype=bfloat16).view(np.uint16)[0])


def _softcap_words(soft_cap):
    """Split cap and 1/cap into bf16 hi/lo pairs, packed two per word."""
    if not soft_cap:
        return 0, 0
    cap = np.float32(soft_cap)
    cap_hi = bfloat16(cap)
    cap_lo = bfloat16(cap - np.float32(cap_hi))
    inv = np.float32(1.0) / cap
    inv_hi = bfloat16(inv)
    inv_lo = bfloat16(inv - np.float32(inv_hi))
    cap_word = _bf16_bits(cap_hi) | (_bf16_bits(cap_lo) << 16)
    inv_word = _bf16_bits(inv_hi) | (_bf16_bits(inv_lo) << 16)
    return cap_word, inv_word


def _make_config(
    format_id, tile_bytes, rows_steps, cols_bands, weights_blocks, total_tiles,
    soft_cap=0.0,
):
    config = np.zeros(CONFIG_LEN, dtype=np.int32)
    config[CFG_FORMAT] = format_id
    config[CFG_TILE_BYTES] = tile_bytes
    config[CFG_ROWS_STEPS] = rows_steps
    config[CFG_COLS_BANDS] = cols_bands
    config[CFG_WEIGHTS_BLOCKS] = weights_blocks
    config[CFG_TOTAL_TILES] = total_tiles
    cap_word, inv_word = _softcap_words(soft_cap)
    config[CFG_SOFT_CAP_HI_LO] = np.array(
        [cap_word], dtype=np.uint32
    ).view(np.int32)[0]
    config[CFG_SOFT_INV_HI_LO] = np.array(
        [inv_word], dtype=np.uint32
    ).view(np.int32)[0]
    return config


def _pack_row(inputs, row):
    row_data = inputs[:, row * ROWS_STEP : (row + 1) * ROWS_STEP]
    packed = np.concatenate(
        [
            row_data[:, t * 8 : (t + 1) * 8].reshape(-1)
            for t in range(ROWS_STEP // 8)
        ]
    )
    return packed.reshape(-1).view(np.uint16)


def _make_objects(inputs, rows_steps, cols_bands):
    """Pack one input object per row group.

    Each object holds ``INPUT_GROUP_ROWS`` row slices.  The config object is
    filled separately by the runtime and is not part of this buffer.  Missing
    rows in the final group are left zero.
    """
    groups = _group_count(rows_steps)
    objects = np.zeros((groups, INPUT_GROUP_ELEMS), dtype=np.uint16)
    row_sums = np.zeros(rows_steps, dtype=np.uint64)
    for row in range(rows_steps):
        packed = _pack_row(inputs, row)
        group = row // INPUT_GROUP_ROWS
        within = row % INPUT_GROUP_ROWS
        objects[group, within * INPUT_ELEMS : (within + 1) * INPUT_ELEMS] = packed
        row_sums[row] = (
            packed.astype(np.uint64).sum() & np.uint64(0xFFFFFFFF)
        )
    return objects, row_sums


def _block_major(streams, weights_blocks):
    return np.ascontiguousarray(
        streams.reshape(
            GROUPS,
            COMPUTES_PER_GROUP,
            weights_blocks,
            L1_STREAM_BYTES,
        ).transpose(0, 2, 1, 3)
    ).reshape(-1)


def _tile_words(group, compute, tile_index, tile_bytes, seed):
    count = tile_bytes // 4
    offsets = np.arange(count, dtype=np.uint64)
    base = np.uint64(seed + group * 131 + compute * 17 + tile_index * 1009)
    words = (offsets * np.uint64(2654435761) + base) & np.uint64(0xFFFFFFFF)
    return words.astype(np.uint32)


def build_case(rows_steps, cols_bands, tile_bytes, seed=1,
               format_id=MODE_TILE_CHECK):
    """Build a synthetic no-op case.

    Layouts:
      weights: [group][block][compute][L1]
      inputs:  [config][tile][INPUT_ELEMS] uint16
      expected: [group][compute][tile] uint32 checksum
    """
    _validate_shape(rows_steps, cols_bands, tile_bytes)

    total_tiles = rows_steps * cols_bands
    weights_blocks = _blocks_for(rows_steps, cols_bands, tile_bytes)
    # Diagnostic checksums are 32-bit values written into the bf16 output
    # object, so the tile count must stay within OUTPUT_ELEMS / 2 words.
    if total_tiles * 2 > OUTPUT_ELEMS:
        raise ValueError("diagnostic case exceeds the output checksum space")
    config = _make_config(
        format_id, tile_bytes, rows_steps, cols_bands, weights_blocks,
        total_tiles
    )

    input_offsets = np.arange(
        rows_steps * ROWS_STEP * INPUTS_VECS, dtype=np.uint64
    )
    input_bits = (
        input_offsets * np.uint64(2246822519) + np.uint64(seed)
    ) & np.uint64(0xFFFF)
    input_bits = input_bits.astype(np.uint16).reshape(
        INPUTS_VECS, rows_steps * ROWS_STEP
    )
    objects, input_sums = _make_objects(input_bits, rows_steps, cols_bands)

    streams = np.zeros(
        (GROUPS, COMPUTES_PER_GROUP, weights_blocks * L1_STREAM_BYTES),
        dtype=np.uint8,
    )
    expected = np.zeros(
        (GROUPS, COMPUTES_PER_GROUP, total_tiles), dtype=np.uint32
    )
    for group in range(GROUPS):
        for compute in range(COMPUTES_PER_GROUP):
            for tile_index in range(total_tiles):
                words = _tile_words(
                    group, compute, tile_index, tile_bytes, seed
                )
                row = tile_index // cols_bands
                band = tile_index % cols_bands
                start = (row * cols_bands + band) * tile_bytes
                streams[group, compute, start : start + tile_bytes] = (
                    words.view(np.uint8)
                )
                tile_sum = (
                    words.astype(np.uint64).sum() & np.uint64(0xFFFFFFFF)
                )
                expected[group, compute, tile_index] = np.uint32(
                    (
                        tile_sum
                        + input_sums[tile_index // cols_bands]
                    )
                    & np.uint64(0xFFFFFFFF)
                )

    return {
        "weights": _block_major(streams, weights_blocks),
        "inputs": objects,
        "config": config,
        "expected": expected,
        "rows_steps": rows_steps,
        "cols_bands": cols_bands,
        "tile_bytes": tile_bytes,
        "weights_blocks": weights_blocks,
        "total_tiles": total_tiles,
    }


def _build_quants_case(
    rows_steps,
    cols_bands,
    seed,
    format_name,
    format_id,
    weights_cols_real=None,
    inputs_matrix=None,
    soft_cap=0.0,
):
    """Build a real quantized case and its NumPy reference.

    ``weights_cols_real`` below ``cols_bands * BAND_COLS`` zero-pads the
    canonical tensor and lets the reference cover the padded columns.  Weights
    are packed block-major for the fill: [group][block][compute][L1].
    """
    _validate_shape(rows_steps, cols_bands)

    quants_tests = Path(__file__).resolve().parents[1] / "quants" / "tests"
    if str(quants_tests) not in sys.path:
        sys.path.insert(0, str(quants_tests))
    import common

    weights_rows = rows_steps * ROWS_STEP
    weights_cols = cols_bands * BAND_COLS
    real_cols = weights_cols if weights_cols_real is None else weights_cols_real
    constants = common.layout_constants(format_name)
    tile_cols = constants["kTileCols"]
    if real_cols <= 0 or real_cols > weights_cols or real_cols % tile_cols:
        raise ValueError("weights_cols_real must align with the padded width")

    fmt = common.load_format(format_name)
    rng = np.random.default_rng(seed)
    raw_real = fmt.make_raw(weights_rows, real_cols, rng)
    blocks_per_col = weights_rows // constants["kBlockRows"]
    raw_stride = blocks_per_col * constants["kBlockBytes"]
    if real_cols == weights_cols:
        raw = raw_real
    else:
        raw = np.zeros(weights_cols * raw_stride, dtype=np.uint8)
        real_bytes = real_cols * raw_stride
        raw[:real_bytes] = raw_real
    packed = common.repack_host(
        format_name, raw, weights_rows, weights_cols, raw_stride
    )
    expected_weights = fmt.reference_weights(raw, weights_rows, weights_cols)

    tile_bytes = constants["kPackedTransportTileBytes"]
    total_tiles = rows_steps * cols_bands
    weights_blocks = _blocks_for(rows_steps, cols_bands, tile_bytes)
    config = _make_config(
        format_id,
        tile_bytes,
        rows_steps,
        cols_bands,
        weights_blocks,
        total_tiles,
        soft_cap=soft_cap,
    )

    if inputs_matrix is None:
        inputs_rng = np.random.default_rng(seed + 1000)
        inputs = (
            inputs_rng.integers(-4, 5, (INPUTS_VECS, weights_rows))
            .astype(np.float32)
            .astype(bfloat16)
        )
    else:
        inputs = inputs_matrix
    objects, _ = _make_objects(inputs, rows_steps, cols_bands)

    streams = np.zeros(
        (GROUPS, COMPUTES_PER_GROUP, weights_blocks * L1_STREAM_BYTES),
        dtype=np.uint8,
    )
    cols_pairs = weights_cols // COLS_TILE
    for group in range(GROUPS):
        for compute in range(COMPUTES_PER_GROUP):
            for row in range(rows_steps):
                for band in range(cols_bands):
                    global_tile = (
                        row * cols_pairs
                        + band * TILES_PER_BAND
                        + group * COMPUTES_PER_GROUP
                        + compute
                    )
                    tile = packed[
                        global_tile * tile_bytes :
                        (global_tile + 1) * tile_bytes
                    ]
                    start = (row * cols_bands + band) * tile_bytes
                    streams[group, compute, start : start + tile_bytes] = tile

    cols_tiles = weights_cols // tile_cols
    weights_matrix = (
        expected_weights.reshape(cols_tiles, weights_rows, tile_cols)
        .transpose(1, 0, 2)
        .reshape(weights_rows, weights_cols)
        .astype(np.float32)
    )
    expected_full = inputs.astype(np.float32) @ weights_matrix
    if soft_cap:
        expected_full = (
            np.float32(soft_cap)
            * np.tanh(expected_full / np.float32(soft_cap))
        )
    expected = np.zeros(
        (
            GROUPS,
            COMPUTES_PER_GROUP,
            cols_bands,
            INPUTS_VECS,
            COLS_TILE,
        ),
        dtype=np.float32,
    )
    for group in range(GROUPS):
        for compute in range(COMPUTES_PER_GROUP):
            for band in range(cols_bands):
                col_start = (
                    band * BAND_COLS
                    + (group * COMPUTES_PER_GROUP + compute) * COLS_TILE
                )
                expected[group, compute, band] = expected_full[
                    :, col_start : col_start + COLS_TILE
                ]

    return {
        "weights": _block_major(streams, weights_blocks),
        "inputs": objects,
        "config": config,
        "expected": expected,
        "rows_steps": rows_steps,
        "cols_bands": cols_bands,
        "tile_bytes": tile_bytes,
        "weights_blocks": weights_blocks,
        "total_tiles": total_tiles,
        "format": format_id,
        "weights_rows": weights_rows,
        "weights_cols": weights_cols,
        "weights_cols_real": real_cols,
        "raw": raw,
        "raw_stride": raw_stride,
        "inputs_matrix": inputs,
        "soft_cap": soft_cap,
    }


def build_q4_0_case(
    rows_steps, cols_bands, seed=1, weights_cols_real=None, soft_cap=0.0
):
    return _build_quants_case(
        rows_steps,
        cols_bands,
        seed,
        "q4_0",
        FORMAT_Q4_0,
        weights_cols_real=weights_cols_real,
        soft_cap=soft_cap,
    )


def build_q6_k_case(
    rows_steps, cols_bands, seed=1, weights_cols_real=None
):
    return _build_quants_case(
        rows_steps,
        cols_bands,
        seed,
        "q6_k",
        FORMAT_Q6_K,
        weights_cols_real=weights_cols_real,
    )


def build_q6_k_head_case(
    rows_steps, total_bands=512, chunk_bands=MAX_BANDS_PER_CHUNK, seed=1
):
    """Build one contiguous Q6_K head tensor, chunk by chunk.

    The full reference matrix is too large to materialize at once, so each
    column chunk is built with its own weights while sharing the input matrix.
    The chunk raws concatenate into one canonical tensor.
    """
    expected = None
    inputs = None
    raws = []
    first = None
    for chunk, start in enumerate(range(0, total_bands, chunk_bands)):
        bands = min(chunk_bands, total_bands - start)
        case = _build_quants_case(
            rows_steps,
            bands,
            seed + chunk,
            "q6_k",
            2,
            inputs_matrix=inputs,
        )
        if inputs is None:
            inputs = case["inputs_matrix"]
            first = case
            expected = np.zeros(
                (
                    GROUPS,
                    COMPUTES_PER_GROUP,
                    total_bands,
                    INPUTS_VECS,
                    COLS_TILE,
                ),
                dtype=np.float32,
            )
        expected[:, :, start : start + bands] = case["expected"]
        raws.append(case["raw"])

    total_tiles = rows_steps * total_bands
    tile_bytes = first["tile_bytes"]
    return {
        "format": FORMAT_Q6_K,
        "rows_steps": rows_steps,
        "cols_bands": total_bands,
        "tile_bytes": tile_bytes,
        "weights_blocks": _blocks_for(
            rows_steps, total_bands, tile_bytes
        ),
        "total_tiles": total_tiles,
        "weights_rows": rows_steps * ROWS_STEP,
        "weights_cols": total_bands * BAND_COLS,
        "raw": np.concatenate(raws),
        "raw_stride": first["raw_stride"],
        "inputs_matrix": inputs,
        "expected": expected,
    }
