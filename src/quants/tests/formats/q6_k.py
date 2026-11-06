"""Test-only numpy reference for canonical GGUF Q6_K weights."""

import numpy as np
from ml_dtypes import bfloat16

BLOCK_ROWS = 256
BLOCK_BYTES = 210
QL_BYTES = 128
QH_BYTES = 64
SCALES_BYTES = 16
TILE_COLS = 8
UNPACK_TILE_ROWS = 64
RECORD_ROWS = 256
CHUNKS_PER_RECORD = 4

DEFAULT_ROWS = 256
DEFAULT_COLS = 64


def make_raw(weights_rows, weights_cols, rng):
    """Return canonical column-major Q6_K tensor bytes."""
    if weights_rows % BLOCK_ROWS != 0:
        raise ValueError("weights_rows must be a multiple of 256")
    if weights_cols <= 0:
        raise ValueError("weights_cols must be positive")

    blocks_per_col = weights_rows // BLOCK_ROWS
    raw = np.empty(weights_cols * blocks_per_col * BLOCK_BYTES, dtype=np.uint8)
    offset = 0

    for _col in range(weights_cols):
        for _block in range(blocks_per_col):
            ql = rng.integers(0, 256, QL_BYTES, dtype=np.uint8)
            qh = rng.integers(0, 256, QH_BYTES, dtype=np.uint8)
            scales = (
                rng.integers(1, 33, SCALES_BYTES)
                * rng.choice(np.array([-1, 1], dtype=np.int8), SCALES_BYTES)
            ).astype(np.int8)
            d = np.array([rng.uniform(0.01, 0.1)], dtype=np.float16)
            raw[offset : offset + QL_BYTES] = ql
            raw[offset + QL_BYTES : offset + QL_BYTES + QH_BYTES] = qh
            raw[
                offset + QL_BYTES + QH_BYTES :
                offset + QL_BYTES + QH_BYTES + SCALES_BYTES
            ] = scales.view(np.uint8)
            raw[
                offset + QL_BYTES + QH_BYTES + SCALES_BYTES :
                offset + BLOCK_BYTES
            ] = d.view(np.uint8)
            offset += BLOCK_BYTES

    return raw


def _dequant_block(block):
    ql = block[0:QL_BYTES].astype(np.int32)
    qh = block[QL_BYTES : QL_BYTES + QH_BYTES].astype(np.int32)
    scales = block[
        QL_BYTES + QH_BYTES : QL_BYTES + QH_BYTES + SCALES_BYTES
    ].view(np.int8).astype(np.int32)
    d = np.frombuffer(
        block[BLOCK_BYTES - 2 : BLOCK_BYTES].tobytes(), dtype=np.float16
    )[0].astype(np.float32)

    y = np.empty(BLOCK_ROWS, dtype=np.float32)
    lengths = np.arange(32)
    is_scale = lengths // 16

    for half in range(2):
        ql_offset = half * 64
        qh_offset = half * 32
        scale_offset = half * 8
        y_offset = half * 128

        ql_low = ql[ql_offset : ql_offset + 32]
        ql_high = ql[ql_offset + 32 : ql_offset + 64]
        qh_part = qh[qh_offset : qh_offset + 32]

        q1 = ((ql_low & 0x0F) | (((qh_part >> 0) & 0x03) << 4)) - 32
        q2 = ((ql_high & 0x0F) | (((qh_part >> 2) & 0x03) << 4)) - 32
        q3 = ((ql_low >> 4) | (((qh_part >> 4) & 0x03) << 4)) - 32
        q4 = ((ql_high >> 4) | (((qh_part >> 6) & 0x03) << 4)) - 32

        y[y_offset + lengths] = (
            d * scales[scale_offset + is_scale] * q1
        )
        y[y_offset + lengths + 32] = (
            d * scales[scale_offset + is_scale + 2] * q2
        )
        y[y_offset + lengths + 64] = (
            d * scales[scale_offset + is_scale + 4] * q3
        )
        y[y_offset + lengths + 96] = (
            d * scales[scale_offset + is_scale + 6] * q4
        )

    return y


def _dequant_values(raw, weights_rows, weights_cols):
    blocks_per_col = weights_rows // BLOCK_ROWS
    raw_blocks = raw.reshape(weights_cols, blocks_per_col, BLOCK_BYTES)
    values = np.empty((weights_rows, weights_cols), dtype=np.float32)

    for weights_col in range(weights_cols):
        for weights_block in range(blocks_per_col):
            values[
                weights_block * BLOCK_ROWS : (weights_block + 1) * BLOCK_ROWS,
                weights_col,
            ] = _dequant_block(raw_blocks[weights_col, weights_block])

    return values


def reference_weights(raw, weights_rows, weights_cols):
    """Expected canonical bf16 weights stream.

    Output order is
    [weights_cols_tile][weights_rows_record][chunk][kUnpackTileRows][kTileCols].
    """
    values = _dequant_values(raw, weights_rows, weights_cols)
    cols_tiles = weights_cols // TILE_COLS
    rows_records = weights_rows // RECORD_ROWS
    ordered = (
        values.reshape(rows_records, RECORD_ROWS, cols_tiles, TILE_COLS)
        .transpose(2, 0, 1, 3)
        .reshape(-1)
    )
    return ordered.astype(bfloat16)


def verify(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return False, f"shape mismatch: {actual.shape} != {expected.shape}"

    error = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    expected_scale = np.abs(expected.astype(np.float32)).max()
    tolerance = max(0.02, 0.01 * float(expected_scale))
    if error.max() <= tolerance:
        return True, (
            f"within Q6_K tolerance (max abs error {error.max():.6f})"
        )
    return False, f"max abs error {error.max():.6f}"
