"""Test-only numpy reference for canonical GGUF Q4_0 weights."""

import numpy as np
from ml_dtypes import bfloat16

BLOCK_ROWS = 32
BLOCK_BYTES = 18
TILE_COLS = 8
UNPACK_TILE_ROWS = 64
RECORD_ROWS = 64
CHUNKS_PER_RECORD = 1

DEFAULT_ROWS = 256
DEFAULT_COLS = 64


def make_raw(weights_rows, weights_cols, rng):
    """Return canonical column-major Q4_0 tensor bytes."""
    if weights_rows % BLOCK_ROWS != 0:
        raise ValueError("weights_rows must be a multiple of 32")
    if weights_cols <= 0:
        raise ValueError("weights_cols must be positive")

    blocks_per_col = weights_rows // BLOCK_ROWS
    raw = np.empty(weights_cols * blocks_per_col * BLOCK_BYTES, dtype=np.uint8)
    offset = 0

    for _col in range(weights_cols):
        for _block in range(blocks_per_col):
            quant = rng.integers(0, 16, BLOCK_ROWS, dtype=np.uint8)
            # Mix normal and subnormal scales, including both signs. The first
            # block covers the smallest and largest subnormal magnitudes.
            edge_scale = (0x0001, 0x03FF, 0x8001, 0x83FF, 0x0200, 0x8200)
            if _block == 0 and _col < len(edge_scale):
                scale_bits = edge_scale[_col]
            else:
                if rng.random() < 0.5:
                    scale_bits = int(rng.integers(1, 0x0400))
                else:
                    scale_bits = int(
                        np.array([rng.uniform(0.01, 3.0)], dtype=np.float16)
                        .view(np.uint16)[0]
                    )
                if rng.random() < 0.5:
                    scale_bits |= 0x8000
            scale_bytes = np.array([scale_bits], dtype=np.uint16).view(np.uint8)
            raw[offset : offset + 2] = scale_bytes
            raw[offset + 2 : offset + BLOCK_BYTES] = (
                quant[:16] | (quant[16:] << 4)
            )
            offset += BLOCK_BYTES

    return raw


def _dequant_values(raw, weights_rows, weights_cols):
    blocks_per_col = weights_rows // BLOCK_ROWS
    raw_blocks = raw.reshape(weights_cols, blocks_per_col, BLOCK_BYTES)
    values = np.empty((weights_rows, weights_cols), dtype=np.float32)

    for weights_col in range(weights_cols):
        for weights_block in range(blocks_per_col):
            block = raw_blocks[weights_col, weights_block]
            scale = np.frombuffer(
                block[0:2].tobytes(), dtype=np.float16
            )[0].astype(np.float32)
            payload = block[2:BLOCK_BYTES]
            quant = np.empty(BLOCK_ROWS, dtype=np.int16)
            quant[:16] = payload & 0x0F
            quant[16:] = payload >> 4
            values[
                weights_block * BLOCK_ROWS : (weights_block + 1) * BLOCK_ROWS,
                weights_col,
            ] = (quant.astype(np.float32) - 8.0) * scale

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
    if np.array_equal(actual, expected):
        return True, "exact bf16 match"

    actual_f = actual.astype(np.float32)
    expected_f = expected.astype(np.float32)
    error = np.abs(actual_f - expected_f)
    if error.max() > 0.01:
        return False, f"max abs error {error.max():.6f}"

    # The absolute tolerance cannot see a flushed subnormal block, so nonzero
    # values must also match within a bf16-relative tolerance.
    nonzero = expected_f != 0
    if np.any(nonzero):
        relative = error[nonzero] / np.abs(expected_f[nonzero])
        if relative.max() > 0.02:
            return False, (
                f"max relative error {relative.max():.6f} "
                f"(max abs error {error.max():.6f})"
            )
        return True, (
            f"within bf16 tolerance (max abs error {error.max():.6f}, "
            f"max relative error {relative.max():.6f})"
        )
    return True, f"within bf16 tolerance (max abs error {error.max():.6f})"
