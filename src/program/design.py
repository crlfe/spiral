"""Locked design parameters and buffer types for the Spiral NPU program.

The first target is the Gemma 4 E2B matmul set.  Everything here is fixed;
per-call values (format, tile bytes, rows_steps, cols_bands, weights_blocks)
travel in the config object.
"""

import numpy as np
from ml_dtypes import bfloat16

# Fixed network geometry.
GROUPS = 8
COMPUTES_PER_GROUP = 4
INPUTS_VECS = 4

ROWS_STEP = 256
COLS_TILE = 16
BAND_COLS = GROUPS * COMPUTES_PER_GROUP * COLS_TILE
TILES_PER_BAND = BAND_COLS // COLS_TILE

# Buffer geometry.
L1_STREAM_BYTES = 8192
L2_STREAM_BYTES = COMPUTES_PER_GROUP * L1_STREAM_BYTES
CARRY_BYTES = 4096

# Locked resource limits.
MAX_WEIGHT_BLOCKS = 128
MAX_BANDS_PER_CHUNK = 48
MAX_ROWS_STEPS = 48
MAX_TILES_PER_BLOCK = 5
MAX_INPUT_OBJECTS = 256

INPUT_GROUP_ROWS = 4
INPUT_ELEMS = ROWS_STEP * INPUTS_VECS
INPUT_GROUP_ELEMS = INPUT_GROUP_ROWS * INPUT_ELEMS
OUTPUT_TILE_ELEMS = INPUTS_VECS * COLS_TILE
OUTPUT_ELEMS = MAX_BANDS_PER_CHUNK * OUTPUT_TILE_ELEMS
GROUP_OUTPUTS = COMPUTES_PER_GROUP * OUTPUT_ELEMS
CONFIG_LEN = 8
STATE_WORDS = 8

# Config word indices.
CFG_FORMAT = 0
CFG_TILE_BYTES = 1
CFG_ROWS_STEPS = 2
CFG_COLS_BANDS = 3
CFG_WEIGHTS_BLOCKS = 4
CFG_TOTAL_TILES = 5
CFG_INPUT_PAD = 6
CFG_WEIGHT_PAD = 7

# State word indices.
STATE_CARRY_LEN = 0
STATE_BLOCK_POS = 1
STATE_BLOCK_DONE = 2
STATE_TILE_INDEX = 3
STATE_READY = 4
STATE_TILE_SOURCE = 5  # 0 = carry, 1 = current L1 block
STATE_TILE_OFFSET = 6
STATE_CURRENT_ROW = 7

l1_stream_type = np.ndarray[(L1_STREAM_BYTES,), np.dtype[np.uint8]]
l2_stream_type = np.ndarray[(L2_STREAM_BYTES,), np.dtype[np.uint8]]
l1_carry_type = np.ndarray[(CARRY_BYTES,), np.dtype[np.uint8]]
l1_state_type = np.ndarray[(STATE_WORDS,), np.dtype[np.int32]]
l1_inputs_type = np.ndarray[(INPUT_ELEMS,), np.dtype[bfloat16]]
l1_input_group_type = np.ndarray[
    (INPUT_GROUP_ELEMS,), np.dtype[bfloat16]
]
l1_outputs_type = np.ndarray[(OUTPUT_ELEMS,), np.dtype[np.float32]]
l2_outputs_type = np.ndarray[
    (GROUP_OUTPUTS,), np.dtype[np.float32]
]
l1_config_out_type = np.ndarray[(CONFIG_LEN,), np.dtype[np.int32]]

# Host buffers are 32-bit words for shim DMA compatibility.
MAX_WEIGHT_WORDS = (
    GROUPS * MAX_WEIGHT_BLOCKS * L2_STREAM_BYTES // 4
)
MAX_INPUT_OBJECTS //= INPUT_GROUP_ROWS
MAX_INPUT_WORDS = (1 + MAX_INPUT_OBJECTS) * INPUT_GROUP_ELEMS // 2
MAX_OUTPUT_ELEMS = GROUPS * GROUP_OUTPUTS

l3_weights_type = np.ndarray[
    (MAX_WEIGHT_WORDS,), np.dtype[np.uint32]
]
l3_inputs_type = np.ndarray[
    (MAX_INPUT_WORDS,), np.dtype[np.uint32]
]
l3_outputs_type = np.ndarray[
    (MAX_OUTPUT_ELEMS,), np.dtype[np.float32]
]
