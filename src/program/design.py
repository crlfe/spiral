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
# Static test path: fixed host buffer for input-group objects.
MAX_INPUT_GROUPS = 64

INPUT_GROUP_ROWS = 4
INPUT_ELEMS = ROWS_STEP * INPUTS_VECS
INPUT_GROUP_ELEMS = INPUT_GROUP_ROWS * INPUT_ELEMS
OUTPUT_TILE_ELEMS = INPUTS_VECS * COLS_TILE
OUTPUT_ELEMS = MAX_BANDS_PER_CHUNK * OUTPUT_TILE_ELEMS
GROUP_OUTPUTS = COMPUTES_PER_GROUP * OUTPUT_ELEMS
STATE_WORDS = 8

# Format ids follow ggml_type for quant formats.  Values 0xF0-0xFF are
# diagnostic-only modes and never reach a production build.
FORMAT_Q4_0 = 2
FORMAT_Q6_K = 14
MODE_BASE = 0xF0
MODE_FIFO = 0xF0       # acquire/release only
MODE_RAW_CHECK = 0xF1  # per-block checksum of delivered stream bytes
MODE_TILE_CHECK = 0xF2 # assemble/carry, then per-tile checksum
MODE_LAST = 0xFF

# Config word indices.
CFG_FORMAT = 0
CFG_TILE_BYTES = 1
CFG_ROWS_STEPS = 2
CFG_COLS_BANDS = 3
CFG_WEIGHTS_BLOCKS = 4
CFG_TOTAL_TILES = 5
CFG_INPUT_PAD = 6
CFG_WEIGHT_PAD = 7
CFG_SOFT_CAP_HI_LO = 8    # two bf16 bits: cap hi, cap lo
CFG_SOFT_INV_HI_LO = 9    # two bf16 bits: 1/cap hi, 1/cap lo
CFG_EPILOGUE = 10         # EPILOGUE_* selector
CFG_PRENORM = 11          # nonzero: this call carries the prenorm body
CFG_PRENORM_EPS = 12      # f32 bits
CFG_PRENORM_INV_K = 13    # f32 bits


# Tile epilogue selector.  The value is a single exclusive kind; a separate
# flags word will be added when layout-preserving modifiers (scale/add/clamp)
# gain their operands.
EPILOGUE_NONE = 0
EPILOGUE_SOFTCAP = 1
EPILOGUE_GEGLU = 2

# Config words are copied as int32; round the object up to a 64-byte line.
CONFIG_LEN = 16
# The norm weight starts after the config header; it is interleaved per
# 8-value K chunk as [hi8][lo8] bf16.
PRENORM_WEIGHT_OFFSET_BYTES = CONFIG_LEN * 4

# State word indices.
STATE_CARRY_LEN = 0
STATE_BLOCK_POS = 1
STATE_BLOCK_DONE = 2
STATE_TILE_INDEX = 3
STATE_READY = 4
STATE_TILE_SOURCE = 5  # 0 = carry, 1 = current L1 block
STATE_TILE_OFFSET = 6
# Word 7 is reserved; STATE_WORDS stays 8 for a 32-byte state object.

l1_stream_type = np.ndarray[(L1_STREAM_BYTES,), np.dtype[np.uint8]]
l2_stream_type = np.ndarray[(L2_STREAM_BYTES,), np.dtype[np.uint8]]
l1_carry_type = np.ndarray[(CARRY_BYTES,), np.dtype[np.uint8]]
l1_state_type = np.ndarray[(STATE_WORDS,), np.dtype[np.int32]]
# Norm prelude state: four f32 sums followed by four hi/lo bf16 pairs.
l1_prenorm_state_type = np.ndarray[(16,), np.dtype[np.int32]]
# Stable copy of the config-carried norm weight; the config object itself is a
# stream object the input DMA reuses, so the scale pass cannot read it in place.
l1_prenorm_weight_type = np.ndarray[
    (INPUT_GROUP_ELEMS,), np.dtype[bfloat16]
]
l1_input_group_type = np.ndarray[
    (INPUT_GROUP_ELEMS,), np.dtype[bfloat16]
]
l1_accum_type = np.ndarray[(OUTPUT_ELEMS,), np.dtype[np.float32]]
l1_outputs_type = np.ndarray[(OUTPUT_ELEMS,), np.dtype[bfloat16]]
l2_outputs_type = np.ndarray[
    (GROUP_OUTPUTS,), np.dtype[bfloat16]
]
l1_config_out_type = np.ndarray[(CONFIG_LEN,), np.dtype[np.int32]]

# Host buffers are 32-bit words for shim DMA compatibility.
MAX_WEIGHT_WORDS = (
    GROUPS * MAX_WEIGHT_BLOCKS * L2_STREAM_BYTES // 4
)
MAX_INPUT_WORDS = MAX_INPUT_GROUPS * INPUT_GROUP_ELEMS // 2
MAX_OUTPUT_ELEMS = GROUPS * GROUP_OUTPUTS

l3_weights_type = np.ndarray[
    (MAX_WEIGHT_WORDS,), np.dtype[np.uint32]
]
l3_inputs_type = np.ndarray[
    (MAX_INPUT_WORDS,), np.dtype[np.uint32]
]
l3_config_type = np.ndarray[
    (INPUT_GROUP_ELEMS // 2,), np.dtype[np.uint32]
]
l3_outputs_type = np.ndarray[
    (MAX_OUTPUT_ELEMS,), np.dtype[bfloat16]
]
