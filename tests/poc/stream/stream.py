"""Quantization-emulation byte-stream test for the 256x16 NPU tile.

The design keeps the NPU network fixed while weights arrive as an exact,
mode-specific byte stream over fixed-size DMA objects.  Complete emulated
weight tiles are multiplied directly out of the transport buffer.  When a tile
crosses a block boundary, only the missing prefix is copied into a carry
buffer, which is multiplied and then reused; bytes are never rearranged inside
a work buffer.

Quantization is emulated rather than implemented.  We always send bfloat16
weights, but truncate each tile to a number of 8-row groups that approximates
the real packed size:

    Q4_0-like  72 rows = 2304 bytes   (real Q4_0: 2304)
    Q6_K-like 104 rows = 3328 bytes   (real Q6_K: 3360)
    IQ1_S-like 25 rows =  800 bytes   (real IQ1_S: 800)

The kernel still multiplies all 256 rows.  It wraps back to the start of the
sent complete 64-weight vectors as needed; a trailing partial vector is
skipped.  The reference model simulates the same vector-level wrap.  This
exercises the stream and its bandwidth without depending on the quantization
kernels.

Geometry:

    one compute tile   256 rows x 16 cols
    one group          256 rows x 64 cols  (4 computes)
    whole array        256 rows x 512 cols (8 groups)

The host sends one continuous stream per compute tile.  The tile stride is the
emulated packed size, not a common maximum.  The stream is padded only at its
end to a whole L1 stream block; the receiver stops after the known number of
tiles and discards the tail.

TODO:
  * This file currently specializes the host sequence on tile_bytes, row_steps,
    col_steps, and vec_steps because the Python JIT path has no runtime scalar
    plumbing here.  The device network and kernels are already fixed; the next
    step is to generate the host transaction at runtime as invoke.py does, so
    changing shape or mode does not reload the xclbin.
  * Add the native C++ driver and pipelined submissions for multiple operations
    through one loaded xclbin.
  * Replace the emulation with the real Q4_0/Q6_K unpack+mmul kernels; the
    stream and configuration layout stay the same.
  * Replace the host-packed mmul input layout with the hardware tiling used by
    calc.py once correctness is established.
  * Handle weight fills with more than 64 L2 stream blocks by splitting the
    fill into multiple shim queue commands.
  * Add a benchmark mode that alternates modes and verifies no reconfiguration
    occurred.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aie import iron
from aie.dialects import arith
from aie.dialects._aie_enum_gen import AIETileType
from aie.ir import IntegerType
from aie.iron import (
    Buffer,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.utils.benchmark import run_iters
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# Design geometry and buffer types


@dataclass(frozen=True)
class StreamDesign:
    """Static geometry and buffer types for the 256x16 stream design.

    Build one of these from `main` (or a library caller) and pass it to
    `run_case`/`program`.  It is immutable, so it is safe to share.
    """

    groups: int = 8
    computes_per_group: int = 4
    rows_step: int = 256
    cols_step: int = 16
    vecs_per_step: int = 4

    # Compile-time bounds.  Actual counts are runtime values.
    max_row_steps: int = 8      # rows up to 2048
    max_col_steps: int = 4      # cols up to 2048
    max_vec_steps: int = 1      # vecs up to 4 for now
    max_tile_bytes: int = 5120

    l1_stream_bytes: int = 2048
    work_bytes: int = 5120

    config_len: int = 8
    state_words: int = 4

    def __post_init__(self):
        if self.l1_stream_bytes < 1024 or self.l1_stream_bytes % 1024:
            raise ValueError("l1_stream_bytes must be a multiple of 1024")
        if self.work_bytes < self.max_tile_bytes:
            raise ValueError("work_bytes must hold the largest tile")
        if self.l1_stream_bytes > 1023 * 1024:
            raise ValueError("l1_stream_bytes exceeds the shim BD limit")

    @property
    def band_cols(self):
        return self.groups * self.computes_per_group * self.cols_step

    @property
    def mmul_outputs(self):
        return 2 * self.vecs_per_step * 8

    @property
    def compute_outputs(self):
        return self.max_col_steps * self.max_vec_steps * self.mmul_outputs

    @property
    def group_outputs(self):
        return self.computes_per_group * self.compute_outputs

    @property
    def max_output_elems(self):
        return self.max_col_steps * self.groups * self.group_outputs

    @property
    def max_input_elems(self):
        return (
            self.max_row_steps
            * self.max_vec_steps
            * self.rows_step
            * self.vecs_per_step
        )

    @property
    def l2_stream_bytes(self):
        return self.computes_per_group * self.l1_stream_bytes

    @property
    def max_l2_blocks(self):
        total = (
            self.max_row_steps
            * self.max_col_steps
            * self.max_tile_bytes
        )
        return (total + self.l1_stream_bytes - 1) // self.l1_stream_bytes

    @property
    def max_stream_bytes(self):
        return self.max_l2_blocks * self.l1_stream_bytes

    @property
    def max_l3_weights_bytes(self):
        return (
            self.groups
            * self.computes_per_group
            * self.max_stream_bytes
        )

    @property
    def l1_stream_type(self):
        return np.ndarray[(self.l1_stream_bytes,), np.dtype[np.uint8]]

    @property
    def l2_stream_type(self):
        return np.ndarray[(self.l2_stream_bytes,), np.dtype[np.uint8]]

    @property
    def l1_work_type(self):
        return np.ndarray[(self.work_bytes,), np.dtype[np.uint8]]

    @property
    def l1_state_type(self):
        return np.ndarray[(self.state_words,), np.dtype[np.int32]]

    @property
    def l1_inputs_type(self):
        return np.ndarray[(self.max_input_elems,), np.dtype[bfloat16]]

    @property
    def l1_outputs_type(self):
        return np.ndarray[(self.compute_outputs,), np.dtype[np.float32]]

    @property
    def l2_outputs_type(self):
        return np.ndarray[(self.group_outputs,), np.dtype[np.float32]]

    @property
    def l1_config_out_type(self):
        return np.ndarray[(self.config_len,), np.dtype[np.int32]]

    # Host buffers are viewed as 32-bit words.  The shim DMA requires the
    # innermost stride to be a multiple of four bytes, which rules out byte
    # or bfloat16 strides on the packed streams even though the device FIFOs
    # still carry uint8 / bfloat16 objects.
    @property
    def l3_weights_type(self):
        return np.ndarray[
            (self.max_l3_weights_bytes // 4,), np.dtype[np.uint32]
        ]

    @property
    def l3_inputs_type(self):
        return np.ndarray[
            (self.max_input_elems // 2,), np.dtype[np.uint32]
        ]

    @property
    def l3_outputs_type(self):
        return np.ndarray[(self.max_output_elems,), np.dtype[np.float32]]

    @property
    def config_type(self):
        return np.ndarray[
            (self.max_input_elems // 2,), np.dtype[np.uint32]
        ]


DEFAULT_DESIGN = StreamDesign()

# Benchmark-only parameters.
PEAK_GBS = 57.0        # measured bulk host-to-NPU bandwidth

# Emulated packed sizes: each weight row of 16 columns is 32 bytes.
EMU_Q4_0_ROWS = 72    # 2304 bytes, matching real Q4_0
EMU_Q6_K_ROWS = 104   # 3328 bytes, close to real Q6_K's 3360
EMU_IQ1_S_ROWS = 25   # 800 bytes, matching real IQ1_S


# Kernel sources live beside this file.  Listing them in @iron.jit's
# source_files makes their content part of the JIT cache key, so editing a
# kernel invalidates the cached design automatically.
KERNEL_DIR = Path(__file__).resolve().parent
STREAM_COMMON_SOURCE = KERNEL_DIR / "stream_common.h"
STREAM_INIT_SOURCE = KERNEL_DIR / "stream_init.aie.cc"
STREAM_CONFIG_SOURCE = KERNEL_DIR / "stream_config_decode.aie.cc"
STREAM_ZERO_SOURCE = KERNEL_DIR / "stream_zero.aie.cc"
STREAM_FUSED_SOURCE = KERNEL_DIR / "stream_fused.aie.cc"
KERNEL_SOURCES = [
    STREAM_COMMON_SOURCE,
    STREAM_INIT_SOURCE,
    STREAM_CONFIG_SOURCE,
    STREAM_ZERO_SOURCE,
    STREAM_FUSED_SOURCE,
]
KERNEL_INCLUDE_DIRS = [str(KERNEL_DIR)]


def _l1_stream_bytes(value):
    number = int(value)
    if number < 1024 or number % 1024:
        raise argparse.ArgumentTypeError("must be a multiple of 1024")
    if number > 1023 * 1024:
        raise argparse.ArgumentTypeError("must not exceed 1023 KiB")
    return number


def _at_least(minimum):
    def parse(value):
        number = int(value)
        if number < minimum:
            raise argparse.ArgumentTypeError(
                f"must be at least {minimum}"
            )
        return number

    return parse


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Emulated quantized weight-stream benchmark for the 256x16 NPU "
            "tile."
        )
    )
    parser.add_argument(
        "--l1-stream-bytes",
        type=_l1_stream_bytes,
        default=DEFAULT_DESIGN.l1_stream_bytes,
        metavar="N",
        help=(
            "L1 weight stream block size in bytes, a multiple of 1024 up "
            f"to 1023 KiB (default: {DEFAULT_DESIGN.l1_stream_bytes})"
        ),
    )
    parser.add_argument(
        "--work-bytes",
        type=_at_least(DEFAULT_DESIGN.max_tile_bytes),
        default=DEFAULT_DESIGN.work_bytes,
        metavar="N",
        help=(
            "per-compute work buffer size in bytes; must hold the largest "
            f"tile (default: {DEFAULT_DESIGN.work_bytes})"
        ),
    )
    parser.add_argument(
        "--l2-depth",
        type=_at_least(1),
        default=2,
        metavar="N",
        help="L2 weight FIFO depth (default: 2)",
    )
    parser.add_argument(
        "--l1-depth",
        type=_at_least(1),
        default=2,
        metavar="N",
        help="L1 weight FIFO depth (default: 2)",
    )
    parser.add_argument(
        "--reps",
        type=_at_least(1),
        default=1,
        metavar="N",
        help="repetitions per launch (default: 1)",
    )
    parser.add_argument(
        "--weight-tile-bytes",
        type=int,
        default=None,
        metavar="N",
        help="override the emulated packed tile size",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="skip the carry copy (verification is skipped)",
    )
    parser.add_argument(
        "--skip-mmul",
        action="store_true",
        help="skip the mmul call (verification is skipped)",
    )
    return parser


def _make_stream_init(design):
    return ExternalFunction(
        "stream_init",
        arg_types=[design.l1_state_type],
        inline=False,
        source_file=str(STREAM_INIT_SOURCE),
    )


def _make_config_decode(design):
    return ExternalFunction(
        "stream_config_decode",
        arg_types=[design.l1_inputs_type, design.l1_config_out_type],
        inline=False,
        source_file=str(STREAM_CONFIG_SOURCE),
        include_dirs=KERNEL_INCLUDE_DIRS,
    )


def _make_zero(design):
    return ExternalFunction(
        "stream_zero",
        arg_types=[design.l1_outputs_type, np.int32],
        inline=False,
        source_file=str(STREAM_ZERO_SOURCE),
        include_dirs=KERNEL_INCLUDE_DIRS,
    )


def _make_stream_fused(design, skip_copy, skip_mmul):
    return ExternalFunction(
        "stream_fused",
        arg_types=[
            design.l1_stream_type,
            design.l1_work_type,
            design.l1_state_type,
            design.l1_inputs_type,
            design.l1_outputs_type,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
        compile_flags=[
            f"-DSTREAM_BLOCK_BYTES={design.l1_stream_bytes}",
            f"-DSTREAM_INPUT_ELEMS={design.rows_step * design.vecs_per_step}",
            *(["-DSTREAM_SKIP_COPY=1"] if skip_copy else []),
            *(["-DSTREAM_SKIP_MMUL=1"] if skip_mmul else []),
        ],
        inline=False,
        source_file=str(STREAM_FUSED_SOURCE),
        include_dirs=KERNEL_INCLUDE_DIRS,
    )


def _make_fifos(design, mem_tiles, l2_depth=2, l1_depth=2):
    # One broadcast stream for the config object and the mmul input steps.
    inputs_l2 = ObjectFifo(design.l1_inputs_type, name="inputs_l2", depth=2)
    inputs_l1 = inputs_l2.cons().forward(
        tile=mem_tiles[0],
        obj_type=design.l1_inputs_type,
        name="inputs_l1",
    )

    # Weights are split four ways per group.
    weights_l2 = [
        ObjectFifo(
            design.l2_stream_type, name=f"weights_l2_{g}", depth=l2_depth
        )
        for g in range(design.groups)
    ]
    weights_l1 = [
        weights_l2[g]
        .cons()
        .split(
            offsets=[
                c * design.l1_stream_bytes
                for c in range(design.computes_per_group)
            ],
            tile=mem_tiles[g],
            obj_types=[design.l1_stream_type] * design.computes_per_group,
            depths=[l1_depth] * design.computes_per_group,
            names=[
                f"weights_l1_{g}_{c}" for c in range(design.computes_per_group)
            ],
        )
        for g in range(design.groups)
    ]

    # Outputs are joined four ways per group.
    outputs_l2 = [
        ObjectFifo(design.l2_outputs_type, name=f"outputs_l2_{g}", depth=1)
        for g in range(design.groups)
    ]
    outputs_l1 = [
        outputs_l2[g]
        .prod()
        .join(
            offsets=[
                c * design.compute_outputs
                for c in range(design.computes_per_group)
            ],
            tile=mem_tiles[g],
            obj_types=[design.l1_outputs_type] * design.computes_per_group,
            names=[
                f"outputs_l1_{g}_{c}" for c in range(design.computes_per_group)
            ],
        )
        for g in range(design.groups)
    ]

    return (
        inputs_l2,
        inputs_l1,
        weights_l2,
        weights_l1,
        outputs_l2,
        outputs_l1,
    )


def compute_body(
    weights_cons,
    inputs_cons,
    outputs_prod,
    config_out,
    carry,
    state,
    stream_init,
    config_decode,
    zero,
    fused,
    reps,
):
    for _rep in range_(reps):
        config_buf = inputs_cons.acquire(1)
        config_decode(config_buf, config_out)
        inputs_cons.release(1)

        # Words: 0 tile_bytes, 1 tile_vectors, 2 row_steps, 3 col_steps,
        # 4 vec_steps, 6 weights_blocks (5 is unused).
        tile_bytes = config_out[0]
        tile_vectors = config_out[1]
        row_steps = config_out[2]
        col_steps = config_out[3]
        vec_steps = config_out[4]
        weights_blocks = config_out[6]

        inputs_buf = inputs_cons.acquire(1)
        stream_init(state)

        outputs_buf = outputs_prod.acquire(1)
        zero(outputs_buf, arith.muli(col_steps, vec_steps))

        for _block in range_(weights_blocks):
            l1_block = weights_cons.acquire(1)
            fused(
                l1_block,
                carry,
                state,
                inputs_buf,
                outputs_buf,
                tile_bytes,
                tile_vectors,
                row_steps,
                col_steps,
                vec_steps,
            )
            weights_cons.release(1)

        outputs_prod.release(1)
        inputs_cons.release(1)


# The design is a compile-time parameter, so its geometry is part of the JIT
# cache key and a changed size cannot reuse a stale xclbin.
@iron.jit(source_files=KERNEL_SOURCES)
def program(
    weights: In,
    inputs: In,
    outputs: Out,
    config: In,
    *,
    tile_bytes: iron.CompileTime[np.int32] = None,
    row_steps: iron.CompileTime[np.int32] = None,
    col_steps: iron.CompileTime[np.int32] = None,
    vec_steps: iron.CompileTime[np.int32] = None,
    design: iron.CompileTime[object] = DEFAULT_DESIGN,
    skip_copy: iron.CompileTime[np.int32] = 0,
    skip_mmul: iron.CompileTime[np.int32] = 0,
    reps: iron.CompileTime[np.int32] = 1,
    weight_l2_depth: iron.CompileTime[np.int32] = 2,
    weight_l1_depth: iron.CompileTime[np.int32] = 2,
):
    shim_tiles = [
        Tile(col=g, row=0, tile_type=AIETileType.ShimNOCTile)
        for g in range(design.groups)
    ]
    mem_tiles = [
        Tile(col=g, row=1, tile_type=AIETileType.MemTile)
        for g in range(design.groups)
    ]
    compute_tiles = [
        [
            Tile(col=g, row=2 + c, tile_type=AIETileType.CoreTile)
            for c in range(design.computes_per_group)
        ]
        for g in range(design.groups)
    ]

    (
        inputs_l2,
        inputs_l1,
        weights_l2,
        weights_l1,
        outputs_l2,
        outputs_l1,
    ) = _make_fifos(
        design,
        mem_tiles,
        l2_depth=weight_l2_depth,
        l1_depth=weight_l1_depth,
    )

    stream_init = _make_stream_init(design)
    config_decode = _make_config_decode(design)
    zero = _make_zero(design)
    fused = _make_stream_fused(design, skip_copy, skip_mmul)

    work_buffers = [
        Buffer(design.l1_work_type, name=f"work_{g}_{c}")
        for g in range(design.groups)
        for c in range(design.computes_per_group)
    ]
    state_buffers = [
        Buffer(design.l1_state_type, name=f"state_{g}_{c}")
        for g in range(design.groups)
        for c in range(design.computes_per_group)
    ]
    config_buffers = [
        Buffer(design.l1_config_out_type, name=f"config_{g}_{c}")
        for g in range(design.groups)
        for c in range(design.computes_per_group)
    ]

    workers = [
        Worker(
            compute_body,
            [
                weights_l1[g][c].cons(),
                inputs_l1.cons(),
                outputs_l1[g][c].prod(),
                config_buffers[g * design.computes_per_group + c],
                work_buffers[g * design.computes_per_group + c],
                state_buffers[g * design.computes_per_group + c],
                stream_init,
                config_decode,
                zero,
                fused,
                reps,
            ],
            tile=compute_tiles[g][c],
            stack_size=8192,
        )
        for g in range(design.groups)
        for c in range(design.computes_per_group)
    ]

    def sequence(
        weights_buf: iron.RuntimeData,
        inputs_buf: iron.RuntimeData,
        outputs_buf: iron.RuntimeData,
        config_buf: iron.RuntimeData,
        tile_bytes,
        row_steps,
        col_steps,
        vec_steps,
        weights_prods,
        inputs_prod,
        outputs_conses,
    ):
        i64 = IntegerType.get_signless(64)
        i32 = IntegerType.get_signless(32)

        def i32_const(value):
            return arith.constant(i32, value)

        def as_i32(value):
            return (
                arith.constant(i32, int(value))
                if isinstance(value, (int, np.integer))
                else value
            )

        tile_bytes = as_i32(tile_bytes)
        row_steps = as_i32(row_steps)
        col_steps = as_i32(col_steps)
        vec_steps = as_i32(vec_steps)

        tile_count = arith.muli(row_steps, col_steps)
        stream_bytes = arith.muli(tile_count, tile_bytes)
        stream_l2_blocks = arith.divui(
            arith.addi(stream_bytes, i32_const(design.l1_stream_bytes - 1)),
            i32_const(design.l1_stream_bytes),
        )
        stream_stride = arith.muli(
            stream_l2_blocks, i32_const(design.l1_stream_bytes)
        )
        stream_stride_words = arith.divui(stream_stride, i32_const(4))
        stream_stride_words_64 = arith.extsi(i64, stream_stride_words)
        l2_blocks_64 = arith.extsi(i64, stream_l2_blocks)

        input_words = design.max_input_elems // 2
        l2_stream_words = design.l2_stream_bytes // 4
        # A shim BD access dimension is limited to [0:1023] elements.  Because
        # the L1 block is a multiple of 1 KiB, use 2 KiB inner chunks when they
        # divide evenly and 1 KiB chunks otherwise, hoisting the rest into the
        # next dimension.
        l1_stream_words = design.l1_stream_bytes // 4
        l1_words_lo = 512 if l1_stream_words % 512 == 0 else 256
        l1_words_hi = l1_stream_words // l1_words_lo
        group_outputs = design.group_outputs

        # Each repetition performs the full per-call work: config, inputs,
        # weights, and outputs.  Only the launch/dispatch overhead is shared.
        for _rep in range(reps):
            rep_tg = iron.TaskGroup()

            inputs_prod.fill(
                config_buf,
                offset=0,
                sizes=[1, 1, 1, input_words],
                strides=[1, 1, 1, 1],
                transfer_len=input_words,
                group=rep_tg,
                wait=False,
            )
            inputs_prod.fill(
                inputs_buf,
                offset=0,
                sizes=[1, 1, 1, input_words],
                strides=[1, 1, 1, 1],
                transfer_len=input_words,
                group=rep_tg,
                wait=False,
            )

            for g in range(design.groups):
                weights_prods[g].fill(
                    weights_buf,
                    offset=g
                    * design.computes_per_group
                    * stream_stride_words,
                    sizes=[
                        l2_blocks_64,
                        design.computes_per_group,
                        l1_words_hi,
                        l1_words_lo,
                    ],
                    strides=[
                        l1_stream_words,
                        stream_stride_words_64,
                        l1_words_lo,
                        1,
                    ],
                    transfer_len=l2_stream_words,
                    group=rep_tg,
                    wait=False,
                )

            for g in range(design.groups):
                outputs_conses[g].drain(
                    outputs_buf,
                    offset=g * group_outputs,
                    sizes=[group_outputs],
                    transfer_len=group_outputs,
                    group=rep_tg,
                    wait=True,
                )

            rep_tg.finish()

    rt = Runtime(
        sequence,
        [
            design.l3_weights_type,
            design.l3_inputs_type,
            design.l3_outputs_type,
            design.config_type,
            tile_bytes if tile_bytes is not None else np.int32,
            row_steps if row_steps is not None else np.int32,
            col_steps if col_steps is not None else np.int32,
            vec_steps if vec_steps is not None else np.int32,
            [
                weights_l2[g].prod(tile=shim_tiles[g])
                for g in range(design.groups)
            ],
            inputs_l2.prod(tile=shim_tiles[0]),
            [
                outputs_l2[g].cons(tile=shim_tiles[g])
                for g in range(design.groups)
            ],
        ],
        strict_task_groups=False,
    )

    prog = Program(NPU2(), rt, workers)
    return prog.resolve_program()


# ---------------------------------------------------------------------------
# Host packing and reference helpers


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def pack_weights_stream(
    design,
    weights_data,
    active_rows,
    row_steps,
    col_steps,
    weight_tile_bytes=None,
):
    """Pack one exact-stride stream per compute tile.

    Each tile packs the first ``active_rows // 8`` complete 8-row groups of a
    256x16 weight block as (row_group, col_tile, 8, 8).  A trailing partial
    group is not packed: the kernel only reads complete 64-weight vectors.

    Layout: [group][compute][band][row_step][tile_bytes], with each compute
    stream padded only at its end to an L1 stream block.
    """
    if not 8 <= active_rows <= design.rows_step:
        raise ValueError("active_rows must cover at least one vector")
    cols_tiles = design.cols_step // 8
    raw_tile_bytes = active_rows * design.cols_step * 2
    full_groups = active_rows // 8

    if weight_tile_bytes is None:
        # Standard weight-tile format: 64-byte aligned.
        tile_bytes = _round_up(raw_tile_bytes, 64)
    else:
        if weight_tile_bytes <= 0 or weight_tile_bytes % 64 != 0:
            raise ValueError(
                "weight_tile_bytes must be a positive multiple of 64"
            )
        if weight_tile_bytes < 256:
            raise ValueError(
                "weight_tile_bytes must contain at least one 8-row group"
            )
        if weight_tile_bytes > design.max_tile_bytes:
            raise ValueError(
                "weight_tile_bytes exceeds the design maximum "
                f"{design.max_tile_bytes}"
            )
        tile_bytes = weight_tile_bytes

    # Only whole vector pairs (one 8-row group) can be used by the kernel.
    effective_groups = min(full_groups, tile_bytes // 256)
    if effective_groups <= 0:
        raise ValueError("weight tile has no complete 8-row group")
    tile_vectors = effective_groups * 2

    stream_bytes = row_steps * col_steps * tile_bytes
    stream_stride = _round_up(stream_bytes, design.l1_stream_bytes)
    if stream_stride > design.max_stream_bytes:
        raise ValueError(
            f"stream {stream_stride} exceeds the design maximum "
            f"{design.max_stream_bytes}"
        )
    if stream_stride // design.l1_stream_bytes > 64:
        raise ValueError(
            "weight fill needs more than 64 stream blocks; split the fill "
            "(TODO)"
        )

    l3 = np.zeros(
        design.groups
        * design.computes_per_group
        * stream_stride,
        dtype=np.uint8,
    )
    full_tile_bytes = full_groups * 8 * design.cols_step * 2
    for band in range(col_steps):
        for row_step in range(row_steps):
            for group in range(design.groups):
                for compute in range(design.computes_per_group):
                    stream_base = (
                        group * design.computes_per_group + compute
                    ) * stream_stride
                    tile_base = (
                        stream_base
                        + (band * row_steps + row_step) * tile_bytes
                    )
                    col_start = (
                        band * design.band_cols
                        + group * design.computes_per_group * design.cols_step
                        + compute * design.cols_step
                    )
                    row_start = row_step * design.rows_step
                    tile = weights_data[
                        row_start : row_start + design.rows_step,
                        col_start : col_start + design.cols_step,
                    ]
                    # Only whole 8-row groups are ever read: the kernel wraps
                    # the complete 64-weight vectors and skips the trailing
                    # partial vector.
                    flat = tile[: full_groups * 8].reshape(
                        full_groups, 8, cols_tiles, 8
                    ).transpose(0, 2, 1, 3).reshape(-1)
                    data = np.ascontiguousarray(flat).view(np.uint8)
                    assert data.size == full_tile_bytes
                    copy_bytes = min(data.size, tile_bytes)
                    l3[
                        tile_base : tile_base + copy_bytes
                    ] = data[:copy_bytes]
    return l3, tile_bytes, stream_stride, tile_vectors


def pack_inputs(design, data, row_steps, vec_steps):
    """Pack inputs as [row_step][vec_step][row_tile][vec][8]."""
    out = np.zeros(
        row_steps * vec_steps * design.rows_step * design.vecs_per_step,
        dtype=bfloat16,
    )
    vecs = data.shape[0]
    for row_step in range(row_steps):
        for vec_step in range(vec_steps):
            base = (
                row_step * vec_steps + vec_step
            ) * design.rows_step * design.vecs_per_step
            for row_tile in range(design.rows_step // 8):
                for vec in range(design.vecs_per_step):
                    for column in range(8):
                        absolute_vec = vec_step * design.vecs_per_step + vec
                        index = (
                            row_step * design.rows_step
                            + row_tile * 8
                            + column
                        )
                        value = (
                            data[absolute_vec, index]
                            if absolute_vec < vecs
                            else bfloat16(0)
                        )
                        out[
                            base
                            + (row_tile * design.vecs_per_step + vec) * 8
                            + column
                        ] = value
    return out


def unpack_outputs(design, output, col_steps, vec_steps, vecs, cols):
    cols_tiles = design.cols_step // 8
    shaped = output[
        : design.groups * design.computes_per_group * design.compute_outputs
    ].reshape(
        design.groups,
        design.computes_per_group,
        design.max_col_steps,
        design.max_vec_steps,
        design.mmul_outputs,
    )
    actual = np.zeros(
        (vec_steps * design.vecs_per_step, col_steps * design.band_cols),
        dtype=np.float32,
    )
    for band in range(col_steps):
        for group in range(design.groups):
            for compute in range(design.computes_per_group):
                for vec_step in range(vec_steps):
                    tile = shaped[group, compute, band, vec_step]
                    for col_tile in range(cols_tiles):
                        for vec in range(design.vecs_per_step):
                            for column in range(8):
                                actual[
                                    vec_step * design.vecs_per_step + vec,
                                    band * design.band_cols
                                    + group
                                    * design.computes_per_group
                                    * design.cols_step
                                    + compute * design.cols_step
                                    + col_tile * 8
                                    + column,
                                ] = tile[
                                    col_tile * design.vecs_per_step * 8
                                    + vec * 8
                                    + column
                                ]
    return actual[:vecs, :cols]


def expected_output(
    design, inputs_data, weights_data, row_steps, active_rows, cols
):
    """Reference for the vector-wrapped weights actually sent to the NPU.

    The kernel fills the full 256-row multiply by repeating the complete
    64-weight vectors from the emulated tile.  A trailing partial vector is
    unused.
    """
    full_groups = active_rows // 8
    tile_vectors = full_groups * 2
    if tile_vectors <= 0:
        raise ValueError("need at least one complete 64-weight vector")

    cols_tiles = design.cols_step // 8
    emulated = np.zeros((row_steps * design.rows_step, cols), dtype=np.float32)
    vector_order = np.arange(design.rows_step // 8 * cols_tiles) % tile_vectors

    for row_step in range(row_steps):
        row_start = row_step * design.rows_step
        for band in range(cols // design.band_cols):
            for group in range(design.groups):
                for compute in range(design.computes_per_group):
                    col_start = (
                        band * design.band_cols
                        + group * design.computes_per_group * design.cols_step
                        + compute * design.cols_step
                    )
                    tile = weights_data[
                        row_start : row_start + design.rows_step,
                        col_start : col_start + design.cols_step,
                    ]
                    vectors = (
                        tile[: full_groups * 8]
                        .reshape(full_groups, 8, cols_tiles, 8)
                        .transpose(0, 2, 1, 3)
                        .reshape(tile_vectors, 64)
                    )
                    expanded = vectors[vector_order]
                    matrix = (
                        expanded.reshape(
                            design.rows_step // 8, cols_tiles, 8, 8
                        )
                        .transpose(0, 2, 1, 3)
                        .reshape(design.rows_step, design.cols_step)
                    )
                    emulated[
                        row_start : row_start + design.rows_step,
                        col_start : col_start + design.cols_step,
                    ] = matrix.astype(np.float32)

    return inputs_data.astype(np.float32) @ emulated


def run_case(
    design,
    label,
    active_rows,
    rows,
    cols,
    vecs,
    seed,
    warmup=5,
    iters=20,
    reps=1,
    weight_l2_depth=2,
    weight_l1_depth=2,
    weight_tile_bytes=None,
    skip_copy=0,
    skip_mmul=0,
):
    if rows % design.rows_step != 0:
        raise ValueError(f"rows must be a multiple of {design.rows_step}")
    if cols % design.band_cols != 0:
        raise ValueError(f"cols must be a multiple of {design.band_cols}")
    if vecs % design.vecs_per_step != 0:
        raise ValueError(f"vecs must be a multiple of {design.vecs_per_step}")

    row_steps = rows // design.rows_step
    col_steps = cols // design.band_cols
    vec_steps = vecs // design.vecs_per_step
    if row_steps > design.max_row_steps or col_steps > design.max_col_steps:
        raise ValueError("shape exceeds the compiled maximum")

    rng = np.random.default_rng(seed)
    weights_data = rng.integers(-4, 5, (rows, cols)).astype(bfloat16)
    inputs_data = rng.integers(-4, 5, (vecs, rows)).astype(bfloat16)

    l3_weights, tile_bytes, stream_stride, tile_vectors = (
        pack_weights_stream(
            design,
            weights_data,
            active_rows,
            row_steps,
            col_steps,
            weight_tile_bytes=weight_tile_bytes,
        )
    )
    effective_active_rows = (tile_vectors // 2) * 8
    packed_inputs = pack_inputs(design, inputs_data, row_steps, vec_steps)

    weights_host = np.zeros(design.max_l3_weights_bytes // 4, dtype=np.uint32)
    weights_host.view(np.uint8)[: l3_weights.size] = l3_weights
    inputs_host = np.zeros(design.max_input_elems // 2, dtype=np.uint32)
    inputs_host.view(bfloat16)[: packed_inputs.size] = packed_inputs

    config_host = np.zeros(design.max_input_elems // 2, dtype=np.uint32)
    # Word 5 is unused; the worker drives repetitions itself.
    config_host[:5] = [
        tile_bytes,
        tile_vectors,
        row_steps,
        col_steps,
        vec_steps,
    ]
    config_host[6] = stream_stride // design.l1_stream_bytes

    weights_tensor = iron.tensor(weights_host, dtype=np.uint32)
    inputs_tensor = iron.tensor(inputs_host, dtype=np.uint32)
    outputs_tensor = iron.zeros(design.max_output_elems, dtype=np.float32)
    config_tensor = iron.tensor(config_host, dtype=np.uint32)

    compile_kwargs = dict(
        design=design,
        tile_bytes=tile_bytes,
        row_steps=row_steps,
        col_steps=col_steps,
        vec_steps=vec_steps,
        reps=reps,
        weight_l2_depth=weight_l2_depth,
        weight_l1_depth=weight_l1_depth,
        skip_copy=skip_copy,
        skip_mmul=skip_mmul,
    )

    def run():
        return program(
            weights_tensor,
            inputs_tensor,
            outputs_tensor,
            config_tensor,
            **compile_kwargs,
        )

    bench = run_iters(run, warmup=warmup, iters=iters)

    if skip_copy or skip_mmul:
        max_error = 0.0
    else:
        actual = unpack_outputs(
            design,
            np.asarray(outputs_tensor),
            col_steps,
            vec_steps,
            vecs,
            cols,
        )
        expected = expected_output(
            design,
            inputs_data,
            weights_data,
            row_steps,
            effective_active_rows,
            cols,
        )
        max_error = float(np.abs(actual - expected).max())

    stats = bench.npu if bench.npu is not None else bench.e2e
    weight_payload = (
        design.groups
        * design.computes_per_group
        * row_steps
        * col_steps
        * tile_bytes
        * reps
    )
    input_bytes = design.max_input_elems * 2
    output_bytes = design.groups * design.group_outputs * 4 * reps
    config_bytes = input_bytes
    total_bytes = (
        l3_weights.size * reps + input_bytes + output_bytes + config_bytes
    )
    per_rep_us = stats.avg_us / reps
    avg_seconds = stats.avg_us * 1e-6
    best_seconds = stats.min_us * 1e-6
    avg_bw = total_bytes / avg_seconds / 1e9
    best_bw = total_bytes / best_seconds / 1e9
    useful_bw = weight_payload / avg_seconds / 1e9
    if skip_copy or skip_mmul:
        status = "skip"
    else:
        status = "ok" if max_error <= 1e-2 else "FAIL"
    print(
        f"{label:8s} rows={rows} cols={cols} vecs={vecs} "
        f"tile={tile_bytes} stream={stream_stride} reps={reps} "
        f"max_err={max_error:.6f} {status}"
    )
    print(
        f"  npu {stats.avg_us:8.1f} us "
        f"({per_rep_us:.1f} us/rep) "
        f"(min {stats.min_us:.1f}, max {stats.max_us:.1f})  "
        f"xfer {total_bytes / 1e6:.2f} MB  "
        f"bw avg {avg_bw:5.1f} / best {best_bw:5.1f} GB/s "
        f"({100 * avg_bw / PEAK_GBS:.1f}% of peak)  "
        f"payload {useful_bw:5.1f} GB/s"
    )
    return max_error <= 1e-2


def main():
    args = _build_parser().parse_args()
    design = StreamDesign(
        l1_stream_bytes=args.l1_stream_bytes,
        work_bytes=args.work_bytes,
    )
    reps = args.reps
    l2_depth = args.l2_depth
    l1_depth = args.l1_depth
    weight_tile_bytes = args.weight_tile_bytes
    skip_copy = int(args.skip_copy)
    skip_mmul = int(args.skip_mmul)

    cases = (
        ("q4_0", EMU_Q4_0_ROWS, 1),
        ("q6_k", EMU_Q6_K_ROWS, 2),
        # IQ1_S-like 832-byte tiles fit several tiles in one L1 stream block.
        ("iq1_s", EMU_IQ1_S_ROWS, 4),
    )
    status = True
    for label, active_rows, seed in cases:
        status &= run_case(
            design,
            label,
            active_rows,
            2048,
            2048,
            4,
            seed=seed,
            reps=reps,
            weight_l2_depth=l2_depth,
            weight_l1_depth=l1_depth,
            weight_tile_bytes=weight_tile_bytes,
            skip_copy=skip_copy,
            skip_mmul=skip_mmul,
        )
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
