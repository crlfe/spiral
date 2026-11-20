"""Config-driven Spiral NPU program and Python test runner.

The same fixed-max design serves every shape: the host fills the maximum
input objects and weight blocks, and each compute tile drains the unused tail.
Per-call format and geometry travel in the config object.  `build.py` reuses
the same design for the C++ transaction path.
"""

import sys
from pathlib import Path

import numpy as np
from aie import iron
from aie.dialects import arith
from aie.dialects._aie_enum_gen import AIETileType
from aie.dialects._arith_enum_gen import CmpIPredicate
from aie.dialects.scf import IfOp
from aie.dialects.scf import yield_ as scf_yield
from aie.helpers.dialects.scf import else_, if_
from aie.ir import InsertionPoint, IntegerType
from aie.extras.dialects.arith import index_cast
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
from aie.iron.controlflow import yield_ as iron_yield
from aie.iron.device import NPU2, Tile

PROGRAM_DIR = Path(__file__).resolve().parent
if str(PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_DIR))

import design
import pack

KERNEL_DIR = PROGRAM_DIR / "kernels"
QUANTS_ROOT = PROGRAM_DIR.parent
CONFIG_SOURCE = KERNEL_DIR / "program_config.aie.cc"
INIT_SOURCE = KERNEL_DIR / "program_init.aie.cc"
ZERO_SOURCE = KERNEL_DIR / "program_zero.aie.cc"
ASSEMBLE_SOURCE = KERNEL_DIR / "program_assemble.aie.cc"
BLOCK_SOURCE = KERNEL_DIR / "program_block.aie.cc"
PROBE_SOURCE = KERNEL_DIR / "program_probe.aie.cc"
STREAM_SOURCE = KERNEL_DIR / "program_stream.aie.cc"
Q4_0_BLOCK_SOURCE = KERNEL_DIR / "program_q4_block.aie.cc"
Q6_K_BLOCK_SOURCE = KERNEL_DIR / "program_q6_block.aie.cc"
# Shared block body and helpers; included by every `program_*_block` source.
BLOCK_IMPL_HEADER = KERNEL_DIR / "program_block_impl.h"
STATE_HEADER = KERNEL_DIR / "program_state.h"
Q4_0_HEADERS = [
    QUANTS_ROOT / "quants/q4_0/mmul.aie.h",
    QUANTS_ROOT / "quants/q4_0/unpack.aie.h",
    QUANTS_ROOT / "quants/q4_0/layout.h",
    QUANTS_ROOT / "quants/common/mmul.aie.h",
    QUANTS_ROOT / "quants/common/scale.aie.h",
    QUANTS_ROOT / "quants/common/tile.h",
]
Q6_K_HEADERS = [
    QUANTS_ROOT / "quants/q6_k/mmul.aie.h",
    QUANTS_ROOT / "quants/q6_k/unpack.aie.h",
    QUANTS_ROOT / "quants/q6_k/layout.h",
    QUANTS_ROOT / "quants/common/q6_scale.aie.h",
    QUANTS_ROOT / "quants/common/mmul.aie.h",
    QUANTS_ROOT / "quants/common/scale.aie.h",
    QUANTS_ROOT / "quants/common/tile.h",
]
# design.py and this runner are part of the JIT recipe: their content is
# hashed with the kernel sources, so fixed constants and module-level helpers
# invalidate the cache automatically.
SOURCE_FILES = [
    Path(design.__file__).resolve(),
    Path(__file__).resolve(),
    CONFIG_SOURCE,
    INIT_SOURCE,
    ZERO_SOURCE,
    ASSEMBLE_SOURCE,
    BLOCK_SOURCE,
    PROBE_SOURCE,
    STREAM_SOURCE,
    Q4_0_BLOCK_SOURCE,
    Q6_K_BLOCK_SOURCE,
    BLOCK_IMPL_HEADER,
    STATE_HEADER,
    *Q4_0_HEADERS,
    *Q6_K_HEADERS,
]

# Production builds omit the diagnostic split path and timing probes.  Keep
# this list separate from SOURCE_FILES so the JIT cache keys stay distinct.
PRODUCTION_SOURCE_FILES = [
    Path(design.__file__).resolve(),
    Path(__file__).resolve(),
    CONFIG_SOURCE,
    INIT_SOURCE,
    ZERO_SOURCE,
    BLOCK_SOURCE,
    Q4_0_BLOCK_SOURCE,
    Q6_K_BLOCK_SOURCE,
    BLOCK_IMPL_HEADER,
    STATE_HEADER,
    *Q4_0_HEADERS,
    *Q6_K_HEADERS,
]

# Transport tile sizes from the quant layouts.
Q4_0_TILE_BYTES = 2304
Q6_K_TILE_BYTES = 3392


def _factor_bd(objects):
    """Factor an object count into two BD dimensions of at most 64."""
    if objects <= 64:
        return 1, objects
    for inner in range(1, 65):
        if objects % inner == 0 and objects // inner <= 64:
            return objects // inner, inner
    raise ValueError(f"{objects} objects do not factor into two BD dimensions")


def _unpack_output_tile(raw):
    """Map the kernel's [col_half][vec][8 cols] tile into [vec][16]."""
    half = design.COLS_TILE // 2
    return (
        raw.reshape(2, design.INPUTS_VECS, half)
        .transpose(1, 0, 2)
        .reshape(design.INPUTS_VECS, design.COLS_TILE)
    )


def _real_max_error(actual, expected, cols_bands):
    max_error = 0.0
    for g in range(design.GROUPS):
        for c in range(design.COMPUTES_PER_GROUP):
            base = (
                g * design.GROUP_OUTPUTS
                + c * design.OUTPUT_ELEMS
            )
            for band in range(cols_bands):
                start = base + band * design.OUTPUT_TILE_ELEMS
                tile = _unpack_output_tile(
                    actual[start : start + design.OUTPUT_TILE_ELEMS]
                )
                error = float(np.abs(tile - expected[g, c, band]).max())
                max_error = max(max_error, error)
    return max_error


def _noop_mismatches(actual, expected, total_tiles):
    mismatches = 0
    first = None
    for g in range(design.GROUPS):
        for c in range(design.COMPUTES_PER_GROUP):
            compute_base = (
                g * design.GROUP_OUTPUTS
                + c * design.OUTPUT_ELEMS
            )
            for tile_index in range(total_tiles):
                got = actual[compute_base + tile_index]
                want = expected[g, c, tile_index]
                if got != want:
                    mismatches += 1
                    if first is None:
                        first = (g, c, tile_index, int(got), int(want))
    return mismatches, first


def _make_kernels(production=False):
    config = ExternalFunction(
        "program_config_decode",
        arg_types=[design.l1_input_group_type, design.l1_config_out_type],
        source_file=str(CONFIG_SOURCE),
        inline=False,
    )
    init = ExternalFunction(
        "program_init",
        arg_types=[design.l1_state_type],
        source_file=str(INIT_SOURCE),
        inline=False,
    )
    zero = ExternalFunction(
        "program_zero",
        arg_types=[design.l1_outputs_type, np.int32],
        source_file=str(ZERO_SOURCE),
        inline=False,
    )
    assemble = None
    stream = None
    probe = None
    if not production:
        assemble = ExternalFunction(
            "program_assemble",
            arg_types=[
                design.l1_stream_type,
                design.l1_carry_type,
                design.l1_state_type,
                np.int32,
                np.int32,
            ],
            compile_flags=[
                f"-DPROGRAM_BLOCK_BYTES={design.L1_STREAM_BYTES}",
                f"-DPROGRAM_CARRY_BYTES={design.CARRY_BYTES}",
            ],
            source_file=str(ASSEMBLE_SOURCE),
            inline=False,
        )
    block_begin = ExternalFunction(
        "program_block_begin",
        arg_types=[design.l1_state_type],
        source_file=str(BLOCK_SOURCE),
        inline=False,
    )
    if not production:
        stream = ExternalFunction(
            "program_stream",
            arg_types=[
                design.l1_stream_type,
                design.l1_carry_type,
                design.l1_input_group_type,
                design.l1_outputs_type,
                design.l1_state_type,
                np.int32,
            ],
            source_file=str(STREAM_SOURCE),
            inline=False,
        )
    block_args = [
        design.l1_stream_type,
        design.l1_carry_type,
        design.l1_input_group_type,
        design.l1_outputs_type,
        design.l1_state_type,
        np.int32,
        np.int32,
        np.int32,
    ]
    block_flags = [
        f"-DPROGRAM_BLOCK_BYTES={design.L1_STREAM_BYTES}",
        f"-DPROGRAM_MAX_TILES={design.MAX_TILES_PER_BLOCK}",
        f"-DPROGRAM_INPUT_ELEMS={design.INPUT_ELEMS}",
        f"-DPROGRAM_INPUT_GROUP_ROWS={design.INPUT_GROUP_ROWS}",
    ]
    q6_k_block = ExternalFunction(
        "program_q6_block",
        arg_types=block_args,
        compile_flags=block_flags,
        include_dirs=[str(QUANTS_ROOT), str(KERNEL_DIR)],
        source_file=str(Q6_K_BLOCK_SOURCE),
        inline=False,
    )
    q4_0_block = ExternalFunction(
        "program_q4_block",
        arg_types=block_args,
        compile_flags=block_flags,
        include_dirs=[str(QUANTS_ROOT), str(KERNEL_DIR)],
        source_file=str(Q4_0_BLOCK_SOURCE),
        inline=False,
    )
    if not production:
        probe = ExternalFunction(
            "program_probe",
            arg_types=[
                design.l1_stream_type,
                design.l1_carry_type,
                design.l1_input_group_type,
                design.l1_outputs_type,
                design.l1_state_type,
                np.int32,
                np.int32,
            ],
            compile_flags=[
                f"-DPROGRAM_INPUT_ELEMS={design.INPUT_ELEMS}",
                f"-DPROGRAM_INPUT_GROUP_ROWS={design.INPUT_GROUP_ROWS}",
            ],
            source_file=str(PROBE_SOURCE),
            inline=False,
        )
    return (
        config,
        init,
        zero,
        assemble,
        block_begin,
        probe,
        stream,
        q4_0_block,
        q6_k_block,
    )


def _make_fifos(mem_tiles):
    inputs_l2 = ObjectFifo(
        design.l1_input_group_type, name="inputs_l2", depth=4
    )
    inputs_l1 = inputs_l2.cons().forward(
        tile=mem_tiles[0],
        obj_type=design.l1_input_group_type,
        name="inputs_l1",
    )

    weights_l2 = [
        ObjectFifo(
            design.l2_stream_type, name=f"weights_l2_{g}", depth=2
        )
        for g in range(design.GROUPS)
    ]
    weights_l1 = [
        weights_l2[g]
        .cons()
        .split(
            offsets=[
                c * design.L1_STREAM_BYTES
                for c in range(design.COMPUTES_PER_GROUP)
            ],
            tile=mem_tiles[g],
            obj_types=[design.l1_stream_type]
            * design.COMPUTES_PER_GROUP,
            depths=[2] * design.COMPUTES_PER_GROUP,
            names=[
                f"weights_l1_{g}_{c}"
                for c in range(design.COMPUTES_PER_GROUP)
            ],
        )
        for g in range(design.GROUPS)
    ]

    outputs_l2 = [
        ObjectFifo(
            design.l2_outputs_type, name=f"outputs_l2_{g}", depth=1
        )
        for g in range(design.GROUPS)
    ]
    outputs_l1 = [
        outputs_l2[g]
        .prod()
        .join(
            offsets=[
                c * design.OUTPUT_ELEMS
                for c in range(design.COMPUTES_PER_GROUP)
            ],
            tile=mem_tiles[g],
            obj_types=[design.l1_outputs_type]
            * design.COMPUTES_PER_GROUP,
            names=[
                f"outputs_l1_{g}_{c}"
                for c in range(design.COMPUTES_PER_GROUP)
            ],
        )
        for g in range(design.GROUPS)
    ]
    return inputs_l2, inputs_l1, weights_l2, weights_l1, outputs_l2, outputs_l1


def _make_program(dynamic, production=False):
    shim_tiles = [
        Tile(col=g, row=0, tile_type=AIETileType.ShimNOCTile)
        for g in range(design.GROUPS)
    ]
    mem_tiles = [
        Tile(col=g, row=1, tile_type=AIETileType.MemTile)
        for g in range(design.GROUPS)
    ]
    compute_tiles = [
        [
            Tile(col=g, row=2 + c, tile_type=AIETileType.CoreTile)
            for c in range(design.COMPUTES_PER_GROUP)
        ]
        for g in range(design.GROUPS)
    ]

    (
        inputs_l2,
        inputs_l1,
        weights_l2,
        weights_l1,
        outputs_l2,
        outputs_l1,
    ) = _make_fifos(mem_tiles)

    (
        config,
        init,
        zero,
        assemble,
        block_begin,
        probe,
        stream,
        q4_0_block,
        q6_k_block,
    ) = _make_kernels(production)

    carry_buffers = [
        Buffer(design.l1_carry_type, name=f"carry_{g}_{c}")
        for g in range(design.GROUPS)
        for c in range(design.COMPUTES_PER_GROUP)
    ]
    state_buffers = [
        Buffer(design.l1_state_type, name=f"state_{g}_{c}")
        for g in range(design.GROUPS)
        for c in range(design.COMPUTES_PER_GROUP)
    ]
    config_buffers = [
        Buffer(design.l1_config_out_type, name=f"config_{g}_{c}")
        for g in range(design.GROUPS)
        for c in range(design.COMPUTES_PER_GROUP)
    ]

    def body(
        weights_cons,
        inputs_cons,
        outputs_prod,
        config_out,
        carry,
        state,
        init,
        decode,
        zero,
        block_begin,
        q4_0_block,
        q6_k_block,
        assemble=None,
        probe=None,
        stream=None,
    ):
        config_buf = inputs_cons.acquire(1)
        decode(config_buf, config_out)
        inputs_cons.release(1)

        fmt = config_out[design.CFG_FORMAT]
        tile_bytes = config_out[design.CFG_TILE_BYTES]
        cols_bands = config_out[design.CFG_COLS_BANDS]
        weights_blocks = config_out[design.CFG_WEIGHTS_BLOCKS]
        total_tiles = config_out[design.CFG_TOTAL_TILES]
        rows_steps = config_out[design.CFG_ROWS_STEPS]

        init(state)
        i32 = IntegerType.get_signless(32)
        zero_i32 = arith.constant(i32, 0)
        one_i32 = arith.constant(i32, 1)
        two_i32 = arith.constant(i32, 2)
        three_i32 = arith.constant(i32, 3)
        four_i32 = arith.constant(i32, 4)
        num_groups = arith.divui(
            arith.addi(rows_steps, three_i32), four_i32
        )

        def emit_loop():
            outputs_buf = outputs_prod.acquire(1)
            zero(
                outputs_buf,
                arith.muli(
                    cols_bands,
                    arith.constant(i32, design.OUTPUT_TILE_ELEMS),
                ),
            )
            max_units = arith.addi(
                arith.addi(weights_blocks, num_groups), two_i32
            )
            block = weights_cons.acquire(1)
            block_begin(state)
            input_group = inputs_cons.acquire(1)
            group0 = zero_i32

            for _unit, carried, _result in range_(
                max_units,
                iter_args=[block, input_group, group0],
                insert_yield=False,
            ):
                current_block, current_input, current_group = carried

                next_rows = arith.select(
                    arith.cmpi(
                        CmpIPredicate.sgt,
                        arith.muli(
                            arith.addi(current_group, one_i32), four_i32
                        ),
                        rows_steps,
                    ),
                    rows_steps,
                    arith.muli(
                        arith.addi(current_group, one_i32), four_i32
                    ),
                )
                group_limit = arith.muli(next_rows, cols_bands)

                advance = arith.andi(
                    arith.cmpi(
                        CmpIPredicate.slt,
                        current_group,
                        arith.subi(num_groups, one_i32),
                    ),
                    arith.cmpi(
                        CmpIPredicate.sge,
                        state[design.STATE_TILE_INDEX],
                        group_limit,
                    ),
                )
                next_input = IfOp(
                    advance, [current_input.type, i32], has_else=True
                )
                with InsertionPoint(next_input.then_block):
                    inputs_cons.release(1)
                    new_input = inputs_cons.acquire(1)
                    scf_yield(
                        [new_input, arith.addi(current_group, one_i32)]
                    )
                with InsertionPoint(next_input.else_block):
                    scf_yield([current_input, current_group])

                active_group = next_input.results[1]
                active_input = next_input.results[0]
                active_rows = arith.select(
                    arith.cmpi(
                        CmpIPredicate.sgt,
                        arith.muli(
                            arith.addi(active_group, one_i32), four_i32
                        ),
                        rows_steps,
                    ),
                    rows_steps,
                    arith.muli(
                        arith.addi(active_group, one_i32), four_i32
                    ),
                )
                active_limit = arith.muli(active_rows, cols_bands)

                need_block = arith.andi(
                    arith.cmpi(
                        CmpIPredicate.ne,
                        state[design.STATE_BLOCK_DONE],
                        zero_i32,
                    ),
                    arith.cmpi(
                        CmpIPredicate.slt,
                        state[design.STATE_TILE_INDEX],
                        active_limit,
                    ),
                )
                next_block = IfOp(
                    need_block, [current_block.type], has_else=True
                )
                with InsertionPoint(next_block.then_block):
                    weights_cons.release(1)
                    block_begin(state)
                    fresh_block = weights_cons.acquire(1)
                    scf_yield([fresh_block])
                with InsertionPoint(next_block.else_block):
                    scf_yield([current_block])

                active_block = next_block.results[0]
                active = arith.andi(
                    arith.cmpi(
                        CmpIPredicate.slt, active_group, num_groups
                    ),
                    arith.cmpi(
                        CmpIPredicate.slt,
                        state[design.STATE_TILE_INDEX],
                        active_limit,
                    ),
                )

                with if_(active):
                    with if_(
                        arith.cmpi(CmpIPredicate.eq, fmt, one_i32)
                    ) as is_q4:
                        q4_0_block(
                            active_block,
                            carry,
                            active_input,
                            outputs_buf,
                            state,
                            tile_bytes,
                            active_limit,
                            cols_bands,
                        )
                    with else_(is_q4):
                        if production:
                            q6_k_block(
                                active_block,
                                carry,
                                active_input,
                                outputs_buf,
                                state,
                                tile_bytes,
                                active_limit,
                                cols_bands,
                            )
                        else:
                            with if_(
                                arith.cmpi(
                                    CmpIPredicate.eq, fmt, two_i32
                                )
                            ) as is_q6:
                                q6_k_block(
                                    active_block,
                                    carry,
                                    active_input,
                                    outputs_buf,
                                    state,
                                    tile_bytes,
                                    active_limit,
                                    cols_bands,
                                )
                            with else_(is_q6):
                                for _attempt in range_(
                                    design.MAX_TILES_PER_BLOCK
                                ):
                                    assemble(
                                        active_block,
                                        carry,
                                        state,
                                        tile_bytes,
                                        active_limit,
                                    )
                                    ready = arith.cmpi(
                                        CmpIPredicate.ne,
                                        state[design.STATE_READY],
                                        zero_i32,
                                    )
                                    with if_(ready):
                                        with if_(
                                            arith.cmpi(
                                                CmpIPredicate.eq,
                                                fmt,
                                                three_i32,
                                            )
                                        ) as is_stream:
                                            stream(
                                                active_block,
                                                carry,
                                                active_input,
                                                outputs_buf,
                                                state,
                                                cols_bands,
                                            )
                                        with else_(is_stream):
                                            probe(
                                                active_block,
                                                carry,
                                                active_input,
                                                outputs_buf,
                                                state,
                                                tile_bytes,
                                                cols_bands,
                                            )

                iron_yield(
                    [active_block, active_input, active_group]
                )

            weights_cons.release(1)
            inputs_cons.release(1)
            outputs_prod.release(1)

        if production:
            emit_loop()
        else:
            with if_(
                arith.cmpi(CmpIPredicate.eq, fmt, four_i32)
            ) as is_cycle:
                for _ in range_(weights_blocks):
                    l1_block = weights_cons.acquire(1)
                    weights_cons.release(1)
                for _ in range_(num_groups):
                    inputs_buf = inputs_cons.acquire(1)
                    inputs_cons.release(1)
                outputs_buf = outputs_prod.acquire(1)
                outputs_prod.release(1)
            with else_(is_cycle):
                emit_loop()

        if dynamic:
            input_pad = config_out[design.CFG_INPUT_PAD]
            weight_pad = config_out[design.CFG_WEIGHT_PAD]
            for _extra in range_(input_pad):
                inputs_buf = inputs_cons.acquire(1)
                inputs_cons.release(1)
            for _extra in range_(weight_pad):
                l1_block = weights_cons.acquire(1)
                weights_cons.release(1)
        else:
            extra_blocks = arith.subi(
                arith.constant(i32, design.MAX_WEIGHT_BLOCKS),
                weights_blocks,
            )
            for _extra in range_(extra_blocks):
                l1_block = weights_cons.acquire(1)
                weights_cons.release(1)

            extra_inputs = arith.subi(
                arith.constant(i32, design.MAX_INPUT_OBJECTS - 1),
                num_groups,
            )
            for _extra in range_(extra_inputs):
                inputs_buf = inputs_cons.acquire(1)
                inputs_cons.release(1)


    worker_kernels = [
        init,
        config,
        zero,
        block_begin,
        q4_0_block,
        q6_k_block,
    ]
    if not production:
        worker_kernels += [assemble, probe, stream]

    workers = [
        Worker(
            body,
            [
                weights_l1[g][c].cons(),
                inputs_l1.cons(),
                outputs_l1[g][c].prod(),
                config_buffers[g * design.COMPUTES_PER_GROUP + c],
                carry_buffers[g * design.COMPUTES_PER_GROUP + c],
                state_buffers[g * design.COMPUTES_PER_GROUP + c],
                *worker_kernels,
            ],
            tile=compute_tiles[g][c],
            stack_size=8192,
        )
        for g in range(design.GROUPS)
        for c in range(design.COMPUTES_PER_GROUP)
    ]

    def sequence_static(
        weights_buf,
        inputs_buf,
        outputs_buf,
        weights_prods,
        inputs_prod,
        outputs_conses,
    ):
        input_words = design.INPUT_GROUP_ELEMS // 2
        l1_words = design.L1_STREAM_BYTES // 4
        l2_words = design.L2_STREAM_BYTES // 4
        group_stride_words = design.MAX_WEIGHT_BLOCKS * l2_words
        l1_lo = 512 if l1_words % 512 == 0 else 256
        l1_hi = l1_words // l1_lo
        combined = design.COMPUTES_PER_GROUP * l1_hi
        outer, inner = _factor_bd(design.MAX_WEIGHT_BLOCKS)
        outer_in, inner_in = _factor_bd(design.MAX_INPUT_OBJECTS)

        tg = iron.TaskGroup()
        inputs_prod.fill(
            inputs_buf,
            offset=0,
            sizes=[outer_in, inner_in, 1, input_words],
            strides=[inner_in * input_words, input_words, 0, 1],
            transfer_len=inner_in * input_words,
            group=tg,
            wait=False,
        )

        for g in range(design.GROUPS):
            weights_prods[g].fill(
                weights_buf,
                offset=g * group_stride_words,
                sizes=[outer, inner, combined, l1_lo],
                strides=[
                    inner * l2_words,
                    l2_words,
                    l1_lo,
                    1,
                ],
                transfer_len=inner * l2_words,
                group=tg,
                wait=False,
            )

        for g in range(design.GROUPS):
            outputs_conses[g].drain(
                outputs_buf,
                offset=g * design.GROUP_OUTPUTS,
                sizes=[1, 1, 1, design.GROUP_OUTPUTS],
                strides=[0, 0, 0, 1],
                transfer_len=design.GROUP_OUTPUTS,
                group=tg,
                wait=True,
            )

        tg.finish()

    def sequence_dynamic(
        weights_buf,
        inputs_buf,
        outputs_buf,
        weights_prods,
        inputs_prod,
        outputs_conses,
        input0,
        input1,
        input2,
        input3,
        weight0,
        weight1,
        group_stride_words,
    ):
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)

        def as_i64(value):
            return arith.extsi(i64, value)

        input_words = design.INPUT_GROUP_ELEMS // 2
        l1_words = design.L1_STREAM_BYTES // 4
        l2_words = design.L2_STREAM_BYTES // 4
        l1_lo = 512 if l1_words % 512 == 0 else 256
        l1_hi = l1_words // l1_lo
        combined = design.COMPUTES_PER_GROUP * l1_hi
        input_step = arith.constant(i32, input_words)
        weight_step = arith.constant(i32, l2_words)

        tg = iron.TaskGroup()
        inputs_prod.fill(
            inputs_buf,
            offset=0,
            sizes=[1, 1, 1, input_words],
            strides=[input_words, 0, 0, 1],
            transfer_len=input_words,
            group=tg,
            wait=False,
        )

        offset = arith.constant(i32, input_words)
        for part in (input0, input1, input2, input3):
            inputs_prod.fill(
                inputs_buf,
                offset=offset,
                sizes=[as_i64(part), 1, 1, input_words],
                strides=[input_words, 0, 0, 1],
                transfer_len=input_words,
                group=tg,
                wait=False,
            )
            offset = arith.addi(
                offset, arith.muli(part, input_step)
            )

        for g in range(design.GROUPS):
            offset = arith.muli(
                arith.constant(i32, g), group_stride_words
            )
            for part in (weight0, weight1):
                weights_prods[g].fill(
                    weights_buf,
                    offset=offset,
                    sizes=[as_i64(part), 1, combined, l1_lo],
                    strides=[l2_words, l2_words, l1_lo, 1],
                    transfer_len=l2_words,
                    group=tg,
                    wait=False,
                )
                offset = arith.addi(
                    offset, arith.muli(part, weight_step)
                )

        for g in range(design.GROUPS):
            outputs_conses[g].drain(
                outputs_buf,
                offset=g * design.GROUP_OUTPUTS,
                sizes=[1, 1, 1, design.GROUP_OUTPUTS],
                strides=[0, 0, 0, 1],
                transfer_len=design.GROUP_OUTPUTS,
                group=tg,
                wait=True,
            )

        tg.finish()

    runtime_args = [
        design.l3_weights_type,
        design.l3_inputs_type,
        design.l3_outputs_type,
        [
            weights_l2[g].prod(tile=shim_tiles[g])
            for g in range(design.GROUPS)
        ],
        inputs_l2.prod(tile=shim_tiles[0]),
        [
            outputs_l2[g].cons(tile=shim_tiles[g])
            for g in range(design.GROUPS)
        ],
    ]
    if dynamic:
        runtime_args += [np.int32] * 7
        rt = Runtime(
            sequence_dynamic, runtime_args, strict_task_groups=False
        )
    else:
        rt = Runtime(
            sequence_static, runtime_args, strict_task_groups=False
        )
    return Program(NPU2(), rt, workers).resolve_program()


@iron.jit(source_files=SOURCE_FILES)
def program(
    weights: In,
    inputs: In,
    outputs: Out,
):
    return _make_program(False)


@iron.jit(source_files=SOURCE_FILES)
def dynamic_program(
    weights: In,
    inputs: In,
    outputs: Out,
):
    return _make_program(True)


@iron.jit(source_files=PRODUCTION_SOURCE_FILES)
def production_program(
    weights: In,
    inputs: In,
    outputs: Out,
):
    return _make_program(False, production=True)


@iron.jit(source_files=PRODUCTION_SOURCE_FILES)
def production_dynamic_program(
    weights: In,
    inputs: In,
    outputs: Out,
):
    return _make_program(True, production=True)


def run_case(rows_steps, cols_bands, tile_bytes, seed=1, format=0):
    if format == 0:
        case = pack.build_case(rows_steps, cols_bands, tile_bytes, seed)
    elif format == 1:
        case = pack.build_q4_0_case(rows_steps, cols_bands, seed)
    elif format == 2:
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
    input_objects = 1 + input_groups
    inputs_words[: input_objects * design.INPUT_GROUP_ELEMS] = case[
        "inputs"
    ].reshape(-1)

    weights_tensor = iron.tensor(weights_host, dtype=np.uint32)
    inputs_tensor = iron.tensor(inputs_host, dtype=np.uint32)
    outputs_tensor = iron.zeros(
        design.MAX_OUTPUT_ELEMS, dtype=np.float32
    )

    program(weights_tensor, inputs_tensor, outputs_tensor)

    if format != 0:
        max_error = _real_max_error(
            np.asarray(outputs_tensor), case["expected"], cols_bands
        )
        label = "q4_0" if format == 1 else "q6_k"
        tolerance = 1e-2 if format == 1 else 2e-2
        print(
            f"{label} rows_steps={rows_steps} cols_bands={cols_bands} "
            f"tile={tile_bytes} blocks={weights_blocks} "
            f"tiles={total_tiles} max_error={max_error:.6f}"
        )
        return max_error <= tolerance

    mismatches, first = _noop_mismatches(
        np.asarray(outputs_tensor).view(np.uint32),
        case["expected"],
        total_tiles,
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
            format=2,
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

    status = True
    for rows_steps, cols_bands, tile_bytes in noop_cases:
        status &= run_case(rows_steps, cols_bands, tile_bytes)
    for rows_steps, cols_bands in q4_0_cases:
        status &= run_case(
            rows_steps, cols_bands, Q4_0_TILE_BYTES, format=1
        )
    for rows_steps, cols_bands in q6_k_cases:
        status &= run_case(
            rows_steps, cols_bands, Q6_K_TILE_BYTES, format=2
        )
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
