"""Config-driven Spiral NPU program and Python test runner.

The same fixed-max design serves every shape: the host fills the maximum
input objects and weight blocks, and each compute tile drains the unused tail.
Per-call format and geometry travel in the config object.  `build.py` reuses
the same design for the C++ transaction path.
"""

import sys
from pathlib import Path
from typing import NamedTuple

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

KERNEL_DIR = PROGRAM_DIR / "kernels"
QUANTS_ROOT = PROGRAM_DIR.parent
BEGIN_SOURCE = KERNEL_DIR / "program_begin.aie.cc"
ZERO_SOURCE = KERNEL_DIR / "program_zero.aie.cc"
OUTPUT_SOURCE = KERNEL_DIR / "program_output.aie.cc"
ASSEMBLE_SOURCE = KERNEL_DIR / "program_assemble.aie.cc"
BLOCK_SOURCE = KERNEL_DIR / "program_block.aie.cc"
PROBE_SOURCE = KERNEL_DIR / "program_probe.aie.cc"
RAW_CHECK_SOURCE = KERNEL_DIR / "program_raw_check.aie.cc"
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
COMMON_SOURCE_FILES = [
    Path(design.__file__).resolve(),
    Path(__file__).resolve(),
    BEGIN_SOURCE,
    ZERO_SOURCE,
    OUTPUT_SOURCE,
    BLOCK_SOURCE,
    Q4_0_BLOCK_SOURCE,
    Q6_K_BLOCK_SOURCE,
    BLOCK_IMPL_HEADER,
    STATE_HEADER,
    *Q4_0_HEADERS,
    *Q6_K_HEADERS,
]
DIAGNOSTIC_SOURCE_FILES = [
    ASSEMBLE_SOURCE,
    PROBE_SOURCE,
    RAW_CHECK_SOURCE,
]
SOURCE_FILES = COMMON_SOURCE_FILES + DIAGNOSTIC_SOURCE_FILES
# Production builds omit the diagnostic split path and timing probes.  Keep
# this list separate from SOURCE_FILES so the JIT cache keys stay distinct.
PRODUCTION_SOURCE_FILES = COMMON_SOURCE_FILES

# Shared stream geometry, in 32-bit words where the shim DMA needs words.
_INPUT_WORDS = design.INPUT_GROUP_ELEMS // 2
_L1_WORDS = design.L1_STREAM_BYTES // 4
_L2_WORDS = design.L2_STREAM_BYTES // 4
_L1_LO = 512 if _L1_WORDS % 512 == 0 else 256
_L1_HI = _L1_WORDS // _L1_LO
_COMBINED = design.COMPUTES_PER_GROUP * _L1_HI


def _factor_bd(objects):
    """Factor an object count into two BD dimensions of at most 64."""
    if objects <= 64:
        return 1, objects
    for inner in range(1, 65):
        if objects % inner == 0 and objects // inner <= 64:
            return objects // inner, inner
    raise ValueError(f"{objects} objects do not factor into two BD dimensions")


def _fill_config(inputs_prod, config_buf, group):
    """Fill the config object separately from the input-group objects."""
    inputs_prod.fill(
        config_buf,
        offset=0,
        sizes=[1, 1, 1, _INPUT_WORDS],
        strides=[_INPUT_WORDS, 0, 0, 1],
        transfer_len=_INPUT_WORDS,
        group=group,
        wait=True,
    )


def _drain_outputs(outputs_conses, outputs_buf, group):
    """Drain each group into the N-major [N/8][vec][8] layout."""
    for g in range(design.GROUPS):
        outputs_conses[g].drain(
            outputs_buf,
            offset=g * design.COMPUTES_PER_GROUP * design.COLS_TILE *
            design.INPUTS_VECS,
            sizes=[
                design.COMPUTES_PER_GROUP,
                design.MAX_BANDS_PER_CHUNK,
                design.OUTPUT_TILE_ELEMS,
            ],
            strides=[
                design.COLS_TILE * design.INPUTS_VECS,
                design.BAND_COLS * design.INPUTS_VECS,
                1,
            ],
            transfer_len=design.GROUP_OUTPUTS,
            group=group,
            wait=True,
        )


class Fifos(NamedTuple):
    """Object FIFOs: shared input/output streams and per-group weights."""

    inputs_l2: object
    inputs_l1: object
    weights_l2: list
    weights_l1: list
    outputs_l2: list
    outputs_l1: list


class Kernels(NamedTuple):
    """External kernels referenced by the worker and the host sequences."""

    begin: object
    zero: object
    output: object
    block_begin: object
    q4_block: object
    q6_block: object
    assemble: object = None
    probe: object = None
    raw_check: object = None


class ComputeBuffers(NamedTuple):
    """Per-compute L1 buffers, indexed by g, c."""

    carry: list
    state: list
    config: list
    accum: list


def _compute_index(group, compute):
    return group * design.COMPUTES_PER_GROUP + compute


def _make_kernels(production=False):
    """Build the external kernels; production omits the diagnostic split."""
    begin = ExternalFunction(
        "program_begin",
        arg_types=[
            design.l1_input_group_type,
            design.l1_config_out_type,
            design.l1_state_type,
        ],
        compile_flags=[
            f"-DPROGRAM_CONFIG_LEN={design.CONFIG_LEN}",
            f"-DPROGRAM_STATE_WORDS={design.STATE_WORDS}",
        ],
        source_file=str(BEGIN_SOURCE),
        inline=False,
    )
    zero = ExternalFunction(
        "program_zero",
        arg_types=[design.l1_accum_type, np.int32],
        source_file=str(ZERO_SOURCE),
        inline=False,
    )
    output = ExternalFunction(
        "program_output",
        arg_types=[
            design.l1_accum_type,
            design.l1_outputs_type,
            np.int32,
            np.int32,
            np.int32,
        ],
        compile_flags=[
            f"-DPROGRAM_OUTPUT_TILE_ELEMS={design.OUTPUT_TILE_ELEMS}",
            f"-DPROGRAM_COLS_TILE={design.COLS_TILE}",
        ],
        source_file=str(OUTPUT_SOURCE),
        inline=False,
    )
    assemble = None
    probe = None
    raw_check = None
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
    block_args = [
        design.l1_stream_type,
        design.l1_carry_type,
        design.l1_input_group_type,
        design.l1_accum_type,
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
        f"-DPROGRAM_OUTPUT_TILE_ELEMS={design.OUTPUT_TILE_ELEMS}",
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
        raw_check = ExternalFunction(
            "program_raw_check",
            arg_types=[
                design.l1_stream_type,
                design.l1_outputs_type,
                np.int32,
                np.int32,
            ],
            source_file=str(RAW_CHECK_SOURCE),
            inline=False,
        )
    return Kernels(
        begin=begin,
        zero=zero,
        output=output,
        block_begin=block_begin,
        q4_block=q4_0_block,
        q6_block=q6_k_block,
        assemble=assemble,
        probe=probe,
        raw_check=raw_check,
    )


def _make_fifos(mem_tiles):
    """Build the input, weight, and output object FIFOs."""
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
    return Fifos(
        inputs_l2=inputs_l2,
        inputs_l1=inputs_l1,
        weights_l2=weights_l2,
        weights_l1=weights_l1,
        outputs_l2=outputs_l2,
        outputs_l1=outputs_l1,
    )


def _make_compute_buffers():
    """Allocate the four per-compute L1 buffers in one place."""

    def alloc(buffer_type, prefix):
        return [
            Buffer(
                buffer_type,
                name=f"{prefix}_{g}_{c}",
            )
            for g in range(design.GROUPS)
            for c in range(design.COMPUTES_PER_GROUP)
        ]

    return ComputeBuffers(
        carry=alloc(design.l1_carry_type, "carry"),
        state=alloc(design.l1_state_type, "state"),
        config=alloc(design.l1_config_out_type, "config"),
        accum=alloc(design.l1_accum_type, "accum"),
    )


def _worker_kernel_list(kernels, production):
    """Order the worker's kernels; diagnostic builds append the probes."""
    common = [
        kernels.begin,
        kernels.zero,
        kernels.output,
        kernels.block_begin,
        kernels.q4_block,
        kernels.q6_block,
    ]
    if production:
        return common
    return common + [kernels.assemble, kernels.probe, kernels.raw_check]


def _sequence_static(weights_buf, inputs_buf, config_buf, outputs_buf,
                     weights_prods, inputs_prod, outputs_conses):
    """Fixed-max host path used by the Python test runner."""
    group_stride_words = design.MAX_WEIGHT_BLOCKS * _L2_WORDS
    outer, inner = _factor_bd(design.MAX_WEIGHT_BLOCKS)
    outer_in, inner_in = _factor_bd(design.MAX_INPUT_GROUPS)

    tg = iron.TaskGroup()
    config_tg = iron.TaskGroup()
    _fill_config(inputs_prod, config_buf, config_tg)
    config_tg.finish()

    inputs_prod.fill(
        inputs_buf,
        offset=0,
        sizes=[outer_in, inner_in, 1, _INPUT_WORDS],
        strides=[inner_in * _INPUT_WORDS, _INPUT_WORDS, 0, 1],
        transfer_len=inner_in * _INPUT_WORDS,
        group=tg,
        wait=False,
    )

    for g in range(design.GROUPS):
        weights_prods[g].fill(
            weights_buf,
            offset=g * group_stride_words,
            sizes=[outer, inner, _COMBINED, _L1_LO],
            strides=[inner * _L2_WORDS, _L2_WORDS, _L1_LO, 1],
            transfer_len=inner * _L2_WORDS,
            group=tg,
            wait=False,
        )

    _drain_outputs(outputs_conses, outputs_buf, tg)
    tg.finish()


def _sequence_dynamic(weights_buf, inputs_buf, config_buf, outputs_buf,
                      weights_prods, inputs_prod, outputs_conses,
                      input0, input1, input2, input3,
                      weight0, weight1, group_stride_words):
    """Runtime-sized host path used by the C++ production caller."""
    i32 = IntegerType.get_signless(32)
    i64 = IntegerType.get_signless(64)

    def as_i64(value):
        return arith.extsi(i64, value)

    input_step = arith.constant(i32, _INPUT_WORDS)
    weight_step = arith.constant(i32, _L2_WORDS)

    tg = iron.TaskGroup()
    config_tg = iron.TaskGroup()
    _fill_config(inputs_prod, config_buf, config_tg)
    config_tg.finish()

    offset = arith.constant(i32, 0)
    for part in (input0, input1, input2, input3):
        inputs_prod.fill(
            inputs_buf,
            offset=offset,
            sizes=[as_i64(part), 1, 1, _INPUT_WORDS],
            strides=[_INPUT_WORDS, 0, 0, 1],
            transfer_len=_INPUT_WORDS,
            group=tg,
            wait=False,
        )
        offset = arith.addi(offset, arith.muli(part, input_step))

    for g in range(design.GROUPS):
        offset = arith.muli(arith.constant(i32, g), group_stride_words)
        for part in (weight0, weight1):
            weights_prods[g].fill(
                weights_buf,
                offset=offset,
                sizes=[as_i64(part), 1, _COMBINED, _L1_LO],
                strides=[_L2_WORDS, _L2_WORDS, _L1_LO, 1],
                transfer_len=_L2_WORDS,
                group=tg,
                wait=False,
            )
            offset = arith.addi(offset, arith.muli(part, weight_step))

    _drain_outputs(outputs_conses, outputs_buf, tg)
    tg.finish()


def _make_program(dynamic, production=False):
    """Build the fixed-max program.

    Flow: tiles -> object FIFOs -> external kernels -> per-compute buffers ->
    runtime sequence -> workers.  ``dynamic`` selects the runtime-sized host
    sequence used by the C++ caller; production omits the diagnostic path.
    """
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

    fifos = _make_fifos(mem_tiles)
    kernels = _make_kernels(production)
    bufs = _make_compute_buffers()

    def body(
        weights_cons,
        inputs_cons,
        outputs_prod,
        config_out,
        carry,
        state,
        accum,
        begin,
        zero,
        output,
        block_begin,
        q4_0_block,
        q6_k_block,
        assemble=None,
        probe=None,
        raw_check=None,
    ):
        """Per-compute worker: decode config, stream blocks, drain output."""
        config_buf = inputs_cons.acquire(1)
        begin(config_buf, config_out, state)
        inputs_cons.release(1)

        fmt = config_out[design.CFG_FORMAT]
        tile_bytes = config_out[design.CFG_TILE_BYTES]
        cols_bands = config_out[design.CFG_COLS_BANDS]
        weights_blocks = config_out[design.CFG_WEIGHTS_BLOCKS]
        rows_steps = config_out[design.CFG_ROWS_STEPS]

        i32 = IntegerType.get_signless(32)
        zero_i32 = arith.constant(i32, 0)
        one_i32 = arith.constant(i32, 1)
        two_i32 = arith.constant(i32, 2)
        group_rows_i32 = arith.constant(i32, design.INPUT_GROUP_ROWS)
        fmt_q4_i32 = arith.constant(i32, design.FORMAT_Q4_0)
        if not production:
            fmt_q6_i32 = arith.constant(i32, design.FORMAT_Q6_K)
            mode_fifo_i32 = arith.constant(i32, design.MODE_FIFO)
            mode_raw_i32 = arith.constant(i32, design.MODE_RAW_CHECK)
        num_groups = arith.divui(
            arith.addi(rows_steps, arith.subi(group_rows_i32, one_i32)),
            group_rows_i32,
        )

        def group_rows(group):
            """Rows covered after ``group`` has been consumed, clamped."""
            next_group = arith.muli(
                arith.addi(group, one_i32), group_rows_i32
            )
            return arith.select(
                arith.cmpi(CmpIPredicate.sgt, next_group, rows_steps),
                rows_steps,
                next_group,
            )

        def emit_tile_dispatch(
            active, active_block, active_input, active_limit, outputs_buf
        ):
            """Emit one tile of the streaming loop for the selected format."""
            with if_(active):
                with if_(
                    arith.cmpi(CmpIPredicate.eq, fmt, fmt_q4_i32)
                ) as is_q4:
                    q4_0_block(
                        active_block,
                        carry,
                        active_input,
                        accum,
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
                            accum,
                            state,
                            tile_bytes,
                            active_limit,
                            cols_bands,
                        )
                    else:
                        with if_(
                            arith.cmpi(
                                CmpIPredicate.eq, fmt, fmt_q6_i32
                            )
                        ) as is_q6:
                            q6_k_block(
                                active_block,
                                carry,
                                active_input,
                                accum,
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
                                    probe(
                                        active_block,
                                        carry,
                                        active_input,
                                        outputs_buf,
                                        state,
                                        tile_bytes,
                                        cols_bands,
                                    )

        def emit_loop():
            """Stream every weight block and input group through the tiles."""
            # The diagnostic split path stores tile checksums while streaming;
            # the real-format path writes its only output after the loop.
            outputs_buf = None
            if not production:
                outputs_buf = outputs_prod.acquire(1)
            zero(
                accum,
                arith.muli(
                    cols_bands,
                    arith.constant(i32, design.OUTPUT_TILE_ELEMS),
                ),
            )
            # Two slack units let the final block and input group drain after
            # their tile limit has been reached.
            max_units = arith.addi(
                arith.addi(weights_blocks, num_groups), two_i32
            )
            block = weights_cons.acquire(1)
            block_begin(state)
            input_group = inputs_cons.acquire(1)
            group0 = zero_i32

            # Carried state: the current weight block, the current input
            # group, and the group index that the input belongs to.
            for _unit, carried, _result in range_(
                max_units,
                iter_args=[block, input_group, group0],
                insert_yield=False,
            ):
                current_block, current_input, current_group = carried

                group_limit = arith.muli(
                    group_rows(current_group), cols_bands
                )

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
                active_limit = arith.muli(
                    group_rows(active_group), cols_bands
                )

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

                emit_tile_dispatch(
                    active, active_block, active_input, active_limit,
                    outputs_buf,
                )

                iron_yield(
                    [active_block, active_input, active_group]
                )

            weights_cons.release(1)
            inputs_cons.release(1)
            if production:
                outputs_buf = outputs_prod.acquire(1)
                output(
                    accum,
                    outputs_buf,
                    cols_bands,
                    config_out[design.CFG_SOFT_CAP_HI_LO],
                    config_out[design.CFG_SOFT_INV_HI_LO],
                )
            else:
                is_real = arith.ori(
                    arith.cmpi(CmpIPredicate.eq, fmt, fmt_q4_i32),
                    arith.cmpi(CmpIPredicate.eq, fmt, fmt_q6_i32),
                )
                with if_(is_real):
                    output(
                        accum,
                        outputs_buf,
                        cols_bands,
                        config_out[design.CFG_SOFT_CAP_HI_LO],
                        config_out[design.CFG_SOFT_INV_HI_LO],
                    )
            outputs_prod.release(1)

        def emit_diagnostic_paths():
            """Cycle and raw modes replace the compute loop in diag builds."""
            with if_(
                arith.cmpi(CmpIPredicate.eq, fmt, mode_fifo_i32)
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
                with if_(
                    arith.cmpi(CmpIPredicate.eq, fmt, mode_raw_i32)
                ) as is_raw:
                    outputs_buf = outputs_prod.acquire(1)
                    for block_index in range_(weights_blocks):
                        l1_block = weights_cons.acquire(1)
                        raw_check(
                            l1_block,
                            outputs_buf,
                            block_index,
                            arith.constant(
                                i32, design.L1_STREAM_BYTES
                            ),
                        )
                        weights_cons.release(1)
                    for _ in range_(num_groups):
                        inputs_buf = inputs_cons.acquire(1)
                        inputs_cons.release(1)
                    outputs_prod.release(1)
                with else_(is_raw):
                    emit_loop()

        def emit_padding():
            """Drain host-filled objects that the live geometry did not use."""
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
                    arith.constant(i32, design.MAX_INPUT_GROUPS),
                    num_groups,
                )
                for _extra in range_(extra_inputs):
                    inputs_buf = inputs_cons.acquire(1)
                    inputs_cons.release(1)

        if production:
            emit_loop()
        else:
            emit_diagnostic_paths()
        emit_padding()

    worker_kernels = _worker_kernel_list(kernels, production)
    workers = [
        Worker(
            body,
            [
                fifos.weights_l1[g][c].cons(),
                fifos.inputs_l1.cons(),
                fifos.outputs_l1[g][c].prod(),
                bufs.config[_compute_index(g, c)],
                bufs.carry[_compute_index(g, c)],
                bufs.state[_compute_index(g, c)],
                bufs.accum[_compute_index(g, c)],
                *worker_kernels,
            ],
            tile=compute_tiles[g][c],
            stack_size=8192,
        )
        for g in range(design.GROUPS)
        for c in range(design.COMPUTES_PER_GROUP)
    ]

    runtime_args = [
        design.l3_weights_type,
        design.l3_inputs_type,
        design.l3_config_type,
        design.l3_outputs_type,
        [
            fifos.weights_l2[g].prod(tile=shim_tiles[g])
            for g in range(design.GROUPS)
        ],
        fifos.inputs_l2.prod(tile=shim_tiles[0]),
        [
            fifos.outputs_l2[g].cons(tile=shim_tiles[g])
            for g in range(design.GROUPS)
        ],
    ]
    if dynamic:
        runtime_args += [np.int32] * 7
        rt = Runtime(
            _sequence_dynamic, runtime_args, strict_task_groups=False
        )
    else:
        rt = Runtime(
            _sequence_static, runtime_args, strict_task_groups=False
        )
    return Program(NPU2(), rt, workers).resolve_program()

@iron.jit(source_files=SOURCE_FILES)
def program(
    weights: In,
    inputs: In,
    config: In,
    outputs: Out,
):
    """Fixed-max program used by the Python test runner."""
    return _make_program(False)


@iron.jit(source_files=SOURCE_FILES)
def dynamic_program(
    weights: In,
    inputs: In,
    config: In,
    outputs: Out,
):
    """Runtime-sized diagnostic program used by ``build.py``."""
    return _make_program(True)


@iron.jit(source_files=PRODUCTION_SOURCE_FILES)
def production_dynamic_program(
    weights: In,
    inputs: In,
    config: In,
    outputs: Out,
):
    return _make_program(True, production=True)


if __name__ == "__main__":
    from tests import main

    raise SystemExit(main())
