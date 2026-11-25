"""Worker program: the per-compute device loop and kernel dispatch.

The body decodes the config object, streams weight blocks and input groups,
and dispatches the format-specific block kernel.  `npu.py` supplies the static
device handles; `build_workers` places the body on every compute tile.
"""

import sys
from pathlib import Path

from aie.dialects import arith
from aie.dialects._arith_enum_gen import CmpIPredicate
from aie.dialects.scf import IfOp
from aie.dialects.scf import yield_ as scf_yield
from aie.helpers.dialects.scf import else_, if_
from aie.ir import InsertionPoint, IntegerType
from aie.iron import Worker
from aie.iron.controlflow import range_
from aie.iron.controlflow import yield_ as iron_yield

PROGRAM_DIR = Path(__file__).resolve().parent
if str(PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_DIR))

import design  # noqa: E402
import npu  # noqa: E402


def build_workers(device, dynamic, production=False):
    """Place the worker body on every compute tile."""

    def body(
        weights_cons,
        inputs_cons,
        outputs_prod,
        config_out,
        carry,
        state,
        accum,
        prenorm_state,
        prenorm_weight,
        begin,
        zero,
        output,
        block_begin,
        q4_0_block,
        q6_k_block,
        prenorm_sum,
        prenorm_finish,
        prenorm_scale,
        prenorm_load,
        assemble=None,
        probe=None,
        raw_check=None,
    ):
        """Per-compute worker: decode config, stream blocks, drain output."""
        config_buf = inputs_cons.acquire(1)
        begin(config_buf, config_out, state)

        i32 = IntegerType.get_signless(32)
        zero_i32 = arith.constant(i32, 0)
        one_i32 = arith.constant(i32, 1)
        two_i32 = arith.constant(i32, 2)
        group_rows_i32 = arith.constant(i32, design.INPUT_GROUP_ROWS)
        fmt_q4_i32 = arith.constant(i32, design.FORMAT_Q4_0)
        prenorm = config_out[design.CFG_PRENORM]
        # Copy the config-carried norm weight into its stable L1 buffer before
        # the input stream reuses the config slot, then release it as usual.
        if production:
            with if_(arith.cmpi(CmpIPredicate.ne, prenorm, zero_i32)):
                prenorm_load(
                    config_buf,
                    prenorm_weight,
                    arith.muli(
                        config_out[design.CFG_ROWS_STEPS],
                        arith.constant(i32, 2 * design.ROWS_STEP),
                    ),
                )
        inputs_cons.release(1)

        fmt = config_out[design.CFG_FORMAT]
        tile_bytes = config_out[design.CFG_TILE_BYTES]
        cols_bands = config_out[design.CFG_COLS_BANDS]
        weights_blocks = config_out[design.CFG_WEIGHTS_BLOCKS]
        rows_steps = config_out[design.CFG_ROWS_STEPS]
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

        def emit_prenorm_sum():
            """Consume the first input fill set and reduce it to r hi/lo."""
            with if_(arith.cmpi(CmpIPredicate.ne, prenorm, zero_i32)):
                if dynamic:
                    input_pad = config_out[design.CFG_INPUT_PAD]
                    sum_groups = arith.addi(num_groups, input_pad)
                else:
                    sum_groups = arith.constant(
                        i32, design.MAX_INPUT_GROUPS
                    )
                for group_index in range_(sum_groups):
                    group = inputs_cons.acquire(1)
                    prenorm_sum(
                        group,
                        arith.muli(
                            arith.index_cast(i32, group_index),
                            group_rows_i32,
                        ),
                        rows_steps,
                        prenorm_state,
                    )
                    inputs_cons.release(1)
                prenorm_finish(
                    prenorm_state,
                    config_out[design.CFG_PRENORM_EPS],
                    config_out[design.CFG_PRENORM_INV_K],
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
            if production:
                with if_(arith.cmpi(CmpIPredicate.ne, prenorm, zero_i32)):
                    prenorm_scale(
                        input_group,
                        zero_i32,
                        rows_steps,
                        prenorm_weight,
                        prenorm_state,
                    )
            group0 = zero_i32

            # Carried state: the current weight block, the current input
            # group, and the group index that the input belongs to.
            for _unit, carried, _result in range_(
                max_units,
                iter_args=[block, input_group, group0],
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
                    if production:
                        with if_(
                            arith.cmpi(
                                CmpIPredicate.ne, prenorm, zero_i32
                            )
                        ):
                            prenorm_scale(
                                new_input,
                                arith.muli(
                                    arith.addi(current_group, one_i32),
                                    group_rows_i32,
                                ),
                                rows_steps,
                                prenorm_weight,
                                prenorm_state,
                            )
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
                    config_out[design.CFG_EPILOGUE],
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
                        config_out[design.CFG_EPILOGUE],
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
            emit_prenorm_sum()
            emit_loop()
        else:
            emit_diagnostic_paths()
        emit_padding()


    worker_kernels = npu.worker_kernel_list(device.kernels, production)
    return [
        Worker(
            body,
            [
                device.fifos.weights_l1[g][c].cons(),
                device.fifos.inputs_l1.cons(),
                device.fifos.outputs_l1[g][c].prod(),
                device.buffers.config[npu._compute_index(g, c)],
                device.buffers.carry[npu._compute_index(g, c)],
                device.buffers.state[npu._compute_index(g, c)],
                device.buffers.accum[npu._compute_index(g, c)],
                device.buffers.prenorm[npu._compute_index(g, c)],
                device.buffers.prenorm_weight[npu._compute_index(g, c)],
                *worker_kernels,
            ],
            tile=device.compute_tiles[g][c],
            stack_size=8192,
        )
        for g in range(design.GROUPS)
        for c in range(design.COMPUTES_PER_GROUP)
    ]
