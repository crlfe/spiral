"""Host runtime sequence: the DMA fills and drains around the NPU work.

The sequence walks the fixed stream layouts in `design.py`; it is independent
of the worker program.  `run.py` selects one of the builders and hands it to
`aie.iron.Runtime` with `npu.runtime_args()`.
"""

import sys
from functools import partial
from pathlib import Path

from aie import iron
from aie.dialects import arith
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import IntegerType

PROGRAM_DIR = Path(__file__).resolve().parent
if str(PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_DIR))

import design  # noqa: E402

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
        tap=TensorAccessPattern(
            (design.INPUT_GROUP_ELEMS // 2,),
            0,
            [1, 1, 1, _INPUT_WORDS],
            [_INPUT_WORDS, 0, 0, 1],
        ),
        group=group,
        wait=True,
    )


def _drain_outputs(outputs_conses, outputs_buf, group):
    """Drain each group into the N-major [N/8][vec][8] layout."""
    for g in range(design.GROUPS):
        outputs_conses[g].drain(
            outputs_buf,
            tap=TensorAccessPattern(
                (design.MAX_OUTPUT_ELEMS,),
                g * design.COMPUTES_PER_GROUP * design.COLS_TILE *
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
            ),
            group=group,
            wait=True,
        )


def _sequence_static(weights_buf, inputs_buf, config_buf, outputs_buf,
                     weights_prods, inputs_prod, outputs_conses,
                     prenorm=False):
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
        tap=TensorAccessPattern(
            (design.MAX_INPUT_WORDS,),
            0,
            [outer_in, inner_in, 1, _INPUT_WORDS],
            [inner_in * _INPUT_WORDS, _INPUT_WORDS, 0, 1],
        ),
        group=tg,
        wait=False,
    )
    if prenorm:
        # Sum-pass set first; the main loop consumes the second set.
        inputs_prod.fill(
            inputs_buf,
            tap=TensorAccessPattern(
                (design.MAX_INPUT_WORDS,),
                0,
                [outer_in, inner_in, 1, _INPUT_WORDS],
                [inner_in * _INPUT_WORDS, _INPUT_WORDS, 0, 1],
            ),
            group=tg,
            wait=False,
        )

    for g in range(design.GROUPS):
        weights_prods[g].fill(
            weights_buf,
            tap=TensorAccessPattern(
                (design.MAX_WEIGHT_WORDS,),
                g * group_stride_words,
                [outer, inner, _COMBINED, _L1_LO],
                [inner * _L2_WORDS, _L2_WORDS, _L1_LO, 1],
            ),
            group=tg,
            wait=False,
        )

    _drain_outputs(outputs_conses, outputs_buf, tg)
    tg.finish()


def _sequence_dynamic(weights_buf, inputs_buf, config_buf, outputs_buf,
                      weights_prods, inputs_prod, outputs_conses,
                      input0, input1, input2, input3,
                      weight0, weight1, group_stride_words,
                      prenorm=False):
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

    def fill_inputs():
        offset = arith.constant(i32, 0)
        for part in (input0, input1, input2, input3):
            inputs_prod.fill(
                inputs_buf,
                tap=TensorAccessPattern(
                    (design.MAX_INPUT_WORDS,),
                    offset,
                    [as_i64(part), 1, 1, _INPUT_WORDS],
                    [_INPUT_WORDS, 0, 0, 1],
                ),
                group=tg,
                wait=False,
            )
            offset = arith.addi(offset, arith.muli(part, input_step))

    if prenorm:
        # Sum-pass set first; the main loop consumes the second set.
        fill_inputs()
    fill_inputs()

    for g in range(design.GROUPS):
        offset = arith.muli(arith.constant(i32, g), group_stride_words)
        for part in (weight0, weight1):
            weights_prods[g].fill(
                weights_buf,
                tap=TensorAccessPattern(
                    (design.MAX_WEIGHT_WORDS,),
                    offset,
                    [as_i64(part), 1, _COMBINED, _L1_LO],
                    [_L2_WORDS, _L2_WORDS, _L1_LO, 1],
                ),
                group=tg,
                wait=False,
            )
            offset = arith.addi(offset, arith.muli(part, weight_step))

    _drain_outputs(outputs_conses, outputs_buf, tg)
    tg.finish()


def build(runtime_sized, prenorm=False):
    """Return the host sequence used by the C++ caller or Python runner."""
    fn = _sequence_dynamic if runtime_sized else _sequence_static
    return partial(fn, prenorm=True) if prenorm else fn
