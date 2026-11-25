"""Static NPU configuration: tiles, object FIFOs, buffers and kernels.

Nothing here depends on the runtime sequence or the worker program.  The
composition root (`run.py`) builds a `Device` once and hands it to both.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import Buffer, ExternalFunction, ObjectFifo
from aie.iron.device import Tile

PROGRAM_DIR = Path(__file__).resolve().parent
if str(PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_DIR))

import design  # noqa: E402

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
PRENORM_SUM_SOURCE = KERNEL_DIR / "program_prenorm_sum.aie.cc"
PRENORM_FINISH_SOURCE = KERNEL_DIR / "program_prenorm_finish.aie.cc"
PRENORM_SCALE_SOURCE = KERNEL_DIR / "program_prenorm_scale.aie.cc"
PRENORM_LOAD_SOURCE = KERNEL_DIR / "program_prenorm_load.aie.cc"
PRENORM_IMPL_HEADER = KERNEL_DIR / "program_prenorm_impl.h"
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
# Kernel sources and headers are part of the JIT recipe: their content is
# hashed with the Python sources, so a kernel edit invalidates the cache.
KERNEL_SOURCE_FILES = [
    BEGIN_SOURCE,
    ZERO_SOURCE,
    OUTPUT_SOURCE,
    BLOCK_SOURCE,
    Q4_0_BLOCK_SOURCE,
    Q6_K_BLOCK_SOURCE,
    PRENORM_SUM_SOURCE,
    PRENORM_FINISH_SOURCE,
    PRENORM_SCALE_SOURCE,
    PRENORM_LOAD_SOURCE,
    PRENORM_IMPL_HEADER,
    QUANTS_ROOT / "funcs/rmsnorm/rmsnorm.aie.h",
    QUANTS_ROOT / "funcs/rmsnorm/layout.h",
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
    prenorm_sum: object
    prenorm_finish: object
    prenorm_scale: object
    prenorm_load: object = None
    assemble: object = None
    probe: object = None
    raw_check: object = None


class ComputeBuffers(NamedTuple):
    """Per-compute L1 buffers, indexed by g, c."""

    carry: list
    state: list
    config: list
    accum: list
    prenorm: list
    prenorm_weight: list


class Device(NamedTuple):
    """The fixed part of the design: placement, channels, buffers, kernels."""

    shim_tiles: list
    mem_tiles: list
    compute_tiles: list
    fifos: Fifos
    buffers: ComputeBuffers
    kernels: Kernels


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
            np.int32,
        ],
        compile_flags=[
            f"-DPROGRAM_OUTPUT_TILE_ELEMS={design.OUTPUT_TILE_ELEMS}",
            f"-DPROGRAM_COLS_TILE={design.COLS_TILE}",
            f"-DPROGRAM_EPILOGUE_NONE={design.EPILOGUE_NONE}",
            f"-DPROGRAM_EPILOGUE_SOFTCAP={design.EPILOGUE_SOFTCAP}",
            f"-DPROGRAM_EPILOGUE_GEGLU={design.EPILOGUE_GEGLU}",
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
    # The real prelude is production-only: the diagnostic build's probe
    # kernels already fill the tile program memory.
    if production:
        prenorm_sum = ExternalFunction(
            "program_prenorm_sum",
            arg_types=[
                design.l1_input_group_type,
                np.int32,
                np.int32,
                design.l1_prenorm_state_type,
            ],
            include_dirs=[str(QUANTS_ROOT), str(KERNEL_DIR)],
            compile_flags=[
                f"-DPROGRAM_PRENORM_WEIGHT_OFFSET_BYTES="
                f"{design.PRENORM_WEIGHT_OFFSET_BYTES}",
                "-Os",
            ],
            source_file=str(PRENORM_SUM_SOURCE),
            inline=False,
        )
        prenorm_finish = ExternalFunction(
            "program_prenorm_finish",
            arg_types=[
                design.l1_prenorm_state_type,
                np.int32,
                np.int32,
            ],
            include_dirs=[str(QUANTS_ROOT), str(KERNEL_DIR)],
            compile_flags=[
                f"-DPROGRAM_PRENORM_WEIGHT_OFFSET_BYTES="
                f"{design.PRENORM_WEIGHT_OFFSET_BYTES}",
                "-Oz",
            ],
            source_file=str(PRENORM_FINISH_SOURCE),
            inline=False,
        )
        prenorm_load = ExternalFunction(
            "program_prenorm_load",
            arg_types=[
                design.l1_input_group_type,
                design.l1_prenorm_weight_type,
                np.int32,
            ],
            include_dirs=[str(QUANTS_ROOT), str(KERNEL_DIR)],
            compile_flags=[
                f"-DPROGRAM_PRENORM_WEIGHT_OFFSET_BYTES="
                f"{design.PRENORM_WEIGHT_OFFSET_BYTES}",
                "-Oz",
            ],
            source_file=str(PRENORM_LOAD_SOURCE),
            inline=False,
        )
        prenorm_scale = ExternalFunction(
            "program_prenorm_scale",
            arg_types=[
                design.l1_input_group_type,
                np.int32,
                np.int32,
                design.l1_prenorm_weight_type,
                design.l1_prenorm_state_type,
            ],
            include_dirs=[str(QUANTS_ROOT), str(KERNEL_DIR)],
            compile_flags=[
                f"-DPROGRAM_PRENORM_WEIGHT_OFFSET_BYTES="
                f"{design.PRENORM_WEIGHT_OFFSET_BYTES}",
                "-Os",
            ],
            source_file=str(PRENORM_SCALE_SOURCE),
            inline=False,
        )
    else:
        # Placeholders keep the worker argument list stable; the body
        # never calls them in the diagnostic build.
        prenorm_sum = prenorm_finish = prenorm_scale = zero
        prenorm_load = zero
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
        prenorm_sum=prenorm_sum,
        prenorm_finish=prenorm_finish,
        prenorm_scale=prenorm_scale,
        prenorm_load=prenorm_load,
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
        depth=1,
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
        prenorm=alloc(design.l1_prenorm_state_type, "prenorm"),
        prenorm_weight=alloc(design.l1_prenorm_weight_type, "prenorm_weight"),
    )


def worker_kernel_list(kernels, production):
    """Order the worker's kernels; diagnostic builds append the probes."""
    common = [
        kernels.begin,
        kernels.zero,
        kernels.output,
        kernels.block_begin,
        kernels.q4_block,
        kernels.q6_block,
        kernels.prenorm_sum,
        kernels.prenorm_finish,
        kernels.prenorm_scale,
        kernels.prenorm_load,
    ]
    if production:
        return common
    return common + [kernels.assemble, kernels.probe, kernels.raw_check]


def build(production=False):
    """Build the static device configuration."""
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
    return Device(
        shim_tiles=shim_tiles,
        mem_tiles=mem_tiles,
        compute_tiles=compute_tiles,
        fifos=_make_fifos(mem_tiles),
        buffers=_make_compute_buffers(),
        kernels=_make_kernels(production),
    )


def runtime_args(device, dynamic):
    """Runtime sequence arguments: buffer types and shim-side FIFO handles."""
    args = [
        design.l3_weights_type,
        design.l3_inputs_type,
        design.l3_config_type,
        design.l3_outputs_type,
        [
            device.fifos.weights_l2[g].prod(tile=device.shim_tiles[g])
            for g in range(design.GROUPS)
        ],
        device.fifos.inputs_l2.prod(tile=device.shim_tiles[0]),
        [
            device.fifos.outputs_l2[g].cons(tile=device.shim_tiles[g])
            for g in range(design.GROUPS)
        ],
    ]
    if dynamic:
        # input parts x4, weight parts x2, group stride.
        args += [np.int32] * 7
    return args
