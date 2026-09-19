#
# Proof-of-concept matrix memory transfer benchmark.
#
# This follows the shape of a plausible matrix multiplication without
# performing the actual calculation.
#
# TODO: Add Compute Tile code to verify that we are seeing all the values and
# weights. Or just press on with the actual implementation?
#
# TODO: Use the hardware address generation to re-order values to/from the flat
# main memory representation into the smaller blocks used by the npu tiles. For
# the moment we are simply verifying that the right number of elements pass
# through every part of the processing.
#


import aie.iron.device
import numpy as np
from aie import iron
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron.controlflow import range_
from aie.utils.benchmark import run_iters
from ml_dtypes import bfloat16


class ComputeConfig:
    rows = 64
    cols = 64
    vecs = 16

    def make_kernel(this):
        return iron.ExternalFunction(
            "flow_kernel",
            arg_types=[
                np.ndarray[(this.rows, this.cols), np.dtype[bfloat16]],
                np.ndarray[(this.vecs, this.rows), np.dtype[bfloat16]],
                np.ndarray[(this.vecs, this.cols), np.dtype[bfloat16]],
            ],
            compile_flags=[
                f"-DSTEP_ROWS={this.rows}",
                f"-DSTEP_COLS={this.cols}",
                f"-DSTEP_VECS={this.vecs}",
            ],
            source_string="""
    #include <aie_api/aie.hpp>
    #include <algorithm>

    extern "C" void flow_kernel(
        const bfloat16 * restrict weights,
        const bfloat16 * restrict inputs,
        bfloat16 * restrict outputs
    ) {
        // TODO: Verify that we have actually seen all the weights and inputs.
    }
    """,
        )


@iron.jit
def program(
    inputs: iron.In,
    weights: iron.In,
    outputs: iron.Out,
    *,
    rows: iron.CompileTime[np.int32],
    cols: iron.CompileTime[np.int32],
    vecs: iron.CompileTime[np.int32],
    reps: iron.CompileTime[np.int32],
):
    num_groups = 8
    num_computes_per_group = 4

    mem_tiles = [
        aie.iron.device.Tile(col=g, row=1, tile_type=AIETileType.MemTile)
        for g in range(num_groups)
    ]
    comp_tiles = [
        [
            aie.iron.device.Tile(col=g, row=2 + i, tile_type=AIETileType.CoreTile)
            for i in range(num_computes_per_group)
        ]
        for g in range(num_groups)
    ]

    l1_config = ComputeConfig()

    # Data in the memory tiles
    l2_rows = l1_config.rows
    l2_cols = l1_config.cols * num_computes_per_group
    l2_vecs = l1_config.vecs
    l2_weights_type = np.ndarray[(l2_rows, l2_cols), np.dtype[bfloat16]]
    l2_inputs_type = np.ndarray[(l2_vecs, l2_rows), np.dtype[bfloat16]]
    l2_outputs_type = np.ndarray[(l2_vecs, l2_cols), np.dtype[bfloat16]]

    # Data in the compute tiles
    l1_weights_type = np.ndarray[(l1_config.rows, l1_config.cols), np.dtype[bfloat16]]
    l1_inputs_type = np.ndarray[(l1_config.vecs, l1_config.rows), np.dtype[bfloat16]]
    l1_outputs_type = np.ndarray[(l1_config.vecs, l1_config.cols), np.dtype[bfloat16]]

    # Weights are streamed concurrently from main memory to each Memory Tile,
    # then split across their associated Compute Tiles.
    l3_l2_weights_fifos = [
        iron.ObjectFifo(l2_weights_type, name=f"weights_m{g}")
        for g in range(num_groups)
    ]
    l2_l1_weights_fifos = [
        l3_l2_weights_fifos[g]
        .cons()
        .split(
            offsets=[
                l1_config.rows * l1_config.cols * i
                for i in range(num_computes_per_group)
            ],
            tile=mem_tiles[g],
            obj_types=[l1_weights_type] * num_computes_per_group,
        )
        for g in range(num_groups)
    ]

    # Inputs are streamed from main memory to the first Memory Tile, and then
    # broadcast to every Compute Tile.
    l3_l2_inputs_fifo = iron.ObjectFifo(
        l2_inputs_type,
        consumer_obj_type=l1_inputs_type,
        name="inputs",
    )
    l2_l1_inputs_fifo = l3_l2_inputs_fifo.cons().forward(
        tile=mem_tiles[0],
        obj_type=l1_inputs_type,
    )

    # Compute Tiles accumulate their outputs internally until the end of the
    # matrix multiplication. They finally send all of their outputs to their
    # associated Memory Tile, which buffers the outputs for transfer to main
    # memory.
    l2_l3_outputs_fifos = [
        iron.ObjectFifo(l2_outputs_type, name=f"outputs_m{g}")
        for g in range(num_groups)
    ]
    l1_l2_outputs_fifos = [
        l2_l3_outputs_fifos[g]
        .prod()
        .join(
            tile=mem_tiles[g],
            offsets=[
                l1_config.vecs * l1_config.cols * i
                for i in range(num_computes_per_group)
            ],
            #            depths=[8] * num_computes_per_group,
            obj_types=[l1_outputs_type] * num_computes_per_group,
        )
        for g in range(num_groups)
    ]

    def compute(
        weights_cons: iron.dataflow.ObjectFifoHandle,
        inputs_cons: iron.dataflow.ObjectFifoHandle,
        outputs_prod: iron.dataflow.ObjectFifoHandle,
        kernel: iron.ExternalFunction,
        row_loops: iron.CompileTime[np.int32],
        col_loops: iron.CompileTime[np.int32],
    ):
        outputs = outputs_prod.acquire(col_loops)
        for _ in range_(row_loops):
            inputs = inputs_cons.acquire(1)

            # TODO: This range has to be compile-time because outputs[i] can
            # not be evaluated by the runtime environment. Either this example
            # or a future one will support runtime arguments, probably by
            # writing outputs into a max-sized static buffer that is then
            # partially streamed out in col_loops pieces.
            for i in range(col_loops):
                weights = weights_cons.acquire(1)
                kernel(weights, inputs, outputs[i])
                weights_cons.release(1)
            inputs_cons.release(1)
        outputs_prod.release(col_loops)

    kernel = l1_config.make_kernel()
    row_loops = rows // l1_config.rows
    col_loops = cols // (l1_config.cols * num_computes_per_group * num_groups)
    workers = [
        iron.Worker(
            compute,
            [
                l2_l1_weights_fifos[g][i].cons(),
                l2_l1_inputs_fifo.cons(),
                l1_l2_outputs_fifos[g][i].prod(depth=8),
                kernel,
                row_loops,
                col_loops,
            ],
            tile=comp_tiles[g][i],
        )
        for i in range(num_computes_per_group)
        for g in range(num_groups)
    ]

    # These are easier to compute outside the sequence due to type errors.
    weights_per_group = rows * cols // num_groups
    inputs_len = vecs * rows
    outputs_per_group = vecs * cols // num_groups

    def sequence(
        weights: iron.RuntimeData,
        inputs: iron.RuntimeData,
        outputs: iron.RuntimeData,
        weights_prod: list[iron.dataflow.ObjectFifoHandle],
        inputs_prod: iron.dataflow.ObjectFifoHandle,
        outputs_cons: list[iron.dataflow.ObjectFifoHandle],
        rows: iron.CompileTime[np.int32],
        cols: iron.CompileTime[np.int32],
        vecs: iron.CompileTime[np.int32],
        reps: iron.CompileTime[np.int32],
    ):
        for _ in range_(reps):
            tg = iron.TaskGroup()
            inputs_prod.fill(
                inputs,
                offset=0,
                sizes=[1, 1, 1, inputs_len],
                group=tg,
            )
            for g in range(num_groups):
                weights_prod[g].fill(
                    weights,
                    offset=weights_per_group * g,
                    sizes=[1, 1, 1, weights_per_group],
                    group=tg,
                )
                outputs_cons[g].drain(
                    outputs,
                    offset=outputs_per_group * g,
                    sizes=[1, 1, 1, outputs_per_group],
                    group=tg,
                    wait=True,
                )
            tg.finish()

    rt = iron.Runtime(
        sequence,
        [
            np.ndarray[(rows, cols), np.dtype[bfloat16]],
            np.ndarray[(vecs, rows), np.dtype[bfloat16]],
            np.ndarray[(vecs, cols), np.dtype[bfloat16]],
            [l3_l2_weights_fifos[g].prod() for g in range(num_groups)],
            l3_l2_inputs_fifo.prod(),
            [l2_l3_outputs_fifos[g].cons() for g in range(num_groups)],
            rows,
            cols,
            vecs,
            reps,
        ],
    )

    prog = iron.Program(aie.iron.device.NPU2(), rt, workers)
    return prog.resolve_program()


def main():
    rows = 16 * 1024
    cols = 16 * 1024
    vecs = 16
    reps = 64

    weights = iron.rand((rows, cols), dtype=bfloat16)
    inputs = iron.rand((vecs, rows), dtype=bfloat16)
    outputs = iron.zeros((vecs, cols), dtype=bfloat16)

    bench = run_iters(
        program,
        weights,
        inputs,
        outputs,
        rows=rows,
        cols=cols,
        vecs=vecs,
        reps=reps,
        warmup=1,
        iters=20,
    )

    total_bytes = reps * (rows * cols + vecs * rows + vecs * cols) * 2
    bytes_per_gb = 1024**3
    bw_numerator = (total_bytes / bytes_per_gb) * 1e6
    avg_bw = bw_numerator / bench.npu.avg_us
    min_bw = bw_numerator / bench.npu.max_us
    max_bw = bw_numerator / bench.npu.min_us
    print(
        f"Transfer: avg {avg_bw:.1f} GB/s",
        f"  (min {min_bw:.1f} to",
        f"max {max_bw:.1f})",
    )


if __name__ == "__main__":
    main()
