#
# Proof-of-concept matrix multiplication.
#
# The multiplication is organized so that each of the NPU's 32 compute tiles
# is responsible for producing 64 elements of each output vector. Input values
# are broadcast to all compute tiles, and each weight is sent to only the tile
# that needs it.
#
# weights: bfloat16 (rows, cols) = (2048, 2048)
# inputs:  float32  (vecs, rows) = (4, 2048)
# outputs: float32  (vecs, cols) = (4, 2048)
#
# The kernel sources are part of the JIT cache key, so editing either .cc file
# invalidates the cached design.
#

from pathlib import Path

import numpy as np
from aie import iron
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import (
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

rows = 2048  # rows of the weight matrix = size of each input vector
cols = 2048  # cols of the weight matrix = size of each output vector
vecs = 4  # number of input/output vectors = concurrency
reps = 64  # repeat the work to amortize launch overhead

groups = 8
computes = 4

l1_rows = 64
l1_cols = 64
l2_cols = computes * l1_cols
row_steps = rows // l1_rows

l1_weights_elems = l1_rows * l1_cols  # bf16 elements per compute tile
l2_weights_elems = computes * l1_weights_elems  # bf16 elements per group

kernel_dir = Path(__file__).resolve().parent
calc_zero_source = kernel_dir / "calc_zero.cc"
calc_mmul_source = kernel_dir / "calc_mmul.cc"
kernel_sources = [calc_zero_source, calc_mmul_source]


def make_zero():
    return ExternalFunction(
        "calc_zero",
        arg_types=[np.ndarray[(vecs, l1_cols), np.dtype[np.float32]]],
        source_file=str(calc_zero_source),
        inline=True,
    )


def make_mmul():
    return ExternalFunction(
        "calc_mmul",
        arg_types=[
            np.ndarray[(l1_weights_elems,), np.dtype[bfloat16]],
            np.ndarray[(vecs, l1_rows), np.dtype[np.float32]],
            np.ndarray[(vecs, l1_cols), np.dtype[np.float32]],
        ],
        source_file=str(calc_mmul_source),
        inline=True,
    )


def _pack_weights(weights):
    """Pack (rows, cols) bf16 weights for one fill per group.

    Each compute gets a contiguous l1_rows x l1_cols tile in
    (rows/8, cols/8, 8, 8) order. The packed order is
    (group, step, compute, rows/8, cols/8, 8, 8).
    """
    packed = np.empty(
        (
            groups,
            row_steps,
            computes,
            8,
            8,
            8,
            8,
        ),
        dtype=bfloat16,
    )
    for group in range(groups):
        for step in range(row_steps):
            for compute in range(computes):
                col_start = group * l2_cols + compute * l1_cols
                tile = weights[
                    step * l1_rows : (step + 1) * l1_rows,
                    col_start : col_start + l1_cols,
                ]
                # (row_tile, row_in, col_tile, col_in)
                #     -> (row_tile, col_tile, row_in, col_in)
                packed[group, step, compute] = tile.reshape(
                    l1_rows // 8, 8, l1_cols // 8, 8
                ).transpose(0, 2, 1, 3)
    return packed.reshape(-1)


@iron.jit(source_files=kernel_sources)
def program(
    inputs: In,
    weights: In,
    outputs: Out,
):
    l1_inputs_type = np.ndarray[(vecs, l1_rows), np.dtype[np.float32]]
    l1_weights_type = np.ndarray[(l1_weights_elems,), np.dtype[bfloat16]]
    l1_outputs_type = np.ndarray[(vecs, l1_cols), np.dtype[np.float32]]
    l2_inputs_type = np.ndarray[(vecs, l1_rows), np.dtype[np.float32]]
    l2_weights_type = np.ndarray[(l2_weights_elems,), np.dtype[bfloat16]]
    l2_outputs_type = np.ndarray[(vecs, l2_cols), np.dtype[np.float32]]

    inputs_type = np.ndarray[(vecs, rows), np.dtype[np.float32]]
    weights_type = np.ndarray[
        (groups * row_steps * l2_weights_elems,),
        np.dtype[bfloat16],
    ]
    outputs_type = np.ndarray[(vecs, cols), np.dtype[np.float32]]

    # The NPU's FPGA heritage indexes hardware tiles by row and column. To
    # avoid confusion with matrix and vector math, we will organize the tiles
    # by group (= FPGA column) and by index for the four compute tiles.

    shim_tiles = [
        Tile(col=group, row=0, tile_type=AIETileType.ShimNOCTile)
        for group in range(groups)
    ]
    mem_tiles = [
        Tile(col=group, row=1, tile_type=AIETileType.MemTile) for group in range(groups)
    ]
    compute_tiles = [
        [
            Tile(col=group, row=2 + compute, tile_type=AIETileType.CoreTile)
            for compute in range(computes)
        ]
        for group in range(groups)
    ]

    # One input fifo, broadcast to every compute tile. Each L2 element is one
    # 64-row input step; the forward re-tiles it as (rows/8, vecs, 8).
    inputs_l2 = ObjectFifo(
        l2_inputs_type,
        consumer_obj_type=l1_inputs_type,
        name="inputs_l2",
    )
    inputs_l1 = inputs_l2.cons().forward(
        tile=mem_tiles[0],
        obj_type=l1_inputs_type,
        name="inputs_l1",
        dims_to_stream=[
            (vecs // 4, 4 * l1_rows),  # vec block
            (l1_rows // 8, 8),  # row tile
            (4, l1_rows),  # vec within block
            (8, 1),  # row within tile
        ],
    )

    # One weight fifo per group, split into pre-tiled fifo for each compute tile.
    weights_l2 = [
        ObjectFifo(l2_weights_type, name=f"weights_{group}") for group in range(groups)
    ]
    weights_l1 = [
        weights_l2[group]
        .cons()
        .split(
            offsets=[compute * l1_weights_elems for compute in range(computes)],
            tile=mem_tiles[group],
            obj_types=[l1_weights_type] * computes,
            names=[f"weights_{group}_{compute}" for compute in range(computes)],
        )
        for group in range(groups)
    ]

    # The output buffer accumulates temporary values tiled as (cols/8, vecs, 8).
    # The join transform un-tiles each into a (vecs, l1_cols) output tile.
    output_dims_to_stream = [
        (vecs // 4, 4 * l1_cols),
        (4, 8),
        (l1_cols // 8, 4 * 8),
        (8, 1),
    ]
    outputs_l2 = [
        ObjectFifo(
            l2_outputs_type,
            name=f"outputs_{group}",
            dims_to_stream=output_dims_to_stream,
        )
        for group in range(groups)
    ]
    outputs_l1 = [
        outputs_l2[group]
        .prod()
        .join(
            offsets=[compute * vecs * l1_cols for compute in range(computes)],
            tile=mem_tiles[group],
            obj_types=[l1_outputs_type] * computes,
            names=[f"outputs_{group}_{compute}" for compute in range(computes)],
        )
        for group in range(groups)
    ]

    def compute_body(
        weights_cons,
        inputs_cons,
        outputs_prod,
        zero,
        mmul,
    ):
        outputs_buf = outputs_prod.acquire(1)
        zero(outputs_buf)

        for _ in range_(row_steps):
            inputs_buf = inputs_cons.acquire(1)
            weights_buf = weights_cons.acquire(1)
            mmul(weights_buf, inputs_buf, outputs_buf)
            weights_cons.release(1)
            inputs_cons.release(1)

        outputs_prod.release(1)

    zero = make_zero()
    mmul = make_mmul()

    workers = [
        Worker(
            compute_body,
            [
                weights_l1[group][compute].cons(),
                inputs_l1.cons(),
                outputs_l1[group][compute].prod(),
                zero,
                mmul,
            ],
            tile=compute_tiles[group][compute],
        )
        for group in range(groups)
        for compute in range(computes)
    ]

    def sequence(
        inputs_buf: iron.RuntimeData,
        weights_buf: iron.RuntimeData,
        outputs_buf: iron.RuntimeData,
        inputs_prod,
        weights_prods,
        outputs_conses,
    ):
        block_elems = row_steps * l2_weights_elems

        for _ in range(reps):
            tg = iron.TaskGroup()

            inputs_prod.fill(
                inputs_buf,
                sizes=[1, row_steps, vecs, l1_rows],
                strides=[0, l1_rows, rows, 1],
                group=tg,
            )

            for group in range(groups):
                weights_prods[group].fill(
                    weights_buf,
                    offset=group * block_elems,
                    sizes=[1, 1, 1, block_elems],
                    group=tg,
                )

                outputs_conses[group].drain(
                    outputs_buf,
                    offset=group * l2_cols,
                    sizes=[computes, vecs, l1_cols],
                    strides=[l1_cols, cols, 1],
                    group=tg,
                    wait=True,
                )

            tg.finish()

    rt = Runtime(
        sequence,
        [
            inputs_type,
            weights_type,
            outputs_type,
            inputs_l2.prod(tile=shim_tiles[0]),
            [weights_l2[group].prod(tile=shim_tiles[group]) for group in range(groups)],
            [outputs_l2[group].cons(tile=shim_tiles[group]) for group in range(groups)],
        ],
    )
    return Program(NPU2(), rt, workers).resolve_program()


def main():
    rng = np.random.default_rng(0)
    inputs_data = rng.integers(0, 8, (vecs, rows), dtype=np.int32).astype(np.float32)
    weights_data = rng.integers(0, 8, (rows, cols), dtype=np.int32).astype(bfloat16)
    packed_weights = _pack_weights(weights_data)

    inputs = iron.tensor(inputs_data, dtype=np.float32)
    weights = iron.tensor(packed_weights, dtype=bfloat16)
    outputs = iron.zeros((vecs, cols), dtype=np.float32)

    bench = run_iters(
        program,
        inputs,
        weights,
        outputs,
        warmup=10,
        iters=20,
    )

    stats = bench.npu if bench.npu is not None else bench.e2e
    avg_seconds = stats.avg_us * 1e-6
    best_seconds = stats.min_us * 1e-6
    weight_bytes = packed_weights.size * 2
    input_bytes = vecs * rows * 4
    output_bytes = vecs * cols * 4
    total_bytes = reps * (weight_bytes + input_bytes + output_bytes)
    flops = reps * 2 * vecs * rows * cols

    print(f"reps: {reps}")
    print(
        f"npu: {stats.avg_us:.1f} us "
        f"(min {stats.min_us:.1f}, max {stats.max_us:.1f}), "
        f"e2e {bench.e2e.avg_us:.1f} us"
    )
    print(
        f"bandwidth: avg {total_bytes / avg_seconds / 1e9:.2f} GB/s, "
        f"best {total_bytes / best_seconds / 1e9:.2f} GB/s"
    )
    print(f"compute: {flops / avg_seconds / 1e9:.2f} GFLOP/s")

    actual = np.asarray(outputs)
    expected = inputs_data @ weights_data.astype(np.float32)
    if not np.array_equal(actual, expected):
        error = np.abs(actual - expected)
        print(f"FAILED: max abs error {error.max():.6f}")
        raise SystemExit(1)
    print("verified against numpy")


if __name__ == "__main__":
    main()
