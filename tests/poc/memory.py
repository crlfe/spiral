#
# Proof-of-concept bulk memory transfer benchmark.
#

import numpy as np
from aie import iron
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    RuntimeData,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifoHandle
from aie.iron.device import NPU2
from aie.utils.benchmark import run_iters

# TODO: Figure out a less confusing terminology for whole-buffer/block/tile.
# Xilinx/AMD use L3 (Host), L2 (Memory Tile), and L1 (Compute Tile), which
# carries the unfortunate suggestion that they are cache levels but is at least
# simple to stick in variable names (see flow.py).
#
# For the moment: The input is divided into `num_channels` blocks of length
# `input_block_len`, each of which is run through a distinct pipeline of one
# Shim MM2S DMA, one Compute Tile, and one Shim S2MM DMA. The `input_block_len`
# int32 items processed by each Compute Tile are further divided into groups
# of length `input_tile_len`. When all of the input has been processed, the
# Compute Tile sends the final sum as the first value in a block of length
# `output_block_len`.


@iron.jit
def program(
    input: In,
    output: Out,
    *,
    func_zero: CompileTime[ExternalFunction],
    func_sum: CompileTime[ExternalFunction],
    input_len: CompileTime[int],
    num_channels: CompileTime[int],
    input_tile_len: CompileTime[int],
    output_block_len: CompileTime[int],
):
    assert input_len % num_channels == 0
    input_block_len = input_len // num_channels

    assert input_block_len % input_tile_len == 0
    input_tiles_per_block = input_block_len // input_tile_len

    input_type = np.ndarray[(input_len,), np.dtype[np.int32]]
    input_block_type = np.ndarray[(input_block_len,), np.dtype[np.int32]]
    input_tile_type = np.ndarray[(input_tile_len,), np.dtype[np.int32]]
    output_tile_type = np.ndarray[(1,), np.dtype[np.int64]]

    input_fifos = [
        ObjectFifo(input_block_type, consumer_obj_type=input_tile_type)
        for _ in range(num_channels)
    ]

    output_type = np.ndarray[(num_channels, output_block_len), np.dtype[np.int64]]

    output_fifos = [ObjectFifo(output_tile_type) for _ in range(num_channels)]

    def run_core(
        input_cons: ObjectFifoHandle,
        output_prod: ObjectFifoHandle,
        func_zero: ExternalFunction,
        func_sum: ExternalFunction,
        input_tiles_per_block: CompileTime[int],
    ):
        output_buf = output_prod.acquire(1)
        func_zero(output_buf)

        for i in range_(input_tiles_per_block):
            input_buf = input_cons.acquire(1)
            func_sum(input_buf, output_buf)
            input_cons.release(1)
        output_prod.release(1)

    workers = [
        Worker(
            run_core,
            [
                input_fifos[i].cons(),
                output_fifos[i].prod(),
                func_zero,
                func_sum,
                input_tiles_per_block,
            ],
        )
        for i in range(num_channels)
    ]

    def run_sequence(
        input: RuntimeData,
        output: RuntimeData,
        input_prods: list[ObjectFifoHandle],
        output_conses: list[ObjectFifoHandle],
    ):
        for i, input_prod in enumerate(input_prods):
            input_prod.fill(input, offset=input_block_len * i, sizes=[input_block_len])

        for i, output_cons in enumerate(output_conses):
            output_cons.drain(output, offset=output_block_len * i, sizes=[1], wait=True)

    rt = Runtime(
        run_sequence,
        [
            input_type,
            output_type,
            [fifo.prod() for fifo in input_fifos],
            [fifo.cons() for fifo in output_fifos],
        ],
    )

    prog = Program(NPU2(), rt, workers)
    return prog.resolve_program()


def _run_and_verify(input, output, *args, **kwargs):
    result = program(input, output, *args, **kwargs)

    expected = np.sum(input, dtype=np.int64)
    actual = np.sum(output)
    if expected != actual:
        err_abs = abs(expected - actual)
        err_pct = 100 * err_abs / min(expected, actual)
        print(
            "  output validation failed:",
            f"got {actual}, ",
            f"expected {expected},",
            f"diff {err_abs} ({err_pct}%)",
        )

    return result


def main():
    # It may take quite a while before the first program is compiled and run,
    # so print a message just to tell the user we are doing something.
    print("Loading...")

    input_tile_len = 1024
    input_tile_type = np.ndarray[(input_tile_len,), np.dtype[np.int32]]

    output_block_len = 8
    output_tile_type = np.ndarray[(1,), np.dtype[np.int64]]

    max_channels = 16
    per_channel_input_len = 64 * 1024 * 1024
    max_input_len = max_channels * per_channel_input_len
    max_input = iron.randint(0, 255, (max_input_len,), dtype=np.int32)
    output = iron.zeros((max_channels, output_block_len), dtype=np.int64)

    def make_zero():
        return ExternalFunction(
            "zero",
            arg_types=[output_tile_type],
            source_string="""
#include <aie_api/aie.hpp>

extern "C" void zero(int64_t * restrict output) {
    *output = 0;
}
""",
        )

    def make_sum():
        return ExternalFunction(
            "sum",
            arg_types=[input_tile_type, output_tile_type],
            compile_flags=[f"-DINPUT_TILE_LEN={input_tile_len}"],
            source_string="""
#include <aie_api/aie.hpp>

extern "C" void sum(const int32_t * restrict input, int64_t * restrict output) {
    aie::vector<int32_t, 16> accum(0);

    for (int32_t i = 0; i < INPUT_TILE_LEN; i += 16) {
        accum = accum + aie::load_v<16>(input + i);
    }

    *output += aie::reduce_add(accum);
}
""",
        )

    for num_channels in 1, 2, 3, 4, 6, 8, 10, 12, 14, 16:
        input_len = num_channels * per_channel_input_len
        input = max_input.subview(0, (input_len,))

        bench = run_iters(
            _run_and_verify,
            input,
            output.subview(0, (num_channels, output_block_len)),
            func_zero=make_zero(),
            func_sum=make_sum(),
            input_len=input_len,
            num_channels=num_channels,
            input_tile_len=input_tile_len,
            output_block_len=output_block_len,
            warmup=1,
            iters=5,
        )

        bytes_per_gb = 1024 * 1024 * 1024
        bw_num = (input_len * 4 / bytes_per_gb) * 1e6
        avg_bw = bw_num / bench.npu.avg_us
        min_bw = bw_num / bench.npu.max_us
        max_bw = bw_num / bench.npu.min_us
        print(
            f"Tested {num_channels:>2d} channels:",
            f"avg {avg_bw:.1f} GB/s",
            f"  (min {min_bw:.1f} to",
            f"max {max_bw:.1f})",
        )


if __name__ == "__main__":
    main()
