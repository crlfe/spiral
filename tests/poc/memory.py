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
from aie.iron.device import NPU2, Tile
from aie.utils.benchmark import run_iters


@iron.jit
def program(
    input_buf: In,
    output_buf: Out,
    *,
    num_channels: CompileTime[int],
    input_len: CompileTime[int],
    output_len: CompileTime[int],
):
    # Compute Tile has 64 KB, use 2x 16KB input buffers.
    l1_input_len = 4 * 1024

    # Memory Tile has 512 KB, two channels each get 2x 128 KB buffers.
    l2_input_len = 32 * 1024

    l1_input_type = np.ndarray[(l1_input_len,), np.dtype[np.int32]]
    l2_input_type = np.ndarray[(l2_input_len,), np.dtype[np.int32]]
    l3_input_type = np.ndarray[(input_len,), np.dtype[np.int32]]

    assert input_len % (num_channels * l2_input_len) == 0
    l1_input_reps = input_len // num_channels // l1_input_len
    l2_input_reps = input_len // num_channels // l2_input_len

    assert output_len % num_channels == 0
    l2_output_len = output_len // num_channels
    l1_output_len = l2_output_len
    l1_output_type = np.ndarray[(l1_output_len,), np.dtype[np.int64]]
    l3_output_type = np.ndarray[(output_len,), np.dtype[np.int64]]

    l2_input_fifos = [
        ObjectFifo(
            l2_input_type,
            name=f"l2_input_{i}",
        )
        for i in range(num_channels)
    ]

    # Automatic assignment of L2 input buffers failed,
    # so anchor each to the appropriate memory tile.
    mem_tiles = [Tile(col=g, row=1) for g in range(8)]

    l1_input_fifos = [
        l2_input_fifos[i]
        .cons()
        .forward(
            tile=mem_tiles[i % 8],
            obj_type=l1_input_type,
            name=f"l1_input_{i}",
        )
        for i in range(num_channels)
    ]

    l1_output_fifos = [
        ObjectFifo(
            l1_output_type,
            name=f"output_{i}",
        )
        for i in range(num_channels)
    ]

    compute_zero = ExternalFunction(
        "zero",
        arg_types=[l1_output_type],
        compile_flags=[f"-DL1_OUTPUT_LEN={l1_output_len}"],
        source_string="""
#include <aie_api/aie.hpp>

extern "C" void zero(int64_t * restrict output) {
    for (int32_t i = 0; i < L1_OUTPUT_LEN; i++) {
        output[i] = 0;
    }
}
""",
    )

    compute_sum = ExternalFunction(
        "sum",
        arg_types=[l1_input_type, l1_output_type],
        compile_flags=[f"-DL1_INPUT_LEN={l1_input_len}"],
        source_string="""
#include <aie_api/aie.hpp>

extern "C" void sum(const int32_t * restrict input, int64_t * restrict output) {
    aie::vector<int32_t, 16> accum(0);

    for (int32_t i = 0; i < L1_INPUT_LEN; i += 16) {
        accum = aie::add(accum, aie::load_v<16>(input + i));
    }

    *output += aie::reduce_add(accum);
}
""",
    )

    def compute(
        input_cons: ObjectFifoHandle,
        output_prod: ObjectFifoHandle,
        compute_zero: ExternalFunction,
        compute_sum: ExternalFunction,
        inputs_per_output: CompileTime[int],
    ):
        output_buf = output_prod.acquire(1)
        compute_zero(output_buf)

        for i in range_(inputs_per_output):
            input_buf = input_cons.acquire(1)
            compute_sum(input_buf, output_buf)
            input_cons.release(1)
        output_prod.release(1)

    workers = [
        Worker(
            compute,
            [
                l1_input_fifos[i].cons(),
                l1_output_fifos[i].prod(),
                compute_zero,
                compute_sum,
                l1_input_reps,
            ],
        )
        for i in range(num_channels)
    ]

    def run_sequence(
        input_buf: RuntimeData,
        output_buf: RuntimeData,
        input_prods: list[ObjectFifoHandle],
        output_conses: list[ObjectFifoHandle],
    ):
        for i in range(num_channels):
            input_prods[i].fill(
                input_buf,
                offset=l2_input_reps * l2_input_len * i,
                sizes=[l2_input_reps * l2_input_len],
            )
            output_conses[i].drain(
                output_buf,
                offset=l2_output_len * i,
                sizes=[l2_output_len],
                wait=True,
            )

    rt = Runtime(
        run_sequence,
        [
            l3_input_type,
            l3_output_type,
            [fifo.prod() for fifo in l2_input_fifos],
            [fifo.cons() for fifo in l1_output_fifos],
        ],
    )

    prog = Program(NPU2(), rt, workers)
    return prog.resolve_program()


def _run_and_verify(input_buf, output_buf, *args, **kwargs):
    result = program(input_buf, output_buf, *args, **kwargs)

    expected = np.sum(input_buf, dtype=np.int64)
    actual = np.sum(output_buf)
    if expected != actual:
        err_abs = abs(expected - actual)
        err_pct = (
            100 * err_abs / min(expected, actual)
            if expected > 0 and actual > 0
            else 100
        )
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

    max_channels = 16
    input_len_per_channel = 128 * 1024 * 1024
    output_len_per_channel = 8

    input_buf = iron.randint(
        0, 255, (max_channels * input_len_per_channel,), dtype=np.int32
    )
    output_buf = iron.zeros((max_channels * output_len_per_channel), dtype=np.int64)

    for num_channels in 1, 2, 3, 4, 6, 8, 10, 12, 14, 16:

        bench = run_iters(
            _run_and_verify,
            input_buf.subview(0, (num_channels * input_len_per_channel,)),
            output_buf.subview(0, (num_channels * output_len_per_channel,)),
            num_channels=num_channels,
            input_len=num_channels * input_len_per_channel,
            output_len=num_channels * output_len_per_channel,
            warmup=1,
            iters=5,
        )

        bytes_per_gb = 1024 * 1024 * 1024
        bw_num = (num_channels * input_len_per_channel * 4 / bytes_per_gb) * 1e6
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
