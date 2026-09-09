import aie.iron as iron
import numpy as np
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
from aie.iron.kernels import reduce_add
from aie.iron.device import NPU2
from aie.utils.benchmark import run_iters, print_benchmark


@iron.jit
def program(
    input: In,
    output: Out,
    *,
    input_len: CompileTime[int],
    input_tile_len: CompileTime[int],
    num_channels: CompileTime[int],
):
    assert input_len % num_channels == 0
    input_block_len = input_len // num_channels

    assert input_block_len % input_tile_len == 0
    input_tiles_per_block = input_block_len // input_tile_len

    input_type = np.ndarray[(input_len,), np.dtype[np.int32]]
    input_block_type = np.ndarray[(input_block_len,), np.dtype[np.int32]]
    input_tile_type = np.ndarray[(input_tile_len,), np.dtype[np.int32]]
    input_fifos = [
        ObjectFifo(input_block_type, consumer_obj_type=input_tile_type)
        for _ in range(num_channels)
    ]

    output_type = np.ndarray[(num_channels,), np.dtype[np.int32]]
    output_block_type = np.ndarray[(1,), np.dtype[np.int32]]

    output_fifos = [ObjectFifo(output_block_type) for _ in range(num_channels)]

    def run_core(
        input_cons: ObjectFifoHandle,
        output_prod: ObjectFifoHandle,
        kernel: ExternalFunction,
        input_tile_len: CompileTime[int],
        input_tiles_per_block: CompileTime[int],
    ):
        output_buf = output_prod.acquire(1)
        for _ in range_(input_tiles_per_block):
            input_buf = input_cons.acquire(1)
            # TODO: Replace reduce_add with an inline kernel that accumulates.
            kernel(input_buf, output_buf, input_tile_len)
            input_cons.release(1)
        output_prod.release(1)

    workers = [
        Worker(
            run_core,
            [
                input_fifos[i].cons(),
                output_fifos[i].prod(),
                reduce_add(input_tile_len),
                input_tile_len,
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
            output_cons.drain(output, offset=i, sizes=[1], wait=True)

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


def main():
    print("Loading...")
    input_len = 1024 * 1024 * 1024
    input = iron.randint(0, 64, (input_len,), dtype=np.int32, device="npu")

    for num_channels in [1, 2, 4, 8, 16]:
        output = iron.zeros(num_channels, dtype=np.int32, device="npu")
        bench = run_iters(
            program,
            input,
            output,
            input_len=input_len,
            input_tile_len=4096,
            num_channels=num_channels,
            warmup=1,
            iters=20,
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
