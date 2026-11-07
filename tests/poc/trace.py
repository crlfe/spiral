#
# Minimal AIE event-tracing example.
#
# One worker runs a tiny kernel that brackets its work with event0()/event1().
# The program enables trace with a host TraceConfig, the run writes a raw trace
# file, and trace_to_json() + get_cycles_summary() turn it into per-invocation
# cycle counts. Use this to check the tracing API, not as a benchmark.
#
#     source mlir-aie/iron_env.sh && python tests/poc/trace.py
#

import argparse
import os
import sys

import numpy as np
from aie import iron
from aie.iron import (
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.device import NPU2
from aie.utils.trace import TraceConfig
from aie.utils.trace.utils import get_cycles_summary
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

num_elements = 1024

KERNEL_SOURCE = """
#include <aie_api/aie.hpp>

extern "C" void trace_add(
    const bfloat16 * restrict in,
    bfloat16 * restrict out
) {
    event0();
    for (int r = 0; r < REPEAT; r++) {
        for (int i = 0; i < NUM_ELEMS; i++) {
            out[i] = in[i] + (bfloat16)1.0f;
        }
    }
    event1();
}
"""


@iron.jit
def trace_add_program(
    inp: In,
    out: Out,
    *,
    trace_config: iron.CompileTime[object] = None,
    trace_size: iron.CompileTime[np.int32] = 0,
    repeat: iron.CompileTime[np.int32] = 1,
):
    tile_ty = np.ndarray[(num_elements,), np.dtype[bfloat16]]

    kernel = ExternalFunction(
        "trace_add",
        arg_types=[tile_ty, tile_ty],
        compile_flags=[f"-DNUM_ELEMS={num_elements}", f"-DREPEAT={repeat}"],
        source_string=KERNEL_SOURCE,
    )

    of_in = ObjectFifo(tile_ty, name="in", depth=1)
    of_out = ObjectFifo(tile_ty, name="out", depth=1)

    def core_body(in_cons, out_prod, kernel):
        in_buf = in_cons.acquire(1)
        out_buf = out_prod.acquire(1)
        kernel(in_buf, out_buf)
        in_cons.release(1)
        out_prod.release(1)

    worker = Worker(
        core_body,
        [of_in.cons(), of_out.prod(), kernel],
        trace=1,
    )

    def sequence(A, B, in_prod, out_cons):
        in_prod.fill(A)
        out_cons.drain(B, wait=True)

    rt = Runtime(
        sequence,
        [tile_ty, tile_ty, of_in.prod(), of_out.cons()],
    )

    prog = Program(NPU2(), rt, [worker])
    if trace_size > 0:
        prog.enable_trace(trace_size, workers=[worker])
    return prog.resolve_program()


def main():
    parser = argparse.ArgumentParser(description="Minimal AIE trace example.")
    parser.add_argument("--trace-size", type=int, default=65536)
    parser.add_argument("--trace-file", default="/tmp/poc_trace.txt")
    parser.add_argument("--trace-json", default="/tmp/poc_trace.json")
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    inp_data = np.ones(num_elements, dtype=bfloat16)
    inp = iron.tensor(inp_data, dtype=bfloat16)
    out = iron.zeros((num_elements,), dtype=bfloat16)

    trace_config = TraceConfig(
        trace_size=args.trace_size,
        trace_file=args.trace_file,
    )

    for _ in range(args.iters):
        trace_add_program(
            inp,
            out,
            trace_size=args.trace_size,
            trace_config=trace_config,
        )

    trace_config.trace_to_json(trace_config.physical_mlir_path, args.trace_json)

    print(f"raw trace:  {args.trace_file}")
    print(f"json trace: {args.trace_json}")
    print(f"lowered MLIR: {trace_config.physical_mlir_path}")

    cycles = get_cycles_summary(args.trace_json)
    for entry in cycles:
        print(f"tile {entry[0]}: {len(entry) - 1} event0->event1 intervals")
        if len(entry) > 1:
            samples = entry[1:]
            print(
                f"  first/min/avg/max cycles: {samples[0]}/ {min(samples)}/ "
                f"{sum(samples) / len(samples):.1f}/ {max(samples)}"
            )

    actual = np.asarray(out)
    expected = inp_data + bfloat16(1.0)
    print("verified" if np.array_equal(actual, expected) else "FAILED")


if __name__ == "__main__":
    main()
