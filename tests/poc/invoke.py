#
# Proof-of-concept dynamic memory transfer benchmark.
#
# The default path compiles the step counts into the design.  The EmitC
# dispatch variants keep one compiled design and pass the step counts per call:
#
#   python tests/poc/invoke.py --emitc       # runtime sizes, static body
#   python tests/poc/invoke.py --emitc-loop  # runtime sizes and a host loop
#
# The loop variant exercises the dynamic BD pool.  It must use explicit
# TaskGroups inside the loop and await each transfer; otherwise the compiler
# warns that BD ids can be recycled while a DMA is still in flight.
#

import itertools
import os
import re
import subprocess
import sys
import sysconfig

import aie.dialects.arith as arith
import numpy as np
from aie import iron
from aie.dialects._arith_enum_gen import CmpIPredicate
from aie.helpers.dialects.scf import if_
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import IndexType, IntegerType
from aie.iron import (
    DispatchTime,
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    RuntimeData,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifoHandle
from aie.iron.device import NPU2
from aie.utils.benchmark import run_iters
from aie.utils.compile import compile_external_kernels, compile_mlir_module

num_groups = 8
num_computes_per_group = 4

# Inputs are split across each worker

l1_inputs_len = 64
l2_inputs_len = l1_inputs_len * num_computes_per_group
l3_inputs_len = l2_inputs_len * num_groups

inputs_dtype = np.int32
l1_inputs_type = np.ndarray[(l1_inputs_len,), np.dtype[inputs_dtype]]
l2_inputs_type = np.ndarray[(l2_inputs_len,), np.dtype[inputs_dtype]]
l3_inputs_type = np.ndarray[(l3_inputs_len,), np.dtype[inputs_dtype]]

# Outputs are concatenated from every worker

l1_outputs_len = 64
l2_outputs_len = l1_outputs_len * num_computes_per_group
l3_outputs_len = l2_outputs_len * num_groups

outputs_dtype = np.int32
l1_outputs_type = np.ndarray[(l1_outputs_len,), np.dtype[outputs_dtype]]
l2_outputs_type = np.ndarray[(l2_outputs_len,), np.dtype[outputs_dtype]]
l3_outputs_type = np.ndarray[(l3_outputs_len,), np.dtype[outputs_dtype]]

# The EmitC dispatch variant keeps one compiled design and passes the step
# counts at run time.  Its tensor types are fixed at the maximum steps the
# benchmark can request; the sequence transfers only the live prefix.
DISPATCH_MAX_INPUT_STEPS = 64
DISPATCH_MAX_OUTPUT_STEPS = 2
l3_inputs_max_type = np.ndarray[
    (l3_inputs_len * DISPATCH_MAX_INPUT_STEPS,), np.dtype[inputs_dtype]
]
l3_outputs_max_type = np.ndarray[
    (l3_outputs_len * DISPATCH_MAX_OUTPUT_STEPS,), np.dtype[outputs_dtype]
]


# inputs_len is the number of input elements one worker consumes per output
# step; outputs_len is the total output volume over every worker and step.
def shape_to_steps(inputs_len, outputs_len):
    assert inputs_len % l1_inputs_len == 0
    assert outputs_len % l3_outputs_len == 0
    return inputs_len // l1_inputs_len, outputs_len // l3_outputs_len


# Build the correctly-sized runtime buffers for a given (inputs_steps, outputs_steps).
def make_args(inputs_steps, outputs_steps):
    # The config travels as one l2_inputs_len-sized buffer, so after the split
    # worker i sees its own pair of slots at [i * l1_inputs_len] and
    # [i * l1_inputs_len + 1].
    config = iron.zeros(l2_inputs_len, dtype=inputs_dtype)
    for i in range(num_computes_per_group):
        config[i * l1_inputs_len + 0] = inputs_steps
        config[i * l1_inputs_len + 1] = outputs_steps
    inputs = iron.arange(l3_inputs_len * inputs_steps, dtype=inputs_dtype)
    outputs = iron.zeros(l3_outputs_len * outputs_steps, dtype=outputs_dtype)
    return config, inputs, outputs


def make_dispatch_args(inputs_steps, outputs_steps):
    """Fixed-max buffers for the DispatchTime variant.

    The design is compiled once with the maximum shapes; the live step counts
    travel in the config object and in the runtime sequence's DMA sizes.
    """
    if not (1 <= inputs_steps <= DISPATCH_MAX_INPUT_STEPS):
        raise ValueError("inputs_steps outside the dispatch maximum")
    if not (1 <= outputs_steps <= DISPATCH_MAX_OUTPUT_STEPS):
        raise ValueError("outputs_steps outside the dispatch maximum")
    config = iron.zeros(l2_inputs_len, dtype=inputs_dtype)
    for i in range(num_computes_per_group):
        config[i * l1_inputs_len + 0] = inputs_steps
        config[i * l1_inputs_len + 1] = outputs_steps
        config[i * l1_inputs_len + 2] = 0
    # The unused tail is harmless; the tap only transfers the live prefix.
    inputs = iron.arange(
        l3_inputs_len * DISPATCH_MAX_INPUT_STEPS, dtype=inputs_dtype
    )
    outputs = iron.zeros(
        l3_outputs_len * DISPATCH_MAX_OUTPUT_STEPS, dtype=outputs_dtype
    )
    return config, inputs, outputs


def make_dispatch_if_args(inputs_steps, outputs_steps, extra_groups):
    config, inputs, outputs = make_dispatch_args(inputs_steps, outputs_steps)
    for i in range(num_computes_per_group):
        config[i * l1_inputs_len + 2] = extra_groups
    return config, inputs, outputs


def make_compute_init():
    return ExternalFunction(
        "compute_init",
        arg_types=[l1_outputs_type],
        inline=True,
        source_string="""
#include <aie_api/aie.hpp>

extern "C" void compute_init(int32_t * restrict outputs) {
    constexpr int32_t N = 16;

    aie::vector<int32_t, N> init = aie::broadcast<int32_t, N>(0);
    for (int32_t i = 0; i < 64; i += N) {
        aie::store_v(outputs + i, init);
    }
}
""",
    )


def make_compute_step():
    return ExternalFunction(
        "compute_step",
        arg_types=[l1_inputs_type, l1_outputs_type],
        inline=True,
        source_string="""
#include <aie_api/aie.hpp>

extern "C" void compute_step(
        const int32_t * restrict inputs,
        int32_t * restrict outputs)
{
    constexpr int32_t N = 16;

    for (int32_t i = 0; i < 64; i += N) {
        aie::store_v(outputs + i, aie::add(
            aie::load_v<N>(outputs + i),
            aie::load_v<N>(inputs + i)
        ));
    }
}
""",
    )


def compute(
    inputs_cons: ObjectFifoHandle,
    outputs_prod: ObjectFifoHandle,
    compute_init: ExternalFunction,
    compute_step: ExternalFunction,
):
    # The config arrives as the first buffer on the inputs FIFO.
    config_buf = inputs_cons.acquire(1)
    inputs_steps = config_buf[0]
    outputs_steps = config_buf[1]
    extra_groups = config_buf[2]
    inputs_cons.release(1)

    for _ in range_(outputs_steps):
        outputs_buf = outputs_prod.acquire(1)
        compute_init(outputs_buf)
        for _ in range_(inputs_steps):
            inputs_buf = inputs_cons.acquire(1)
            compute_step(inputs_buf, outputs_buf)
            inputs_cons.release(1)
        outputs_prod.release(1)

    # Optional runtime-if probe: the host conditionally fills one extra input
    # group and the worker drains it.  Normal calls leave this at zero.
    for _ in range_(extra_groups):
        inputs_buf = inputs_cons.acquire(1)
        inputs_cons.release(1)


def sequence(
    config: RuntimeData,
    inputs: RuntimeData,
    outputs: RuntimeData,
    inputs_steps: np.int32,
    outputs_steps: np.int32,
    inputs_prods: list[ObjectFifoHandle],
    outputs_conses: list[ObjectFifoHandle],
):
    int64_type = IntegerType.get_signless(64)
    inputs_steps_64 = arith.extsi(int64_type, inputs_steps)
    outputs_steps_64 = arith.extsi(int64_type, outputs_steps)

    for g in range(num_groups):
        inputs_prods[g].fill(config)
        # The leading size/stride pair (outputs_steps, 0) re-reads the same
        # input block once per output step, which is why _run multiplies the
        # accumulated inputs by outputs_steps.
        inputs_prods[g].fill(
            inputs,
            offset=l2_inputs_len * g,
            sizes=[outputs_steps_64, 1, inputs_steps_64, l2_inputs_len],
            strides=[0, 0, l3_inputs_len, 1],
            transfer_len=inputs_steps * l2_inputs_len,
        )
        outputs_conses[g].drain(
            outputs,
            offset=l2_outputs_len * g,
            sizes=[1, 1, outputs_steps_64, l2_outputs_len],
            strides=[0, 0, l3_outputs_len, 1],
            transfer_len=outputs_steps * l2_outputs_len,
            wait=True,
        )


def sequence_dispatch(
    config: RuntimeData,
    inputs: RuntimeData,
    outputs: RuntimeData,
    inputs_steps,
    outputs_steps,
    inputs_prods: list[ObjectFifoHandle],
    outputs_conses: list[ObjectFifoHandle],
):
    """Runtime-sized host sequence for the EmitC dispatch variant.

    ``inputs_steps`` and ``outputs_steps`` are DispatchTime SSA values.  The
    tensor types are the fixed maximum shapes, so the taps are what make the
    live prefix dynamic.  The input block is repeated across output steps with
    a zero stride, exactly as the static sequence does.
    """
    for g in range(num_groups):
        inputs_prods[g].fill(config)
        inputs_prods[g].fill(
            inputs,
            tap=TensorAccessPattern(
                (l3_inputs_len * DISPATCH_MAX_INPUT_STEPS,),
                l2_inputs_len * g,
                [outputs_steps, 1, inputs_steps, l2_inputs_len],
                [0, 0, l3_inputs_len, 1],
            ),
        )
        outputs_conses[g].drain(
            outputs,
            tap=TensorAccessPattern(
                (l3_outputs_len * DISPATCH_MAX_OUTPUT_STEPS,),
                l2_outputs_len * g,
                [1, 1, outputs_steps, l2_outputs_len],
                [0, 0, l3_outputs_len, 1],
            ),
            wait=True,
        )


def sequence_dispatch_loop(
    config: RuntimeData,
    inputs: RuntimeData,
    outputs: RuntimeData,
    inputs_steps,
    outputs_steps,
    inputs_prods: list[ObjectFifoHandle],
    outputs_conses: list[ObjectFifoHandle],
):
    """Like sequence_dispatch, but with a runtime-bound host loop.

    This deliberately exercises the dynamic EmitC path and its BD pool: each
    loop step issues its own input fills and output drain.  It is opt-in
    because runtime loops plus the BD pool are the configuration that
    deadlocked the fixed program once live counts exceeded FIFO depth.
    """
    i32 = IntegerType.get_signless(32)
    index = IndexType.get()
    loop_n = arith.index_cast(index, outputs_steps)
    for g in range(num_groups):
        config_tg = iron.TaskGroup()
        inputs_prods[g].fill(config, group=config_tg, wait=True)
        config_tg.finish()
        for step in range_(loop_n):
            tg = iron.TaskGroup()
            inputs_prods[g].fill(
                inputs,
                tap=TensorAccessPattern(
                    (l3_inputs_len * DISPATCH_MAX_INPUT_STEPS,),
                    l2_inputs_len * g,
                    [1, 1, inputs_steps, l2_inputs_len],
                    [0, 0, l3_inputs_len, 1],
                ),
                group=tg,
                wait=True,
            )
            step_offset = arith.muli(
                arith.index_cast(i32, step),
                arith.constant(i32, l3_outputs_len),
            )
            output_offset = arith.addi(
                arith.constant(i32, l2_outputs_len * g), step_offset
            )
            outputs_conses[g].drain(
                outputs,
                tap=TensorAccessPattern(
                    (l3_outputs_len * DISPATCH_MAX_OUTPUT_STEPS,),
                    output_offset,
                    [1, 1, 1, l2_outputs_len],
                    [0, 0, 1, 1],
                ),
                group=tg,
                wait=True,
            )
            tg.finish()


def sequence_dispatch_if(
    config: RuntimeData,
    inputs: RuntimeData,
    outputs: RuntimeData,
    inputs_steps,
    outputs_steps,
    extra_groups,
    inputs_prods: list[ObjectFifoHandle],
    outputs_conses: list[ObjectFifoHandle],
):
    """Runtime-sized sequence with a runtime ``if`` around an extra fill.

    This is the shape the RMS norm prelude needs: a host-side scalar decides
    whether a prelude transfer is issued.  The worker reads the same scalar
    from its config and drains the extra group.  All tasks use explicit task
    groups so nothing is hoisted out of the conditional.
    """
    i32 = IntegerType.get_signless(32)
    zero = arith.constant(i32, 0)
    for g in range(num_groups):
        config_tg = iron.TaskGroup()
        inputs_prods[g].fill(config, group=config_tg, wait=True)
        config_tg.finish()

        inputs_tg = iron.TaskGroup()
        inputs_prods[g].fill(
            inputs,
            tap=TensorAccessPattern(
                (l3_inputs_len * DISPATCH_MAX_INPUT_STEPS,),
                l2_inputs_len * g,
                [outputs_steps, 1, inputs_steps, l2_inputs_len],
                [0, 0, l3_inputs_len, 1],
            ),
            group=inputs_tg,
            wait=True,
        )
        inputs_tg.finish()

        with if_(arith.cmpi(CmpIPredicate.ne, extra_groups, zero)):
            extra_tg = iron.TaskGroup()
            inputs_prods[g].fill(
                inputs,
                tap=TensorAccessPattern(
                    (l3_inputs_len * DISPATCH_MAX_INPUT_STEPS,),
                    l2_inputs_len * g,
                    [1, 1, 1, l2_inputs_len],
                    [0, 0, l3_inputs_len, 1],
                ),
                group=extra_tg,
                wait=True,
            )
            extra_tg.finish()

        outputs_tg = iron.TaskGroup()
        outputs_conses[g].drain(
            outputs,
            tap=TensorAccessPattern(
                (l3_outputs_len * DISPATCH_MAX_OUTPUT_STEPS,),
                l2_outputs_len * g,
                [1, 1, outputs_steps, l2_outputs_len],
                [0, 0, l3_outputs_len, 1],
            ),
            group=outputs_tg,
            wait=True,
        )
        outputs_tg.finish()


def _make_design(inputs_type, outputs_type, sequence_fn, step_args):
    """Build the FIFOs, workers and Runtime shared by the static/dispatch variants."""
    # Read inputs into a memory tile and then split to its group.
    l2_inputs_fifos = [
        ObjectFifo(l2_inputs_type, name=f"inputs_{g}") for g in range(num_groups)
    ]
    l1_inputs_fifos = [
        l2_inputs_fifos[g]
        .cons()
        .split(
            offsets=[l1_inputs_len * i for i in range(num_computes_per_group)],
            obj_types=[l1_inputs_type] * num_computes_per_group,
            names=[f"inputs_{g}_{i}" for i in range(num_computes_per_group)],
        )
        for g in range(num_groups)
    ]

    # Collect outputs from a group into its memory tile then write out.
    l2_outputs_fifos = [
        ObjectFifo(l2_outputs_type, name=f"outputs_{g}") for g in range(num_groups)
    ]
    l1_outputs_fifos = [
        l2_outputs_fifos[g]
        .prod()
        .join(
            offsets=[l1_outputs_len * i for i in range(num_computes_per_group)],
            obj_types=[l1_outputs_type] * num_computes_per_group,
            names=[f"outputs_{g}_{i}" for i in range(num_computes_per_group)],
        )
        for g in range(num_groups)
    ]

    compute_init = make_compute_init()
    compute_step = make_compute_step()
    workers = [
        Worker(
            compute,
            [
                l1_inputs_fifos[g][i].cons(),
                l1_outputs_fifos[g][i].prod(),
                compute_init,
                compute_step,
            ],
        )
        for i in range(num_computes_per_group)
        for g in range(num_groups)
    ]

    rt = Runtime(
        sequence_fn,
        [
            l2_inputs_type,
            inputs_type,
            outputs_type,
            *step_args,
            [fifo.prod() for fifo in l2_inputs_fifos],
            [fifo.cons() for fifo in l2_outputs_fifos],
        ],
    )

    return Program(NPU2(), rt, workers).resolve_program()


@iron.jit
def program(
    config: iron.In,
    inputs: iron.In,
    outputs: iron.Out,
    *,
    inputs_steps: iron.CompileTime[np.int32],
    outputs_steps: iron.CompileTime[np.int32],
):
    # Grow the runtime buffers when the step counts are baked in at compile
    # time; keep the base size when they are passed as runtime scalars.
    inputs_type = l3_inputs_type
    outputs_type = l3_outputs_type
    if inputs_steps is not None:
        inputs_type = np.ndarray[
            (l3_inputs_len * inputs_steps,), np.dtype[inputs_dtype]
        ]
    if outputs_steps is not None:
        outputs_type = np.ndarray[
            (l3_outputs_len * outputs_steps,), np.dtype[outputs_dtype]
        ]

    return _make_design(
        inputs_type,
        outputs_type,
        sequence,
        [
            inputs_steps if inputs_steps is not None else np.int32,
            outputs_steps if outputs_steps is not None else np.int32,
        ],
    )


@iron.jit
def program_dispatch(
    config: iron.In,
    inputs: iron.In,
    outputs: iron.Out,
    *,
    inputs_steps: DispatchTime[np.int32],
    outputs_steps: DispatchTime[np.int32],
):
    """One compiled design; step counts arrive per call through the EmitC bridge."""
    return _make_design(
        l3_inputs_max_type,
        l3_outputs_max_type,
        sequence_dispatch,
        [inputs_steps, outputs_steps],
    )


@iron.jit
def program_dispatch_loop(
    config: iron.In,
    inputs: iron.In,
    outputs: iron.Out,
    *,
    inputs_steps: DispatchTime[np.int32],
    outputs_steps: DispatchTime[np.int32],
):
    """Dispatch variant with a runtime-bound host loop (dynamic BD pool)."""
    return _make_design(
        l3_inputs_max_type,
        l3_outputs_max_type,
        sequence_dispatch_loop,
        [inputs_steps, outputs_steps],
    )


@iron.jit
def program_dispatch_if(
    config: iron.In,
    inputs: iron.In,
    outputs: iron.Out,
    *,
    inputs_steps: DispatchTime[np.int32],
    outputs_steps: DispatchTime[np.int32],
    extra_groups: DispatchTime[np.int32],
):
    """Dispatch variant with a runtime ``if`` around one extra input fill."""
    return _make_design(
        l3_inputs_max_type,
        l3_outputs_max_type,
        sequence_dispatch_if,
        [inputs_steps, outputs_steps, extra_groups],
    )



def _bench_native(
    *,
    num_calls: int,
    warmup: int,
    modes=(
        "repeat-a",
        "repeat-b",
        "alternate",
        "reuse",
        "pipeline",
    ),
):
    print(f"\nC++ API, {num_calls} calls (warmup {warmup})")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(script_dir, "build/invoke.dynamic")
    os.makedirs(work_dir, exist_ok=True)

    def run_tool(*args):
        subprocess.run(list(args), check=True, cwd=work_dir)

    # as_mlir() must come first: it instantiates the external kernels, which
    # compile_external_kernels() then collects from ExternalFunction._instances.
    mlir = program.as_mlir(inputs_steps=None, outputs_steps=None)
    compile_external_kernels(ExternalFunction._instances, work_dir, "aie2p")
    compile_mlir_module(mlir, work_dir=work_dir, device=NPU2())

    # Lower the MLIR enough to generate the C++ header.
    run_tool(
        "aie-opt",
        "--aie-substitute-shim-dma-allocations",
        "--aie-assign-runtime-sequence-bd-ids",
        "--aie-dma-tasks-to-npu",
        "--aie-dma-to-npu",
        "-o",
        "lowered.mlir",
        "input_with_addresses.mlir",
    )

    # Generate the C++ code that creates runtime instructions.
    run_tool(
        "aie-translate",
        "--aie-npu-to-cpp",
        "-o",
        "generate_txn.h",
        "lowered.mlir",
    )

    # Patch the generated header so overflows throw with a line number.
    txn_path = os.path.join(work_dir, "generate_txn.h")
    with open(txn_path, "r") as f:
        txn_src = f.read()
    patched, count = re.subn(
        r"return\s+std::nullopt\s*;",
        'throw std::runtime_error(std::string("Overflow at ") + __FILE__ + '
        '":" + std::to_string(__LINE__));',
        txn_src,
    )
    if count:
        patched = "#include <stdexcept>\n" + patched
        with open(txn_path, "w") as f:
            f.write(patched)

    # Compile the lowered MLIR to the final xclbin.
    run_tool("aiecc", "--get-xclbin", "--tmpdir=xclbin_build", "lowered.mlir")

    site_packages = sysconfig.get_paths()["purelib"]
    run_tool(
        "clang++",
        "-O2",
        "-g",
        f"-I{site_packages}/mlir_aie/include",
        f"-I{work_dir}",
        "-oinvoke",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "invoke.cc"),
        "-lxrt++",
        "-lxrt_coreutil",
    )

    # invoke.cc prints a total for the pipeline mode instead of a min-max range.
    print(f"  {'mode':<20} {'avg us':>7}  {'(min-max)':>14}")
    for mode in modes:
        run_tool("./invoke", mode, str(num_calls), str(warmup))

    print("\n2x size:")
    for mode in modes:
        run_tool("./invoke", mode, str(num_calls), str(warmup), str(2))

    print("\n4x size:")
    for mode in modes:
        run_tool("./invoke", mode, str(num_calls), str(warmup), str(4))


def _compile_static():
    out_dir = "build/invoke.static"
    program.specialize(
        inputs_steps=2,
        outputs_steps=2,
    ).compile(
        xclbin_path=f"{out_dir}/aie.xclbin",
        inst_path=f"{out_dir}/insts.bin",
    )


def _run(inputs_steps, outputs_steps):
    config, inputs, outputs = make_args(inputs_steps, outputs_steps)
    outputs_len = l3_outputs_len * outputs_steps

    program(
        config,
        inputs,
        outputs,
        inputs_steps=inputs_steps,
        outputs_steps=outputs_steps,
    )

    expected = []
    for _ in range(outputs_steps):
        for g in range(num_groups):
            for i in range(num_computes_per_group):
                for c in range(l1_outputs_len):
                    expected.append(
                        sum(
                            inputs[
                                l2_inputs_len * g
                                + l1_inputs_len * i
                                + c
                                + r * l3_inputs_len
                            ]
                            for r in range(inputs_steps)
                        )
                    )

    if outputs_len != len(expected):
        print("length differs: ", outputs_len, len(expected))
        return

    errors = 0
    for i in range(outputs_len):
        if outputs[i] != expected[i]:
            print(i, outputs[i], expected[i])
            errors += 1
            if errors >= 10:
                print("... too many errors, stopping.")
                break


def _bench_python(*, num_calls: int, warmup: int):
    shape_a = shape_to_steps(2048, 4096)
    shape_b = shape_to_steps(4096, 2048)
    args_a = make_args(*shape_a)
    args_b = make_args(*shape_b)

    # iron.jit can not run dynamic step counts, so specialize one program per
    # shape and switch between them.
    def run_a():
        return program(*args_a, inputs_steps=shape_a[0], outputs_steps=shape_a[1])

    def run_b():
        return program(*args_b, inputs_steps=shape_b[0], outputs_steps=shape_b[1])

    bench_a = run_iters(run_a, warmup=warmup, iters=num_calls)
    bench_b = run_iters(run_b, warmup=warmup, iters=num_calls)

    alt_runs = itertools.cycle((run_a, run_b))
    bench_alt = run_iters(lambda: next(alt_runs)(), warmup=warmup, iters=num_calls)

    def report(name, bench):
        print(
            f"  {name:<20} {bench.npu.avg_us:7.1f}"
            f"    ({bench.npu.min_us:6.1f}-{bench.npu.max_us:6.1f})"
            f"       {bench.e2e.avg_us:7.1f}"
        )

    print(f"\nPython API, {num_calls} calls (warmup {warmup}):")
    print(f"  {'workload':<20} {'avg us':>7}  {'(min-max)':>14}  {'e2e avg us':>15}")
    report("repeat A", bench_a)
    report("repeat B", bench_b)
    report("reconfigure A/B", bench_alt)

    # The reconfiguring runs switch between two separately-compiled designs on
    # every call. The extra time over the average of the two uniform runs is the
    # cost of reloading/reconfiguring the NPU for the other design, not the
    # compute itself.
    uniform_e2e = 0.5 * (bench_a.e2e.avg_us + bench_b.e2e.avg_us)
    uniform_npu = 0.5 * (bench_a.npu.avg_us + bench_b.npu.avg_us)
    print(
        "  reconfiguration per switch:",
        f"npu {bench_alt.npu.avg_us - uniform_npu:.1f} us",
        f"e2e {bench_alt.e2e.avg_us - uniform_e2e:.1f} us",
    )


def _expected_outputs(inputs, inputs_steps, outputs_steps):
    expected = []
    for _ in range(outputs_steps):
        for g in range(num_groups):
            for i in range(num_computes_per_group):
                for c in range(l1_outputs_len):
                    expected.append(
                        sum(
                            inputs[
                                l2_inputs_len * g
                                + l1_inputs_len * i
                                + c
                                + r * l3_inputs_len
                            ]
                            for r in range(inputs_steps)
                        )
                    )
    return expected


def _check_dispatch(fn, inputs_steps, outputs_steps, label):
    config, inputs, outputs = make_dispatch_args(inputs_steps, outputs_steps)
    fn(
        config,
        inputs,
        outputs,
        inputs_steps=inputs_steps,
        outputs_steps=outputs_steps,
    )
    expected = _expected_outputs(inputs, inputs_steps, outputs_steps)
    if len(expected) != l3_outputs_len * outputs_steps:
        raise AssertionError("expected-output size mismatch")
    errors = 0
    for i, want in enumerate(expected):
        if outputs[i] != want:
            errors += 1
            if errors <= 5:
                print(f"  {label}: index {i} got {outputs[i]} want {want}")
    print(f"  {label}: ({inputs_steps}, {outputs_steps}) errors={errors}")
    return errors


def _bench_dispatch(*, num_calls: int, warmup: int):
    shape_a = shape_to_steps(2048, 4096)
    shape_b = shape_to_steps(4096, 2048)
    args_a = make_dispatch_args(*shape_a)
    args_b = make_dispatch_args(*shape_b)

    def run_a():
        return program_dispatch(
            *args_a,
            inputs_steps=shape_a[0],
            outputs_steps=shape_a[1],
        )

    def run_b():
        return program_dispatch(
            *args_b,
            inputs_steps=shape_b[0],
            outputs_steps=shape_b[1],
        )

    bench_a = run_iters(run_a, warmup=warmup, iters=num_calls)
    bench_b = run_iters(run_b, warmup=warmup, iters=num_calls)
    alt_runs = itertools.cycle((run_a, run_b))
    bench_alt = run_iters(lambda: next(alt_runs)(), warmup=warmup, iters=num_calls)

    print(f"\nPython API, EmitC dispatch, {num_calls} calls (warmup {warmup}):")
    print(f"  {'workload':<20} {'avg us':>7}  {'(min-max)':>14}  {'e2e avg us':>15}")
    for name, bench in (
        ("repeat A", bench_a),
        ("repeat B", bench_b),
        ("alternate A/B", bench_alt),
    ):
        print(
            f"  {name:<20} {bench.npu.avg_us:7.1f}"
            f"    ({bench.npu.min_us:6.1f}-{bench.npu.max_us:6.1f})"
            f"       {bench.e2e.avg_us:7.1f}"
        )
    print("  one compiled design; A/B differ only in runtime step counts")


def _bench_dispatch_loop(*, num_calls: int, warmup: int):
    shape_a = (4, 1)
    shape_b = (2, 2)
    args_a = make_dispatch_args(*shape_a)
    args_b = make_dispatch_args(*shape_b)

    def run_a():
        return program_dispatch_loop(
            *args_a,
            inputs_steps=shape_a[0],
            outputs_steps=shape_a[1],
        )

    def run_b():
        return program_dispatch_loop(
            *args_b,
            inputs_steps=shape_b[0],
            outputs_steps=shape_b[1],
        )

    bench_a = run_iters(run_a, warmup=warmup, iters=num_calls)
    bench_b = run_iters(run_b, warmup=warmup, iters=num_calls)
    print(f"\nPython API, EmitC dispatch loop, {num_calls} calls (warmup {warmup}):")
    print(f"  repeat A {bench_a.npu.avg_us:7.1f} us   e2e {bench_a.e2e.avg_us:7.1f}")
    print(f"  repeat B {bench_b.npu.avg_us:7.1f} us   e2e {bench_b.e2e.avg_us:7.1f}")


def _check_dispatch_if(inputs_steps, outputs_steps, extra_groups):
    config, inputs, outputs = make_dispatch_if_args(
        inputs_steps, outputs_steps, extra_groups
    )
    program_dispatch_if(
        config,
        inputs,
        outputs,
        inputs_steps=inputs_steps,
        outputs_steps=outputs_steps,
        extra_groups=extra_groups,
    )
    expected = _expected_outputs(inputs, inputs_steps, outputs_steps)
    errors = sum(
        1
        for i, want in enumerate(expected)
        if outputs[i] != want
    )
    print(
        f"  if: ({inputs_steps}, {outputs_steps}, extra={extra_groups}) "
        f"errors={errors}"
    )
    return errors


def _bench_dispatch_if(*, num_calls: int, warmup: int):
    shape = (2, 1)
    args_off = make_dispatch_if_args(*shape, 0)
    args_on = make_dispatch_if_args(*shape, 1)

    def run_off():
        return program_dispatch_if(
            *args_off,
            inputs_steps=shape[0],
            outputs_steps=shape[1],
            extra_groups=0,
        )

    def run_on():
        return program_dispatch_if(
            *args_on,
            inputs_steps=shape[0],
            outputs_steps=shape[1],
            extra_groups=1,
        )

    bench_off = run_iters(run_off, warmup=warmup, iters=num_calls)
    bench_on = run_iters(run_on, warmup=warmup, iters=num_calls)
    print(f"\nPython API, EmitC dispatch if, {num_calls} calls (warmup {warmup}):")
    print(f"  extra=0 {bench_off.npu.avg_us:7.1f} us   e2e {bench_off.e2e.avg_us:7.1f}")
    print(f"  extra=1 {bench_on.npu.avg_us:7.1f} us   e2e {bench_on.e2e.avg_us:7.1f}")


def main():
    args = sys.argv[1:]
    if "--emitc-if" in args:
        _check_dispatch_if(2, 1, 0)
        _check_dispatch_if(2, 1, 1)
        _bench_dispatch_if(num_calls=64, warmup=4)
        return
    if "--emitc-loop" in args:
        _check_dispatch(program_dispatch_loop, 2, 1, "loop")
        _check_dispatch(program_dispatch_loop, 2, 2, "loop")
        _bench_dispatch_loop(num_calls=32, warmup=4)
        return
    if "--emitc" in args:
        _check_dispatch(program_dispatch, 4, 2, "dispatch")
        _check_dispatch(program_dispatch, 2, 1, "dispatch")
        _bench_dispatch(num_calls=128, warmup=8)
        return

    _bench_python(num_calls=128, warmup=8)
    _bench_native(num_calls=128, warmup=8)

    # Other options:
    # _compile_static()  # build a fully static design (xclbin + insts.bin)
    # _run(3, 2)         # one dynamic call, checked against the reference model


if __name__ == "__main__":
    main()
