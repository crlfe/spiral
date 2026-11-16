"""Verify ObjectFifo buffers carried through `range_` iter_args on device.

This exercises the pattern needed by the main program's continuous weight
stream:

* acquire one ObjectFifo object outside a loop;
* carry the acquired memref (and an index) through `scf.for` iter_args;
* conditionally release it and acquire the next object using an
  `scf.if` that returns the new memref;
* release the final object after the loop.

The worker sums the first word of four input objects (1+2+3+4) and writes the
result to an output object.  Expected output is 10.

Run:

    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/iter_args.py
"""

import numpy as np
from aie import iron
from aie.dialects import arith
from aie.dialects._aie_enum_gen import AIETileType
from aie.dialects._arith_enum_gen import CmpIPredicate
from aie.dialects.scf import IfOp
from aie.dialects.scf import yield_ as scf_yield
from aie.ir import IndexType, InsertionPoint
from aie.iron import (
    Buffer,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.controlflow import yield_ as iron_yield
from aie.iron.device import NPU2, Tile

OBJECT_ELEMS = 16
NUM_OBJECTS = 4
EXPECTED = 10  # 1 + 2 + 3 + 4

l1_object_type = np.ndarray[(OBJECT_ELEMS,), np.dtype[np.int32]]
l1_state_type = np.ndarray[(4,), np.dtype[np.int32]]

l3_inputs_type = np.ndarray[
    (NUM_OBJECTS * OBJECT_ELEMS,), np.dtype[np.int32]
]
l3_outputs_type = np.ndarray[(OBJECT_ELEMS,), np.dtype[np.int32]]


def _make_kernels():
    zero = ExternalFunction(
        "iter_args_zero",
        arg_types=[l1_state_type],
        inline=True,
        source_string="""
extern "C" void iter_args_zero(int32_t * restrict state) {
  state[0] = 0;
  state[1] = 0;
  state[2] = 0;
  state[3] = 0;
}
""",
    )
    accumulate = ExternalFunction(
        "iter_args_accumulate",
        arg_types=[l1_object_type, l1_state_type],
        inline=True,
        source_string="""
extern "C" void iter_args_accumulate(
    const int32_t * restrict object,
    int32_t * restrict state) {
  state[0] += object[0];
  state[1] = object[0];
  state[2] += 1;
  state[3] = object[1];
}
""",
    )
    copy_out = ExternalFunction(
        "iter_args_copy_out",
        arg_types=[l1_state_type, l1_object_type],
        inline=True,
        source_string="""
extern "C" void iter_args_copy_out(
    const int32_t * restrict state,
    int32_t * restrict out) {
  out[0] = state[0];
  out[1] = state[1];
  out[2] = state[2];
  out[3] = state[3];
}
""",
    )
    return zero, accumulate, copy_out


@iron.jit(use_cache=False)
def program(inputs: In, outputs: Out):
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    core = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)

    inputs_l2 = ObjectFifo(
        l1_object_type, name="iter_inputs_l2", depth=NUM_OBJECTS
    )
    inputs_l1 = inputs_l2.cons().forward(
        tile=mem,
        obj_type=l1_object_type,
        name="iter_inputs_l1",
    )
    outputs_l2 = ObjectFifo(
        l1_object_type, name="iter_outputs_l2", depth=1
    )
    outputs_l1 = outputs_l2.prod().join(
        offsets=[0],
        tile=mem,
        obj_types=[l1_object_type],
        names=["iter_outputs_l1"],
    )

    zero, accumulate, copy_out = _make_kernels()
    state = Buffer(l1_state_type, name="iter_state")

    def body(
        inputs_cons,
        outputs_prod,
        state,
        zero,
        accumulate,
        copy_out,
    ):
        zero(state)
        block = inputs_cons.acquire(1)
        last = arith.constant(IndexType.get(), NUM_OBJECTS - 1)

        for i, current, _result in range_(
            NUM_OBJECTS, iter_args=[block], insert_yield=False
        ):
            accumulate(current, state)

            take_next = arith.cmpi(CmpIPredicate.slt, i, last)
            branch = IfOp(take_next, [current.type], has_else=True)
            with InsertionPoint(branch.then_block):
                inputs_cons.release(1)
                next_block = inputs_cons.acquire(1)
                scf_yield([next_block])
            with InsertionPoint(branch.else_block):
                scf_yield([current])
            iron_yield([branch.results[0]])

        inputs_cons.release(1)

        out_buf = outputs_prod.acquire(1)
        copy_out(state, out_buf)
        outputs_prod.release(1)

    worker = Worker(
        body,
        [
            inputs_l1.cons(),
            outputs_l1[0].prod(),
            state,
            zero,
            accumulate,
            copy_out,
        ],
        tile=core,
    )

    def sequence(inputs_buf, outputs_buf, inputs_prod, outputs_cons):
        tg = iron.TaskGroup()
        inputs_prod.fill(
            inputs_buf,
            offset=0,
            sizes=[NUM_OBJECTS, 1, 1, OBJECT_ELEMS],
            strides=[OBJECT_ELEMS, OBJECT_ELEMS, 0, 1],
            transfer_len=OBJECT_ELEMS,
            group=tg,
            wait=False,
        )
        outputs_cons.drain(
            outputs_buf,
            offset=0,
            sizes=[1, 1, 1, OBJECT_ELEMS],
            strides=[0, 0, 0, 1],
            transfer_len=OBJECT_ELEMS,
            group=tg,
            wait=True,
        )
        tg.finish()

    rt = Runtime(
        sequence,
        [
            l3_inputs_type,
            l3_outputs_type,
            inputs_l2.prod(tile=shim),
            outputs_l2.cons(tile=shim),
        ],
    )
    return Program(NPU2(), rt, [worker]).resolve_program()


def main():
    inputs = np.zeros(NUM_OBJECTS * OBJECT_ELEMS, dtype=np.int32)
    for object_index in range(NUM_OBJECTS):
        inputs[object_index * OBJECT_ELEMS] = object_index + 1

    inputs_tensor = iron.tensor(inputs, dtype=np.int32)
    outputs_tensor = iron.zeros(OBJECT_ELEMS, dtype=np.int32)
    program(inputs_tensor, outputs_tensor)
    outputs = np.asarray(outputs_tensor)

    got = int(outputs[0])
    last = int(outputs[1])
    count = int(outputs[2])
    print(
        f"iter_args carried block: sum {got}, last {last}, count {count} "
        f"(expected {EXPECTED}, {NUM_OBJECTS}, {NUM_OBJECTS})"
    )
    if got != EXPECTED or last != NUM_OBJECTS or count != NUM_OBJECTS:
        raise SystemExit(1)
    print("ok")


if __name__ == "__main__":
    main()
