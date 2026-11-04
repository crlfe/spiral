#!/usr/bin/env python3
#
# Verify and profile the Q4_0 -> bf16 dequant tile.
#
#     source ~/Projects/mlir-aie/iron_env.sh
#     python src/dequant/q4_0.py
#
# One tile is 16 canonical 18-byte Q4_0 blocks = 8 output columns x 2 K blocks
# (64 reduction values). The runner repacks them into aligned signed-int4
# payloads with grouped fp16 scales before sending the tile to the NPU. The
# output is the 512-value bf16 B-tile layout expected by aie::mmul<4, 8, 8>.
#

import argparse
import json
from pathlib import Path

import numpy as np
from aie import iron
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.utils.trace import TraceConfig
from ml_dtypes import bfloat16

KERNEL_SOURCE = Path(__file__).resolve().with_name("q4_0.cc")

# Must match the constants in q4_0.cc.
BLOCK_WEIGHTS = 32
BLOCK_BYTES = 18
COLS = 8
K_BLOCKS_PER_COL = 2
TILE_BLOCKS = COLS * K_BLOCKS_PER_COL
TILE_K = K_BLOCKS_PER_COL * BLOCK_WEIGHTS
K_SUBTILES = TILE_K // 8
OUT_VALUES = TILE_K * COLS

PAYLOAD_ROWS = 4
PAYLOADS_PER_KBLOCK = BLOCK_WEIGHTS // PAYLOAD_ROWS  # 8 payloads of 4 K x 8 N
PAYLOAD_BYTES = (PAYLOAD_ROWS * COLS) // 2  # 32 signed int4 values = 16 B
PAYLOAD_REGION = PAYLOADS_PER_KBLOCK * PAYLOAD_BYTES  # 128 B
SCALE_BYTES = COLS * 2  # 8 fp16 scales = 16 B
SCALE_OFFSET = PAYLOAD_REGION
UNIT_BYTES = PAYLOAD_REGION + SCALE_BYTES  # 144 B
TILE_BYTES = K_BLOCKS_PER_COL * UNIT_BYTES

TRACE_SIZE = 4096
TRACE_FILE = "/tmp/q4_0_trace.txt"
TRACE_JSON = "/tmp/q4_0_trace.json"


def repack_tile(raw_blocks):
    """Repack canonical Q4_0 blocks into the kernel's transposed int4 tile.

    Layout per k block: [8 x 16-byte payloads][8 fp16 scales]. Each payload is
    a 4 K x 8 N block in row-major order with signed (q - 8) nibbles.
    """
    raw = raw_blocks.reshape(TILE_BLOCKS, BLOCK_BYTES)
    packed = np.zeros((K_BLOCKS_PER_COL, UNIT_BYTES), dtype=np.uint8)

    for kb in range(K_BLOCKS_PER_COL):
        blocks = np.stack(
            [raw[col * K_BLOCKS_PER_COL + kb] for col in range(COLS)]
        )
        packed[kb, SCALE_OFFSET : SCALE_OFFSET + SCALE_BYTES] = blocks[:, 0:2].reshape(
            -1
        )

        # Canonical Q4_0 stores q[0:16] in low nibbles and q[16:32] in high
        # nibbles. Reassemble natural K order, center on zero, then transpose
        # into 4 K x 8 N payloads.
        q = np.concatenate((blocks[:, 2:18] & 0x0F, blocks[:, 2:18] >> 4), axis=1)
        signed = ((q.astype(np.int16) - 8) & 0x0F).astype(np.uint8)

        for payload in range(PAYLOADS_PER_KBLOCK):
            rows = signed[
                :, payload * PAYLOAD_ROWS : (payload + 1) * PAYLOAD_ROWS
            ].T
            values = rows.reshape(-1)
            base = payload * PAYLOAD_BYTES
            packed[kb, base : base + PAYLOAD_BYTES] = (
                values[0::2] | (values[1::2] << 4)
            )

    return packed.reshape(-1)


def make_tile(seed):
    """Build a 16-block Q4_0 tile and its expected mmul-layout dequantization."""
    rng = np.random.default_rng(seed)

    # [column][k block][k value], unsigned as stored in canonical Q4_0.
    q = rng.integers(0, 16, (COLS, K_BLOCKS_PER_COL, BLOCK_WEIGHTS), dtype=np.uint8)

    # Powers of two keep every (q - 8) * scale product exactly representable in
    # bf16, so the result can be compared elementwise instead of with tolerance.
    scales = rng.choice(
        np.array([0.5, 1.0, 2.0], dtype=np.float16),
        (COLS, K_BLOCKS_PER_COL),
    )

    # Canonical Q4_0 block: [fp16 scale][16 packed bytes], where byte j holds
    # q[j] in the low nibble and q[j + 16] in the high nibble.
    payload = q[:, :, :16] | (q[:, :, 16:] << 4)
    scale_bytes = scales.view(np.uint8).reshape(COLS, K_BLOCKS_PER_COL, 2)
    raw_blocks = np.concatenate((scale_bytes, payload), axis=2).reshape(-1)
    blocks = repack_tile(raw_blocks)
    assert blocks.size == TILE_BYTES

    # Reference in [k subtile][k inside][column] order, matching the kernel.
    values = (q.astype(np.float32) - 8.0) * scales.astype(np.float32)[..., None]
    expected = (
        values.reshape(COLS, K_BLOCKS_PER_COL, 4, 8)
        .transpose(1, 2, 3, 0)
        .reshape(-1)
        .astype(bfloat16)
    )
    return blocks, expected


@iron.jit(source_files=[str(KERNEL_SOURCE)])
def dequant_program(
    blocks: In,
    out: Out,
    *,
    trace_size: iron.CompileTime[np.int32] = 0,
    trace_config: iron.CompileTime[object] = None,
):
    blocks_ty = np.ndarray[(TILE_BYTES,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(OUT_VALUES,), np.dtype[bfloat16]]

    kernel = ExternalFunction(
        "q4_0_dequant_tile",
        source_file=str(KERNEL_SOURCE),
        arg_types=[blocks_ty, out_ty],
    )

    in_fifo = ObjectFifo(blocks_ty, name="blocks", depth=1)
    out_fifo = ObjectFifo(out_ty, name="out", depth=1)

    def core_body(in_cons, out_prod, kernel):
        in_buf = in_cons.acquire(1)
        out_buf = out_prod.acquire(1)
        kernel(in_buf, out_buf)
        in_cons.release(1)
        out_prod.release(1)

    worker = Worker(
        core_body,
        [in_fifo.cons(), out_fifo.prod(), kernel],
        trace=1,
        stack_size=8192,
    )

    def sequence(blocks_buf, out_buf, in_prod, out_cons):
        in_prod.fill(blocks_buf)
        out_cons.drain(out_buf, wait=True)

    rt = Runtime(
        sequence,
        [blocks_ty, out_ty, in_fifo.prod(), out_fifo.cons()],
    )

    prog = Program(NPU2(), rt, [worker])
    if trace_size > 0:
        prog.enable_trace(trace_size, workers=[worker])
    return prog.resolve_program()


def trace_event_cycles(trace_path):
    with open(trace_path, "r") as f:
        data = json.load(f)

    events = sorted(
        (entry["ts"], entry["name"])
        for entry in data
        if entry.get("ph") == "B"
        and entry.get("name", "").startswith("INSTR_EVENT_")
    )
    intervals = [
        (events[i][1], events[i + 1][1], events[i + 1][0] - events[i][0])
        for i in range(len(events) - 1)
    ]
    return events, intervals


def event_label(name):
    return name.replace("INSTR_EVENT_", "event")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify and profile Q4_0 dequant.")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not KERNEL_SOURCE.exists():
        raise SystemExit(f"kernel source not found: {KERNEL_SOURCE.name}")

    blocks, expected = make_tile(args.seed)
    blocks_t = iron.tensor(blocks, dtype=np.uint8)
    out = iron.zeros((OUT_VALUES,), dtype=bfloat16)

    trace_config = TraceConfig(
        trace_size=TRACE_SIZE,
        trace_file=TRACE_FILE,
    )
    dequant_program(
        blocks_t,
        out,
        trace_size=TRACE_SIZE,
        trace_config=trace_config,
    )
    trace_config.trace_to_json(trace_config.physical_mlir_path, TRACE_JSON)

    print(f"kernel: {KERNEL_SOURCE.name}")
    print(
        f"tile: {TILE_BLOCKS} Q4_0 blocks = {COLS} columns x "
        f"{TILE_K} K values, {OUT_VALUES} bf16 outputs"
    )

    events, intervals = trace_event_cycles(TRACE_JSON)
    if not intervals:
        raise SystemExit("trace contained no successive event intervals")
    if len(intervals) > 1:
        for start, end, cycles in intervals:
            print(f"{event_label(start)} -> {event_label(end)}: {cycles} cycles")

    kernel_cycles = events[-1][0] - events[0][0]
    weights = TILE_BLOCKS * BLOCK_WEIGHTS
    print(
        f"{event_label(events[0][1])} -> {event_label(events[-1][1])}: "
        f"{kernel_cycles} cycles = {weights / kernel_cycles:.3f} weights/cycle"
    )

    actual = np.asarray(out)
    if np.array_equal(actual, expected):
        print("verified against numpy (exact bf16 match)")
        return

    bad = np.flatnonzero(actual != expected)
    print(f"FAILED: {len(bad)} of {actual.size} values differ")
    for idx in bad[:8]:
        print(f"  out[{idx}]: got {actual[idx]}, expected {expected[idx]}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
