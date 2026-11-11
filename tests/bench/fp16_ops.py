#!/usr/bin/env python3
"""Capability probes for float16 on the AMD XDNA2 NPU.

XDNA2 is arch 21 (`aie2p`). The AIE API headers only define `float16` for
arch 22 (`aie2ps`), so these probes check which float16 operations the
compiler accepts and which bf16 fallbacks the quant kernels must use.

The driver is deliberately standalone: it does not import the `src/quants`
test harness. Each probe is compiled in isolation, so one unsupported
operation fails only its own row. Probes that compile are then run under the
IRON harness and timed with one `event0()` / `event1()` pair.

Results are printed as a table. The durable findings belong in
`src/quants/NOTES.md`; this script is the evidence.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

BENCH_ROOT = Path(__file__).resolve().parent
SOURCE = BENCH_ROOT / "fp16_ops.aie.cc"

IN_BYTES = 8192
OUT_BYTES = 8192
ITERS = 32
Q6K_STRIDE = 128
TRACE_SIZE = 65536

_TMP_DIR = tempfile.TemporaryDirectory(prefix="bench_fp16_")
TMP_ROOT = Path(_TMP_DIR.name)


class NpuUnavailable(Exception):
    """Raised when the NPU or its trace tooling is unavailable."""


def npu_available():
    if os.environ.get("BENCH_NO_NPU"):
        return False
    patterns = ("/dev/accel/accel*", "/dev/dri/accel*")
    return any(glob.glob(pattern) for pattern in patterns)


@dataclass
class Probe:
    name: str
    macro: str
    timed: bool = False
    build_input: object = None
    check: object = None
    note: str = ""


# ---------------------------------------------------------------------------
# Probe references. Each builder returns the full input buffer and the expected
# output prefix. Each checker returns (max error, note).
# ---------------------------------------------------------------------------


def _seed_input(seed):
    return np.random.default_rng(seed)


def make_u16(seed):
    rng = _seed_input(seed)
    payload = rng.integers(0, 65536, size=ITERS * 64, dtype=np.uint16)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: payload.nbytes] = payload.view(np.uint8)
    return data, data[: payload.nbytes].copy()


def check_bytes(out, expected):
    used = expected.size
    if np.array_equal(out[:used], expected):
        return 0.0, "exact"
    diff = np.abs(out[:used].astype(np.int16) - expected.astype(np.int16))
    return float(diff.max()), "byte mismatch"


def make_bf16(seed):
    rng = _seed_input(seed)
    values = rng.standard_normal(ITERS * 64).astype(np.float32).astype(bfloat16)
    payload = values.view(np.uint16)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: payload.nbytes] = payload.view(np.uint8)
    return data, data[: payload.nbytes].copy()


def make_fp16_bits(seed):
    rng = _seed_input(seed)
    values = rng.uniform(-8.0, 8.0, size=ITERS * 8).astype(np.float16)
    payload = values.view(np.uint16)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: payload.nbytes] = payload.view(np.uint8)
    return data, values.astype(np.float32)


def check_f32(out, expected):
    used = expected.nbytes
    actual = out[:used].view(np.float32)
    error = np.abs(actual - expected)
    return float(error.max()), "emulated f32" if error.max() else "exact"


def make_split(seed):
    data, expected = make_fp16_bits(seed)
    hi = expected.astype(bfloat16)
    lo = (expected - hi.astype(np.float32)).astype(bfloat16)
    paired = np.stack(
        [hi.reshape(ITERS, 8), lo.reshape(ITERS, 8)], axis=1
    ).reshape(-1)
    return data, paired


def check_split(out, expected):
    used = expected.nbytes
    actual = out[:used].view(bfloat16)
    error = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return float(error.max()), "hi/lo pair"


def check_split_sum(out, expected):
    used = expected.size
    paired = out[: used * 4].view(bfloat16).reshape(ITERS, 2, 8)
    total = paired[:, 0, :].astype(np.float32) + paired[:, 1, :].astype(
        np.float32
    )
    error = np.abs(total.reshape(-1) - expected.astype(np.float32))
    return float(error.max()), "hi+lo"


def make_bf16_scale(seed):
    rng = _seed_input(seed)
    scale = np.array([rng.uniform(0.01, 0.2)], dtype=bfloat16)
    q = rng.integers(-128, 128, size=ITERS * 64, dtype=np.int16).astype(np.int8)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[:2] = scale.view(np.uint16).view(np.uint8)
    data[64 : 64 + q.nbytes] = q.view(np.uint8)
    expected = (q.astype(np.float32) * float(scale[0])).astype(bfloat16)
    return data, expected


def check_bf16(out, expected):
    used = expected.nbytes
    actual = out[:used].view(bfloat16)
    error = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return float(error.max()), "single bf16 scale"


def make_f32_scale(seed):
    rng = _seed_input(seed)
    scale = np.array([rng.uniform(0.01, 0.2)], dtype=np.float32)
    q = rng.integers(-128, 128, size=ITERS * 64, dtype=np.int16).astype(np.int8)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[:4] = scale.view(np.uint8)
    data[64 : 64 + q.nbytes] = q.view(np.uint8)
    expected = q.astype(np.float32) * float(scale[0])
    return data, expected


def make_fp16_scale(seed):
    rng = _seed_input(seed)
    scale = np.array([rng.uniform(0.01, 0.2)], dtype=np.float16)
    q = rng.integers(-128, 128, size=ITERS * 64, dtype=np.int16).astype(np.int8)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[:2] = scale.view(np.uint16).view(np.uint8)
    data[64 : 64 + q.nbytes] = q.view(np.uint8)
    hi = scale.astype(bfloat16)
    lo = (scale.astype(np.float32) - hi.astype(np.float32)).astype(bfloat16)
    expected = (
        q.astype(np.float32) * hi.astype(np.float32)
        + q.astype(np.float32) * lo.astype(np.float32)
    ).astype(bfloat16)
    return data, expected


def _fp16_scale_bits(data):
    return int(data[:2].view(np.uint16)[0])


def make_fp16_int_mul(seed):
    data, _ = make_fp16_scale(seed)
    bits = _fp16_scale_bits(data)
    mantissa = (bits & 0x03FF) | 0x0400
    q = data[64 : 64 + ITERS * 64].view(np.int8)
    expected = q.astype(np.int32) * mantissa
    return data, expected


def make_fp16_int_round(seed):
    data, expected_int = make_fp16_int_mul(seed)
    expected = expected_int.astype(np.float32).astype(bfloat16)
    return data, expected


def check_int32(out, expected):
    used = expected.nbytes
    actual = out[:used].view(np.int32)
    error = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
    return float(error.max()), "int32 product"


def _make_q6k_records(seed):
    rng = _seed_input(seed)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    scales = []
    payloads = []
    for i in range(ITERS):
        rec = i * Q6K_STRIDE
        d = rng.uniform(0.005, 0.25, 8).astype(np.float16)
        s = rng.integers(-127, 128, 8).astype(np.int8)
        s[s == 0] = 1
        q = rng.integers(-128, 128, 64, dtype=np.int16).astype(np.int8)
        data[rec : rec + 16] = d.view(np.uint16).view(np.uint8)
        data[rec + 16 : rec + 24] = s.view(np.uint8)
        data[rec + 64 : rec + 64 + 64] = q.view(np.uint8)
        scales.append((d, s))
        payloads.append(q)
    return data, scales, payloads


def _q6k_expected(d, s, q):
    scale = d.astype(np.float32) * s.astype(np.float32)
    return (
        q.reshape(8, 8).astype(np.float32) * scale[None, :]
    ).reshape(-1).astype(bfloat16)


def make_q6k_full(seed):
    data, scales, payloads = _make_q6k_records(seed)
    expected = np.concatenate(
        [
            _q6k_expected(d, s, q)
            for (d, s), q in zip(scales, payloads)
        ]
    )
    return data, expected


def make_q6k_fixed(seed):
    data, scales, payloads = _make_q6k_records(seed)
    d, s = scales[0]
    expected = np.concatenate([_q6k_expected(d, s, q) for q in payloads])
    return data, expected


def make_q6k_int_fixed(seed):
    data, scales, payloads = _make_q6k_records(seed)
    d, s = scales[0]
    bits = d.view(np.uint16)
    exponent = (bits & 0x7C00) >> 10
    mantissa = ((bits & 0x03FF) | 0x0400).astype(np.int64)
    meff = mantissa * s.astype(np.int64)
    mhi = (meff >> 8).astype(np.int16)
    mlo = (meff & 0xFF).astype(np.int16)
    factor_bits = ((exponent.astype(np.uint16) + 102) << 7).astype(np.uint16)
    data[0:16] = mhi.view(np.uint8)
    data[16:32] = mlo.view(np.uint8)
    data[32:48] = factor_bits.view(np.uint8)
    expected = np.concatenate([_q6k_expected(d, s, q) for q in payloads])
    return data, expected


PROBES = [
    Probe("u16_load_store", "U16_LOAD_STORE", True, make_u16, check_bytes),
    Probe("bf16_load_store", "BF16_LOAD_STORE", True, make_bf16, check_bytes),
    Probe(
        "f16_scalar_add",
        "F16_SCALAR_ADD",
        note="expect failure: scalar fp16 arithmetic",
    ),
    Probe(
        "f16_vector_decl",
        "F16_VECTOR_DECL",
        note="expect failure: vector<float16>",
    ),
    Probe(
        "f16_vector_load_store",
        "F16_VECTOR_LOAD_STORE",
        note="expect failure: fp16 load/store",
    ),
    Probe(
        "f16_size",
        "F16_SIZE",
        note="expect failure: sizeof(float16) != 2",
    ),
    Probe(
        "f16_int8_convert",
        "F16_INT8_CONVERT",
        note="expect failure: int8 -> fp16",
    ),
    Probe(
        "f16_acc_convert",
        "F16_ACC_CONVERT",
        note="expect failure: accfloat -> fp16",
    ),
    Probe("f16_add", "F16_ADD", note="expect failure: fp16 add"),
    Probe("f16_mul", "F16_MUL", note="expect failure: fp16 mul"),
    Probe("f16_mmul", "F16_MMUL", note="expect failure: fp16 mmul"),
    Probe(
        "fp16_bits_f32",
        "FP16_BITS_F32",
        True,
        make_fp16_bits,
        check_f32,
    ),
    Probe(
        "fp16_bits_split_bf16",
        "FP16_BITS_SPLIT_BF16",
        True,
        make_split,
        check_split,
    ),
    Probe(
        "fp16_shift_f32",
        "FP16_SHIFT_F32",
        True,
        make_fp16_bits,
        check_f32,
    ),
    Probe(
        "fp16_shift_split_bf16",
        "FP16_SHIFT_SPLIT_BF16",
        True,
        make_split,
        check_split,
    ),
    Probe(
        "fp16_direct_split_bf16",
        "FP16_DIRECT_SPLIT_BF16",
        True,
        make_fp16_bits,
        check_split_sum,
    ),
    Probe(
        "int8_scale_f32",
        "INT8_SCALE_F32",
        True,
        make_f32_scale,
        check_f32,
    ),
    Probe(
        "int8_scale_bf16",
        "INT8_SCALE_BF16",
        True,
        make_bf16_scale,
        check_bf16,
    ),
    Probe(
        "int8_scale_bf16_2term",
        "INT8_SCALE_BF16_2TERM",
        True,
        make_fp16_scale,
        check_bf16,
    ),
    Probe(
        "int8_scale_bf16_accum",
        "INT8_SCALE_BF16_ACCUM",
        True,
        make_fp16_scale,
        check_bf16,
    ),
    Probe(
        "int8_scale_int_split",
        "INT8_SCALE_INT_SPLIT",
        True,
        make_fp16_scale,
        check_bf16,
    ),
    Probe(
        "int8_scale_int_mul",
        "INT8_SCALE_INT_MUL",
        True,
        make_fp16_int_mul,
        check_int32,
    ),
    Probe(
        "int8_scale_int_round",
        "INT8_SCALE_INT_ROUND",
        True,
        make_fp16_int_round,
        check_bf16,
    ),
    Probe(
        "q6k_3term_apply",
        "Q6K_3TERM_APPLY",
        True,
        make_q6k_fixed,
        check_bf16,
    ),
    Probe(
        "q6k_3term_full",
        "Q6K_3TERM_FULL",
        True,
        make_q6k_full,
        check_bf16,
    ),
    Probe(
        "q6k_int_apply",
        "Q6K_INT_APPLY",
        True,
        make_q6k_int_fixed,
        check_bf16,
    ),
    Probe(
        "q6k_int_full",
        "Q6K_INT_FULL",
        True,
        make_q6k_full,
        check_bf16,
    ),
]


# ---------------------------------------------------------------------------
# Compile phase
# ---------------------------------------------------------------------------


def compile_probe(probe):
    """Compile one probe and return (ok, diagnostic)."""
    from aie.utils.compile.utils import compile_cxx_core_function

    output = TMP_ROOT / f"compile_{probe.name}.o"
    try:
        compile_cxx_core_function(
            source_path=str(SOURCE),
            target_arch="aie2p",
            output_path=str(output),
            compile_args=[f"-DBENCH_PROBE_{probe.macro}"],
            cwd=str(TMP_ROOT),
        )
        return True, ""
    except Exception as exc:  # noqa: BLE001 - compiler diagnostics are the result
        text = str(exc)
        errors = [
            line.strip()
            for line in text.splitlines()
            if "error:" in line
        ]
        if errors:
            return False, errors[0]
        tail = [line.strip() for line in text.splitlines() if line.strip()]
        return False, tail[-1] if tail else "unknown compiler failure"


# ---------------------------------------------------------------------------
# Run phase
# ---------------------------------------------------------------------------


def build_program(probe):
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

    flags = [f"-DBENCH_PROBE_{probe.macro}"]
    source = TMP_ROOT / f"fp16_ops_{probe.name}.aie.cc"
    shutil.copyfile(SOURCE, source)

    def build(
        inp: In,
        out: Out,
        *,
        trace_size: iron.CompileTime[np.int32] = 0,
        trace_config: iron.CompileTime[object] = None,
    ):
        in_ty = np.ndarray[(IN_BYTES,), np.dtype[np.uint8]]
        out_ty = np.ndarray[(OUT_BYTES,), np.dtype[np.uint8]]
        kernel = ExternalFunction(
            "bench_probe",
            object_file_name=f"bench_{probe.name}.o",
            source_file=str(source),
            arg_types=[in_ty, out_ty],
            compile_flags=flags,
        )
        in_fifo = ObjectFifo(in_ty, name="in", depth=1)
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

        def sequence(in_buf, out_buf, in_prod, out_cons):
            in_prod.fill(in_buf)
            out_cons.drain(out_buf, wait=True)

        runtime = Runtime(
            sequence,
            [in_ty, out_ty, in_fifo.prod(), out_fifo.cons()],
        )
        program = Program(NPU2(), runtime, [worker])
        if trace_size > 0:
            program.enable_trace(trace_size, workers=[worker])
        return program.resolve_program()

    return iron.jit(source_files=[str(source)], compile_flags=flags)(build)


def parse_trace_intervals(path):
    with open(path, "r") as trace_file:
        data = json.load(trace_file)

    by_tile = defaultdict(list)
    for entry in data:
        if entry.get("ph") != "B":
            continue
        name = entry.get("name")
        if name not in ("INSTR_EVENT_0", "INSTR_EVENT_1"):
            continue
        by_tile[entry.get("pid", 0)].append((entry["ts"], name))

    intervals = []
    leftover = False
    for events in by_tile.values():
        events.sort()
        stack = []
        for timestamp, name in events:
            if name == "INSTR_EVENT_0":
                stack.append(timestamp)
            else:
                if not stack:
                    raise RuntimeError("trace stack underflow")
                start = stack.pop()
                intervals.append((start, timestamp))
        leftover = leftover or bool(stack)
    if leftover:
        raise RuntimeError("trace stack not empty at end of stream")
    return intervals


def total_cycles(intervals):
    if not intervals:
        raise RuntimeError("no trace intervals found")
    return max(end for _, end in intervals) - min(
        start for start, _ in intervals
    )


def run_program(program, probe, seed):
    from aie import iron
    from aie.utils.trace import TraceConfig

    data, expected = probe.build_input(seed)
    in_tensor = iron.tensor(data, dtype=np.uint8)
    out = iron.zeros((OUT_BYTES,), dtype=np.uint8)

    # The trace ring occasionally leaves the closing event unflushed; the
    # kernel result is unaffected, so retry the whole run a few times.
    trace_file = TMP_ROOT / f"{probe.name}_trace.txt"
    trace_json = TMP_ROOT / f"{probe.name}_trace.json"
    trace_config = TraceConfig(
        trace_size=TRACE_SIZE, trace_file=str(trace_file)
    )
    last_error = None
    for _ in range(3):
        program(
            in_tensor,
            out,
            trace_size=TRACE_SIZE,
            trace_config=trace_config,
        )
        trace_config.trace_to_json(
            trace_config.physical_mlir_path, str(trace_json)
        )
        try:
            cycles = total_cycles(parse_trace_intervals(trace_json))
        except RuntimeError as exc:
            last_error = exc
            continue
        return np.asarray(out).copy(), expected, cycles

    raise last_error or RuntimeError("trace capture failed")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="XDNA2 float16 probes")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        help="run only the named probe (repeatable)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="compile every probe but skip device execution",
    )
    parser.add_argument("--list", action="store_true", help="list probes")
    return parser.parse_args()


def _shorten(text, limit=96):
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main():
    args = parse_args()
    probes = PROBES
    if args.probe:
        wanted = set(args.probe)
        probes = [probe for probe in probes if probe.name in wanted]
        missing = wanted - {probe.name for probe in probes}
        if missing:
            print(f"unknown probe(s): {', '.join(sorted(missing))}")
            return 2

    if args.list:
        for probe in probes:
            print(probe.name)
        return 0

    if not args.compile_only and not npu_available():
        print("ERROR: no NPU device available (use --compile-only)")
        return 1

    print(
        f"{'probe':<24} {'compile':<8} {'run':<5} "
        f"{'cycles':>8} {'cyc/iter':>9} {'max err':>10}  notes"
    )
    print("-" * 108)

    compiled = 0
    timed = 0
    for probe in probes:
        ok, diagnostic = compile_probe(probe)
        note = probe.note
        if not ok:
            note = diagnostic
        elif probe.timed:
            compiled += 1
            if args.compile_only:
                note = "compile-only mode"
            else:
                try:
                    program = build_program(probe)
                    out, expected, cycles = run_program(
                        program, probe, args.seed
                    )
                    error, check_note = probe.check(out, expected)
                    timed += 1
                    note = f"{note}; {check_note}" if note else check_note
                    print(
                        f"{probe.name:<24} {'yes':<8} {'yes':<5} "
                        f"{cycles:>8} {cycles / ITERS:>9.1f} "
                        f"{error:>10.4g}  {_shorten(note)}"
                    )
                    continue
                except NpuUnavailable as exc:
                    note = str(exc)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    note = _shorten(str(exc))
                print(
                    f"{probe.name:<24} {'yes':<8} {'FAIL':<5} "
                    f"{'-':>8} {'-':>9} {'-':>10}  {note}"
                )
                continue
        else:
            compiled += 1
            note = note or "compile-only probe"

        print(
            f"{probe.name:<24} "
            f"{('yes' if ok else 'no'):<8} {'-':<5} "
            f"{'-':>8} {'-':>9} {'-':>10}  {_shorten(note)}"
        )

    print("-" * 108)
    print(f"compiled {compiled}/{len(probes)} probes; timed {timed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
