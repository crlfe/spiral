#!/usr/bin/env python3
"""Shared harness for the standalone float16 bench experiments.

Each experiment lives in its own Python driver and `.aie.cc` shim so that one
unsupported operation fails only its own row and the driver can be run without
the others. This module holds the parts that are identical across experiments:
probe metadata, isolated compilation, the IRON runtime wrapper, trace parsing,
and the result table.
"""

import argparse
import glob
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IN_BYTES = 8192
OUT_BYTES = 8192
ITERS = 32
TRACE_SIZE = 65536

_TMP_DIR = tempfile.TemporaryDirectory(prefix="bench_float16_")
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
    values_per_iter: int = 8


def _seed_input(seed):
    return np.random.default_rng(seed)


def compile_probe(probe, source):
    """Compile one probe and return (ok, diagnostic)."""
    from aie.utils.compile.utils import compile_cxx_core_function

    output = TMP_ROOT / f"compile_{probe.name}.o"
    try:
        compile_cxx_core_function(
            source_path=str(source),
            target_arch="aie2p",
            output_path=str(output),
            compile_args=[f"-DBENCH_PROBE_{probe.macro}"],
            cwd=str(TMP_ROOT),
        )
        return True, ""
    except Exception as exc:  # noqa: BLE001 - compiler diagnostics are the result
        text = str(exc)
        errors = [
            line.strip() for line in text.splitlines() if "error:" in line
        ]
        if errors:
            return False, errors[0]
        tail = [line.strip() for line in text.splitlines() if line.strip()]
        return False, tail[-1] if tail else "unknown compiler failure"


def build_program(probe, source):
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

    flags = [f"-DBENCH_PROBE_{probe.macro}", f"-I{source.parent}"]
    local_source = TMP_ROOT / f"{source.stem}_{probe.name}.aie.cc"
    shutil.copyfile(source, local_source)

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
            source_file=str(local_source),
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

    return iron.jit(source_files=[str(local_source)], compile_flags=flags)(build)


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


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
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


def run(description, source, probes):
    args = parse_args(description)
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
        f"{'probe':<40} {'compile':<8} {'run':<5} "
        f"{'cycles':>8} {'vals/iter':>9} {'cyc/iter':>9} "
        f"{'max err':>10}  notes"
    )
    print("-" * 136)

    compiled = 0
    timed = 0
    for probe in probes:
        ok, diagnostic = compile_probe(probe, source)
        note = probe.note
        if not ok:
            note = diagnostic
        elif probe.timed:
            compiled += 1
            if args.compile_only:
                note = "compile-only mode"
            else:
                try:
                    program = build_program(probe, source)
                    out, expected, cycles = run_program(
                        program, probe, args.seed
                    )
                    error, check_note = probe.check(out, expected)
                    timed += 1
                    note = f"{note}; {check_note}" if note else check_note
                    print(
                        f"{probe.name:<40} {'yes':<8} {'yes':<5} "
                        f"{cycles:>8} {probe.values_per_iter:>9} "
                        f"{cycles / ITERS:>9.1f} "
                        f"{error:>10.4g}  {_shorten(note)}"
                    )
                    continue
                except NpuUnavailable as exc:
                    note = str(exc)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    note = _shorten(str(exc))
                print(
                    f"{probe.name:<40} {'yes':<8} {'FAIL':<5} "
                    f"{'-':>8} {'-':>9} {'-':>9} {'-':>10}  {note}"
                )
                continue
        else:
            compiled += 1
            note = note or "compile-only probe"

        print(
            f"{probe.name:<40} "
            f"{('yes' if ok else 'no'):<8} {'-':<5} "
            f"{'-':>8} {'-':>9} {'-':>9} {'-':>10}  {_shorten(note)}"
        )

    print("-" * 136)
    print(f"compiled {compiled}/{len(probes)} probes; timed {timed}")
    return 0
