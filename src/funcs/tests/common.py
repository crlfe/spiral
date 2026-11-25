"""Shared helpers for the `src/funcs` device tests.

These are deliberately format-independent: no quant layouts, repack or mmul
code enters this tree.
"""

import glob
import json
import os
import re
import struct
from collections import defaultdict
from pathlib import Path
import tempfile

import numpy as np
from ml_dtypes import bfloat16

TESTS_ROOT = Path(__file__).resolve().parent
FUNCS_ROOT = TESTS_ROOT.parent
SRC_ROOT = FUNCS_ROOT.parent
RMSNORM_ROOT = FUNCS_ROOT / "rmsnorm"
RMSNORM_HEADER = RMSNORM_ROOT / "rmsnorm.aie.h"
RMSNORM_LAYOUT = RMSNORM_ROOT / "layout.h"
RMSNORM_SHIM = TESTS_ROOT / "rmsnorm.aie.cc"
CORE_PROGRAM_MEMORY_BYTES = 16 * 1024
_TRACE_SIZE = 65536
_TMP_DIR = tempfile.TemporaryDirectory(prefix="funcs_tests_")
TMP_ROOT = Path(_TMP_DIR.name)


class NpuUnavailable(Exception):
    pass


def npu_available():
    if os.environ.get("FUNCS_NO_NPU"):
        return False
    if os.environ.get("FUNCS_FORCE_NPU"):
        return True
    return any(
        glob.glob(pattern)
        for pattern in ("/dev/accel/accel*", "/dev/dri/accel*")
    )


def is_device_error(exc):
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "no such device",
            "device not found",
            "no device",
            "device unavailable",
            "xrt",
            "amdxdna",
        )
    )


def layout_constants():
    """Parse the simple integer `constexpr` expressions in rmsnorm/layout.h."""
    values = {}
    pattern = re.compile(r"^constexpr int ([A-Za-z_]\w*) = (.+);$")
    for line in RMSNORM_LAYOUT.read_text().splitlines():
        line = line.split("//", 1)[0].strip()
        match = pattern.match(line)
        if not match:
            continue
        name, expression = match.groups()
        value = eval(expression, {"__builtins__": {}}, dict(values))
        values[name] = int(value)
    required = {
        "kChunk",
        "kVecs",
        "kRowStep",
        "kRowsPerGroup",
        "kChunksPerGroup",
        "kGroupInputElems",
    }
    missing = required - values.keys()
    if missing:
        raise RuntimeError(f"layout.h is missing {sorted(missing)}")
    return values


def kernel_compile_flags():
    return [
        "-Rpass-missed=loop-unroll",
        *os.environ.get("FUNCS_KERNEL_FLAGS", "").split(),
    ]


def build_rmsnorm_program(rows_steps, k, eps_bits, inv_k_bits):
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

    const = layout_constants()
    groups = (rows_steps + const["kRowsPerGroup"] - 1) // const["kRowsPerGroup"]
    input_elems = groups * const["kGroupInputElems"]
    input_ty = np.ndarray[(input_elems,), np.dtype[bfloat16]]
    weight_ty = np.ndarray[(2 * k,), np.dtype[bfloat16]]
    compile_flags = kernel_compile_flags()

    def build(
        inputs: In,
        weights: In,
        outputs: Out,
        *,
        trace_size: iron.CompileTime[np.int32] = 0,
        trace_config: iron.CompileTime[object] = None,
    ):
        kernel = ExternalFunction(
            "funcs_rmsnorm",
            arg_types=[
                input_ty,
                weight_ty,
                input_ty,
                np.int32,
                np.int32,
                np.int32,
            ],
            source_file=str(RMSNORM_SHIM),
            include_dirs=[str(SRC_ROOT)],
            compile_flags=compile_flags,
        )
        input_fifo = ObjectFifo(input_ty, name="inputs", depth=1)
        weight_fifo = ObjectFifo(weight_ty, name="weights", depth=1)
        output_fifo = ObjectFifo(input_ty, name="outputs", depth=1)

        def core_body(inputs_cons, weights_cons, outputs_prod, kernel):
            inputs_buf = inputs_cons.acquire(1)
            weights_buf = weights_cons.acquire(1)
            outputs_buf = outputs_prod.acquire(1)
            kernel(
                inputs_buf,
                weights_buf,
                outputs_buf,
                rows_steps,
                eps_bits,
                inv_k_bits,
            )
            inputs_cons.release(1)
            weights_cons.release(1)
            outputs_prod.release(1)

        worker = Worker(
            core_body,
            [
                input_fifo.cons(),
                weight_fifo.cons(),
                output_fifo.prod(),
                kernel,
            ],
            trace=1,
            stack_size=8192,
        )

        def sequence(
            inputs_buf,
            weights_buf,
            outputs_buf,
            inputs_prod,
            weights_prod,
            outputs_cons,
        ):
            inputs_prod.fill(inputs_buf)
            weights_prod.fill(weights_buf)
            outputs_cons.drain(outputs_buf, wait=True)

        runtime = Runtime(
            sequence,
            [
                input_ty,
                weight_ty,
                input_ty,
                input_fifo.prod(),
                weight_fifo.prod(),
                output_fifo.cons(),
            ],
        )
        program = Program(NPU2(), runtime, [worker])
        if trace_size > 0:
            program.enable_trace(trace_size, workers=[worker])
        return program.resolve_program()

    return iron.jit(
        source_files=[
            str(RMSNORM_SHIM),
            str(RMSNORM_HEADER),
            str(RMSNORM_LAYOUT),
        ],
        compile_flags=compile_flags,
    )(build)


def run_traced(program, args, tag):
    from aie.utils.trace import TraceConfig

    trace_file = TMP_ROOT / f"{tag}_trace.txt"
    trace_json = TMP_ROOT / f"{tag}_trace.json"
    trace_config = TraceConfig(
        trace_size=_TRACE_SIZE, trace_file=str(trace_file)
    )
    program(
        *args,
        trace_size=_TRACE_SIZE,
        trace_config=trace_config,
    )
    trace_config.trace_to_json(trace_config.physical_mlir_path, str(trace_json))
    kernel_dir = None
    if trace_config.physical_mlir_path:
        kernel_dir = Path(trace_config.physical_mlir_path).parent
    return parse_trace_intervals(trace_json), kernel_dir


def parse_trace_intervals(trace_path):
    with open(trace_path, "r") as trace_file:
        data = json.load(trace_file)
    by_tile = defaultdict(list)
    for entry in data:
        if entry.get("ph") != "B":
            continue
        name = entry.get("name")
        if name not in ("INSTR_EVENT_0", "INSTR_EVENT_1"):
            continue
        by_tile[entry.get("pid", 0)].append((entry["ts"], name))
    results = {}
    for pid, events in by_tile.items():
        events.sort()
        stack = []
        intervals = []
        for timestamp, name in events:
            if name == "INSTR_EVENT_0":
                stack.append((timestamp, len(stack)))
            else:
                if not stack:
                    raise RuntimeError("trace stack underflow")
                start, depth = stack.pop()
                intervals.append(
                    {
                        "depth": depth,
                        "start": start,
                        "end": timestamp,
                        "cycles": timestamp - start,
                    }
                )
        if stack:
            raise RuntimeError("trace stack not empty at end of stream")
        results[pid] = intervals
    return results


def flatten_intervals(intervals_by_tile):
    return sorted(
        (
            interval
            for intervals in intervals_by_tile.values()
            for interval in intervals
        ),
        key=lambda interval: interval["start"],
    )


def top_level_intervals(intervals_by_tile):
    return [
        interval
        for interval in flatten_intervals(intervals_by_tile)
        if interval["depth"] == 0
    ]


def total_cycles(intervals):
    if not intervals:
        raise RuntimeError("no trace intervals found")
    return max(interval["end"] for interval in intervals) - min(
        interval["start"] for interval in intervals
    )


def _elf_section(path, wanted):
    data = Path(path).read_bytes()
    if data[:6] != b"\x7fELF\x01\x01":
        raise ValueError(f"{path}: not a 32-bit little-endian ELF")
    section_offset = struct.unpack_from("<I", data, 0x20)[0]
    section_entry_size = struct.unpack_from("<H", data, 0x2E)[0]
    section_count = struct.unpack_from("<H", data, 0x30)[0]
    names_index = struct.unpack_from("<H", data, 0x32)[0]

    def section(index):
        return struct.unpack_from(
            "<IIIIII", data, section_offset + index * section_entry_size
        )

    names_offset, names_size = section(names_index)[4:6]
    names = data[names_offset : names_offset + names_size]
    for index in range(section_count):
        name_offset, _, _, _, offset, size = section(index)
        name = names[name_offset:].split(b"\0", 1)[0]
        if name == wanted:
            return data[offset : offset + size]
    return None


def _elf_text_size(path):
    section = _elf_section(path, b".text")
    if section is None:
        raise ValueError(f"{path}: no .text section")
    return len(section)


def _read_uleb128(data, offset):
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated ULEB128 in .stack_sizes")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("invalid ULEB128")


def _elf_max_stack_size(path):
    section = _elf_section(path, b".stack_sizes")
    if section is None:
        return None
    values = []
    offset = 0
    # One 4-byte function address then one ULEB128 frame size per entry; stop
    # before trailing alignment padding.
    while offset + 5 <= len(section):
        offset += 4
        value, offset = _read_uleb128(section, offset)
        values.append(value)
    return max(values) if values else 0


def print_program_size(kernel_dir):
    if kernel_dir is None:
        print("program memory: unavailable")
        return
    elfs = sorted(Path(kernel_dir).glob("elfs_*/*.elf"))
    if not elfs:
        print("program memory: unavailable (no core ELF)")
        return
    sizes = []
    for elf in elfs:
        text = _elf_text_size(elf)
        stack = _elf_max_stack_size(elf)
        sizes.append((text, stack))
    text, stack = sizes[0]
    stack_text = f"{stack} B" if stack is not None else "n/a"
    print(
        f"program memory: {text} B / {CORE_PROGRAM_MEMORY_BYTES} B "
        f"({100 * text / CORE_PROGRAM_MEMORY_BYTES:.1f}%), "
        f"max stack frame: {stack_text}"
    )
