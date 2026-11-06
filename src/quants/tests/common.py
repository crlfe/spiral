"""Shared helpers for the `src/quants` Python test harness."""

import ast
import ctypes
import functools
import glob
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

TESTS_ROOT = Path(__file__).resolve().parent
QUANTS_ROOT = TESTS_ROOT.parent
SRC_ROOT = QUANTS_ROOT.parent
_TMP_DIR = tempfile.TemporaryDirectory(prefix="quants_tests_")
TMP_ROOT = Path(_TMP_DIR.name)
_TRACE_SIZE = 4096


class SkipTest(Exception):
    """Raised when the NPU is unavailable or a test cannot run on this host."""


_CONST_RE = re.compile(
    r"^constexpr\s+(?:int|int64_t)\s+([A-Za-z_]\w*)\s*=\s*(.+);$"
)


def _eval_const_expr(expression, values, path, line_number):
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid constant expression {expression!r}"
        ) from exc

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise ValueError(
                    f"{path}:{line_number}: unknown identifier {node.id!r}"
                )
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div)
        ):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ValueError(
                    f"{path}:{line_number}: division by zero in {expression!r}"
                )
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            return int(left / right)
        raise ValueError(
            f"{path}:{line_number}: unsupported constant expression "
            f"{expression!r}"
        )

    return evaluate(tree.body)


def parse_layout(path):
    """Parse namespace-scope `constexpr int NAME = expression;` lines."""
    path = Path(path)
    values = {}
    for line_number, raw_line in enumerate(
        path.read_text().splitlines(), start=1
    ):
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            directive = line.split(None, 1)[0]
            if directive not in ("#pragma", "#include"):
                raise ValueError(
                    f"{path}:{line_number}: preprocessor conditionals are "
                    "not allowed in layout headers"
                )
            continue
        if "inline constexpr" in line:
            raise ValueError(
                f"{path}:{line_number}: inline constexpr is not allowed"
            )

        match = _CONST_RE.match(line)
        if not match:
            if line.startswith("constexpr"):
                raise ValueError(
                    f"{path}:{line_number}: unsupported constexpr declaration"
                )
            continue

        name, expression = match.groups()
        values[name] = _eval_const_expr(
            expression, values, path, line_number
        )

    return values


@functools.cache
def layout_constants(format_name):
    values = parse_layout(QUANTS_ROOT / "common" / "tile.h")
    values.update(parse_layout(QUANTS_ROOT / format_name / "layout.h"))
    return values


def _compiler():
    return os.environ.get("CXX", "g++")


def _build_shared(source, tag):
    path = TMP_ROOT / f"{tag}.so"
    command = [
        _compiler(),
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-I",
        str(SRC_ROOT),
        "-x",
        "c++",
        "-",
        "-o",
        str(path),
    ]
    result = subprocess.run(
        command,
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"compiler failed for {tag}:\n{result.stderr.strip()}"
        )
    return ctypes.CDLL(str(path))


def _host_shim_source(format_name):
    namespace = f"spiral::quants::{format_name}"
    return f"""
#include <stdint.h>

#include "quants/{format_name}/repack.h"

extern "C" {{
int64_t quants_record_count(int64_t weights_rows, int64_t weights_cols) {{
  return {namespace}::record_count(weights_rows, weights_cols);
}}

int64_t quants_packed_bytes(int64_t weights_rows, int64_t weights_cols) {{
  return {namespace}::packed_bytes(weights_rows, weights_cols);
}}

int64_t quants_repack(const uint8_t* src,
                      int64_t src_col_stride_bytes,
                      uint8_t* dst,
                      int64_t dst_capacity,
                      int64_t weights_rows,
                      int64_t weights_cols) {{
  return {namespace}::repack_weights(
      src, src_col_stride_bytes, dst, dst_capacity, weights_rows,
      weights_cols);
}}
}}
"""


@functools.cache
def host_shim(format_name):
    library = _build_shared(
        _host_shim_source(format_name), f"quants_host_{format_name}"
    )
    library.quants_record_count.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    library.quants_record_count.restype = ctypes.c_int64
    library.quants_packed_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64]
    library.quants_packed_bytes.restype = ctypes.c_int64
    library.quants_repack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    library.quants_repack.restype = ctypes.c_int64
    return library


def host_record_count(format_name, weights_rows, weights_cols):
    return host_shim(format_name).quants_record_count(
        weights_rows, weights_cols
    )


def host_packed_bytes(format_name, weights_rows, weights_cols):
    return host_shim(format_name).quants_packed_bytes(
        weights_rows, weights_cols
    )


def repack_host(format_name, raw, weights_rows, weights_cols, stride):
    """Run the host repack shim and return the packed bytes."""
    if weights_rows <= 0 or weights_cols <= 0:
        raise ValueError("weights_rows and weights_cols must be positive")
    if stride < 0:
        raise ValueError("stride must be non-negative")

    library = host_shim(format_name)
    expected_bytes = library.quants_packed_bytes(weights_rows, weights_cols)
    if expected_bytes <= 0:
        raise ValueError(
            f"{format_name}: invalid shape {weights_rows}x{weights_cols}"
        )

    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    if raw.size < weights_cols * stride:
        raise ValueError(
            f"{format_name}: source buffer is smaller than the strided view"
        )
    packed = np.zeros(expected_bytes, dtype=np.uint8)
    written = library.quants_repack(
        raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        stride,
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        expected_bytes,
        weights_rows,
        weights_cols,
    )
    if written != expected_bytes:
        raise RuntimeError(
            f"{format_name}: repack wrote {written} bytes, "
            f"expected {expected_bytes}"
        )
    return packed


def npu_available():
    if os.environ.get("QUANTS_SKIP_NPU"):
        return False
    if os.environ.get("QUANTS_FORCE_NPU"):
        return True
    patterns = ("/dev/accel/accel*", "/dev/dri/accel*")
    return any(glob.glob(pattern) for pattern in patterns)


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


def _write_shim(source, tag):
    path = TMP_ROOT / f"{tag}.cc"
    path.write_text(source)
    return path


def _npu_unpack_source(format_name, weights_rows, weights_cols):
    namespace = f"spiral::quants::{format_name}"
    return f"""
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/{format_name}/layout.h"
#include "quants/{format_name}/unpack.h"

extern "C" void quants_unpack_all(const uint8_t* packed, bfloat16* out) {{
  using namespace {namespace};
  const int32_t cols_tiles = {weights_cols} / kTileCols;
  const int32_t rows_records = {weights_rows} / kRecordRows;
  const int32_t chunk_elems = kUnpackTileRows * kTileCols;

  for (int32_t weights_cols_tile = 0; weights_cols_tile < cols_tiles;
       ++weights_cols_tile) {{
    for (int32_t weights_rows_record = 0;
         weights_rows_record < rows_records; ++weights_rows_record) {{
      const int32_t record_index =
          weights_cols_tile * rows_records + weights_rows_record;
      const uint8_t* record = packed + record_index * kPackedRecordBytes;

      for (int chunk_index = 0; chunk_index < kChunksPerRecord;
           ++chunk_index) {{
        bfloat16* tile =
            out + (record_index * kChunksPerRecord + chunk_index) *
                      chunk_elems;
        unpack_weights_chunk(common_scales(record),
                             chunk(record, chunk_index), tile);
      }}
    }}
  }}
}}
"""


def _npu_mmul_source(format_name, weights_rows, weights_cols):
    namespace = f"spiral::quants::{format_name}"
    return f"""
#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/mmul.h"
#include "quants/common/tile.h"
#include "quants/{format_name}/layout.h"
#include "quants/{format_name}/unpack.h"

extern "C" void quants_unpack_mmul_all(const uint8_t* packed,
                                       const bfloat16* inputs,
                                       float* outputs) {{
  using namespace {namespace};
  const int32_t cols_tiles = {weights_cols} / kTileCols;
  const int32_t rows_records = {weights_rows} / kRecordRows;
  constexpr int kWeightsRowsTiles =
      kUnpackTileRows / spiral::quants::kMmulWeightsRows;

  for (int32_t weights_cols_tile = 0; weights_cols_tile < cols_tiles;
       ++weights_cols_tile) {{
    float* outputs_tile =
        outputs + weights_cols_tile * spiral::quants::kMmulOutputsElems;
    for (int i = 0; i < spiral::quants::kMmulOutputsElems; ++i) {{
      outputs_tile[i] = 0.0f;
    }}

    for (int32_t weights_rows_record = 0;
         weights_rows_record < rows_records; ++weights_rows_record) {{
      const int32_t record_index =
          weights_cols_tile * rows_records + weights_rows_record;
      const uint8_t* record = packed + record_index * kPackedRecordBytes;

      for (int chunk_index = 0; chunk_index < kChunksPerRecord;
           ++chunk_index) {{
        alignas(64) bfloat16 weights_tile[kUnpackTileRows * kTileCols];
        unpack_weights_chunk(common_scales(record),
                             chunk(record, chunk_index), weights_tile);

        const int32_t weights_rows_start =
            weights_rows_record * kRecordRows +
            chunk_index * kUnpackTileRows;
        const bfloat16* inputs_slice =
            inputs +
            (weights_rows_start / spiral::quants::kMmulWeightsRows) *
                spiral::quants::kMmulInputsElems;
        spiral::quants::mmul_weights_tile<kWeightsRowsTiles>(
            inputs_slice, weights_tile, outputs_tile);
      }}
    }}
  }}
}}
"""


def _header_files(format_name, include_mmul=False):
    paths = [
        QUANTS_ROOT / "common" / "tile.h",
        QUANTS_ROOT / "common" / "trace.h",
        QUANTS_ROOT / format_name / "layout.h",
        QUANTS_ROOT / format_name / "unpack.h",
    ]
    if include_mmul:
        paths.append(QUANTS_ROOT / "common" / "mmul.h")
    return [str(path) for path in paths]


def _require_iron():
    try:
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
    except ImportError as exc:
        raise SkipTest(f"AIE/IRON is not installed: {exc}") from exc
    return (
        iron,
        ExternalFunction,
        In,
        ObjectFifo,
        Out,
        Program,
        Runtime,
        Worker,
        NPU2,
    )


def build_unpack_program(
    format_name, packed_bytes, output_elems, weights_rows, weights_cols
):
    (
        iron,
        ExternalFunction,
        In,
        ObjectFifo,
        Out,
        Program,
        Runtime,
        Worker,
        NPU2,
    ) = _require_iron()
    shim = _write_shim(
        _npu_unpack_source(format_name, weights_rows, weights_cols),
        f"quants_unpack_{format_name}_{weights_rows}x{weights_cols}",
    )
    source_files = [str(shim)] + _header_files(format_name)
    include_dirs = [str(SRC_ROOT)]

    def build(
        packed: In,
        out: Out,
        *,
        trace_size: iron.CompileTime[np.int32] = 0,
        trace_config: iron.CompileTime[object] = None,
    ):
        packed_ty = np.ndarray[(packed_bytes,), np.dtype[np.uint8]]
        out_ty = np.ndarray[(output_elems,), np.dtype[bfloat16]]

        kernel = ExternalFunction(
            "quants_unpack_all",
            source_file=str(shim),
            arg_types=[packed_ty, out_ty],
            include_dirs=include_dirs,
            compile_flags=["-DTRACE=1"],
        )
        packed_fifo = ObjectFifo(packed_ty, name="packed", depth=1)
        out_fifo = ObjectFifo(out_ty, name="weights", depth=1)

        def core_body(packed_cons, out_prod, kernel):
            packed_buf = packed_cons.acquire(1)
            out_buf = out_prod.acquire(1)
            kernel(packed_buf, out_buf)
            packed_cons.release(1)
            out_prod.release(1)

        worker = Worker(
            core_body,
            [packed_fifo.cons(), out_fifo.prod(), kernel],
            trace=1,
            stack_size=8192,
        )

        def sequence(packed_buf, out_buf, packed_prod, out_cons):
            packed_prod.fill(packed_buf)
            out_cons.drain(out_buf, wait=True)

        runtime = Runtime(
            sequence,
            [
                packed_ty,
                out_ty,
                packed_fifo.prod(),
                out_fifo.cons(),
            ],
        )
        program = Program(NPU2(), runtime, [worker])
        if trace_size > 0:
            program.enable_trace(trace_size, workers=[worker])
        return program.resolve_program()

    return iron.jit(source_files=source_files)(build)


def build_mmul_program(
    format_name,
    packed_bytes,
    inputs_elems,
    outputs_elems,
    weights_rows,
    weights_cols,
):
    (
        iron,
        ExternalFunction,
        In,
        ObjectFifo,
        Out,
        Program,
        Runtime,
        Worker,
        NPU2,
    ) = _require_iron()
    shim = _write_shim(
        _npu_mmul_source(format_name, weights_rows, weights_cols),
        f"quants_mmul_{format_name}_{weights_rows}x{weights_cols}",
    )
    source_files = [str(shim)] + _header_files(format_name, include_mmul=True)
    include_dirs = [str(SRC_ROOT)]

    def build(
        packed: In,
        inputs: In,
        outputs: Out,
        *,
        trace_size: iron.CompileTime[np.int32] = 0,
        trace_config: iron.CompileTime[object] = None,
    ):
        packed_ty = np.ndarray[(packed_bytes,), np.dtype[np.uint8]]
        inputs_ty = np.ndarray[(inputs_elems,), np.dtype[bfloat16]]
        outputs_ty = np.ndarray[(outputs_elems,), np.dtype[np.float32]]

        kernel = ExternalFunction(
            "quants_unpack_mmul_all",
            source_file=str(shim),
            arg_types=[packed_ty, inputs_ty, outputs_ty],
            include_dirs=include_dirs,
            compile_flags=["-DTRACE=1"],
        )
        packed_fifo = ObjectFifo(packed_ty, name="packed", depth=1)
        inputs_fifo = ObjectFifo(inputs_ty, name="inputs", depth=1)
        outputs_fifo = ObjectFifo(outputs_ty, name="outputs", depth=1)

        def core_body(packed_cons, inputs_cons, outputs_prod, kernel):
            packed_buf = packed_cons.acquire(1)
            inputs_buf = inputs_cons.acquire(1)
            outputs_buf = outputs_prod.acquire(1)
            kernel(packed_buf, inputs_buf, outputs_buf)
            packed_cons.release(1)
            inputs_cons.release(1)
            outputs_prod.release(1)

        worker = Worker(
            core_body,
            [
                packed_fifo.cons(),
                inputs_fifo.cons(),
                outputs_fifo.prod(),
                kernel,
            ],
            trace=1,
            stack_size=8192,
        )

        def sequence(
            packed_buf,
            inputs_buf,
            outputs_buf,
            packed_prod,
            inputs_prod,
            outputs_cons,
        ):
            packed_prod.fill(packed_buf)
            inputs_prod.fill(inputs_buf)
            outputs_cons.drain(outputs_buf, wait=True)

        runtime = Runtime(
            sequence,
            [
                packed_ty,
                inputs_ty,
                outputs_ty,
                packed_fifo.prod(),
                inputs_fifo.prod(),
                outputs_fifo.cons(),
            ],
        )
        program = Program(NPU2(), runtime, [worker])
        if trace_size > 0:
            program.enable_trace(trace_size, workers=[worker])
        return program.resolve_program()

    return iron.jit(source_files=source_files)(build)


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


def total_cycles(intervals):
    if not intervals:
        raise RuntimeError("no trace intervals found")
    return max(interval["end"] for interval in intervals) - min(
        interval["start"] for interval in intervals
    )


def run_traced(program, args, tag):
    try:
        from aie.utils.trace import TraceConfig
    except ImportError as exc:
        raise SkipTest(f"AIE trace utilities are not installed: {exc}") from exc

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
    trace_config.trace_to_json(
        trace_config.physical_mlir_path, str(trace_json)
    )
    return parse_trace_intervals(trace_json)
