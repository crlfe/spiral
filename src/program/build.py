"""Build the Spiral NPU design for the C++ transaction path."""

import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np

PROGRAM_DIR = Path(__file__).resolve().parent
if str(PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_DIR))

import design  # noqa: E402
import pack  # noqa: E402
import run  # noqa: E402
from aie.iron import ExternalFunction  # noqa: E402
from aie.iron.device import NPU2  # noqa: E402
from aie.utils import config as aie_config  # noqa: E402
from aie.utils.compile import (  # noqa: E402
    compile_external_kernels,
    compile_mlir_module,
)

WORK_DIR = PROGRAM_DIR / "build"
PRODUCTION = False
AIE_BIN = Path(aie_config.aiecc_path()).parent
PEANO_BIN = Path(aie_config.peano_install_dir()) / "bin"
AIE_INCLUDE = Path(sysconfig.get_paths()["purelib"]) / "mlir_aie/include"


def _tool(*args):
    env = os.environ.copy()
    env["PATH"] = (
        f"{PEANO_BIN}:{AIE_BIN}:{env.get('PATH', '')}"
    )
    subprocess.run(
        list(args), check=True, cwd=WORK_DIR, env=env
    )


def _build_q4_0_softcap(rows_steps, cols_bands):
    return pack.build_q4_0_case(rows_steps, cols_bands, soft_cap=30.0)


GOLDEN_CASES = (
    ("q4_0_1536x512", pack.build_q4_0_case, 6, 1),
    ("q4_0_1536x2048", pack.build_q4_0_case, 6, 4),
    ("q4_0_1536x4096", pack.build_q4_0_case, 6, 8),
    ("q4_0_1536x6144", pack.build_q4_0_case, 6, 12),
    ("q4_0_1536x12288", pack.build_q4_0_case, 6, 24),
    ("q4_0_1536x512_sc30", _build_q4_0_softcap, 6, 1),
    ("q4_0_2048x1536", pack.build_q4_0_case, 8, 3),
    ("q4_0_2048x2048", pack.build_q4_0_case, 8, 4),
    ("q4_0_4096x1536", pack.build_q4_0_case, 16, 3),
    ("q4_0_6144x1536", pack.build_q4_0_case, 24, 3),
    ("q4_0_12288x1536", pack.build_q4_0_case, 48, 3),
    ("q4_0_256x1536", pack.build_q4_0_case, 1, 3),
    ("q4_0_256x512", pack.build_q4_0_case, 1, 1),
    ("q4_0_512x512", pack.build_q4_0_case, 2, 1),
    ("q4_0_1024x512", pack.build_q4_0_case, 4, 1),
    ("q4_0_1792x4608", pack.build_q4_0_case, 7, 9),
    ("q4_0_2048x4096", pack.build_q4_0_case, 8, 8),
    ("q4_0_1792x8192", pack.build_q4_0_case, 7, 16),
    ("q4_0_1536x9728", pack.build_q4_0_case, 6, 19),
    ("q4_0_3840x3840", pack.build_q4_0_case, 15, 15),
    ("q6_k_512x1536", pack.build_q6_k_case, 2, 3),
    ("q6_k_1792x5632", pack.build_q6_k_case, 7, 11),
    ("q6_k_1792x11264", pack.build_q6_k_case, 7, 22),
    ("q6_k_1536x2048", pack.build_q6_k_case, 6, 4),
    ("q6_k_2048x2048", pack.build_q6_k_case, 8, 4),
    ("q6_k_1536x4096", pack.build_q6_k_case, 6, 8),
    ("q6_k_1536x6144", pack.build_q6_k_case, 6, 12),
)

# Explicit 256-column cases padded to the 512-column device band.
PADDED_CASES = (
    ("q4_0_1536x256p512", 6, 1, 256),
    ("q4_0_2048x256p512", 8, 1, 256),
)

# One contiguous Q6_K head tensor split into band chunks, used to check the
# wrapper's weight-column and output-column offsets.
HEAD_GOLDEN = (
    "q6_k_head", 6, 512, design.MAX_BANDS_PER_CHUNK
)  # rows_steps, total_bands, chunk_bands


def _emit_design_meta():
    """Single-source the fixed design constants for the C++ host."""
    int32_values = {
        "kGroups": design.GROUPS,
        "kComputesPerGroup": design.COMPUTES_PER_GROUP,
        "kInputsVecs": design.INPUTS_VECS,
        "kRowsStep": design.ROWS_STEP,
        "kColsTile": design.COLS_TILE,
        "kBandCols": design.BAND_COLS,
        "kInputElems": design.INPUT_ELEMS,
        "kInputGroupRows": design.INPUT_GROUP_ROWS,
        "kInputGroupElems": design.INPUT_GROUP_ELEMS,
        "kMaxInputGroups": (
            design.MAX_ROWS_STEPS + design.INPUT_GROUP_ROWS - 1
        )
        // design.INPUT_GROUP_ROWS,
        "kOutputTileElems": design.OUTPUT_TILE_ELEMS,
        "kOutputElems": design.OUTPUT_ELEMS,
        "kGroupOutputs": design.GROUP_OUTPUTS,
        "kMaxWeightBlocks": design.MAX_WEIGHT_BLOCKS,
        "kMaxBandsPerChunk": design.MAX_BANDS_PER_CHUNK,
        "kMaxRowsSteps": design.MAX_ROWS_STEPS,
        "kMaxTotalTiles": (
            design.MAX_ROWS_STEPS * design.MAX_BANDS_PER_CHUNK
        ),
        "kCfgFormat": design.CFG_FORMAT,
        "kCfgTileBytes": design.CFG_TILE_BYTES,
        "kCfgRowsSteps": design.CFG_ROWS_STEPS,
        "kCfgColsBands": design.CFG_COLS_BANDS,
        "kCfgWeightsBlocks": design.CFG_WEIGHTS_BLOCKS,
        "kCfgTotalTiles": design.CFG_TOTAL_TILES,
        "kCfgInputPad": design.CFG_INPUT_PAD,
        "kCfgWeightPad": design.CFG_WEIGHT_PAD,
        "kCfgSoftCapHiLo": design.CFG_SOFT_CAP_HI_LO,
        "kCfgSoftInvHiLo": design.CFG_SOFT_INV_HI_LO,
        "kCfgEpilogue": design.CFG_EPILOGUE,
        "kConfigLen": design.CONFIG_LEN,
        "kFormatQ4_0": design.FORMAT_Q4_0,
        "kFormatQ6K": design.FORMAT_Q6_K,
        "kEpilogueNone": design.EPILOGUE_NONE,
        "kEpilogueSoftcap": design.EPILOGUE_SOFTCAP,
        "kEpilogueGeglu": design.EPILOGUE_GEGLU,
        "kModeBase": design.MODE_BASE,
        "kModeFifo": design.MODE_FIFO,
        "kModeRawCheck": design.MODE_RAW_CHECK,
        "kModeTileCheck": design.MODE_TILE_CHECK,
        "kModeLast": design.MODE_LAST,
        "kProduction": 1 if PRODUCTION else 0,
    }
    size_values = {
        "kL1StreamBytes": design.L1_STREAM_BYTES,
        "kL2StreamBytes": design.L2_STREAM_BYTES,
        "kWeightWords": design.MAX_WEIGHT_WORDS,
        "kInputWords": design.MAX_INPUT_WORDS,
        "kMaxOutputElems": design.MAX_OUTPUT_ELEMS,
    }
    lines = [
        "// Generated by build.py; do not edit.",
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "namespace design {",
    ]
    lines += [
        f"constexpr int32_t {name} = {value};"
        for name, value in int32_values.items()
    ]
    lines += [
        f"constexpr size_t {name} = {value};"
        for name, value in size_values.items()
    ]
    lines += ["}  // namespace design", ""]
    (WORK_DIR / "design_meta.h").write_text("\n".join(lines))


def _write_golden(golden_dir, name, case, extra_meta=None):
    raw = np.ascontiguousarray(case["raw"], dtype=np.uint8)
    # Keep bfloat16 bit patterns; a dtype= cast would convert them
    # numerically instead of reinterpreting.
    inputs = np.ascontiguousarray(case["inputs_matrix"]).view(np.uint16)
    expected = np.ascontiguousarray(case["expected"], dtype=np.float32)
    (golden_dir / f"{name}.raw").write_bytes(raw.tobytes())
    (golden_dir / f"{name}.inputs").write_bytes(
        inputs.astype("<u2").tobytes()
    )
    (golden_dir / f"{name}.expected").write_bytes(
        expected.astype("<f4").tobytes()
    )
    meta = {
        "format": case["format"],
        "rows_steps": case["rows_steps"],
        "cols_bands": case["cols_bands"],
        "tile_bytes": case["tile_bytes"],
        "weights_blocks": case["weights_blocks"],
        "total_tiles": case["total_tiles"],
        "weights_rows": case["weights_rows"],
        "weights_cols": case["weights_cols"],
        "raw_stride": case["raw_stride"],
        "raw_bytes": raw.size,
        "inputs_elems": inputs.size,
        "expected_count": expected.size,
        "soft_cap_bits": int(
            np.array([case.get("soft_cap", 0.0)], dtype=np.float32).view(
                np.int32
            )[0]
        ),
    }
    if extra_meta:
        meta.update(extra_meta)
    (golden_dir / f"{name}.meta").write_text(
        "".join(f"{k} {v}\n" for k, v in meta.items())
    )


def _emit_golden():
    """Write canonical weights, inputs, and reference outputs for C++ checks.

    The C++ driver repacks the canonical weights with the real
    ``src/quants/*/repack.h`` code, so these files only need to carry the
    host-side input data and the Python/NumPy reference.
    """
    golden_dir = WORK_DIR / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, build, rows_steps, cols_bands in GOLDEN_CASES:
        _write_golden(golden_dir, name, build(rows_steps, cols_bands))
        entries.append((name, "matmul"))

    for name, rows_steps, cols_bands, real_cols in PADDED_CASES:
        case = pack.build_q4_0_case(
            rows_steps, cols_bands, weights_cols_real=real_cols
        )
        _write_golden(golden_dir, name, case)
        entries.append((name, "matmul"))

    head_name, head_rows, total_bands, chunk_bands = HEAD_GOLDEN
    head = pack.build_q6_k_head_case(
        head_rows, total_bands, chunk_bands
    )
    _write_golden(
        golden_dir, head_name, head, {"chunk_bands": chunk_bands}
    )
    entries.append((head_name, "head"))
    (golden_dir / "index.txt").write_text(
        "".join(f"{name} {mode}\n" for name, mode in entries)
    )


def main():
    global WORK_DIR, PRODUCTION
    PRODUCTION = "--production" in sys.argv[1:]
    WORK_DIR = PROGRAM_DIR / (
        "build-production" if PRODUCTION else "build"
    )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    # compile_external_kernels skips existing objects, even after a kernel
    # source change.  Clear them so a rebuild always reflects the sources.
    for obj in WORK_DIR.glob("*.o"):
        obj.unlink()

    builder = (
        run.production_dynamic_program
        if PRODUCTION
        else run.dynamic_program
    )
    mlir = builder.as_mlir()
    compile_external_kernels(
        ExternalFunction._instances, WORK_DIR, "aie2p"
    )
    compile_mlir_module(mlir, work_dir=WORK_DIR, device=NPU2())

    _tool(
        str(AIE_BIN / "aie-opt"),
        "--aie-substitute-shim-dma-allocations",
        "--aie-assign-runtime-sequence-bd-ids",
        "--aie-dma-tasks-to-npu",
        "--aie-dma-to-npu",
        "-o",
        "lowered.mlir",
        "input_with_addresses.mlir",
    )
    _tool(
        str(AIE_BIN / "aie-translate"),
        "--aie-npu-to-cpp",
        "-o",
        "generate_txn.h",
        "lowered.mlir",
    )

    txn_path = WORK_DIR / "generate_txn.h"
    txn_src = txn_path.read_text()
    patched, count = re.subn(
        r"return\s+std::nullopt\s*;",
        'throw std::runtime_error(std::string("Overflow at ") + __FILE__ + '
        '":" + std::to_string(__LINE__));',
        txn_src,
    )
    if count:
        txn_path.write_text("#include <stdexcept>\n" + patched)

    if PRODUCTION:
        # Reuse the diagnostic build's golden data when available; production
        # builds do not need to regenerate the large head tensor.
        shared = PROGRAM_DIR / "build" / "golden"
        local = WORK_DIR / "golden"
        if shared.exists() and not local.exists():
            local.symlink_to(shared, target_is_directory=True)
        elif not local.exists():
            _emit_golden()
    else:
        _emit_golden()
    _emit_design_meta()

    _tool(
        str(AIE_BIN / "aiecc"),
        "--get-xclbin",
        "--tmpdir=xclbin_build",
        "lowered.mlir",
    )
    _compile_test()
    print(f"artifacts in {WORK_DIR}")
    print(f"run: cd {WORK_DIR} && ./test")


def _compile_test():
    subprocess.run(
        [
            "clang++",
            "-O2",
            "-g",
            f"-I{AIE_INCLUDE}",
            f"-I{PROGRAM_DIR.parent}",
            f"-I{WORK_DIR}",
            "-o",
            str(WORK_DIR / "test"),
            str(PROGRAM_DIR / "host" / "test.cc"),
            str(PROGRAM_DIR / "host" / "program.cc"),
            "-lxrt++",
            "-lxrt_coreutil",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
