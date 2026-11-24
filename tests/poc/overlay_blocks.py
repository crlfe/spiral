#!/usr/bin/env python3
"""Link-only PoC for overlaying the production dequant+mmul blocks.

This shows the build/linker shape needed by ``src/program``:

* the resident worker calls one external ``overlay_entry`` symbol;
* each format builds the same resident core plus a different overlay object;
* the overlay object is linked last, so the resident symbols and the overlay
  entry keep the same address across variants;
* the host can therefore block-write one variant's overlay tail into program
  memory while the core is idle, without relinking or reloading the xclbin.

The script uses the real Q4_0 and Q6_K block sources, not synthetic kernels.
It builds a minimal core around them because the point here is the layout; the
runtime block-write side is demonstrated separately by ``overlay.py``.

Run with the mlir-aie environment:

    python tests/poc/overlay_blocks.py
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from aie.iron import Buffer, ExternalFunction, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.utils import config as aie_config
from aie.utils.compile import compile_mlir_module

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROGRAM = ROOT / "src" / "program"
WORK_ROOT = HERE / "build" / "overlay_blocks"

if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))
import design  # noqa: E402

PEANO = Path(aie_config.peano_install_dir()) / "bin"
AIE_BIN = Path(aie_config.aiecc_path()).parent
INCLUDES = [
    f"-I{aie_config.cxx_header_path()}",
    f"-I{ROOT / 'src'}",
    f"-I{PROGRAM / 'kernels'}",
]
FLAGS = [
    "-std=c++20",
    "-O2",
    "-DNDEBUG",
    "-Wno-parentheses",
    "-Wno-attributes",
    "-Wno-macro-redefined",
    "-Wno-empty-body",
    "--system-header-prefix=aie_api/",
    "-Werror=undef",
    "-D__AIE_API_AIE_ADF_HPP__",
    "--target=aie2p-none-unknown-elf",
    "-ffunction-sections",
    "-fdata-sections",
    "-fstack-size-section",
    f"-DPROGRAM_BLOCK_BYTES={design.L1_STREAM_BYTES}",
    f"-DPROGRAM_MAX_TILES={design.MAX_TILES_PER_BLOCK}",
    f"-DPROGRAM_INPUT_ELEMS={design.INPUT_ELEMS}",
    f"-DPROGRAM_INPUT_GROUP_ROWS={design.INPUT_GROUP_ROWS}",
]

# The block's C signature, as an IRON type list.
BLOCK_ARGS = [
    design.l1_stream_type,
    design.l1_carry_type,
    design.l1_input_group_type,
    design.l1_outputs_type,
    design.l1_state_type,
    np.int32,
    np.int32,
    np.int32,
]


def _run(args, cwd: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = f"{PEANO}:{AIE_BIN}:" + env.get("PATH", "")
    proc = subprocess.run(
        list(map(str, args)), cwd=cwd, env=env, capture_output=True, text=True
    )
    if proc.returncode:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, proc.args)


def _build_overlay(src: Path, dst: Path) -> None:
    stem = src.name.replace(".aie.cc", "")
    _run(
        [
            PEANO / "clang++",
            *FLAGS,
            *INCLUDES,
            # Give the format block the stable name the resident core calls.
            f"-D{stem}=overlay_entry",
            "-c",
            src,
            "-o",
            dst,
        ],
        WORK_ROOT,
    )


def _make_variant(tag: str, overlay_obj: Path) -> Path:
    work = WORK_ROOT / tag
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / overlay_obj.name).write_bytes(overlay_obj.read_bytes())

    overlay = ExternalFunction(
        "overlay_entry",
        object_file_name=overlay_obj.name,
        source_string="",
        arg_types=BLOCK_ARGS,
    )
    block = Buffer(design.l1_stream_type, name="block")
    carry = Buffer(design.l1_carry_type, name="carry")
    inputs = Buffer(design.l1_input_group_type, name="inputs")
    outputs = Buffer(design.l1_outputs_type, name="outputs")
    state = Buffer(design.l1_state_type, name="state")

    def body(blk, car, inp, out, st, kernel):
        kernel(blk, car, inp, out, st, 2304, 1, 1)

    worker = Worker(
        body,
        [block, carry, inputs, outputs, state, overlay],
        tile=Tile(0, 2),
        stack_size=8192,
    )

    def sequence():
        pass

    program = Program(NPU2(), Runtime(sequence, []), [worker])
    mlir = program.resolve_program()
    (work / "raw.mlir").write_text(str(mlir))
    compile_mlir_module(mlir, work_dir=work, device=NPU2())
    _run(
        [
            "aie-opt",
            "--aie-substitute-shim-dma-allocations",
            "--aie-assign-runtime-sequence-bd-ids",
            "--aie-dma-tasks-to-npu",
            "--aie-dma-to-npu",
            "-o",
            "lowered.mlir",
            "input_with_addresses.mlir",
        ],
        work,
    )
    _run(
        ["aiecc", "--get-xclbin", "--tmpdir=xclbin_build", "lowered.mlir"],
        work,
    )
    return next((work / "xclbin_build").glob("elfs_*_core_0_2/*.elf"))


def _symbols(elf: Path) -> dict[str, int]:
    out = subprocess.check_output(["readelf", "-sW", str(elf)], text=True)
    result = {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].rstrip(":").isdigit():
            continue
        try:
            result[fields[7]] = int(fields[1], 16)
        except ValueError:
            pass
    return result


def _text(elf: Path) -> tuple[int, bytes]:
    """Return the loadable .text bytes and its load address."""
    data = elf.read_bytes()
    e_phoff = struct.unpack_from("<I", data, 28)[0]
    e_phentsize = struct.unpack_from("<H", data, 42)[0]
    e_phnum = struct.unpack_from("<H", data, 44)[0]
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_offset, _vaddr, p_paddr, p_filesz, _memsz, _flags, _align = (
            struct.unpack_from("<IIIIIIII", data, off)
        )
        if p_type == 1 and p_paddr == 0:
            return p_paddr, data[p_offset : p_offset + p_filesz]
    raise RuntimeError(f"{elf}: no loadable program-memory segment")


def _build_one(tag: str, source: str) -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    obj = WORK_ROOT / f"{tag}_overlay.o"
    _build_overlay(PROGRAM / "kernels" / source, obj)
    elf = _make_variant(tag, obj)
    symbols = _symbols(elf)
    start, text = _text(elf)
    entry = symbols["overlay_entry"]
    overlay = text[entry - start :]
    (WORK_ROOT / f"overlay_{tag}.bin").write_bytes(overlay)
    (WORK_ROOT / f"meta_{tag}.json").write_text(
        json.dumps(
            {
                "elf": str(elf),
                "symbols": symbols,
                "text_bytes": len(text),
                "overlay_entry": entry,
                "overlay_bytes": len(overlay),
            },
            indent=2,
        )
    )
    print(
        f"  {tag}: core=0x{symbols['core_0_2']:x} "
        f"overlay=0x{entry:x} ({len(overlay)} B) "
        f"text={len(text)} B"
    )


def main() -> None:
    if len(sys.argv) > 1:
        _build_one(sys.argv[1], sys.argv[2])
        return

    print("building Q4_0 and Q6_K block overlay variants")
    for tag, source in (
        ("q4", "program_q4_block.aie.cc"),
        ("q6", "program_q6_block.aie.cc"),
    ):
        subprocess.run([sys.executable, __file__, tag, source], check=True)

    meta = {
        tag: json.loads((WORK_ROOT / f"meta_{tag}.json").read_text())
        for tag in ("q4", "q6")
    }
    a_entry = meta["q4"]["overlay_entry"]
    b_entry = meta["q6"]["overlay_entry"]
    if a_entry != b_entry:
        raise SystemExit(
            f"overlay entry moved between variants: 0x{a_entry:x} vs 0x{b_entry:x}"
        )

    # Every symbol below the overlay entry is resident code/data.  They must
    # agree exactly, not just __start and the core function.
    def resident(tag):
        return {
            name: addr
            for name, addr in meta[tag]["symbols"].items()
            if 0 < addr < meta[tag]["overlay_entry"]
        }

    ra, rb = resident("q4"), resident("q6")
    if ra != rb:
        missing = sorted(set(ra) ^ set(rb))
        moved = sorted(k for k in ra.keys() & rb.keys() if ra[k] != rb[k])
        raise SystemExit(
            "resident symbols differ between variants: "
            f"missing={missing[:8]} moved={moved[:8]}"
        )

    print("\nresident symbols are identical; overlay entry is stable:")
    for name in ("__start", "core_0_2", "overlay_entry"):
        addr = meta["q4"]["symbols"].get(name, a_entry if name == "overlay_entry" else 0)
        print(f"  {name:20s} 0x{addr:x}")
    print(f"  resident symbols    {len(ra)} checked below overlay_entry")
    print("\noverlay tails written for the runtime loader:")
    for tag in ("q4", "q6"):
        m = meta[tag]
        end = m["overlay_entry"] + m["overlay_bytes"]
        if end > 0x4000:
            raise SystemExit(f"{tag}: overlay ends at 0x{end:x}, past 16 KiB")
        print(
            f"  {tag}: {m['overlay_bytes']:5d} B at 0x{m['overlay_entry']:x} "
            f"-> {WORK_ROOT / f'overlay_{tag}.bin'}"
        )


if __name__ == "__main__":
    main()
