#!/usr/bin/env python3
"""Build and run the TXN-concatenation / chained-matmul prototype.

The prototype links against the production build's generated TXN header and
xclbin, so build that first:

    ~/Projects/mlir-aie/ironenv/bin/python src/program/build.py --production

Then:

    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/txn_concat/run.py
"""

import subprocess
import sysconfig
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROGRAM = REPO / "src" / "program"
WORK = PROGRAM / "build-production"
SOURCE = Path(__file__).resolve().parent / "txn_concat.cc"
PROGRAM_SOURCE = PROGRAM / "host" / "program.cc"

CASES = (
    ("q4 aligned", "q4_0_1536x2048", "q4_0_2048x1536"),
    ("q4 partial group", "q4_0_12288x1536", "q4_0_1536x512"),
    ("q6_k square", "q6_k_2048x2048", "q6_k_2048x2048"),
    ("q6_k partial group", "q6_k_512x1536", "q6_k_1536x2048"),
    ("mixed q4 -> q6", "q4_0_1536x2048", "q6_k_2048x2048"),
)


def main():
    for needed in ("aie.xclbin", "generate_txn.h", "design_meta.h", "golden"):
        if not (WORK / needed).exists():
            raise SystemExit(
                f"missing {WORK / needed}; run "
                "`~/Projects/mlir-aie/ironenv/bin/python "
                "src/program/build.py --production` first"
            )

    include = Path(sysconfig.get_paths()["purelib"]) / "mlir_aie/include"
    if not include.exists():
        raise SystemExit(
            f"missing {include}; run this script with the mlir-aie ironenv "
            "python (see the module docstring)"
        )
    exe = WORK / "txn_concat"
    cmd = [
        "clang++",
        "-O2",
        "-g",
        "-std=c++17",
        f"-I{include}",
        f"-I{WORK}",
        f"-I{REPO / 'src'}",
        "-o",
        str(exe),
        str(SOURCE),
        str(PROGRAM_SOURCE),
        "-lxrt++",
        "-lxrt_coreutil",
    ]
    subprocess.run(cmd, check=True)

    for label, op0, op1 in CASES:
        print(f"\n== {label}: {op0} -> {op1} ==")
        subprocess.run(
            [str(exe), "golden", op0, op1, "20"],
            check=True,
            cwd=WORK,
        )

    print("\n== q4 GEGLU gate+up -> down ==")
    subprocess.run(
        [str(exe), "golden", "q4_0_1536x12288", "q4_0_6144x1536", "20",
         "geglu"],
        check=True,
        cwd=WORK,
    )

    print("\n== production Program GEGLU epilogue ==")
    subprocess.run(
        [str(exe), "golden", "q4_0_1536x12288", "q4_0_6144x1536", "20",
         "program"],
        check=True,
        cwd=WORK,
    )


if __name__ == "__main__":
    main()
