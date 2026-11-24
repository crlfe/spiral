> Role: reference (pins). Current as of 2026-11-24. History: `git log -- docs/upstream.md`.

# Upstream references

Spiral builds against **installed wheels**, not against a source checkout of
mlir-aie. Upstream source is a reading reference: its docs, examples, headers
and TableGen give the machine model. Pin both, but do not confuse them.

## Toolchain of record

| component | version | location |
| :-- | :-- | :-- |
| `mlir_aie` Python package | `1.4.6.dev66+ga1fd967` | `ironenv/lib/python3.12/site-packages/mlir_aie` |
| Peano / `llvm-aie` | `22.0.0.2026102001+4f5df13d` | `ironenv/lib/python3.12/site-packages/llvm-aie` |
| mlir-aie source (reference) | `a1fd96791` (`latest-wheels-5-20-ga1fd96791`) | sparse clone, see below |
| llvm-aie source (reference) | `eafecd8716` (2026-08-06) | sparse clone, see below |

The installed `mlir_aie` wheel matches the mlir-aie source commit, so its docs
and examples describe the library that is installed. **The installed Peano is
not the same commit as the local llvm-aie checkout** (`4f5df13d` vs `eafecd8`).
Use the checkout's `AIE2Schedule.td` and `PERF_OPT_GUIDE.md` for the model, but
confirm instruction latencies against generated assembly whenever a schedule
argument depends on them.

## Where to read what

Read the installed headers first; they are the API that compiles:

| question | path |
| :-- | :-- |
| vector API, conversions, LUTs, mmul modes | `mlir_aie/include/aie_api/aie.hpp` and siblings |
| prebuilt kernels | `mlir_aie/include/aie_kernels/` |
| transaction encoding | `mlir_aie/include/aie/Runtime/TxnEncoding.h` |
| IRON Python API | `mlir_aie/python/aie/` |
| dialect and pass reference | mlir-aie `docs/AIEDesignPatterns.md`, `docs/AIE*Dialect.md` |
| end-to-end LLM design patterns | mlir-aie `programming_examples/ml/llama_3_2_1b/` |
| building mlir-aie from source | mlir-aie `docs/Building.md`, `README.md` |
| compressed upstream overview | mlir-aie `docs/llms.txt` (37 lines) |
| Peano kernel optimisation | llvm-aie `PERF_OPT_GUIDE.md` §2 restrict, §3 DM banks, §4 store width, §5-6 loop pragmas/hints, §7 pipelining, §8 loop versioning |
| VLIW slots and bundle forms | llvm-aie `llvm/lib/Target/AIE/AIE2Slots.td`, `AIE2MultiSlotPseudoInstrInfo.td` |
| instruction latencies and hazards | llvm-aie `llvm/lib/Target/AIE/AIE2Schedule.td` |
| assembler and bundle validation | llvm-aie `llvm/test/MC/AIE/` |

Latencies worth remembering (from `AIE2Schedule.td`): `vldb` 7, `vlda` 7,
`vmul`/`vmac` to accumulator 5, `vadd`/`vsub`/`vextbcst`/`vshuffle` 2, `vconv`
2, `vups` 3, `vsrs` 4, `vst` completion 5, ALU 1.

## Fetching the reference trees

`tools/fetch-refs.sh` sparse-clones both pins into `.refs/` (gitignored):

```sh
tools/fetch-refs.sh            # mlir-aie and llvm-aie
tools/fetch-refs.sh mlir-aie   # one of them
```

It uses `--filter=blob:none --sparse` and checks out only the directories used
above. That is tens of MB, not the full 343 MB history plus 367 MB board store
that a submodule would pull. Update a pin by editing the variables at the top
of the script and this table together.

## Path resolution

Nothing in the repository should hardcode `~/Projects/mlir-aie`. Source
`tools/env.sh` and use `python`:

```sh
source tools/env.sh
python src/program/build.py
```

`tools/env.sh` resolves the environment in this order and stops at the first
hit:

1. `$SPIRAL_MLIR_AIE` (explicit override),
2. `../mlir-aie` (the conventional sibling checkout),
3. `third_party/mlir-aie` (if a vendored tree is ever added),
4. `$MLIR_AIE_INSTALL_DIR` already set by the caller.

It sources that tree's `iron_env.sh` when present, otherwise it uses the
environment as given. `SPIRAL_PY` then points at the interpreter to use.
