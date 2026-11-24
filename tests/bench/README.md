> Role: orientation. Current as of 2026-11-24. History: `git log -- tests/bench/README.md`.

# Bench experiments

Small standalone experiments that answer one question at a time about the AMD
XDNA2 NPU. This is a staging area for understanding and optimisation, not a
test suite:

- nothing under `src/` imports these files;
- nothing here is gated by CI;
- a proven helper moves to `src/quants/common/`;
- a confirmed finding is recorded in `docs/measurements.md` and, if it is a
  technique, in `src/quants/DESIGN.md`.

## Experiments

| file | question |
| :-- | :-- |
| `float16_support.py` | Which float16 operations does the compiler and XDNA2 accept? |
| `float16_convert.py` | How fast is raw float16 to f32? |
| `float16_scale.py` | How do we split and apply float16-derived scales with two or three bf16 terms? |

Each experiment is a Python driver plus a `.aie.cc` kernel shim. Shared
compile/run/trace code is in `bench_common.py`; shared device helpers are in
`float16_bench.aie.h`. Every probe reports cycles, values per iteration, a
correctness number against a NumPy reference, and the compiler diagnostic for
anything that does not build.

## Running

```sh
source tools/env.sh
cd tests/bench
python float16_scale.py
```

Useful flags:

```sh
python float16_scale.py --list            # probe names
python float16_scale.py --compile-only    # no device or tracing
python float16_scale.py --probe float16_shift_split_bf16
python float16_scale.py --seed 7
```

`--compile-only` uses the same Peano invocation as the JIT path, so it needs the
AIE environment but not the device. The trace ring occasionally drops the
closing `event1()`; the driver retries that run up to three times.

## Best choices

| condition | choice |
| :-- | :-- |
| Canonical two-byte float16 scales, exact, denorms required | exponent-add denorm, 32-wide: `float16_expadd_split_bf16_denorm` |
| Canonical two-byte float16 scales, exact, denorms can be flushed or host pre-biased | exponent-add flush, 32-wide: `float16_expadd_split_bf16_flush` |
| Host may re-encode the scale bytes, ~0.1% scale error is acceptable | packed 16-bit encoding, 32-wide: `float16_packed16_split_bf16` |
| Exact host-precomputed pair, and two extra bytes per scale is affordable | host bf16 hi/lo planes, as used by `q4_0_hilo` |
| Three-term Q6_K scale preparation | rounded hardware split with the 32-lane `q6k_3term_prep` |
| Two-term int8 scale application | `int8_scale_bf16_2term` |

The default probes are 32-wide because 32 bf16 lanes is the native 512-bit
width; `_x8` and `_x16` names remain for history and format-shaped cases. The
packed 16-bit format is the fastest two-term path and handles denormals without
a device branch, but it is not bit-exact. Cycles and error numbers are in
[`../../docs/measurements.md`](../../docs/measurements.md).

## Device and soak harnesses

`soak/` measures whole-system behaviour (`Tctl`, iGPU temperature, both fan
tachometers, NPU clocks) for the CPU, iGPU and NPU paths. The measurement
configuration, procedure and acceptance bar are in
[`../../docs/measurements.md`](../../docs/measurements.md); the harnesses verify
that configuration first and refuse to measure when it drifts.

| file | purpose |
| :-- | :-- |
| `soak/hostconfig.py` | read and verify the measurement configuration; prints restore commands and exits 2 on drift (`--allow-drift` to override) |
| `soak/soak.py` | long generation per device (`cpu`, `npu`, `igpu`) until temperatures and fan RPM plateau, then cool back to idle |
| `soak/headsoak.py` | NPU-only busy loop (`src/program ./test --head-soak`) with no CPU fallback |
| `soak/device_ab.py` | one short generation per device with cooldown and temperature/fan sampling |
| `soak/endpoint.py` | soak any local OpenAI-compatible endpoint, so servers can be compared on the same machine state |
| `soak/xdna.py` | read NPU clocks and power mode through `DRM_IOCTL_AMDXDNA_GET_INFO` |

```sh
python3 tests/bench/soak/hostconfig.py
python3 tests/bench/soak/soak.py npu 3000
python3 tests/bench/soak/headsoak.py
python3 tests/bench/soak/device_ab.py cpu 300
```

Paths are resolved from the repo and from hwmon names, not fixed indices. The
harnesses take `LLAMA_CPP_DIR`, `SPIRAL_MODEL` and `SPIRAL_PROGRAM_BUILD_DIR`
overrides for the llama.cpp checkout, the GGUF and the device build; defaults
are relative to the repository and `$HOME` and are listed in the script
docstrings. Power is deliberately not measured.

## Deferred

- **`MEMORY_STALL` trace event.** The default trace records it (AIE2P value
  23) but it has never fired for these probes. Revisit only if a kernel shows
  an unexplained cycle gap that looks like bank contention; the experiment is a
  matched same-bank/separate-bank pair with identical instruction counts.
- **Phasing out `_x8`/`_x16` probes** once no consumer needs them. The narrow
  variants remain for format-shaped cases such as a Q4_0 block's eight column
  scales.
- **A full 4-subgroup Q6_K apply demo** consuming the 32-lane prepared terms,
  if the Q4_0/Q6_K work needs it.
