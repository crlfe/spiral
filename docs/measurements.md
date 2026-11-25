> Role: reference (numbers). Current as of 2026-11-25. History: `git log -- docs/measurements.md`.

# Measurements

The single home for performance numbers. Other documents link here instead of
restating figures. Every table names the build, the machine state and the date
where it is known; if a number has none of those, it is not quotable.

Machine: Ryzen AI 9 HX 475 (XDNA2, `aie2p`, arch 21), 64 GB.
`/dev/accel/accel0`; targets from `tests/poc`, `src/quants`, `src/program`.

## 1. How to measure

### Configuration baseline

Numbers are only comparable when the machine is in the same state. The baseline
is the normal boot state plus two deliberate changes: iGPU SCLK capped at
2400 MHz and NPU pmode `turbo`.

| setting | value | source |
| :-- | :-- | :-- |
| NPU power mode | `turbo` | `DRM_AMDXDNA_GET_POWER_MODE` |
| iGPU DPM force level | `manual` | `power_dpm_force_performance_level` |
| iGPU SCLK OD bounds | 600 / 2400 MHz | `pp_od_clk_voltage` |
| platform profile | `balanced` | `/sys/firmware/acpi/platform_profile` |
| CPU governor | `powersave` on every policy | `cpufreq/policy*/scaling_governor` |
| CPU EPP | `balance_performance` (warning only) | `cpufreq/policy*/energy_performance_preference` |
| PM QoS | off (`SPIRAL_DMA_LATENCY` unset) | environment |

`tests/bench/soak/hostconfig.py` verifies all of it, prints restore commands,
and exits `2` on drift. `--allow-drift` reports and continues; those runs must
not be compared with the tables here.

fclk and socclk are deliberately **not** pinned: pinning them measured no
change, so they stay on DVFS and are printed as observations. The iGPU SCLK cap
is the one setting that drifts silently (a reboot or driver reload restores the
3100 MHz hardware range).

### Procedure

- Cool to the idle baseline: fans off, `Tctl` below ~45 C.
- Run a long greedy generation per device (thousands of tokens), sampling about
  once per 0.5 s, until `Tctl` and fan RPM have been flat for 20 s.
- Report peak `Tctl`, peak fan RPM, throughput, and both fans.
- Cool back to idle (fan 0, `Tctl` within 2 C) before the next device.
- Short runs mislead: the iGPU and CPU look fanless until the task outlasts the
  thermal time constant.

### Signals

| signal | source |
| :-- | :-- |
| CPU temperature | `k10temp` `temp1_input` (Tctl) |
| iGPU temperature | `amdgpu` hwmon `temp1_input` |
| fan speed | both `hp` fans (`fan1_input`, `fan2_input`) |
| host CPU utilisation | `/proc/stat` busy delta |
| NPU clocks | `DRM_IOCTL_AMDXDNA_GET_INFO` (MP-NPU and H) |
| iGPU clock | `amdgpu` hwmon `pp_dpm_sclk` / `amdgpu` sysfs |
| throughput | `pp8`, `tg16`, a 300-token generation |

Both `hp` fans follow one EC curve and always move together (150-200 RPM
apart). The curve has a knee just above 53 C: 53.1 C can finish at zero RPM
while 54.0 C sits at 2.8k, and CPU-heavy runs jump to ~4.8k.

**Power is not measured.** RAPL `energy_uj` is root-only in the measurement
container and `SOCKET_POWER` is too coarse, so fan RPM and `Tctl` are the proxy.

### Acceptance bar

Productive-and-quiet, not peak throughput: a usable rate with fan RPM within
~150 of idle and `Tctl` within ~10 C of idle. No current path meets it (the
quietest, FastFlowLM, still reaches 54 C and 2.8k RPM), so it is a target, not a
gate.

## 2. Device and engine comparison

`gemma-4-E2B_q4_0-it.gguf` (Gemma 4 E2B Q4_0, native Q6_K head). 3000-token
greedy soaks, one submission thread, peaks after the 20 s plateau window,
measured 2026-11 under the configuration baseline.

| engine | tokens | t/s | peak Tctl | peak iGPU | fan1 | fan2 | plateau | cool-back |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| iGPU (`ROCm0`, gfx1150) | 3000 | 52.42 | 54.2 C | 54.0 C | 2793 | 2597 | none in 59 s | 123 s |
| FastFlowLM (`gemma4-it:e2b`) | 3000 | 25.75 | 54.0 C | 48.0 C | 2818 | 2626 | none in 117 s | 194 s |
| CPU (12 threads) | 3000 | 22.17 | 91.0 C | 56.0 C | 4845 | 4657 | none in 138 s | 126 s |
| Spiral NPU | 3000 | 16.15 | 73.6 C | 56.0 C | 4834 | 4657 | 65 s | 121 s |

FastFlowLM uses its own quantisation and packing, so that row compares engine
classes; the other three share the GGUF. The CPU row varied 19.6-22.8 t/s
between runs at the same settings.

The NPU path's heat is mostly host work: an NPU-only busy loop
(`src/program ./test --head-soak 20000`, 1% CPU busy) reaches `Tctl` 60.6 C,
iGPU 49.0 C and 2818/2626 RPM over 152 s. The extra heat during decode is the
CPU fallback and host submission. The same head loop runs ~4% faster under
pmode `turbo` (152.3 s vs 155-159 s).

### Per-call path, 300-token answer

30-token prompt, 300-token greedy answer, native Q6_K head, one CPU fallback
thread, `llama-completion -no-cnv --ignore-eos --temp 0`.

| device | pp8 | tg16 | 300-token generation | per token |
| :-- | --: | --: | --: | --: |
| iGPU (`ROCm0`) | 383.1 t/s | 55.30 t/s | 55.11 t/s | 18.1 ms |
| CPU (`-ngl 0`, 12 threads) | 224.1 t/s | 35.34 t/s | 35.78 t/s | 28.0 ms |
| Spiral NPU | 111.5 t/s | 18.47 t/s | 17.78 t/s | 56.2 ms |

All three stream ~1.4 GB/token of active weights. The NPU path adds the ~80 us
per-call floor plus the PCIe hop; decode makes ~240 calls/token, so ~19 ms/token
is fixed launch/fill overhead.

Thread counts, `tg16` / `pp8` on the NPU path: 1 thread 18.47 / 111.5,
2 16.8 / 106, 4 14.5 / 95, 8 10.5 / 68. CPU inference peaks at 12 threads
(36.1 / 225). These were taken before the pmode baseline; treat them as ratios.

### Batched decode rows (parallel slots)

Gemma 4 E2B Q4_0, native Q6_K head. `llama-batched-bench` with
`-dev Spiral0 -ngl 999 -npp 64 -ntg 128`, `-c 4096 -b 2048 -ub 512 -fa off`,
`SPIRAL_CPU_THREADS=1`, separate prompts (no `-pps`). One fresh process per
`npl`; the numbers were taken after the device had recovered from a transient
TDR cascade (see `TODO.md`). Build: llama.cpp `d26efdb`, `src/program`
`1ab7793` plus the zero-row no-op fix, `build.py --production`. Measured
2026-11-24.

| npl | T_PP s | S_PP t/s | T_TG s | S_TG t/s | per-slot t/s | step ms | vs npl=1 |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 0.249 | 256.5 | 8.289 | 15.44 | 15.44 | 64.8 | 1.00x |
| 2 | 1.051 | 121.8 | 8.543 | 29.97 | 14.99 | 66.7 | 1.94x |
| 3 | 1.537 | 125.0 | 8.670 | 44.29 | 14.76 | 67.7 | 2.87x |
| 4 | 2.064 | 124.0 | 8.948 | 57.22 | 14.31 | 69.9 | 3.71x |

The backend chunks batch rows by `kSpiralInputsVecs = 4`, and the NPU mmul is
4 rows wide, so npl 1-4 share one weight sweep per decode step. Step time grows
8% while four tokens are produced per step. Effective wall bandwidth is nearly
constant at ~20-21 GB/s (`1.38 GB/token * npl / T_TG`): the gain is amortising
bytes over four rows, not higher bandwidth. npl=1 settles at 15.4-15.6 t/s once
warm; a cold first run measured 13.6 t/s.

Short-run thermal context (each run ~15 s, fans off throughout, so not the
steady-state acceptance measurement):

| npl | idle Tctl | peak Tctl | peak iGPU | peak fan |
| --: | --: | --: | --: | --: |
| 1 | 35.9 C | 61.1 C | 41.0 C | 0 |
| 2 | 43.9 C | 68.6 C | 45.0 C | 0 |
| 3 | 44.8 C | 70.1 C | 45.0 C | 0 |
| 4 | 45.0 C | 72.0 C | 47.0 C | 0 |

Prefill aggregate falls from ~256 t/s for one sequence to ~124 t/s for multiple
sequences. The four-row limit is the current design; npl 8/16 would re-stream
weights once per four rows and are not measured.

#### Steady-state 4-slot soak

Same model and command shape, but long enough to plateau: `npp 64`, separate
prompts, `-fa off`, `SPIRAL_CPU_THREADS=1`. NPU `ntg 2048`; iGPU `ntg 4096` in
the ROCm build. Configuration baseline verified; measured 2026-11-24 after
`4959ba1` and the production xclbin.

| engine | slots | aggregate t/s | per-slot t/s | step ms | peak Tctl | peak iGPU | fan1/fan2 | plateau | cool-back |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Spiral NPU | 1 | 15.30 | 15.30 | 65.3 | 62.9 C | 47.0 C | 2837/2645 | none in 139 s | 93 s |
| Spiral NPU | 4 | **54.44** | 13.61 | 73.5 | 67.2 C | 48.0 C | 2782/2636 | 82 s (75 s steady) | 103 s |
| iGPU (`ROCm0`) | 4 | **87.88** | 21.97 | 45.5 | 60.9 C | 61.0 C | 3648/3348 | 57 s (131 s steady) | 100 s |

The NPU's four slots are a **3.56x** throughput gain over one slot for +4.3 C
peak Tctl and essentially unchanged peak fan RPM; per-token heat drops sharply.
The iGPU is 61% faster in aggregate but runs both itself and the fans hotter.
Neither device meets the quiet acceptance bar: both ramp from 0 to roughly
2.8k-3.6k RPM. The NPU row's Tctl is the host CPU, not the NPU: the iGPU stays
near idle while the CPU fallback and submission raise Tctl.

The remaining E2B heat is therefore host work. Device-side RMS norm, RoPE and
attention links are the next E2B lever; more slots or A4B would add work before
that is addressed.

#### Stage timings, one vs four rows

`llama-batched-bench -npp 16 -ntg 32` for 32 decode steps, with
`SPIRAL_PROFILE_GRAPH=1 SPIRAL_PROFILE_NODES=1 -lv 4` (info logs are hidden
below `-lv 4`). Node and chain lines were filtered to `rows=1` and `rows=4`.
Chain stage splits are derived from the run totals minus the node stages; a
step's `kernel` includes `start + wait + sync`.

| per decode step | x1 | x4 |
| :-- | --: | --: |
| wall (`T_TG / 32`) | 62.9 ms | 68.3 ms |
| **NPU total** | **43.1 ms** | **42.6 ms** |
| node matmuls | 23.2 ms | 23.1 ms |
| — kernel | 22.30 | 22.17 |
| — txn / input / unpack | 0.15 / 0.40 / 0.40 | 0.15 / 0.40 / 0.41 |
| FFN chains (35 per step) | 19.9 ms | 19.5 ms |
| — kernel (derived) | 19.70 | 19.33 |
| — txn / input / unpack | 0.04 / 0.11 / 0.03 | 0.04 / 0.10 / 0.03 |
| outside the NPU (wall − NPU) | 19.8 ms | 25.7 ms |
| per-token wall | 62.9 ms | 17.1 ms |

The NPU does the same work per step for one or four rows: ~42-43 ms, 97% of it
in the kernel stage, with input/txn/unpack together under 1.2 ms. Four rows
therefore divide per-token NPU time by four. The x4 step costs 5.4 ms more
wall, all of it outside the NPU: cumulative CPU fallback rises from 42 ms to
134 ms over the 32-step run, and host pack/gather from ~0.18 to ~0.69 ms per
step; the rest is graph scheduling and submission between backend views. The
NPU is not the thing that fails to scale with slots.

#### Remaining CPU fallback after the fusions

`SPIRAL_PROFILE_CPU=1`, x4, `npp 16 -ntg 16`, decode nodes with four rows.
Per-node logging inflates the totals slightly; the ranking is the useful part.

| op | nodes/step | ms/step |
| :-- | --: | --: |
| `RMS_NORM` | 242 | 1.24 |
| `MUL` | 298 | 0.43 |
| `SOFT_MAX` | 35 | 0.33 |
| `ROPE` | 50 | 0.24 |
| `UNARY` | 35 | 0.17 |
| `CONT` | 35 | 0.16 |
| `ADD` | 106 | 0.09 |
| `GET_ROWS` (PLE lookup) | 6 | 0.05 |

The measured CPU nodes are ~2.7 ms/step; the graph profile's cumulative CPU
fallback is ~4.2 ms/step over the same run. The rest of the ~25.7 ms/step
outside the NPU is graph scheduling and host submission between backend views.

### Backend fusions and chains

Gemma 4 E2B Q4_0, 64-token greedy, 3 alternating runs, `-n 64`, `temp 0`.
Output is byte-identical across all rows.

| mode | CPU fallback | eval (median) |
| :-- | --: | --: |
| CPU GLU (`SPIRAL_NO_GEGLU_FUSE=1`) | 93.7 ms / 32 graphs | 51.65 ms/token (19.36 t/s) |
| host-mediated GEGLU (`SPIRAL_NO_FFN_CHAIN=1`) | 63.8 ms / 32 graphs | 51.64 ms/token (19.37 t/s) |
| chained TXN (default) | 63.8 ms / 32 graphs | **47.81 ms/token (20.91 t/s)** |

The chained TXN is the current best decode number. It is a 64-token run, not a
soak; the soak table above predates it.

| experiment | before | after |
| :-- | --: | --: |
| gate+up combined (individual -> chained) | 106.1 pp8 / 16.2 tg1 / 17.3 tg16 | 122.2 / 17.5 / 18.4 |
| pipeline depth 1 -> 2 (`pp8` / `tg16`) | 95.3 / 18.0 | 112.5 / 18.3 |
| pipeline depth 4 (`pp8` / `tg16`) | - | 111.6 / 16.4 |
| softcap fused into the epilogue, 29 runs | ~53.7 ms/token | ~50.4 ms/token (6%) |

The gate+up and pipeline rows predate the pmode baseline; the relative wins are
the durable part. `pp8`/`tg16` carry roughly ±3-9 t/s and ±0.7 t/s run variance.

64-token decode profile, one CPU fallback thread:

| component | per token |
| :-- | --: |
| clean NPU calls | ~48 ms |
| CPU fallback | 9.4 ms |
| host/sampling | ~10 ms |
| **total** | **~56 ms (17.7 t/s)** |

Fallback breakdown before the softcap/GEGLU links: `UNARY` (tanh over 262144
values) 5.1 ms, `GLU` 1.2, `FLASH_ATTN_EXT` 1.2, `RMS_NORM` 0.8, `MUL` 0.6,
`ROPE` 0.2. More `SPIRAL_CPU_THREADS` speeds the tanh but slows the token.

PM QoS (`SPIRAL_DMA_LATENCY=1`), three alternating runs:

| metric | on | off |
| :-- | --: | --: |
| `pp8` | 118.1 t/s | 111.4 t/s |
| `tg16` | 16.84 t/s | 18.11 t/s |
| 300-token generation | 15.2-15.3 t/s | 16.8-17.3 t/s |

### NPU tuning

Same 3000-token NPU soak, normal boot state otherwise:

| configuration | t/s |
| :-- | --: |
| boot state (pmode `default`) | 14.70 |
| + fclk 7 (1960 MHz) | 14.64 |
| + fclk 7 + socclk 7 (1472 MHz) | 14.65 |
| pmode `turbo`, no clock pins | 16.38 |

pmode is the whole difference (~11% decode) and it does not survive a reboot;
a `spiral-npu-tune` systemd unit re-applies it. MP/H clocks are identical
(1408/2112 MHz) under both settings, so the gain is not a clock effect.

### Backend verification pass

Build: production xclbin (`src/program/build-production`, 2026-11-24) with
llama.cpp `build-spiral` and a server-only `build-server`, model
`gemma-4-E2B_q4_0-it.gguf`, prelude on unless noted. Short runs; no soak, so
no `Tctl`/fan context and none of these are thermal results. Measured
2026-11-25.

| check | result |
| :-- | :-- |
| shapes on the NPU | chained path exercises K = 256/1536/2048/4096; `SPIRAL_NO_FFN_CHAIN=1` adds K=6144 (45 calls) and K=12288 (60 calls), all as `spiral node` NPU calls |
| M=1 and M=17 | `-ub 17` on a 24-token prompt: prefill rows 17, 7 and 2, decode rows 1, no failures |
| batched decode, `-np 3` | 3/3 slots byte-equal to the `-np 1` greedy text, 48 tokens; 2.55-2.62 s/slot, 3.29 s wall |
| batched decode, `-np 5` | 5/5 slots byte-equal, 48 tokens; 4.37-4.52 s/slot, 5.57 s wall (≈55 t/s aggregate vs 20.4 t/s for one slot) |
| decode graph | 1524 nodes, **3** scheduler splits (`-lv 4`), `max_slots=2` |
| 16-token profile | 34 graphs, 4692 NPU calls, 595 FFN chains, 595 prenorm, 4962 row chunks, 3882 col chunks; kernel 801.8 ms, txn 4.6, input 12.6, unpack 11.4; pack 3.2, gather 2.3 |
| resident weights | 1326 MiB in the Spiral weight BOs (llama memory breakdown) |
| greedy vs CPU | agree to character 112 of the completion (≈25 of 64 tokens) then diverge (`inherent` vs `physical`); `SPIRAL_NO_PRENORM=1` diverges at the same character |
| CPU fallback with unsupported matmuls | **fails**: bf16 E2B with `--device Spiral0 -ngl 999`, and q4_0 with `-ngl 0`, decode garbage; `SPIRAL_MATMUL_ONLY=1` and `--device none` are correct |

The divergence is the documented bf16/quant rounding difference, not the
prelude; the fallback failure is a defect in the backend, not in the model or
the NPU path. See `integrations/llama.cpp/TODO.md`.

## 3. Fixed program (`src/program`)

Production build, device kernel time, twenty runs, measured after the bf16 tile
epilogue. Ranges, not medians: run-to-run spread is large enough that a single
number is misleading.

| case | kernel (us) |
| :-- | --: |
| Q4_0 1536x512 | 71-79 |
| Q4_0 2048x2048 | 103-113 |
| Q6_K 2048x2048 | 127-151 |
| Q4_0 1536x12288 | 267-280 |
| Q6_K 1536x6144 | 210-230 |

Q6_K head (262144 columns), prepared weights:

| path | wall | kernel sum |
| :-- | --: | --: |
| 11 chunks, one slot | 7.7-8.2 ms | 7.2-7.5 ms |
| 11 chunks, two pipeline slots | 6.5-6.9 ms | 5.9-6.2 ms |
| 4 slots | 6.5 ms | 5.9 ms |
| 8 slots | 6.6 ms | 6.0 ms |

Head chunk sweep (bands per chunk -> total): 25 -> 9462.6 us, 32 -> 8764.7,
40 -> 8461.2, 48 -> 7984.1. `MAX_BANDS_PER_CHUNK=48` is kept.

Cost model for a cold, streaming call: `kernel_us ~= 80 us + bytes / 48 GB/s`.
This is linear in weight bytes plus a fixed launch; naive `bytes / time`
understates the streaming rate. Cold-timing medians that fit the line:

| case | weight bytes | p50 kernel |
| :-- | --: | --: |
| 4096x1536 | 3.54 MB | 153.3 us |
| 6144x1536 | 5.31 MB | 194.5 us |
| 12288x1536 | 10.60 MB | 299.7 us |

Idle gaps are expensive: inserting a sleep between calls slows the next call,
and the penalty grows with the gap. The model's interleaving of NPU calls with
CPU fallback therefore costs more than a back-to-back loop would, and keeping
the NPU busy is worth more than raw bandwidth. `SPIRAL_COLD_GAP_US` reproduces
the measurement.

Combined column matmuls (prepared weights, production):

| combined call | columns | kernel us | total us | separate total |
| :-- | --: | --: | --: | --: |
| gate + up, 6144 | 12288 | 318.8 | 339.3 | 436.0 |
| gate + up, 12288 | 24576 | 548.9 | 595.1 | 684.4 |
| Q + K + V, sliding | 3072 | 139.8 | 148.0 | 290.0 |
| Q + K + V, full | 5120 | 178.8 | 189.6 | 337.8 |

FFN 6144 chain (`q4_0_1536x12288` gate+up -> `q4_0_6144x1536`), 20 iterations:

| path | total |
| :-- | --: |
| two launches with device GEGLU intermediate | 447-458 us |
| one chained TXN | 401-437 us |

Launch-limited E2B estimate before chaining: 286 launches, 51.97 ms kernel /
55.83 ms wall (19.2 / 17.9 tok/s); after combining gate+up and QKV, 221 launches
and ~19.8 tok/s. A 286-op token-shaped stream in one TXN simulates at 29.7 ms
against 55.8 ms per-call (memory bound 25.1 ms); an empty TXN is ~44 us.

Program memory: production core `.text` 11872 bytes before the GEGLU epilogue,
12544 bytes (77% of 16 KiB) after; diagnostic build 14880 bytes. Worker stack
8192 bytes. `program_output` 1360 B, no stack frame.

Golden tolerance in the shipped code: `max(1e-3, 6e-3 * column_scale)`; softcap
and packed16 use `5e-2`. Worst observed golden ratio ~0.65. Absolute error
grows ~`1.2e-7 * K`: max abs 0.0026 at K=2048, 0.0049 at 4096, 0.0071 at 6144,
0.0127 at 12288. That is an unbiased RTNE accumulation envelope, not a bug.

### RMS norm kernel (`src/funcs/rmsnorm`), standalone

Build: `python src/funcs/tests/rmsnorm.py` on the baseline machine state,
2026-11-24. One compute tile, `rows_steps = 6` (K=1536, two groups, the second
with two real row steps). Trace around each pass, seed 1:

| pass | cycles |
| :-- | --: |
| sum of squares | 765 |
| in-place scale | 3904 |
| **total** | **5315** |
| `.text` | 3600 B |
| max stack frame | 384 B |

Rows sweep, same build: 4 rows 3696 cycles, 6 rows 5315, 8 rows 6693. The
second group is partial at 6 rows, which is why the per-row figure is not flat.

Accuracy: `max abs error 0.0156`, worst tolerance ratio `0.464` across seeds
1-5 under `max(1e-3, 6e-3 * row_scale)` (seed 1 gives `0.450`). Reordering the
f32 sum of squares across four partial accumulators did not change it.

The same harness before this change measured 9197 total (sum 1059, scale 7486)
at `.text` 3408 B. The scale pass gained ~48% from two independent chunk
chains, a 512-bit two-chunk weight load, and one-instruction 128-bit lane
broadcasts; the sum gained ~28% from four f32 partial accumulators.

### Integrated prelude (`src/program`), production

Build: `python src/program/build.py --production`, 2026-11-24, one process,
`./test --prenorm-spike`. The normal body runs the matmul on the raw input; the
prenorm body sums the same input, scales each group by `r * weight`, then runs
the same matmul.

| shape | normal | prenorm | device cost |
| :-- | --: | --: | --: |
| `q4_0_1536x512` | 92.1 us | 94.9 us | 2.8 us |
| `q4_0_1536x2048` | 126.4 us | 129.9 us | 3.5 us |

Interleaved 20-run totals, normal call first each iteration. Both bodies are
bit-identical to a host-normalized bf16 reference through the same mmul (max
error 0) when the reference mirrors the device's `r*weight` hi/lo split and
single rounding of `x*hi + x*lo`.

| property | value |
| :-- | --: |
| `.text` | 16208 B of 16384 (98.9%) |
| stack | 8192 B |
| TXN stream, normal | 973 words / 3892 B |
| TXN stream, prenorm | 1121 words / 4484 B |

The weight is copied out of the config object into an 8 KiB L1 buffer, paid for
by dropping the input FIFO to depth 1. The config object caps the weight at
`K <= 2032`; the L1 buffer caps it at `K <= 2048`.

`--prenorm-spike` also covers the fused forms, measured 2026-11-25 on the same
build: a two-segment wide call (`prenorm_wide_1536x1024`, 0/4096 values differ)
and a two-operation chain with the prelude on op0
(`prenorm_chain_1536x512x512`, K=1536 then 512, 0/2048 values differ). Both are
bit-exact against the same host-normalized reference, so the prelude survives
concatenation and chaining without an extra rounding.

### E2B soak, FFN prelude on vs off

Build: production, 2026-11-24, `tests/bench/soak/soak.py npu 3000` with the
host configuration verified (turbo, balanced, powersave, 600/2400 MHz iGPU,
PM QoS off). The prelude covers the FFN gate+up input only; the attention norm
stays on the host.

| build | wall | tokens/s | Tctl max | iGPU max | fans | plateau |
| :-- | --: | --: | --: | --: | --: | --: |
| prelude off (`SPIRAL_NO_PRENORM=1`) | 179.5 s | 17.29 | 74.9 C | 56.0 C | 4834/4667 | 74.4 s |
| prelude on | 181.5 s | 17.09 | 74.0 C | 56.0 C | 4845/4667 | 108.6 s |

The prelude removes ~1190 host norm/gain nodes per run but is 1.2% slower and
does not move the fans; the Tctl difference is inside run-to-run noise. It is
not a thermal win at this coverage.

## 4. Quant kernels (`src/quants`)

Fused single-core `mmul.py`, 256x64 shape, measured 2026-11-18 on Linux 7.3.5
at the measured 1.8 GHz H clock; the `q6_k` and `q6_k_int8` rows were
re-measured 2026-11-24 after the immediate-lane scale broadcast. `% PM` is
kernel `.text` as a share of the 16 KiB compute-tile program memory.
`est w/s` is the full-NPU estimate:
`min(weights/cycle * 1.5 GHz * 32, 55 GB/s / bytes-per-weight)`.

| format | bits/w | cycles | cycles/tile | weights/cycle | % PM | est w/s | limit |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| `q4_0` | 4.5 | 7883 | 246.34 | 2.078 | 24.7% | 94B | compute at 1.5, memory 98B at 1.8 GHz |
| `q4_0_packed16` | 4.5 | 7397 | 231.16 | 2.215 | 23.2% | 98B | near-exact, memory-bound at 1.8 GHz |
| `q4_0_hilo` | 5.0 | 7268 | 227.13 | 2.254 | 22.2% | 88B | memory, +11% size |
| `q6_k` | 6.625 | 10115 | 316.09 | 1.620 | 29.8% | 66B | memory-bound, 66B |
| `q6_k_int8` | 8.625 | 8440 | 263.75 | 1.941 | 25.6% | 51B | memory, +31% size |
| `q8_0` | 8.5 | 7707 | 240.84 | 2.126 | 43.8% | 52B | memory |

The q6 rows gained from `repeat_lane64` (register-lane `vextbcst.128`) instead
of `broadcast_vector` for the subgroup scale expansion: `q6_k_int8` 9101 ->
8440 cycles (−7.3%) and stack 2240 -> 1728 B, `q6_k` 10180 -> 10115 (−0.6%) and
stack 3200 -> 2432 B. At 256x128 the same change is 20329 -> 20200 (`q6_k`) and
18178 -> 16841 (`q6_k_int8`). The same edit regressed `q4_0` 7883 -> 7929 and
`q4_0_packed16` 7397 -> 7573, so those keep `broadcast_vector`.

Standalone unpack (diagnostic only, never a selection criterion): `q4_0` 6579,
`q4_0_packed16` 5459, `q4_0_hilo` 2916, `q6_k` 8498, `q6_k_int8` 4210, `q8_0`
2920 cycles.

Exact and packed16 use the same 9216 packed bytes at this shape, so only cycles
differ: 6.2% (7397 vs 7883). The production path is the exact `q4_0`.

Program-memory bytes per kernel: `q6_k` 4880, `q8_0` 7168, `q4_0` 4048,
`q4_0_packed16` 3808, `q4_0_hilo` 3632, `q6_k_int8` 4192.

Target-model traffic, derived from the allocations in
[`targets.md`](targets.md): the formats each artifact actually uses, one row per
token for an embedding lookup, the tied head counted as a full matmul, and the
routed-expert fraction of each artifact's experts. These are not device
measurements; the gate is bandwidth, so they are the ceiling before dispatch and
host cost. The E2B row is the earlier figure, kept unchanged.

| target | GB/token | ms/token at 55 GB/s | ceiling |
| :-- | --: | --: | --: |
| Gemma 4 E2B q4_0 | 1.381 | 25.1 | 39.8 tok/s |
| Gemma 4 26B-A4B qat q4_0 | 2.377 | 43.2 | 23.1 tok/s |
| Qwen3.6 35B-A3B UD-Q4_K_XL | 2.789 | 50.7 | 19.7 tok/s |
| ISTA Qwen3.8-Flash-Next Q2_0, `Q2_0` share only | 0.664 | 12.1 | — |
| Gemma 4 31B qat q4_0 | 17.27 | 314.0 | 3.2 tok/s |
| Qwen3.8 27B UD-Q4_K_XL | 16.83 | 306.0 | 3.3 tok/s |

E2B carries canonical `q4_0` plus the native `q6_k` head (1.497 GB/token with
`q4_0_hilo`, 1.482 GB with `q6_k_int8`). The two dense models sit an order of
magnitude above the rest: every matmul weight is read per token, so ~3 tok/s is
the bandwidth wall and no kernel work moves it. The 26B reads 22.8B expert
parameters at 8-of-128; the Qwen 3.6 reads 33.0B at 8-of-256, but its dense path
is almost entirely `Q8_0`, which costs it more than the sparser expert selection
saves. The `Q2_0` row is the routed-expert share of that artifact alone, at
2.36B active parameters; the rest of its path is at formats this repository does
not implement.

## 5. Loop and compiler experiments

Standalone 64-byte copy, 16 KiB total (256 x 64 B), measured on the same part:

| C++ form / hand asm | cycles | bytes/cycle | kernel `.text` | program `.text` |
| :-- | --: | --: | --: | --: |
| indexed, rolled | 2313 | 7.1 | 176 B | 512 B |
| indexed, `unroll(enable)` | 516 | 31.8 | 272 B | 608 B |
| indexed, `unroll_count(16)` | 441 | 37.2 | 320 B | 656 B |
| indexed, `unroll(full)` | 263 | 62.3 | 1568 B | 1904 B |
| **pointer, no pragma** | **266** | **61.6** | **208 B** | **544 B** |
| pointer, `unroll_count(8)` | 263 | 62.3 | 272 B | 560 B |
| indexed, manual 8x | 670 | 24.5 | 576 B | 912 B |
| hand asm, 8-deep | 271 | 60.5 | 436 B | 800 B |

The q4_0 counterexample: pointer increments on a recurrence-bound loop regress
2-4%. See `rejected.md`.

`-O3` produced byte-identical objects and identical cycles; `-Os` grew `.text`
and was slower on large shapes. A fixed-shape 256x16 Q4_0 experiment reached
2316 cycles against 2204 for the existing schedule. The full list of neutral
`aie-*` flags is in `rejected.md`.

## 6. float16 scale probes (`tests/bench`)

One core, 32 iterations. `vals/iter` differs between probes; compare per-value
ratios, not raw cycles. Re-measured 2026-11-18.

| probe | cycles | vals/iter | cyc/iter | error |
| :-- | --: | --: | --: | :-- |
| `float16_packed16_split_bf16` | 904 | 32 | 28.2 | 9.8e-4 relative |
| `float16_expadd_split_bf16_flush` | 1387 | 32 | 43.3 | 6.0e-5 abs |
| `float16_expadd_split_bf16_denorm` | 2089 | 32 | 65.3 | 0 |
| `float16_shift_split_bf16` | 1450 | 32 | 45.3 | 0 |
| `float16_bits_split_bf16` | 1740 | 32 | 54.4 | 0 |
| `float16_direct_split_bf16` | 2252 | 32 | 70.4 | 0 |
| `int8_scale_bf16` | 866 | 64 | 27.1 | 0 |
| `int8_scale_bf16_2term` | 964 | 64 | 30.1 | 0 |
| `int8_scale_int_mul` | 642 | 64 | 20.1 | 0 |
| `q6k_3term_prep` | 3434 | 32 | 107.3 | 0 |
| `q6k_3term_apply_x8` | 1063 | 64 | 33.2 | 0 |
| `q6k_int_apply` | 2054 | 64 | 64.2 | 0 |

XDNA2 native vector width is 512 bits: 64 int8, 32 bf16/int16, 16 f32, 128
int4 lanes per register. The `_x8`/`_x16` probes exercise a quarter/half of a
native bf16 vector and are kept only for format-shaped cases.

## 7. PoC records (`tests/poc`)

These belong to completed experiments and are frozen; see the linked records
for conditions and caveats.

- Bulk host-to-NPU DMA: 1 channel 14.2, 4 channels 41.0, 8 channels 55.5,
  16 channels 57.4 GB/s. Matrix-shaped transfer 56.5 GB/s.
- 2048x2048 bf16 mmul with f32 I/O: 10022.6 us, 53.98 GB/s, 214.26 GFLOP/s.
- Quantized weight stream: q4_0 52.8, q6_k 54.6 (96% of the bulk ceiling),
  iq1_s 33.5 GB/s.
- Call overhead, C++: ~80-92 us per repeat, ~31.8 us per call pipelined;
  reconfiguration per switch ~2.6 ms; Python ~105 us per call.
- Program-memory overlay: full PDI reload ~2.6 ms; blockwrite 1.3-1.5 us/KiB,
  ~0.45-0.5 ms for 32 tiles; control packets 68.8/138.6 us per tile against
  50.7/96.3 us for blockwrite.
- TXN concatenation: 2-op chains 1.18-1.39x, 8-op chains 1.57-1.94x over
  independent launches; every identity case bit-identical.

Details: [tests/poc/README.md](../tests/poc/README.md),
[overlay.md](../tests/poc/overlay.md),
[stream/README.md](../tests/poc/stream/README.md),
[txn_concat/README.md](../tests/poc/txn_concat/README.md).

## 8. Recording rules

- Name the build (production/diagnostic), the shape, and the date.
- Report cycles, `weights/cycle`, bytes/weight, `.text`, and stack frame for a
  kernel change; report `Tctl` and both fans for anything that runs long.
- Prefer medians/p95 from repeated interleaved runs over means or best-of-N.
- Cycles and `weights/cycle` are clock-independent; weights/s and GB/s scale
  with the clock. Say which clock a rate assumes.
- Compare within one boot where possible. Cross-boot absolute cycles have
  matched closely, but per-value ratios from one session are safer.
- A number with no configuration and no date is not evidence.
