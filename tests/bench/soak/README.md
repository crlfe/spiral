# Soak and device harnesses

Temperature, fan and clock measurements are only comparable when the machine
is in the same state each run.  These harnesses verify that state
first and refuse to measure when it drifts.

| file | purpose |
| :-- | :-- |
| `hostconfig.py` | read and verify the measurement configuration; prints restore commands and exits 2 on drift (`--allow-drift` to override) |
| `soak.py` | long generation per device (`cpu`, `npu`, `igpu`) until temperatures and fan RPM plateau, then cool back to idle |
| `headsoak.py` | NPU-only busy loop (`src/program ./test --head-soak`) with no CPU fallback |
| `device_ab.py` | one short generation per device with cooldown and temperature/fan sampling |
| `endpoint.py` | soak any local OpenAI-compatible endpoint, so different servers/frontends can be compared on the same machine state |
| `xdna.py` | read NPU clocks and power mode through `DRM_IOCTL_AMDXDNA_GET_INFO` |

## Check the configuration

```bash
python3 tests/bench/soak/hostconfig.py
```

Expected baseline: the machine's normal boot state with two deliberate
changes, a 2400 MHz iGPU SCLK cap and NPU pmode `turbo`.

| setting | value | source |
| :-- | :-- | :-- |
| NPU power mode | `turbo` | `DRM_AMDXDNA_GET_POWER_MODE` |
| iGPU DPM force level | `manual` | `power_dpm_force_performance_level` |
| iGPU SCLK OD bounds | 600 / 2400 MHz | `pp_od_clk_voltage` |
| platform profile | `balanced` | `/sys/firmware/acpi/platform_profile` |
| CPU governor | `powersave` on every policy | `cpufreq/policy*/scaling_governor` |
| CPU EPP | `balance_performance` (warning only) | `cpufreq/policy*/energy_performance_preference` |
| PM QoS | off (`SPIRAL_DMA_LATENCY` unset) | environment |

The NPU power mode is set to `turbo` by the `spiral-npu-tune` systemd unit and
is asserted here, because it is worth ~11% on NPU decode and does not survive a
reboot.  fclk and socclk are deliberately **not** controlled: pinning them was
measured to change nothing, so they stay on DVFS and are printed as
observations, never asserted.

A mismatch exits `2` and prints the commands that restore the baseline; the
harness never runs `sudo` itself.  `--allow-drift` reports the drift and keeps
going for exploratory runs, but those numbers must not go into the notes.

The one setting that can silently drift is the iGPU SCLK cap: rebooting or a
driver reload restores the hardware range (to 3100 MHz), which raises energy
per token and does not fail loudly.  The DPM force level and `SPIRAL_DMA_LATENCY`
are checked for the same reason.

## Run

```bash
# long soaks (tokens per device, plateau detection, cool-back)
python3 tests/bench/soak/soak.py igpu 3000
python3 tests/bench/soak/soak.py npu 3000

# NPU-only heat source, default 20000 head passes
python3 tests/bench/soak/headsoak.py

# quick device comparison
python3 tests/bench/soak/device_ab.py cpu 300

# any OpenAI-compatible server, with the same sampling and cool-back
python3 tests/bench/soak/endpoint.py http://127.0.0.1:52625 300 \
    --model gemma4-it:e2b --label fastflowlm
```

`endpoint.py` streams `/v1/completions` (or `--api chat` for
`/v1/chat/completions`), counts the streamed tokens, and measures decode from
the first token to the last so the number is comparable to `soak.py`'s
`eval time`.  Prefill is reported separately.  The model defaults to the first
entry of `/v1/models`; `--ignore-eos` adds llama.cpp's `ignore_eos` field for
servers that stop early.  Cool-back runs until Tctl is within 2 C of idle or
420 s pass; `--max-cool-back` shortens that for a quick look (numbers taken
with a shortened cool-back are not comparable with a full soak).  The only requirement is that the server streams SSE
(non-streaming responses fall back to `usage.completion_tokens` but report a
single sample).

Paths are resolved from the repo and from hwmon names, not fixed indices:

| variable | default |
| :-- | :-- |
| `LLAMA_CPP_DIR` | `~/Projects/llama.cpp` |
| `SPIRAL_MODEL` | `~/Projects/models/gemma-4-E2B_q4_0-it.gguf` |
| `SPIRAL_PROGRAM_BUILD_DIR` | `src/program/build-production` |

## Signals

- NPU clocks: no sysfs entry exists.  `xdna.py` reads
  `DRM_IOCTL_AMDXDNA_GET_INFO` on `/dev/accel/accel0` for the MP-NPU and H
  clocks (it can also read power and the power mode, which the harnesses leave
  unused).
- `amd-smi metric --clock` reports `APU_AVERAGE_IPUCLK_FREQUENCY` (H clock) and
  `APU_AVERAGE_MPIPU_FREQUENCY` (MP-NPU clock) as a root-free cross-check.
- iGPU temperature/clock, CPU `Tctl` and the fans are resolved by hwmon name
  (`amdgpu`, `k10temp`, `hp`), so index changes do not silently move a reading.
  The laptop has two `hp` fans and both are tracked: peak output is
  `fan1/fan2`, and they always move together.  `acpi_fan` is a zero
  placeholder, used only when `hp` is absent.

Power is deliberately not measured: RAPL `energy_uj` is root-only in this
container and `SOCKET_POWER` is too coarse to be worth integrating.  Fan RPM
and Tctl are the user-visible proxy, with far fewer approximations.

See `src/program/NOTES.md` ("Efficiency acceptance") and
`integrations/llama.cpp/TODO.md` ("Efficiency baseline") for the recorded
results and the conditions they were taken in.
