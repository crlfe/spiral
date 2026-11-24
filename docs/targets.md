> Role: reference. Current as of 2026-11-24. History: `git log -- docs/targets.md`.

# Target artifacts

The GGUF builds this repository is developed against. Read this before proposing
a quantization format or a capability: an artifact's allocation says which
format buys most of the model, and its tensor shapes say what the transport
contract has to accept.

Per-token traffic for every entry is in [`measurements.md`](measurements.md) § 4.
Plans and status are in `src/quants/TODO.md` and `src/program/TODO.md`. The
numbers below are properties of the artifacts, not measurements of this
project.

## Priority

| artifact | state | what it needs |
| :-- | :-- | :-- |
| Gemma 4 E2B q4_0 | current target | nothing new: `q4_0` plus a native `q6_k` head are implemented, and every matmul `ne0` is a multiple of 256 |
| Gemma 4 26B-A4B qat q4_0 | next | no new format; `ne0` 2112 and 704 are not 256-multiples, so it needs 64-row tiles, where a `q4_0` tile is 576 bytes and packs cleanly; exercises 8-of-128 sparse expert dispatch |
| Qwen3.6 35B-A3B UD-Q4_K_XL | next | no geometry work — every matmul `ne0` is a multiple of 256; needs `Q4_K` and `Q5_K`, which are already the first formats on the plan, plus `Q8_0` and `Q6_K`, which are done; 8-of-256 experts |
| ISTA Qwen3.8-Flash-Next Q2_0 | proposed | one new format (`Q2_0`) covering 96.3% of the parameters; `ne0` 640 and 320 also need 64-row tiles, where a `Q2_0` tile is 288 bytes and takes padding or a wider column tile |
| Gemma 4 31B qat q4_0 | parked | nothing new — all `q4_0`, every matmul `ne0` a multiple of 256 — but dense, so every weight is on the token path |
| Qwen3.8 27B UD-Q4_K_XL | parked | eight quantization types including `IQ4_XS`, `IQ4_NL`, `Q3_K` and `IQ3_S`, and dense |

64 is the largest row granularity that divides every offending `K` across both
targets that need it. It is free for `q4_0`: 64 rows x 16 columns is 1024
weights at 4.5 bits, 576 bytes, already a 64-byte multiple. It is not free for
`Q2_0`, which is 288 bytes at the same tile, 32 short, so that format either
pads to 320 (+11%) or moves to two column pairs per tile.

The next two entries sit on different axes. The Gemma 26B needs no format at
all and exercises the tile change; the Qwen 3.6 needs no tile change and
exercises the K-quant core. Doing the Gemma first means the `Q2_0` bring-up
inherits a proven 64-row tile; doing the Qwen first means the K-quant work,
which every artifact in this list wants, lands on a target that also runs
sparse experts. Either order leaves the other one cheap.

## ISTA Qwen3.8-Flash-Next GSQ-RCO `Q2_0`

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, file
`Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`, revision
`ed59f92`. The repository also publishes
`tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt`,
so the per-tensor allocation can be audited without the weights.

Shard 1 of 2: 1223 tensors, 125.74B parameters, 37.61 GB. 48 blocks, hidden
2560, context 262144, 24 heads with 2 KV heads. 512 experts per block with 10
active per token, a separate shared expert in the `*_shexp` tensors, and expert
intermediate 640.

| ggml type | tensors | GB | parameters | bpw |
| :-- | --: | --: | --: | --: |
| `Q2_0` | 202 | 34.05 | 121.06B | 2.250 |
| BF16 | 483 | 1.47 | 738M | 15.97 |
| `Q3_K` | 91 | 0.77 | 1.80B | 3.438 |
| `Q5_K` | 16 | 0.52 | 757M | 5.500 |
| `IQ4_XS` | 56 | 0.44 | 824M | 4.250 |
| `Q4_K` | 38 | 0.23 | 413M | 4.500 |
| `Q6_K` | 12 | 0.08 | 103M | 6.562 |
| `Q4_0` | 16 | 0.02 | 26M | 4.500 |
| `Q5_0` | 7 | 0.01 | 11M | 5.500 |
| `IQ4_NL` | 7 | 0.01 | 11M | 4.500 |
| `Q8_0` | 2 | 0.00 | 3M | 8.500 |
| F32 | 292 | 0.01 | 3M | 32.0 |
| F16 | 1 | 0.00 | 41K | 16.0 |

`Q2_0` is the whole routed-expert plane: all 144 `ffn_{gate,up,down}_exps`
tensors in all 48 blocks, 121.06B parameters, 96.3% of the artifact and 90.5% of
its bytes. The remaining 58 `Q2_0` tensors are dense, and together hold 0.26B
parameters. This is the argument for the build: one scalar format, 34 GB.

Shapes, per block:

| tensor | shape | K | N |
| :-- | :-- | --: | --: |
| `ffn_gate_exps`, `ffn_up_exps` | [2560, 640, 512] | 2560 | 640 |
| `ffn_down_exps` | [640, 2560, 512] | 640 | 2560 |

`ffn_down_exps` and `ffn_down_shexp` are `K` 640 (96 tensors), and the
gated-residual up-projections `hc_*_up` are `K` 320 (97 tensors). Both are
multiples of 64 and neither is a multiple of 256, so the current repack contract
(`weights_rows % 256 == 0`) rejects them. The 48 `ffn_down_exps` alone are
11.3 GB, a third of the `Q2_0` bytes, so the tile question gates the format, and
it gates the gated-residual path with it. 640 is 10 blocks of 64 and 320 is 5,
so the block layout fits 64 rows exactly; whether the tile geometry can (see
the packing note under Priority) is the open question.

Shard 2 is a single tensor, `per_layer_token_embd.weight` [160, 320001536],
51.20B parameters at `IQ4_NL`, 28.8 GB. It is an n-gram lookup table: one row
per token, never a matmul. It is out of scope for the kernels and only matters
for resident bytes.

`Q2_0` is ggml type 42. Treat the block layout as unconfirmed until it is read
from the ggml source that defines it; the 2.250 bits/weight here is exact
(18 bytes per 64 weights), but that implies rather than proves one fp16 scale
per block.

## Gemma 4 E2B q4_0

`gemma-4-E2B_q4_0-it.gguf`, architecture `gemma4`: 4.63B parameters, 3.33 GB,
541 tensors, 35 blocks, hidden 1536, context 131072, 8 heads, 1 KV head.

| ggml type | tensors | GB | parameters | bpw |
| :-- | --: | --: | --: | --: |
| `Q6_K` | 2 | 2.26 | 2.75B | 6.562 |
| `Q4_0` | 275 | 1.05 | 1.86B | 4.500 |
| F16 | 1 | 0.03 | 14M | 16.0 |
| F32 | 263 | 0.00 | 0.3M | 32.0 |

The two embedding tables hold 59% of the parameters, and only one of them is on
the token path:

| tensor | shape | role |
| :-- | :-- | :-- |
| `per_layer_token_embd` | [8960, 262144] `Q6_K` | PLE lookup, 2.35B parameters |
| `token_embd` | [1536, 262144] `Q6_K` | lookup and tied output head, 403M |

Matmul weights are therefore 1.86B `Q4_0` parameters in 1.05 GB. FFN is 6144 in
blocks 0-14 and 12288 in blocks 15-34; 15 blocks carry `attn_k`/`attn_v` at 512
wide and all 35 carry a 256-wide `inp_gate`/`proj` pair. `per_layer_model_proj`
[1536, 8960] is the one F16 tensor.

Every matmul `ne0` is a multiple of 256, which is why this is the first target
and why no contract work was needed for it; the only other `ne0` is 1, on
`layer_output_scale`.

## Gemma 4 26B-A4B qat q4_0

`gemma-4-26B-A4B-it-qat-q4_0.gguf`, architecture `gemma4`: 25.23B parameters,
14.42 GB, 658 tensors, 30 blocks, hidden 2816, FFN 2112, context 262144, 16
heads, 128 experts with 8 active.

| ggml type | tensors | GB | parameters | bpw |
| :-- | --: | --: | --: | --: |
| `Q4_0` | 265 | 13.77 | 24.48B | 4.500 |
| `Q6_K` | 1 | 0.61 | 738M | 6.562 |
| F32 | 392 | 0.05 | 12M | 32.0 |

No new format: it is `Q4_0` with a `Q6_K` `token_embd` head, the same pair the
current program runs. The experts are fused and sparse:

| tensor | shape | K | per block |
| :-- | :-- | --: | --: |
| `ffn_gate_up_exps` | [2816, 1408, 128] | 2816 | 507M parameters |
| `ffn_down_exps` | [704, 2816, 128] | 704 | 254M |

The expert matrices are 22.8B of the 24.5B `Q4_0` parameters. `ffn_down_exps`
carries a per-expert F32 `scale` [128] alongside the weight, which the other
targets do not have. The dense `ffn_down` is `ne0` 2112 and the expert
`ffn_down_exps` is 704; those are the only matmul `K` values in the file that are
not multiples of 256. Both are multiples of 64, and 704 is 11 blocks of 64, so
no padding is involved: 64 rows x 16 columns is 576 bytes at `q4_0`, a clean
64-byte multiple.

## Gemma 4 31B qat q4_0

`gemma-4-31B-it-qat-q4_0.gguf`, architecture `gemma4`: 30.70B parameters,
17.27 GB, 833 tensors, 60 blocks, hidden 5376, FFN 21504, context 262144, 32
heads.

| ggml type | tensors | GB | parameters | bpw |
| :-- | --: | --: | --: | --: |
| `Q4_0` | 411 | 17.27 | 30.70B | 4.500 |
| F32 | 422 | 0.005 | 1.3M | 32.0 |

The file is `Q4_0` throughout and `token_embd` [5376, 262144] is `Q4_0` as
well, doubling as the output head. Like E2B, every matmul `ne0` is a multiple of
256, so it needs no contract work. It was renamed locally from the upstream
`UD-Q4_K_XL` label, which does not describe its contents; a fresh download of
the original will land under that name again.

It is parked because it is dense: all 30.7B parameters are read per token. See
[`measurements.md`](measurements.md) § 4 for the traffic and the resulting
ceiling.

## Qwen3.6 35B-A3B UD-Q4_K_XL

`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, architecture `qwen35moe`: 35.51B parameters,
22.84 GB, 753 tensors, 41 blocks, hidden 2048, context 262144, 16 heads with 2
KV heads, 256 experts with 8 active.

| ggml type | tensors | GB | parameters | bpw |
| :-- | --: | --: | --: | --: |
| `Q4_K` | 80 | 12.08 | 21.47B | 4.500 |
| `Q5_K` | 40 | 7.38 | 10.74B | 5.500 |
| `Q8_0` | 260 | 2.62 | 2.46B | 8.500 |
| `Q6_K` | 3 | 0.66 | 805M | 6.562 |
| F32 | 368 | 0.11 | 26M | 32.0 |
| BF16 | 2 | 0.001 | 0.5M | 16.0 |

Every format in it is either implemented or already first on the plan: `Q8_0`
and `Q6_K` are done, and `Q4_K` plus `Q5_K` are the first two entries under
`src/quants/TODO.md` § Format bring-up plans. Nothing here needs a deferred
format.

| tensor | shape | K | N |
| :-- | :-- | --: | --: |
| `ffn_gate_exps`, `ffn_up_exps` | [2048, 512, 256] `Q4_K` | 2048 | 512 |
| `ffn_down_exps` | [512, 2048, 256] `Q5_K` | 512 | 2048 |

The experts are 33.0B of the 35.5B parameters, and every matmul `ne0` in the
file is a multiple of 256, so it needs no tile work. The allocation is nearly
but not exactly uniform per block: 40 of the 41 `ffn_down_exps` are `Q5_K` and
80 of the 82 gate/up matrices are `Q4_K`, the remainder being `Q6_K`/`Q5_K`. It
is a hybrid stack like E2B: 30 of the 41 blocks carry
`attn_qkv`/`attn_gate`/`ssm_out`, and 11 carry
`attn_q`/`attn_k`/`attn_v`/`attn_output`.

The dense path is almost entirely `Q8_0`, which is why its per-token traffic
exceeds the Gemma 26B's despite 8 active experts of 256 rather than 128.

## Qwen3.8 27B UD-Q4_K_XL

`Qwen3.8-27B-UD-Q4_K_XL.gguf`, architecture `qwen35`: 27.32B parameters, 17.55
GB, 866 tensors, 65 blocks, hidden 5120, FFN 17408, context 262144, 24 heads
with 4 KV heads.

| ggml type | tensors | GB | parameters | bpw |
| :-- | --: | --: | --: | --: |
| `Q5_K` | 191 | 7.93 | 11.53B | 5.500 |
| `IQ4_XS` | 70 | 3.13 | 5.88B | 4.250 |
| `Q4_K` | 69 | 3.06 | 5.44B | 4.500 |
| `Q6_K` | 56 | 2.86 | 3.49B | 6.562 |
| `IQ4_NL` | 6 | 0.28 | 498M | 4.500 |
| `Q8_0` | 110 | 0.13 | 123M | 8.500 |
| `Q3_K` | 3 | 0.12 | 267M | 3.438 |
| `IQ3_S` | 1 | 0.04 | 89M | 3.438 |
| F32 | 360 | 0.01 | 3M | 32.0 |

Eight quantization types, mixed per tensor and per layer: `blk.0.ffn_down` is
`IQ4_XS`, `blk.1.ffn_down` is `IQ4_NL`, and `blk.0.ffn_up` is `Q3_K`, so no
single format covers one projection across the stack. It therefore needs the
K-quant core, both IQ4 formats, `Q3_K`, and `IQ3_S`, which is currently
deferred.

It is dense, so every matmul weight is on the token path, and it is parked for
the same reason as Gemma 4 31B. Every matmul `ne0` is a multiple of 256, so it
needs no tile work — the cost is formats and bandwidth, not geometry.

## Not targets

- `gemma-4-E2B-it-bf16.gguf` (4.65B parameters, BF16, 9.29 GB) is an exactness
  oracle for the E2B kernels, not something to offload. It is also the only
  copy of `per_layer_model_proj` at BF16.
- `gemma-4-26B-A4B-it-qat-q4_0-mtp.gguf` and
  `gemma-4-26B-A4B-it-DFlash-Q8_0.gguf` are speculative-decoding draft heads,
  not full models.
- `Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf`, `Qwen3.6-35B-A3B-DFlash-Q8_0.gguf` and
  `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` are the same thing under the `dflash`
  architecture: 6 or 5 blocks and 0.39-1.92B parameters.
- `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (35.51B parameters, 39.10 GB) is the
  quality reference for the 3.6: `Q8_0` except for 14 expert matrices in blocks
  34-41 held at BF16. It needs `BF16` on the expert path, which is not started,
  and it carries more per-token traffic than the `Q4_K_XL` build.
- `gemma-4-E2B-it-mmproj.gguf` is the vision projector. It is not on the text
  path.

## Reading an artifact

Everything above comes from the GGUF tensor table: type ids, dimensions and
offsets, which give exact bits per weight and exactly which tensor carries
which format. `gguf-dump` or a short header parse reproduces all of it without
downloading the weights.
