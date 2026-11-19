# Spiral

The AMD XDNA2 NPU is an intriguing addition to some recent AMD processors:
it promises a lot of compute for local inference with much lower power
consumption than a GPU. Unfortunately, software support for the AMD NPUs
remains poor. What is available has limited model support and seems slower
than it should be. So we're going to fix it.

## Roadmap

I have hacked together enough of a llama.cpp GGML backend that I believe the
NPU can be used as an accelerator, starting with offloading large matrix ops.
The backend lives in [`integrations/llama.cpp`](integrations/llama.cpp); it is
attached to a local llama.cpp checkout with a symlink and a small hook patch,
so this repository stays the place for proving capabilities and benchmarking.
The eventual code may build as a proper backend in the llama.cpp tree.

My current focus is bringing up a basic set of dequantization routines, which
take the highly-compressed weights typical of consumer models and expand them
on the NPU to the bf16 blocks that its hardware can efficiently multiply.
Optimizing those well enough to run at full bandwidth will be a challenge,
so I have added a framework for fast unit testing and cycle counting. The first
real model test will be Gemma 4 26B-A4B IT QAT Q4_0. The objective is
productive, quiet inference: the NPU is used for efficiency, and an NPU+iGPU
split is parked until the NPU path is closer to competitive, ideally on a much
larger model.

## Development Hardware

Thanks to a well-timed clearance sale, I managed to get my hands on a laptop
with an AMD Ryzen AI 9 HX 475 and 64 GB of RAM for something approaching a
reasonable price. A major drawback to this particular device is that the BIOS
is crippled to prevent booting Linux (hacked that) or adjusting clock speeds.

The [FastFlowLM project](https://github.com/ROCm/FastFlowLM) is currently the
best choice for running inference on the NPU. Testing on my laptop it is
substantially slower than the Radeon 890M iGPU, but runs much cooler:

| Engine             | Model                          | Size       | Decode |
| :---               | :---                           | :---:      | :---:  |
| FastFlowLM-1.0.7   | Qwen 3.6 35B-A3B Q4+MTP Aug26  | 20.6 GB    | 17 tps |
| FastFlowLM-1.0.7   | Qwen 3.8 27B Q4+MTP Sep29      | 16.8 GB    |  2 tps |
| llama-b10839 ROCm  | Qwen 3.6 35B-A3B UD-Q4_K_XL    | 21.3 GB    | 20 tps |
| llama-b10839 ROCm  | Qwen 3.6 35B-A3B UD-Q8_K_XL    | 36.4 GB    | 18 tps |
| llama-b10839 ROCm  | Qwen 3.8 27B UD-Q4_K_XL        | 16.4 GB    |  4 tps |
| llama-b10839 ROCm  | Gemma 4 26B-A4B IT QAT Q4_0    | 13.4 GB    | 25 tps |
|

Interestingly, on the iGPU Qwen 3.6 35B-A3B runs as fast in Q8 as the smaller
Q4 versions, suggesting that the bottleneck is not the system memory bandwidth.
The dense Qwen 3.8 27B model used roughly `16.4 GB/tok * 4 tok/s = 65.6 GB/s`.
If the same memory bandwidth were saturated by Qwen 35B-A3B Q4, we would double
the basic decode rate to 36 tps. Adding speculative decoding like MTP or DFlash
could then easily get us over 50 tps when writing code.

## Gemma 4 E2B: engine comparison

3000-token greedy generations on this laptop, same model class.  The iGPU SCLK
is capped at 2400 MHz and the NPU runs pmode `turbo`.  The three llama.cpp rows
share the same Q4_0 GGUF with its native Q6_K head; FastFlowLM uses its own
quantisation and packing, so that row is a comparison of engines rather than of
identical weights.

| engine | model | decode |
| :-- | :-- | --: |
| llama.cpp ROCm (gfx1150 iGPU) | Gemma 4 E2B Q4_0, Q6_K head | 52.4 t/s |
| FastFlowLM | `gemma4-it:e2b` | 25.8 t/s |
| llama.cpp CPU (12 threads) | Gemma 4 E2B Q4_0, Q6_K head | 22.2 t/s |
| Spiral NPU backend | Gemma 4 E2B Q4_0, Q6_K head | 16.2 t/s |

Temperature, fan and clock behaviour for these runs, including the NPU's
per-fan comparison with FastFlowLM, are in
[`src/program/NOTES.md`](src/program/NOTES.md) under "Efficiency acceptance".

# License and Warranty Disclaimer

```
MIT License

Copyright (c) 2023-2026 Chris Wolfe (https://crlfe.ca)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
