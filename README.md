# Spiral

The AMD XDNA2 NPU is an intriguing addition to some recent AMD processors:
it promises a lot of compute for local inference with much lower power
consumption than a GPU. Unfortunately software support for the AMD NPUs
remains poor, and what is available has limited model support and seems slow.
So we're going to fix it.

## Roadmap

I have hacked together enough of a llama.cpp GGML backend that I believe the
NPU can be used as an accelerator, starting with offloading large matrix ops.
The eventual code will build in a fork of the llama.cpp tree, but for now this
repository will be for proving capabilities and benchmarking.

### Next Steps

* tests/poc/memory.py: Check the available host->NPU memory bandwidth.
* tests/poc/invoke.py: Check the latency of running multiple operations.
* tests/poc/flow.py: Check the max banwidth through a matrix multiply.
* tests/poc/calc.py: Check the throughput of a matrix multiply kernel.

### Future Plans

* Backend for llama.cpp that offloads matrix multiplies to the NPU.

## Development Hardware

Thanks to a well-timed clearance sale, I managed to get my hands on a laptop
with an AMD Ryzen AI 9 HX 475 and 64 GB of RAM for something approaching a
reasonable price. A major drawback to this particular device is that the BIOS
is crippled to prevent booting Linux (hacked that) or adjusting clock speeds.

The [FastFlowLM project](https://github.com/ROCm/FastFlowLM) is currently the
best choice for running inference on the NPU. Testing on my laptop it is
roughly equivalent to the Radeon 890M iGPU, but runs much cooler:

| Engine             | Model                       | Size       | Decode |
| :---               | :---                        | :---:      | :---:  |
| FastFlowLM-1.0.4   | Qwen 3.6 35B-A3B Q4_K Aug26 | 21.3 GB    | 18 tps |
| llama-b10839 ROCm  | Qwen 3.6 35B-A3B UD-Q4_K_XL | 21.3 GB    | 20 tps |
| llama-b10839 ROCm  | Qwen 3.6 35B-A3B UD-Q8_K_XL | 36.4 GB    | 18 tps |
| llama-b10839 ROCm  | Gemma 4 26B-A4B IT QAT Q4_0 | 13.4 GB    | 25 tps |
| llama-b10839 ROCm  | Qwen 3.8 27B UD-Q4_K_XL     | 16.4 GB    | 4 tps  |

Interestingly, Qwen 3.6 35B-A3B runs as fast in Q8 as the smaller Q4 versions,
suggesting that the bottleneck is not the system memory bandwidth. The dense
Qwen 3.8 27B model used roughly `16.4 GB/tok * 4 tok/s = 65.6 GB/s`. If the
same memory bandwidth were saturated by Qwen 35B-A3B Q4, we would double the
basic decode rate to 36 tps. Adding speculative decoding like MTP or DFlash
could easily get us over 50 tps.

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
