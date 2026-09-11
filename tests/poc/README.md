# Proof of Concept

These tests use the AMD MLIR-AIE Python library to evaluate basic functionality
and performance of the AMD XDNA 2 NPU. For download and setup instructions, see
[their code repository](https://github.com/Xilinx/mlir-aie).

To run examples with inline kernels, you may need to add the Peano LLVM compiler
that MLIR-AIE installed to your PATH after following the MLIR-AIE instructions
to enter their `ironenv` virtual environment. In my case, the default location
was `${VIRTUAL_ENV}/../my_install/mlir/bin`.

## Memory Bandwidth

`memory.py` reports speed from host memory to the NPU in GB/s, using a varying
number of DMA channels.

```
# Ryzen AI 9 HX 475 Laptop

Tested  1 channels: avg 13.8 GB/s   (min 13.5 to max 14.1)
Tested  2 channels: avg 25.9 GB/s   (min 25.1 to max 26.4)
Tested  3 channels: avg 36.9 GB/s   (min 36.3 to max 37.5)
Tested  4 channels: avg 43.9 GB/s   (min 43.3 to max 44.8)
Tested  6 channels: avg 49.5 GB/s   (min 48.4 to max 50.0)
Tested  8 channels: avg 54.8 GB/s   (min 54.5 to max 55.2)
Tested 10 channels: avg 56.2 GB/s   (min 56.0 to max 56.3)
Tested 12 channels: avg 56.7 GB/s   (min 56.4 to max 56.9)
Tested 14 channels: avg 56.7 GB/s   (min 56.4 to max 57.0)
Tested 16 channels: avg 56.9 GB/s   (min 56.8 to max 57.1)
```
