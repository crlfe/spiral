# Proof of Concept

These tests use the AMD MLIR-AIE Python library to evaluate basic functionality
and performance of the AMD XDNA 2 NPU. For download and setup instructions, see
[their code repository](https://github.com/Xilinx/mlir-aie).

To run examples with inline kernels, you may need to add the Peano LLVM compiler
that MLIR-AIE installed to your PATH after following the MLIR-AIE instructions
to enter their `ironenv` virtual environment. In my case, the default location
was `${VIRTUAL_ENV}/../my_install/mlir/bin`.

## Bulk Memory Bandwidth

`memory.py` reports the bulk transfer speed from host memory to the NPU in GB/s,
testing a varying number of DMA channels.

```
# Ryzen AI 9 HX 475 Laptop

Tested  1 channels: avg 14.3 GB/s   (min 14.2 to max 14.4)
Tested  2 channels: avg 26.7 GB/s   (min 26.1 to max 27.0)
Tested  3 channels: avg 35.2 GB/s   (min 34.7 to max 35.7)
Tested  4 channels: avg 40.5 GB/s   (min 37.5 to max 44.3)
Tested  6 channels: avg 48.9 GB/s   (min 45.5 to max 50.1)
Tested  8 channels: avg 54.4 GB/s   (min 50.7 to max 55.6)
Tested 10 channels: avg 54.6 GB/s   (min 51.8 to max 56.8)
Tested 12 channels: avg 55.2 GB/s   (min 51.2 to max 57.2)
Tested 14 channels: avg 57.6 GB/s   (min 57.4 to max 57.7)
Tested 16 channels: avg 53.0 GB/s   (min 45.8 to max 57.4)
```

## Matrix Memory Bandwidth

`flow.py` reports the transfer speed through the NPU following a path similar
to a matrix multiply, but without performing the actual calculations. This
result is unoptimized and the actual output data has not been verified.

```
# Ryzen AI 9 HX 475 Laptop

Transfer: avg 56.1 GB/s   (min 56.0 to max 56.2)
```
