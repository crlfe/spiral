# Proof of Concept

These tests use the AMD mlir-aie Python library to evaluate basic functionality
and performance of the AMD XDNA 2 NPU. For download and setup instructions, see
[their code repository](https://github.com/Xilinx/mlir-aie).

## Memory Bandwidth

`memory.py` reports speed from host memory to the NPU in GB/s, using a varying
number of DMA channels.

| Device                       |   1   |   2   |   4   |   8   |   16  |
| :---                         | :---: | :---: | :---: | :---: | :---: |
| Ryzen AI 9 HX 475 Laptop     |  14.3 |  28.3 |  46.2 |  55.7 |  56.5 |
|