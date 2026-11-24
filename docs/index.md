> Role: navigation. Current as of 2026-11-24. History: `git log -- docs/index.md`.

# Documentation index

Every document in the repository, what it is for, and when to read it.
`AGENTS.md` is the only file you must read before changing code; this page is
how you find the rest.

## Start here

| document | role | read when |
| :-- | :-- | :-- |
| [`../README.md`](../README.md) | orientation | first contact with the project, hardware, headline results |
| [`../AGENTS.md`](../AGENTS.md) | contract | every agent session; invariants, traps, workflow, navigation |
| [`measurements.md`](measurements.md) | numbers | quoting or comparing any performance figure |
| [`targets.md`](targets.md) | reference | choosing a model target, or sizing a format or capability against one |
| [`rejected.md`](rejected.md) | negative results | before trying an optimisation or a format variant |
| [`upstream.md`](upstream.md) | pins | reading mlir-aie/llvm-aie sources, or reproducing the toolchain |

## Component reference

| document | role | read when |
| :-- | :-- | :-- |
| [`../src/quants/README.md`](../src/quants/README.md) | orientation | using the quant formats; package layout |
| [`../src/quants/DESIGN.md`](../src/quants/DESIGN.md) | reference | writing or optimising a kernel; format/tile/API contracts; exactness |
| [`../src/quants/TODO.md`](../src/quants/TODO.md) | tasks | picking up the next format or multi-core work |
| [`../src/program/README.md`](../src/program/README.md) | orientation | building, running and calling the fixed program |
| [`../src/program/DESIGN.md`](../src/program/DESIGN.md) | reference | changing the worker, data flow, config words or epilogues |
| [`../src/program/TODO.md`](../src/program/TODO.md) | tasks | launch consolidation, chain links, open risks |
| [`../integrations/llama.cpp/README.md`](../integrations/llama.cpp/README.md) | orientation | attaching and building the ggml backend |
| [`../integrations/llama.cpp/DESIGN.md`](../integrations/llama.cpp/DESIGN.md) | reference | backend matchers, fallback, buffer and preparation design |
| [`../integrations/llama.cpp/TODO.md`](../integrations/llama.cpp/TODO.md) | tasks | backend milestones, open questions |

## Experiments

| document | role | read when |
| :-- | :-- | :-- |
| [`../tests/bench/README.md`](../tests/bench/README.md) | orientation | probing a compiler or float16 question |
| [`../tests/poc/README.md`](../tests/poc/README.md) | record | bandwidth, call overhead, overlay and TXN results |
| [`../tests/poc/overlay.md`](../tests/poc/overlay.md) | record | program-memory overlays and control packets |
| [`../tests/poc/stream/README.md`](../tests/poc/stream/README.md) | record | quantized weight streaming layout |
| [`../tests/poc/txn_concat/README.md`](../tests/poc/txn_concat/README.md) | record | chained matmuls in one command |

## Conventions

- **Orientation** documents answer "what is this and how do I run it".
- **Reference** documents are edited in place and contain no history.
- **Tasks** documents contain open work only; finished work is deleted.
- **Records** are frozen evidence from a completed experiment; they keep
  their original numbers and dates and are not updated.
- Numbers live in [`measurements.md`](measurements.md) or in a record. Other
  documents link instead of restating them.
- Every document opens with a role/date line. `git log -- <path>` is the
  history; there are no changelogs in the tree.
