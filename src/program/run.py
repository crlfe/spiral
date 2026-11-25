"""Composition root for the fixed NPU program.

The design is assembled from three parts: the static device configuration
(`npu.py`), the host runtime sequence (`sequence.py`), and the worker program
(`worker.py`).  `build.py` and the Python test runner enter through the JIT
entry points at the bottom.
"""

import sys
from pathlib import Path

from aie import iron
from aie.iron import In, Out, Program, Runtime
from aie.iron.device import NPU2

PROGRAM_DIR = Path(__file__).resolve().parent
if str(PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_DIR))

import design  # noqa: E402
import npu  # noqa: E402
import sequence  # noqa: E402
import worker  # noqa: E402

# The Python modules and kernel sources are part of the JIT recipe: their
# content is hashed with the kernel sources, so a fixed-constant or helper edit
# invalidates the cache automatically.
COMMON_SOURCE_FILES = [
    Path(design.__file__).resolve(),
    Path(__file__).resolve(),
    Path(npu.__file__).resolve(),
    Path(sequence.__file__).resolve(),
    Path(worker.__file__).resolve(),
    *npu.KERNEL_SOURCE_FILES,
]
SOURCE_FILES = COMMON_SOURCE_FILES + npu.DIAGNOSTIC_SOURCE_FILES
# Production builds omit the diagnostic split path and timing probes.  Keep
# this list separate from SOURCE_FILES so the JIT cache keys stay distinct.
PRODUCTION_SOURCE_FILES = COMMON_SOURCE_FILES


def _make_program(dynamic, production=False, prenorm=False):
    """Build the fixed-max program.

    Flow: static device configuration -> worker program -> runtime sequence.
    ``dynamic`` selects the runtime-sized host sequence used by the C++
    caller; production omits the diagnostic path.  ``prenorm`` selects the
    second, prelude-carrying host body.
    """
    device = npu.build(production)
    workers = worker.build_workers(device, dynamic, production)
    rt = Runtime(
        sequence.build(dynamic, prenorm),
        npu.runtime_args(device, dynamic),
        name="prenorm_sequence" if prenorm else "sequence",
        strict_task_groups=False,
    )
    return Program(NPU2(), rt, workers).resolve_program()


@iron.jit(source_files=SOURCE_FILES)
def program(
    weights: In,
    inputs: In,
    config: In,
    outputs: Out,
):
    """Fixed-max program used by the Python test runner."""
    return _make_program(False)


@iron.jit(source_files=SOURCE_FILES)
def dynamic_program(
    weights: In,
    inputs: In,
    config: In,
    outputs: Out,
):
    """Runtime-sized diagnostic program used by ``build.py``."""
    return _make_program(True)


@iron.jit(source_files=PRODUCTION_SOURCE_FILES)
def production_dynamic_program(
    weights: In,
    inputs: In,
    config: In,
    outputs: Out,
):
    return _make_program(True, production=True)


@iron.jit(source_files=SOURCE_FILES)
def prenorm_dynamic_program(
    weights: In,
    inputs: In,
    config: In,
    outputs: Out,
):
    """Diagnostic program with the prelude-carrying host body."""
    return _make_program(True, prenorm=True)


@iron.jit(source_files=PRODUCTION_SOURCE_FILES)
def production_prenorm_dynamic_program(
    weights: In,
    inputs: In,
    config: In,
    outputs: Out,
):
    """Production program with the prelude-carrying host body."""
    return _make_program(True, production=True, prenorm=True)


if __name__ == "__main__":
    from tests import main

    raise SystemExit(main())
