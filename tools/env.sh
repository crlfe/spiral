#!/bin/sh
# Source this to get the MLIR-AIE / Peano environment Spiral expects.
#
#   source tools/env.sh
#
# Resolution order:
#   1. $SPIRAL_MLIR_AIE          explicit override
#   2. ../mlir-aie               conventional sibling checkout
#   3. third_party/mlir-aie      optional vendored tree
#   4. $MLIR_AIE_INSTALL_DIR     already configured by the caller
#
# The first tree that has iron_env.sh is sourced, which also activates its
# virtualenv. SPIRAL_PY then names the interpreter to use for repository
# scripts.

_spiral_root=${SPIRAL_ROOT:-}
if [ -z "${_spiral_root}" ] && [ -n "${BASH_SOURCE:-}" ]; then
    _spiral_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [ -z "${_spiral_root}" ]; then
    # Sourced from a POSIX shell without BASH_SOURCE: find the repository by
    # walking up from the current directory.
    _d=$(pwd)
    while [ "${_d}" != "/" ]; do
        if [ -f "${_d}/src/program/build.py" ]; then
            _spiral_root=${_d}
            break
        fi
        _d=$(dirname -- "${_d}")
    done
    unset _d
fi
if [ -z "${_spiral_root}" ]; then
    echo "tools/env.sh: source this from inside the Spiral repository," >&2
    echo "or set SPIRAL_ROOT to the repository root." >&2
    return 1 2>/dev/null || exit 1
fi

_spiral_mlir_aie=
for _cand in "${SPIRAL_MLIR_AIE:-}" \
             "${_spiral_root}/../mlir-aie" \
             "${_spiral_root}/third_party/mlir-aie"; do
    [ -n "$_cand" ] || continue
    if [ -f "${_cand}/iron_env.sh" ]; then
        _spiral_mlir_aie=$(CDPATH= cd -- "$_cand" && pwd)
        break
    fi
done

if [ -n "$_spiral_mlir_aie" ]; then
    . "${_spiral_mlir_aie}/iron_env.sh"
elif [ -n "${MLIR_AIE_INSTALL_DIR:-}" ]; then
    _spiral_mlir_aie=${MLIR_AIE_INSTALL_DIR%/lib/python*/site-packages/mlir_aie}
else
    echo "tools/env.sh: no MLIR-AIE tree found." >&2
    echo "Set SPIRAL_MLIR_AIE to a checkout with iron_env.sh, or source the" >&2
    echo "environment first and export MLIR_AIE_INSTALL_DIR." >&2
    unset _spiral_root _spiral_mlir_aie _cand
    return 1 2>/dev/null || exit 1
fi

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    SPIRAL_PY="${VIRTUAL_ENV}/bin/python"
else
    SPIRAL_PY=python3
fi
export SPIRAL_PY
export SPIRAL_MLIR_AIE="${_spiral_mlir_aie}"

unset _spiral_root _cand
