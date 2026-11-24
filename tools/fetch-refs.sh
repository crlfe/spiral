#!/usr/bin/env bash
# Sparse, blobless clones of the pinned upstream references into .refs/.
#
#   tools/fetch-refs.sh             # both
#   tools/fetch-refs.sh mlir-aie    # one of them
#
# The pins here and in docs/upstream.md are the same record; change both.
# A submodule is deliberately not used: these trees are read-only reference
# material, and a full checkout is several hundred MB plus its own submodules.

set -euo pipefail

MLIR_AIE_REPO=https://github.com/Xilinx/mlir-aie.git
MLIR_AIE_REF=a1fd96791
MLIR_AIE_DIRS="docs programming_examples aie_kernels include"

LLVM_AIE_REPO=https://github.com/Xilinx/llvm-aie.git
LLVM_AIE_REF=eafecd8716ecf70d88577fddf5ab81cd17884650
LLVM_AIE_DIRS="llvm/lib/Target/AIE llvm/test/MC/AIE"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REFS="${ROOT}/.refs"

fetch() {
    name=$1 repo=$2 ref=$3 dirs=$4
    dst="${REFS}/${name}"

    if [ -d "${dst}/.git" ] && git -C "${dst}" cat-file -e "${ref}^{commit}" 2>/dev/null; then
        echo "${name}: already present at ${ref}"
    else
        rm -rf "${dst}"
        mkdir -p "${REFS}"
        echo "${name}: cloning ${repo} (blobless)"
        git clone --filter=blob:none --no-checkout "${repo}" "${dst}"
        git -C "${dst}" sparse-checkout init --cone
        # shellcheck disable=SC2086
        git -C "${dst}" sparse-checkout set ${dirs}
    fi

    git -C "${dst}" checkout --detach "${ref}" >/dev/null
    git -C "${dst}" sparse-checkout reapply
    echo "${name}: ${ref} in ${dst}"
}

case "${1:-all}" in
    mlir-aie) fetch mlir-aie "${MLIR_AIE_REPO}" "${MLIR_AIE_REF}" "${MLIR_AIE_DIRS}" ;;
    llvm-aie) fetch llvm-aie "${LLVM_AIE_REPO}" "${LLVM_AIE_REF}" "${LLVM_AIE_DIRS}" ;;
    all)
        fetch mlir-aie "${MLIR_AIE_REPO}" "${MLIR_AIE_REF}" "${MLIR_AIE_DIRS}"
        fetch llvm-aie "${LLVM_AIE_REPO}" "${LLVM_AIE_REF}" "${LLVM_AIE_DIRS}"
        ;;
    *)
        echo "usage: $0 [mlir-aie|llvm-aie|all]" >&2
        exit 2
        ;;
esac
