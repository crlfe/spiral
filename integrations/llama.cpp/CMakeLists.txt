cmake_minimum_required(VERSION 3.14)

find_package(PkgConfig REQUIRED)
pkg_check_modules(XRT xrt)

if (NOT XRT_FOUND)
    message(FATAL_ERROR "Spiral: XRT not found (pkg-config module 'xrt')")
endif()

# This backend is developed inside the Spiral tree and reached from llama.cpp
# through a symlink at ggml/src/ggml-spiral.  Resolve the link first, then
# default the device design and host wrapper to the sibling src/program so a
# checkout needs no configuration.  Override these paths when vendoring the
# generated artifacts.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    get_filename_component(_spiral_here "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
else()
    set(_spiral_here "${CMAKE_CURRENT_LIST_DIR}")
endif()

set(SPIRAL_PROGRAM_DIR "${_spiral_here}/../../src/program" CACHE PATH
    "Directory containing src/program (design.py, host/program.cc, ...)")
set(SPIRAL_PROGRAM_BUILD_DIR "${SPIRAL_PROGRAM_DIR}/build-production" CACHE PATH
    "Directory containing the production aie.xclbin, design_meta.h, generate_txn.h")
set(SPIRAL_GOLDEN_DIR "${SPIRAL_PROGRAM_DIR}/build/golden" CACHE PATH
    "Directory containing the golden smoke-test cases")
set(SPIRAL_AIE_INCLUDE_DIR
    "$ENV{HOME}/Projects/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/include"
    CACHE PATH "Directory containing aie/Runtime/TxnEncoding.h")

set(SPIRAL_PROGRAM_SOURCE "${SPIRAL_PROGRAM_DIR}/host/program.cc")
set(SPIRAL_XCLBIN "${SPIRAL_PROGRAM_BUILD_DIR}/aie.xclbin")

if (NOT EXISTS "${SPIRAL_PROGRAM_SOURCE}" OR
    NOT EXISTS "${SPIRAL_XCLBIN}" OR
    NOT EXISTS "${SPIRAL_PROGRAM_BUILD_DIR}/design_meta.h" OR
    NOT EXISTS "${SPIRAL_PROGRAM_BUILD_DIR}/generate_txn.h")
    message(FATAL_ERROR
            "Spiral: set SPIRAL_PROGRAM_DIR and SPIRAL_PROGRAM_BUILD_DIR to a "
            "built src/program tree")
endif()

if (NOT EXISTS "${SPIRAL_AIE_INCLUDE_DIR}/aie/Runtime/TxnEncoding.h")
    message(FATAL_ERROR
            "Spiral: set SPIRAL_AIE_INCLUDE_DIR to the mlir_aie/include "
            "directory containing aie/Runtime/TxnEncoding.h")
endif()

ggml_add_backend_library(ggml-spiral
                         ggml-spiral.cpp
                         "${SPIRAL_PROGRAM_SOURCE}"
                        )

target_compile_features(ggml-spiral PRIVATE cxx_std_17)
# ggml_add_backend_library() adds "${CMAKE_CURRENT_SOURCE_DIR}/..".  Because
# this directory is a symlink, `..` is resolved after the link and lands in the
# Spiral tree, so name ggml's source directory explicitly for
# ggml-backend-impl.h.
if (DEFINED ggml_SOURCE_DIR)
    target_include_directories(ggml-spiral PRIVATE "${ggml_SOURCE_DIR}/src")
endif()
target_include_directories(ggml-spiral PRIVATE
                           "${SPIRAL_PROGRAM_DIR}/host"
                           "${SPIRAL_PROGRAM_BUILD_DIR}"
                           "${SPIRAL_PROGRAM_DIR}/..")
target_include_directories(ggml-spiral SYSTEM PRIVATE
                           ${XRT_INCLUDE_DIRS}
                           "${SPIRAL_AIE_INCLUDE_DIR}")
target_link_libraries(ggml-spiral PRIVATE
                      ${XRT_LIBRARIES}
                      xrt++
                      xrt_coreutil
                      ${CMAKE_DL_LIBS})

option(GGML_SPIRAL_SMOKE "ggml: build the standalone Spiral smoke test" ON)

if (GGML_SPIRAL_SMOKE)
    add_executable(spiral-smoke smoke.cpp)
    target_compile_features(spiral-smoke PRIVATE cxx_std_17)
    target_include_directories(spiral-smoke PRIVATE
                               "${SPIRAL_PROGRAM_DIR}/host"
                               "${SPIRAL_PROGRAM_BUILD_DIR}")
    target_compile_definitions(spiral-smoke PRIVATE
                               SPIRAL_GOLDEN_DIR="${SPIRAL_GOLDEN_DIR}")
    target_link_libraries(spiral-smoke PRIVATE ggml)

    # spiral::Program loads "aie.xclbin" from the working directory.
    add_custom_command(TARGET spiral-smoke POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                               "${SPIRAL_XCLBIN}"
                               "$<TARGET_FILE_DIR:spiral-smoke>/aie.xclbin")
endif()
