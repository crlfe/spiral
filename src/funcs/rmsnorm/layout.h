#pragma once

// Geometry for the standalone RMS norm kernel.  These constants describe the
// program's input-group layout, not a quant format: one group is four K-row
// steps of 256 values, with four input vectors interleaved per 8-value chunk.

namespace spiral::funcs::rmsnorm {

constexpr int kChunk = 8;
constexpr int kVecs = 4;
constexpr int kRowStep = 256;
constexpr int kRowsPerGroup = 4;
constexpr int kChunksPerGroup = kRowStep / kChunk;
constexpr int kSliceRow = kChunksPerGroup * 2 * kChunk;
constexpr int kGroupInputElems = kRowsPerGroup * kRowStep * kVecs;

static_assert(kRowStep % kChunk == 0, "row step must contain whole chunks");
static_assert(kSliceRow == 2 * kRowStep, "weight slice must cover hi/lo");

}  // namespace spiral::funcs::rmsnorm
