#pragma once

#include <stdint.h>

#include "quants/common/tile.h"

namespace spiral::quants::q6_k {

// Canonical GGUF Q6_K block: 256 rows of one weights column.
constexpr int kBlockRows = 256;
constexpr int kBlockBytes = 210;
constexpr int kBlockQlBytes = 128;
constexpr int kBlockQhBytes = 64;
constexpr int kBlockScalesBytes = 16;
constexpr int kBlockScaleBytes = 2;

// Repacked record/chunk layout. The intra-chunk packing is provisional.
constexpr int kTileCols = 8;
constexpr int kUnpackTileRows = 64;
constexpr int kRecordRows = 256;
constexpr int kChunksPerRecord = 4;
constexpr int kCommonScaleBytes = 16;
constexpr int kPackedChunkBytes = 416;
constexpr int kPackedRecordBytes = kCommonScaleBytes + kChunksPerRecord * kPackedChunkBytes;

static_assert(kTileCols == ::spiral::quants::kMmulWeightsCols,
              "Q6_K tile columns must match the mmul tile width");
static_assert(kRecordRows == kChunksPerRecord * kUnpackTileRows,
              "Q6_K record and chunk depths disagree");
static_assert(kPackedChunkBytes == 416, "Q6_K packed chunk size changed");
static_assert(kPackedRecordBytes == 1680, "Q6_K packed record size changed");

inline const uint8_t* common_scales(const uint8_t* record) {
  return record;
}

inline const uint8_t* chunk(const uint8_t* record, int chunk_index) {
  return record + kCommonScaleBytes + chunk_index * kPackedChunkBytes;
}

}  // namespace spiral::quants::q6_k
