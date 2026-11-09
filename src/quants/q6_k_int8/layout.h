#pragma once

#include <stdint.h>

#include "quants/common/tile.h"

namespace spiral::quants::q6_k_int8 {

// Canonical GGUF Q6_K block: 256 rows of one weights column.
constexpr int kBlockRows = 256;
constexpr int kBlockBytes = 210;
constexpr int kBlockQlBytes = 128;
constexpr int kBlockQhBytes = 64;
constexpr int kBlockScalesBytes = 16;
constexpr int kBlockScaleBytes = 2;

// Repacked record layout. Weights are stored as signed (q - 32) bytes. All
// payload chunks come first, then all per-chunk scale bytes, then the shared
// fp16 d array. This keeps every 64-byte tile load aligned without per-chunk
// padding.
constexpr int kTileCols = 8;
constexpr int kUnpackTileRows = 64;
constexpr int kRecordRows = 256;
constexpr int kChunksPerRecord = 4;
constexpr int kTileRows = 8;

constexpr int kPackedChunkScalesBytes = 32;
constexpr int kPackedChunkPayloadBytes = kUnpackTileRows * kTileCols;
constexpr int kPackedPayloadsBytes =
    kChunksPerRecord * kPackedChunkPayloadBytes;
constexpr int kPackedScalesBytes = kChunksPerRecord * kPackedChunkScalesBytes;
constexpr int kPackedScalesOffset = kPackedPayloadsBytes;
constexpr int kPackedChunkBytes = kPackedChunkPayloadBytes;

constexpr int kCommonScaleBytes = 16;
constexpr int kPackedRecordDataBytes =
    kPackedPayloadsBytes + kPackedScalesBytes + kCommonScaleBytes;
constexpr int kPackedRecordPaddingBytes =
    (64 - kPackedRecordDataBytes % 64) % 64;
constexpr int kPackedRecordBytes =
    kPackedRecordDataBytes + kPackedRecordPaddingBytes;

static_assert(kTileCols == ::spiral::quants::kMmulWeightsCols,
              "Q6_K tile columns must match the mmul tile width");
static_assert(kTileRows == ::spiral::quants::kMmulWeightsRows,
              "Q6_K tile rows must match the mmul tile rows");
static_assert(kPackedChunkPayloadBytes == kUnpackTileRows * kTileCols,
              "Q6_K payload must be one byte per weight");
static_assert(kRecordRows == kChunksPerRecord * kUnpackTileRows,
              "Q6_K record and chunk depths disagree");
static_assert(kPackedRecordBytes == 2240, "Q6_K packed record size changed");
static_assert(kPackedRecordBytes % 64 == 0,
              "Q6_K records must keep tile loads 64-byte aligned");
static_assert(kPackedPayloadsBytes % 64 == 0,
              "Q6_K payload region must keep tile loads 64-byte aligned");

inline const uint8_t* chunk(const uint8_t* record, int chunk_index) {
  return record + chunk_index * kPackedChunkBytes;
}

inline const uint8_t* chunk_scales(const uint8_t* record, int chunk_index) {
  return record + kPackedScalesOffset + chunk_index * kPackedChunkScalesBytes;
}

inline const uint8_t* record_scales(const uint8_t* record) {
  return record + kPackedScalesOffset + kPackedScalesBytes;
}

}  // namespace spiral::quants::q6_k_int8
