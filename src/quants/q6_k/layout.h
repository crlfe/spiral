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

// Base Q6_K repack: the native six-bit payload is preserved. One record is
// 256 rows x 8 columns and contains four 64-row chunks. All four payload
// chunks come first, then all per-chunk scale bytes, then the shared fp16 d
// array. The record is exactly the canonical 1680 bytes.
constexpr int kTileCols = 8;
constexpr int kTileRows = 8;
constexpr int kUnpackTileRows = 64;
constexpr int kRecordRows = 256;
constexpr int kChunksPerRecord = 4;

constexpr int kPackedChunkScalesBytes = 32;
constexpr int kPackedChunkLowBytes = kUnpackTileRows * kTileCols / 2;
constexpr int kPackedChunkHighBytes = kUnpackTileRows * kTileCols / 4;
constexpr int kPackedChunkPayloadBytes =
    kPackedChunkLowBytes + kPackedChunkHighBytes;
constexpr int kPackedChunkBytes = kPackedChunkPayloadBytes;
constexpr int kPackedPayloadsBytes =
    kChunksPerRecord * kPackedChunkPayloadBytes;
constexpr int kPackedScalesBytes = kChunksPerRecord * kPackedChunkScalesBytes;
constexpr int kPackedScalesOffset = kPackedPayloadsBytes;
constexpr int kCommonScaleBytes = kTileCols * kBlockScaleBytes;
constexpr int kPackedRecordDataBytes =
    kPackedPayloadsBytes + kPackedScalesBytes + kCommonScaleBytes;
constexpr int kPackedRecordPaddingBytes = 0;
constexpr int kPackedRecordBytes =
    kPackedRecordDataBytes + kPackedRecordPaddingBytes;

static_assert(kTileCols == ::spiral::quants::kMmulWeightsCols,
              "Q6_K tile columns must match the mmul tile width");
static_assert(kTileRows == ::spiral::quants::kMmulWeightsRows,
              "Q6_K tile rows must match the mmul tile rows");
static_assert(kPackedChunkPayloadBytes == kUnpackTileRows * kTileCols * 3 / 4,
              "Q6_K packed payload must be six bits per weight");
static_assert(kRecordRows == kChunksPerRecord * kUnpackTileRows,
              "Q6_K record and chunk depths disagree");
static_assert(kPackedChunkPayloadBytes % 64 == 0,
              "Q6_K payload chunk must keep 64-byte alignment");
static_assert(kPackedPayloadsBytes % 64 == 0,
              "Q6_K payload region must keep 64-byte alignment");
static_assert(kPackedRecordBytes == 1680, "Q6_K packed record size changed");

inline const uint8_t* chunk(const uint8_t* record, int chunk_index) {
  return record + chunk_index * kPackedChunkPayloadBytes;
}

inline uint8_t* chunk(uint8_t* record, int chunk_index) {
  return record + chunk_index * kPackedChunkPayloadBytes;
}

inline const uint8_t* chunk_scales(const uint8_t* record, int chunk_index) {
  return record + kPackedScalesOffset + chunk_index * kPackedChunkScalesBytes;
}

inline uint8_t* chunk_scales(uint8_t* record, int chunk_index) {
  return record + kPackedScalesOffset + chunk_index * kPackedChunkScalesBytes;
}

inline const uint8_t* record_scales(const uint8_t* record) {
  return record + kPackedScalesOffset + kPackedScalesBytes;
}

inline uint8_t* record_scales(uint8_t* record) {
  return record + kPackedScalesOffset + kPackedScalesBytes;
}

}  // namespace spiral::quants::q6_k
