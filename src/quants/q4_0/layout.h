#pragma once

#include <stdint.h>

#include "quants/common/tile.h"

namespace spiral::quants::q4_0 {

// Canonical GGUF Q4_0 block: 32 rows of one weights column.
constexpr int kBlockRows = 32;
constexpr int kBlockBytes = 18;
constexpr int kBlockScaleBytes = 2;
constexpr int kPayloadRows = 4;
constexpr int kPayloadsPerBlock = 8;
constexpr int kPayloadBytes = 16;
constexpr int kBlockPayloadRegionBytes = 128;

// Repacked record/chunk layout. Scales for both blocks are stored first, then
// all payloads, so every 32-byte tile payload is 32-byte aligned.
constexpr int kTileCols = 8;
constexpr int kBlocksPerRecord = 2;
constexpr int kScaleBytesPerBlock = kTileCols * kBlockScaleBytes;
constexpr int kScalesBytes = kBlocksPerRecord * kScaleBytesPerBlock;
constexpr int kPayloadsBytes = kBlocksPerRecord * kBlockPayloadRegionBytes;
constexpr int kScalesOffset = 0;
constexpr int kPayloadsOffset = kScalesBytes;
constexpr int kPayloadsPerTile = 2;
constexpr int kTilePayloadBytes = kPayloadsPerTile * kPayloadBytes;
constexpr int kUnpackTileRows = 64;
constexpr int kRecordRows = 64;
constexpr int kChunksPerRecord = 1;
constexpr int kCommonScaleBytes = 0;
constexpr int kPackedChunkBytes = kScalesBytes + kPayloadsBytes;
constexpr int kPackedRecordBytes =
    kCommonScaleBytes + kChunksPerRecord * kPackedChunkBytes;

static_assert(kTileCols == ::spiral::quants::kMmulWeightsCols,
              "Q4_0 tile columns must match the mmul tile width");
static_assert(kRecordRows == kChunksPerRecord * kUnpackTileRows,
              "Q4_0 record and chunk depths disagree");
static_assert(kPackedChunkBytes == 288, "Q4_0 packed chunk size changed");
static_assert(kPackedRecordBytes == 288, "Q4_0 packed record size changed");
static_assert(kPayloadsOffset % 32 == 0,
              "Q4_0 payload region must be 32-byte aligned");
static_assert(kBlockPayloadRegionBytes % 32 == 0,
              "Q4_0 block payload region must contain whole tiles");

inline const uint8_t* chunk(const uint8_t* record, int chunk_index) {
  return record + kCommonScaleBytes + chunk_index * kPackedChunkBytes;
}

inline const uint8_t* block_scales(const uint8_t* packed_chunk, int block) {
  return packed_chunk + kScalesOffset + block * kScaleBytesPerBlock;
}

inline const uint8_t* block_payloads(const uint8_t* packed_chunk, int block) {
  return packed_chunk + kPayloadsOffset + block * kBlockPayloadRegionBytes;
}

}  // namespace spiral::quants::q4_0
