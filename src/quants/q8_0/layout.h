#pragma once

#include <stdint.h>

#include "quants/common/tile.h"

namespace spiral::quants::q8_0 {

// Canonical GGUF Q8_0 block: 32 rows of one weights column.
constexpr int kBlockRows = 32;
constexpr int kBlockBytes = 34;
constexpr int kBlockScaleBytes = 2;
constexpr int kBlockQuantBytes = 32;

// Repacked record/chunk layout. A record is 256 rows x 8 columns and holds
// eight canonical blocks per column (64 blocks total). Payloads come first so
// every 64-byte tile load is naturally aligned, and the 2176-byte record
// stride is itself a multiple of 64.
constexpr int kTileCols = 8;
constexpr int kTileRows = 8;
constexpr int kBlocksPerRecord = 8;
constexpr int kUnpackTileRows = 256;
constexpr int kRecordRows = 256;
constexpr int kChunksPerRecord = 1;
constexpr int kCommonScaleBytes = 0;
constexpr int kScaleBytesPerBlock = kTileCols * kBlockScaleBytes;
constexpr int kScalesBytes = kBlocksPerRecord * kScaleBytesPerBlock;
constexpr int kBlockPayloadRegionBytes = kTileCols * kBlockRows;
constexpr int kPayloadsBytes = kBlocksPerRecord * kBlockPayloadRegionBytes;
constexpr int kPayloadsOffset = 0;
constexpr int kScalesOffset = kPayloadsBytes;
constexpr int kTilesPerBlock = kBlockRows / kTileRows;
constexpr int kTilePayloadBytes = kTileRows * kTileCols;
constexpr int kPackedChunkBytes = kPayloadsBytes + kScalesBytes;
constexpr int kPackedRecordBytes =
    kCommonScaleBytes + kChunksPerRecord * kPackedChunkBytes;

static_assert(kTileCols == ::spiral::quants::kMmulWeightsCols,
              "Q8_0 tile columns must match the mmul tile width");
static_assert(kTileRows == ::spiral::quants::kMmulWeightsRows,
              "Q8_0 tile rows must match the mmul tile rows");
static_assert(kRecordRows == kChunksPerRecord * kUnpackTileRows,
              "Q8_0 record and chunk depths disagree");
static_assert(kRecordRows == kBlocksPerRecord * kBlockRows,
              "Q8_0 record must contain whole canonical blocks");
static_assert(kPayloadsOffset % 64 == 0,
              "Q8_0 payload region must be 64-byte aligned");
static_assert(kPackedRecordBytes % 64 == 0,
              "Q8_0 records must keep tile loads 64-byte aligned");
static_assert(kBlockPayloadRegionBytes % 32 == 0,
              "Q8_0 block payload region must contain whole tiles");
static_assert(kBlockBytes == kBlockScaleBytes + kBlockQuantBytes,
              "Q8_0 canonical block size changed");
static_assert(kPackedChunkBytes == 2176, "Q8_0 packed chunk size changed");
static_assert(kPackedRecordBytes == 2176, "Q8_0 packed record size changed");

inline const uint8_t* chunk(const uint8_t* record, int chunk_index) {
  return record + kCommonScaleBytes + chunk_index * kPackedChunkBytes;
}

inline const uint8_t* block_scales(const uint8_t* packed_chunk, int block) {
  return packed_chunk + kScalesOffset + block * kScaleBytesPerBlock;
}

inline const uint8_t* block_payloads(const uint8_t* packed_chunk, int block) {
  return packed_chunk + kPayloadsOffset + block * kBlockPayloadRegionBytes;
}

}  // namespace spiral::quants::q8_0
