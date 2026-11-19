#pragma once

#include <stdint.h>

#include "quants/common/tile.h"

namespace spiral::quants::q4_0_packed16 {

// Canonical GGUF Q4_0 block: 32 rows of one weights column.
constexpr int kBlockRows = 32;
constexpr int kBlockBytes = 18;
constexpr int kBlockScaleBytes = 2;
constexpr int kPayloadRows = 4;
constexpr int kPayloadsPerBlock = 8;
constexpr int kPayloadBytes = 16;
constexpr int kBlockPayloadRegionBytes = 128;

// One decoder tile is 8 rows x 8 columns.
constexpr int kTileCols = 8;
constexpr int kBlocksPerRecord = 2;
constexpr int kScaleBytesPerBlock = kTileCols * kBlockScaleBytes;
constexpr int kPackedScalesPerRecordBytes =
    kBlocksPerRecord * kScaleBytesPerBlock;
constexpr int kPackedPayloadsPerRecordBytes =
    kBlocksPerRecord * kBlockPayloadRegionBytes;
constexpr int kPackedRecordBytes =
    kPackedScalesPerRecordBytes + kPackedPayloadsPerRecordBytes;
constexpr int kPayloadsPerTile = 2;
constexpr int kTilePayloadBytes = kPayloadsPerTile * kPayloadBytes;
constexpr int kUnpackTileRows = 64;
constexpr int kRecordRows = 64;
constexpr int kChunksPerRecord = 1;

// The repack and transport unit is 256 rows x 16 columns. Each 64-row record
// keeps its scales first and its payloads second, and records are grouped into
// transport tiles.
constexpr int kTransportTileRows = 256;
constexpr int kTransportTileCols = 16;
constexpr int kRowsRecordsPerTile = kTransportTileRows / kRecordRows;
constexpr int kRecordsPerTransportTile = kTransportTileCols / kTileCols;
constexpr int kPackedTransportTileBytes =
    kRowsRecordsPerTile * kRecordsPerTransportTile * kPackedRecordBytes;

static_assert(kTileCols == ::spiral::quants::kMmulWeightsCols,
              "Q4_0 tile columns must match the mmul tile width");
static_assert(kRecordRows == kChunksPerRecord * kUnpackTileRows,
              "Q4_0 record and chunk depths disagree");
static_assert(kTransportTileRows == kRowsRecordsPerTile * kRecordRows,
              "Q4_0 transport tile must contain whole records");
static_assert(kTransportTileCols == kRecordsPerTransportTile * kTileCols,
              "Q4_0 transport tile must contain whole columns");
static_assert(kPackedTransportTileBytes == 2304,
              "Q4_0 transport tile size changed");
static_assert(kPackedTransportTileBytes % 64 == 0,
              "Q4_0 transport tiles must be 64-byte aligned");

inline const uint8_t* record_ptr(const uint8_t* tile, int local_col,
                                 int local_row) {
  return tile + (local_col * kRowsRecordsPerTile + local_row) *
                    kPackedRecordBytes;
}

inline uint8_t* record_ptr(uint8_t* tile, int local_col, int local_row) {
  return tile + (local_col * kRowsRecordsPerTile + local_row) *
                    kPackedRecordBytes;
}

inline const uint8_t* record_payloads(const uint8_t* tile, int local_col,
                                      int local_row) {
  return record_ptr(tile, local_col, local_row) +
         kPackedScalesPerRecordBytes;
}

inline uint8_t* record_payloads(uint8_t* tile, int local_col, int local_row) {
  return record_ptr(tile, local_col, local_row) +
         kPackedScalesPerRecordBytes;
}

inline const uint8_t* record_scales(const uint8_t* tile, int local_col,
                                    int local_row) {
  return record_ptr(tile, local_col, local_row);
}

inline uint8_t* record_scales(uint8_t* tile, int local_col, int local_row) {
  return record_ptr(tile, local_col, local_row);
}

inline const uint8_t* block_scales(const uint8_t* scales, int block) {
  return scales + block * kScaleBytesPerBlock;
}

inline uint8_t* block_scales(uint8_t* scales, int block) {
  return scales + block * kScaleBytesPerBlock;
}

inline const uint8_t* block_payloads(const uint8_t* payloads, int block) {
  return payloads + block * kBlockPayloadRegionBytes;
}

inline uint8_t* block_payloads(uint8_t* payloads, int block) {
  return payloads + block * kBlockPayloadRegionBytes;
}

}  // namespace spiral::quants::q4_0_packed16
