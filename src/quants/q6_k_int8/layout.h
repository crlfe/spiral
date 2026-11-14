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

// One decoder tile is 8 rows x 8 columns.
constexpr int kTileCols = 8;
constexpr int kTileRows = 8;
constexpr int kUnpackTileRows = 64;
constexpr int kRecordRows = 256;
constexpr int kChunksPerRecord = 4;

// The repack and transport unit is 256 rows x 16 columns. Weights are stored
// as signed (q - 32) bytes. All payload chunks come first, then the per-chunk
// scale bytes for both records, then the per-column fp16 d values, then a
// single shared 32-byte tail pad.
constexpr int kTransportTileRows = 256;
constexpr int kTransportTileCols = 16;
constexpr int kRecordsPerTransportTile = kTransportTileCols / kTileCols;

constexpr int kPackedChunkScalesBytes = 32;
constexpr int kPackedChunkPayloadBytes = kUnpackTileRows * kTileCols;
constexpr int kPackedPayloadsPerRecordBytes =
    kChunksPerRecord * kPackedChunkPayloadBytes;
constexpr int kPackedScalesPerRecordBytes =
    kChunksPerRecord * kPackedChunkScalesBytes;
constexpr int kCommonScaleBytes = 16;

constexpr int kTransportTilePayloadsBytes =
    kRecordsPerTransportTile * kPackedPayloadsPerRecordBytes;
constexpr int kTransportTileScalesBytes =
    kRecordsPerTransportTile * kPackedScalesPerRecordBytes;
constexpr int kTransportTileCommonScalesBytes =
    kRecordsPerTransportTile * kCommonScaleBytes;

constexpr int kTilePayloadsOffset = 0;
constexpr int kTileScalesOffset = kTransportTilePayloadsBytes;
constexpr int kTileCommonScalesOffset =
    kTileScalesOffset + kTransportTileScalesBytes;
constexpr int kPackedTransportTileDataBytes =
    kTileCommonScalesOffset + kTransportTileCommonScalesBytes;
constexpr int kPackedTransportTileBytes =
    (kPackedTransportTileDataBytes + 63) / 64 * 64;

static_assert(kTileCols == ::spiral::quants::kMmulWeightsCols,
              "Q6_K tile columns must match the mmul tile width");
static_assert(kTileRows == ::spiral::quants::kMmulWeightsRows,
              "Q6_K tile rows must match the mmul tile rows");
static_assert(kPackedChunkPayloadBytes == kUnpackTileRows * kTileCols,
              "Q6_K payload must be one byte per weight");
static_assert(kRecordRows == kChunksPerRecord * kUnpackTileRows,
              "Q6_K record and chunk depths disagree");
static_assert(kTransportTileRows == kRecordRows,
              "Q6_K transport tile must hold one record depth");
static_assert(kTransportTileCols % kTileCols == 0,
              "Q6_K transport tile must contain whole records");
static_assert(kTransportTilePayloadsBytes % 64 == 0,
              "Q6_K payload region must keep 64-byte alignment");
static_assert(kPackedTransportTileDataBytes == 4384,
              "Q6_K transport tile data size changed");
static_assert(kPackedTransportTileBytes == 4416,
              "Q6_K transport tile size changed");
static_assert(kPackedTransportTileBytes % 64 == 0,
              "Q6_K transport tiles must be 64-byte aligned");

inline const uint8_t* record_payloads(const uint8_t* tile, int local_col) {
  return tile + kTilePayloadsOffset +
         local_col * kPackedPayloadsPerRecordBytes;
}

inline uint8_t* record_payloads(uint8_t* tile, int local_col) {
  return tile + kTilePayloadsOffset +
         local_col * kPackedPayloadsPerRecordBytes;
}

inline const uint8_t* chunk(const uint8_t* payloads, int chunk_index) {
  return payloads + chunk_index * kPackedChunkPayloadBytes;
}

inline uint8_t* chunk(uint8_t* payloads, int chunk_index) {
  return payloads + chunk_index * kPackedChunkPayloadBytes;
}

inline const uint8_t* chunk_scales(const uint8_t* tile, int local_col,
                                   int chunk_index) {
  return tile + kTileScalesOffset +
         local_col * kPackedScalesPerRecordBytes +
         chunk_index * kPackedChunkScalesBytes;
}

inline uint8_t* chunk_scales(uint8_t* tile, int local_col, int chunk_index) {
  return tile + kTileScalesOffset +
         local_col * kPackedScalesPerRecordBytes +
         chunk_index * kPackedChunkScalesBytes;
}

inline const uint8_t* record_scales(const uint8_t* tile, int local_col) {
  return tile + kTileCommonScalesOffset + local_col * kCommonScaleBytes;
}

inline uint8_t* record_scales(uint8_t* tile, int local_col) {
  return tile + kTileCommonScalesOffset + local_col * kCommonScaleBytes;
}

}  // namespace spiral::quants::q6_k_int8
