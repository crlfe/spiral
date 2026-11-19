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

// One decoder tile is 8 rows x 8 columns.
constexpr int kTileCols = 8;
constexpr int kTileRows = 8;
constexpr int kUnpackTileRows = 64;
constexpr int kRecordRows = 256;
constexpr int kChunksPerRecord = 4;

// The repack unit and transport unit is 256 rows x 16 columns, i.e. two
// adjacent 8-column records. All payloads come first, then the per-chunk
// scales, then the per-column bf16 hi/lo d pairs. The d region absorbs the
// former 32-byte tail padding, so the tile still ends on a 64-byte boundary
// while sharing its layout between the two records.
constexpr int kTransportTileRows = 256;
constexpr int kTransportTileCols = 16;
constexpr int kRecordsPerTransportTile = kTransportTileCols / kTileCols;

constexpr int kPackedChunkScalesBytes = 32;
constexpr int kPackedChunkLowBytes = kUnpackTileRows * kTileCols / 2;
constexpr int kPackedChunkHighBytes = kUnpackTileRows * kTileCols / 4;
constexpr int kPackedChunkPayloadBytes =
    kPackedChunkLowBytes + kPackedChunkHighBytes;
constexpr int kPackedPayloadsPerRecordBytes =
    kChunksPerRecord * kPackedChunkPayloadBytes;
constexpr int kPackedScalesPerRecordBytes =
    kChunksPerRecord * kPackedChunkScalesBytes;
// The per-column fp16 d value is pre-split on the host into an exact bf16
// hi/lo pair, stored as two planes. This keeps subnormal canonical d values
// exactly representable without any device-side fp16 decode.
constexpr int kPackedScaleBytesPerCol = 4;
constexpr int kCommonScalePlaneBytes = kTileCols * kBlockScaleBytes;
constexpr int kCommonScaleBytes = kTileCols * kPackedScaleBytesPerCol;

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
static_assert(kPackedChunkPayloadBytes == kUnpackTileRows * kTileCols * 3 / 4,
              "Q6_K packed payload must be six bits per weight");
static_assert(kRecordRows == kChunksPerRecord * kUnpackTileRows,
              "Q6_K record and chunk depths disagree");
static_assert(kTransportTileRows == kRecordRows,
              "Q6_K transport tile must hold one record depth");
static_assert(kTransportTileCols % kTileCols == 0,
              "Q6_K transport tile must contain whole records");
static_assert(kPackedChunkPayloadBytes % 64 == 0,
              "Q6_K payload chunk must keep 64-byte alignment");
static_assert(kPackedChunkLowBytes % 64 == 0,
              "Q6_K low plane must keep 64-byte alignment");
static_assert(kPackedChunkHighBytes % 64 == 0,
              "Q6_K high plane must keep 64-byte alignment");
static_assert(kTilePayloadsOffset % 64 == 0,
              "Q6_K payload region must start 64-byte aligned");
static_assert(kTileScalesOffset % 32 == 0,
              "Q6_K chunk scales must start 32-byte aligned");
static_assert(kTileCommonScalesOffset % 16 == 0,
              "Q6_K d values must start 16-byte aligned");
static_assert(kCommonScalePlaneBytes % 16 == 0,
              "Q6_K d hi/lo planes must keep 16-byte alignment");
static_assert(kCommonScaleBytes == 2 * kCommonScalePlaneBytes,
              "Q6_K d record must hold one bf16 hi and lo plane");
static_assert(kTransportTilePayloadsBytes % 64 == 0,
              "Q6_K payload region must keep 64-byte alignment");
static_assert(kPackedTransportTileDataBytes == 3392,
              "Q6_K transport tile data size changed");
static_assert(kPackedTransportTileBytes == 3392,
              "Q6_K transport tile size changed");
static_assert(kPackedTransportTileBytes % 64 == 0,
              "Q6_K transport tiles must be 64-byte aligned");

// Resolve a logical column tile / row record to its 256x16 transport tile.
inline const uint8_t* transport_tile(const uint8_t* packed,
                                     int weights_cols_tile,
                                     int weights_rows_record,
                                     int cols_pairs) {
  return packed + (weights_rows_record * cols_pairs + weights_cols_tile / 2) *
                      kPackedTransportTileBytes;
}

inline uint8_t* transport_tile(uint8_t* packed,
                               int weights_cols_tile,
                               int weights_rows_record,
                               int cols_pairs) {
  return packed + (weights_rows_record * cols_pairs + weights_cols_tile / 2) *
                      kPackedTransportTileBytes;
}

// Payloads start at the beginning of each record's tile-local region.
inline const uint8_t* record_payloads(const uint8_t* tile, int local_col) {
  return tile + kTilePayloadsOffset + local_col * kPackedPayloadsPerRecordBytes;
}

inline uint8_t* record_payloads(uint8_t* tile, int local_col) {
  return tile + kTilePayloadsOffset + local_col * kPackedPayloadsPerRecordBytes;
}

inline const uint8_t* chunk(const uint8_t* payloads, int chunk_index) {
  return payloads + chunk_index * kPackedChunkPayloadBytes;
}

inline uint8_t* chunk(uint8_t* payloads, int chunk_index) {
  return payloads + chunk_index * kPackedChunkPayloadBytes;
}

inline const uint8_t* chunk_scales(const uint8_t* tile,
                                   int local_col,
                                   int chunk_index) {
  return tile + kTileScalesOffset + local_col * kPackedScalesPerRecordBytes +
         chunk_index * kPackedChunkScalesBytes;
}

inline uint8_t* chunk_scales(uint8_t* tile, int local_col, int chunk_index) {
  return tile + kTileScalesOffset + local_col * kPackedScalesPerRecordBytes +
         chunk_index * kPackedChunkScalesBytes;
}

inline const uint8_t* record_scales(const uint8_t* tile, int local_col) {
  return tile + kTileCommonScalesOffset + local_col * kCommonScaleBytes;
}

inline uint8_t* record_scales(uint8_t* tile, int local_col) {
  return tile + kTileCommonScalesOffset + local_col * kCommonScaleBytes;
}

}  // namespace spiral::quants::q6_k
