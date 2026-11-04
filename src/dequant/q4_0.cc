//
// Q4_0 -> bfloat16 dequantization tile.
//
// The kernel consumes a host-repacked tile of 16 Q4_0 blocks: 8 output columns
// by 2 k blocks (64 reduction values). The host transposes the int4 values
// while repacking, so the kernel can emit the mmul B layout directly without
// an in-register transpose.
//
// Layout per k block (144 bytes, still 4.5 bits/weight):
//
//     [8 x (4 K x 8 N) signed-int4 payloads][8 fp16 scales]
//
// - Each payload is 16 bytes and covers four K rows by eight columns, in
//   row-major order with signed values (q - 8).
// - The scale group is 8 contiguous fp16 scales, one per column.
// - Payloads start at offset 0, so every payload load is 16-byte aligned; the
//   scales sit at offset 128.
//
// Each fp16 scale is split into hi/lo bf16 components:
//
//     scale ~= hi_bf16 + lo_bf16
//
// The kernel multiplies each payload by both components and accumulates in
// the f32 accumulator before the single final bf16 round. This recovers the
// fp16 scale precision without converting weights through f32, and matches the
// float dequant path on random fp16 scales.
//
// The output is the bf16 B-tile layout consumed by aie::mmul<4, 8, 8>:
//
//     [k subtile: 8][k inside: 8][column: 8]
//
// which is 512 bf16 values: eight 8x8 K-major subtiles. Each 4x8 payload is
// stored as one half of an 8x8 subtile.
//
// The kernel brackets its body with event0()/event1() so NPU event tracing can
// measure it directly.
//

#include <stdint.h>
#include <aie_api/aie.hpp>

namespace {

constexpr int kBlockWeights = 32;  // weights per Q4_0 block
constexpr int kCols = 8;           // output columns in one tile
constexpr int kKBlocksPerCol = 2;  // 2 * 32 = 64 reduction values
constexpr int kTileBlocks = kCols * kKBlocksPerCol;  // 16 blocks per tile

constexpr int kPayloadRows = 4;     // 4 K rows per transposed payload
constexpr int kPayloadsPerKBlock = 8;  // 8 payloads = 32 K rows
constexpr int kPayloadBytes = 16;   // 32 signed int4 values
constexpr int kPayloadRegion = kPayloadsPerKBlock * kPayloadBytes;  // 128
constexpr int kScaleOffset = kPayloadRegion;
constexpr int kScaleBytes = kCols * 2;  // 8 fp16 scales
constexpr int kUnitBytes = kPayloadRegion + kScaleBytes;  // 144

constexpr int kTileK = kKBlocksPerCol * kBlockWeights;
constexpr int kMmulK = 8;
constexpr int kOutValues = kTileK * kCols;  // 512 bf16 values

static inline aie::vector<float, 8> load_scales(const uint8_t* unit) {
  // Convert the eight normal fp16 scales with one packed bit trick:
  //   f32 = sign | ((exp16 + 112) << 23) | (mantissa << 13)
  // Subnormal fp16 scales are not supported; Q4_0 scales are normal in
  // practice, and this keeps the path branch-free.
  const auto h = aie::load_v<8>(
      reinterpret_cast<const uint16_t*>(unit + kScaleOffset));
  const auto u = h.unpack();

  const auto sign = aie::upshift(aie::bit_and(0x8000u, u), 16);
  const auto exponent = aie::upshift(
      aie::add(aie::logical_downshift(aie::bit_and(0x7c00u, u), 10), 112u),
      23);
  const auto mantissa = aie::upshift(aie::bit_and(0x03ffu, u), 13);
  const auto bits = aie::bit_or(sign, aie::bit_or(exponent, mantissa));
  return bits.template cast_to<float>();
}

static inline aie::vector<int4, 32> load_payload(const uint8_t* unit,
                                                 int payload) {
  const uint8_t* p = unit + payload * kPayloadBytes;
  return aie::load_v<16>(p).template cast_to<int4>();
}

}  // namespace

extern "C" void q4_0_dequant_tile(const uint8_t* restrict blocks,
                                  bfloat16* restrict out) {
  event0();

#pragma unroll
  for (int kb = 0; kb < kKBlocksPerCol; ++kb) {
    const uint8_t* unit = blocks + kb * kUnitBytes;
    const aie::vector<float, 8> scales = load_scales(unit);
    const auto hi_bf =
        aie::to_vector<bfloat16>(aie::accum<accfloat, 8>(scales));
    const auto hi_f = aie::accum<accfloat, 8>(hi_bf).template to_vector<float>();
    const auto lo_bf = aie::to_vector<bfloat16>(
        aie::accum<accfloat, 8>(aie::sub(scales, hi_f)));
    const auto hi_rows = aie::concat(hi_bf, hi_bf, hi_bf, hi_bf);
    const auto lo_rows = aie::concat(lo_bf, lo_bf, lo_bf, lo_bf);

#pragma unroll
    for (int payload = 0; payload < kPayloadsPerKBlock; ++payload) {
      const auto q = load_payload(unit, payload);
      const auto qbf = aie::to_float<bfloat16>(q, 0);
      auto acc = aie::mul(qbf, hi_rows);
      acc = aie::mac(acc, qbf, lo_rows);
      const auto vals = acc.template to_vector<bfloat16>();

      // Payloads are 4 K rows; two payloads make one 8x8 mmul B subtile.
      const int subtile = kb * 4 + payload / 2;
      const int half = payload % 2;
      aie::store_v(out + subtile * (kMmulK * kCols) +
                       half * (kPayloadRows * kCols),
                   vals);
    }
  }

  event1();
}
