#pragma once

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "quants/common/scale.aie.h"
#include "quants/common/tile.h"
#include "quants/q6_k/layout.h"

namespace spiral::quants::q6_k {

namespace detail {

// Load the per-column fp16 `d` values once per record and expand them to 32
// lanes. The product of `d` and an 8-bit subgroup scale needs up to 18
// significand bits, so the product is split into three bf16 terms: two terms
// are not always enough to round to the same bf16 as an fp32 multiply.
static inline
    __attribute__((always_inline)) aie::vector<float, kPackedChunkScalesBytes>
    load_d32(const uint8_t* record) {
  const auto d8 = load_fp16x8(record_scales(record));
  return aie::concat(d8, d8, d8, d8);
}

static inline __attribute__((always_inline)) void load_chunk_scales32(
    const aie::vector<float, kPackedChunkScalesBytes>& d32,
    const uint8_t* packed_scales,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& hi,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& mid,
    aie::vector<bfloat16, kPackedChunkScalesBytes>& lo) {
  const auto subgroup_scales = aie::to_float<float>(
      aie::load_unaligned_v<kPackedChunkScalesBytes>(packed_scales, 16)
          .template cast_to<int8>());
  const auto product =
      aie::mul(d32, subgroup_scales).template to_vector<float>();

  hi = aie::to_vector<bfloat16>(
      aie::accum<accfloat, kPackedChunkScalesBytes>(product));
  const auto hi_f32 =
      aie::accum<accfloat, kPackedChunkScalesBytes>(hi).to_vector<float>();
  const auto residual1 = aie::sub(product, hi_f32);

  mid = aie::to_vector<bfloat16>(
      aie::accum<accfloat, kPackedChunkScalesBytes>(residual1));
  const auto mid_f32 =
      aie::accum<accfloat, kPackedChunkScalesBytes>(mid).to_vector<float>();
  const auto residual2 = aie::sub(residual1, mid_f32);

  lo = aie::to_vector<bfloat16>(
      aie::accum<accfloat, kPackedChunkScalesBytes>(residual2));
}

// Expand one subgroup's eight per-column scale terms to 64 lanes.
static inline __attribute__((always_inline)) void load_subgroup_scales64(
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& hi32,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& mid32,
    const aie::vector<bfloat16, kPackedChunkScalesBytes>& lo32,
    int subgroup,
    aie::vector<bfloat16, kMmulWeightsElems>& hi,
    aie::vector<bfloat16, kMmulWeightsElems>& mid,
    aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  hi = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(hi32,
                                                                     subgroup);
  mid = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(mid32,
                                                                      subgroup);
  lo = aie::broadcast_vector<kTileCols, bfloat16, kMmulWeightsElems>(lo32,
                                                                     subgroup);
}

// Decode one 8-row x 8-column tile from the native six-bit payload. The low
// nibble plane is reinterpreted as uint4 lanes; the high-two-bit plane is
// expanded through 16-bit shifts and interleaved back to 64 byte lanes. The
// decoded values then follow the shared bf16 scale/dequant path.
static inline
    __attribute__((always_inline)) aie::vector<bfloat16, kMmulWeightsElems>
    dequantize_tile(const uint8_t* packed_chunk,
                    int weights_rows_tile,
                    const aie::vector<bfloat16, kMmulWeightsElems>& hi,
                    const aie::vector<bfloat16, kMmulWeightsElems>& mid,
                    const aie::vector<bfloat16, kMmulWeightsElems>& lo) {
  const uint8_t* packed_low = packed_chunk;
  const uint8_t* packed_high = packed_low + kPackedChunkLowBytes;
  const int low_offset = weights_rows_tile * (kTileRows * kTileCols / 2);
  const int high_offset = weights_rows_tile * (kTileRows * kTileCols / 4);
  const uint8_t* low = packed_low + low_offset;
  const uint8_t* high = packed_high + high_offset;

  const auto low_bytes =
      aie::load_unaligned_v<kTileRows * kTileCols / 2>(low, 16);
  const auto low_u8 =
      low_bytes.template cast_to<uint4>().template unpack<uint8>();
  const auto high_bytes = aie::load_v<kTileRows * kTileCols / 4>(high);
  const auto high8 =
      high_bytes.template cast_to<uint4>().template unpack<uint8>();
  const auto high16 = high8.template unpack<uint16>();
  const auto mask8 = aie::broadcast<uint8_t, kTileRows * kTileCols / 2>(
      static_cast<uint8_t>(0x03));
  const auto mask16 = aie::broadcast<uint16_t, kTileRows * kTileCols / 2>(
      static_cast<uint16_t>(0x03));

  const auto h_even = aie::bit_and(high8, mask8);
  const auto h_odd_16 = aie::bit_and(aie::logical_downshift(high16, 2), mask16);
  const auto h_odd = h_odd_16.template pack<uint8>();
  const auto [c0, c1] = aie::interleave_zip(h_even, h_odd, 1);
  const auto high_u8 = aie::concat(c0, c1);
  const auto values_u8 = aie::bit_or(low_u8, aie::upshift(high_u8, 4));
  const auto centered = aie::add(
      values_u8,
      aie::broadcast<uint8_t, kMmulWeightsElems>(static_cast<uint8_t>(0xE0)));
  const auto quant = centered.template cast_to<int8>();
  const auto values = aie::to_float<bfloat16>(quant);
  auto accumulator = aie::mul(values, hi);
  accumulator = aie::mac(accumulator, values, mid);
  accumulator = aie::mac(accumulator, values, lo);
  return accumulator.template to_vector<bfloat16>();
}

}  // namespace detail

// Unpack one 64-row x 8-column chunk into the canonical bf16 weights tile.
inline void unpack_weights_chunk(const uint8_t* record,
                                 const uint8_t* __restrict packed_chunk,
                                 bfloat16* __restrict weights_tile) {
  using OutputVec = aie::vector<bfloat16, kMmulWeightsElems>;
  auto output_tiles = reinterpret_cast<OutputVec*>(weights_tile);
  constexpr int kTilesPerChunk = kUnpackTileRows / kTileRows;
  constexpr int kTilesPerSubgroup = 16 / kTileRows;

  aie::vector<bfloat16, kPackedChunkScalesBytes> hi32;
  aie::vector<bfloat16, kPackedChunkScalesBytes> mid32;
  aie::vector<bfloat16, kPackedChunkScalesBytes> lo32;
  const int chunk_index =
      static_cast<int>((packed_chunk - record) / kPackedChunkBytes);
  const auto d32 = detail::load_d32(record);
  detail::load_chunk_scales32(d32, chunk_scales(record, chunk_index), hi32,
                              mid32, lo32);

#pragma unroll
  for (int subgroup = 0; subgroup < kTilesPerChunk / kTilesPerSubgroup;
       ++subgroup) {
    aie::vector<bfloat16, kMmulWeightsElems> hi;
    aie::vector<bfloat16, kMmulWeightsElems> mid;
    aie::vector<bfloat16, kMmulWeightsElems> lo;
    detail::load_subgroup_scales64(hi32, mid32, lo32, subgroup, hi, mid, lo);

    static_assert(kTilesPerSubgroup == 2,
                  "Q6_K pair decode expects two tiles per subgroup");
    const int tile0 = subgroup * kTilesPerSubgroup;
    const auto values0 =
        detail::dequantize_tile(packed_chunk, tile0, hi, mid, lo);
    const auto values1 =
        detail::dequantize_tile(packed_chunk, tile0 + 1, hi, mid, lo);
    output_tiles[tile0] = values0;
    output_tiles[tile0 + 1] = values1;
  }
}

}  // namespace spiral::quants::q6_k
