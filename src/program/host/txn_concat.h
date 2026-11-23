#pragma once

// Host-side TXN concatenation and patch rewriting for the fixed Spiral
// program.  The compiler-generated body stays the source of truth for BD
// setup, queue assignment, and tile addresses; this header only appends
// bodies and rewrites the (arg_idx, arg_plus) fields of the DDR patches.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace txn_concat {

// Runtime-sequence argument order.  These map to XRT kernel arguments
// 3, 4, 5, 6 respectively.
enum Arg : uint32_t {
  kArgWeights = 0,
  kArgInputs = 1,
  kArgConfig = 2,
  kArgOutputs = 3,
};

constexpr size_t kHeaderWords = 4;
constexpr size_t kPatchOpcode = 129;
constexpr size_t kPatchWords = 12;

// Per-operation patch policy.  Offsets are byte offsets added to the
// compiler-generated arg_plus for that argument.
struct OpPatch {
  uint64_t config_offset = 0;
  uint64_t weight_offset = 0;
  uint64_t output_offset = 0;
  // True for chained operations: retarget input patches to the output
  // argument so the next matmul reads the previous activation in place.
  bool input_from_output = false;
};

// Number of 32-bit words occupied by the operation at `pos`.
inline size_t op_words(const std::vector<uint32_t>& txn, size_t pos) {
  const uint32_t op = txn[pos];
  switch (op) {
    case 0:  // write32
      return 6;
    case 1: {  // blockwrite: size field is at +3, in bytes
      const size_t words = txn[pos + 3] / 4;
      if (words < 4) {
        throw std::runtime_error("malformed blockwrite size");
      }
      return words;
    }
    case 3:  // maskwrite32
    case 4:  // maskpoll32
      return 7;
    case 5:  // noop
    case 6:  // preempt
      return 1;
    case 8:  // loadpdi
      return 4;
    case 128:  // sync / TCT
      return 4;
    case 129:  // DDR patch
      return kPatchWords;
    default:
      throw std::runtime_error("unknown TXN opcode " +
                               std::to_string(static_cast<unsigned>(op)));
  }
}

inline uint64_t get_plus(const std::vector<uint32_t>& txn, size_t patch) {
  return static_cast<uint64_t>(txn[patch + 10]) |
         (static_cast<uint64_t>(txn[patch + 11]) << 32);
}

inline void set_plus(std::vector<uint32_t>& txn, size_t patch, uint64_t value) {
  txn[patch + 10] = static_cast<uint32_t>(value & 0xFFFFFFFFu);
  txn[patch + 11] = static_cast<uint32_t>(value >> 32);
}

// Rewrite the patches in one copied body starting at `body_start`.
// Returns the number of patches touched.
inline int patch_body(std::vector<uint32_t>& dst,
                      size_t body_start,
                      const OpPatch& op) {
  int touched = 0;
  const size_t end = dst.size();
  size_t pos = body_start;
  while (pos < end) {
    const size_t words = op_words(dst, pos);
    if (pos + words > end) {
      throw std::runtime_error("TXN operation runs past the body");
    }
    if (dst[pos] == kPatchOpcode) {
      const uint32_t arg = dst[pos + 8];
      uint64_t plus = get_plus(dst, pos);
      switch (arg) {
        case kArgConfig:
          plus += op.config_offset;
          break;
        case kArgWeights:
          plus += op.weight_offset;
          break;
        case kArgInputs:
          if (op.input_from_output) {
            dst[pos + 8] = kArgOutputs;
          }
          break;
        case kArgOutputs:
          plus += op.output_offset;
          break;
        default:
          throw std::runtime_error("unexpected TXN patch arg " +
                                   std::to_string(arg));
      }
      set_plus(dst, pos, plus);
      ++touched;
    }
    pos += words;
  }
  if (pos != end) {
    throw std::runtime_error("TXN body did not end on an operation boundary");
  }
  return touched;
}

// Concatenate compiler-generated TXNs.  The first TXN supplies the header;
// op_count is summed and byte_length is fixed up.
inline std::vector<uint32_t> concat(
    const std::vector<std::vector<uint32_t>>& bases,
    const std::vector<OpPatch>& ops) {
  if (bases.empty() || bases.size() != ops.size()) {
    throw std::invalid_argument("concat needs one patch policy per base");
  }
  const auto& first = bases.front();
  if (first.size() < kHeaderWords) {
    throw std::invalid_argument("base TXN is too short");
  }
  std::vector<uint32_t> out(first.begin(), first.begin() + kHeaderWords);
  uint32_t op_count = 0;
  for (size_t i = 0; i < bases.size(); ++i) {
    const auto& base = bases[i];
    if (base.size() < kHeaderWords) {
      throw std::invalid_argument("base TXN is too short");
    }
    if (base[0] != first[0] || base[1] != first[1]) {
      throw std::invalid_argument("base TXN headers do not match");
    }
    op_count += base[2];
    const size_t body_start = out.size();
    out.insert(out.end(), base.begin() + kHeaderWords, base.end());
    patch_body(out, body_start, ops[i]);
  }
  out[2] = op_count;
  out[3] = static_cast<uint32_t>(out.size() * sizeof(uint32_t));
  return out;
}

inline std::vector<uint32_t> single(const std::vector<uint32_t>& base,
                                    const OpPatch& op) {
  return concat({base}, {op});
}

}  // namespace txn_concat
