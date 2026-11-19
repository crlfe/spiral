#!/usr/bin/env python3
"""float16 scale split and application probes.

These start from a float16 scale and integer weights. The two-term probes
produce a bf16 hi/lo pair from one float16 value; the three-term and integer
probes cover the Q6_K-style scale preparation. The int8_scale probes apply a
float16-derived scale to int8 data with the same two-term fallback.

All probes except the 64-value application cases process float16 scales per
iteration; `values_per_iter` records the count. Unsuffixed probe names are the
default 32-wide batches, while `_x8` and `_x16` are retained historical widths.
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import bench_common
from bench_common import IN_BYTES, ITERS, Probe, _seed_input
from float16_convert import check_f32, make_float16_bits

BENCH_ROOT = Path(__file__).resolve().parent
SOURCE = BENCH_ROOT / "float16_scale.aie.cc"

Q6K_STRIDE = 128


def make_split_biased(seed):
    """Canonical float16 values pre-multiplied by 2^10 on the host."""
    _, expected = make_float16_bits(seed)
    biased = (expected * 1024.0).astype(np.float16)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: biased.nbytes] = biased.view(np.uint8)
    hi = expected.astype(bfloat16)
    lo = (expected - hi.astype(np.float32)).astype(bfloat16)
    paired = np.stack(
        [hi.reshape(ITERS, 8), lo.reshape(ITERS, 8)], axis=1
    ).reshape(-1)
    return data, paired


def make_split(seed):
    data, expected = make_float16_bits(seed)
    hi = expected.astype(bfloat16)
    lo = (expected - hi.astype(np.float32)).astype(bfloat16)
    paired = np.stack(
        [hi.reshape(ITERS, 8), lo.reshape(ITERS, 8)], axis=1
    ).reshape(-1)
    return data, paired


def check_split(out, expected):
    used = expected.nbytes
    actual = out[:used].view(bfloat16)
    error = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return float(error.max()), "hi/lo pair"


def check_split_sum(out, expected):
    used = expected.size
    paired = out[: used * 4].view(bfloat16).reshape(ITERS, 2, 8)
    total = paired[:, 0, :].astype(np.float32) + paired[:, 1, :].astype(
        np.float32
    )
    error = np.abs(total.reshape(-1) - expected.astype(np.float32))
    return float(error.max()), "hi+lo"


# Host-packed 16-value encoding for the branch-free two-term split. Each float16
# is normalized to a bf16 exponent and a nine-bit significand fraction, then
# packed into the same two bytes as:
#   bit 15      sign
#   bits 14:9   exponent, biased by 102 (0 marks zero)
#   bits 8:2    top seven fraction bits (the hi term)
#   bits 1:0    residual code {0, 1.5, 2, 3} in 1/512 units (the lo term)
# The residual collapses the low fraction bits so the device never has to
# renormalize them: lo's exponent delta and mantissa bit are already in the
# code. Both normal and subnormal float16 values use the same path.
PACK16_Q = np.array([0.0, 1.5, 2.0, 3.0])


def encode_float16x16(values):
    values = values.astype(np.float16)
    bits = values.view(np.uint16).astype(np.uint32)
    sign = bits & 0x8000
    magnitude = np.abs(values.astype(np.float32))
    nonzero = magnitude > 0
    mantissa, exponent = np.frexp(
        np.where(nonzero, magnitude, 1.0).astype(np.float64)
    )
    bf16_exponent = np.where(nonzero, exponent - 1 + 127, 0)
    fraction = np.where(nonzero, (mantissa * 2.0 - 1.0) * 512.0, 0.0)

    hi_fraction = np.floor(fraction / 4.0).astype(np.int64)
    residual = fraction - 4.0 * hi_fraction
    code = np.argmin(
        np.abs(residual[..., None] - PACK16_Q[None, :]), axis=-1
    )

    packed_exponent = np.where(nonzero, bf16_exponent - 102, 0)
    packed = (
        sign
        | ((packed_exponent.astype(np.uint32) & 0x3F) << 9)
        | ((hi_fraction.astype(np.uint32) & 0x7F) << 2)
        | code.astype(np.uint32)
    ).astype(np.uint16)
    return packed.reshape(-1)


def split_packed16(packed):
    """Device model: packed code -> exact expected hi/lo bf16 bit patterns."""
    packed = packed.astype(np.uint16)
    code = packed & 3
    sign = packed & 0x8000
    fraction = (packed & 0x7FFF) >> 2
    nonzero = (packed & 0x7E00) != 0
    hi = ((fraction.astype(np.uint32) + (102 << 7)) | sign).astype(np.uint16)
    hi = np.where(nonzero, hi, 0).astype(np.uint16)
    lo = (
        (fraction & 0x7F80).astype(np.int32)
        + ((102 - 9) << 7)
        + (code.astype(np.int32) << 6)
        + sign.astype(np.int32)
    ).astype(np.uint16)
    lo = np.where(nonzero & (code != 0), lo, 0).astype(np.uint16)
    return hi, lo


def make_packed16_split(seed):
    rng = _seed_input(seed)
    scale = rng.uniform(0.5, 1.5, size=ITERS * 16).astype(np.float32)
    magnitude = np.exp2(rng.uniform(-24, 15, size=ITERS * 16))
    values = (rng.choice([-1.0, 1.0], size=ITERS * 16) * magnitude * scale)
    values = values.astype(np.float16)
    values[::97] = 0.0
    packed = encode_float16x16(values)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: packed.nbytes] = packed.view(np.uint8)
    hi, lo = split_packed16(packed)
    expected = {
        "hi": hi.reshape(ITERS, 16),
        "lo": lo.reshape(ITERS, 16),
        "values": values.reshape(ITERS, 16).astype(np.float32),
    }
    return data, expected


def check_packed16_split(out, expected):
    used = ITERS * 32 * 2
    pairs = out[:used].view(bfloat16).reshape(ITERS, 32)
    actual_hi = pairs[:, :16]
    actual_lo = pairs[:, 16:]
    exact = np.array_equal(actual_hi.view(np.uint16), expected["hi"]) and (
        np.array_equal(actual_lo.view(np.uint16), expected["lo"])
    )
    total = actual_hi.astype(np.float32) + actual_lo.astype(np.float32)
    reference = expected["values"]
    error = np.abs(total - reference)
    relative = error / np.maximum(np.abs(reference), np.finfo(np.float32).tiny)
    note = "packed16 hi/lo rel" if exact else "packed16 hi/lo mismatch"
    return float(relative.max()), note


def _bf16_int_bits(value):
    """Bit pattern of a non-negative integer 0..255 as bf16."""
    value = value.astype(np.int32)
    nonzero = value > 0
    leading = np.zeros(value.shape, dtype=np.int32)
    for bit in range(8):
        leading = np.where((value >> bit) & 1, bit, leading)
    mantissa = value - (1 << leading)
    mantissa = np.where(nonzero, mantissa << (7 - leading), 0)
    bits = np.where(
        nonzero,
        ((127 + leading).astype(np.uint16) << 7) | mantissa.astype(np.uint16),
        0,
    )
    return bits.astype(np.uint16)


def _expadd_split(bits, denorm):
    """Device model for the exponent-add split; bits are canonical float16."""
    bits = bits.astype(np.uint16)
    sign = (bits & 0x8000).astype(np.uint16)
    exponent = ((bits >> 10) & 0x1F).astype(np.int32)
    top = ((bits >> 8) & 3).astype(np.uint16)
    low = (bits & 0xFF).astype(np.uint16)
    is_zero_exp = exponent == 0

    hi_bits = (
        sign
        | ((exponent + 112).astype(np.uint16) << 7)
        | (top << 5)
    )
    if denorm:
        hi_sub = (
            sign
            | ((0x3780 + ((top & 2) << 6)).astype(np.uint16))
            | ((top == 3).astype(np.uint16) << 6)
        )
        hi_sub = np.where(top != 0, hi_sub, 0).astype(np.uint16)
        hi_bits = np.where(is_zero_exp, hi_sub, hi_bits)
    else:
        hi_bits = np.where(is_zero_exp, 0, hi_bits)

    delta = exponent - 25
    if denorm:
        delta = np.where(is_zero_exp, -24, delta)
    low_bf16 = _bf16_int_bits(low)
    lo_bits = (low_bf16.astype(np.int32) + (delta << 7)).astype(np.uint16)
    lo_bits = (lo_bits | sign).astype(np.uint16)
    lo_bits = np.where(low != 0, lo_bits, 0).astype(np.uint16)
    if not denorm:
        lo_bits = np.where(is_zero_exp, 0, lo_bits).astype(np.uint16)
    return hi_bits.astype(np.uint16), lo_bits.astype(np.uint16)


def make_expadd_split(seed, denorm):
    rng = _seed_input(seed)
    # Finite float16 patterns, deliberately including E == 0 (zero and denorms).
    sign = rng.integers(0, 2, size=ITERS * 16, dtype=np.uint16)
    exponent = rng.integers(0, 31, size=ITERS * 16, dtype=np.uint16)
    mantissa = rng.integers(0, 1024, size=ITERS * 16, dtype=np.uint16)
    bits = (sign << 15) | (exponent << 10) | mantissa
    # Force a run of exact zeros and a run of denorms.
    bits[0:16] = 0
    bits[16:48] &= 0x83FF
    values = bits.view(np.float16)
    hi, lo = _expadd_split(bits, denorm)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: values.nbytes] = values.view(np.uint8)
    expected = {
        "hi": hi.reshape(ITERS, 16),
        "lo": lo.reshape(ITERS, 16),
        "values": values.reshape(ITERS, 16).astype(np.float32),
    }
    return data, expected


def make_expadd_split_flush(seed):
    return make_expadd_split(seed, False)


def make_expadd_split_denorm(seed):
    return make_expadd_split(seed, True)


def check_expadd_split(out, expected, denorm):
    used = ITERS * 32 * 2
    pairs = out[:used].view(bfloat16).reshape(ITERS, 32)
    actual_hi = pairs[:, :16]
    actual_lo = pairs[:, 16:]
    exact = np.array_equal(actual_hi.view(np.uint16), expected["hi"]) and (
        np.array_equal(actual_lo.view(np.uint16), expected["lo"])
    )
    total = actual_hi.astype(np.float32) + actual_lo.astype(np.float32)
    error = np.abs(total - expected["values"])
    note = (
        "expadd denorm hi/lo"
        if denorm and exact
        else "expadd flush hi/lo"
        if exact
        else "expadd hi/lo mismatch"
    )
    return float(error.max()), note


def _make_float16_values(seed, count):
    rng = _seed_input(seed)
    return rng.uniform(-8.0, 8.0, size=ITERS * count).astype(np.float16)


def _make_split32(seed, biased):
    values = _make_float16_values(seed, 32)
    expected = values.astype(np.float32)
    stored = (expected * 1024.0).astype(np.float16) if biased else values
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: stored.nbytes] = stored.view(np.uint8)
    hi = expected.astype(bfloat16)
    lo = (expected - hi.astype(np.float32)).astype(bfloat16)
    paired = np.stack(
        [hi.reshape(ITERS, 32), lo.reshape(ITERS, 32)], axis=1
    ).reshape(-1)
    return data, paired


def make_split32(seed):
    return _make_split32(seed, False)


def make_split_biased32(seed):
    return _make_split32(seed, True)


def make_bits_split32(seed):
    return _make_split32(seed, False)


def make_direct_split32(seed):
    values = _make_float16_values(seed, 32)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: values.nbytes] = values.view(np.uint8)
    return data, values.astype(np.float32)


def check_split_sum32(out, expected):
    used = expected.size
    paired = out[: used * 4].view(bfloat16).reshape(ITERS, 2, 32)
    total = paired[:, 0, :].astype(np.float32) + paired[:, 1, :].astype(
        np.float32
    )
    error = np.abs(total.reshape(-1) - expected.astype(np.float32))
    return float(error.max()), "hi+lo"


def make_packed16_split32(seed):
    values = _make_float16_values(seed, 32)
    values[::193] = 0.0
    packed = encode_float16x16(values)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: packed.nbytes] = packed.view(np.uint8)
    hi, lo = split_packed16(packed)
    expected = {
        "hi": hi.reshape(ITERS, 32),
        "lo": lo.reshape(ITERS, 32),
        "values": values.reshape(ITERS, 32).astype(np.float32),
    }
    return data, expected


def check_packed16_split32(out, expected):
    used = ITERS * 64 * 2
    pairs = out[:used].view(bfloat16).reshape(ITERS, 64)
    actual_hi = pairs[:, :32]
    actual_lo = pairs[:, 32:]
    exact = np.array_equal(actual_hi.view(np.uint16), expected["hi"]) and (
        np.array_equal(actual_lo.view(np.uint16), expected["lo"])
    )
    total = actual_hi.astype(np.float32) + actual_lo.astype(np.float32)
    reference = expected["values"]
    error = np.abs(total - reference)
    relative = error / np.maximum(np.abs(reference), np.finfo(np.float32).tiny)
    note = "packed16 rel" if exact else "packed16 mismatch"
    return float(relative.max()), note


def make_expadd_split32(seed, denorm):
    rng = _seed_input(seed)
    sign = rng.integers(0, 2, size=ITERS * 32, dtype=np.uint16)
    exponent = rng.integers(0, 31, size=ITERS * 32, dtype=np.uint16)
    mantissa = rng.integers(0, 1024, size=ITERS * 32, dtype=np.uint16)
    bits = (sign << 15) | (exponent << 10) | mantissa
    bits[0:32] = 0
    bits[32:96] &= 0x83FF
    values = bits.view(np.float16)
    hi, lo = _expadd_split(bits, denorm)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[: values.nbytes] = values.view(np.uint8)
    expected = {
        "hi": hi.reshape(ITERS, 32),
        "lo": lo.reshape(ITERS, 32),
        "values": values.reshape(ITERS, 32).astype(np.float32),
    }
    return data, expected


def _check_expadd_split32(out, expected, denorm):
    used = ITERS * 64 * 2
    pairs = out[:used].view(bfloat16).reshape(ITERS, 64)
    actual_hi = pairs[:, :32]
    actual_lo = pairs[:, 32:]
    exact = np.array_equal(actual_hi.view(np.uint16), expected["hi"]) and (
        np.array_equal(actual_lo.view(np.uint16), expected["lo"])
    )
    total = actual_hi.astype(np.float32) + actual_lo.astype(np.float32)
    error = np.abs(total - expected["values"])
    note = (
        "expadd denorm"
        if denorm and exact
        else "expadd flush"
        if exact
        else "expadd mismatch"
    )
    return float(error.max()), note


def make_expadd_split_flush32(seed):
    return make_expadd_split32(seed, False)


def make_expadd_split_denorm32(seed):
    return make_expadd_split32(seed, True)


def check_expadd_split_flush32(out, expected):
    return _check_expadd_split32(out, expected, False)


def check_expadd_split_denorm32(out, expected):
    return _check_expadd_split32(out, expected, True)


def make_bf16_scale(seed):
    rng = _seed_input(seed)
    scale = np.array([rng.uniform(0.01, 0.2)], dtype=bfloat16)
    q = rng.integers(-128, 128, size=ITERS * 64, dtype=np.int16).astype(np.int8)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[:2] = scale.view(np.uint16).view(np.uint8)
    data[64 : 64 + q.nbytes] = q.view(np.uint8)
    expected = (q.astype(np.float32) * float(scale[0])).astype(bfloat16)
    return data, expected


def check_bf16(out, expected):
    used = expected.nbytes
    actual = out[:used].view(bfloat16)
    error = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return float(error.max()), "single bf16 scale"


def make_f32_scale(seed):
    rng = _seed_input(seed)
    scale = np.array([rng.uniform(0.01, 0.2)], dtype=np.float32)
    q = rng.integers(-128, 128, size=ITERS * 64, dtype=np.int16).astype(np.int8)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[:4] = scale.view(np.uint8)
    data[64 : 64 + q.nbytes] = q.view(np.uint8)
    expected = q.astype(np.float32) * float(scale[0])
    return data, expected


def make_float16_scale(seed):
    rng = _seed_input(seed)
    scale = np.array([rng.uniform(0.01, 0.2)], dtype=np.float16)
    q = rng.integers(-128, 128, size=ITERS * 64, dtype=np.int16).astype(np.int8)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    data[:2] = scale.view(np.uint16).view(np.uint8)
    data[64 : 64 + q.nbytes] = q.view(np.uint8)
    hi = scale.astype(bfloat16)
    lo = (scale.astype(np.float32) - hi.astype(np.float32)).astype(bfloat16)
    expected = (
        q.astype(np.float32) * hi.astype(np.float32)
        + q.astype(np.float32) * lo.astype(np.float32)
    ).astype(bfloat16)
    return data, expected


def _float16_scale_bits(data):
    return int(data[:2].view(np.uint16)[0])


def make_float16_int_mul(seed):
    data, _ = make_float16_scale(seed)
    bits = _float16_scale_bits(data)
    mantissa = (bits & 0x03FF) | 0x0400
    q = data[64 : 64 + ITERS * 64].view(np.int8)
    expected = q.astype(np.int32) * mantissa
    return data, expected


def make_float16_int_round(seed):
    data, expected_int = make_float16_int_mul(seed)
    expected = expected_int.astype(np.float32).astype(bfloat16)
    return data, expected


def check_int32(out, expected):
    used = expected.nbytes
    actual = out[:used].view(np.int32)
    error = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
    return float(error.max()), "int32 product"


def _make_q6k_records(seed):
    rng = _seed_input(seed)
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    scales = []
    payloads = []
    for i in range(ITERS):
        rec = i * Q6K_STRIDE
        d = rng.uniform(0.005, 0.25, 8).astype(np.float16)
        s = rng.integers(-127, 128, 8).astype(np.int8)
        s[s == 0] = 1
        q = rng.integers(-128, 128, 64, dtype=np.int16).astype(np.int8)
        data[rec : rec + 16] = d.view(np.uint16).view(np.uint8)
        data[rec + 16 : rec + 24] = s.view(np.uint8)
        data[rec + 64 : rec + 64 + 64] = q.view(np.uint8)
        scales.append((d, s))
        payloads.append(q)
    return data, scales, payloads


def _q6k_expected(d, s, q):
    scale = d.astype(np.float32) * s.astype(np.float32)
    return (
        q.reshape(8, 8).astype(np.float32) * scale[None, :]
    ).reshape(-1).astype(bfloat16)


def make_q6k_full(seed):
    data, scales, payloads = _make_q6k_records(seed)
    expected = np.concatenate(
        [
            _q6k_expected(d, s, q)
            for (d, s), q in zip(scales, payloads)
        ]
    )
    return data, expected


def make_q6k_fixed(seed):
    data, scales, payloads = _make_q6k_records(seed)
    d, s = scales[0]
    expected = np.concatenate([_q6k_expected(d, s, q) for q in payloads])
    return data, expected


def make_q6k_int_fixed(seed):
    data, scales, payloads = _make_q6k_records(seed)
    d, s = scales[0]
    bits = d.view(np.uint16)
    exponent = (bits & 0x7C00) >> 10
    mantissa = ((bits & 0x03FF) | 0x0400).astype(np.int64)
    meff = mantissa * s.astype(np.int64)
    mhi = (meff >> 8).astype(np.int16)
    mlo = (meff & 0xFF).astype(np.int16)
    factor_bits = ((exponent.astype(np.uint16) + 102) << 7).astype(np.uint16)
    data[0:16] = mhi.view(np.uint8)
    data[16:32] = mlo.view(np.uint8)
    data[32:48] = factor_bits.view(np.uint8)
    expected = np.concatenate([_q6k_expected(d, s, q) for q in payloads])
    return data, expected



def _split3_terms(product):
    hi = product.astype(bfloat16)
    r1 = product - hi.astype(np.float32)
    mid = r1.astype(bfloat16)
    r2 = r1 - mid.astype(np.float32)
    return hi, mid, r2.astype(bfloat16)


def make_q6k_prep(seed, count):
    rng = _seed_input(seed)
    d = rng.uniform(0.005, 0.25, size=ITERS * 8).astype(np.float16)
    scales = rng.integers(-127, 128, size=ITERS * count, dtype=np.int16).astype(
        np.int8
    )
    scales[scales == 0] = 1
    stride = 64 if count == 32 else 32
    data = np.zeros(IN_BYTES, dtype=np.uint8)
    for i in range(ITERS):
        rec = i * stride
        data[rec : rec + 16] = d[i * 8 : (i + 1) * 8].view(np.uint8)
        offset = 32 if count == 32 else 16
        data[rec + offset : rec + offset + count] = scales[
            i * count : (i + 1) * count
        ].view(np.uint8)
    repeats = count // 8
    d32 = np.tile(d.reshape(ITERS, 8).astype(np.float32), (1, repeats))
    product = d32 * scales.reshape(ITERS, count).astype(np.float32)
    return data, product


def make_q6k_prep32(seed):
    return make_q6k_prep(seed, 32)


def make_q6k_prep8(seed):
    return make_q6k_prep(seed, 8)


def _check_q6k_prep(out, expected, count):
    terms = out[: ITERS * 3 * count * 2].view(bfloat16).reshape(
        ITERS, 3, count
    )
    total = (
        terms[:, 0, :].astype(np.float32)
        + terms[:, 1, :].astype(np.float32)
        + terms[:, 2, :].astype(np.float32)
    )
    error = np.abs(total - expected)
    note = f"q6k prep x{count}"
    return float(error.max()), note


def check_q6k_prep32(out, expected):
    return _check_q6k_prep(out, expected, 32)


def check_q6k_prep8(out, expected):
    return _check_q6k_prep(out, expected, 8)


PROBES = [
    Probe(
        "float16_bits_split_bf16_x8",
        "FLOAT16_BITS_SPLIT_BF16_X8",
        True,
        make_split,
        check_split,
    ),
    Probe(
        "float16_shift_split_bf16_x8",
        "FLOAT16_SHIFT_SPLIT_BF16_X8",
        True,
        make_split,
        check_split,
    ),
    Probe(
        "float16_biased_shift_split_bf16_x8",
        "FLOAT16_BIASED_SHIFT_SPLIT_BF16_X8",
        True,
        make_split_biased,
        check_split,
    ),
    Probe(
        "float16_direct_split_bf16_x8",
        "FLOAT16_DIRECT_SPLIT_BF16_X8",
        True,
        make_float16_bits,
        check_split_sum,
    ),
    Probe(
        "float16_packed16_split_bf16_x16",
        "FLOAT16_PACKED16_SPLIT_BF16_X16",
        True,
        make_packed16_split,
        check_packed16_split,
        values_per_iter=16,
    ),
    Probe(
        "float16_expadd_split_bf16_flush_x16",
        "FLOAT16_EXPADD_SPLIT_BF16_FLUSH_X16",
        True,
        make_expadd_split_flush,
        lambda out, expected: check_expadd_split(out, expected, False),
        values_per_iter=16,
    ),
    Probe(
        "float16_expadd_split_bf16_denorm_x16",
        "FLOAT16_EXPADD_SPLIT_BF16_DENORM_X16",
        True,
        make_expadd_split_denorm,
        lambda out, expected: check_expadd_split(out, expected, True),
        values_per_iter=16,
    ),
    Probe(
        "float16_bits_split_bf16",
        "FLOAT16_BITS_SPLIT_BF16",
        True,
        make_bits_split32,
        check_split,
        values_per_iter=32,
    ),
    Probe(
        "float16_shift_split_bf16",
        "FLOAT16_SHIFT_SPLIT_BF16",
        True,
        make_split32,
        check_split,
        values_per_iter=32,
    ),
    Probe(
        "float16_biased_shift_split_bf16",
        "FLOAT16_BIASED_SHIFT_SPLIT_BF16",
        True,
        make_split_biased32,
        check_split,
        values_per_iter=32,
    ),
    Probe(
        "float16_direct_split_bf16",
        "FLOAT16_DIRECT_SPLIT_BF16",
        True,
        make_direct_split32,
        check_split_sum32,
        values_per_iter=32,
    ),
    Probe(
        "float16_packed16_split_bf16",
        "FLOAT16_PACKED16_SPLIT_BF16",
        True,
        make_packed16_split32,
        check_packed16_split32,
        values_per_iter=32,
    ),
    Probe(
        "float16_expadd_split_bf16_flush",
        "FLOAT16_EXPADD_SPLIT_BF16_FLUSH",
        True,
        make_expadd_split_flush32,
        check_expadd_split_flush32,
        values_per_iter=32,
    ),
    Probe(
        "float16_expadd_split_bf16_denorm",
        "FLOAT16_EXPADD_SPLIT_BF16_DENORM",
        True,
        make_expadd_split_denorm32,
        check_expadd_split_denorm32,
        values_per_iter=32,
    ),
    Probe(
        "int8_scale_f32",
        "INT8_SCALE_F32",
        True,
        make_f32_scale,
        check_f32,
        values_per_iter=64,
    ),
    Probe(
        "int8_scale_bf16",
        "INT8_SCALE_BF16",
        True,
        make_bf16_scale,
        check_bf16,
        values_per_iter=64,
    ),
    Probe(
        "int8_scale_bf16_2term",
        "INT8_SCALE_BF16_2TERM",
        True,
        make_float16_scale,
        check_bf16,
        values_per_iter=64,
    ),
    Probe(
        "int8_scale_bf16_accum",
        "INT8_SCALE_BF16_ACCUM",
        True,
        make_float16_scale,
        check_bf16,
        values_per_iter=64,
    ),
    Probe(
        "int8_scale_int_split",
        "INT8_SCALE_INT_SPLIT",
        True,
        make_float16_scale,
        check_bf16,
        values_per_iter=64,
    ),
    Probe(
        "int8_scale_int_mul",
        "INT8_SCALE_INT_MUL",
        True,
        make_float16_int_mul,
        check_int32,
        values_per_iter=64,
    ),
    Probe(
        "int8_scale_int_round",
        "INT8_SCALE_INT_ROUND",
        True,
        make_float16_int_round,
        check_bf16,
        values_per_iter=64,
    ),
    Probe(
        "q6k_3term_prep",
        "Q6K_3TERM_PREP",
        True,
        make_q6k_prep32,
        check_q6k_prep32,
        values_per_iter=32,
    ),
    Probe(
        "q6k_3term_prep_x8",
        "Q6K_3TERM_PREP_X8",
        True,
        make_q6k_prep8,
        check_q6k_prep8,
        values_per_iter=8,
    ),
    Probe(
        "q6k_3term_apply_x8",
        "Q6K_3TERM_APPLY_X8",
        True,
        make_q6k_fixed,
        check_bf16,
        values_per_iter=64,
    ),
    Probe(
        "q6k_3term_full_x8",
        "Q6K_3TERM_FULL_X8",
        True,
        make_q6k_full,
        check_bf16,
        values_per_iter=64,
    ),
    Probe(
        "q6k_int_apply",
        "Q6K_INT_APPLY",
        True,
        make_q6k_int_fixed,
        check_bf16,
        values_per_iter=64,
    ),
    Probe(
        "q6k_int_full",
        "Q6K_INT_FULL",
        True,
        make_q6k_full,
        check_bf16,
        values_per_iter=64,
    ),
]


if __name__ == "__main__":
    sys.exit(bench_common.run(__doc__, SOURCE, PROBES))
