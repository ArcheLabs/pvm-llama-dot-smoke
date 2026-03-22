// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs
use crate::{
    consts::PVM_DOT_Q8_0_VALUES,
    primitives::{Q8Block, Q8Input},
};

/// A minimal little-endian reader without bounds checks.
pub fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    let bytes: [u8; 4] = buf[off..off + 4].try_into().unwrap();
    u32::from_le_bytes(bytes)
}

/// Build the fixed input vector `x`.
///
/// For smoke-test purposes, this is a simple deterministic test input rather
/// than one loaded from a model file or derived from a real prompt. This helps
/// keep the verification focused on operator-level consistency, avoiding
/// additional sources of variability such as tokenization, embedding lookup,
/// and graph execution, while ensuring the result is fully reproducible across
/// runs.
pub fn fixed_input_vec() -> [f32; PVM_DOT_Q8_0_VALUES] {
    let mut v = [0.0f32; PVM_DOT_Q8_0_VALUES];

    for i in 0..PVM_DOT_Q8_0_VALUES {
        let k = ((i * 37 + 11) % 29) as i32 - 14;
        v[i] = (k as f32) / 8.0;
    }
    v
}

/// Manually convert a 16-bit float (`fp16` / `half`) to `f32` to avoid an
/// external dependency.
///
/// fp16 layout:
/// - 1-bit sign
/// - 5-bit exponent
/// - 10-bit fraction
///
/// Reference:
/// https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Software_implementation
///
/// f32 layout:
/// - 1-bit sign
/// - 8-bit exponent
/// - 23-bit fraction
///
/// Reference:
/// https://en.wikipedia.org/wiki/Single-precision_floating-point_format
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let out_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut mant = frac;
            let mut e: i32 = -14;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            let exp32 = (e + 127) as u32;
            sign | (exp32 << 23) | (mant << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        let exp32 = exp + 112;
        sign | (exp32 << 23) | (frac << 13)
    };

    f32::from_bits(out_bits)
}

/// A simple reference implementation used during the smoke-test stage to
/// validate result consistency.
///
/// It computes the dot product between a `Q8_0` block and the input vector
/// `x`, with the following semantics:
///
/// ```text
/// result = d * sum(q[i] * x[i])
/// ```
pub fn dot_q8_0_reference(block: &Q8Block, x: &Q8Input) -> f32 {
    let d = f16_to_f32(u16::from_le_bytes([block.0[0], block.0[1]]));

    let qs = &block.0[2..2 + PVM_DOT_Q8_0_VALUES];
    let sum: f32 = qs
        .iter()
        .zip(x.0.iter())
        .map(|(&q, &xi)| (q as i8 as f32) * xi)
        .sum();

    d * sum
}