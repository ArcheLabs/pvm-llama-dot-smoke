// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs
use crate::{
    consts::PVM_DOT_Q8_0_VALUES,
    primitives::{Q8Block, Q8Input},
};

// Standard 64-bit golden-ratio increment used by SplitMix64 / SplittableRandom.
//
// Reference: https://gee.cs.oswego.edu/dl/papers/oopsla14.pdf
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

// Standard 64-bit FNV-1a offset basis used by the FNV hash algorithm.
//
// Reference: https://www.ietf.org/archive/id/draft-eastlake-fnv-21.html
const FNV64_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

// https://gee.cs.oswego.edu/dl/papers/oopsla14.pdf
const MIX64_VARIANT13_MUL1: u64 = 0xbf58_476d_1ce4_e5b9;
const MIX64_VARIANT13_MUL2: u64 = 0x94d0_49bb_1331_11eb;

/// A simple method to compare two floating-point numbers for approximate equality
pub fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-6
}

/// Map an input prompt into a deterministic fixed-length vector.
pub fn prompt_to_vec(prompt: &str) -> Q8Input {
    let mut seed = fnv1a64(prompt.as_bytes());

    if prompt.is_empty() {
        seed ^= GOLDEN_GAMMA;
    }

    let mut vec = [0.0f32; PVM_DOT_Q8_0_VALUES];

    for (i, v) in vec.iter_mut().enumerate() {
        let seed = splitmix64(seed.wrapping_add(i as u64));
        let u = ((seed >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        *v = u * 2.0 - 1.0;
    }

    vec.into()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV64_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    hash
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(GOLDEN_GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(MIX64_VARIANT13_MUL1);
    z = (z ^ (z >> 27)).wrapping_mul(MIX64_VARIANT13_MUL2);
    z ^ (z >> 31)
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
