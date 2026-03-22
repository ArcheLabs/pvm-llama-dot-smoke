// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs
use crate::consts::PVM_DOT_Q8_0_VALUES;
use jam_codec::{Decode, Encode};

#[derive(Debug, Default, Clone, Decode, Encode)]
pub struct Q8_0(pub [f32; PVM_DOT_Q8_0_VALUES]);

/// Custom DI01 header.
/// Used as the agreed full input buffer format between the guest and the host.
#[derive(Debug, Clone, Decode, Encode)]
pub struct DotInput {
    /// Protocol identifier.
    pub magic: u32,
    /// Protocol version.
    pub version: u32,
    /// Flags.
    pub flags: u32,
    /// Data provider identifier.
    pub provider: u32,
    /// Lower 32 bits of the block file offset.
    pub file_off_lo: u32,
    /// Upper 32 bits of the block file offset.
    pub file_off_hi: u32,
    /// Block length.
    pub block_len: u32,
    /// Quantization kind.
    /// This is a temporary design choice, fixed to Q8_0 for the smoke test.
    pub quant_kind: u32,
    /// Offset of vector x within the full input buffer.
    pub vec_off: u32,
    /// Vector length, fixed to 32 for the smoke test.
    pub vec_len: u32,
    /// Reserved field, currently unused. Defaults to 0.
    pub reserved0: u32,
    /// Reserved field, currently unused. Defaults to 0.
    pub reserved1: u32,
}

/// DO01 header.
/// Used as the agreed full output buffer format between the guest and the host.
#[derive(Debug, Clone, Decode, Encode)]
pub struct DotOutput {
    /// Protocol identifier.
    pub magic: u32,
    /// Protocol version.
    pub version: u32,
    /// Stage marker indicating how far the guest has progressed.
    pub stage: u32,
    /// Quantization kind.
    /// This is a temporary design choice, fixed to Q8_0 for the smoke test.
    pub quant_kind: u32,
    /// Raw bit representation of the result.
    /// The guest converts the computed result back to f32 through this value.
    pub result_bits: u32,
    /// Vector length, fixed to 32 for the smoke test.
    pub vec_len: u32,
    /// Block length.
    pub block_len: u32,
    /// Reserved field, currently unused. Defaults to 0.
    pub reserved: u32,
}