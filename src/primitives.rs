// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs
use crate::consts::{
    PVM_DI_MAGIC, PVM_DOT_PROVIDER_HOST_READ_AT, PVM_DOT_Q8_0_BLOCK_LEN, PVM_DOT_Q8_0_VALUES,
    PVM_DOT_QUANT_Q8_0, PVM_DO_MAGIC, SMOKE_TEST_VERSION,
};
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

const DI01_LEN: u32 = std::mem::size_of::<DotInput>() as u32;

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

const DO01_LEN: u32 = std::mem::size_of::<DotOutput>() as u32;

impl Default for DotInput {
    fn default() -> Self {
        Self {
            magic: PVM_DI_MAGIC,
            version: SMOKE_TEST_VERSION,
            flags: 0,
            provider: PVM_DOT_PROVIDER_HOST_READ_AT,
            file_off_lo: 0,
            file_off_hi: 0,
            block_len: 0,
            quant_kind: 0,
            vec_off: DI01_LEN,
            vec_len: 0,
            reserved0: 0,
            reserved1: 0,
        }
    }
}

impl DotInput {
    /// Helper function to create a DotInput instance for a Q8_0 block at a given file offset.
    pub fn q8_0(block_file_off: u64) -> Self {
        Self {
            quant_kind: PVM_DOT_Q8_0_BLOCK_LEN,
            vec_len: PVM_DOT_QUANT_Q8_0,
            file_off_lo: block_file_off as u32,
            file_off_hi: (block_file_off >> 32) as u32,
            ..Default::default()
        }
    }
}

impl Default for DotOutput {
    fn default() -> Self {
        Self {
            magic: PVM_DO_MAGIC,
            version: SMOKE_TEST_VERSION,
            stage: 0,
            quant_kind: PVM_DOT_QUANT_Q8_0,
            result_bits: 0,
            vec_len: PVM_DOT_Q8_0_VALUES as u32,
            block_len: 0,
            reserved: 0,
        }
    }
}
