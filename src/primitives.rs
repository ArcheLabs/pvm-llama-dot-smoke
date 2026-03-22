// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs
use crate::consts::{
    PVM_DI_MAGIC, PVM_DOT_PROVIDER_HOST_READ_AT, PVM_DOT_Q8_0_BLOCK_LEN, PVM_DOT_Q8_0_VALUES,
    PVM_DOT_QUANT_Q8_0, PVM_DO_MAGIC, SMOKE_TEST_VERSION,
};
use anyhow::{bail, Result};
use jam_codec::{Decode, Encode};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

#[derive(Debug, Default, Clone, Decode, Encode)]
pub struct Q8Input(pub [f32; PVM_DOT_Q8_0_VALUES]);

impl From<&[f32]> for Q8Input {
    fn from(slice: &[f32]) -> Self {
        let mut arr = [0.0f32; PVM_DOT_Q8_0_VALUES];
        arr.copy_from_slice(&slice[..PVM_DOT_Q8_0_VALUES]);
        Self(arr)
    }
}

#[derive(Debug, Clone, Decode, Encode)]
pub struct Q8Block(pub [u8; PVM_DOT_Q8_0_BLOCK_LEN as usize]);

impl From<&[u8]> for Q8Block {
    fn from(bytes: &[u8]) -> Self {
        let mut block = [0u8; PVM_DOT_Q8_0_BLOCK_LEN as usize];
        block.copy_from_slice(&bytes[..PVM_DOT_Q8_0_BLOCK_LEN as usize]);
        Self(block)
    }
}

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

/// Fixed sample: `output.weight` / `Q8_0` / `block 0`
///
/// The smoke test is bound to the fixed block offset in the real model file.
/// If the model file is changed, this value must be updated accordingly.
///
/// Note: the tensor/type information of a GGUF model file can be inspected with:
/// https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/scripts/gguf_dump.py
const FIXED_BLOCK_FILE_OFF: u64 = 5_947_744;

#[derive(Debug)]
pub struct HostState {
    /// Open handle to the model file.
    model_file: File,
    /// Cached total length of the model file, used for bounds checks.
    model_len: u64,
}

impl HostState {
    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let len = buf.len() as u64;

        let end = offset.checked_add(len).ok_or_else(|| {
            anyhow::anyhow!("offset overflow: offset {:#x}, length {:#x}", offset, len)
        })?;

        if end > self.model_len {
            bail!(
                "read out of bounds: offset {:#x}, length {:#x}, but model length is only {:#x}",
                offset,
                len,
                self.model_len
            );
        }
        self.model_file.seek(SeekFrom::Start(offset))?;
        self.model_file.read_exact(buf)?;
        Ok(())
    }

    fn read_page_zero_padded(&mut self, off: u64, buf: &mut [u8]) -> Result<()> {
        buf.fill(0);

        let avail_len = self.model_len.saturating_sub(off);
        if avail_len == 0 {
            return Ok(());
        }

        let read_len = avail_len.min(buf.len() as u64) as usize;
        self.read_exact_at(off, &mut buf[..read_len])?;

        Ok(())
    }
}
