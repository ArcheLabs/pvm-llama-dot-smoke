// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs
use crate::consts::{
    PVM_DI_MAGIC, PVM_DOT_OK, PVM_DOT_PROVIDER_HOST_READ_AT, PVM_DOT_Q8_0_BLOCK_LEN,
    PVM_DOT_Q8_0_VALUES, PVM_DOT_QUANT_Q8_0, PVM_DO_MAGIC, SMOKE_TEST_VERSION,
};
use anyhow::{anyhow, bail, Error, Result};
use jam_codec::{Decode, Encode};
use polkavm::Instance;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

/// Fixed-length floating-point input vector for the Q8_0 dot-product smoke test.
///
/// This wraps the dense input vector `x` used by the host-side reference path
/// and by the guest-side verifier input protocol.
///
/// The length is fixed by `PVM_DOT_Q8_0_VALUES`, which matches the logical
/// number of values represented by one Q8_0 block.
#[derive(Debug, Default, Clone, Decode, Encode)]
pub struct Q8Input(pub [f32; PVM_DOT_Q8_0_VALUES]);

impl From<[f32; PVM_DOT_Q8_0_VALUES]> for Q8Input {
    fn from(slice: [f32; PVM_DOT_Q8_0_VALUES]) -> Self {
        let mut arr = [0.0f32; PVM_DOT_Q8_0_VALUES];
        arr.copy_from_slice(&slice[..PVM_DOT_Q8_0_VALUES]);
        Self(arr)
    }
}

/// Raw on-disk bytes of a single Q8_0 quantized block.
///
/// This type stores the exact block payload as read from the model file,
/// without interpreting it at the type level. The block layout is expected
/// to follow the Q8_0 encoding convention:
/// - a leading fp16 scale;
/// - followed by quantized 8-bit values.
///
/// The total byte size is fixed by `PVM_DOT_Q8_0_BLOCK_LEN`.
#[derive(Debug, Clone, Decode, Encode)]
pub struct Q8Block(pub [u8; PVM_DOT_Q8_0_BLOCK_LEN as usize]);

impl From<&[u8]> for Q8Block {
    fn from(bytes: &[u8]) -> Self {
        let mut block = [0u8; PVM_DOT_Q8_0_BLOCK_LEN as usize];
        block.copy_from_slice(&bytes[..PVM_DOT_Q8_0_BLOCK_LEN as usize]);
        Self(block)
    }
}

impl Default for Q8Block {
    fn default() -> Self {
        Self([0u8; PVM_DOT_Q8_0_BLOCK_LEN as usize])
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

/// The total byte length of the DI01 header, used for buffer layout calculations.
pub const DI01_LEN: u32 = std::mem::size_of::<DotInput>() as u32;

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

/// The total byte length of the DO01 header, used for buffer layout calculations.
pub const DO01_LEN: u32 = std::mem::size_of::<DotOutput>() as u32;

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
            block_len: PVM_DOT_Q8_0_BLOCK_LEN,
            quant_kind: PVM_DOT_QUANT_Q8_0,
            vec_len: PVM_DOT_Q8_0_VALUES as u32,
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

/// Resolved guest memory layout for the current PVM instance.
///
/// These fields are obtained from guest-exported helper functions and define
/// where the host should write the encoded input and where it should read the
/// fixed-size output buffer.
///
/// `*_ptr` fields are guest memory addresses, and `*_cap` fields are the
/// corresponding capacities in bytes.
#[derive(Debug, Clone, Decode, Encode)]
pub struct InstancePosition {
    /// Guest pointer to the start of the input buffer.
    pub input_ptr: u32,

    /// Total capacity of the guest input buffer, in bytes.
    pub input_cap: u32,

    /// Guest pointer to the start of the output buffer.
    pub output_ptr: u32,

    /// Total capacity of the guest output buffer, in bytes.
    pub output_cap: u32,
}

/// Fixed sample: `output.weight` / `Q8_0` / `block 0`
///
/// The smoke test is bound to the fixed block offset in the real model file.
/// If the model file is changed, this value must be updated accordingly.
///
/// Note: the tensor/type information of a GGUF model file can be inspected with:
/// https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/scripts/gguf_dump.py
pub const FIXED_BLOCK_FILE_OFF: u64 = 5_947_744;

#[derive(Debug)]
pub struct HostState {
    /// Open handle to the model file.
    model_file: File,
    /// Cached total length of the model file, used for bounds checks.
    model_len: u64,
}

impl HostState {
    /// Creates a new host state from the opened model file.
    ///
    /// This records the total file length up front so that later reads can
    /// perform explicit bounds checks before touching the file.
    pub fn new(model_file: File) -> Result<Self> {
        let model_len = model_file.metadata()?.len();
        Ok(Self {
            model_file,
            model_len,
        })
    }

    /// Reads exactly `buf.len()` bytes starting at `offset`.
    ///
    /// This is a strict, bounds-checked read:
    /// - it fails if `offset + len` overflows;
    /// - it fails if the requested range exceeds the model file length;
    /// - otherwise it seeks to `offset` and fills the entire buffer.
    ///
    /// This helper is used for protocol-critical reads where partial data
    /// would indicate a logic error rather than an expected short read.
    pub fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let len = buf.len() as u64;

        let end = offset
            .checked_add(len)
            .ok_or_else(|| anyhow!("offset overflow: offset {:#x}, length {:#x}", offset, len))?;

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

    /// Reads a page-like chunk from the model file and zero-pads the remainder.
    ///
    /// The destination buffer is first cleared to zero. Then:
    /// - if `off` is already beyond EOF, the buffer remains all zeroes;
    /// - otherwise, up to `buf.len()` available bytes are read from `off`;
    /// - any unread tail stays zero-filled.
    ///
    /// This is useful when emulating page-based reads where the final page may
    /// be only partially backed by file data.
    pub fn read_page_zero_padded(&mut self, off: u64, buf: &mut [u8]) -> Result<()> {
        buf.fill(0);

        let avail_len = self.model_len.saturating_sub(off);
        if avail_len == 0 {
            return Ok(());
        }

        let read_len = avail_len.min(buf.len() as u64) as usize;
        self.read_exact_at(off, &mut buf[..read_len])?;

        Ok(())
    }

    /// Queries the guest-exported input/output buffer layout and validates it.
    ///
    /// This method calls the guest exports:
    /// - `pvm_input_ptr`
    /// - `pvm_input_cap`
    /// - `pvm_output_ptr`
    /// - `pvm_output_cap`
    ///
    /// It then verifies that:
    /// - the prepared input fits into the guest input buffer;
    /// - the fixed `DO01` output area fits into the guest output buffer.
    ///
    /// On success, it returns the resolved buffer pointers and capacities for
    /// the current instance.
    pub fn init(
        &mut self,
        instance: &mut Instance<HostState, Error>,
        input_len: u32,
    ) -> Result<InstancePosition> {
        let input_ptr: u32 = instance
            .call_typed_and_get_result(self, "pvm_input_ptr", ())
            .map_err(|e| anyhow!("call pvm_input_ptr failed: {:?}", e))?;

        let input_cap: u32 = instance
            .call_typed_and_get_result(self, "pvm_input_cap", ())
            .map_err(|e| anyhow!("call pvm_input_cap failed: {:?}", e))?;

        let output_ptr: u32 = instance
            .call_typed_and_get_result(self, "pvm_output_ptr", ())
            .map_err(|e| anyhow!("call pvm_output_ptr failed: {:?}", e))?;

        let output_cap: u32 = instance
            .call_typed_and_get_result(self, "pvm_output_cap", ())
            .map_err(|e| anyhow!("call pvm_output_cap failed: {:?}", e))?;

        if input_len > input_cap {
            bail!(
                "input length {} exceeds input capacity {}",
                input_len,
                input_cap
            );
        }

        if DO01_LEN > output_cap {
            bail!(
                "DO01 header length {} exceeds output capacity {}",
                DO01_LEN,
                output_cap
            );
        }

        Ok(InstancePosition {
            input_ptr,
            input_cap,
            output_ptr,
            output_cap,
        })
    }

    /// Executes the guest entrypoint and returns the raw `DO01` output bytes.
    ///
    /// The guest `main` function is invoked with:
    /// - the input pointer and actual input length;
    /// - the output pointer and fixed `DO01` output length.
    ///
    /// The guest is expected to return `PVM_DOT_OK` on success. Any other status
    /// is treated as a guest-side failure and returned as an error.
    ///
    /// If execution succeeds, this method reads back exactly `DO01_LEN` bytes
    /// from guest memory and returns them to the host for decoding.
    pub fn run(
        &mut self,
        instance: &mut Instance<HostState, Error>,
        pos: &InstancePosition,
        input_len: u32,
    ) -> Result<Vec<u8>> {
        let status: u64 = instance
            .call_typed_and_get_result(
                self,
                "main",
                (
                    pos.input_ptr,
                    input_len,
                    pos.output_ptr,
                    DO01_LEN as u32,
                ),
            )
            .map_err(|e| anyhow!("guest call failed: {:?}", e))?;

        if status != PVM_DOT_OK {
            return Err(anyhow!("guest returned error status: {:#x}", status));
        }

        let out = instance.read_memory(pos.output_ptr, DO01_LEN as u32)?;

        Ok(out)
    }
}