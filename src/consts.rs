// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs

/// Quantization format enum value: Q8_0.
/// This is a temporary convention used for the smoke test.
pub const PVM_DOT_QUANT_Q8_0: u32 = 8;

/// A single Q8_0 block contains 32 quantized values.
pub const PVM_DOT_Q8_0_VALUES: usize = 32;

/// Total byte length of a Q8_0 block.
///
/// Layout:
/// - 2 bytes: d (fp16)
/// - 32 bytes: 32 i8 quantized values
pub const PVM_DOT_Q8_0_BLOCK_LEN: u32 = PVM_DOT_Q8_0_VALUES as u32 + 2;

/// Input protocol magic: `DI01`.
pub const PVM_DI_MAGIC: u32 = u32::from_le_bytes(*b"DI01");

/// Output protocol magic: `DO01`.
pub const PVM_DO_MAGIC: u32 = u32::from_le_bytes(*b"DO01");

/// Smoke test protocol version.
pub const SMOKE_TEST_VERSION: u32 = 1;

/// Provider kind: read the model file at a given offset through the host callback
/// `provider_read_at`.
///
/// Notes:
/// - This is a minimal provider ABI.
/// - The guest does not hold the GGUF file directly; the actual read is performed
///   by the host, which writes the result back into guest memory.
/// - The production design will keep this mechanism. In the future, commitments
///   over the model file can be used to verify that the data has not been tampered with.
pub const PVM_DOT_PROVIDER_HOST_READ_AT: u32 = 4;