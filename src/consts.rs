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

/// provider 类型：通过 host 回调 `provider_read_at` 按 offset 读取模型文件。
/// The provider type: reading model files at given offsets through the host callback `provider_read_at`.
///
/// Notes:
/// - 这是一个简单的 provider ABI
/// - guest 自己并不直接持有 GGUF 文件，真正的读取动作由 host 完成，并将结果写回 guest 内存
/// - 正式版本中也将延续该机制，未来会通过对模型文件生成承诺来验证数据是否被篡改
pub const PVM_DOT_PROVIDER_HOST_READ_AT: u32 = 4;