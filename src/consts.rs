// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs

/// Quantization format enum value: Q8_0.
/// This is a temporary convention used for the smoke test.
pub const PVM_DOT_QUANT_Q8_0: u32 = 8;

/// A single Q8_0 block contains 32 quantized values.
pub const PVM_DOT_Q8_0_VALUES: usize = 32;