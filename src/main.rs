// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs
use anyhow::Result;

pub mod consts;
pub mod primitives;

mod utils;

fn main() -> Result<()> {
    println!("Hello, JAM!");
    Ok(())
}
