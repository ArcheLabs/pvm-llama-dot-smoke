// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 ArcheLabs
pub mod consts;
pub mod primitives;

mod utils;

use crate::consts::*;
use crate::primitives::*;
use crate::utils::*;

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use jam_codec::{Decode, Encode};
use polkavm::{Caller, Config, Engine, Linker, Module, ModuleConfig, ProgramBlob};
use std::{fs, path::PathBuf};

type PvmLinker = Linker<HostState, anyhow::Error>;

const DEFAULT_GUEST_BLOB_PATH: &str = "./guest/pvm-guest.polkavm";
const DEFAULT_MODEL_PATH: &str = "./models/qwen2.5-0.5b-instruct-q2_k.gguf";
const DEFAULT_PROMPT: &str = "Hello JAM";

#[derive(Debug, Parser)]
#[command(name = "pvm-host-runner")]
#[command(about = "Run the PVM Q8_0 dot-product smoke test")]
struct Cli {
    /// Path to the compiled PVM blob.
    #[arg(default_value = DEFAULT_GUEST_BLOB_PATH)]
    guest_blob_path: PathBuf,

    /// Path to the GGUF model file.
    #[arg(default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

    /// Input prompt used to derive a deterministic pseudo-embedding vector.
    #[arg(short, long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// Block offset to read from the GGUF model file.
    #[arg(long, default_value_t = FIXED_BLOCK_FILE_OFF)]
    block_file_off: u64,
}

fn build_input(block_file_off: u64, x: &Q8Input) -> Vec<u8> {
    let mut buf = Vec::with_capacity(DI01_LEN as usize + PVM_DOT_Q8_0_VALUES * 4);
    let di = DotInput::q8_0(block_file_off);
    buf.extend_from_slice(di.encode().as_slice());
    buf.extend_from_slice(x.encode().as_slice());
    buf
}

fn build_linker() -> Result<PvmLinker> {
    let mut linker = PvmLinker::new();
    linker.define_typed(
        "provider_read_at",
        |caller: Caller<'_, HostState>,
         off_lo: u32,
         off_hi: u32,
         dst_ptr: u32,
         len: u32|
         -> Result<u64> {
            let off = ((off_hi as u64) << 32) | (off_lo as u64);
            let mut tmp = vec![0u8; len as usize];
            caller.user_data.read_page_zero_padded(off, &mut tmp)?;
            caller
                .instance
                .write_memory(dst_ptr, &tmp)
                .map_err(|e| anyhow!("write_memory failed: {e}"))?;
            Ok(PVM_DOT_OK)
        },
    )?;
    Ok(linker)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let block_file_off = cli.block_file_off;

    // 1. Load the blob bytes and parse them
    let raw_blob = fs::read(&cli.guest_blob_path)?;
    let blob = ProgramBlob::parse(raw_blob.into())?;

    // 2. Initialize the PVM environment
    let config = Config::from_env()?;
    let engine = Engine::new(&config)?;
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob)?;

    // 3. Open the model file and initialize HostState
    let model_file = fs::File::open(&cli.model_path)?;
    let mut host = HostState::new(model_file)?;

    // 4. Build the input data and compute the reference result
    let x: Q8Input = prompt_to_vec(&cli.prompt).into();

    let mut block = Q8Block::default();
    host.read_exact_at(block_file_off, &mut block.0)?;

    let reference_result = dot_q8_0_reference(&block, &x);

    // 5. Build the PVM instance and initialize the memory pointers
    let input = build_input(block_file_off, &x);
    let input_len = input.len() as u32;

    let linker = build_linker()?;
    let pre = linker.instantiate_pre(&module)?;
    let mut instance = pre.instantiate()?;

    let pos = host.init(&mut instance, input_len)?;

    instance.write_memory(pos.input_ptr, &input)?;
    instance.write_memory(pos.output_ptr, &[0u8; DO01_LEN as usize])?;

    // 6. Execute the PVM and retrieve the output
    let output_bytes = host.run(&mut instance, &pos, input_len)?;
    let output = DotOutput::decode(&mut output_bytes.as_slice())?;

    // 7. Validate the output and print the result
    if output.magic != PVM_DO_MAGIC {
        bail!(
            "invalid output magic: expected {:#x}, got {:#x}",
            PVM_DO_MAGIC,
            output.magic
        );
    }

    if output.version != SMOKE_TEST_VERSION {
        bail!(
            "invalid output version: expected {}, got {}",
            SMOKE_TEST_VERSION,
            output.version
        );
    }

    let result = f32::from_bits(output.result_bits);
    let is_approx_equal = approx_eq(result, reference_result);

    println!("fixed_block_off = {:#x}", block_file_off);
    println!("quant_kind      = {}", output.quant_kind);
    println!("vec_len         = {}", output.vec_len);
    println!("block_len       = {}", output.block_len);
    println!("stage           = {}", output.stage);
    println!("guest_result    = {:.9}", result);
    println!("reference       = {:.9}", reference_result);
    println!("approx_equal    = {}", is_approx_equal);

    Ok(())
}
