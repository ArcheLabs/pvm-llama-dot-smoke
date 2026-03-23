# **pvm-llama-dot-smoke**

This project aims to build a **minimal verification prototype** for LLM computation. The current version is still a very early **smoke test**: rather than replaying the full inference process, it extracts a real quantized block from an actual GGUF model file, executes a minimal quantized dot product inside the PVM, and then performs an offline reference check on the host side.

The goal of this prototype is not to fully prove that complete LLM inference can already be reproduced and verified inside the PVM. Instead, it is meant to show the feasibility of verifying LLM inference in the PVM through minimal operators.

## Principles

Models such as Llama are, in essence, **decoder-only Transformers**. The inference process can be roughly described as follows:

1. Split the input text into tokens.
2. Map tokens into vectors.
3. Pass them through multiple Transformer blocks, repeatedly performing computations such as attention and MLPs.
4. Produce the final logits.
5. Select the next token and return to step 2.

Across these steps, whether in attention or MLPs, the most common primitive is in fact a large number of **dot products**. This makes it possible to design verification mechanisms that do not require replaying the full inference process.

**Sampling**: We sample a subset of these massive numbers of primitive computations for verification, thereby assigning a degree of confidence to the overall inference process. This sampling mechanism does not require loading the full model into the PVM. In a complete design, verifiers would not need to hold the full model; they would only need the sampled data together with its proof. In the future, this could be combined with commitment schemes such as KZG / Pedersen / Merkle to prove that the sampled data indeed comes from a specific model and inference context.

**Optimistic verification**: Similar to the mechanism of optimistic rollups, interactive verification is triggered only when needed, narrowing the dispute down to the minimal erroneous computation and verifying it inside the PVM.

If you are not very familiar with LLM fundamentals, the following two articles are recommended:

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2)

## Properties

A verification mechanism based on minimal computation units naturally brings the following properties:

1. **Lightweight**: It avoids the engineering complexity and the large runtime cost of replay-based inference verification.
2. **Weaker trust assumptions**: Compared with replaying full inference inside a TEE, this approach does not require correctness to rest on proprietary hardware black boxes. Instead, it reduces disputes to minimal computation samples that can be checked publicly.
3. **Adaptivity**: For the verification of a single operator, we can tolerate floating-point differences across hardware and implementations; such differences are almost unavoidable in full inference replay.
4. **Model-size independence**: Verifiers only need the sampled data. Commitment/proof data is negligible compared with the size of the full model.

## Architecture

- **llama.cpp / ggml runtime**: a minimal subset of C/C++ computation logic extracted from `llama.cpp` / `ggml` that can run inside the PVM environment.
- **Guest side**
    - define the guest entry point
    - define the input/output ABI
    - call host callbacks to read the model file
    - invoke the runtime above
    - write the result back into the output buffer
- **Host side**
    - load the `.polkavm`
    - open the GGUF file
    - write input into guest memory
    - provide `provider_read_at`
    - start the guest
    - read the output and compare it against the host reference

### Notes

- This repository currently provides the host runner and the related verification logic.
- The compiled guest blob is located in the `guest/` directory.
- The current runtime and guest depend on experimental upstream modifications and have not yet been independently cleaned up and open-sourced.
- A later goal is to extract a clean and reusable runtime and guest project.

## Limitations

This project is still at a very early stage. At present, we have achieved the following:

1. Llama’s core primitives and computations can be compiled to the PVM side.
2. The result can be returned through the ABI and checked deterministically by the host.

This shows that the PVM can serve as a minimal prototype for a future LLM computation verification system.

However, the following areas are not yet covered:

1. **The full LLM inference pipeline is not yet covered**: in theory, verification can be extended to more inference stages, but the engineering complexity is high, and the current stage does not aim for full replay or full verification.
2. Verification of other inference-stage components: tokenizer / vocab / sampler / KV cache.
3. In the current smoke test, only specific types are supported. General tensor scanning and general quantization support will be implemented in the next stage.
4. **Cross-implementation consistency testing**: for toolchain completeness, this project currently uses the PolkaVM implementation. Cross-PVM consistency testing is not yet supported, and this is not the best stage to prioritize it. If you specifically want to test other PVM clients, you can implement support yourself by referring to this project; a compiled `.polkavm` is already available under the [guest](https://github.com/ArcheLabs/pvm-llama-dot-smoke/tree/main/guest) directory.

## Usage

### 1. Setup

Clone this repository and download the model.

```
# Clone this repository
git clonegit@github.com:ArcheLabs/pvm-llama-dot-smoke.git
cd pvm-llama-dot-smoke

# Download the model
huggingface-cli download \
  Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  qwen2.5-0.5b-instruct-q2_k.gguf \
--local-dir ./models
```

### 2. Test

Run the test. Note:

- the current default test target is `qwen2.5-0.5b-instruct-q2_k.gguf`
- the current input protocol, quantization parsing, and fixed block selection are still bound to this model
- after changing the model or quantization format, the current version may not work directly

```
# Run directly
cargo run --release

# Custom prompt input
cargo run --release -- --prompt "Hello JAM"
```

If you see output similar to the following, the test has succeeded:

```
fixed_block_off = 0x5ac160
quant_kind      = 8
vec_len         = 32
block_len       = 34
stage           = 6
guest_result    = -0.046125144
reference       = -0.046125144
approx_equal    = true
```

## Roadmap

- further abstract the input protocol and sample description
- add minimal verification support for more quantization formats
- extend from a single dot product to single-row matmul verification
- design and implement a sampling mechanism

## References

- https://graypaper.com/
- https://jalammar.github.io/illustrated-gpt2/
- https://jalammar.github.io/illustrated-transformer
- https://github.com/ggml-org/llama.cpp