# GTX 1080 validation — September 7, 2026

**The pinned Linux CUDA build and both the 4B and 9B Q4 models passed hardware tests on a GTX 1080. Both 16K and 32K contexts work with all 34 offloadable layers on the GPU.** The 32K tests processed 32,000 input tokens without truncation.

## Hardware and build

| Component | Tested configuration |
|---|---|
| GPU | NVIDIA GeForce GTX 1080, 8,192 MiB VRAM, compute capability 6.1 |
| Host | Ubuntu 24.04.4 LTS, approximately 16 GB system RAM |
| Driver | 570.211.01; existing driver retained |
| Toolchain | CUDA 12.6.85, GCC 13.3, CMake 3.28.3 |
| Engine | llama.cpp `73a43d1f69345aee8bb186ef4b3172cef892f2e5` |
| Architecture check | `cuobjdump --list-elf` found only `sm_61` in the server executable |
| Models | Empero Qwen3.8-4B and 9B Distill, Q4_K_M |

The model download was **2,783,446,304 bytes** and passed SHA-256 verification against `dec96e8cf2e11b613bb46513dec485377f9ca5a351e71712ee0e244f287c6790`.

The engine was compiled from the pinned, unmodified upstream source with `GGML_CUDA_FORCE_MMQ=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_FA_ALL_QUANTS=OFF`, and `GGML_CUDA_GRAPHS=OFF`. The packaged launcher is custom integration around existing llama.cpp kernels, not a FlashInfer port.

## Automated tests and fixes found on hardware

All **43 tests passed on Ubuntu with Python 3.12** after adding MTP support. Windows with Python 3.14 ran 42 successfully and skipped the POSIX socket test. The added checks cover completed MTP initialization, invalid draft lengths, fixed-context command construction, and stopping a child that becomes healthy without initializing requested MTP.

Coverage includes verified/resumable downloads, invalid files, toolkit selection, CUDA 13 rejection, GPU selection, fixed-context fitting, positive/full offload checks, startup/shutdown, allocation retries, occupied ports, streaming, attention/cache validation, and benchmark comparability.

Two deployment fixes were verified:

- Toolkit detection now prefers the newest versioned CUDA 12 installation over a stale PATH compiler. This host had CUDA 12.0 on PATH and CUDA 12.6 installed.
- Linux readiness preflight now allows a recently closed socket in TIME_WAIT while still rejecting an active listener. A regression test failed before the fix and passed afterward. Sequential real-engine restarts then completed.

## GPU acceptance checks

- Confirmed **34/34 offloadable layers on CUDA**, Flash Attention active, and the exact allocated context via `/props`.
- Passed health, model listing, non-streaming chat, and streaming chat at 16K/f16 and 32K/q8_0.
- With thinking disabled, the model answered “What is 2 plus 2?” with **“4”**.
- Processed a 4,096-token input at 16K/f16 and an 8,192-token input at 32K/q8_0, generating 32 tokens after each.
- Rejected inputs beyond each configured context with HTTP 400 and remained healthy afterward.
- Successfully processed **32,000-token inputs plus 64 generated tokens** at 32K with both f16 and q8_0 K/V storage.
- Confirmed the benchmark's temporary engine processes exited and subsequent profiles started normally.
- Started a separate running 32K/f16 service and verified both chat modes through SSH forwarding from the Windows computer.

“All layers on GPU” refers to the engine's offloadable-layer count. Input embeddings and some operations may still use the CPU; this is not a claim of zero CPU work.

## 4B measured serving performance

The 16K comparison used exactly the same **2,048 input token IDs**, **64 generated tokens**, one sequence, four CPU threads, and **three cold repetitions** per profile. Warm-up was excluded. Cold requests disabled cache reuse and processed the full input. A separate related follow-up tested prefix reuse.

| Profile | Median prompt tokens/s | Median generated tokens/s | Median cold request | Follow-up request |
|---|---:|---:|---:|---:|
| compat: ordinary attention, old batching, caching disabled | 738.0 | 46.0 | 4.15 s | 4.22 s |
| flash: f16 cache, larger batches, caching enabled | 993.3 | 49.1 | 3.36 s | 1.42 s |
| flash-q8: same, with 8-bit K/V | 979.3 | 47.5 | 3.43 s | 1.46 s |

The flash profile had **1.23× the cold-request throughput** of compat in this test. Its related follow-up reused **2,044 prefix tokens**, while compat reused zero. The comparison measures the complete serving configurations; it does not isolate the effect of Flash Attention from batching and caching.

Near-full-window checks allocated 32,768 tokens and fed **32,000 input tokens**, followed by 64 generated tokens. These are **single cold measurements** per profile, not a statistical performance study.

| 32K cache format | Prompt tokens/s | Generated tokens/s | Cold request | Cached follow-up |
|---|---:|---:|---:|---:|
| f16 | 619.7 | 40.6 | 53.26 s | 1.72 s |
| q8_0 | 552.1 | 36.1 | 59.77 s | 1.93 s |

Both follow-ups reused 31,996 input tokens. f16 was faster in this workload and had ample memory headroom, so the deployed server uses **32K with f16**. The portable launcher's default remains 16K; q8_0 remains available when memory savings matter.

## 4B memory observations

| Allocated context | K/V storage | GPU K/V buffer |
|---|---|---:|
| 16,384 | f16 | 512 MiB |
| 32,768 | f16 | 1,024 MiB |
| 32,768 | q8_0 | 544 MiB |

The 16K/32K acceptance and 16K benchmark run peaked at **3,747 MiB** total GPU memory in one-second samples. The near-full 32K comparison peaked at **4,103 MiB**, about **4.01 GiB**, including the desktop and other driver allocations. Sampling can miss brief peaks; this is an observation, not an enforced memory cap.

The model also allocated 2,572.86 MiB of GPU weights and 50.25 MiB of recurrent state, with additional CPU mappings, compute buffers, and prompt/checkpoint storage. K/V savings do not halve total memory.

## 9B hardware test

The `qwen38-9b-q4` preset is [Empero Qwen3.8-9B Distill Q4_K_M](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF/tree/760121cd70bb4c36b2b5ec58eb765e0df5987efe). Its **5,780,090,176-byte** download passed SHA-256 verification against `df13d66021cef676f82be74053220fd75af6bf2a6a7fb77f5222ab9e50744a7a`. The source revision, size, and hash are pinned in `models.lock.json`.

The same native build loaded **34/34 offloadable layers**, activated Flash Attention, and allocated exactly **32,768 tokens** with the f16 context cache. Health, model listing, normal chat, and streamed chat passed. A deterministic arithmetic check returned `4`. An oversized request returned HTTP 400, and the server remained healthy.

| Model / cache | Allocated context | Input / output tokens | Prompt tokens/s | Generated tokens/s | Cold request | Cached follow-up |
|---|---:|---:|---:|---:|---:|---:|
| 4B / f16 | 16,384 | 2,048 / 64 | 993.3 | 49.1 | 3.36 s | 1.42 s |
| 9B / f16 | 16,384 | 2,048 / 64 | 604.8 | 31.6 | 5.40 s | 2.17 s |
| 4B / f16 | 32,768 | 32,000 / 64 | 619.7 | 40.6 | 53.26 s | 1.72 s |
| 9B / f16 | 32,768 | 32,000 / 64 | 428.9 | 27.4 | 76.99 s | 2.51 s |
| 9B / q8_0 | 32,768 | 32,000 / 64 | 394.6 | 25.8 | 83.62 s | 2.65 s |

The 16K figures are medians of three cold runs; the 32K figures are single cold measurements. Both 9B 32K profiles processed the complete input without truncation and reused 31,996 tokens on their cached follow-ups. Tokenization of the synthetic 2,048- and 32,000-token prompts was checked against the 4B server and produced identical token IDs. The comparison uses the same engine, sampling, thread count, batching, and cache settings. It is a serving-speed comparison, not a model-quality evaluation.

The 9B f16 32K run peaked at **6,349 MiB (6.20 GiB)** total GPU memory in one-second samples; q8_0 peaked at **5,989 MiB (5.85 GiB)**. These totals include the desktop and driver, and sampling may miss short peaks. The f16 model allocated 4,812.25 MiB of GPU weights, 1,024 MiB of K/V storage, and 50.25 MiB of recurrent state, plus compute and other allocations. Some weights remained CPU-mapped despite full offloadable-layer residency.

The standard f16 cache was faster and fit with room remaining, so it is the basis for subsequent 9B optimization. The portable default remains 4B; this machine's service has been switched to 9B at the user's request.

## 9B optimization and deployed configuration

The selected service runs **Q4_K_M, 32,768 context, f16 K/V, batch 1,024 / microbatch 256, four CPU threads, one sequence, 512 MiB prompt cache, four checkpoints, and `--mtp 1`**. The one-token MTP setting uses the prediction head already present in this GGUF and the pinned engine's existing speculative decoder. It does not require another model download or a kernel rebuild. The launcher records MTP activation and refuses readiness if requested MTP did not initialize.

Batch tuning compared microbatches 128, 256, 512, and 1,024, with two cold repetitions using 2,048 input tokens and 128 output tokens at a 32K allocation. Microbatch 256 processed about 625 prompt tokens/s versus 600 at the initial baseline; larger batches offered no useful combined request-time improvement. The final baseline repeat was slower, so this small batching gain should not be treated as a precise isolated effect.

MTP tests used two distinct short chat prompts: a traffic-policy explanation and a Python interval-merging function. Each generated 256 tokens twice, with thinking disabled, greedy sampling, seed 42, and prompt reuse disabled. All MTP comparisons used the same 1,024 / 256 batching. The table reports mean end-to-end tokens/s over each task's two repetitions, including request and prompt-processing overhead.

| Setting | Writing tokens/s | Coding tokens/s | Total time for four requests |
|---|---:|---:|---:|
| MTP off, initial baseline | 30.34 | 30.27 | 33.79 s |
| MTP 1 token, selected | 37.83 | 43.23 | 25.38 s |
| MTP 2 tokens | 35.81 | 45.77 | 25.49 s |
| MTP 3 tokens | 31.26 | 43.60 | 28.12 s |
| MTP off, repeated baseline | 30.29 | 30.22 | 33.85 s |

One-token MTP improved these writing and coding rates by about **25% and 43%**, respectively. All tested MTP outputs matched the baseline text exactly, including the repeated runs. This checks those specific greedy continuations; it does not establish universal output equivalence or broad model quality. The selected head accepted approximately 68% of writing proposals and 92% of coding proposals. Acceptance and speed depend on the workload.

The exact selected configuration then passed a separate **32,000-input / 128-output** comparison at 32K, with one cold measurement per configuration:

| Configuration | Prompt tokens/s | Generated tokens/s | Cold request | Cached follow-up |
|---|---:|---:|---:|---:|
| Original 9B: 512 / 128 batch, MTP off | 431.35 | 27.38 | 78.90 s | 4.89 s |
| Selected 9B: 1,024 / 256 batch, MTP 1 | 390.50 | 39.16 | 85.26 s | 3.78 s |

**Tradeoff:** MTP increased generation throughput about 43% on this nearly full context, but reduced prompt-processing throughput about 9.5%. The fresh long-document request took about 8% longer overall, while the cached follow-up took about 23% less time. Both follow-ups reused 31,996 input tokens. MTP is selected for interactive generation; `--mtp 0` remains available for workloads dominated by fresh long inputs.

The selected full-context run peaked at **6,771 MiB (6.61 GiB)** total GPU memory in one-second samples, including desktop/driver allocations. All **34/34 offloadable layers** stayed on the GPU. The MTP model allocated 4,956.24 MiB of GPU weights, 1,024 MiB of main K/V, 128 MiB of prediction-head K/V, and 100.50 MiB of recurrent state, plus compute and other allocations. These are measured allocations, not a memory cap.

After the full-context checks, the selected configuration passed health, model listing, non-streaming and streaming chat, and oversized-request rejection. The running service's actual child-process report confirmed the 9B model, exact 32K context, full offload, active Flash Attention, and active one-token MTP. The same service passed both chat modes through the existing SSH tunnel from Windows. Its start script now preserves the selected 9B settings.

GPU temperature and clocks varied during the sweeps, with the card reaching the mid-80s Celsius under load. No clock, voltage, power-limit, driver, or fan settings were changed. These short tests are not a controlled thermal or concurrency benchmark.

## CUDA kernel experiments (7 September 2026)

CUDA activity traces on the selected 9B/MTP-1 server attributed **85-87% of generation GPU time** to quantized matrix-vector multiplication and **74-78% of prompt GPU time** to quantized matrix-matrix multiplication. These are shares of traced GPU kernel duration, not end-to-end latency. They guided the optimization work toward quantized matrix operations. Attention kernels remain the pinned engine's existing implementation.

Candidates were built separately, preserving the serving engine. Each matrix-vector candidate passed 153 matrix/expert-matrix checks and 108 fused-operation checks against upstream CPU references before response timing. The tests used the same 9B model, 32K context, 1,024 / 256 batching, f16 cache, MTP 1, and fixed 256-token writing/coding probes. Baseline runs bracketed each sweep.

| Experiment | Observed result | Decision |
|---|---|---|
| One warp for Q4_K/Q6_K, 1-2 columns | Writing/coding about 9-11% slower | Rejected |
| Two warps for Q4_K/Q6_K, 1-2 columns | Writing roughly flat; coding about 4% slower | Rejected |
| One row per block for two columns | Writing/coding about 8% slower | Rejected |
| Four rows per block for one/two columns | Writing/coding about 30% slower | Rejected |
| Smaller prompt tiles: 16, 32, 48 columns | All slower than the existing 64-column selection | Rejected |
| Floating-point BLAS for prompt matrices | Roughly half the prompt throughput | Rejected |
| Packed Q6 subtraction | Isolated Q6 kernels about 3-5% faster; response probes identical | Retained as an optional patch |

The Q6 microbenchmark used matrices with 4,096 output rows, inner dimension 14,336, and one or two input columns. Bracketing stock measurements averaged 290.15 and 289.41 microseconds, versus 275.94 and 281.03 with the patch. The initial two-run response medians moved from 38.65 to 38.83 tokens/s for writing and 43.51 to 43.96 for coding. These small end-to-end differences are close to thermal and timing variation; they do not establish a substantial application-level speedup.

The selected conversion additionally passed an exhaustive independent scalar check over **all 16,777,216 four-byte Q6 combinations**. It preserves the signed integer values exactly and does not change the reduction order. The shipped patch is a small change to the existing Q6 unpacking function, with a Pascal architecture guard; experimental thread tables and row layouts are excluded.

Detailed measurements, output agreement, accuracy counts, and tracing evidence are in [validation/kernel-experiments.json](validation/kernel-experiments.json). The stock engine remains the generic default, and [KERNELS.md](KERNELS.md) documents the optional patched build and reproducible comparison. Portable tests passed **51/51 on Ubuntu** and **50 passed / 1 platform-specific skip on Windows**. The supplied GitHub Actions workflow is prepared but has not run on GitHub. No repository has been published.

## Packaged CUDA release verification

Both engine profiles built successfully from the packaged workflow with CUDA 12.6. Inspection of their embedded GPU code found **sm_61 only**. Both passed all **261 CUDA accuracy checks**. The release comparison, full-context acceptance, API checks, and packaged Nsight profiling/export/summary tools passed. Raw results and binary identities are in [validation/release-validation.json](validation/release-validation.json).

The 9B comparison used four alternating rounds per engine, 4,096 cold prompt tokens, then the two fixed 256-token response probes. The table reports medians; the cold synthetic request precedes each pair of response probes, so these timings should not be compared directly with the earlier task-only MTP table.

| 9B workload, MTP 1 | Stock | Pascal patch | Measured difference |
|---|---:|---:|---:|
| Writing, end-to-end tokens/s | 35.71 | 36.02 | +0.9% |
| Coding, end-to-end tokens/s | 41.10 | 41.63 | +1.3% |
| Prompt processing, tokens/s | 563.83 | 566.54 | +0.5% |

All eight patched writing/coding responses matched their baseline hashes. The gain is modest, and clock/temperature variation limits precision. The prompt path is unchanged by the patch; its tiny difference should be treated as measurement variation. The patched profile is selected on the tested 9B server, while the portable default remains stock so other users can measure their own setup.

The 4B model also passed a two-round comparison with MTP off, a 16K allocation, 2,048 prompt tokens, and 128-token responses. All four patched responses matched baseline hashes. Writing medians were 45.59 versus 45.78 tokens/s. Coding medians were 42.96 versus 46.58, but the stock samples ranged from 40.09 to 45.84; this small sample is not evidence of a dependable 8% improvement.

The patched 9B engine then processed **32,000 input tokens and 128 generated tokens** at a **32,768-token allocation**, retaining all **34/34 offloadable layers**, Flash Attention, and MTP 1. The cold request took 85.66 seconds (389.17 prompt tokens/s and 37.76 generated tokens/s). A follow-up reused **31,996 tokens** and took **3.60 seconds**. The long-context test passed without truncation. These are acceptance measurements, not a paired claim of further long-context speedup. Normal and streamed API generation both passed afterward.

The long-context run peaked at **6,771 MiB (6.61 GiB)** total GPU memory in one-second samples. The separate Nsight profiling run peaked at 6,799 MiB. Sampling can miss brief peaks. The card reached 88 C during the long-input test, reinforcing the need to account for temperature and clock changes in small performance comparisons.

The final packaged binaries repeated the Q6 microbenchmark: **274.58 / 279.63 microseconds** for one/two columns, versus bracketing stock averages of **287.64 / 288.12 microseconds**, approximately **4.8% / 3.0% faster**. The [final deployment record](validation/final-deployment.json) contains these samples and the verified running engine identity. A final comment-wording correction was rebuilt and produced byte-identical server, benchmark, accuracy-test, and perplexity binaries; this explains the patch-hash difference from the earlier comparison record. The [trace summary](validation/profile-summary.json) was produced by the packaged profiling tools.

The running 9B service was verified again through the existing SSH connection after deployment: health, model listing, non-streaming generation, and streamed generation passed. The service start script retains the patched profile and the selected 32K/MTP-1 configuration. The stock build remains available for comparison and rollback.

## Practical limits

These are synthetic throughput checks plus simple chat checks. No broad answer-quality or long-document retrieval evaluation was performed. The accepted 64K setting, Q8 weight preset, concurrent-client stress, and other GPUs remain untested. The optional CUDA 12.9 Docker recipe has not been built here; hardware validation used the native CUDA 12.6 build.

Earlier Windows CPU integration checks used upstream release b10823 (`7620399f58aebfd2196b74021f9581bcf7218cb9`). The GPU results above use the actual packaged source pin and supersede the earlier lack of CUDA validation.
