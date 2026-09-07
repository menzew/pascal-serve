# CUDA kernel work on GTX 1080

This package has two engine profiles: `baseline` is the pinned upstream engine; `pascal` applies the checksum-verified patch in `patches/`. Both compile only for compute capability 6.1. The patch changes Q6_K weight unpacking on Pascal DP4A devices. The serving launcher and patch do not modify the operating-system kernel, GPU driver, clocks, voltage, or power settings.

The stock profile remains the portable default. Select the patched profile explicitly after measuring it on your card. Q4_K_M models commonly contain Q6_K tensors, so the patch can affect them too. Models without Q6_K weights receive no benefit from this patch.

The optional Docker recipe builds the patched profile; its `PASCAL_SERVER` setting selects that bundled binary. The updated container recipe has not been built or GPU-tested here. Native Linux builds are the validated route.

## Reproduce the build and comparison

```bash
python3 pascal.py build --kernels baseline --tests --jobs 2
python3 pascal.py build --kernels pascal --tests --jobs 2
python3 pascal.py kernel-check --kernels pascal

# Stop the serving process and other GPU workloads before benchmarking.
python3 pascal.py kernel-bench --preset qwen38-9b-q4 --mtp 1 \
  --ctx 32768 --batch 1024 --ubatch 256 \
  --prompt-tokens 4096 --generate-tokens 256 --repeats 4 \
  --output kernel-comparison.json

python3 pascal.py serve --preset qwen38-9b-q4 --kernels pascal \
  --ctx 32768 --batch 1024 --ubatch 256 --mtp 1 --require-full-gpu
```

Use `--model /path/to/model.gguf` with another supported single-file GGUF. Leave MTP off unless the model contains a compatible prediction head. The benchmark uses the same settings for both engines; it does not silently tune model quantization or context. To return to stock, stop the server and use the same serving command with `--kernels baseline`. No rebuild is needed once both profiles exist.

Builds use a clean pinned checkout and a separate verified patched checkout. A changed pin, modified patch, unexpected source change, or changed engine binary is rejected. Build manifests record compiler, configuration, patch, and binary hashes. `--tests` adds upstream `test-backend-ops` and `llama-perplexity`; ordinary serving needs only `llama-server`. Keep source and build directories outside your public repository; `.gitignore` excludes the default `.pascal/` directory.

`PASCAL_SERVER` deliberately overrides normal engine selection for external-engine experiments. Unset it to use `--kernels`, and before running `kernel-bench`. External binaries without a matching manifest are identified as unverified in startup records.

## What changed

Upstream converts four packed unsigned six-bit values into signed bytes using a saturating SIMD subtraction. Each input byte is already limited to 0-63. On Pascal, the patch sets the high bit of every byte, subtracts 32 with ordinary packed integer arithmetic, then toggles the high bits. Setting those bits prevents subtraction from borrowing between adjacent bytes. The result is exactly the same signed byte value, from -32 through 31.

This reduces the unpacking instruction cost. Weight precision, accumulation order, quantization, attention, and sampling settings are retained. The architecture guard leaves other GPU generations on the original implementation. No model dimensions or vocabulary sizes are hardcoded into the patch.

The portable launcher currently targets sm_61. Hardware measurements cover one GTX 1080, CUDA 12.6, driver 570.211.01, Ubuntu 24.04, and the included 4B/9B model family. Other sm_61 cards and models need their own measurements. This is a measured optimization, not a claim of universally optimal kernels.

## Validation and measurement

`kernel-check` runs 153 quantized matrix and expert-matrix checks plus 108 fused-operation checks against the upstream CPU reference. It checks process status and the number of matched tests: an empty run is a failure. These tests exercise the changed quantization types and upstream shape, layout, and fusion cases.

The byte transformation also has an exhaustive independent scalar check:

```bash
c++ -O2 -std=c++11 tests/check_q6_packing.cpp -o /tmp/check-q6-packing
/tmp/check-q6-packing
```

It checks all 16,777,216 combinations of four six-bit values. GPU reference tests remain necessary because an algebraic check alone does not validate compiled CUDA code.

`kernel-bench` first runs numerical checks for both profiles. It alternates engine order between rounds, excludes warm-up, uses identical synthetic prompt token IDs, and rejects incomplete generation, unexpected cache reuse, or partial GPU layer offload. It separately measures cold prompt work and writing/coding response throughput. Reports include GPU temperature, clocks, power, and memory snapshots. Four rounds balance the starting order. Thermal throttling and background applications can still affect results; examine the samples instead of interpreting tiny differences as guaranteed gains.

Greedy response hashes are agreement probes, not a broad answer-quality evaluation. The benchmark does not automatically select a winner or restart an existing service. Existing conversations should be stopped explicitly before loading comparison engines.

See [VALIDATION.md](VALIDATION.md) and [validation/kernel-experiments.json](validation/kernel-experiments.json) for hardware results and rejected candidates.

## Profile another workload

Install compatible CUDA 12.x profiling tools separately if they are not already present. These commands were exercised with Nsight Systems 2024.5. They trace CUDA activity and label request phases with NVTX:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --wait=all --kill=none --output=pascal-profile \
  python3 tools/profile_workload.py --model models/Qwen3.8-9B-Q4_K_M.gguf \
  --kernels pascal --mtp 1 --ctx 32768 --prompt-tokens 8192 \
  --output workload.json

nsys export --type=sqlite --output=pascal-profile.sqlite pascal-profile.nsys-rep
python3 tools/summarize_trace.py pascal-profile.sqlite --output kernel-profile.json
```

The workload tool starts and stops its own loopback server. Stop other GPU work first. Request ranges distinguish prompt work, cached long-context generation, and a short writing task. The summary reports kernel names, call counts, accumulated duration, and register usage. Profiling adds overhead; use unprofiled `kernel-bench` runs for performance claims. Summed kernel durations are not wall time when kernels overlap.

## Extending the patch

Keep proposed changes in a separate checkout, preserve a stock engine, inspect activity traces, then run numerical and representative model comparisons. Record losses as well as wins. If a change affects additional quantization types or operations, expand the validation filters and expected counts. The current checks do not certify arbitrary new kernels.

The patch is derived from MIT-licensed llama.cpp; its license is included in `licenses/llama.cpp.txt`. The serving tools use the root MIT license. Rejected thread-table experiments considered Animesh Srivastava's [Pascal MMVQ proposal](https://github.com/ggml-org/llama.cpp/commit/daec666d1416c4a5cda55fcb8fad87a3f10c928d). That thread-table code is not included in the selected patch. This package is a local patch set, not an upstream-approved contribution.
