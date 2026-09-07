# Pascal Serve

A small Linux serving stack for a **GTX 1080 8 GB**, built around a pinned llama.cpp engine compiled for **Pascal / sm_61 with CUDA 12.x**. The Python launcher uses only the standard library. There is no PyTorch, vLLM, or Python CUDA wheel dependency.

The default model is **Empero Qwen3.8-4B Distill, Q4_K_M**, a 2.783 GB download (2.592 GiB). This is a **community distillation of Qwen3.8 into Qwen3.5-4B**, not an official Qwen 4B release. See the [publisher's model card](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF). Its hybrid Qwen3.5 / Gated DeltaNet architecture is supported by the pinned engine.

This project supplies build configuration, verified downloads, launch policy, diagnostics, and API checks. llama.cpp supplies the inference kernels, model loader, memory fitting, scheduling, and HTTP API. This is not a newly written inference engine.

An optional **Pascal CUDA kernel patch** and reproducible stock-versus-patched build, accuracy, benchmark, and profiling tools are now included. See [KERNELS.md](KERNELS.md). The stock engine remains the default; `--kernels pascal` selects the patched build.

The default context is **16,384 tokens**. This version enables llama.cpp's existing Flash Attention path, larger prompt batches, and conversation caching, with an optional 8-bit K/V cache for longer contexts. The Linux CUDA build and real GTX 1080 execution have passed hardware tests at 16K and 32K. The 16K benchmark reached about **49 generated tokens/second**; see [VALIDATION.md](VALIDATION.md) for the workload and measured limits.

## Start on Linux

Use x86-64 Linux with an AVX2-capable CPU, Python 3.10+, Git, CMake 3.20+, a C++ compiler, and CUDA Toolkit **12.x**. CUDA **12.6 with GCC 13.3 on Ubuntu 24.04** was verified on hardware; CUDA **12.9** is the recommended version for a fresh installation and is used by the optional container recipe. Use a proprietary NVIDIA driver that supports Pascal; NVIDIA identifies **R580** as the final supporting branch. The tested driver was **570.211.01**. Do not replace a working driver just because `nvidia-smi` prints a different CUDA version: that field is the driver's supported CUDA level, not the installed compiler.

On Ubuntu 22.04, install the ordinary build tools:

```bash
sudo apt-get update
sudo apt-get install -y python3 git cmake build-essential ca-certificates
```

Install CUDA Toolkit 12.9 using [NVIDIA's CUDA 12.9 archive](https://developer.nvidia.com/cuda-12-9-1-download-archive), selecting your distribution. Keep the existing compatible driver when installing just the toolkit. CUDA 13 cannot compile for this card. The build command checks the actual `nvcc` version and refuses CUDA 13.

Copy or unzip this directory onto the Linux machine, open a terminal in it, and run:

```bash
python3 pascal.py doctor
python3 pascal.py build --jobs 2
python3 pascal.py pull
python3 pascal.py serve
```

If upgrading from the initial 2K version, run `build` again: that version compiled out CUDA Flash Attention. Reusing its old binary will not enable the new kernels.

`build` downloads the pinned engine source and compiles it locally. `pull` downloads only the selected model, resumes interrupted downloads, and verifies its pinned SHA-256. Budget about 6–10 GB of free disk space for the model and build; compilation needs additional system RAM. Start with two build workers on a machine with limited RAM.

The launcher selects the newest versioned CUDA 12 toolkit under `/usr/local` before consulting `PATH`, avoiding a stale compiler when several toolkits are installed. To select a specific installation:

```bash
python3 pascal.py build --nvcc /path/to/cuda-12.9/bin/nvcc --jobs 2
```

Once you see `Ready`, open a second terminal in the directory:

```bash
python3 pascal.py check
python3 pascal.py chat --prompt "Write a short Python function to remove duplicates from a list." --max-tokens 512 --no-thinking
```

This is a reasoning model. The token limit includes reasoning, and a short limit can end before a final answer. `--no-thinking` requests a direct answer. Otherwise, the chat client hides separately reported reasoning unless `--show-reasoning` is set. The `check` command validates generation and streaming, not answer quality.

Press Ctrl+C in the serving terminal to stop the launcher and its engine.

### Optional 9B model

The `qwen38-9b-q4` preset downloads **Empero Qwen3.8-9B Distill, Q4_K_M** (5.780 GB). It is a community distillation into Qwen3.5-9B. The exact revision and SHA-256 are pinned in `models.lock.json`; the default remains the 4B model. Stop the current GPU server before loading another model.

```bash
python3 pascal.py pull qwen38-9b-q4
python3 pascal.py serve --preset qwen38-9b-q4 --ctx 32768 --require-full-gpu
```

On the tested GTX 1080, the 9B model benefited from one-token MTP speculation and a 256-token microbatch:

```bash
python3 pascal.py serve --preset qwen38-9b-q4 --ctx 32768 \
  --batch 1024 --ubatch 256 --mtp 1 --require-full-gpu
```

`--mtp 1` uses the model's built-in prediction head to propose one token ahead, which the target model verifies. It needs a compatible head in the GGUF and additional runtime memory. Startup fails if MTP does not initialize. `--mtp 0` disables it; lengths 1–3 are available for workload-specific testing. This setting uses the same model file and the pinned engine's existing MTP implementation. Writing and coding tests produced identical greedy output to the baseline while improving throughput; this is not a broad equivalence or quality evaluation.

Consult [VALIDATION.md](VALIDATION.md) for measured memory and throughput. To compare it with the 4B model using the same workload:

```bash
python3 pascal.py bench --preset qwen38-9b-q4 --ctx 16384 \
  --profiles flash --prompt-tokens 2048 --generate-tokens 64 --repeats 3 \
  --output benchmark-9b-16k.json
```

Add `--mtp 1` to a benchmark command to measure MTP with that command's selected batch/cache profile. The standard `flash` benchmark profile still uses batch 512 / microbatch 128; the deployed 9B tuning above uses 1024 / 256. Full tuning settings and measurements are recorded in `VALIDATION.md`.

## Connect an application

- API base URL: `http://127.0.0.1:8080/v1`
- Model ID: `pascal-qwen`
- API key: any placeholder for local clients that require one; authentication is off by default on loopback.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"pascal-qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":512,"stream":true}'
```

For access from a laptop, leave the server on loopback and forward its port:

```bash
ssh -L 8080:127.0.0.1:8080 your-user@your-linux-machine
```

To bind directly to the network, supply `--host 0.0.0.0 --api-key-file /path/to/key.txt`. The file should contain one key on one line. Use TLS at a reverse proxy for untrusted networks. This launcher does not install a service or alter your firewall.

## How it uses 8 GB

The default configuration is:

| Setting | Value | Purpose |
|---|---:|---|
| CUDA architecture | 61 | Generates code for the GTX 1080 |
| Quantized matrix multiplication | MMQ forced | Uses quantized kernels without Tensor Cores |
| Flash Attention | Built in; automatic device probe | Enables supported fused attention kernels |
| CUDA graphs | Off | The pinned engine disables them on pre-Volta GPUs |
| Context | 16,384 tokens | Shared budget for input, history, and generated tokens |
| Parallel sequences | 1 | Keeps cache use and scheduling simple |
| Batch / microbatch | 512 / 128 | Processes more prompt tokens per batch |
| K/V cache storage | f16 / f16 | Default attention-cache precision |
| GPU reserve | 768 MiB | Leaves space above the engine's estimated allocation |
| Prompt RAM cache / checkpoints | 512 MiB / 4 | Enables prefix reuse for repeated conversations |

Weights are only part of runtime memory. The engine's native `--fit` planner accounts for model tensors, context state (including hybrid recurrent state), and compute buffers. It selects how many layers to offload using currently free VRAM. Context stays at the requested size; the launcher does not silently lower it. Layers that do not fit stay in system RAM and run on the CPU.

The reserve is an allocation target, **not an operating-system-enforced VRAM limit**. Other applications can consume memory after the plan is made. During startup only, a CUDA allocation failure is retried at most twice, increasing the reserve by 512 MiB each time. Unsupported architectures and kernel/driver failures stop immediately. Failures after startup stop the service; in-flight requests are not replayed.

The launcher requires evidence of at least one offloaded layer before declaring GPU readiness. Use `--require-full-gpu` to require all offloadable layers on the card. Input embeddings and some operations may still execute on the CPU even with every offloadable layer on the GPU.

```bash
# Run 32K with the standard cache, verified on the GTX 1080.
python3 pascal.py serve --ctx 32768 --require-full-gpu

# Reduce K/V storage if another model needs more memory headroom.
python3 pascal.py serve --ctx 32768 --kv-cache q8_0 --require-full-gpu

# Leave more space if the card also drives a display.
python3 pascal.py serve --reserve-mib 1280

# Select one card on a machine with multiple GPUs.
python3 pascal.py serve --gpu 0
```

No model format or file size alone guarantees compatibility. Other **single-file GGUFs supported by this engine** can be served with `--model /path/to/model.gguf`. Safetensors, AWQ, GPTQ, split GGUFs, and multimodal projector loading are outside this launcher's current interface. This release serves text. It does not promise every model under 8 GB will fit completely in VRAM or run quickly.

## Attention acceleration and longer context

[FlashInfer supports Turing (sm_75) and newer GPUs](https://github.com/flashinfer-ai/flashinfer#gpu-support); the GTX 1080 is Pascal (sm_61). This stack integrates llama.cpp's existing generic Flash Attention kernels that have a path for GPUs without Tensor Cores. It does not contain a FlashInfer port. The pinned CUDA implementation selects vector or tile kernels for these devices; see its [attention kernel selection](https://github.com/ggml-org/llama.cpp/blob/73a43d1f69345aee8bb186ef4b3172cef892f2e5/ggml/src/ggml-cuda/fattn.cu).

`--flash-attn auto` asks the engine to check whether the loaded model's operations are supported. The launcher records the resolved result at startup. With the default f16 cache, an unsupported path can fall back to ordinary attention. The optional q8_0 cache requires Flash Attention to activate; otherwise startup fails with a diagnostic. Only matching f16/f16 and q8_0/q8_0 K/V formats are exposed, allowing a smaller CUDA build with `GGML_CUDA_FA_ALL_QUANTS=OFF`.

`--kv-cache q8_0` changes the attention cache, independently of the model's Q4 or Q8 weight quantization. It roughly halves K/V storage. For this model, the CPU engine allocated **512 MiB** of K/V at 16K with f16, **272 MiB** at 16K with q8_0, and **544 MiB** at 32K with q8_0. The model also needs recurrent state, weights, compute buffers, and cache/checkpoint storage. K/V savings do not halve total memory, and quantized attention may add dequantization work. Measure both formats on the GPU before choosing for speed.

The launcher accepts contexts up to **65,536** tokens. The GTX 1080 passed a **32,000-token input** at a 32K allocation with both cache formats; f16 was faster and is the deployed choice. The 64K setting remains untested. The model metadata advertises a larger native context, so these settings need no RoPE scaling override; useful long-context accuracy has not been evaluated. No context setting guarantees full GPU residency for every model. `--require-full-gpu` catches partial layer offload, which can substantially slow generation. The input and output must fit together; oversized requests are rejected, and automatic context shifting is disabled.

Prefix caching can reduce the work when an application resends a conversation or common document. Hybrid recurrent models also need checkpoints to restore a reusable prefix. This stack enables four checkpoints and a 512 MiB system-RAM prompt cache. Checkpoint allocations are additional to the prompt-cache budget. Applications must actually resend the shared prefix to benefit; unrelated prompts do not. These controls trade memory for reuse.

For smaller working buffers or a compatibility comparison:

```bash
python3 pascal.py serve --ctx 16384 --flash-attn off --kv-cache f16 \
  --batch 128 --ubatch 32 --cache-ram 0 --checkpoints 0
```

## Measure on your card

After building and downloading the model, stop other GPU workloads and run:

```bash
python3 pascal.py bench --ctx 16384 --prompt-tokens 2048 --generate-tokens 64 \
  --repeats 3 --output benchmark-16k.json

# Compare the two accelerated profiles at 32K.
python3 pascal.py bench --ctx 32768 --prompt-tokens 4096 --profiles flash flash-q8 \
  --output benchmark-32k.json
```

The benchmark starts and stops its own server on loopback port 18080. It compares `compat` (old batching, ordinary attention, caching disabled), `flash` (new defaults), and `flash-q8` (new defaults plus quantized K/V). Each profile uses the same model, exact prompt token IDs, context allocation, CPU thread count, and output-token count. Warm-up is excluded, cold requests explicitly disable prefix reuse, and a separate follow-up records actual prefix reuse. Comparing profiles measures the combined serving configuration, not Flash Attention in isolation.

The JSON records engine/device details, actual layer placement, resolved Flash Attention status, samples, median prompt and generation throughput, and request duration. Failed profiles are recorded and return a nonzero exit code. It rejects truncated requests, unexpected cache reuse in cold runs, and incomplete generation. Context allocation is not context occupancy: the first command allocates 16K but feeds a 2K prompt. Increase `--prompt-tokens` for a representative long document while leaving room for output and a 32-token suffix.

`--cpu` exercises the benchmark without a GPU, but its timings do not predict Pascal performance. The benchmark does not measure answer quality or peak VRAM. The hardware validation used a separate one-second GPU-memory sampler; see [VALIDATION.md](VALIDATION.md) for the measured results.

## A higher precision option

The same model is also pinned as Q8_0 (4.611 GB / 4.294 GiB):

```bash
python3 pascal.py pull qwen38-4b-q8
python3 pascal.py serve --preset qwen38-4b-q8
```

Start with Q4, which was benchmarked on the GTX 1080. Q8 reduces weight quantization at a larger memory and bandwidth cost; the Q8 weight preset has not been benchmarked on the card.

Qwen3.8-Flash-Next is not an alternative 6B-sized model: its activated parameter count does not describe stored weights. It is outside the intended memory budget of this machine.

## Optional container build

The Dockerfile uses CUDA 12.9.1 for both build and runtime. The host still needs a Pascal-compatible NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). A container cannot restore support removed from the host driver.

```bash
docker build -t pascal-serve --build-arg BUILD_JOBS=2 .
mkdir -p models state
docker run --rm -v "$PWD/models:/models" pascal-serve pull
docker run --rm --init --gpus all --network host \
  -v "$PWD/models:/models:ro" -v "$PWD/state:/state" pascal-serve serve
```

These commands are for Linux. Host networking keeps the service on the host's loopback address; it does not publish a public port. `--init` forwards signals to the launcher. The model is not baked into the image.

## Diagnostics and development

Logs are written under `.pascal/logs`. Each successful start also produces `ready-PID.json` with the actual engine version, configured source pin, device, offloaded layer count, context, reserve, attention/cache/batch settings, and log path. These are run records and can be stale after shutdown. `PASCAL_SERVER=/path/to/llama-server` allows an external engine for testing. Its version is recorded separately from the configured source pin; equivalent flags and architecture support are the caller's responsibility. Use the bundled build command for the intended CUDA configuration.

```bash
python3 -m unittest discover -s tests -v
python3 pascal.py build --cpu --jobs 2
python3 pascal.py serve --cpu
```

The CPU mode is explicit and intended for diagnostics; it does not test GPU kernels. Use `--state-dir` and `--models-dir` **before** the subcommand to relocate generated data.

See [VALIDATION.md](VALIDATION.md) for what was actually verified and what remains to test on the GTX 1080.

## Source pins and attribution

- llama.cpp: [`73a43d1f69345aee8bb186ef4b3172cef892f2e5`](https://github.com/ggml-org/llama.cpp/tree/73a43d1f69345aee8bb186ef4b3172cef892f2e5), MIT license. [CUDA build documentation](https://github.com/ggml-org/llama.cpp/blob/73a43d1f69345aee8bb186ef4b3172cef892f2e5/docs/build.md).
- Model: [Empero Qwen3.8-4B Distill GGUF](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF/tree/391fc7d103e3942a408def3e4f51c2f85d464417), Apache-2.0 per the publisher. File revisions, byte sizes and hashes are in `models.lock.json`. No model weights or upstream source are bundled in this package.
- [NVIDIA Pascal / CUDA toolchain support guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers).

Pins are intentional. Updating to another engine commit requires checking the CUDA build and command-line options again; the launcher never auto-upgrades it.
