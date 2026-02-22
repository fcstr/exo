# ARM64 Performance Optimization for LlamaCpp Runner

This document covers the runtime tuning applied to the llama-cpp-python `Llama()` constructor
for ARM64 targets (aarch64 Linux, Android/proot), and compilation-level optimizations that
can further improve throughput.

---

## Runtime tuning (current)

All parameters are set in `runner.py` at model load time and can be overridden via
environment variables.

### Thread configuration

| Parameter | Default | Env var | Rationale |
|-----------|---------|---------|-----------|
| `n_threads` | `cpu_count // 2` | `EXO_LLAMACPP_N_THREADS` | Generation (autoregressive decode) is memory-bandwidth-bound. Fewer threads reduce cache contention and inter-core traffic on ARM big.LITTLE SoCs. |
| `n_threads_batch` | `cpu_count` | `EXO_LLAMACPP_N_THREADS_BATCH` | Prompt processing (prefill) is compute-bound and embarrassingly parallel — use all cores. |

On a typical 8-core phone SoC (4 big + 4 little), this gives 4 threads for decode and
8 for prefill. For devices where little cores hurt more than help, set
`EXO_LLAMACPP_N_THREADS=2` and `EXO_LLAMACPP_N_THREADS_BATCH=4` to pin to big cores only.

### Flash attention

| Parameter | Default | Env var |
|-----------|---------|---------|
| `flash_attn` | `True` | `EXO_LLAMACPP_FLASH_ATTN` (`"1"`/`"0"`) |

Flash attention fuses the Q·K^T·V computation into a single pass, cutting memory reads
by ~2x during the attention step. On ARM64, where memory bandwidth is the primary
bottleneck during decode, this is one of the highest-impact single flags.

### KV cache quantization

| Parameter | Default | Env var | GGML type |
|-----------|---------|---------|-----------|
| `type_k` | `8` | `EXO_LLAMACPP_TYPE_K` | `GGML_TYPE_Q8_0` |
| `type_v` | `8` | `EXO_LLAMACPP_TYPE_V` | `GGML_TYPE_Q8_0` |

By default llama.cpp stores KV cache entries in FP16 (2 bytes/element). Q8_0 uses ~1
byte/element, halving KV cache memory. This has two benefits:

1. **Fits larger contexts** — a 4096-context 0.5B model drops from ~256 MB to ~128 MB of KV cache.
2. **Reduces memory traffic** — every attention step reads the full KV cache, so halving its size directly speeds up decode.

The quality impact of Q8_0 KV quantization is negligible for most models. For aggressive
memory savings, `GGML_TYPE_Q4_0` (type `2`) can be used but may degrade output quality.

Common GGML type IDs for reference:

| ID | Type | Bytes/element |
|----|------|---------------|
| 0  | F32  | 4.0 |
| 1  | F16  | 2.0 |
| 8  | Q8_0 | ~1.1 |
| 2  | Q4_0 | ~0.6 |

### Batch / micro-batch size

| Parameter | Default | Env var |
|-----------|---------|---------|
| `n_batch` | `512` | `EXO_LLAMACPP_N_BATCH` |
| `n_ubatch` | `512` | `EXO_LLAMACPP_N_UBATCH` |

`n_batch` controls the maximum number of tokens processed in a single batch during
prompt evaluation. `n_ubatch` controls the micro-batch size — the number of tokens
processed per internal iteration within a batch. Both default to 512, a good balance
between memory usage and GEMM efficiency. Increase to 1024 or 2048 if you have RAM
headroom and long prompts.

### Memory mapping and prefetch

| Parameter | Value | Notes |
|-----------|-------|-------|
| `use_mmap` | `True` | Maps the GGUF file into memory. Good for proot — lets the OS page in weights on demand and share pages across processes. |
| `use_mlock` | `False` (default) | proot cannot call `mlock()`. Attempting it would fail silently or error. |
| `madvise` prefetch | Automatic | Before loading, `MADV_SEQUENTIAL` + `MADV_WILLNEED` are called on the GGUF file to trigger kernel read-ahead into page cache, reducing page-fault stalls during model init. |

### CPU affinity (big.LITTLE)

On ARM big.LITTLE SoCs (common in Android phones), the runner automatically detects
big vs little cores via `/sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq` and
pins itself to the big (high-performance) cores. This prevents the OS scheduler from
migrating inference threads to power-efficient little cores mid-computation.

Falls back silently if sysfs is unavailable (e.g. proot without `/sys` bind).

### Context size

| Parameter | Default | Env var |
|-----------|---------|---------|
| `n_ctx` | `4096` | `EXO_LLAMACPP_N_CTX` |

Maximum sequence length. Larger values consume more KV cache memory (linearly). For
resource-constrained devices, reduce to 2048 or 1024.

---

## Compilation-level optimizations (further work)

The pre-built `llama-cpp-python` wheel from PyPI is compiled with generic settings.
Building from source with platform-specific flags can yield significant speedups.

### 1. Native ARM NEON/ASIMD/i8mm/SVE compilation

The current device exposes these CPU features (from `/proc/cpuinfo`):

```
fp asimd aes pmull sha1 sha2 crc32 atomics fphp asimdhp
asimdrdm i8mm bf16 asimddp sha512 asimdfhm dotprod
```

To build llama-cpp-python with native tuning:

```bash
CMAKE_ARGS="-DCMAKE_C_FLAGS='-mcpu=native' -DCMAKE_CXX_FLAGS='-mcpu=native'" \
  pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall
```

This enables:
- **NEON/ASIMD**: 128-bit SIMD (already used by default on aarch64, but `-mcpu=native` unlocks device-specific scheduling)
- **i8mm** (`FEAT_I8MM`): Hardware int8 matrix multiply instructions — llama.cpp's Q4_0/Q4_K/Q8_0 GEMM kernels use `smmla`/`ummla` when available, giving ~2x speedup on quantized matmuls vs pure NEON dot-product path
- **bf16** (`FEAT_BF16`): Native bfloat16 operations for BF16 GGUF models
- **dotprod** (`asimddp` / `FEAT_DOTPROD`): 4-element dot product instructions, used in Q4/Q8 kernels

### 2. SVE/SME (if available)

Some newer ARM cores (Cortex-X3/X4, Neoverse V2) support SVE or SVE2:

```bash
CMAKE_ARGS="-DGGML_SVE=ON -DCMAKE_C_FLAGS='-mcpu=native -msve-vector-bits=128'" \
  pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall
```

SVE provides scalable vector operations that can be wider than NEON's fixed 128-bit.
Check for `sve` in `/proc/cpuinfo` Features before enabling.

### 3. OpenBLAS / BLAS backend

For prompt processing (large matrix multiplications), a tuned BLAS library can help:

```bash
apt install libopenblas-dev
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall
```

OpenBLAS has hand-optimized ARM64 kernels. This primarily speeds up prompt evaluation
(prefill), not autoregressive decode. However, for quantized models (Q4/Q8), llama.cpp's
built-in quantized GEMM kernels are often faster than dequantize→BLAS→FP32, so benchmark
before committing to this.

### 4. Arm Compute Library (ACL)

ARM's own Compute Library provides highly optimized GEMM/GEMV for Cortex-A and Neoverse:

```bash
# Build ACL first (or install from package manager)
git clone https://github.com/ARM-software/ComputeLibrary.git
cd ComputeLibrary
scons Werror=0 neon=1 opencl=0 os=linux arch=armv8.2-a -j$(nproc)

# Then build llama.cpp against it
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=ACL" \
  pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall
```

ACL is usually the fastest BLAS option on ARM64 for FP32/FP16 matmuls.

### 5. Optimized memory allocator

The default glibc `malloc` has contention under multithreaded workloads. Using jemalloc
or mimalloc can reduce allocation overhead:

```bash
apt install libjemalloc2
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 uv run exo
```

Or at compile time:

```bash
CMAKE_ARGS="-DCMAKE_C_FLAGS='-mcpu=native' -DCMAKE_EXE_LINKER_FLAGS='-ljemalloc'" \
  pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall
```

### 6. LTO (Link-Time Optimization)

Enables cross-translation-unit inlining and dead code elimination:

```bash
CMAKE_ARGS="-DCMAKE_C_FLAGS='-mcpu=native -flto' -DCMAKE_CXX_FLAGS='-mcpu=native -flto' -DCMAKE_EXE_LINKER_FLAGS='-flto'" \
  pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall
```

Increases build time but can yield 5-15% throughput improvement.

### 7. Combined build command (recommended)

For maximum performance on this device (Cortex-A series with i8mm + bf16):

```bash
CMAKE_ARGS="\
  -DCMAKE_C_FLAGS='-mcpu=native -O3 -flto' \
  -DCMAKE_CXX_FLAGS='-mcpu=native -O3 -flto' \
  -DCMAKE_EXE_LINKER_FLAGS='-flto' \
  -DGGML_NATIVE=ON" \
  pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall
```

For uv-based projects (like exo):

```bash
CMAKE_ARGS="\
  -DCMAKE_C_FLAGS='-mcpu=native -O3 -flto' \
  -DCMAKE_CXX_FLAGS='-mcpu=native -O3 -flto' \
  -DCMAKE_EXE_LINKER_FLAGS='-flto' \
  -DGGML_NATIVE=ON" \
  uv pip install llama-cpp-python --no-binary llama-cpp-python --reinstall
```

### Expected impact

| Optimization | Estimated speedup | Status |
|-------------|-------------------|--------|
| `-mcpu=native` (i8mm/bf16/dotprod) | 1.5-2.0x | Done (source build) |
| Flash attention | 1.2-1.5x decode | Done |
| KV cache Q8_0 | 1.1-1.3x decode | Done |
| Thread tuning (decode/prefill split) | 1.1-1.2x | Done |
| madvise prefetch | Faster cold start | Done |
| CPU affinity (big.LITTLE pinning) | 1.1-1.3x | Done |
| n_ubatch tuning | 1.0-1.1x | Done |
| jemalloc | 1.05-1.10x | Available (LD_PRELOAD) |
| LTO | 1.05-1.15x | Done (source build) |
| OpenBLAS/ACL | 1.2-1.5x prefill | Not yet |

To use jemalloc at runtime:
```bash
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 uv run exo
```
