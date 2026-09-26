# tulpar.cpp

Independent performance-optimization fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) targeting the AMD Radeon RX 7800 XT (RDNA3, `gfx1101`) for ultra-long-context (128k-131k) LLM inference in 16 GiB VRAM.

Fork: https://github.com/gencer-altan/tulpar.cpp

Scope: decode/prefill kernel-level optimization and profiling of Qwen3.8-27B-class GGUF models (hybrid gated-delta-net + full-attention architecture) with quantized KV cache (`q4_0`). All performance claims in this repository are backed by committed artifacts; every number is cited to a file or commit hash in [PERFORMANCE.md](PERFORMANCE.md).

## Verified optimization results (merged into master)

| Change | Commit | Measured gain (MTP OFF) | Status |
|--------|--------|--------------------------|--------|
| Tile flash attention for quantized KV decode, head size 256 (RDNA3) | `66dcba5eb` | +5% @1k, +17% @16k, +40% @63k decode tok/s; corroborated @63k: 11.29 -> 15.74 (+39.5%) | MEASURED |
| PATH A: fused q4_0 KV dequant inside tile FA (staging dequant eliminated) | `2e033a696` | +28.79% @128k (12.507 -> 16.108 tok/s), +29.03% @131k; `dequantize_block_q4_0` launches 32/token -> 0 | MEASURED |
| EXP-006: `iq3_xxs` GEMV gather hoist (restored, T14 standalone) | `f7c4436c3`, `068f581e2` | +12.97% @63k (63.15 -> 54.96 ms/tok), average +4.6% (see CONFLICTING note) | MEASURED |
| EXP-023: HIP sign-application fastpath for IQ3_XXS/IQ3_S GEMV | `628507055` | op-level -50% to -55% (n=1); full-model wall pending (T-9) | MEASURED (op-level) |
| EXP-025: GQA<6> head batching for decode attention (default ON) | `3e1ebebbc` | 15.30 -> 19.59 tok/s @131k (llama-bench -d 131072), FA decode 37.3 -> 22.13 ms/tok | MEASURED |

Prefill regression accepted as tradeoff of PATH A: fresh prefill -2% to -3.6% at 63k-131k (see [PERFORMANCE.md](PERFORMANCE.md)).

## Current state

- Master is the integrated optimization branch (merge `41f467c1b`): PATH A staging elimination, Phase-1 profiling baseline, Phase-2 benchmark suite, the EXP-006 GEMV hoist, the EXP-023 sign fastpath, and the EXP-025 GQA<6> decode batching (default ON) are all in-tree.
- Production model: V2 GGUF (IQ3_XXS-dominant, 75.9% of streamed weights), swapped in `4f72448eb` (EXP-002).
- Current production decode (USER-REPORTED llama-server log, 2026-09-26, ctx ~1.3k, MTP OFF): 34.18 tok/s decode (50 tokens, 29.26 ms/tok), 548.20 tok/s prefill.
- Latest measured production decode (V2, MTP OFF, from `experiments/v2_baseline/summary/table_v2.json`):

| Context | 1k | 16k | 63k | 128k | 131k |
|---------|-----|-----|-----|------|------|
| tg tok/s (MTP OFF) | 22.214 | 21.112 | 18.31 | 15.557 | 15.581 |
| tg tok/s (MTP ON) | 26.895 (acc 0.844) | 32.729 (acc 0.886) | 29.39 (acc 0.869) | 27.047 (acc 1.0) | 22.776 (acc 0.935) |

- Target: 40 tok/s decode. Gap to target from the current V2 baseline (MTP OFF): +80.1% @1k, +89.5% @16k, +118.5% @63k, +157.1% @128k. See [ROADMAP.md](ROADMAP.md).
- Known decode root cause (diagnosed, ATTRIBUTION): the IQ3_XXS GEMV kernel is memory-latency bound on the serialized `iq3xxs_grid` gather chain, 76.7% of wave-cycles in instruction-wait stalls. See [ROADMAP.md](ROADMAP.md) and [TODO.md](TODO.md) for the active plan.

## Documentation

- [PERFORMANCE.md](PERFORMANCE.md) - evidence ledger: MEASURED vs ATTRIBUTION vs EXPECTED vs UNMEASURED vs CONFLICTING, all numbers cited.
- [ROADMAP.md](ROADMAP.md) - three-phase plan with explicit projected-target labels.
- [TODO.md](TODO.md) - active and upcoming optimization tasks.
- [experiments/EXPERIMENT_LOG.md](experiments/EXPERIMENT_LOG.md) - append-only experiment log (EXP-000 ... EXP-025).
- Experiment artifacts: `experiments/phase0` ... `experiments/phase5`, `experiments/phase_v2_trace`, `experiments/v2_baseline`.

## Build and usage

This fork builds and runs like upstream llama.cpp. See upstream docs:

- Build: [docs/build.md](docs/build.md) (use `GGML_HIP=ON` for the ROCm/HIP backend; this fork's validated config additionally uses `GGML_HIP_GRAPHS=ON`, `GPU_TARGETS=gfx1101`).
- Server: [tools/server/README.md](tools/server/README.md)

Validated production flags (from `ops/manifest/prod_flags.env`): `-ngl 999 --load-mode mmap -fa on -ctk q4_0 -ctv q4_0 --cache-prompt --ctx-checkpoints 4 -t 8 -np 1`, MTP via `--spec-type draft-mtp`.
