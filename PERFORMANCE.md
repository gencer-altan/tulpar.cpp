# PERFORMANCE.md

Evidence-based performance ledger for the tulpar.cpp fork (RX 7800 XT / RDNA3 / `gfx1101`).

Rules applied here:

- MEASURED: valid before/after comparison, same workload, same binary-protocol.
- ATTRIBUTION: measurement-only bottleneck identification, no before/after.
- EXPECTED: projected gain without a passing benchmark.
- UNMEASURED: code change committed without performance metrics.
- CONFLICTING: incompatible measurements. Both values are reported; they are never averaged.

Speedup formulas:

- throughput: `speedup_percent = (after - before) / before * 100` (tok/s)
- latency: `latency_reduction_percent = (before - after) / before * 100` (ms/tok)
- A claim is only a speedup when both before and after exist.

Environment for all measurements: AMD Radeon RX 7800 XT (gfx1101, 60 CU, 16 GiB), ROCm 7.2.4, rocprofv3 1.1.0, Linux (cachyos-x8664), build `build-p3/bin/llama-server` (except where noted). Sampling: greedy temp 0, top_k 1, fixed seed 1234, fresh instance per arm, 0.5 s VRAM monitor, guard 15.5 GiB sampled.

Models:

- V3: `Qwen3.8-27B-UD-Q2_K_XL.gguf` (sha256 `fd4730dd...` per `experiments/phase1/phase1_attribution.md`).
- V2: swapped in via `4f72448eb` (EXP-002), IQ3_XXS dominant (75.9% of streamed weights). Current production model.

---

## 1. MEASURED results

### 1.1 Tile flash attention for quantized KV decode, head size 256 (commit `66dcba5eb`)

Change: routes RDNA3 decode at head size 256 to `flash_attn_tile` instead of the forced `flash_attn_ext_vec` fallback (`ggml/src/ggml-cuda/fattn.cu`).

Claimed in commit message (Qwen3.8-27B, q4_0 KV, MTP OFF): decode tok/s +5% @1k, +17% @16k, +40% @63k.

Corroborated with explicit before/after (from `experiments/phase2a/evidence_ledger.md`, section 3.4): legacy 11.29 -> Phase-1B baseline 15.74 tok/s @63k = +39.5%.

Classification: MEASURED for @63k (before/after both recorded). @1k and @16k are commit-message percentages without surviving before/after numbers in-tree: treat as MEASURED-without-surviving-raws.

### 1.2 PATH A: fused q4_0 KV dequant, staging eliminated (commit `2e033a696`)

Change: `flash_attn_tile` reads q4_0 K/V blocks directly (planar nibbles, block-dense strides); whole-cache F16 staging removed. Fused variants instantiated only for DKQ==DV==256.

Source: `experiments/phase2/phase2_benchmark.md` (MTP OFF, P1 baseline = Phase-1 untraced ladder, V3 model).

| arm | before (P1) tok/s | after (P2) tok/s | throughput | before ms/tok | after ms/tok | latency reduction |
|-----|-------------------:|------------------:|-----------:|---------------:|--------------:|------------------:|
| 1k | 23.481 | 23.384 | -0.41% | 42.59 | 42.76 | -0.42% (noise) |
| 16k | 21.594 | 22.419 | +3.82% | 46.31 | 44.61 | +3.67% |
| 63k | 16.477 | 19.279 | +17.01% | 60.69 | 51.87 | +14.54% |
| 128k | 12.507 | 16.108 | +28.79% | 79.95 | 62.08 | +22.35% |
| 131k | 12.481 | 16.104 | +29.03% | 80.12 | 62.10 | +22.50% |

Fresh prefill (pp) regression, same file:

| arm | before pp | after pp | delta |
|-----|----------:|---------:|------:|
| 1k | 518.17 | 519.84 | +0.32% |
| 16k | 494.42 | 490.98 | -0.70% |
| 63k | 339.66 | 331.89 | -2.29% |
| 128k | 234.39 | 226.05 | -3.56% |
| 131k | 232.57 | 224.86 | -3.32% |

Staging verification (`experiments/phase2/phase2_staging_verification.md`, traced, 131k, MTP OFF):

| metric | P1 baseline | P2 fused |
|--------|------------:|---------:|
| `dequantize_block_q4_0` launches | 32/token | 0 |
| staging dequant ms/tok | ~19.65 | 0 |
| `flash_attn_tile` ms/tok | 16.960 | 18.903 |
| net attention ms/tok | ~36.61 | 18.90 (-17.71) |

Cross-check (same file): removing ~17.7 ms/tok from the Phase-1 82.8 ms step period predicts ~15.4 tok/s; measured untraced 16.108 tok/s @128k. Consistent.

Success criteria (file): 128k target >= +15%: PASS (measured +28.79%, 16.108 > 14.4). 131k VRAM under 15.5 GiB guard: PASS (13.70 GiB).

### 1.3 EXP-006: `iq3_xxs` GEMV gather hoist, restored (commits `f7c4436c3`, `068f581e2`)

Change: hoist 8 `iq3xxs_grid` lookups + 4 sign computations + 8 Q8_1 reads before the dp4a chain in `vec_dot_iq3_xxs_q8_1` (`ggml/src/ggml-cuda/vecdotq.cuh`). Bit-exact, zero numerical drift.

Decode speed (MTP OFF, 5 reps, fixed seed, 256 toks, cached; from EXP-006 in `experiments/EXPERIMENT_LOG.md`, V2 model):

| ctx | before ms/tok | after ms/tok | recorded gain |
|-----|--------------:|--------------:|--------------:|
| 1k | 45.32 | 45.64 | -0.71% (noise) |
| 16k | 48.84 | 48.07 | +1.59% |
| 63k | 63.15 | 54.96 | +12.97% |
| average | | | +4.6% |

Decode speed (MTP ON, from the same entry):

| ctx | before ms/tok | after ms/tok | recorded gain |
|-----|--------------:|--------------:|--------------:|
| 1k | 25.02 | 23.13 | +7.55% |
| 16k | 28.38 | 26.11 | +8.01% |
| 63k | 26.97 | 26.54 | +1.59% |

Prefill regression check (same entry): 1k 479.68 -> 478.06, 16k 471.72 -> 478.25, 63k 321.36 -> 320.47 tok/s. No regression.

Correctness gates (same entry + `experiments/EXP-006-restore/correctness_patched.log`): `test-backend-ops -o MUL_MAT -t iq3_xxs` 11/11 PASS (log shows `11/11 tests passed`), all quant types no regression, deterministic greedy suite 12/12 (MTP ON and OFF), 9/12 short-context tests bit-exact vs `results2/correctness/baseline_v2.json`.

Restored baseline reference (V2, 131k, from `experiments/EXP-006-restore/`): MTP OFF tg 15.164 tok/s (65.944 ms/tok), MTP ON tg 23.66 tok/s.

### 1.4 MTP ON vs MTP OFF on V2 baseline (config comparison, measured)

From `experiments/v2_baseline/summary/table_v2.json` (V2 model, q4_0 KV, fresh process per arm):

| ctx | OFF tok/s | ON tok/s | ON vs OFF | MTP acceptance |
|-----|----------:|---------:|-----------:|---------------:|
| 1k | 22.214 | 26.8945 | +21.1% | 0.8435 |
| 16k | 21.112 | 32.729 | +55.0% | 0.8861 |
| 63k | 18.31 | 29.39 | +60.5% | 0.8691 |
| 128k | 15.557 | 27.047 | +73.9% | 1.0 |
| 131k | 15.581 | 22.776 | +46.2% | 0.9349 |

This is a configuration comparison (MTP draft on/off), recorded for context; it is not an optimization gain.

### 1.5 Gap to the 40 tok/s target

Target: 40 tok/s decode. Gap computed as `(target - current) / current * 100` from the current V2 baseline (section 1.4, MTP OFF):

| ctx | current tok/s | required increase |
|-----|--------------:|------------------:|
| 1k | 22.214 | +80.1% |
| 16k | 21.112 | +89.5% |
| 63k | 18.31 | +118.5% |
| 128k | 15.557 | +157.1% |
| 131k | 15.581 | +156.7% |

Historical record (Phase-1B era, V3 model, from `experiments/phase2a/evidence_ledger.md`): +81% @1k, +96% @16k, +154% @63k, +234% @128k.

### 1.6 EXP-023: HIP sign-application fastpath for IQ3_XXS/IQ3_S (commit `628507055`)

Change: `iq_apply_sign4()` (HIP-only, `ggml/src/ggml-cuda/vecdotq.cuh`) replaces the scalarized `__vcmpne4`/`__vsub4` byte-wise sign chain in `vec_dot_iq3_xxs_q8_1` and `vec_dot_iq3_s_q8_1` with a native `v_mul_u32_u24_e32` + xor/add. Bit-exact; non-HIP (CUDA) path unchanged.

Op-level (MEASURED, test-backend-ops perf, m=4096 k=14336, same workload both sides):

| quant   | n    | baseline us/run | patched us/run | delta   |
|---------|------|----------------:|----------------:|---------|
| iq3_xxs |    1 |           114.28 |            51.33 | -55.1%  |
| iq3_xxs |    2 |           121.89 |            64.96 | -46.7%  |
| iq3_xxs |    3 |           130.77 |            79.57 | -39.2%  |
| iq3_xxs |    4 |           141.78 |            90.34 | -36.3%  |
| iq3_xxs |    5 |           153.28 |           100.83 | -34.2%  |
| iq3_xxs |    8 |           199.40 |           138.92 | -30.3%  |
| iq3_xxs |  512 |          2380.69 |          2382.77 | +0.1% (GEMM, flat) |
| iq3_s   |    1 |           113.65 |            57.32 | -49.6%  |
| iq3_s   |    2 |           120.74 |            70.05 | -42.0%  |
| iq3_s   |    3 |           128.02 |            80.82 | -36.9%  |
| iq3_s   |    4 |           138.55 |            89.31 | -35.5%  |
| iq3_s   |    5 |           149.66 |           102.53 | -31.5%  |
| iq3_s   |    8 |           221.26 |           147.83 | -33.2%  |
| iq3_s   |  512 |          2419.66 |          2417.26 | -0.1% (GEMM, flat) |

Per-dispatch HW counters (rocprofv3 PMU, `mul_mat_vec_q<(ggml_type)18, 1, ...>`): duration 105.20 -> 45.34 us (-56.9%), SQ_INSTS_VALU 21.01M -> 6.32M (-69.9%), GRBM_GUI_ACTIVE 276,607 -> 124,819 (-54.9%).

Kernel resources (same trace): VGPR type18 n=1 80 -> 48, type21 n=1 80 -> 40; all type18/type21 instantiations reduced; type29 unchanged. SGPR 128, LDS 0, scratch 0 (both builds).

Classification: MEASURED (op-level, same protocol). The full-model wall before/after (Phase-3 gate: MTP OFF, 5 reps, fixed seed, 256 toks) was NOT measured this run: build-baseline was rebuilt against the patched tree (contaminated) and the GPU is occupied by a running production server. Tracked as T-9.

Production single sample (USER-REPORTED, no surviving log, NOT a before/after pair): 31.62 tok/s decode (2972.53 ms), prefill 441.41 tok/s, ctx ~1330-1425, V2 model, patched build. MTP state of the run unknown.

Reference numbers (USER-REPORTED, NOT measured in this commit): stock kernel ~18 tok/s; user's own decode kernel ~22 tok/s; Vulkan ~31-33 tok/s (1k ctx, MTP off).

---

## 2. ATTRIBUTION (measurement-only)

### 2.1 Phase-1 full decode attribution, V3 model (commit `ca49ee25b`; `experiments/phase1/phase1_attribution.md`, twin `phase1_attribution.json`)

Decode attention-side cost per token (MTP OFF, kernel busy share):

| context | total busy ms/tok | staging dequant | FA tile | KV store |
|---------|------------------:|----------------:|---------:|----------:|
| 1k | 40.986 | 0.000 (0.0%) | 4.464 (10.9%) | 0.017 (0.0%) |
| 16k | 41.517 | 1.424 (3.4%) | 10.647 (25.6%) | 0.086 (0.2%) |
| 63k | 43.061 | 8.744 (20.3%) | 15.053 (35.0%) | 0.174 (0.4%) |
| 128k | 43.740 | 19.646 (44.9% of busy* see note) | 17.031 (38.9%) | 0.210 (0.5%) |

Note: the report's headline shares (staging 26.4% and tile 22.9% of busy @128k) are computed against busy+gap-normalized period shares; the table above is raw ms/tok. Both forms appear in the report; ms/tok values are quoted from the report table.

Decode dispatches/token: 599 (1k) -> 629 (128k). Effective aggregate GEMV bandwidth @128k: 265.6 GB/s (42.6% of nominal 624 GB/s).

Per-quant effective GEMV bandwidth @128k (V3):

| quant | GB/s | % of 624 |
|-------|-----:|---------:|
| IQ3_XXS | 261 | 41.8% |
| IQ3_S | 285 | 45.7% |
| Q3_K | 336 | 53.9% |
| Q2_K | 400 | 64.1% |
| Q4_K | 540 | 86.5% |
| IQ2_S | 222 | 35.6% |
| IQ2_XS | 200 | 32.1% |
| IQ2_XXS | 183 | 29.3% |
| IQ1_S | 368 | 58.9% |
| IQ1_M | 404 | 64.7% |
| IQ4_XS | 551 | 88.3% |
| F16 | 615 | 98.6% |

Decode wall (untraced, V3): OFF 23.481 / 21.594 / 16.477 / 12.507 / 12.481 tok/s at 1k/16k/63k/128k/131k; ON (MTP) 40.692 / 44.181 (r2 median) / 31.908 / 24.795 / 24.994. Traced decode @128k OFF: 12.080 tok/s vs untraced 12.507 (3.6% traced overhead).

Untraced ON deviation: U-16k-ON-r1 43.814 vs r2 44.181 (-0.8%), recorded in `experiments/phase1/phase1_blockers.md`.

### 2.2 Clean V2 Phase-2-only attribution baseline (EXP-005 in `experiments/EXPERIMENT_LOG.md`)

Reset baseline after dropping two Phase-3 commits (`1d50f66ff`, `11c95eb56`), V2 model, MTP OFF, 4 arms (rocprofv2-wrapped server per arm):

| arm | decode kernel ms | FA_tile % | GEMV_IQ3_XXS % | staging dequant |
|-----|-----------------:|----------:|---------------:|----------------:|
| T-V2-1k-OFF | 2554.4 (64 tok) | 0.6% | 74.1% | 0 |
| T-V2-16k-OFF | 2592.2 | 0.9% | 73.7% | 0 |
| T-V2-63k-OFF | 2602.1 | 2.2% | 72.8% | 0 |
| T-V2-128k-OFF | 2653.7 | 3.9% | 71.5% | 0 |

Prefill FA_tile share: 0.7% (1k) -> 12.1% (128k); MMQ_IQ3_XXS 74.4% -> 64.4%.

Bandwidth @128k (V2): decode GEMV_IQ3_XXS 350 GB/s (56% of nominal 624); decode GEMV_Q3_K (lm_head) 503 GB/s (80%).

Decode dispatches/token: 1838-1859 (all 4 arms, graph fully reused). Per-token decode: ~40 ms kernel-sum vs ~58 ms wall; inter-dispatch gap = 43% of decode wall (median gap 8.4 us, p99 180-187 us, max 15-17 ms per 64-token generation).

VRAM peaks (under 15.5 GiB guard): 11.92 / 12.20 / 13.05 / 14.29 GiB at 1k/16k/63k/128k.

Decode kernel total grows 4% from 1k to 128k entirely in FA_tile (KV-cache-size dependent); all other decode categories constant.

### 2.3 Phase-3 GEMV root-cause diagnosis (`experiments/phase3/phase3_report.md`)

Verdict: DIAGNOSIS_COMPLETE. Root cause: IQ3_XXS MMVQ GEMV is memory-LATENCY bound, not DRAM-bandwidth bound, not directly occupancy-starved.

Key measured evidence:

- HW counters (rocprofv3 --pmc): IQ3_XXS spends 76.7% of wave-cycles in instruction-wait stalls vs 2.0% for Q4_0; resident waves/SIMD parity (5.44 vs 5.64); L2 hit 6.1% (pure streaming, one cold read per dispatch, no redundant DRAM traffic).
- Kernel metadata: `mul_mat_vec_q<IQ3_XXS,1,...>` at 76 VGPRs (allocates to 80) -> ~6 waves/SIMD max; Q4_0 at 18 VGPRs. No spills.
- ISA: 452 instructions/trip for IQ3_XXS vs 76 for Q4_0; 14 global loads/trip, 8 of them dependent b32 gathers into the 1 KB `iq3xxs_grid` table; no cross-trip software pipelining (vmcnt(0) drain per trip).
- Synthetic harness (ISA-diffed against production): IQ3_XXS 165-184 GB/s weight-only vs Q4_0 1068 GB/s at batch=1 (cache-inflation caveat recorded). Batch scaling fused gateup: 184 -> 304 GB/s b1->b2 (+65%), then decays (260 @b4, 210 @b8): the ILP effect of a second independent chain.

### 2.4 Negative and marginal variant results (measurement-only)

- Phase-4A V1 (LDS-staged grid table, `experiments/phase4a/phase4a_report.md`): V1_BIT_EXACT_BUT_SLOW. Bit-exact, does not reduce the stall ratio, no GB/s gain; LDS bank-conflict rate 62-66% measured.
- Phase-4E subagent variants (`experiments/phase4e/`): prefetch-distance, VDR-increase (VDR=2 is optimum), independent accumulators (E1/E2/E3: MARGINAL, do NOT adopt).
- Phase-4 grid analysis (`experiments/phase4grid/grid_analysis_report.md`): `iq3xxs_grid` is 256 tuned centroid quadruples over 8 symbols; table is not the misbehavior source.
- Phase-5 (`experiments/phase5/subagent_g/README.md`): RDNA3 MMQ for IQ3_XXS is a genuine WMMA tensor-core path, but GEMM routing for MTP verify is NOT viable: MMQ costs +30%/+50% more per matmul than MMVQ at ne11=4-5.

### 2.5 Phase-1B baseline era (V3, pre-PATH A)

From `experiments/phase2a/evidence_ledger.md` (binary at `66dcba5eb`): TG MTP OFF 22.055 / 20.451 / 15.740 / 11.981 tok/s at 1k/16k/63k/128k; MTP ON 25.788 / 30.626 / 31.276 / 20.977 (acc 0.79/0.85/0.97/0.95). Historical legacy vs current @63k: 11.29 -> 15.74 (+39.5%) = the merged tile-FA gain.

Note: this era differs from the Phase-1 ladder (section 2.1) by measurement protocol era. The PATH A before/after pair (section 1.2) uses the Phase-1 ladder, which is the direct same-protocol baseline of the Phase-2 benchmark. Do not mix eras.

---

## 3. UNMEASURED (committed without performance metrics)

- `92f0807df` (EXP-000): dynamic repo-root resolution, experiment log standard, fork-only policy. Hygiene only.
- `966f25454` (EXP-001): benchmark driver failure classification (HEALTH_FAIL, VRAM_MONITOR_FAIL) and VRAM monitor hardening; plus line-level audit of the fused q4_0 loader (no defect found). Robustness only.
- `ca49ee25b`: Phase-1 profiling baseline commit (the measurement itself, not an optimization).
- `4f72448eb` (EXP-002): V2 model swap + baseline measurement (baseline, not a code optimization).

---

## 4. CONFLICTING (both values recorded; never averaged)

### 4.1 EXP-006 verdict vs restored-in-tree

- EXP-006 entry in `experiments/EXPERIMENT_LOG.md`: DECISION "+4.6% average decode speedup is well below the protocol's 15% success threshold. Change REVERTED." VERDICT "FAILED (insufficient gain)".
- Commits `f7c4436c3` + `068f581e2`: the same change is restored to master ("T14 standalone"), claiming "Measured: +12.97% TG at 63k context (MTP OFF)", "Bit-exact, zero numerical drift", "Zero occupancy cost on gfx1101 (HW-capped 16 waves/CU)".

Both are recorded. Current master contains the restored change. The long-context gain (+12.97% @63k) and the short-context noise (-0.71% @1k) coexist in the same before/after set; the +4.6% average is the mean of that set.

### 4.2 131k VRAM peak for PATH A

- Commit `2e033a696` message: "VRAM peak at 131k 13.70 GiB (baseline 13.30)" = +0.40 GiB vs baseline.
- `experiments/phase2/phase2_benchmark.md` reading section: "Peak VRAM at 128k/131k is slightly BELOW baseline".

The table in the same file shows 128k 13.33 -> 13.27 (below) and 131k 13.30 -> 13.70 (above). The "below baseline" statement is only true for 128k. Both the table and the commit message agree on the raw values: 13.70 GiB @131k vs 13.30 GiB baseline.

### 4.3 IQ3_XXS effective bandwidth across eras

- Phase-1 (V3 model): IQ3_XXS 261 GB/s, aggregate 265.6 GB/s @128k.
- EXP-005 (V2 model): decode GEMV_IQ3_XXS 350 GB/s.
- Phase-3 (V2-era production): ~262 GB/s in the in-model kernel.

These are different models (V3 vs V2) and/or different measurement layers (in-model vs isolated). They are not averaged; each is cited to its own artifact.

---

## 5. EXPECTED / PROJECTED (no passing benchmark yet)

Labeled EXPECTED; none of these has a committed before/after result:

- Recover the PATH A prefill regression (-2% to -3.6% pp @63k-131k) via shape-gated hybrid dispatch (recorded as a future tuning target in `experiments/phase2/phase2_benchmark.md`).
- Phase-3 ranked GEMV recommendations not yet benchmarked in-tree: software-pipeline the K-loop (hoist next-trip loads), two independent chains per thread (synthetic b1->b2 +65% evidence, `experiments/phase3/phase3_report.md` section 4), nwarps=2/4 calibration sweep.
- Reduce the decode long-tail gap stall (max 15-17 ms per 64-token generation; 43% of decode wall is gap, EXP-005). EXP-007 (kernel fusion) was rejected at feasibility: dispatch-count reduction cannot address long-tail stalls (upper bound ~9% of wall, below the 10% threshold).
- Weight-layout repack (swizzle) to make grid indices contiguous: deferred, largest engineering cost.
