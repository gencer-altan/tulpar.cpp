# TODO.md

Active and upcoming optimization tasks for the tulpar.cpp fork. Every task lists its evidence base and the gate it must pass. Projected gains are labeled PROJECTED; nothing below is a measured result until the gate passes.

## Active (Phase 3: GEMV decode latency)

### T-1. Software-pipeline the IQ3_XXS K-loop (hoist next-trip loads)

- Evidence: Phase-3 diagnosis - no cross-trip overlap; each trip drains to `vmcnt(0)`; 76.7% of wave-cycles in instruction-wait stalls (`experiments/phase3/phase3_report.md` section 1.3).
- Approach: template bucketing of `blocks_per_row` or manual double-buffering of X/Y loads so the next trip's loads issue before the current trip consumes.
- First target: synthetic harness (`experiments/phase3/synthetic/mmvq_bench.hip`), then in-tree with the full gate.
- PROJECTED: reduction of the per-trip stall serialization. No number is committed; the gate decides.

### T-2. Two independent dependency chains per thread

- Evidence: synthetic fused gateup b1->b2 = 184 -> 304 GB/s (+65%) purely from a second independent chain (`experiments/phase3/phase3_report.md` section 2.2).
- Approach: process two kbx groups (or two output rows) per thread to create two independent chains without touching the numerical path of a single dot product.
- PROJECTED: multi-hundred GB/s upside in-model if the b1->b2 effect transfers. Gate required.

### T-3. nwarps=2/4 calibration sweep for IQ3_XXS on RDNA3

- Evidence: `mmvq.cu` RDNA3 table sends all non-K-quant types to nwarps=1 at ncols_dst==1; Phase-3 calls this a structural (indirect) contributor, not the direct mechanism (occupancy parity measured).
- Approach: table change only; watch VGPR pressure and spills. Treat as calibration.
- PROJECTED: uncertain gain; low effort.

### T-4. Decode long-tail gap stall reduction (43% of decode wall is gap)

- Evidence: EXP-005 - median gap 8.4 us, p99 180-187 us, max 15-17 ms per 64-token generation; 43% of decode wall time is gap.
- Evidence (negative): EXP-007 rejected - dispatch-count reduction (RMSNorm+Quantize+GEMV fusion) cannot address long-tail stalls; upper bound ~9% of wall, below the 10% threshold.
- Approach: first instrument the stall source (graph re-launches, scratch buffer allocations, or specific large-kernel boundaries); a dedicated EXP with root-cause evidence before any kernel change.
- PROJECTED: unknown until sourced. Do not re-propose dispatch-count fusion.

### T-5. Prefill pp recovery (shape-gated hybrid dispatch)

- Evidence: PATH A accepted tradeoff - fresh prefill -0.70% to -3.56% at 1k-131k (worst -3.56% @128k), `experiments/phase2/phase2_benchmark.md`.
- Approach: shape-gated hybrid dispatch: keep fused q4_0 tile for decode, restore staged path for large-ncols prefill tiles where the nibble-unpack cost exceeds the staging savings.
- PROJECTED: recover -2% to -3.6% pp regression at 63k-131k without losing the +17% to +29% decode gains.

### T-6. Resolve the EXP-006 verdict record

- Evidence: CONFLICTING - EXP-006 entry says FAILED/REVERTED (average +4.6% < 15% threshold) while `f7c4436c3`/`068f581e2` restored the change claiming +12.97% @63k (MTP OFF). Both are recorded in PERFORMANCE.md section 4.1.
- Approach: add a follow-up EXP entry documenting the restore decision (why T14 standalone was kept) with the 1k/16k/63k before/after and the 131k re-measurement.
- No code change; documentation/closure task.

### T-7. 128k-131k VRAM margin watch

- Evidence: EXP-005 - V2 at 128k uses 14.29 GiB (1.2 GiB margin under the 15.5 GiB guard); PATH A 131k peak 13.70 GiB.
- Approach: track per-experiment VRAM peaks; any variant that raises peak VRAM at 128k-131k must justify itself against the guard.
- No projected gain; guardrail task.

### T-9. EXP-023 full-model wall before/after (Phase-3 gate) + H13 occupancy falsification

- Evidence: EXP-023 (sign fastpath, `628507055`) shows op-level -50%/-55% (n=1) and PMU SQ_INSTS_VALU -69.9% per dispatch, but full-model wall before/after is missing (build-baseline contaminated: mmvq.cu.o byte-identical to build-patched; GPU occupied by a running production server).
- Approach: (1) restore clean vecdotq.cuh and incrementally rebuild build-baseline; (2) Phase-3 gate benchmark (MTP OFF, 5 reps, fixed seed, 256 toks, 1k/16k/63k ctx) on baseline and patched; (3) rocprofv3 occupancy check (resident waves/SIMD) on the patched build as the H13 falsification test; (4) restore the patch and record the EXP-023 follow-up numbers.
- Gate: fresh process per arm, >= 3 reps, spread < 1%, correctness gates PASS. If the wall gain is below the 15% threshold or occupancy regresses, H13 is falsified and the change is revisited.
- PROJECTED: unknown until measured.

### T-10. Fix fattn-prefill-d256-rdna3 hard assert crashing MTP draft context at load

- Evidence (phase-6, 2026-09-26): production restore per manifest (build-p3 + `--spec-type draft-mtp`) aborts at load: `fattn-prefill-d256-rdna3.cu:392: GGML_ASSERT(K->type == GGML_TYPE_Q4_0) failed`, reached via `common_context_can_seq_rm(ctx_dft)` (server-context.cpp:1194) probing the MTP draft context with a 2-token decode. Reproduced on build-p3, build-patched, and build/bin (segfault). MTP OFF loads and runs cleanly on all current builds. Regression window: fattn-rdna3 Phase 1-3 (`bc474fe7d`, `72431fd53`, `f3f27aa41`, 2026-09-17); last successful MTP production run 2026-08-26. Production currently runs MTP OFF on build-patched as a documented deviation.
- Approach: in the `ggml_cuda_flash_attn_ext_prefill_d256_rdna3` entry check, turn the K/V `GGML_ASSERT(type == Q4_0)` pair (and the `nb[0]` stride asserts) into capability `return false` so the fattn.cu:587 dispatcher falls back to the TILE kernel for non-q4_0 KV (the MTP draft context). Full correctness gates + an MTP-ON load test + draft-acceptance check required.
- Gate: MTP-ON server loads and generates with q4_0 target KV; greedy A/B on target outputs unchanged vs TILE path on a small ctx; production restorable per manifest verbatim.
- PROJECTED: restores the documented production config (MTP ON); no decode-path perf change expected on q4_0 KV (assert only ever fires on the draft context path).

## Later (deferred)

### T-8. Weight-layout repack (swizzle) for contiguous grid indices

- Evidence: Phase-3 recommendation 5 - largest engineering cost; defers until T-1/T-2 under-deliver.
- PROJECTED: only relevant if the gather chain survives T-1/T-2.

## Done (reference)

- EXP-025 GQA<6> head batching for decode attention (default ON, opt-out `GGML_FA_DECODE_GQA_BATCH_OFF=1`) - `3165bee7e` (MEASURED 15.30 -> 19.59 t/s @131k, FA decode 37.3 -> 22.13 ms/tok).
- Tile FA for quantized KV decode @hsk 256 - `66dcba5eb` (MEASURED +39.5% @63k).
- PATH A fused q4_0 KV staging elimination - `2e033a696` (MEASURED +28.79% @128k).
- EXP-006 gather hoist restored - `f7c4436c3`, `068f581e2` (MEASURED +12.97% @63k; see T-6 for the record conflict).
- Phase-1 baselines (V3), Phase-1B matrix, Phase-2A ledger, Phase-2B inventory, Phase-3 diagnosis, Phase-4A V1 (negative), Phase-4E variants (negative/marginal), Phase-5 WMMA finding, EXP-001/EXP-002 infrastructure.
- Phase-4A V1 LDS grid table: negative result, closed.
- Phase-4E E-variants (independent accumulators): MARGINAL, closed (do not adopt).
- EXP-007 fusion: rejected at feasibility, closed.

## Gates (apply to all performance work)

- Correctness: `test-backend-ops -o MUL_MAT` (and `-o FLASH_ATTN_EXT -p q4_0`) all PASS.
- Deterministic greedy suite 12/12 (MTP ON and OFF).
- MTP acceptance unchanged within run-to-run spread.
- Fresh instance per arm, fixed seed, >= 3 reps, spread < 1%.
- Single change, single commit; EXP entry in `experiments/EXPERIMENT_LOG.md`.
