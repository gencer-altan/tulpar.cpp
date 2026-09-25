# ROADMAP.md

Three-phase plan for the tulpar.cpp fork (RX 7800 XT, RDNA3, `gfx1101`). Labels: DONE (merged + measured or verified), ATTRIBUTION (measurement-only findings), EXPECTED/PROJECTED (projected targets, no passing benchmark yet).

Target: 40 tok/s decode at 128k-131k context (current V2 baseline MTP OFF: 15.557 / 15.581 tok/s; required increase +157.1% / +156.7%, see PERFORMANCE.md section 1.5).

---

## Phase 1: Profiling baselines and infrastructure (DONE)

Goal: trusted before-numbers and a root-cause picture before any kernel change.

- DONE: Phase-0 hygiene (dynamic repo root, fork-only policy) - `92f0807df`.
- DONE: Phase-1 full decode trace + untraced wall ladder (V3) - `ca49ee25b`, `experiments/phase1/`.
- DONE: Phase-1B baseline matrix (V3) - `experiments/phase1b/`.
- DONE: Phase-2A evidence ledger consolidation - `experiments/phase2a/evidence_ledger.md`.
- DONE: Phase-2B kernel census + byte-exact streamed-weight inventory (10,237,562,880 B = 9.530 GiB per decode step; IQ3_XXS 7.537 GiB) - `experiments/phase2b/`, corrected in Phase-2C.
- DONE: Phase-2C ops fixes + ledger drift correction - `experiments/phase2c/`.
- DONE: EXP-001 benchmark failure classification + VRAM monitor hardening - `966f25454`.
- DONE: EXP-002 V2 model swap + baseline (V2 baseline table) - `4f72448eb`, `experiments/v2_baseline/`.
- DONE: EXP-005 clean V2 Phase-2-only attribution baseline (reset) - `experiments/EXPERIMENT_LOG.md`.
- ATTRIBUTION: Phase-3 GEMV root-cause diagnosis (memory-latency bound, 76.7% instruction-wait stall share) - `experiments/phase3/phase3_report.md`.

Phase-1 exit criteria met: every number below Phase-2 is traceable to an artifact; root cause of the IQ3_XXS GEMV underperformance is identified and evidenced.

## Phase 2: Staging elimination (DONE)

Goal: remove the F16 KV staging round trip from decode attention.

- DONE (merged in fork base): tile flash attention for quantized KV decode at head size 256 - `66dcba5eb`. Measured +39.5% @63k (11.29 -> 15.74 tok/s), corroborated.
- DONE: PATH A fused q4_0 dequant inside tile FA; `dequantize_block_q4_0` launches 32/token -> 0 - `2e033a696`.
  - MEASURED: +28.79% @128k (12.507 -> 16.108 tok/s), +29.03% @131k, +17.01% @63k, +3.82% @16k, -0.41% @1k (MTP OFF). Net attention ms/tok ~36.61 -> 18.90 (131k trace).
  - ACCEPTED TRADEOFF (recorded): fresh prefill -2% to -3.6% at 63k-131k.
  - ACCEPTED TRADEOFF (recorded): 131k VRAM peak 13.30 -> 13.70 GiB (+0.40 GiB; 128k is 13.33 -> 13.27, below).
- DONE: correctness gates (deterministic greedy suite, MTP smoke, bit-exact signature in both MTP configs).

Phase-2 exit criteria met: 128k target >= +15% (measured +28.79%); staging kernel gone from the launch stream; VRAM under the 15.5 GiB guard.

## Phase 3: GEMV decode latency optimization (IN PROGRESS)

Goal: attack the diagnosed latency-bound IQ3_XXS GEMV (76.7% instruction-wait stall share; decode GEMV_IQ3_XXS = 71.5-74.1% of decode kernel time, 350 GB/s = 56% of nominal; 43% of decode wall is inter-dispatch gap).

### DONE in this phase

- DONE: EXP-006 gather hoist restored in-tree - `f7c4436c3`, `068f581e2`.
  - MEASURED: +12.97% @63k (MTP OFF), +1.59% @16k, -0.71% @1k (noise), average +4.6%; MTP ON +7.55% @1k.
  - CONFLICTING (recorded, not averaged): original EXP-006 verdict FAILED/REVERTED in `experiments/EXPERIMENT_LOG.md` vs the restored-in-tree claim; see PERFORMANCE.md section 4.1.
- DONE (negative results, keep as evidence):
  - Phase-4A V1 LDS-staged grid table: bit-exact but slower (V1_BIT_EXACT_BUT_SLOW); 62-66% LDS bank-conflict rate.
  - Phase-4E prefetch-distance / VDR-increase / independent-accumulator variants: marginal to negative (E variants: MARGINAL, do NOT adopt; VDR=2 is optimum).
  - Phase-5: MMQ WMMA is genuine on RDNA3, but GEMM routing for MTP verify is NOT viable (+30%/+50% per-matmul cost at ne11=4-5).
  - EXP-007 (RMSNorm+Quantize+GEMV fusion): rejected at feasibility; dispatch-count reduction cannot address the 15-17 ms long-tail gap stalls (upper bound ~9% of wall < 10% threshold).
- DONE: EXP-023 HIP sign-application fastpath (IQ3_XXS/IQ3_S `iq_apply_sign4`) - `628507055`.
  - MEASURED (op-level, same-protocol): iq3_xxs n=1 114.28 -> 51.33 us/run (-55.1%), iq3_s n=1 113.65 -> 57.32 (-49.6%); n=512 flat. PMU: duration -56.9%, SQ_INSTS_VALU -69.9% per dispatch; VGPR type18 n=1 80 -> 48. Bit-exact (11/11 x2, greedy 3/3 byte-identical, PPL final identical).
  - CAVEAT: full-model wall before/after not measured this run (user-reported production sample only); see PERFORMANCE.md 1.6 and T-9.

### EXPECTED / PROJECTED (no passing benchmark; each requires a full gate pass)

Ranked by expected leverage against the diagnosed cause (Phase-3 report section 4):

1. EXPECTED: software-pipeline the K-loop (hoist next-trip X/Y loads ahead of current-trip consumption). Attacks the vmcnt(0)-per-trip serialization. Target: measurable reduction of the 76.7% stall share on the synthetic harness before any in-tree change.
2. EXPECTED: two independent dependency chains per thread (two kbx groups or two output rows). Evidence: synthetic fused gateup b1->b2 = +65% (184 -> 304 GB/s). Projected: multi-hundred GB/s upside if the effect holds in-model.
3. EXPECTED: nwarps=2/4 calibration sweep for IQ3_XXS on RDNA3 (table change). Lowest effort; treat as calibration, gain uncertain given measured occupancy parity.
4. EXPECTED: decode long-tail gap reduction (max 15-17 ms stalls; 43% of decode wall is gap). Source not yet identified (graph re-launches, scratch allocations, or large-kernel boundaries); a dedicated EXP is required before any change.
5. EXPECTED: recover the PATH A prefill regression (-2% to -3.6% pp @63k-131k) via shape-gated hybrid dispatch.
6. EXPECTED (deferred): weight-layout repack (swizzle) to make grid indices contiguous. Largest engineering cost; only if items 1-5 under-deliver.

### Explicit projected end-state (PROJECTED, not a claim)

PROJECTED: with items 1-4 realized, decode GEMV_IQ3_XXS moves from 350 GB/s (56%) toward the Q3_K-class 503 GB/s (80%) observed on the same V2 model, and the 40 tok/s @128k-131k target is reached. This is a projection from measured per-quant ceilings; it is NOT a measured result and has no date.

### Gates (all Phase-3 changes)

- `test-backend-ops -o MUL_MAT` all quant types PASS (no regression).
- Deterministic greedy suite 12/12 (MTP ON and OFF) bit-identical logits.
- MTP acceptance rate unchanged within run-to-run spread.
- Fresh-process per arm, >= 3 reps, spread < 1% (protocol from Phase-1/2).
- Single change, single commit; EXP entry appended to `experiments/EXPERIMENT_LOG.md` with before/after.
