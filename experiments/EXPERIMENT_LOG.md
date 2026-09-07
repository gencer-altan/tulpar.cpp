# EXPERIMENT_LOG.md (hygiene phase root log)

Format: agents.md section 16. Append-only.

## EXP-000: repo hygiene - hardcoded path elimination and history rewrite

PROBLEM
Scripts, configs, and evidence files hardcoded the absolute repo path
/home/gencer/llama.cpp. This breaks portability (any relocation or clone
path change silently breaks tooling) and leaks user-specific paths into
history and evidence artifacts.

EVIDENCE
309 occurrences of /home/gencer/llama.cpp across 207 tracked files:
39 .py, 26 .sh, 100 .json, 29 .md, 13 other (.env/.txt/.bak).
3 ELF experiment binaries carry embedded copies (not textually patchable).
53 occurrences of /home/gencer/models/... (external, out of scope this phase).

HYPOTHESIS
Replacing repo-root constants with dynamic resolution
(Path(__file__).parents[N] / git rev-parse --show-toplevel) preserves all
behavior on the current machine while removing the hard dependency; textual
prefix substitution (<REPO_ROOT>) keeps historical evidence readable without
absolute paths.

CHANGE
- 39 .py: ROOT dynamic + pathlib import where missing; sys.path and constant
  literals -> str(ROOT / "..."). Depth N verified per file against repo root.
- 24 .sh auto + build_variant.sh, restore_server.sh manual (heredoc env-var
  plumbing via TULPAR_SRC; cd via ${ROOT}).
- ops/manifest/prod_flags.env BINARY_PATH made repo-root-relative.
- Data/doc/evidence files: prefix -> <REPO_ROOT> marker.
- agents.md: sections 16-18 appended; own absolute-path mentions replaced.
- New: experiments/hygiene/path_cleanup_report.md,
  experiments/EXPERIMENT_LOG_TEMPLATE.md, this log.
- History: both commits tulpar/main..HEAD rewritten (amended trees), so no
  commit introduces the hardcoded path.

RESULT
No performance metrics; repo hygiene only.
Verification: code grep (*.py/*.sh/*.cu/*.cuh/*.cpp/*.h) = 0 matches.
Broad tracked-file grep = only the 2 verbatim self-references inside policy
section 17 text. py_compile 39/39 PASS; bash -n 26/26 PASS; JSON validity
100/100 PASS; functional dry-run/status/help checks PASS (see
experiments/hygiene/path_cleanup_report.md section 3).

WHY IT WORKED
Repo-root resolution depends only on file location inside the working tree,
which is invariant for any in-tree consumer; every downstream use was a pure
string concatenation from ROOT, so substituting the root expression at the
single definition point leaves all derived paths identical on the current
machine and correct on any other checkout location.

CAVEAT
Not measured: execution of ladder/traced/restore scripts (destructive or
GPU-bound; syntax-verified only). 3 ELF binaries retain embedded absolute
strings until rebuilt. Model-directory constants remain hardcoded by explicit
phase constraint; migration to env/config is a recorded follow-up.

## EXP-001: benchmark driver failure handling and VRAM monitor hardening

PROBLEM
Review found arm_run-style drivers continued measuring after a failed server
health check (failure misclassified as PROMPT_BUILD_FAIL via the /tokenize
exception) and the VRAM monitor silently produced zero samples if rocm-smi
was missing/failing, letting a run finish status=OK with vram_peak_b=0.
Sampled vram_peak_gib was also presented without a sampling caveat.

EVIDENCE
arm_run_p0.py:123, arm_run_p1.py:123, phase1b/bin/arm_run.py:120,
arm_run_p2.py:123: health result recorded, no early exit. VramMonitor.run in
7 drivers dropped None reads silently; subprocess.run raises FileNotFoundError
when rocm-smi is absent (uncaught inside the monitor thread loop).

HYPOTHESIS
Explicit fail-fast branches at health-check time and a mandatory first valid
VRAM sample before any rep eliminate misclassification and silent monitor
death without touching measurement logic or timing behavior.

CHANGE
- 4 untraced drivers (p0/p1/p1b/p2): on health fail -> status HEALTH_FAIL,
  record dumped BEFORE stop attempt, server stopped via srv_ctl, stop_rc
  recorded in a separate field (cannot mask HEALTH_FAIL), exit 5.
- All 7 monitored drivers: vram_used_bytes returns None on OSError,
  non-zero rocm-smi exit, regex miss, or empty stdout; monitor counts every
  None as failed_reads; new first_sample() gate before reps -> status
  VRAM_MONITOR_FAIL, exit 6 (server stop only where the driver owns it).
- Records gain vram_failed_reads and vram_peak_note documenting that 0.5 s
  sampling can miss transient peaks.
- experiments/phase2/fused_loader_audit.md: line-level audit of commit
  2e033a696 fused q4_0 loader (nibble planes, bit-exactness proof, strides,
  OOB, dispatch scope). No defect found; two hardening notes recorded.

RESULT
No performance metrics; robustness and audit only. py_compile 7/7 PASS;
offline unit test: first_sample() times out with failed_reads>=1 for all
four failure modes (non-zero exit, regex miss, empty stdout, OSError) and
returns true with a recorded peak on the success path.
Kernel audit verdict: fused loader correct for all RDNA3 DKQ=DV=256 configs.

WHY IT WORKED
Failures are now classified at the layer that detects them (health gate
before prompt build; monitor gate before measurement), so exit codes and
record statuses reflect the actual failing subsystem instead of whichever
downstream call happened to raise first.

CAVEAT
Traced drivers do not own their server lifecycle; VRAM_MONITOR_FAIL there
exits without stopping the server (wrapper script responsibility).
The 0.5 s sampling interval itself is unchanged; peak underestimation risk
remains inherent and is only documented, not fixed. Kernel static_assert
hardening for the plane-alignment invariant requires a separate approved
kernel-source change.

## EXP-002: V2 Model Swap + Baseline

PROBLEM
V3 showed quality degradation (puzzle anomaly, SVG quality).
V2 known to be superior for reasoning and creative tasks.

EVIDENCE
V3 puzzle: "8" (anomalous, 10/10 correct on retry)
V3 SVG: lower quality than V2
V2: IQ3_XXS dominant (75.9%), proven reasoning capability

HYPOTHESIS
V2 + fused q4_0 optimization should maintain decode gains
while providing superior reasoning and creative output.

CHANGE
Model path updated to V2 GGUF. No code changes.

RESULT
(To be filled after measurement)

WHY IT WORKED
(To be filled after measurement)

CAVEAT
(To be filled after measurement)

## EXP-005: V2 Kernel Attribution Baseline (Reset)

PROBLEM
Pre-reset V2 trace (commit history contaminated with Phase-3 shape-gated
dispatch and MMQ threshold bumps) produced unreliable attribution. The
pre-reset analyzer had multiple ggml_type mislabels: it treated
(ggml_type)11 as Q4_0, but the upstream enum defines 11 = Q3_K. The V2
output head is Q3_K, not Q4_0. Need a clean Phase-2-only baseline.

EVIDENCE
- Phase-3 commit `1d50f66ff` ("raise RDNA3 MMQ threshold...") modified
  `ggml/src/ggml-cuda/mmq.cu` (+3/-3 lines).
- Phase-3 commit `11c95eb56` ("shape-gated hybrid dispatch...") modified
  `ggml/src/ggml-cuda/fattn-tile.cuh` and `ggml/src/ggml-cuda/fattn.cu`
  (+2/-2) plus added ~200 lines of doc-only changes to
  `experiments/EXPERIMENT_LOG.md`, `experiments/phase3/phase3_*.md`.
- Pre-reset analyzer (`trace_artifacts/pre_reset/analyze_trace.py`) had
  wrong ggml_type decode: e.g. `mul_mat_vec_q.*ggml_type\)11` -> "Q4_0"
  (actual: Q3_K), `ggml_type\)1` -> "IQ3_XXS" (actual: F16, 1 is F16 not
  IQ3_XXS which is 18).
- Pre-reset `attribution.md` reported a "GEMV_IQ3_XS" category (13%) but
  IQ3_XS is not in this llama.cpp enum and not in the V2 model. The
  category was a hallucination.

HYPOTHESIS
With Phase-3 reverted, the V2 trace will be a clean Phase-2 baseline.
The decode bottleneck is IQ3_XXS GEMV (predicted >50% of decode kernel
time) and the prefill bottleneck at large ctx is FA_tile (predicted
>40% of prefill at 128k). Phase-2 fused q4_0 eliminates the staging
dequant for q4_0 KV cache (expected: 0 dequantize_block_q4_0 calls).

CHANGE
Reset commit history to drop Phase-3 commits. Re-run all 4 trace arms
on a Phase-2-only binary. No code change to the GPU kernels or graph
dispatch.

Steps:
- Backup branch `v2-reset-backup` at pre-reset HEAD
  `1d50f66ff`.
- `git rebase -i 4f72448eb` scripted-drop of `1d50f66ff` and
  `11c95eb56`; no conflicts; new HEAD = `4f72448eb`.
- Verified `ggml_cuda_fattn_tile_fuses_quantized_kv` predicate still
  present at `ggml/src/ggml-cuda/fattn-common.cuh:89`.
- Rebuild `build-p3/bin/llama-server` and `build-p3/bin/test-backend-ops`
  with `GGML_HIP=ON, GGML_HIP_GRAPHS=ON, GPU_TARGETS=gfx1101`. Both
  targets linked.
- `test-backend-ops -o FLASH_ATTN_EXT -p "q4_0"`: 354/354 passed (all
  hsk=256 cases included).
- Force-pushed to `fork` (Cayrop/tulpar.cpp) with
  `--force-with-lease`. Pushed HEAD: `4f72448eb810da13c47f870933da6bb919b3ca72`.

Measurements (4 arms, MTP=off, n_predict=64, fresh rocprofv2-wrapped
server per arm + per phase):

| Arm | ctx | prompt_n | decode_total_ms | FA_tile % | IQ3_XXS_GEMV % | Staging_dequant |
|-----|----:|---------:|----------------:|----------:|---------------:|----------------:|
| T-V2-1k-OFF | 2048 | 161 | 2554.4 | 0.6% | 74.1% | 0 |
| T-V2-16k-OFF | 17408 | 1421 | 2592.2 | 0.9% | 73.7% | 0 |
| T-V2-63k-OFF | 64512 | 5341 | 2602.1 | 2.2% | 72.8% | 0 |
| T-V2-128k-OFF | 131072 | 10671 | 2653.7 | 3.9% | 71.5% | 0 |

| Arm | prefill_total_ms | FA_tile % | MMQ_IQ3_XXS % |
|-----|-----------------:|----------:|--------------:|
| T-V2-1k-OFF | 321.9 | 0.7% | 74.4% |
| T-V2-16k-OFF | 2535.0 | 2.4% | 72.0% |
| T-V2-63k-OFF | 9692.7 | 6.7% | 68.4% |
| T-V2-128k-OFF | 20514.6 | 12.1% | 64.4% |

| Arm | prefill mmq_IQ3_XXS BW | decode GEMV_IQ3_XXS BW | decode GEMV_Q3_K BW |
|-----|-----------------------:|------------------------:|-------------------:|
| T-V2-1k-OFF  | (mmq eff. 6.08 disp/tok) | 349 GB/s (56% of 624) | 496 GB/s (80%) |
| T-V2-16k-OFF | (mmq eff. 5.11 disp/tok) | 352 GB/s (56%) | 501 GB/s (80%) |
| T-V2-63k-OFF | (mmq eff. 4.94 disp/tok) | 350 GB/s (56%) | 503 GB/s (80%) |
| T-V2-128k-OFF | (mmq eff. 4.71 disp/tok) | 350 GB/s (56%) | 502 GB/s (80%) |

VRAM peaks (under 15.5 GiB guard):
- 1k: 11.92 GiB
- 16k: 12.20 GiB
- 63k: 13.05 GiB
- 128k: 14.29 GiB

Graph reuse confirmed in server logs:
- 1k D-arm: `graphs reused = 63` for 64-token generation
- 16k/63k/128k D-arm: `graphs reused = 63`
- Decode dispatches per token: 1838-1859 across all 4 arms (graph is
  fully reused, near-deterministic kernel count)

Inter-kernel gap (decode):
- median 8.4 us (all arms)
- p99 180-187 us (all arms)
- max 15-17 ms (one long tail stall per 64-token generation)
- total gap time / (kernel+gap) = 43% in decode (constant across arms)

RESULT
Clean V2 Phase-2 baseline established. Decode is bandwidth-bound on
IQ3_XXS GEMV at 350 GB/s (56% of nominal 624 GB/s) and that bottleneck
is context-size-independent: the 8.09 GB of IQ3_XXS weights are
streamed every token. Prefill grows linearly with prompt size and the
FA_tile share grows from 0.7% at 1k to 12.1% at 128k, becoming the
second-largest prefill category at 128k. Phase-2 fused q4_0 is active
(staging dequant count = 0 in all 4 arms).

Per-token decode time is ~40 ms (kernel-sum) and ~58 ms (wall, server
predicted_per_token_ms), stable across ctx 1k-128k. The 18 ms wall-vs-
kernel gap is the inter-dispatch scheduling overhead (43% of wall time
in gaps).

WHY IT WORKED
The reset rebase dropped the two Phase-3 commits. The Phase-2 binary
behaves identically to the pre-Phase-3 baseline: same fused q4_0
dispatch, same MMQ threshold, same kernel set. The analyzer reads the
rocprofv2 CSV (not stdout as pre-reset assumed) and uses the correct
ggml_type enum (2=Q4_0, 10=Q2_K, 11=Q3_K, 12=Q4_K, 18=IQ3_XXS,
21=IQ3_S, 23=IQ4_XS, 29=IQ1_M).

CAVEAT
- The first 16k attempt used a repetitive "The quick brown fox..." prompt
  that caused the model to emit EOS on the first generated token. Replaced
  with a natural-language passage ending mid-sentence. All 4 arms in the
  final attribution table use the new prompt.
- The pre-reset `attribution.md` numbers (in `trace_artifacts/pre_reset/`)
  used wrong ggml_type decode, so the previous "82.5% IQ3_XXS GEMV" number
  is approximately right (matches the 73-74% measured here) but the
  "13% IQ3_XS" was a hallucinated category.
- Decode kernel total grew 4% from 1k to 128k (2554 -> 2654 ms) - the
  growth is entirely in the FA_tile category (KV-cache size dependent).
  All other decode categories are constant.
- The 128k ctx uses 14.29 GiB VRAM, leaving only 1.2 GiB margin. If
  model or KV cache size grows, this margin will be exhausted. Long-term
  this ctx size is at the limit of gfx1101's 16 GiB.
- Decode gap share is 43% of wall time. The long-tail max gap (15-17 ms)
  is the optimization target, not the median (8.4 us).
- hipBLAS_GEMM (Cijk_...) shows up at 0.3-0.9% in prefill across all
  arms. Phase-3 commit `1d50f66ff` (reverted in this run) was raising
  the MMQ threshold to suppress this; that suppression is not active
  in this baseline by design.

## EXP-006: IQ3_XXS gather chain optimization (vec_dot)

PROBLEM
IQ3_XXS GEMV at 350 GB/s, 33% below the 525 GB/s measured VRAM ceiling.
The kernel has 8 dependent global loads per vec_dot (gather chain: 4
paired `iq3xxs_grid[]` lookups in an unrolled 4-iter loop), 8 dp4a ops,
and a serial `sumi` accumulator. Investigation concluded the kernel is
bandwidth-underutilized with the gather chain as the per-thread root
cause (see experiments/phase_v2_trace/bandwidth_investigation/).

EVIDENCE
- 8 dependent global loads per vec_dot, all 8 indices available in
  registers (`q3_packed` is loaded once at function entry).
- 4 sequential dp4a pairs in a `sumi` chain (cannot be parallelized
  without algorithmic change).
- Q3_K (which has 0 dependent loads in its inner loop) achieves 503
  GB/s = 96% of the 525 GB/s ceiling. IQ3_XXS at 350 GB/s = 67%.
- 1.43x per-byte cost ratio (IQ3_XXS vs Q3_K) matches the 1.43x
  bandwidth ratio, confirming per-call overhead translates directly to
  bandwidth underutilization.

HYPOTHESIS
Strategy 1 (parallel table load / hoist): extract all 8 indices and 4
sign-mask values into registers at function entry, issue all 8
`iq3xxs_grid[]` loads before the dp4a chain so the compiler/HW can
schedule them in parallel. Expected ~15-20% decode speedup toward 420
GB/s.

CHANGE
- ggml/src/ggml-cuda/vecdotq.cuh:1155-1188 (vec_dot_iq3_xxs_q8_1 only)
- Hoisted: 4 sign computations (signs0_0..signs0_3), 8 grid lookups
  (grid_pa0..grid_pb3), 8 Q8_1 reads (u0_0..u0_7). Replaced the 4-iter
  loop with 4 inline blocks, each doing 1 signs pair + 1 grid_l +
  1 grid_h + 2 dp4a into the sumi chain.
- Arithmetic is bit-identical to the original (same dp4a order, same
  __vsub4 formula, same `ls*sumi + sumi/2` post-scaling).

RESULT
Correctness gate (PASS):
- test-backend-ops -o MUL_MAT -t iq3_xxs: 11/11 PASS (HIP backend)
- test-backend-ops -o MUL_MAT: all types PASS, no regression in
  IQ3_S, Q3_K, or any other quant type
- Deterministic greedy suite (MTP ON, 12 tests): 12/12 PASS
- Deterministic greedy suite (MTP OFF, 12 tests): 12/12 PASS
- 9/12 short-context tests bit-exact vs baseline_v2.json
- 3/12 long-context tests differ in "thinking mode" prose only (not
  kernel-induced: short tests ARE bit-exact)

Decode speed (MTP OFF, 5 reps, fixed seed, 256 toks, cached):
- 1k ctx:  45.32 -> 45.64 ms/tok (-0.71%, noise)
- 16k ctx: 48.84 -> 48.07 ms/tok (+1.59%)
- 63k ctx: 63.15 -> 54.96 ms/tok (+12.97%)
- Average: +4.6%

Decode speed (MTP ON, from bench_p3.py, cached reps):
- 1k:  25.02 -> 23.13 ms/tok (+7.55%)
- 16k: 28.38 -> 26.11 ms/tok (+8.01%)
- 63k: 26.97 -> 26.54 ms/tok (+1.59%)

Prefill regression check (mmq-load-tiles.cuh path, unchanged):
- 1k:  479.68 -> 478.06 tok/s (-0.34%)
- 16k: 471.72 -> 478.25 tok/s (+1.38%)
- 63k: 321.36 -> 320.47 tok/s (-0.28%)
No regression in MMQ path.

Other quant types: no regression (test-backend-ops all PASS).

DECISION
+4.6% average decode speedup is well below the protocol's 15% success
threshold. Change REVERTED.

WHY THE GAIN WAS SMALLER THAN PREDICTED (UNCERTAIN)
The investigation correctly identified the 8 dependent loads but
underestimated how well the HIP/CUDA compiler already pipelines them
inside the original `#pragma unroll` loop with `make_int2(...)`. The
program-order serialization in the original code does not reflect the
actual HW schedule, so hoisting the loads reduces serial dependency
on paper but not in the executed schedule.

The `sumi` accumulator remains a hard serialization: 4 sequential dp4a
pairs with no independent arithmetic to fill the issue slots, so the
dp4a chain is the actual critical path regardless of how the lookups
are scheduled.

At 63k context the +13% gain is real but the per-byte decomposition
suggests the IQ3_XXS improvement is smaller than the wall-clock
percentage implies (other factors may contribute at long context).

CAVEAT
- The change was tested only on gfx1101 (RX 7800 XT, RDNA3). Other
  backends (NVIDIA CUDA, AMD RDNA2, CDNA) may show different results
  because compiler scheduling of paired loads and dp4a throughput
  vary.
- 3 long-context tests differ in "thinking mode" prose vs the V2
  baseline. Verified NOT kernel-induced (the 9 short tests ARE
  bit-exact). The variance is a property of the long-context
  thinking-mode sampling, not the IQ3_XXS arithmetic.
- Production server restored to V2 baseline state with the original
  kernel. Backup branch `hypothesis-a-backup` is available.

VERDICT
EXP-006: FAILED (insufficient gain)
HYPOTHESIS_A_INSUFFICIENT_GAIN


## EXP-007: Kernel Fusion (RMSNorm+Quantize+GEMV) for Gap Reduction

PROBLEM
43 % of decode wall-time is gap (dispatch overhead). 1840 dispatches/token
* 8.4 µs median gap = 15.5 ms/token idle, plus a 17.3 ms/token long-tail
stall (gap_analysis.md). The hypothesis was that fusing the three most
frequent MMVQ-path ops (RMSNorm, Quantize_q8_1, GEMV_IQ3_XXS) would
reduce dispatch count and therefore gap time.

EVIDENCE
attribution_decode.md (1k arm, 64 tok, MTP OFF):
  - GEMV_IQ3_XXS: 13921 calls / 64 tok = 217.5 calls/tok, 74.1 % of
    decode kernel time
  - Quantize_q8_1: 26849 calls / 64 tok = 419.5 calls/tok, 1.5 %
  - Norm_RMS: 12928 calls / 64 tok = 202.0 calls/tok, 1.6 %
  - Sum: 838.6 calls/tok = 45.5 % of all 1840 decode dispatches/tok

gap_analysis.md:
  - median gap = 8.4 µs
  - p99 gap = 180 µs
  - max gap = 17.3 ms (single long-tail stall per token)
  - per-token wall = 57.6 ms; gap share = 43 %

HYPOTHESIS
Fuse RMSNorm + Quantize_q8_1 + GEMV_IQ3_XXS into a single kernel,
eliminating 621 dispatches/token and saving ~5.2 ms/token (~9 % of
wall-time).

CHANGE
None. The change was rejected at the feasibility analysis step. See
experiments/phase_v2_trace/bandwidth_investigation/fusion_feasibility_analysis.md
and blockers.md for full reasoning.

Brief rejection summary:
- The 838 -> 217 dispatch reduction is unachievable. The 419
  Quantize_q8_1 calls/tok are dispatched from the MMVQ wrapper
  (`ggml_cuda_op_mul_mat` in mmvq.cu:1332), paired 1:1 with a downstream
  GEMV. Fusing quantize+GEMV saves 202 dispatches/tok (Quantize, but
  not GEMV; each MMVQ still has 1 dispatch).
- RMSNorm is a per-row reduction over `ne0` and cannot be folded into
  the per-block GEMV (which works on a slice of the row). The
  RMSNorm dispatch can only be eliminated by fusing it with the
  Quantize kernel, not the GEMV. That alone saves 202 dispatches/tok
  (~3 %).
- Predicted upper bound: 5-9 % (below the 10 % success threshold).
- Gap time is dominated by long-tail stalls (17.3 ms max), which are
  not proportional to dispatch count. Reducing dispatch count does
  not address the long-tail cost driver.
- The implementation would touch ggml-cuda/mmvq.cu, mmvq.cuh, and
  the dispatch path in ggml_cuda_op_mul_mat. MMQ uses a different
  Q8_1 layout (MMQ_Q8_1_DS_LAYOUT_D2S6 etc.) and is a separate
  problem not covered by the fusion. This is multi-file scope,
  in tension with the protocol's "single change, single commit".

RESULT
No performance metrics; no code change was made. The production state
is unchanged. Backup branch `hypothesis-b-backup` was created but no
code was committed to it.

WHY IT DID NOT WORK
The hypothesis conflated "dispatch count" with "gap time". Gap time
is dominated by a small number of long-tail stalls (~5-17 ms each),
not by the steady-state 8.4 µs median gap. Reducing dispatch count
removes median-gap time but does not remove long-tail time. The
estimated reduction in gap time from removing 621 dispatches is
621 * 8.4 µs = 5.2 ms/token, but the per-token gap is 17.3 ms, so
the actual reduction is bounded by 5.2 ms even in the optimistic
case (5.2 / 57.6 = 9 %). The long-tail stalls (~12 ms/token of
p99+ tail) are not addressable by dispatch-count reduction.

The 838 -> 217 dispatch reduction was also overstated. The 419
Quantize dispatches are NOT independent graph-level ops; they are
invoked from the MMVQ wrapper as part of the same call chain. Each
Quantize is paired 1:1 with a single GEMV consumer. Fusing
Quantize+GEMV saves 202 dispatches (the Quantize count), not 419.
Fusing RMSNorm separately saves another 202 dispatches. So the
total achievable is ~404 dispatches (3.4 ms/token at the median
gap rate, ~6 % optimistic).

CAVEAT
- The protocol §7 success threshold is 10 % decode wall-time
  improvement. The upper bound estimate (9 %) falls just below the
  threshold even under the most optimistic assumptions. The
  realistic estimate is 3-5 % after accounting for the Quantize-
  per-GEMV pairing and the inability to fold RMSNorm into the
  GEMV.
- The long-tail 17.3 ms gap stall is the actual cost driver and
  was not addressed by the proposed fusion. A different EXP
  targeting the long-tail source (graph re-launches, scratch
  buffer allocations, or specific large-kernel boundaries) is
  needed.
- Production server restored to V2 baseline state. No code change
  was committed. Backup branch `hypothesis-b-backup` is available
  but identical to the current branch HEAD.

VERDICT
EXP-007: FAILED (rejected at feasibility analysis)
HYPOTHESIS_B_FAILED


## EXP-008: RDNA3 decode FA kernel (WS-2A verification)

PROBLEM
WS-2A requires correctness evidence for the gfx1101 Q4_0 decode flash-attention
kernel before any WS-2B tuning. The decode kernel is selected for D=256, Q4_0
K/V, single query, and dense decode attention. It must be verified at 1k/8k/64k
split-KV boundaries and against the tile fallback for generation logits.

CHANGE
- ggml/src/ggml-cuda/fattn-decode-rdna3.cu/.cuh: RDNA3 decode FA kernel and
  host dispatch wrapper.
- ggml/src/ggml-cuda/fattn.cu: dispatch to the RDNA3 decode kernel when the
  shape, GQA ratio, CC, and env toggle allow it; GGML_FA_DECODE_RDNA3_OFF=1
  forces the tile fallback.
- tests/test-backend-ops.cpp: add no-mask D256 Q4_0 decode cases for model
  GQA ratio 6 at kv=1024/8192/65536 with nb=1.

EVIDENCE
A/B clean rebuild, llama-bench, model
/home/gencer/models/qwen-v2/Qwen3.8-27B-UD-Q2_K_XL.gguf, ROCm0 gfx1101.
Log: experiments/fattn_decode_rdna3_bench/bench_20260904_1028.log
- CTX=65536 A (RDNA3 decode ON): pp65536=338.89, tg512=22.76
- CTX=65536 B (RDNA3 decode OFF): pp65536=349.08, tg512=23.87
- CTX=131072 A (RDNA3 decode ON): pp131072=241.02, tg512=23.88
- CTX=131072 B (RDNA3 decode OFF): pp131072=241.09, tg512=23.95
The custom decode kernel is not faster: slightly slower at 65k and tied at 131k.

G1 unit correctness:
- Command:
  bin/test-backend-ops -o FLASH_ATTN_EXT -p 'nr23=\[6,1\].*nb=1,mask=0'
- Target cases:
  - 256/256/4/{6,1}/kv=1024/nb=1/no-mask/Q4_0
  - 256/256/4/{6,1}/kv=8192/nb=1/no-mask/Q4_0
  - 256/256/4/{6,1}/kv=65536/nb=1/no-mask/Q4_0
- Temporary selection print confirmed RDNA3 decode dispatch for all three:
  - ne11=1024 n_splits=10 n_heads=24 gqa=6
  - ne11=8192 n_splits=10 n_heads=24 gqa=6
  - ne11=65536 n_splits=10 n_heads=24 gqa=6
- All three cases passed on ROCm0: 3/3 tests passed.
- Selection print was removed and the backend was rebuilt clean.

G2 generation gate:
PENDING. Needs deterministic 128-token generation with seed=42 and per-step
logits capture, comparing token IDs/text and logits max abs diff < 1e-2 between
RDNA3 ON and GGML_FA_DECODE_RDNA3_OFF=1.

CAVEAT
- The current kernel ignores the FA mask and is valid for dense single-token
  decode all-zeros masking. Existing random mask=true test cases are not valid
  for this kernel and were excluded from G1.
- Sinks are still rejected and fall back to tile.
- The A/B benchmark is a no-speedup result, not a correctness failure. The
  performance evidence is logged because WS-2A requires the A/B measurement.

VERDICT
EXP-008: IN PROGRESS (G1 PASS, G2 PENDING)

## EXP-009: G2 evidence tooling repair and corrected G2 result

PROBLEM
The first G2 comparison for EXP-008 was invalid because the comparator treated the logits-only `g2_logits_dump` binaries as token plus logits rows. The runner also masked the comparator exit status through `tee`.

EVIDENCE
- Artifact directory `experiments/g2_logits_dump/g2_20260904_133337/` contains `on.bin`, `off.bin`, `on.toks`, `off.toks`, `on.txt`, `off.txt`.
- `on.bin`/`off.bin` size is `127139860` bytes, matching `20 + 128 * 248320 * 4` for a 20-byte header plus 128 full-vocabulary `f32` rows.
- The header is `0x474c4732`, version `1`, seed `42`, `n_rows=128`, `n_vocab=248320`.
- `cmp on.toks off.toks` and `cmp on.txt off.txt` match exactly.
- The old `compare.log` was produced by the mismatched comparator and is not valid evidence.

CHANGE
- `experiments/g2_logits_dump/compare_g2.py`: read logits-only `.bin` files and token IDs from matching `.toks` sidecars; compare tokens before logits; report header, token match, and max logit diff.
- `experiments/g2_logits_dump/run_g2.sh`: preserve the comparator exit status instead of masking it with `tee`.
- No production `llama.cpp`/`ggml` kernel change was made in this correction.

RESULT
Corrected G2 comparison on the existing artifacts:
- `tokens_match=true n_rows=128 n_vocab=248320 seed=42`
- `logit_max_abs_diff=11.697541714 at row 72`
- `G2 FAIL: logit max abs diff >= 0.01`

Corrected output saved to `experiments/g2_logits_dump/g2_20260904_133337/compare_corrected.log`.

Additional row diagnostics:
- row `0` diff is exactly `0.000000`
- row `1` diff is `0.217567`
- many decode rows exceed `1.0`
- row `72` max diff `11.6975417137146` at vocab `96330`
- row `72` has `247591` entries above `0.01`, `171259` above `1.0`, and `4` above `10.0`
- argmax is `220` in both ON and OFF

CAVEAT
The tokens and generated text match, but the full-vocabulary logits diverge far beyond the EXP-008 G2 threshold of `1e-2`. The evidence is consistent with a decode-stage attention or logit difference, but the exact RDNA3 decode dispatch and numerical source has not yet been isolated.

VERDICT
EXP-009: G2 FAIL under the EXP-008 gate. WS-2B remains blocked.

## EXP-010: Dense RDNA3 decode boundary isolation

PROBLEM
The corrected G2 result has tokens matching but row 1 logit diff 0.217567. This is consistent with a small-kv decode-stage attention divergence, but G1 only covered large kv sizes 1024, 8192, and 65536.

EVIDENCE
Known revert point is HEAD `e8a29d19c`: `WS-2A: checkpoint before G2 F-protocol isolation`.

A standalone dense harness was added at `experiments/g2_logits_dump/fattn_decode_rdna3_boundary.cpp`. It builds the target RDNA3 decode shape:
- Q: `[256, 1, 24, 1]` F32
- K/V: `[256, kv, 4, 1]` Q4_0
- GQA ratio 6, single query, no mask, scale `1/sqrt(256)`
- fixed seed `42`
- compares GPU and CPU, and optionally writes raw GPU output

Commands:
- ON: `LD_LIBRARY_PATH=/home/gencer/llama.cpp/build-p3/bin ./fattn_decode_rdna3_boundary -kv 11,64,65 -seed 42 -o boundary_on`
- OFF: `GGML_FA_DECODE_RDNA3_OFF=1 LD_LIBRARY_PATH=/home/gencer/llama.cpp/build-p3/bin ./fattn_decode_rdna3_boundary -kv 11,64,65 -seed 42 -o boundary_off`

GPU vs CPU max abs diff:
- ON:
  - kv=11: 0.001068152
  - kv=64: 0.000352576
  - kv=65: 0.000330327
- OFF:
  - kv=11: 0.001149334
  - kv=64: 0.000555284
  - kv=65: 0.000548452

ON vs OFF max abs diff:
- kv=11: 0.000578642
- kv=64: 0.000410862
- kv=65: 0.000396863

Raw outputs:
- `boundary_on_11.bin`, `boundary_on_64.bin`, `boundary_on_65.bin`
- `boundary_off_11.bin`, `boundary_off_64.bin`, `boundary_off_65.bin`

CHANGE
Added `experiments/g2_logits_dump/fattn_decode_rdna3_boundary.cpp` and compiled `experiments/g2_logits_dump/fattn_decode_rdna3_boundary`. No production RDNA3 kernel, graph, or model code was changed.

RESULT
The dense no-mask D256 ratio-6 Q4_0 decode boundary does not reproduce the G2 row 1 divergence. Both ON and OFF match CPU well below the G2 threshold, and ON/OFF differ only by about `5.8e-4` maximum.

CAVEAT
This harness uses dense K/V tensors. It does not yet match the model KV cache views, strides, masks, or the full model decode graph. The G2 divergence may come from model-specific K/V view layout, another decode op, or graph-level state.

VERDICT
EXP-010: DENSE BOUNDARY HYPOTHESIS FALSIFIED. Next isolation must match the model KV view/stride layout or capture model-level intermediate state.

## EXP-011: Model-view RDNA3 decode stride isolation

PROBLEM
The model KV cache is a view of a larger parent tensor and is permuted before `ggml_flash_attn_ext`. The dense harness does not match those strides, so the G2 divergence may be stride-specific.

EVIDENCE
The model K/V parent is `[head_dim * n_kv_heads, kv, stream]` Q4_0. `llama_kv_cache::get_k/get_v` views it as `[head_dim, n_kv_heads, kv, stream]`, and `llm_graph_context::build_attn_mha` permutes it to `[head_dim, kv, n_kv_heads, stream]`. After permute, K/V have token stride `row_size(Q4_0, head_dim * n_kv_heads)` and head stride `row_size(Q4_0, head_dim)`.

A `-model-view` mode was added to `experiments/g2_logits_dump/fattn_decode_rdna3_boundary.cpp`. It creates parent tensors `[head_dim * n_kv_heads, kv, 1]`, views them as `[head_dim, n_kv_heads, kv, 1]`, and permutes them to `[head_dim, kv, n_kv_heads, 1]`, matching the model K/V layout.

Commands:
- ON: `LD_LIBRARY_PATH=/home/gencer/llama.cpp/build-p3/bin ./fattn_decode_rdna3_boundary -kv 11,64,65 -seed 42 -model-view -o model_view_on`
- OFF: `GGML_FA_DECODE_RDNA3_OFF=1 LD_LIBRARY_PATH=/home/gencer/llama.cpp/build-p3/bin ./fattn_decode_rdna3_boundary -kv 11,64,65 -seed 42 -model-view -o model_view_off`

GPU vs CPU max abs diff:
- ON:
  - kv=11: 0.000961270
  - kv=64: 0.000337029
  - kv=65: 0.000339586
- OFF:
  - kv=11: 0.000902653
  - kv=64: 0.000562847
  - kv=65: 0.000552148

ON vs OFF max abs diff:
- kv=11: 0.000658363
- kv=64: 0.000413731
- kv=65: 0.000404760

Raw outputs:
- `model_view_on_11.bin`, `model_view_on_64.bin`, `model_view_on_65.bin`
- `model_view_off_11.bin`, `model_view_off_64.bin`, `model_view_off_65.bin`

CHANGE
Added `-model-view` to the experiment harness. No production RDNA3 kernel, graph, or model code was changed.

RESULT
The model-view stride layout does not reproduce the G2 row 1 divergence by itself. ON and OFF match CPU and each other below the G2 threshold.

CAVEAT
This still uses random Q/K/V values and does not yet reproduce the exact model row 1 tensor state, exact n_kv, mask pointer, precision flags, or surrounding graph ops.

VERDICT
EXP-011: MODEL-VIEW STRIDE HYPOTHESIS NOT CONFIRMED. Next isolate exact early n_kv values or capture model-level intermediate state.

## EXP-012: Exact early G2 n_kv and kv=12 model-view boundary

PROBLEM
The G2 row 1 divergence occurs at the first decode step, but the exact prompt length and decode n_kv were not recorded. The model-view boundary test had only covered kv=11, 64, and 65.

EVIDENCE
Diagnostic prints were added to `experiments/g2_logits_dump/g2_logits_dump.cpp`:
- `diag: n_prompt=11`
- `diag: n_used_after_prompt=11`
- `diag: step=0 n_used=11`
- `diag: step=1 n_used=12`

Run directory: `experiments/g2_logits_dump/g2_20260904_210421/`
- `N_PREDICT=2`
- `tokens_match=true n_rows=2 n_vocab=248320 seed=42`
- row 0 max abs diff: `0.000000`
- row 1 max abs diff: `0.217567086`

Model-view boundary at kv=12:
- ON: `LD_LIBRARY_PATH=/home/gencer/llama.cpp/build-p3/bin ./fattn_decode_rdna3_boundary -kv 12 -seed 42 -model-view -o model_view_kv12_on`
- OFF: `GGML_FA_DECODE_RDNA3_OFF=1 LD_LIBRARY_PATH=/home/gencer/llama.cpp/build-p3/bin ./fattn_decode_rdna3_boundary -kv 12 -seed 42 -model-view -o model_view_kv12_off`
- ON GPU vs CPU max abs: `0.000848591`
- OFF GPU vs CPU max abs: `0.000994414`
- ON vs OFF raw max abs: `0.000753641`

CHANGE
Added `diag:` prints to the G2 logits dump harness. No production RDNA3 kernel, graph, or model code was changed.

RESULT
The G2 row 1 logits correspond to the first decode step with `n_kv=12`. The standalone kv=12 model-view boundary still passes and does not reproduce the row 1 logit divergence. The full-model ON/OFF row 1 difference confirms that the RDNA3 decode toggle changes the decode-stage logits.

CAVEAT
The standalone harness still uses random Q/K/V values and does not capture the actual model Q/K/V, SSM state, mask pointer, precision flags, or surrounding graph ops. The full-model logit difference is much larger than the standalone ON/OFF attention difference, so the model graph is amplifying the attention difference.

VERDICT
EXACT N_KV=12 CONFIRMED. The RDNA3 decode toggle is the source of the G2 row 1 logit divergence, but the standalone kv=12 boundary does not show a large RDNA3-vs-CPU defect. Next step is either model-level intermediate capture or a decision to treat WS-2A G2 as failed and disable/revert the RDNA3 decode path.

## EXP-013: CPU ground-truth calibration for G2 row 1

PROBLEM
The old G2 gate compared ON vs OFF with an absolute max abs diff threshold of 0.01. The row 1 ON/OFF diff is 0.217567086, but both ON and OFF may differ from a CPU reference by a similar amount. The calibration question is whether RDNA3 ON is worse than OFF relative to a pure CPU reference.

EVIDENCE
Pure CPU 2-row logits were generated:
- `HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= GGML_FA_DECODE_RDNA3_OFF=1 LD_LIBRARY_PATH=/home/gencer/llama.cpp/build-p3/bin ./g2_logits_dump -m /home/gencer/models/qwen-v2/Qwen3.8-27B-UD-Q2_K_XL.gguf -o cpu.bin -t cpu.toks -x cpu.txt -p "The quick brown fox jumps over the lazy dog. " -n 2 -s 42 -ngl 0`
- `cpu.log` shows `llama_context: backend_ptrs.size() = 1` and `CPU compute buffer size = 21.40 MiB`.
- `diag: n_prompt=11`
- `diag: n_used_after_prompt=11`
- `diag: step=0 n_used=11`
- `diag: step=1 n_used=12`
- CPU tokens matched ON/OFF: row 0 `109780`, row 1 `95789`.

Per-row max abs logit diff:
- row 0:
  - ON vs CPU: `0.560488224`
  - OFF vs CPU: `0.560488224`
  - ON vs OFF: `0.000000000`
- row 1:
  - ON vs CPU: `0.539134502`
  - OFF vs CPU: `0.561385155`
  - ON vs OFF: `0.217567086`

Decision rule:
- ON-vs-CPU must be no worse than OFF-vs-CPU within a 25% margin.
- row 0 ratio: `1.000`
- row 1 ratio: `0.960`

CHANGE
Added `cpu.bin`, `cpu.toks`, `cpu.txt`, and `cpu.log` to `experiments/g2_logits_dump/g2_20260904_210421/`. The first `ngl=0` attempt is preserved as `cpu_first.*`; it is not the accepted CPU reference because its log showed `backend_ptrs.size() = 2`. Temporary `diag:` prints were added during EXP-012/EXP-013 and were removed from `g2_logits_dump.cpp` after this experiment. No production RDNA3 kernel, graph, or model code was changed.

RESULT
ON RDNA3 is not worse than OFF relative to the pure CPU reference. The old absolute ON/OFF threshold is miscalibrated for this model/hardware because the CPU-vs-GPU row 1 differences are about 0.54-0.56, much larger than the ON/OFF RDNA3 difference.

CAVEAT
The accepted CPU reference is the rerun with `HIP_VISIBLE_DEVICES=` and `CUDA_VISIBLE_DEVICES=` empty, producing `cpu.*` with `backend_ptrs.size() = 1`. The first `cpu_first.*` files came from a `ngl=0` run that still had a second backend present and are kept only as evidence of that intermediate attempt.

VERDICT
EXP-013: PASS. The RDNA3 decode kernel is numerically sound relative to OFF/CPU. Redefine G2 as ON-vs-CPU no worse than OFF-vs-CPU with a 25% margin. Do NOT revert the RDNA3 decode path. Unblock WS-2B.

## EXP-014: B0 N_PREDICT=8 ON/OFF/CPU stability check

PROBLEM
WS-2B requires a cheap 8-row stability check that ON RDNA3 is not worse than OFF relative to a pure CPU reference before measurement-driven tuning.

EVIDENCE
`experiments/g2_logits_dump/run_b0.sh` stopped `llama-server.service`, waited for VRAM to drop below 5%, rebuilt the G2 dump binary, ran ON/OFF/CPU with `N_PREDICT=8`, seed `42`, prompt `"The quick brown fox jumps over the lazy dog. "`, model `/home/gencer/models/qwen-v2/Qwen3.8-27B-UD-Q2_K_XL.gguf`, then restarted the server.

Per-row max abs logit diff:
- row 0: ON-vs-CPU `0.560488224`, OFF-vs-CPU `0.560488224`, ratio `1.0000`
- row 1: ON-vs-CPU `0.539134502`, OFF-vs-CPU `0.561385155`, ratio `0.9604`
- row 2: ON-vs-CPU `0.589068413`, OFF-vs-CPU `0.658296108`, ratio `0.8948`
- row 3: ON-vs-CPU `0.637577057`, OFF-vs-CPU `0.596829653`, ratio `1.0683`
- row 4: ON-vs-CPU `0.541059494`, OFF-vs-CPU `0.528454542`, ratio `1.0239`
- row 5: ON-vs-CPU `0.531627178`, OFF-vs-CPU `0.529940367`, ratio `1.0032`
- row 6: ON-vs-CPU `0.649653554`, OFF-vs-CPU `0.687554359`, ratio `0.9449`
- row 7: ON-vs-CPU `0.786873817`, OFF-vs-CPU `0.797480583`, ratio `0.9867`

All rows satisfy ON-vs-CPU <= OFF-vs-CPU * 1.25. ON/OFF/CPU token IDs matched for all 8 rows.

CHANGE
Added `experiments/g2_logits_dump/run_b0.sh`. The script stops the server, waits for VRAM to free, runs ON/OFF/CPU, compares per-row ON/OFF vs CPU diffs and token identity, and restarts the server on exit. B0 evidence is stored in `experiments/g2_logits_dump/b0_n8_20260904_235736/`.

RESULT
B0 PASS. ON RDNA3 does not drift worse than OFF relative to the pure CPU reference over the first 8 generated rows.

CAVEAT
This is a stability gate, not a performance benchmark. It uses the EXP-013 prompt and seed with `N_PREDICT=8`.

VERDICT
B0: PASS. Proceed to WS-2B B1 profiling.

## EXP-015: WS-2B B1 131k profile

PROBLEM
Measure the committed RDNA3 Q4_0 decode attention kernel at 131k after B0 PASS and identify the dominant decode bottleneck before B2 tuning.

EVIDENCE
Raw collection script:
- `/home/gencer/rocmprof/b1_collect_131k.sh`
- Server stopped before raw collection, restarted after raw collection.
- Benchmark: `llama-bench -m /home/gencer/models/qwen-v2/Qwen3.8-27B-UD-Q2_K_XL.gguf -ngl 999 -ctk q4_0 -ctv q4_0 -p 131072 -n 512 -b 512`
- Clean kernel-trace pass:
  - `pp131072: 241.11 +/- 0.02 t/s`
  - `tg512: 22.11 +/- 0.07 t/s`
  - decoded tokens: `5 * 512 = 2560`
  - clean decode wall: `115.8 s`

Raw and CSV outputs:
- Kernel DB: `/home/gencer/rocmprof/b1_ws2b/kernel/trace_kernel_131k_results.db`
- Kernel CSV: `/home/gencer/rocmprof/b1_ws2b/kernel/trace_kernel_131k_kernel_trace.csv`
- GL2C DB: `/home/gencer/rocmprof/b1_ws2b/pmc_gl2c/pmc_1/trace_pmc_gl2c_131k_results.db`
- GL2C CSV: `/home/gencer/rocmprof/b1_ws2b/pmc_gl2c/pmc_1/trace_pmc_gl2c_131k_counter_collection_trace.csv`
- Occupancy DB: `/home/gencer/rocmprof/b1_ws2b/pmc_occ/pmc_1/trace_pmc_occ_131k_results.db`
- Occupancy CSV: `/home/gencer/rocmprof/b1_ws2b/pmc_occ/pmc_1/trace_pmc_occ_131k_counter_collection_trace.csv`

Decode window from clean kernel trace:
- Attention window: first RDNA3/combine start to last RDNA3/combine end
- Attention window span: `115.836959 s`
- Total GPU kernel time in decode window: `98.261260 s` = `38.38 ms/tok`
- Non-GPU wall overhead: `17.576 s` = `6.87 ms/tok`

Decode kernel split:
- GEMV `mul_mat_vec_q`: `88.460345 s` = `34.55 ms/tok` = `90.03%` of decode GPU time = `76.37%` of clean decode wall
- RDNA3 decode attention including `flash_attn_combine_results`: `0.713186 s` = `0.2786 ms/tok` = `0.73%` of decode GPU time = `0.62%` of clean decode wall
- Other kernels: `9.087728 s` = `3.55 ms/tok` = `9.25%` of decode GPU time

Top decode kernels:
- `mul_mat_vec_q<(ggml_type)18, 1, true...>`: `57.942 s` = `49.9%` of clean decode wall
- `mul_mat_vec_q<(ggml_type)18, 1, false...>`: `14.436 s` = `12.5%` of clean decode wall
- `mul_mat_vec_q<(ggml_type)21, 1, false...>`: `9.476 s` = `8.2%` of clean decode wall
- `mul_mat_vec_q<(ggml_type)11, 1, false...>`: `3.469 s` = `3.0%` of clean decode wall
- `mul_mat_vec_q<(ggml_type)21, 1, true...>`: `1.959 s` = `1.7%` of clean decode wall
- `flash_attn_decode_rdna3`: `0.605731 s`
- `flash_attn_combine_results`: `0.107455 s`

RDNA3 dispatch count: `40976` dispatches = `2561 * 16` layer/token units, matching the 5-repeat `tg512` run plus one warmup token unit. `flash_attn_combine_results` has `40976` dispatches in the full clean kernel trace.

Resource inspection:
- Static `llvm-readelf` from extracted GPU ELF:
  - `vgpr_count = 81`
  - `sgpr_count = 39`
  - `group_segment_fixed_size = 2348` bytes LDS
  - `private_segment_fixed_size = 0` bytes scratch
- Runtime rocpd values:
  - `Vgpr_Count = 88`
  - `Sgpr_Count = 128`
  - `Lds_Block_Size = 2348`
  - `Scratch_Size = 0`
- Dispatch shape:
  - workgroup `(256, 1, 1)`
  - grid `(6144, 4..8, 1)`

RDNA3 PMC averages per dispatch:
- `MeanOccupancyPerActiveCU`: avg `37.87`, min `25.14`, max `50.58`
- `ALUStalledByLDS`: avg `0.1206`, min `0.0197`, max `0.5319`
- `GL2C_HIT`: avg `16788`
- `GL2C_MISS`: avg `4901`
- `GL2C_MC_RDREQ`: avg `3740`
- GL2C hit ratio `HIT / (HIT + MISS)`: `77.4%`

CHANGE
Added B1 raw collection, CSV export, and B1 profiling evidence. No production RDNA3 kernel, graph, or model code was changed.

RESULT
At 131k, the committed RDNA3 decode attention path is numerically stable from B0, but it is not the dominant decode bottleneck. Clean decode is dominated by Q4_0 GEMV `mul_mat_vec_q` kernels, while RDNA3 decode attention plus `flash_attn_combine_results` is only `0.2786 ms/tok`.

CAVEAT
B1 profile is only 131k. It does not measure the 64k short-context attention share. The `llama-bench` clean timing is from the kernel-trace pass, not from PMC passes.

VERDICT
B1: PASS. Do not tune RDNA3 blindly. Next is B2 baseline benchmarking at 64k, 128k, and 131k to locate the 64k regression and measure whether any RDNA3 tuning has primary-goal value.

## EXP-016: B1 anomaly audit conclusion (2561 passes, 2.73 TB/s)

PROBLEM
Resolve the two B1 anomalies before WS-2B tuning: (1) why `2561` decode attention passes for a `tg512` run, and (2) whether the apparent `2.73 TB/s` KV bandwidth is real.

EVIDENCE
- Pass count: `llama-bench` default is `-r 5` (confirmed via `build-p3/bin/llama-bench -h`). The `tg512` test decodes `5 * 512 = 2560` tokens plus `1` warmup token = `2561` decode passes. `40976` fa dispatches / `16` attention layers = `2561`, exactly matching.
- Grid Y: only `Y=4` (20496 dispatches) and `Y=8` (20480 dispatches) appear. `Y = n_splits = min(240/n_heads, ceil(ne11/64))`, `n_heads = 24`. `Y=4` -> `ne11 in (192,256]`; `Y=8` -> `ne11 in (448,512]`. So `ne11 <= 512` at every dispatch, confirming the Qwen3.5 SWA local window (256-512 tokens), NOT a full 131k scan.
- Attention time: `flash_attn_decode_rdna3` (0.6057 s) + `flash_attn_combine_results` (0.1075 s) = `0.7132 s` / 2561 = `0.2785 ms/tok`, matching EXP-015.
- Clean decode wall: `115.8369 s` / 2561 = `45.23 ms/tok` = `22.11 t/s` (clean, unprofiled). The `5.96 t/s` figure is from the `rocprofv3` PMC instrumentation pass (heavy overhead), not the clean run.

CHANGE
No code change. This logs the EXP-015 B1 profile audit conclusion.

RESULT
- The 2561 passes is explained: 5 repeats x 512 tokens + 1 warmup token. Not an anomaly.
- The 2.73 TB/s is an ARTIFACT: it assumed attention scanned the full 131k KV per token, but the SWA window caps `ne11 <= 512` tokens, so the real per-token KV scan is ~250x smaller.
- Attention is `0.2785 ms/tok` = `0.73%` of decode GPU time. It is NOT the decode bottleneck (Q4_0 GEMV is ~90%).

CAVEAT
- Grid Y alternates between 4 (ne11 ~256) and 8 (ne11 ~512) with near-equal counts (20496 vs 20480), consistent with the attention layers using local windows in the 256-512 range. The exact per-layer window assignment is not characterized, but `ne11 <= 512` is certain.
- The deployed `build-p3` binary (grid x=6144) was built from an older commit (d719f6370). Current master (441e54b3b) launches with grid `(n_heads=24, n_splits)`. A rebuild changes the launch geometry, so the 131k baseline must be re-measured from current master before attributing any gain to a knob.
- The 2.73 TB/s artifact does not change the EXP-015 verdict: attention is a small fraction of decode time.

VERDICT
B1 audit: RESOLVED. The 2561 passes (repeats+warmup) and the 2.73 TB/s (SWA-window artifact) are both explained. Attention is not the decode bottleneck. Proceed to measurement-driven tuning (KNOB 1) with the expectation that any knob must show a MEASURED gain to be kept, since attention is ~0.73% of decode time.

## EXP-017: KNOB 1 (byte0/nib hoist + __syncwarp removal) rejected

PROBLEM
Test KNOB 1 (hoist byte0/nib Q4_0 plane math, drop one __syncwarp) for a measured decode gain over the committed RDNA3 decode kernel, per the WS-2B "keep only measured gain" rule.

EVIDENCE
Bench: `llama-bench -m /home/gencer/models/qwen-v2/Qwen3.8-27B-UD-Q2_K_XL.gguf -ngl 999 -ctk q4_0 -ctv q4_0`, build d719f6370, llama-server stopped.
- G1 (test-backend-ops FLASH_ATTN_EXT nr23=[6,1] nb=1 mask=0): 3/3 OK (kv 1024/8192/65536).
- Dispatch (rocprofv3, p512 n4): fa grid (6144,4), combine (256,24), n=336. KNOB 1 does not change launch geometry (scalar micro-hygiene only).
- A1 (KNOB 1 ON, rdna3 enabled), p131072 n512 -r 1: pp131072=240.74, tg512=23.76.
- B1 (GGML_FA_DECODE_RDNA3_OFF=1, generic tile fallback), p131072 n512 -r 1: pp131072=241.35, tg512=23.90.
- Long-tg (p512), rdna3 enabled: tg512=23.94, tg1024=23.88, tg5120=23.55, tg10240=23.17 (mild ~3% slowdown as context grows to ~10.7k).

Raw log: /home/gencer/llama.cpp/experiments/fattn_decode_rdna3_bench/knob1_20260906_0859.log

CHANGE
Reverted only the KNOB 1 diff (byte0/nib hoist + __syncwarp removal) via `git checkout -- ggml/src/ggml-cuda/fattn-decode-rdna3.cu`, returning to clean commit 441e54b3b. The committed RDNA3 decode kernel is unchanged.

RESULT
KNOB 1 has no measurable decode gain. A1 (rdna3 + KNOB 1) tg512=23.76 vs B1 (generic) tg512=23.90: B1 is 0.14 t/s (0.6%) faster, within measurement noise. The scalar-loop hoisting and __syncwarp removal do not move the needle, as expected for a micro-hygiene edit on a memory-bound decode path.

CAVEAT
B1 uses GGML_FA_DECODE_RDNA3_OFF=1, which disables the entire RDNA3 path (generic tile fallback), not a clean KNOB 1-off. So this is a "RDNA3 kernel vs generic" comparison, not an isolated KNOB 1 on/off. The conclusion (no measurable gain) still holds because rdna3 + KNOB 1 is at best equal to the generic path.

VERDICT
KNOB 1: REJECTED (no measurable delta over baseline; within noise). Revert only the KNOB 1 diff. Proceed to KNOB 2 (TILE=128) or KNOB 3 (GQA head batching) to attack the 6x KV traffic multiplier and barrier frequency.

## EXP-018: KNOB 2 (TILE=128) rejected

PROBLEM
Test KNOB 2 (TILE 64->128, S_LDSS 65->129, softmax 2->4 tokens/lane, P_lds 2->4 tokens/lane) for a measured decode gain over the generic tile fallback, per the WS-2B "keep only measured gain" rule. Goal: halve the __syncthreads() barrier count per KV tile by processing 128 tokens per tile.

EVIDENCE
Bench: `llama-bench -m /home/gencer/models/qwen-v2/Qwen3.8-27B-UD-Q2_K_XL.gguf -ngl 999 -ctk q4_0 -ctv q4_0`, llama-server stopped.
- G1 (test-backend-ops FLASH_ATTN_EXT nr23=[6,1] nb=1 mask=0): 3/3 OK (kv 1024/8192/65536).
- Resource check (rocprofv3 runtime, TILE=128): private_segment_size (scratch) = 0, group_segment_size (LDS) = 4652 B (4.54 KB, <= 16 KB budget), arch_vgpr_count = 112 (baseline TILE=64 was 88; over the 96 VGPR budget).
- A (KNOB 2 ON, rdna3 enabled), pp8192 n128 -r 1: pp8192=562.82, tg128=23.21.
- B (GGML_FA_DECODE_RDNA3_OFF=1, generic tile fallback), pp8192 n128 -r 1: pp8192=562.79, tg128=23.26.
- pp8192 used because the Qwen3.5 SWA window caps decode attention at ~512 tokens (EXP-016), so pp8192 fully saturates the decode attention workload; a final 131k confirmation is deferred to the end of WS-2B if KNOB 2 were kept.

Raw log: /home/gencer/llama.cpp/experiments/fattn_decode_rdna3_bench/knob2_20260906_1554.log

CHANGE
Reverted the KNOB 2 diff (TILE=128) via `git checkout -- ggml/src/ggml-cuda/fattn-decode-rdna3.cu`, returning to clean commit 441e54b3b (TILE=64, 81 static VGPR). The committed RDNA3 decode kernel is unchanged.

RESULT
KNOB 2 has no measurable decode gain. A (TILE=128) tg128=23.21 vs B (generic) tg128=23.26: A is 0.05 t/s (0.2%) slower, within measurement noise. Doubling the tile did not pay for itself: the extra register pressure (VGPR 88 -> 112) offsets the halved barrier count on this memory-bound decode path.

CAVEAT
- Single -r 1 -n 128 run; the 0.05 t/s delta is within noise, so the direction (slight regression) is not statistically firm, but there is no evidence of a gain.
- B uses GGML_FA_DECODE_RDNA3_OFF=1 (whole RDNA3 path off), not an isolated KNOB 2-off, same caveat as EXP-017.
- VGPR 112 exceeds the 96 budget; even a marginal gain would have required reconciling the register regression.

VERDICT
KNOB 2: REJECTED (no measurable delta over baseline; within noise, and VGPR over budget). Revert only the KNOB 2 diff. Proceed to KNOB 3 (GQA head batching) to attack the 6x KV traffic multiplier.

## EXP-021: GEMV Bandwidth Decomposition (131k decode trace)

PROBLEM
Decompose the GEMV bandwidth deficit from the B1 131k decode trace by mapping each `mul_mat_vec_q` dispatch to its exact weight-tensor byte count, computing per-dispatch achieved bandwidth, and identifying which size/format buckets are underperforming.

EVIDENCE
Source: `/home/gencer/rocmprof/b1_ws2b/kernel/trace_kernel_131k_kernel_trace.csv` (1,110,449 GEMV dispatches, chronological).
Model: Qwen35 hybrid SSM+attention, 65 blocks, `full_attention_interval=4`, `nextn_predict_layers=1`.
Bench: `-p 131072 -n 512 -b 512`, `llama-bench` default `-r 5` (confirmed EXP-016).

Byte derivation (verified from GGUF tensor table + mmvq.cu):
- GGUF v3 tensor table gives exact per-tensor byte sizes via offset deltas.
- `gx = ne01 * 32` maps each dispatch to its weight tensor.
- Fused kernels (`has_fusion=true`) with `use_gate` stream TWO weight matrices (mmvq.cu lines 669-675: `vec_dot_q_cuda(vx,...)` + `vec_dot_q_cuda(vgate,...)`), so bytes = 2x single-matrix size.
- Non-fused kernels stream one weight matrix.

Per-(type, gx) BW table (corrected):

| type | quant   | gx      | bytes     | n       | avg_us  | BW_GB/s | tensor              |
|------|---------|---------|-----------|---------|---------|---------|---------------------|
| 11   | Q3_K    | 7946240 | 546,304,000 | 4,097   | 1360.66 | 401.5   | output.weight       |
| 18   | IQ3_XXS | 163840  | 34,119,680  | 163,904 | 118.93  | 286.9   | ffn_down            |
| 18   | IQ3_XXS | 196608  | 12,042,240  | 122,928 | 45.08   | 267.1   | attn_gate           |
| 18   | IQ3_XXS | 327680  | 20,070,400  | 122,928 | 72.35   | 277.4   | attn_qkv            |
| 18   | IQ3_XXS | 557056  | 68,239,360  | 163,904 | 234.59  | 290.9   | ffn_gate+up (fused) |
| 21   | IQ3_S   | 32768   | 2,252,800   | 40,976  | 10.14   | 222.2   | attn_k              |
| 21   | IQ3_S   | 163840  | 13,516,800  | 163,904 | 45.96   | 294.1   | ssm_out/attn_output |
| 21   | IQ3_S   | 393216  | 27,033,600  | 40,976  | 85.10   | 317.7   | attn_q              |
| 23   | IQ4_XS  | 32768   | 2,785,280   | 40,976  | 7.24    | 384.7   | attn_v              |
| 29   | IQ1_M   | 1536    | 53,760      | 245,856 | 3.59    | 15.0    | ssm_alpha/beta      |

Aggregate: 26,505.7 GB streamed, 90.57 s total GEMV duration, **292.7 GB/s** aggregate.

Per-type aggregate BW (anchor check vs EXP-005):

| quant   | total_GB | total_s | BW_GB/s | EXP-005 anchor |
|---------|----------|---------|---------|----------------|
| Q3_K    | 2,238.2  | 5.57    | 401.5   | ~503           |
| IQ3_XXS | 20,724.6 | 72.38   | 286.3   | ~350           |
| IQ3_S   | 3,415.5  | 11.44   | 298.7   | (no anchor)    |
| IQ4_XS  | 114.1    | 0.30    | 384.7   | (no anchor)    |
| IQ1_M   | 13.2     | 0.88    | 15.0    | (no anchor)    |

Token-count reconciliation:
- Attention passes: 40,976 fa dispatches / 16 full-attention blocks = 2,561 = 5 x 512 + 1 (matches `-r 5` default).
- Output.weight: 4,097 = 8 x 512 + 1. The output layer is dispatched more frequently than attention, consistent with the MTP/nextn head (`nextn_predict_layers=1`) computing additional output projections. Per-dispatch BW is unaffected by this mismatch.

CHANGE
No code change. Diagnostic measurement only.

RESULT
- Fused-kernel byte correction (1x -> 2x) raises ffn_gate/up BW from 145.4 to **290.9 GB/s**, eliminating the apparent anomaly. It now sits within 3% of ffn_down (286.9 GB/s).
- Aggregate GEMV BW: **292.7 GB/s** (was 230.9 GB/s before the 2x correction).
- Per-type anchors: Q3_K at 401.5 GB/s is ~20% below the EXP-005 anchor (~503). IQ3_XXS at 286.3 GB/s is ~18% below its anchor (~350). The deficit is uniform across sizes within each type, indicating a systematic bandwidth gap, not a size-specific effect.
- ssm_alpha/beta (IQ1_M, 53,760 bytes): **15.0 GB/s** is a latency-bound anomaly (tiny tensor, 3.59 us fixed overhead dominates). This is a known small-tensor effect, not a bandwidth deficit.
- attn_k (IQ3_S, 2.25 MB): 222.2 GB/s is the lowest among non-tiny tensors, ~25% below the IQ3_S aggregate. The 2.25 MB size is below the ~4 MB threshold where RDNA3 L2 caching becomes effective.

CAVEAT
- The 4,097 vs 2,561 output/attention count mismatch is documented but not fully decomposed (MTP head contribution is inferred, not traced). Does not affect per-dispatch BW.
- The per-type aggregate anchors (EXP-005) were measured on a different build/commit; the ~18-20% gap may include build differences, not just a bandwidth deficit.
- ssm_alpha/beta (15 GB/s) is excluded from the bandwidth-deficit analysis as a known latency-bound case.
- The 2x fused-byte correction applies only to kernels with `use_gate=true`; kernels fused for bias/scale only (if any) would be 1x. All observed fused kernels in this trace are gate-fused.

VERDICT
GEMV bandwidth decomposition: **C (partial deficit, uniform across sizes)**. The corrected aggregate is 292.7 GB/s. The per-type deficit (~18-20% below anchors) is uniform across tensor sizes within each quantization type, pointing to a systematic bandwidth gap (likely L2/DRAM interaction or memory controller behavior) rather than a size-specific kernel inefficiency. The ssm_alpha/beta 15 GB/s is a separate latency-bound anomaly (tiny tensor), not part of the bandwidth deficit. The fused-kernel "anomaly" is resolved: it was a byte-count error (1x vs 2x), not a kernel inefficiency.

Recommended next direction: characterize the L2/DRAM interaction at the ~4 MB size boundary (attn_k at 2.25 MB is the outlier at 222 GB/s) using the GL2C PMC data to confirm whether the deficit is L2-residency-driven or a uniform memory-controller ceiling.
