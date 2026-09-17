#include "fattn-prefill-d256-rdna3.cuh"
#include "fattn-common.cuh"
#include "mma.cuh"
#include <cfloat>

using namespace ggml_cuda_mma;

// RDNA3 has a single hardware exp2 instruction, C++ expf expands to a ~30-instruction software sequence.
#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float fa_expf(float x) {
    return __builtin_amdgcn_exp2f(x * 1.4426950408889634f);
}
#else
#define fa_expf expf
#endif

// Two nibbles packed as f16 denormal bit patterns (0x000n per half, value n*2^-24).
// Two exact power-of-2 multiplies promote the denormals to n, one packed fma applies d and -8*d.
static __device__ __forceinline__ half2 dequant_nibbles(const uint32_t p, const half2 d_h2, const half2 dm_h2) {
    const half2 n = *reinterpret_cast<const half2 *>(&p);
    return __hfma2(__hmul2(__hmul2(n, make_half2(1024.0f, 1024.0f)), make_half2(16384.0f, 16384.0f)), d_h2, dm_h2);
}

// Dequantize 4 packed Q4_0 nibble-dwords (32 dims) into 16 half2 at tk.
// Split-plane Q4_0: byte b holds dim b (low nibble) and dim b+16 (high nibble).
static __device__ __forceinline__ void dequant_q4_0_h2_qs(const uint32_t * qs32, const half2 d_h2, const half2 dm_h2, half2 * tk) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t v  = qs32[j];
        const uint32_t t  = v  >> 8;
        const uint32_t vh = v  >> 4;
        // p01/p23: low-nibble plane, dims 4j+0..4j+3. q01/q23: high-nibble plane, dims 16+4j+0..+3.
        const uint32_t p01 = (v  & 0x0000000F) | ((v  << 8) & 0x000F0000);
        const uint32_t p23 = ((v  >> 16) & 0x000F) | ( t        & 0x000F0000);
        const uint32_t q01 = (vh & 0x0000000F) | ((vh << 8) & 0x000F0000);
        const uint32_t q23 = ((vh >> 16) & 0x000F) | ((t  >> 4) & 0x000F0000);
        tk[2*j]         = dequant_nibbles(p01, d_h2, dm_h2);
        tk[2*j + 1]     = dequant_nibbles(p23, d_h2, dm_h2);
        tk[8 + 2*j]     = dequant_nibbles(q01, d_h2, dm_h2);
        tk[8 + 2*j + 1] = dequant_nibbles(q23, d_h2, dm_h2);
    }
}

// Load one raw Q4_0 block (18 bytes: ggml_half d + 16 bytes of qs) into 5 dwords:
// p[0] holds d in the low half, p[1..4] hold the qs dwords (bytes 2..17 of the block).
static __device__ __forceinline__ void load_q4_0_block_regs(const char * pb, uint32_t * p) {
    p[0] = *reinterpret_cast<const uint16_t *>(pb);
    p[1] = *reinterpret_cast<const uint32_t *>(pb + 2);
    p[2] = *reinterpret_cast<const uint32_t *>(pb + 6);
    p[3] = *reinterpret_cast<const uint32_t *>(pb + 10);
    p[4] = *reinterpret_cast<const uint32_t *>(pb + 14);
}

// Dequantize one raw Q4_0 block held in 5 dwords into 16 half2 at tk.
static __device__ __forceinline__ void dequant_q4_0_h2_regs(const uint32_t * p, half2 * tk) {
    const half d = (*reinterpret_cast<const half2 *>(p)).x;
    dequant_q4_0_h2_qs(p + 1, __half2half2(d), __half2half2(__float2half(-8.0f * __half2float(d))), tk);
}

// WMMA prefill Flash Attention for RDNA3 (gfx1101), head_dim 256, Q4_0 KV, GQA ratio 6.
// 192 threads (6 warps of Wave32). 1 block = 1 KV head x 96 Q cols (16 tokens x 6 GQA heads).
// KV_TILE = 16 (16 KV rows per tile). LDS layout: tile_KV is SHARED between full K and full V
// (8.44 KB), total LDS = 58.5 KB (<64 KB), zero scratch spill, 2x wave occupancy vs KV_TILE = 64.
// Raw K/V blocks (18 bytes each) are prefetched into registers (5 dwords per thread)
// ahead of the barriers, so global load latency overlaps WMMA compute.
// 4 __syncthreads() per 16-token KV block: tile_KV is reused for K and V,
// so all reads must complete before each overwrite.

template <int DKV>
__launch_bounds__(192, 1)
static __global__ void flash_attn_prefill_d256_rdna3_kernel(
        const float2 * Q,
        const char   * K,
        const char   * V,
        const half   * mask,
        float2       * dst,
        const float scale,
        const int n_kv_head,
        const int ne01,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_mask,
        const int stride_dst1,
        const int stride_dst2,
        const int nb11,
        const int nb12,
        const int nb21,
        const int nb22) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    using T_A_KQ  = tile<16, 8,  half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_B_KQ  = tile<16, 8,  half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;
    using T_A_VKQ = tile<16, 8,  half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_B_VKQ = tile<16, 8,  half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;

    constexpr int N_TOKENS = 16;
    constexpr int N_GQA    = 6;
    constexpr int NCOLS    = N_TOKENS * N_GQA; // 96 cols
    constexpr int KV_TILE  = 16;
    constexpr int N_CH     = DKV / 64;         // 4 chunks
    constexpr int S_Q      = DKV / 2 + 4;      // 132
    constexpr int S_KV     = DKV / 2 + 4;      // 132 (full head dim in half2 + 4 pad)
    constexpr int S_MASK   = KV_TILE + 8;      // 24

    // Total LDS = 50,688 + 8,448 + 768 = 59,904 bytes (58.5 KB < 65,536 bytes)
    // 16-byte alignment keeps every row base legal for ds_read_b128 (all row strides are 16B multiples).
    __shared__ __align__(16) half2 tile_Q   [NCOLS    * S_Q];
    __shared__ __align__(16) half2 tile_KV  [KV_TILE  * S_KV];
    __shared__ __align__(16) half  tile_mask[N_TOKENS * S_MASK];

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp    = tid >> 5; // Exactly 6 warps: 0..5
    const int qcol    = warp * 16 + (lane & 15);
    const int ntile   = (ne01 + N_TOKENS - 1) / N_TOKENS;
    const int jt      = blockIdx.x % ntile;
    const int kv_head = blockIdx.x / ntile;

    // 1. Cooperative Load Q (96 cols x 128 half2 = 12288 elements / 192 threads = 64 per thread)
    {
        const int gc = tid >> 3; // 24 col-groups (4 cols each)
        const int gk = tid & 7;  // 8 k-groups (16 k's each)
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int jc = gc * 4 + l;
            const int j  = jc / N_GQA;
            const int c  = jc % N_GQA;
            half2 * tq = tile_Q + jc * S_Q;
            if (jt * N_TOKENS + j < ne01) {
                const float2 * Qp = Q + (jt * N_TOKENS + j) * stride_Q1 + (kv_head * N_GQA + c) * stride_Q2;
                #pragma unroll
                for (int k0 = 0; k0 < 16; ++k0) {
                    const int k = gk * 16 + k0;
                    const float2 tmp = Qp[k];
                    tq[k] = make_half2(tmp.x * scale, tmp.y * scale);
                }
            } else {
                #pragma unroll
                for (int k0 = 0; k0 < 16; ++k0) {
                    tq[gk * 16 + k0] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }
    __syncthreads();

    float KQ_max    = -FLT_MAX / 2.0f;
    float KQ_rowsum = 0.0f;

    T_C_VKQ VKQ_C[16];
    #pragma unroll
    for (int f = 0; f < 16; ++f) {
        #pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) {
            VKQ_C[f].x[l] = 0.0f;
        }
    }

    const char * K_head = K + nb12 * kv_head;
    const char * V_head = V + nb22 * kv_head;
    const int n_kv_blocks = (ne11 + KV_TILE - 1) / KV_TILE;

    // Register prefetch buffers: one raw Q4_0 block per thread, packed into 5 dwords.
    // Load latency is hidden by the WMMA compute between the barrier that issues the load
    // and the barrier after which the data is consumed.
    uint32_t pref_k[5];
    uint32_t pref_v[5];
    {
        const int row = tid >> 3;
        const int b   = tid & 7;
        if (tid < 128 && row < min(KV_TILE, ne11)) {
            load_q4_0_block_regs((const char *)(K_head + nb11 * row + b * sizeof(block_q4_0)), pref_k);
        }
    }

    for (int kb = 0; kb < n_kv_blocks; ++kb) {
        const int k_sup = min(KV_TILE, ne11 - kb * KV_TILE);

        // 2. Unpack prefetched raw K into tile_KV + load mask
        if (tid < 128) {
            const int row = tid >> 3;
            const int b   = tid & 7;
            half2 * tk = tile_KV + row * S_KV + b * 16;
            if (row < k_sup) {
                dequant_q4_0_h2_regs(pref_k, tk);
            } else {
                #pragma unroll
                for (int l = 0; l < 16; ++l) {
                    tk[l] = make_half2(0.0f, 0.0f);
                }
            }
        }

        if (mask != nullptr) {
            for (int it = 0; it < 2; ++it) {
                const int midx = tid + it * 192;
                if (midx < 256) {
                    const int j  = midx >> 4;
                    const int i  = midx & 15;
                    const int jv = jt * N_TOKENS + j;
                    tile_mask[j * S_MASK + i] = (i < k_sup && jv < ne01) ?
                        mask[(int64_t) jv * stride_mask + kb * KV_TILE + i] : half(0.0f);
                }
            }
        }

        __syncthreads(); // BARRIER 1: Full K and Mask ready

        // Prefetch raw V of this tile: in flight while WMMA computes Q x K^T
        if (tid < 128) {
            const int row = tid >> 3;
            const int b   = tid & 7;
            if (row < k_sup) {
                load_q4_0_block_regs((const char *)(V_head + nb21 * (kb * KV_TILE + row) + b * sizeof(block_q4_0)), pref_v);
            }
        }

        // 3. WMMA Q x K^T: All 6 warps active
        T_C_KQ KQ_C[1];
        #pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            KQ_C[0].x[l] = 0.0f;
        }

        #pragma unroll
        for (int ch = 0; ch < N_CH; ++ch) {
            #pragma unroll
            for (int k0 = 0; k0 < 32; k0 += 8) {
                T_B_KQ Q_B;
                load_ldmatrix(Q_B, tile_Q + warp * 16 * S_Q + ch * 32 + k0, S_Q);
                T_A_KQ A;
                load_ldmatrix(A, tile_KV + ch * 32 + k0, S_KV);
                mma(KQ_C[0], A, Q_B);
            }
        }

        // 4. Softmax (purely warp-local)
        if (mask != nullptr) {
            const int j = qcol / N_GQA;
            #pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int i = 2*l + (lane >> 4);
                KQ_C[0].x[l] += __half2float(tile_mask[j * S_MASK + i]);
            }
        }

        float KQ_max_new = KQ_max;
        #pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (2*l + (lane >> 4) < k_sup) {
                KQ_max_new = fmaxf(KQ_max_new, KQ_C[0].x[l] + FATTN_KQ_MAX_OFFSET);
            }
        }
        KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, 16, 32));
        const float KQ_max_diff = KQ_max - KQ_max_new;
        float KQ_max_scale = fa_expf(KQ_max_diff);
        KQ_max = KQ_max_new;
        *((uint32_t *) &KQ_max_scale) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

        #pragma unroll
        for (int f = 0; f < 16; ++f) {
            #pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[f].x[l] = VKQ_C[f].x[l] * KQ_max_scale;
            }
        }

        float rowsum_add = 0.0f;
        #pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (2*l + (lane >> 4) < k_sup) {
                KQ_C[0].x[l] = fa_expf(KQ_C[0].x[l] - KQ_max);
                rowsum_add += KQ_C[0].x[l];
            } else {
                KQ_C[0].x[l] = 0.0f;
            }
        }
        rowsum_add += __shfl_xor_sync(0xFFFFFFFF, rowsum_add, 16, 32);
        KQ_rowsum = KQ_max_scale * KQ_rowsum + rowsum_add;

        // Convert scores to B fragments for P x V
        T_B_VKQ B[1];
        B[0] = get_half2(KQ_C[0]);

        // K reads done, tile_KV can now be overwritten with V
        __syncthreads(); // BARRIER 2

        // 5. Unpack prefetched raw V into tile_KV (safely overwriting K)
        if (tid < 128) {
            const int row = tid >> 3;
            const int b   = tid & 7;
            half2 * tv = tile_KV + row * S_KV + b * 16;
            if (row < k_sup) {
                dequant_q4_0_h2_regs(pref_v, tv);
            } else {
                #pragma unroll
                for (int l = 0; l < 16; ++l) {
                    tv[l] = make_half2(0.0f, 0.0f);
                }
            }
        }

        __syncthreads(); // BARRIER 3: Full V ready

        // Prefetch raw K of the next tile: in flight while WMMA computes P x V
        if (kb + 1 < n_kv_blocks) {
            const int row = tid >> 3;
            const int b   = tid & 7;
            const int k_sup_next = min(KV_TILE, ne11 - (kb + 1) * KV_TILE);
            if (tid < 128 && row < k_sup_next) {
                load_q4_0_block_regs((const char *)(K_head + nb11 * ((kb + 1) * KV_TILE + row) + b * sizeof(block_q4_0)), pref_k);
            }
        }

        // 6. WMMA P x V: Accumulate into VKQ_C (FP32)
        #pragma unroll
        for (int i_v = 0; i_v < DKV; i_v += 16) {
            T_A_VKQ A;
            load_ldmatrix_trans(A, tile_KV + i_v/2, S_KV);
            mma(VKQ_C[i_v/16], A, B[0]);
        }

        // V reads done, next KV block can overwrite tile_KV with K
        __syncthreads(); // BARRIER 4
    }

    // Epilogue: Normalize and write output to global memory
    const float inv = 1.0f / KQ_rowsum;
    const int j = qcol / N_GQA;
    const int c = qcol % N_GQA;

    // Rematerialize jt and kv_head locally in epilogue to kill the long live range from prologue
    const int jt_epi      = blockIdx.x % ntile;
    const int kv_head_epi = blockIdx.x / ntile;

    if (jt_epi * N_TOKENS + j < ne01) {
        float2 * dstp = dst + (jt_epi * N_TOKENS + j) * stride_dst2 + (kv_head_epi * N_GQA + c) * stride_dst1;
        #pragma unroll
        for (int f = 0; f < 16; ++f) {
            #pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                const int idx = f*8 + l;
                if ((lane >> 4) == 0) {
                    dstp[idx].x = VKQ_C[f].x[l] * inv;
                } else {
                    dstp[idx].y = VKQ_C[f].x[l] * inv;
                }
            }
        }
    }
#endif
}

bool ggml_cuda_flash_attn_ext_prefill_d256_rdna3(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    static const bool disabled = (getenv("GGML_FA_PREFILL_RDNA3_OFF") != nullptr);
    if (disabled) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // Attention sinks and batched queries are not supported, use the generic kernels instead.
    if (dst->src[4] != nullptr || Q->ne[3] != 1) {
        return false;
    }
    if (mask != nullptr && mask->type != GGML_TYPE_F16) {
        return false;
    }

    // Hardware runtime check: gfx1101 only
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (cc != GGML_CUDA_CC_OFFSET_AMD + 0x1101) {
        return false;
    }

    const int64_t ne00 = Q->ne[0]; // 256
    const int64_t ne01 = Q->ne[1]; // n_tokens
    const int64_t ne11 = K->ne[1]; // n_kv
    const int64_t ne02 = Q->ne[2]; // 24
    const int64_t ne12 = K->ne[2]; // 4

    GGML_ASSERT(ne00 == 256);
    GGML_ASSERT(ne02 / ne12 == 6);
    GGML_ASSERT(K->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(V->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(K->nb[0] == sizeof(block_q4_0) && V->nb[0] == sizeof(block_q4_0));

    float scale = 1.0f / sqrtf(float(ne00));
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    {
        const float * params = (const float *) dst->op_params;
        scale = params[0];
        max_bias = params[1];
        logit_softcap = params[2];
    }
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    const int ntile = (ne01 + 16 - 1) / 16;
    const dim3 blocks(ne12 * ntile, 1, 1);
    const dim3 threads(192, 1, 1);

    cudaStream_t stream = ctx.stream();

    flash_attn_prefill_d256_rdna3_kernel<256><<<blocks, threads, 0, stream>>>(
        (const float2 *) Q->data,
        (const char   *) K->data,
        (const char   *) V->data,
        mask ? (const half *) mask->data : nullptr,
        (float2       *) dst->data,
        scale,
        ne12,
        ne01,
        ne11,
        Q->nb[1] / sizeof(float2),
        Q->nb[2] / sizeof(float2),
        mask ? mask->nb[1] / sizeof(half) : 0,
        dst->nb[1] / sizeof(float2),
        dst->nb[2] / sizeof(float2),
        K->nb[1],
        K->nb[2],
        V->nb[1],
        V->nb[2]
    );
    return true;
}
