#include "fattn-decode-rdna3.cuh"
#include "fattn-common.cuh"

// Decode attention for RDNA3 (gfx1101) with Q4_0 KV cache, head_dim 256, single query.
// Fused dequant in registers, split-KV over pooled partials, combine via flash_attn_combine_results.
//
// Thread/wave layout (256 threads, wave size 32):
//   warp w covers dims [32*w, 32*w + 32) of the head, one Q4_0 block per token
//   lane l = (grp, l7): token in a 4-token group is l >> 3, dim group l & 7
//   each lane owns dims [32*w + 4*l7, 32*w + 4*l7 + 4)
static __global__ void flash_attn_decode_rdna3(
        const char * Q,
        const char * K,
        const char * V,
        float  * O_partial,
        float2 * meta,
        const float scale,
        const int gqa_ratio,
        const int ne11,
        const int split_len,
        const int nb02,
        const int nb11,
        const int nb12,
        const int nb21,
        const int nb22) {
    constexpr int D       = 256;
    constexpr int TILE    = 64;
    constexpr int N_WARPS = D / 32;
    constexpr int S_LDSS  = 65; // padded row stride: warp*65 % 32 is unique per warp
    constexpr float S_OOB = -1e30f;

    __shared__ float S_lds[N_WARPS*S_LDSS];
    __shared__ float P_lds[TILE];
    __shared__ float m_shared;
    __shared__ float lsum_shared;
    __shared__ float r_shared;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int grp  = lane >> 3;
    const int l7   = lane & 7;

    const int head  = blockIdx.x;
    const int split = blockIdx.y;

    const int kv_head = head / gqa_ratio;

    const char * K_head = K + nb12*kv_head;
    const char * V_head = V + nb22*kv_head;

    // Dims [32*warp + 4*l7, 32*warp + 4*l7 + 4) for this lane, scaled once.
    const float4 q4 = ((const float4 *)(Q + nb02*head))[warp*8 + l7];
    const float q0 = q4.x * scale;
    const float q1 = q4.y * scale;
    const float q2 = q4.z * scale;
    const float q3 = q4.w * scale;

    float O_acc0 = 0.0f;
    float O_acc1 = 0.0f;
    float O_acc2 = 0.0f;
    float O_acc3 = 0.0f;

    if (tid == 0) {
        m_shared    = -FLT_MAX/2.0f;
        lsum_shared = 0.0f;
        r_shared    = 0.0f;
    }
    __syncthreads();

    const int s0 = split*split_len;
    const int e0 = s0 + split_len < ne11 ? s0 + split_len : ne11;
    const int ntiles = (e0 - s0 + TILE - 1) / TILE;

    for (int tile = 0; tile < ntiles; ++tile) {
        const int t0 = s0 + tile*TILE;

        // KQ: lane (grp, l7) partial-dots dims [4*l7, 4*l7 + 4) of token t0 + 4*grp.
#pragma unroll
        for (int g = 0; g < TILE/4; ++g) {
            const int token = t0 + 4*g + grp;
            const bool valid = token < e0;

            float partial = 0.0f;
            if (valid) {
                const block_q4_0 * blk = (const block_q4_0 *)(K_head + nb11*token) + warp;
                const float d  = __half2float(blk->d);
                const float dm = -8.0f*d;
                // Split-plane Q4_0: byte b holds dim b (low nibble) and dim b+16 (high nibble).
                // Lane l7 owns dims 4*l7..4*l7+3: low plane for l7<4, high plane for l7>=4.
                const int byte0 = 4 * (l7 & 3);
                const uint32_t v = *(const uint16_t *)(blk->qs + byte0)
                                  | (uint32_t(*(const uint16_t *)(blk->qs + byte0 + 2)) << 16);
                const int nib = (l7 & 4) ? 4 : 0;
                partial = fmaf(q0, d*float((v >> (0*8 + nib))  & 0x0F) + dm, partial);
                partial = fmaf(q1, d*float((v >> (1*8 + nib))  & 0x0F) + dm, partial);
                partial = fmaf(q2, d*float((v >> (2*8 + nib))  & 0x0F) + dm, partial);
                partial = fmaf(q3, d*float((v >> (3*8 + nib))  & 0x0F) + dm, partial);
            }
            partial += __shfl_down(partial, 1, 8);
            partial += __shfl_down(partial, 2, 8);
            partial += __shfl_down(partial, 4, 8);
            if (l7 == 0) {
                S_lds[warp*S_LDSS + 4*g + grp] = valid ? partial : S_OOB;
            }
        }
        __syncthreads();

        // Softmax over the 64 tokens of the tile, warp 0 only.
        if (warp == 0) {
            float s0_ = 0.0f;
            float s1_ = 0.0f;
#pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                s0_ += S_lds[w*S_LDSS + 2*lane];
                s1_ += S_lds[w*S_LDSS + 2*lane + 1];
            }
            float mx = fmaxf(s0_, s1_);
            mx = fmaxf(mx, __shfl_down(mx, 16, 32));
            mx = fmaxf(mx, __shfl_down(mx, 8, 32));
            mx = fmaxf(mx, __shfl_down(mx, 4, 32));
            mx = fmaxf(mx, __shfl_down(mx, 2, 32));
            mx = fmaxf(mx, __shfl_down(mx, 1, 32));
            mx = __shfl(mx, 0, 32);

            const float m_old = m_shared;
            __syncwarp();
            const float m_new = fmaxf(m_old, mx);
            const float e0 = expf(s0_ - m_new);
            const float e1 = expf(s1_ - m_new);
            float sum = e0 + e1;
            sum += __shfl_down(sum, 16, 32);
            sum += __shfl_down(sum, 8, 32);
            sum += __shfl_down(sum, 4, 32);
            sum += __shfl_down(sum, 2, 32);
            sum += __shfl_down(sum, 1, 32);

            if (lane == 0) {
                const float r = expf(m_old - m_new);
                lsum_shared = lsum_shared*r + sum;
                m_shared    = m_new;
                r_shared    = r;
            }
            P_lds[2*lane]     = e0;
            P_lds[2*lane + 1] = e1;
        }
        __syncthreads();

        // V: rescale the accumulator, then O_acc += P * V.
        O_acc0 *= r_shared;
        O_acc1 *= r_shared;
        O_acc2 *= r_shared;
        O_acc3 *= r_shared;
#pragma unroll
        for (int g = 0; g < TILE/4; ++g) {
            const int token = t0 + 4*g + grp;
            if (token < e0) {
                const float p = P_lds[4*g + grp];
                const block_q4_0 * blk = (const block_q4_0 *)(V_head + nb21*token) + warp;
                const float d  = __half2float(blk->d);
                const float dm = -8.0f*d;
                const int byte0 = 4 * (l7 & 3);
                const uint32_t v = *(const uint16_t *)(blk->qs + byte0)
                                  | (uint32_t(*(const uint16_t *)(blk->qs + byte0 + 2)) << 16);
                const int nib = (l7 & 4) ? 4 : 0;
                O_acc0 = fmaf(p, d*float((v >> (0*8 + nib)) & 0x0F) + dm, O_acc0);
                O_acc1 = fmaf(p, d*float((v >> (1*8 + nib)) & 0x0F) + dm, O_acc1);
                O_acc2 = fmaf(p, d*float((v >> (2*8 + nib)) & 0x0F) + dm, O_acc2);
                O_acc3 = fmaf(p, d*float((v >> (3*8 + nib)) & 0x0F) + dm, O_acc3);
            }
        }
        __syncthreads();
    }

    // Merge the 4 token groups (grp) that cover the same dims.
    O_acc0 += __shfl_down(O_acc0, 8, 32);
    O_acc0 += __shfl_down(O_acc0, 16, 32);
    O_acc1 += __shfl_down(O_acc1, 8, 32);
    O_acc1 += __shfl_down(O_acc1, 16, 32);
    O_acc2 += __shfl_down(O_acc2, 8, 32);
    O_acc2 += __shfl_down(O_acc2, 16, 32);
    O_acc3 += __shfl_down(O_acc3, 8, 32);
    O_acc3 += __shfl_down(O_acc3, 16, 32);

    if (grp == 0) {
        const int j = (head*(int)gridDim.y + split)*D + 32*warp + 4*l7;
        *(float4 *)(O_partial + j) = make_float4(O_acc0, O_acc1, O_acc2, O_acc3);
    }
    if (tid == 0) {
        meta[head*(int)gridDim.y + split] = make_float2(m_shared, lsum_shared);
    }
}

// GQA head batching: one block per (kv_head, split) processes all gqa_ratio query heads.
// K/V tiles are loaded and dequantized once per tile, reused across the heads.
// Grid: (n_kv_heads, n_splits) instead of (n_heads, n_splits): 6x less KV traffic.
// Per-head arithmetic is identical to flash_attn_decode_rdna3 (bit-exact partials).
template<int NH>
static __global__ void flash_attn_decode_rdna3_gqa(
        const char * Q,
        const char * K,
        const char * V,
        float  * O_partial,
        float2 * meta,
        const float scale,
        const int gqa_ratio,
        const int ne11,
        const int split_len,
        const int nb02,
        const int nb11,
        const int nb12,
        const int nb21,
        const int nb22) {
    constexpr int D       = 256;
    constexpr int TILE    = 64;
    constexpr int N_WARPS = D / 32;
    constexpr int S_LDSS  = 65;
    constexpr float S_OOB = -1e30f;

    __shared__ float S_lds[NH][N_WARPS*S_LDSS];
    __shared__ float P_lds[NH][TILE];
    __shared__ float m_shared[NH];
    __shared__ float lsum_shared[NH];
    __shared__ float r_shared[NH];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int grp  = lane >> 3;
    const int l7   = lane & 7;

    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;

    const char * K_head = K + nb12*kv_head;
    const char * V_head = V + nb22*kv_head;

    // Dims [32*warp + 4*l7, 32*warp + 4*l7 + 4) of each of the NH q heads, scaled once.
    float q0[NH], q1[NH], q2[NH], q3[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        const float4 q4 = ((const float4 *)(Q + nb02*(kv_head*gqa_ratio + h)))[warp*8 + l7];
        q0[h] = q4.x * scale;
        q1[h] = q4.y * scale;
        q2[h] = q4.z * scale;
        q3[h] = q4.w * scale;
    }

    float O_acc0[NH], O_acc1[NH], O_acc2[NH], O_acc3[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        O_acc0[h] = 0.0f;
        O_acc1[h] = 0.0f;
        O_acc2[h] = 0.0f;
        O_acc3[h] = 0.0f;
    }

    if (tid == 0) {
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            m_shared[h]    = -FLT_MAX/2.0f;
            lsum_shared[h] = 0.0f;
            r_shared[h]    = 0.0f;
        }
    }
    __syncthreads();

    const int s0 = split*split_len;
    const int e0 = s0 + split_len < ne11 ? s0 + split_len : ne11;
    const int ntiles = (e0 - s0 + TILE - 1) / TILE;

    for (int tile = 0; tile < ntiles; ++tile) {
        const int t0 = s0 + tile*TILE;

        // KQ: dequant once per (g, lane), partial-dots against all NH heads.
#pragma unroll
        for (int g = 0; g < TILE/4; ++g) {
            const int token = t0 + 4*g + grp;
            const bool valid = token < e0;

            float partial[NH];
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                partial[h] = 0.0f;
            }
            if (valid) {
                const block_q4_0 * blk = (const block_q4_0 *)(K_head + nb11*token) + warp;
                const float d  = __half2float(blk->d);
                const float dm = -8.0f*d;
                const int byte0 = 4 * (l7 & 3);
                const uint32_t v = *(const uint16_t *)(blk->qs + byte0)
                                  | (uint32_t(*(const uint16_t *)(blk->qs + byte0 + 2)) << 16);
                const int nib = (l7 & 4) ? 4 : 0;
                const float k0 = d*float((v >> (0*8 + nib)) & 0x0F) + dm;
                const float k1 = d*float((v >> (1*8 + nib)) & 0x0F) + dm;
                const float k2 = d*float((v >> (2*8 + nib)) & 0x0F) + dm;
                const float k3 = d*float((v >> (3*8 + nib)) & 0x0F) + dm;
#pragma unroll
                for (int h = 0; h < NH; ++h) {
                    partial[h] = fmaf(q0[h], k0, partial[h]);
                    partial[h] = fmaf(q1[h], k1, partial[h]);
                    partial[h] = fmaf(q2[h], k2, partial[h]);
                    partial[h] = fmaf(q3[h], k3, partial[h]);
                }
            }
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                partial[h] += __shfl_down(partial[h], 1, 8);
                partial[h] += __shfl_down(partial[h], 2, 8);
                partial[h] += __shfl_down(partial[h], 4, 8);
                if (l7 == 0) {
                    S_lds[h][warp*S_LDSS + 4*g + grp] = valid ? partial[h] : S_OOB;
                }
            }
        }
        __syncthreads();

        // Softmax over the 64 tokens of the tile, per head, warp 0 only.
        if (warp == 0) {
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                float s0_ = 0.0f;
                float s1_ = 0.0f;
#pragma unroll
                for (int w = 0; w < N_WARPS; ++w) {
                    s0_ += S_lds[h][w*S_LDSS + 2*lane];
                    s1_ += S_lds[h][w*S_LDSS + 2*lane + 1];
                }
                float mx = fmaxf(s0_, s1_);
                mx = fmaxf(mx, __shfl_down(mx, 16, 32));
                mx = fmaxf(mx, __shfl_down(mx, 8, 32));
                mx = fmaxf(mx, __shfl_down(mx, 4, 32));
                mx = fmaxf(mx, __shfl_down(mx, 2, 32));
                mx = fmaxf(mx, __shfl_down(mx, 1, 32));
                mx = __shfl(mx, 0, 32);

                const float m_old = m_shared[h];
                __syncwarp();
                const float m_new = fmaxf(m_old, mx);
                const float e0 = expf(s0_ - m_new);
                const float e1 = expf(s1_ - m_new);
                float sum = e0 + e1;
                sum += __shfl_down(sum, 16, 32);
                sum += __shfl_down(sum, 8, 32);
                sum += __shfl_down(sum, 4, 32);
                sum += __shfl_down(sum, 2, 32);
                sum += __shfl_down(sum, 1, 32);

                if (lane == 0) {
                    const float r = expf(m_old - m_new);
                    lsum_shared[h] = lsum_shared[h]*r + sum;
                    m_shared[h]    = m_new;
                    r_shared[h]    = r;
                }
                P_lds[h][2*lane]     = e0;
                P_lds[h][2*lane + 1] = e1;
            }
        }
        __syncthreads();

        // V: rescale the accumulators, then O_acc += P * V. V dequant shared across heads.
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            O_acc0[h] *= r_shared[h];
            O_acc1[h] *= r_shared[h];
            O_acc2[h] *= r_shared[h];
            O_acc3[h] *= r_shared[h];
        }
#pragma unroll
        for (int g = 0; g < TILE/4; ++g) {
            const int token = t0 + 4*g + grp;
            if (token < e0) {
                const block_q4_0 * blk = (const block_q4_0 *)(V_head + nb21*token) + warp;
                const float d  = __half2float(blk->d);
                const float dm = -8.0f*d;
                const int byte0 = 4 * (l7 & 3);
                const uint32_t v = *(const uint16_t *)(blk->qs + byte0)
                                  | (uint32_t(*(const uint16_t *)(blk->qs + byte0 + 2)) << 16);
                const int nib = (l7 & 4) ? 4 : 0;
                const float v0 = d*float((v >> (0*8 + nib)) & 0x0F) + dm;
                const float v1 = d*float((v >> (1*8 + nib)) & 0x0F) + dm;
                const float v2 = d*float((v >> (2*8 + nib)) & 0x0F) + dm;
                const float v3 = d*float((v >> (3*8 + nib)) & 0x0F) + dm;
#pragma unroll
                for (int h = 0; h < NH; ++h) {
                    const float p = P_lds[h][4*g + grp];
                    O_acc0[h] = fmaf(p, v0, O_acc0[h]);
                    O_acc1[h] = fmaf(p, v1, O_acc1[h]);
                    O_acc2[h] = fmaf(p, v2, O_acc2[h]);
                    O_acc3[h] = fmaf(p, v3, O_acc3[h]);
                }
            }
        }
        __syncthreads();
    }

    // Merge the 4 token groups (grp) that cover the same dims, per head.
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        O_acc0[h] += __shfl_down(O_acc0[h], 8, 32);
        O_acc0[h] += __shfl_down(O_acc0[h], 16, 32);
        O_acc1[h] += __shfl_down(O_acc1[h], 8, 32);
        O_acc1[h] += __shfl_down(O_acc1[h], 16, 32);
        O_acc2[h] += __shfl_down(O_acc2[h], 8, 32);
        O_acc2[h] += __shfl_down(O_acc2[h], 16, 32);
        O_acc3[h] += __shfl_down(O_acc3[h], 8, 32);
        O_acc3[h] += __shfl_down(O_acc3[h], 16, 32);
    }

    if (grp == 0) {
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            const int head = kv_head*gqa_ratio + h;
            const int j = (head*(int)gridDim.y + split)*D + 32*warp + 4*l7;
            *(float4 *)(O_partial + j) = make_float4(O_acc0[h], O_acc1[h], O_acc2[h], O_acc3[h]);
        }
    }
    if (tid == 0) {
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            const int head = kv_head*gqa_ratio + h;
            meta[head*(int)gridDim.y + split] = make_float2(m_shared[h], lsum_shared[h]);
        }
    }
}

bool ggml_cuda_flash_attn_ext_decode_rdna3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool disabled = (getenv("GGML_FA_DECODE_RDNA3_OFF") != nullptr);
    if (disabled) {
        return false;
    }

    // GQA head batching is the default decode path; opt out with GGML_FA_DECODE_GQA_BATCH_OFF=1
    static const bool gqa_batch = (getenv("GGML_FA_DECODE_GQA_BATCH_OFF") == nullptr);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    // gfx1101 only for now, other targets fall back to the tile kernel.
    if (cc != GGML_CUDA_CC_OFFSET_AMD + 0x1101) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (!Q || !K || !V) {
        return false;
    }

    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (Q->ne[1] != 1) { // decode only
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    if (!ggml_is_contiguously_allocated(Q)) {
        return false;
    }
    if (K->nb[0] != sizeof(block_q4_0) || V->nb[0] != sizeof(block_q4_0)) {
        return false;
    }
    if (K->nb[1] % sizeof(block_q4_0) != 0 || V->nb[1] % sizeof(block_q4_0) != 0) {
        return false;
    }
    // mask (dst->src[3]) is always attached for dense single-token decode but is all-zeros:
    // the query attends to every valid cached position, so only sinks (dst->src[4]) matter.
    if (dst->src[4] != nullptr) {
        return false;
    }

    float scale         = 0.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    const int ne11 = K->ne[1];
    if (ne11 <= 0) {
        return false;
    }

    const int n_heads   = Q->ne[2];
    const int gqa_ratio = n_heads / K->ne[2];
    // GQA batching: one block per kv head, so split budget is per kv head.
    const bool  use_gqa     = gqa_batch && gqa_ratio == 6;
    const int   n_kv_heads  = n_heads / gqa_ratio;
    const int   max_splits  = 240 / (use_gqa ? n_kv_heads : n_heads);
    if (max_splits < 1) {
        return false;
    }
    const int n_splits  = std::min(max_splits, (ne11 + 63) / 64);
    const int split_len = (ne11 + n_splits - 1) / n_splits;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream   = ctx.stream();

    ggml_cuda_pool_alloc<float>  O_partial(pool);
    ggml_cuda_pool_alloc<float2> meta(pool);

    O_partial.alloc((size_t)n_heads * n_splits * 256);
    meta.alloc(n_heads * n_splits);

    const dim3 block_dim(256, 1, 1);
    if (use_gqa) {
        const dim3 blocks_num_gqa(n_kv_heads, n_splits, 1);
        const ggml_cuda_kernel_launch_params launch_params_gqa(blocks_num_gqa, block_dim, 0, stream);
        ggml_cuda_kernel_launch(flash_attn_decode_rdna3_gqa<6>, launch_params_gqa,
            (const char *) Q->data,
            (const char *) K->data,
            (const char *) V->data,
            O_partial.ptr, meta.ptr,
            scale, gqa_ratio, ne11, split_len,
            (int)Q->nb[2],
            (int)K->nb[1], (int)K->nb[2],
            (int)V->nb[1], (int)V->nb[2]);
        CUDA_CHECK(cudaGetLastError());
    } else {
        const dim3 blocks_num(n_heads, n_splits, 1);
        const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
        ggml_cuda_kernel_launch(flash_attn_decode_rdna3, launch_params,
            (const char *) Q->data,
            (const char *) K->data,
            (const char *) V->data,
            O_partial.ptr, meta.ptr,
            scale, gqa_ratio, ne11, split_len,
            (int)Q->nb[2],
            (int)K->nb[1], (int)K->nb[2],
            (int)V->nb[1], (int)V->nb[2]);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim_combine(256, 1, 1);
    const dim3 blocks_num_combine(1, n_heads, 1);
    const ggml_cuda_kernel_launch_params launch_params_combine(
        blocks_num_combine, block_dim_combine, n_splits*sizeof(float2), stream);
    ggml_cuda_kernel_launch(flash_attn_combine_results<256>, launch_params_combine,
        O_partial.ptr, meta.ptr, (float *) dst->data, n_splits);
    CUDA_CHECK(cudaGetLastError());

    return true;
}
