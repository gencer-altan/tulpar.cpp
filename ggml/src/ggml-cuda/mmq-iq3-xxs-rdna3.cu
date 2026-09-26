#include "mmq.cuh"
#include "quantize.cuh"
#include "mmq-iq3-xxs-rdna3.cuh"

// RDNA3 (gfx1101) dedicated prefill GEMM for GGML_TYPE_IQ3_XXS.
// I=128, J=128, 256 threads, Q8_0 sram layout, K_vram = MMQ_ITER_K.
//
// The raw block_iq3_xxs bytes of the next x tile are register-prefetched, so the VRAM load
// overlaps the WMMA of the current chunk. The 1 KB iq3xxs_grid LUT lives in LDS.

template <int J, bool fallback>
static __device__ __forceinline__ void iq3_xxs_rdna3_prefetch_x(
        const char * __restrict__ x, const int kbx0, const int stride,
        uint32_t * pref_q3, uint32_t * pref_aux, float * pref_d) {
    constexpr int warp_size         = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps            = ggml_cuda_mmq_get_nthreads(GGML_TYPE_IQ3_XXS, J, fallback) / warp_size;
    constexpr int I                 = ggml_cuda_mmq_get_I(GGML_TYPE_IQ3_XXS, J, fallback);
    constexpr int threads_per_row   = (MMQ_ITER_K / (4 * QR3_XXS)) / 2;
    constexpr int nrows             = warp_size / threads_per_row;
    constexpr int rows_per_thread   = I / (nwarps * nrows);

    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < rows_per_thread; ++i0) {
        const int i = i0*nwarps*nrows + threadIdx.y*nrows + threadIdx.x/threads_per_row;
        const block_iq3_xxs * bxi = (const block_iq3_xxs *) x + kbx0 + i*stride;

        pref_q3[2*i0 + 0] = (uint32_t) get_int_b2(bxi->qs, 2*kqsx + 0);
        pref_q3[2*i0 + 1] = (uint32_t) get_int_b2(bxi->qs, 2*kqsx + 1);
        pref_aux[i0]      = (uint32_t) get_int_b2(bxi->qs, QK_K/16 + kqsx);
        pref_d[i0]        = (float) bxi->d;
    }
}

// Same LUT/grid + sign dequant as ggml_cuda_mmq_load_tiles_iq3_xxs, reading bytes from registers.
template <int J, bool fallback>
static __device__ __forceinline__ void iq3_xxs_rdna3_unpack_x(
        int * __restrict__ x_tile, const uint32_t * __restrict__ lds_grid,
        const uint32_t * pref_q3, const uint32_t * pref_aux, const float * pref_d) {
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(GGML_TYPE_IQ3_XXS, J, fallback) / warp_size;
    constexpr int I               = ggml_cuda_mmq_get_I(GGML_TYPE_IQ3_XXS, J, fallback);
    constexpr int sram_stride     = ggml_cuda_mmq_get_sram_stride(GGML_TYPE_IQ3_XXS, J, fallback);
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_XXS)) / 2;
    constexpr int nrows           = warp_size / threads_per_row;
    constexpr int rows_per_thread = I / (nwarps * nrows);

    int   * x_qs = (int   *) x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < rows_per_thread; ++i0) {
        const int i = i0*nwarps*nrows + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        const int2 q3_packed = make_int2((int) pref_q3[2*i0 + 0], (int) pref_q3[2*i0 + 1]);
        const uint8_t * q3 = (const uint8_t *) &q3_packed;
        const uint32_t aux32 = pref_aux[i0];
        const float d = pref_d[i0];

#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(lds_grid[q3[2*l+0]], lds_grid[q3[2*l+1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7*l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
        }

        // Bit-identical to (ls*d + d/2)/2 for all finite d: 0.5*ls + 0.25 is exact.
        const int ls = aux32 >> 28;
        x_df[i*sram_stride + kqsx] = d * fmaf(0.5f, (float)ls, 0.25f);
    }
}

template <int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(GGML_TYPE_IQ3_XXS, J, fallback), ggml_cuda_mmq_get_occupancy(GGML_TYPE_IQ3_XXS, J, fallback))
static __global__ void mul_mat_q_iq3_xxs_rdna3(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int ncols_y, const int nkb_x, const int stride_row_x, const int stride_col_dst) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    constexpr ggml_type type      = GGML_TYPE_IQ3_XXS;
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride     = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_iter = ggml_cuda_mmq_get_K_vram(type, J, fallback) / qk;
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_XXS)) / 2;
    constexpr int nrows           = warp_size / threads_per_row;
    constexpr int rows_per_thread = I / (nwarps * nrows);

    float sum[J*I / (nwarps*warp_size)] = {0.0f};

    __shared__ uint32_t lds_grid[256];

    extern __shared__ __align__(16) int data[];
    int * ids_dst_shared = data;
    int * tile_y         = data + J;
    int * tile_x         = tile_y + GGML_PAD(J*MMQ_TILE_Y_K, nwarps*warp_size);

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
    constexpr int ne_block = QK8_1_MMQ;

    const int jt = blockIdx.y;
    const int it = blockIdx.x;

    const int tile_x_max_i = nrows_x - it*I - 1;
    const int tile_y_max_j = ncols_y - jt*J - 1;

    const int offset_x = it*I*stride_row_x;
    const int offset_y = jt*J*sz;

    // Register prefetch buffers for one x tile (rows_per_thread rows).
    uint32_t pref_q3[2*rows_per_thread];
    uint32_t pref_aux[rows_per_thread];
    float    pref_d[rows_per_thread];

    {
        const int tid = threadIdx.y*blockDim.x + threadIdx.x;
        if (tid < J) {
            ids_dst_shared[tid] = tid;
        }
        if (tid < 256) {
            lds_grid[tid] = iq3xxs_grid[tid];
        }
    }
    __syncthreads();

    // Prologue: prefetch x[0], load y[0] part 1, unpack x[0].
    iq3_xxs_rdna3_prefetch_x<J, fallback>(x, offset_x, stride_row_x, pref_q3, pref_aux, pref_d);
    {
        const int * by0 = y + offset_y;
#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
            int l = l0 + threadIdx.y*warp_size + threadIdx.x;
            tile_y[l] = by0[l];
        }
    }
    iq3_xxs_rdna3_unpack_x<J, fallback>(tile_x, lds_grid, pref_q3, pref_aux, pref_d);
    __syncthreads();

    for (int kb0 = 0; kb0 < nkb_x; kb0 += blocks_per_iter) {
        // Prefetch the next x tile: in flight during the WMMA of this iteration.
        if (kb0 + blocks_per_iter < nkb_x) {
            iq3_xxs_rdna3_prefetch_x<J, fallback>(x, offset_x + (kb0 + blocks_per_iter), stride_row_x,
                    pref_q3, pref_aux, pref_d);
        }

        ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>(tile_x, tile_y, sum, 0);
        __syncthreads();

        // y part 2 for this kb0.
        {
            const int * by0 = y + offset_y + ncols_y * ((kb0 * qk / ne_block) + 1) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();

        ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();

        // Unpack the next x tile and load y part 1 for the next kb0.
        if (kb0 + blocks_per_iter < nkb_x) {
            const int * by0 = y + offset_y + ncols_y * ((kb0 + blocks_per_iter) * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
            iq3_xxs_rdna3_unpack_x<J, fallback>(tile_x, lds_grid, pref_q3, pref_aux, pref_d);
        }
        __syncthreads();
    }

    ggml_cuda_mmq_write_back_mma<type, J, fallback>(
        sum, ids_dst_shared, dst + it*I + jt*J*stride_col_dst, nullptr, stride_col_dst,
        tile_x_max_i, tile_y_max_j);
#endif
}

bool ggml_cuda_mmq_iq3_xxs_rdna3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (getenv("GGML_CUDA_MMQ_RDNA3_OFF") != nullptr) {
        return false;
    }

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (cc != GGML_CUDA_CC_OFFSET_AMD + 0x1101) {
        return false;
    }

    if (src0->type != GGML_TYPE_IQ3_XXS) {
        return false;
    }

    // Prefill only: single batch, no experts, no extra sources.
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->src[4] != nullptr) {
        return false;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    constexpr int J = 128;

    // Decode filter: the J=128 tile is not worth it for small N.
    if (ne11 < J) {
        return false;
    }

    const int id0 = ggml_cuda_get_device();
    const int cc0 = ggml_cuda_info().devices[id0].cc;
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(GGML_TYPE_IQ3_XXS, J, false, cc0);
    if (config.type == GGML_TYPE_COUNT) {
        return false;
    }
    const int I = config.I;

    // M must be a multiple of I: the prefetch/unpack has no row bounds check.
    if (ne01 < I || ne01 % I != 0) {
        return false;
    }
    // K must be a multiple of the quantization block.
    if (ne00 % ggml_blck_size(src0->type) != 0) {
        return false;
    }

    const int warp_size = ggml_cuda_info().devices[id0].warp_size;
    const int nwarps = config.nthreads / warp_size;
    const size_t sram_stride = ggml_cuda_mmq_get_sram_stride(config.sram_layout);
    const size_t nbytes_shared = mmq_get_nbytes_shared(config, cc0);

    const size_t smpbo = ggml_cuda_info().devices[id0].smpbo;
    if (nbytes_shared > smpbo) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    // Quantize src1 (float activations) to the q8_1_mmq layout expected by the kernel.
    const size_t ts_src1 = ggml_type_size(src1->type);
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s13 = src1->nb[3] / ts_src1;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t y_block_size = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = QK8_1_MMQ;
    const size_t nbytes_src1_q8_1 = ne11*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(GGML_TYPE_IQ3_XXS, false, cc0, ne11) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                            ne11, 1, 1, stream);
    CUDA_CHECK(cudaGetLastError());

    const int nty = (int)((ne01 + I - 1) / I);
    const int ntx = (int)((ne11 + J - 1) / J);
    const dim3 block_nums(nty, ntx, 1);
    const dim3 block_dims(warp_size, nwarps, 1);

    const int stride_row_x = (int)(src0->nb[1] / ggml_type_size(src0->type));
    const int stride_col_dst = (int)(dst->nb[1] / ggml_type_size(dst->type));
    const int nkb_x = (int)(ne00 / ggml_blck_size(src0->type));

    mul_mat_q_iq3_xxs_rdna3<J, false><<<block_nums, block_dims, nbytes_shared, stream>>>(
        (const char *) src0->data, (const int *) src1_q8_1.get(), (float *) dst->data,
        (int) ne01, (int) ne11, nkb_x, stride_row_x, stride_col_dst);
    CUDA_CHECK(cudaGetLastError());

    return true;
}
