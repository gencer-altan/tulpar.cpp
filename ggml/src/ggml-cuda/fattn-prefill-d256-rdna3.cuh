#pragma once

#include "common.cuh"

// WMMA prefill flash attention for RDNA3 (gfx1101), head_dim 256, Q4_0 KV, GQA ratio 6.
// Returns true when the op was handled, false to fall back to the tile kernel.
bool ggml_cuda_flash_attn_ext_prefill_d256_rdna3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
