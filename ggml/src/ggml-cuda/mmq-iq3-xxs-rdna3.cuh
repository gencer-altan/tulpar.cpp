#pragma once

#include "common.cuh"

// RDNA3 (gfx1101) dedicated IQ3_XXS prefill GEMM (I=128, J=128).
// Returns true when the op was handled, false to fall back to the tile kernel.
bool ggml_cuda_mmq_iq3_xxs_rdna3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
