#include <ggml.h>
#include <ggml-cpp.h>
#include <ggml-cuda.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int head_dim   = 256;
constexpr int n_heads    = 24;
constexpr int n_kv_heads = 4;

constexpr double kDevThreshold = 1e-4;

void usage(const char * prog) {
    std::fprintf(stderr, "usage: %s [-ntokens N] [-nkv N] [-seed 42] [-o prefix]\n", prog);
}

bool parse_args(int argc, char * * argv, int & n_tokens, int & n_kv, int & seed, std::string & out_prefix) {
    n_tokens = 512;
    n_kv = 512;
    seed = 42;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-ntokens") == 0 && i + 1 < argc) {
            n_tokens = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-nkv") == 0 && i + 1 < argc) {
            n_kv = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_prefix = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

std::vector<float> make_q(int n_tokens, int seed) {
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(static_cast<size_t>(head_dim) * n_tokens * n_heads);
    for (auto & x : q) {
        x = dist(gen);
    }
    return q;
}

std::vector<uint8_t> make_kv(int n_kv, int seed) {
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t nels = static_cast<int64_t>(head_dim) * n_kv * n_kv_heads;
    std::vector<float> f(nels);
    for (auto & x : f) {
        x = dist(gen);
    }
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q4_0, nels));
    const int64_t nblocks = nels / 32;
    ggml_quantize_chunk(GGML_TYPE_Q4_0, f.data(), q.data(), 0, nblocks, 32, nullptr);
    return q;
}

std::vector<float> run_case(ggml_backend_t backend, int n_tokens, int n_kv, int seed) {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_context * c = ctx.get();
    if (!c) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return {};
    }

    ggml_tensor * q = ggml_new_tensor_4d(c, GGML_TYPE_F32, head_dim, n_tokens, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(c, GGML_TYPE_Q4_0, head_dim, n_kv, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(c, GGML_TYPE_Q4_0, head_dim, n_kv, n_kv_heads, 1);

    ggml_tensor * out = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / std::sqrt(head_dim), 0.0f, 0.0f);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(c, backend));
    if (!buf) {
        std::fprintf(stderr, "failed to allocate tensors\n");
        return {};
    }

    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);

    std::vector<float> qf = make_q(n_tokens, seed);
    std::vector<uint8_t> kf = make_kv(n_kv, seed + 1);
    std::vector<uint8_t> vf = make_kv(n_kv, seed + 2);

    ggml_backend_tensor_set(q, qf.data(), 0, qf.size() * sizeof(float));
    ggml_backend_tensor_set(k, kf.data(), 0, kf.size());
    ggml_backend_tensor_set(v, vf.data(), 0, vf.size());

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed: %d\n", st);
        return {};
    }

    const int64_t nres = out->ne[0] * out->ne[1] * out->ne[2] * out->ne[3];
    std::vector<float> res(nres);
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
    return res;
}

void compare(const std::vector<float> & a, const std::vector<float> & b, int n_tokens) {
    double max_abs = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    long first = -1;
    const int64_t n = static_cast<int64_t>(a.size());
    for (int64_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        const double ad = std::abs(d);
        sum_sq_diff += d * d;
        sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        if (ad > max_abs) max_abs = ad;
        if (first < 0 && ad > kDevThreshold) first = i;
    }
    const double mse = sum_sq_diff / n;
    const double nmse = sum_sq_ref > 0.0 ? mse / (sum_sq_ref / n) : 0.0;
    std::printf("max_abs=%.9f mse=%.9e nmse=%.9e first_large_dev(thresh=%.0e)=",
                max_abs, mse, nmse, kDevThreshold);
    if (first >= 0) {
        const int dim   = first % head_dim;
        const int token = (first / head_dim) % n_tokens;
        const int head  = first / (head_dim * n_tokens);
        std::printf(" (token %d, head %d, dim %d) %.9f vs %.9f\n", token, head, dim, a[first], b[first]);
    } else {
        std::printf(" none\n");
    }
}

void write_raw(const std::string & prefix, int n_tokens, int n_kv, const std::vector<float> & data) {
    if (prefix.empty()) return;
    const std::string path = prefix + "_gpu_nt" + std::to_string(n_tokens) + "_nkv" + std::to_string(n_kv) + ".bin";
    std::FILE * fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "failed to open %s for writing\n", path.c_str());
        return;
    }
    if (std::fwrite(data.data(), sizeof(float), data.size(), fp) != data.size()) {
        std::fprintf(stderr, "failed to write %s\n", path.c_str());
    }
    std::fclose(fp);
}

} // namespace

int main(int argc, char * * argv) {
    int n_tokens, n_kv, seed;
    std::string out_prefix;
    if (!parse_args(argc, argv, n_tokens, n_kv, seed, out_prefix)) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "failed to init cpu backend\n");
        return 1;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) {
        std::fprintf(stderr, "failed to init cuda/hip backend\n");
        return 1;
    }

    std::vector<float> cpu_res = run_case(cpu, n_tokens, n_kv, seed);
    std::vector<float> gpu_res = run_case(gpu, n_tokens, n_kv, seed);
    if (cpu_res.empty() || gpu_res.empty() || cpu_res.size() != gpu_res.size()) {
        std::fprintf(stderr, "failed to get results\n");
        return 1;
    }

    std::printf("ntokens=%d nkv=%d seed=%d\n", n_tokens, n_kv, seed);
    compare(gpu_res, cpu_res, n_tokens);
    if (!out_prefix.empty()) {
        std::string gpu_prefix = out_prefix + "_gpu";
        std::string cpu_prefix = out_prefix + "_cpu";
        write_raw(gpu_prefix, n_tokens, n_kv, gpu_res);
        write_raw(cpu_prefix, n_tokens, n_kv, cpu_res);
    }
    write_raw(out_prefix, n_tokens, n_kv, gpu_res);

    return 0;
}
