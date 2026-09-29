// RDNA3 (gfx1101) flash-attention decode boundary harness, Qwen3.8-27B decode shape:
// d256, 24 q heads, 4 kv heads (GQA ratio 6), Q4_0 KV, single query, no mask.
// GPU vs CPU reference; fails (rc=1) if nmse > 1e-4.
#include <ggml.h>
#include <ggml-cpp.h>
#include <ggml-cuda.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Qwen3.8-27B decode topology
constexpr int head_dim   = 256;
constexpr int n_heads    = 24;
constexpr int n_kv_heads = 4;
constexpr int kv_deep    = 129424; // benched by default
constexpr double nmse_gate = 1e-4;

void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s -kv <11,64,65,129424> [-seed 42] [-o prefix] [-model-view] [-bench]\n",
        prog);
}

bool parse_args(int argc, char * * argv, std::vector<int> & kvs, int & seed, std::string & out_prefix, bool & model_view, bool & bench) {
    seed = 42;
    model_view = false;
    bench = false;
    bool have_kvs = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-kv") == 0 && i + 1 < argc) {
            have_kvs = true;
            const std::string s = argv[++i];
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) continue;
                int kv = std::stoi(item);
                kvs.push_back(kv);
            }
        } else if (std::strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "-model-view") == 0) {
            model_view = true;
        } else if (std::strcmp(argv[i], "-bench") == 0) {
            bench = true;
        } else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return false;
        }
    }
    if (!have_kvs) {
        // deep context exercises the n_splits fold path
        kvs = {11, 64, 65, kv_deep};
    }
    return true;
}

std::vector<float> make_q(int seed) {
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(head_dim * n_heads);
    for (auto & x : q) {
        x = dist(gen);
    }
    return q;
}

std::vector<uint8_t> make_kv(int kv, int seed) {
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t nels = static_cast<int64_t>(head_dim) * kv * n_kv_heads;
    std::vector<float> f(nels);
    for (auto & x : f) {
        x = dist(gen);
    }
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q4_0, nels));
    const int64_t nblocks = nels / 32;
    ggml_quantize_chunk(GGML_TYPE_Q4_0, f.data(), q.data(), 0, nblocks, 32, nullptr);
    return q;
}

struct case_ctx {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * out;
    ggml_cgraph * gf;
    bool ok;
};

case_ctx build_case(ggml_backend_t backend, int kv, int seed, const char * name, bool model_view) {
    case_ctx cc = {};
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    cc.ctx = ggml_context_ptr(ggml_init(params));
    ggml_context * c = cc.ctx.get();
    if (!c) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return cc;
    }

    ggml_tensor * q = ggml_new_tensor_4d(c, GGML_TYPE_F32,  head_dim, 1, n_heads,    1);
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * k_parent = nullptr;
    ggml_tensor * v_parent = nullptr;
    if (model_view) {
        k_parent = ggml_new_tensor_3d(c, GGML_TYPE_Q4_0, head_dim * n_kv_heads, kv, 1);
        v_parent = ggml_new_tensor_3d(c, GGML_TYPE_Q4_0, head_dim * n_kv_heads, kv, 1);

        const size_t nb_head  = ggml_row_size(GGML_TYPE_Q4_0, head_dim);
        const size_t nb_token = ggml_row_size(GGML_TYPE_Q4_0, head_dim * n_kv_heads);
        const size_t nb_stream = ggml_row_size(GGML_TYPE_Q4_0, head_dim * n_kv_heads * kv);

        ggml_tensor * k_view = ggml_view_4d(c, k_parent,
                head_dim, n_kv_heads, kv, 1,
                nb_head, nb_token, nb_stream, 0);
        ggml_tensor * v_view = ggml_view_4d(c, v_parent,
                head_dim, n_kv_heads, kv, 1,
                nb_head, nb_token, nb_stream, 0);

        k = ggml_permute(c, k_view, 0, 2, 1, 3);
        v = ggml_permute(c, v_view, 0, 2, 1, 3);
    } else {
        k = ggml_new_tensor_4d(c, GGML_TYPE_Q4_0, head_dim, kv, n_kv_heads, 1);
        v = ggml_new_tensor_4d(c, GGML_TYPE_Q4_0, head_dim, kv, n_kv_heads, 1);
    }

    cc.out = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / std::sqrt(head_dim), 0.0f, 0.0f);

    cc.buf = ggml_backend_buffer_ptr(ggml_backend_alloc_ctx_tensors(c, backend));
    if (!cc.buf) {
        std::fprintf(stderr, "failed to allocate tensors for %s\n", name);
        return cc;
    }

    cc.gf = ggml_new_graph(c);
    ggml_build_forward_expand(cc.gf, cc.out);

    std::vector<float> qf = make_q(seed);
    std::vector<uint8_t> kf = make_kv(kv, seed + 1);
    std::vector<uint8_t> vf = make_kv(kv, seed + 2);

    ggml_backend_tensor_set(q, qf.data(), 0, qf.size() * sizeof(float));
    if (model_view) {
        ggml_backend_tensor_set(k_parent, kf.data(), 0, kf.size());
        ggml_backend_tensor_set(v_parent, vf.data(), 0, vf.size());
    } else {
        ggml_backend_tensor_set(k, kf.data(), 0, kf.size());
        ggml_backend_tensor_set(v, vf.data(), 0, vf.size());
    }

    cc.ok = true;
    return cc;
}

std::vector<float> run_case(ggml_backend_t backend, int kv, int seed, const char * name, bool model_view) {
    case_ctx cc = build_case(backend, kv, seed, name, model_view);
    if (!cc.ok) return {};

    const ggml_status st = ggml_backend_graph_compute(backend, cc.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed for %s: %d\n", name, st);
        return {};
    }

    const int64_t nres = cc.out->ne[0] * cc.out->ne[1] * cc.out->ne[2] * cc.out->ne[3];
    std::vector<float> res(nres);
    ggml_backend_tensor_get(cc.out, res.data(), 0, res.size() * sizeof(float));
    return res;
}

// synchronous compute (async launch + device sync) per call, so chrono wall time is the kernel latency
void bench_case(ggml_backend_t backend, int kv, int seed, bool model_view) {
    case_ctx cc = build_case(backend, kv, seed, "gpu-bench", model_view);
    if (!cc.ok) return;
    const int n_warmup = 10;
    const int n_iters  = 50;
    for (int i = 0; i < n_warmup; ++i) {
        if (ggml_backend_graph_compute(backend, cc.gf) != GGML_STATUS_SUCCESS) return;
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; ++i) {
        if (ggml_backend_graph_compute(backend, cc.gf) != GGML_STATUS_SUCCESS) return;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_iters;
    std::printf("BENCHMARK RESULT [kv=%d]: Average kernel latency = %.1f us (throughput = %.1f tok/s equivalent)\n",
                kv, us, 1.0 / (us * 1e-6));
}

double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return den > 0.0 ? num / den : num;
}

double max_diff(const std::vector<float> & a, const std::vector<float> & b, std::vector<double> & per_head) {
    per_head.assign(n_heads, 0.0);
    double max_abs = 0.0;
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < head_dim; ++i) {
            const double d = std::abs(static_cast<double>(a[h * head_dim + i]) - static_cast<double>(b[h * head_dim + i]));
            per_head[h] = std::max(per_head[h], d);
            max_abs = std::max(max_abs, d);
        }
    }
    return max_abs;
}

void write_raw(const std::string & prefix, int kv, const std::vector<float> & data) {
    if (prefix.empty()) return;
    const std::string path = prefix + "_" + std::to_string(kv) + ".bin";
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
    std::vector<int> kvs;
    int seed = 42;
    std::string out_prefix;
    bool model_view = false;
    bool bench = false;
    if (!parse_args(argc, argv, kvs, seed, out_prefix, model_view, bench)) {
        usage(argv[0]);
        return 1;
    }

    std::sort(kvs.begin(), kvs.end());

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

    int rc = 0;
    for (int kv : kvs) {
        std::vector<float> cpu_res = run_case(cpu, kv, seed, "cpu", model_view);
        std::vector<float> gpu_res = run_case(gpu, kv, seed, "gpu", model_view);
        if (cpu_res.empty() || gpu_res.empty() || cpu_res.size() != gpu_res.size()) {
            std::fprintf(stderr, "kv=%d: failed to get results\n", kv);
            rc = 1;
            continue;
        }
        std::vector<double> per_head;
        const double d = max_diff(cpu_res, gpu_res, per_head);
        const double n = nmse(gpu_res, cpu_res);
        double head_max = 0.0;
        for (double x : per_head) head_max = std::max(head_max, x);
        std::printf("model_view=%d kv=%d gpu_vs_cpu_max_abs=%.9f gpu_vs_cpu_head_max=%.9f nmse=%.3e seed=%d\n",
                    model_view ? 1 : 0, kv, d, head_max, n, seed);
        for (int h = 0; h < n_heads; ++h) {
            std::printf("  head %02d max_abs=%.9f\n", h, per_head[h]);
        }
        if (n > nmse_gate) {
            std::fprintf(stderr, "FAIL: kv=%d nmse %.3e above gate %.0e\n", kv, n, nmse_gate);
            rc = 1;
        }
        write_raw(out_prefix, kv, gpu_res);
        // deep context is benched by default; -bench times every case
        if (bench || kv == kv_deep) {
            bench_case(gpu, kv, seed, model_view);
        }
    }

    return rc;
}
