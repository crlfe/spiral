// Standalone smoke test for the Spiral device program.
//
// This intentionally does not use ggml yet.  It loads the production
// aie.xclbin, prepares canonical Q4_0/Q6_K weights, runs one matmul, and
// checks the result against the src/program golden references.

#include "program.h"

#include "design_meta.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-spiral.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Golden {
    int32_t format = 0;
    int32_t rows_steps = 0;
    int32_t cols_bands = 0;
    int32_t weights_rows = 0;
    int32_t weights_cols = 0;
    int32_t raw_stride = 0;
    size_t raw_bytes = 0;
    size_t inputs_elems = 0;
    size_t expected_count = 0;
    std::vector<uint8_t> raw;
    std::vector<uint16_t> inputs;
    std::vector<float> expected;
};

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>());
}

template <typename T>
std::vector<T> read_words(const std::string & path, size_t count) {
    auto bytes = read_file(path);
    if (bytes.size() < count * sizeof(T)) {
        throw std::runtime_error("short file: " + path);
    }
    std::vector<T> out(count);
    std::memcpy(out.data(), bytes.data(), count * sizeof(T));
    return out;
}

Golden load_golden(const std::string & name) {
    const std::string dir = std::string(SPIRAL_GOLDEN_DIR) + "/" + name;
    std::ifstream meta(dir + ".meta");
    if (!meta) {
        throw std::runtime_error("cannot open " + dir + ".meta");
    }

    std::map<std::string, long long> values;
    std::string key;
    long long value;
    while (meta >> key >> value) {
        values[key] = value;
    }

    Golden golden;
    golden.format = static_cast<int32_t>(values.at("format"));
    golden.rows_steps = static_cast<int32_t>(values.at("rows_steps"));
    golden.cols_bands = static_cast<int32_t>(values.at("cols_bands"));
    golden.weights_rows = static_cast<int32_t>(values.at("weights_rows"));
    golden.weights_cols = static_cast<int32_t>(values.at("weights_cols"));
    golden.raw_stride = static_cast<int32_t>(values.at("raw_stride"));
    golden.raw_bytes = static_cast<size_t>(values.at("raw_bytes"));
    golden.inputs_elems = static_cast<size_t>(values.at("inputs_elems"));
    golden.expected_count = static_cast<size_t>(values.at("expected_count"));

    golden.raw = read_file(dir + ".raw");
    golden.inputs = read_words<uint16_t>(dir + ".inputs", golden.inputs_elems);
    golden.expected = read_words<float>(dir + ".expected", golden.expected_count);
    if (golden.raw.size() != golden.raw_bytes) {
        throw std::runtime_error("raw size mismatch for " + name);
    }
    return golden;
}

double check_outputs(const Golden & golden, const std::vector<float> & outputs) {
    double max_error = 0.0;
    for (int32_t g = 0; g < design::kGroups; ++g) {
        for (int32_t c = 0; c < design::kComputesPerGroup; ++c) {
            for (int32_t band = 0; band < golden.cols_bands; ++band) {
                const size_t exp_base =
                    ((static_cast<size_t>(g) * design::kComputesPerGroup + c) *
                         golden.cols_bands +
                     band) *
                    design::kOutputTileElems;
                const int32_t col_base =
                    band * design::kBandCols +
                    (g * design::kComputesPerGroup + c) * design::kColsTile;
                for (int32_t half = 0; half < 2; ++half) {
                    for (int32_t vec = 0; vec < design::kInputsVecs; ++vec) {
                        for (int32_t col = 0; col < design::kColsTile / 2; ++col) {
                            const size_t exp_index =
                                exp_base + vec * design::kColsTile +
                                half * (design::kColsTile / 2) + col;
                            const int32_t col_index =
                                col_base + half * (design::kColsTile / 2) + col;
                            const float got =
                                outputs[static_cast<size_t>(vec) *
                                            golden.weights_cols +
                                        col_index];
                            const double diff =
                                std::fabs(static_cast<double>(got) -
                                          static_cast<double>(golden.expected[exp_index]));
                            if (diff > max_error) {
                                max_error = diff;
                            }
                        }
                    }
                }
            }
        }
    }
    return max_error;
}

bool run_case(spiral::Program & program, const std::string & name) {
    const Golden golden = load_golden(name);

    spiral::MatmulShape shape;
    shape.format = golden.format;
    shape.weights_rows = golden.weights_rows;
    shape.weights_cols = golden.weights_cols;
    shape.inputs_vecs = design::kInputsVecs;

    std::vector<float> outputs(
        static_cast<size_t>(design::kInputsVecs) * golden.weights_cols,
        std::numeric_limits<float>::quiet_NaN());

    const auto start = std::chrono::steady_clock::now();
    program.run(shape, golden.raw.data(), golden.raw_stride, 0,
                golden.inputs.data(), outputs.data(), golden.weights_cols, 0);
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_us =
        std::chrono::duration<double, std::micro>(end - start).count();

    const double max_error = check_outputs(golden, outputs);
    const double tolerance = golden.format == GGML_TYPE_Q4_0 ? 1e-2 : 2e-2;
    std::printf("%-22s %8.1f us  max_error %.6f  %s\n", name.c_str(),
                elapsed_us, max_error, max_error <= tolerance ? "ok" : "FAIL");
    return max_error <= tolerance;
}

bool run_q6_slice_case(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "spiral-smoke: cannot open %s\n", path.c_str());
        return false;
    }
    std::vector<uint8_t> raw(std::istreambuf_iterator<char>(in), {});
    constexpr int32_t K = 1536;
    constexpr int32_t N = 512;
    if (raw.size() != static_cast<size_t>(N) * (K / 256 * 210)) {
        std::fprintf(stderr, "spiral-smoke: unexpected slice size %zu\n", raw.size());
        return false;
    }

    const Golden inputs = load_golden("q4_0_1536x512");
    if (inputs.inputs_elems != static_cast<size_t>(design::kInputsVecs) * K) {
        return false;
    }
    std::vector<float> xdata(static_cast<size_t>(design::kInputsVecs) * K);
    ggml_bf16_to_fp32_row(
        reinterpret_cast<const ggml_bf16_t *>(inputs.inputs.data()),
        xdata.data(), static_cast<int64_t>(xdata.size()));

    ggml_init_params params = { 2ull * 1024 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * wc = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, K, N);
    ggml_tensor * ws = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, K, N);
    ggml_tensor * xc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, design::kInputsVecs);
    ggml_tensor * xs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, design::kInputsVecs);
    ggml_tensor * yc = ggml_mul_mat(ctx, wc, xc);
    ggml_tensor * ys = ggml_mul_mat(ctx, ws, xs);

    auto alloc = [&](ggml_tensor * t, ggml_backend_buffer_type_t buft) {
        ggml_backend_buffer_t b =
            ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(t));
        if (b == nullptr) {
            return static_cast<ggml_backend_buffer_t>(nullptr);
        }
        if (ggml_backend_tensor_alloc(b, t, ggml_backend_buffer_get_base(b)) !=
            GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(b);
            return static_cast<ggml_backend_buffer_t>(nullptr);
        }
        return b;
    };

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t wcb = alloc(wc, cpu_buft);
    ggml_backend_buffer_t wsb = alloc(ws, ggml_backend_spiral_weights_buffer_type());
    ggml_backend_buffer_t xcb = alloc(xc, cpu_buft);
    ggml_backend_buffer_t xsb = alloc(xs, cpu_buft);
    ggml_backend_buffer_t ycb = alloc(yc, cpu_buft);
    ggml_backend_buffer_t ysb = alloc(ys, cpu_buft);
    if (!wcb || !wsb || !xcb || !xsb || !ycb || !ysb) {
        std::fprintf(stderr, "spiral-smoke: q6 slice allocation failed\n");
        return false;
    }

    ggml_backend_tensor_set(wc, raw.data(), 0, raw.size());
    ggml_backend_tensor_set(ws, raw.data(), 0, raw.size());
    ggml_backend_tensor_set(xc, xdata.data(), 0, xdata.size() * sizeof(float));
    ggml_backend_tensor_set(xs, xdata.data(), 0, xdata.size() * sizeof(float));

    ggml_cgraph * gc = ggml_new_graph(ctx);
    ggml_build_forward_expand(gc, yc);
    ggml_cgraph * gs = ggml_new_graph(ctx);
    ggml_build_forward_expand(gs, ys);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t spir = ggml_backend_spiral_init(0);
    const enum ggml_status cpu_status = ggml_backend_graph_compute(cpu, gc);
    const enum ggml_status spir_status = ggml_backend_graph_compute(spir, gs);

    bool ok = cpu_status == GGML_STATUS_SUCCESS && spir_status == GGML_STATUS_SUCCESS;
    double max_abs = 0.0;
    double max_cpu = 0.0;
    if (ok) {
        const auto * a = static_cast<const float *>(yc->data);
        const auto * b = static_cast<const float *>(ys->data);
        for (size_t i = 0; i < static_cast<size_t>(design::kInputsVecs) * N; ++i) {
            const double diff = std::fabs(static_cast<double>(a[i]) - b[i]);
            max_abs = std::max(max_abs, diff);
            max_cpu = std::max(max_cpu, std::fabs(static_cast<double>(a[i])));
        }
        ok = max_abs <= 1e-2 * max_cpu + 1e-3;
    }
    std::printf("q6 real slice           cpu %s  spiral %s  max_abs %.6g  max_cpu %.6g  %s\n",
                cpu_status == GGML_STATUS_SUCCESS ? "ok" : "FAIL",
                spir_status == GGML_STATUS_SUCCESS ? "ok" : "FAIL",
                max_abs, max_cpu, ok ? "ok" : "FAIL");

    ggml_backend_free(cpu);
    ggml_backend_free(spir);
    ggml_backend_buffer_free(wcb);
    ggml_backend_buffer_free(wsb);
    ggml_backend_buffer_free(xcb);
    ggml_backend_buffer_free(xsb);
    ggml_backend_buffer_free(ycb);
    ggml_backend_buffer_free(ysb);
    ggml_free(ctx);
    return ok;
}

bool run_residency_case(spiral::Program & program) {
    const Golden a = load_golden("q4_0_1536x512");
    const Golden b = load_golden("q6_k_512x1536");

    auto shape_of = [](const Golden & g) {
        spiral::MatmulShape shape;
        shape.format = g.format;
        shape.weights_rows = g.weights_rows;
        shape.weights_cols = g.weights_cols;
        shape.inputs_vecs = design::kInputsVecs;
        return shape;
    };
    const auto sa = shape_of(a);
    const auto sb = shape_of(b);
    auto pa = program.prepare(sa, a.raw.data(), a.raw_stride, 0);
    auto pb = program.prepare(sb, b.raw.data(), b.raw_stride, 0);

    auto run_and_check = [&](const Golden & g, const spiral::MatmulShape & shape,
                             const spiral::PreparedWeights & prepared) {
        std::vector<float> out(static_cast<size_t>(design::kInputsVecs) * g.weights_cols);
        program.run(shape, prepared, g.inputs.data(), out.data(), g.weights_cols, 0);
        const double err = check_outputs(g, out);
        return err;
    };

    const double e1 = run_and_check(a, sa, pa);
    const double e2 = run_and_check(b, sb, pb);
    const double e3 = run_and_check(a, sa, pa);
    const double tolerance_a = 1e-2;
    const double tolerance_b = 2e-2;
    const bool ok = e1 <= tolerance_a && e2 <= tolerance_b && e3 <= tolerance_a;
    std::printf("residency path          a %.6f  b %.6f  a-again %.6f  %s\n",
                e1, e2, e3, ok ? "ok" : "FAIL");
    return ok;
}

bool run_buffer_case(const std::string & name, int32_t override_cols = 0) {
    const Golden golden = load_golden(name);
    const ggml_type weight_type =
        golden.format == GGML_TYPE_Q4_0 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q6_K;

    const int32_t K = golden.weights_rows;
    const int32_t N = override_cols > 0 ? override_cols : golden.weights_cols;

    ggml_init_params params = {
        /* .mem_size   = */ 2ull * 1024 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "spiral-smoke: ggml_init failed\n");
        return false;
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, weight_type, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, design::kInputsVecs);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_type_t wbuft = ggml_backend_spiral_weights_buffer_type();
    ggml_backend_buffer_t wbuf =
        ggml_backend_buft_alloc_buffer(wbuft,
                                       ggml_backend_buft_get_alloc_size(wbuft, w));
    ggml_backend_buffer_t xbuf =
        ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),
                                       ggml_nbytes(x));
    ggml_backend_buffer_t ybuf =
        ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),
                                       ggml_nbytes(y));
    if (wbuf == nullptr || xbuf == nullptr || ybuf == nullptr) {
        std::fprintf(stderr, "spiral-smoke: buffer allocation failed\n");
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (xbuf) ggml_backend_buffer_free(xbuf);
        if (ybuf) ggml_backend_buffer_free(ybuf);
        ggml_free(ctx);
        return false;
    }

    bool ok = true;
    ok = ok && ggml_backend_tensor_alloc(wbuf, w,
                                         ggml_backend_buffer_get_base(wbuf)) ==
                   GGML_STATUS_SUCCESS;
    ok = ok && ggml_backend_tensor_alloc(xbuf, x,
                                         ggml_backend_buffer_get_base(xbuf)) ==
                   GGML_STATUS_SUCCESS;
    ok = ok && ggml_backend_tensor_alloc(ybuf, y,
                                         ggml_backend_buffer_get_base(ybuf)) ==
                   GGML_STATUS_SUCCESS;
    if (!ok) {
        std::fprintf(stderr, "spiral-smoke: tensor allocation failed\n");
        ggml_backend_buffer_free(wbuf);
        ggml_backend_buffer_free(xbuf);
        ggml_backend_buffer_free(ybuf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(w, golden.raw.data(), 0, ggml_nbytes(w));

    std::vector<float> xdata(golden.weights_rows * design::kInputsVecs);
    ggml_bf16_to_fp32_row(
        reinterpret_cast<const ggml_bf16_t *>(golden.inputs.data()),
        xdata.data(), static_cast<int64_t>(xdata.size()));
    ggml_backend_tensor_set(x, xdata.data(), 0,
                            xdata.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    ggml_backend_t backend = ggml_backend_spiral_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "spiral-smoke: backend init failed\n");
        ggml_backend_buffer_free(wbuf);
        ggml_backend_buffer_free(xbuf);
        ggml_backend_buffer_free(ybuf);
        ggml_free(ctx);
        return false;
    }

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "spiral-smoke: backend graph compute failed\n");
        ok = false;
    } else {
        std::vector<float> outputs(
            static_cast<size_t>(design::kInputsVecs) * golden.weights_cols, 0.0f);
        for (int32_t vec = 0; vec < design::kInputsVecs; ++vec) {
            std::memcpy(outputs.data() +
                            static_cast<size_t>(vec) * golden.weights_cols,
                        static_cast<const float *>(y->data) +
                            static_cast<size_t>(vec) * N,
                        static_cast<size_t>(N) * sizeof(float));
        }
        const double max_error = check_outputs(golden, outputs);
        const double tolerance = golden.format == GGML_TYPE_Q4_0 ? 1e-2 : 2e-2;
        std::printf("%-22s buffer path  max_error %.6f  %s\n", name.c_str(),
                    max_error, max_error <= tolerance ? "ok" : "FAIL");
        ok = max_error <= tolerance;
    }

    ggml_backend_free(backend);
    ggml_backend_buffer_free(wbuf);
    ggml_backend_buffer_free(xbuf);
    ggml_backend_buffer_free(ybuf);
    ggml_free(ctx);
    return ok;
}

bool run_chain_case() {
    constexpr int kSegments = 3;
    const Golden golden = load_golden("q6_k_512x1536");
    const int32_t K = golden.weights_rows;
    const int32_t N = golden.weights_cols;

    ggml_init_params params = { 2ull * 1024 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    std::vector<std::vector<uint8_t>> raws(kSegments, golden.raw);
    for (int seg = 1; seg < kSegments; ++seg) {
        raws[seg][0] ^= static_cast<uint8_t>(seg);
    }

    ggml_tensor * ws[kSegments] = {};
    ggml_tensor * wc[kSegments] = {};
    ggml_tensor * ys[kSegments] = {};
    ggml_tensor * yc[kSegments] = {};
    ggml_tensor * xs =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, design::kInputsVecs);
    ggml_tensor * xc =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, design::kInputsVecs);
    for (int seg = 0; seg < kSegments; ++seg) {
        ws[seg] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, K, N);
        wc[seg] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, K, N);
        ys[seg] = ggml_mul_mat(ctx, ws[seg], xs);
        yc[seg] = ggml_mul_mat(ctx, wc[seg], xc);
    }

    auto alloc = [&](ggml_tensor * t, ggml_backend_buffer_type_t buft) {
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(
            buft, ggml_backend_buft_get_alloc_size(buft, t));
        if (buf == nullptr) {
            return static_cast<ggml_backend_buffer_t>(nullptr);
        }
        if (ggml_backend_tensor_alloc(buf, t,
                                      ggml_backend_buffer_get_base(buf)) !=
            GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            return static_cast<ggml_backend_buffer_t>(nullptr);
        }
        return buf;
    };

    ggml_backend_buffer_type_t sbuft = ggml_backend_spiral_weights_buffer_type();
    ggml_backend_buffer_type_t cbuft = ggml_backend_cpu_buffer_type();
    std::vector<ggml_backend_buffer_t> bufs;
    for (int seg = 0; seg < kSegments; ++seg) {
        bufs.push_back(alloc(ws[seg], sbuft));
        bufs.push_back(alloc(wc[seg], cbuft));
    }
    bufs.push_back(alloc(xs, cbuft));
    bufs.push_back(alloc(xc, cbuft));
    for (int seg = 0; seg < kSegments; ++seg) {
        bufs.push_back(alloc(ys[seg], cbuft));
        bufs.push_back(alloc(yc[seg], cbuft));
    }
    bool ok = true;
    for (ggml_backend_buffer_t buf : bufs) {
        ok = ok && buf != nullptr;
    }
    if (ok) {
        for (int seg = 0; seg < kSegments; ++seg) {
            ggml_backend_tensor_set(ws[seg], raws[seg].data(), 0,
                                    ggml_nbytes(ws[seg]));
            ggml_backend_tensor_set(wc[seg], raws[seg].data(), 0,
                                    ggml_nbytes(wc[seg]));
        }

        std::vector<float> xdata(static_cast<size_t>(K) * design::kInputsVecs);
        ggml_bf16_to_fp32_row(
            reinterpret_cast<const ggml_bf16_t *>(golden.inputs.data()),
            xdata.data(), static_cast<int64_t>(xdata.size()));
        const size_t xbytes = xdata.size() * sizeof(float);
        ggml_backend_tensor_set(xs, xdata.data(), 0, xbytes);
        ggml_backend_tensor_set(xc, xdata.data(), 0, xbytes);

        ggml_cgraph * gs = ggml_new_graph(ctx);
        ggml_cgraph * gc = ggml_new_graph(ctx);
        for (int seg = 0; seg < kSegments; ++seg) {
            ggml_build_forward_expand(gs, ys[seg]);
            ggml_build_forward_expand(gc, yc[seg]);
        }

        ggml_backend_t spir = ggml_backend_spiral_init(0);
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ok = spir != nullptr && cpu != nullptr &&
             ggml_backend_graph_compute(cpu, gc) == GGML_STATUS_SUCCESS &&
             ggml_backend_graph_compute(spir, gs) == GGML_STATUS_SUCCESS;
        if (ok) {
            double max_abs = 0.0;
            double max_cpu = 0.0;
            for (int seg = 0; seg < kSegments; ++seg) {
                const auto * got = static_cast<const float *>(ys[seg]->data);
                const auto * ref = static_cast<const float *>(yc[seg]->data);
                for (size_t i = 0;
                     i < static_cast<size_t>(design::kInputsVecs) * N; ++i) {
                    const double diff =
                        std::fabs(static_cast<double>(got[i]) -
                                  static_cast<double>(ref[i]));
                    max_abs = std::max(max_abs, diff);
                    max_cpu =
                        std::max(max_cpu, std::fabs(static_cast<double>(ref[i])));
                }
            }
            ok = max_abs <= 1e-2 * max_cpu + 1e-3;
            std::printf("3-segment chain        max_abs %.6g  max_cpu %.6g  %s\n",
                        max_abs, max_cpu, ok ? "ok" : "FAIL");
        }
        if (spir != nullptr) {
            ggml_backend_free(spir);
        }
        if (cpu != nullptr) {
            ggml_backend_free(cpu);
        }
    }

    for (ggml_backend_buffer_t buf : bufs) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    }
    ggml_free(ctx);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const bool run_head = argc > 1 && std::string(argv[1]) == "--head";
    if (argc > 2 && std::string(argv[1]) == "--q6-slice") {
        return run_q6_slice_case(argv[2]) ? 0 : 1;
    }
    try {
        std::printf("ggml backend registry:\n");
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            std::printf("  %s (%zu device%s)\n", ggml_backend_reg_name(reg),
                        ggml_backend_reg_dev_count(reg),
                        ggml_backend_reg_dev_count(reg) == 1 ? "" : "s");
            for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); ++j) {
                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
                std::printf("    %s: %s\n", ggml_backend_dev_name(dev),
                            ggml_backend_dev_description(dev));
            }
        }

        ggml_backend_reg_t spiral_reg = ggml_backend_spiral_reg();
        auto get_extra_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(
                spiral_reg, "ggml_backend_dev_get_extra_bufts"));
        ggml_backend_buffer_type_t * extra =
            get_extra_bufts ? get_extra_bufts(ggml_backend_reg_dev_get(spiral_reg, 0))
                            : nullptr;
        const bool buft_ok =
            extra != nullptr &&
            extra[0] == ggml_backend_spiral_weights_buffer_type();
        std::printf("weights buft hook       %s\n", buft_ok ? "ok" : "FAIL");
        if (!buft_ok) {
            return 1;
        }

        spiral::Program program;
        bool ok = true;
        ok = run_case(program, "q4_0_1536x512") && ok;
        ok = run_case(program, "q6_k_512x1536") && ok;
        ok = run_buffer_case("q4_0_1536x512") && ok;
        ok = run_buffer_case("q6_k_512x1536") && ok;
        ok = run_buffer_case("q4_0_1536x256p512", 256) && ok;
        ok = run_residency_case(program) && ok;
        ok = run_chain_case() && ok;
        if (run_head) {
            ok = run_buffer_case("q6_k_head") && ok;
        }
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "spiral-smoke: %s\n", e.what());
        return 1;
    }
}
