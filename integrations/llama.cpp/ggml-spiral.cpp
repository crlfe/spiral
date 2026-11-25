#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-spiral.h"

#include "design_meta.h"
#include "program.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#    include <sys/sysinfo.h>
#endif

namespace {

constexpr const char * kSpiralBackendName = "Spiral";
constexpr int32_t kSpiralInputsVecs = 4;
constexpr int32_t kSpiralBandCols = 512;

struct ggml_backend_spiral_device;
struct ggml_backend_spiral_buffer_type;
struct ggml_backend_spiral_buffer;

ggml_guid_t spiral_guid() {
    static ggml_guid guid = {
        0x6f, 0x3a, 0x0d, 0x51, 0x9c, 0x2b, 0x4e, 0x93,
        0x8a, 0x51, 0x7d, 0x27, 0x31, 0x4f, 0x2c, 0x8e,
    };
    return &guid;
}

int32_t spiral_format(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return GGML_TYPE_Q4_0;
        case GGML_TYPE_Q6_K:
            return GGML_TYPE_Q6_K;
        default:
            return 0;
    }
}

int32_t spiral_ceil_to(int32_t value, int32_t unit) {
    return (value + unit - 1) / unit * unit;
}

double spiral_now_us() {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct ggml_backend_spiral_buffer_type {
    ggml_backend_buffer_type base;
    ggml_backend_spiral_device * dev;
    std::string name;

    ggml_backend_spiral_buffer_type(ggml_backend_spiral_device * dev, std::string name);

    static const char * get_name(ggml_backend_buffer_type_t buft);
    static ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
    static ggml_backend_buffer_t alloc_buffer_n(ggml_backend_buffer_type_t buft,
                                                ggml_tensor ** tensors, int n_tensors);
    static size_t get_alignment(ggml_backend_buffer_type_t buft);
    static size_t get_max_size(ggml_backend_buffer_type_t buft);
    static size_t get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);
    static size_t get_alloc_size_n(ggml_backend_buffer_type_t buft,
                                   ggml_tensor ** tensors, int n_tensors);
    static bool is_host(ggml_backend_buffer_type_t buft);
};

struct ggml_backend_spiral_device {
    ggml_backend_device base;
    int index = 0;
    std::string name;
    std::string description;
    std::unique_ptr<spiral::Program> program;
    uint64_t prepare_tensors = 0;
    double prepare_us = 0.0;
    ggml_backend_spiral_buffer_type weights_buffer_type;
    ggml_backend_buffer_type_t extra_bufts[2] = { nullptr, nullptr };

    ggml_backend_spiral_device(ggml_backend_reg_t reg, int index,
                               std::string name, std::string description);

    bool open();

    static const char * get_name(ggml_backend_dev_t dev);
    static const char * get_description(ggml_backend_dev_t dev);
    static void get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total);
    static enum ggml_backend_dev_type get_type(ggml_backend_dev_t dev);
    static void get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props);
    static ggml_backend_t init_backend(ggml_backend_dev_t dev, const char * params);
    static ggml_backend_buffer_type_t get_buffer_type(ggml_backend_dev_t dev);
    static bool supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);
    static bool supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
};

struct ggml_backend_spiral_buffer {
    ggml_backend_buffer_t base;
    ggml_backend_spiral_buffer_type * buft;
    std::vector<uint8_t> placeholder;
    std::vector<spiral::PreparedWeights> prepared;
    std::vector<spiral::MatmulShape> shapes;
    int64_t raw_stride = 0;
    int32_t format = 0;
    int32_t weights_rows = 0;
    int32_t weights_cols = 0;
    int32_t padded_cols = 0;
    uint64_t epoch = 0;
    uint64_t prepared_epoch = 0;

    ggml_backend_spiral_buffer(ggml_backend_spiral_buffer_type * buft, size_t size);
    static void free_buffer(ggml_backend_buffer_t buffer);
    static void * get_base(ggml_backend_buffer_t buffer);
    static void set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                           const void * data, size_t offset, size_t size);
    static void get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                           void * data, size_t offset, size_t size);
    static void memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                              uint8_t value, size_t offset, size_t size);
    static void clear(ggml_backend_buffer_t buffer, uint8_t value);
    bool ensure_prepared();

    // Forced RMS-norm prelude variants.  The matmul weight stream is shared;
    // only the config-carried norm weight differs, so these are prepared once
    // per norm tensor and cached by pointer and eps.
    std::vector<spiral::PreparedWeights> prenorm_prepared;
    const void * prenorm_weight = nullptr;
    float prenorm_eps = 0.0f;
    uint64_t prenorm_epoch = 0;
    bool ensure_prenorm_prepared(const void * weight, float eps);
};

// Prepared handles for several weight tensors concatenated along columns.
//
// Adjacent matmuls that read the same input (FFN gate/up, attention Q/K/V)
// can be emitted back to back, so one wide call does the work of several.
// The cache is keyed by the weight buffers and their epochs; weights are
// immutable after load.
struct spiral_weight_chain {
    std::vector<ggml_backend_spiral_buffer *> buffers;
    std::vector<uint64_t> epochs;
    int32_t padded_cols = 0;
    std::vector<spiral::PreparedWeights> prepared;
    std::vector<spiral::MatmulShape> shapes;
    // Shared-BO prelude variant; only the TXN and config gain differ.
    std::vector<spiral::PreparedWeights> prenorm_prepared;
    const void * prenorm_weight = nullptr;
    float prenorm_eps = 0.0f;
};

// gate+up (GEGLU) -> down in one TXN.  The PreparedChain references the
// per-op prepared weights, so keep the gate chain and the down buffer alive.
struct spiral_ffn_chain {
    ggml_backend_spiral_buffer * gate = nullptr;
    ggml_backend_spiral_buffer * up = nullptr;
    ggml_backend_spiral_buffer * down = nullptr;
    uint64_t gate_epoch = 0;
    uint64_t up_epoch = 0;
    uint64_t down_epoch = 0;
    spiral::PreparedChain chain;
    // gate+up+down with the prelude on op0's shared gate+up input.
    spiral::PreparedChain prenorm_chain;
    const void * prenorm_weight = nullptr;
    float prenorm_eps = 0.0f;
    bool has_prenorm = false;
};

struct ggml_backend_spiral {
    ggml_backend base;
    ggml_backend_spiral_device * dev;
    ggml_backend_t cpu = nullptr;
    std::vector<uint16_t> input_staging;
    std::vector<uint16_t> output_staging;
    std::vector<spiral_weight_chain> weight_chains;
    std::vector<spiral_ffn_chain> ffn_chains;

    bool profile = false;
    bool profile_graph = false;
    bool profile_nodes = false;
    bool profile_cpu = false;
    uint64_t stats_graphs = 0;
    uint64_t stats_cpu_nodes = 0;
    uint64_t stats_cpu_batches = 0;
    uint64_t stats_mul_mats = 0;
    uint64_t stats_chains = 0;
    uint64_t stats_softcap_fused = 0;
    uint64_t stats_geglu_fused = 0;
    uint64_t stats_ffn_chains = 0;
    uint64_t stats_prenorm_fused = 0;
    uint64_t stats_prenorm_chains = 0;
    uint64_t stats_row_chunks = 0;
    uint64_t stats_col_chunks = 0;
    double stats_wall_us = 0.0;
    double stats_cpu_us = 0.0;
    double stats_npu_us = 0.0;
    double stats_npu_txn_us = 0.0;
    double stats_npu_input_us = 0.0;
    double stats_npu_kernel_us = 0.0;
    double stats_npu_unpack_us = 0.0;
    double stats_pack_us = 0.0;
    double stats_gather_us = 0.0;

    explicit ggml_backend_spiral(ggml_backend_spiral_device * dev) :
        base{
            /* .guid    = */ spiral_guid(),
            /* .iface   = */ {},
            /* .device  = */ &dev->base,
            /* .context = */ this,
        },
        dev(dev) {
        profile = std::getenv("SPIRAL_PROFILE") != nullptr;
        profile_graph = std::getenv("SPIRAL_PROFILE_GRAPH") != nullptr;
        profile_nodes = std::getenv("SPIRAL_PROFILE_NODES") != nullptr;
        profile_cpu = std::getenv("SPIRAL_PROFILE_CPU") != nullptr;
    }
};

const ggml_backend_buffer_i spiral_buffer_iface = {
    /* .free_buffer  = */ ggml_backend_spiral_buffer::free_buffer,
    /* .get_base     = */ ggml_backend_spiral_buffer::get_base,
    /* .init_tensor  = */ nullptr,
    /* .memset_tensor= */ ggml_backend_spiral_buffer::memset_tensor,
    /* .set_tensor   = */ ggml_backend_spiral_buffer::set_tensor,
    /* .get_tensor   = */ ggml_backend_spiral_buffer::get_tensor,
    /* .set_tensor_2d= */ nullptr,
    /* .get_tensor_2d= */ nullptr,
    /* .cpy_tensor   = */ nullptr,
    /* .clear        = */ ggml_backend_spiral_buffer::clear,
    /* .reset        = */ nullptr,
    /* .set_tensor_n = */ nullptr,
    /* .get_tensor_n = */ nullptr,
    /* .cpy_tensor_n = */ nullptr,
};

const ggml_backend_buffer_type_i spiral_buffer_type_iface = {
    /* .get_name        = */ ggml_backend_spiral_buffer_type::get_name,
    /* .alloc_buffer    = */ ggml_backend_spiral_buffer_type::alloc_buffer,
    /* .alloc_buffer_n  = */ ggml_backend_spiral_buffer_type::alloc_buffer_n,
    /* .get_alignment   = */ ggml_backend_spiral_buffer_type::get_alignment,
    /* .get_max_size    = */ ggml_backend_spiral_buffer_type::get_max_size,
    /* .get_alloc_size  = */ ggml_backend_spiral_buffer_type::get_alloc_size,
    /* .get_alloc_size_n= */ ggml_backend_spiral_buffer_type::get_alloc_size_n,
    /* .is_host         = */ ggml_backend_spiral_buffer_type::is_host,
};

ggml_backend_spiral_buffer_type::ggml_backend_spiral_buffer_type(
    ggml_backend_spiral_device * dev, std::string name) :
    base{ spiral_buffer_type_iface, &dev->base, this },
    dev(dev),
    name(std::move(name)) {}

const char * ggml_backend_spiral_buffer_type::get_name(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_spiral_buffer_type *>(buft->context)->name.c_str();
}

size_t ggml_backend_spiral_buffer_type::get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64;
}

size_t ggml_backend_spiral_buffer_type::get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

size_t ggml_backend_spiral_buffer_type::get_alloc_size(ggml_backend_buffer_type_t buft,
                                                       const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    if (tensor->ne[1] > 0 && tensor->nb[1] > 0) {
        return static_cast<size_t>(
                   spiral_ceil_to(static_cast<int32_t>(tensor->ne[1]),
                                  kSpiralBandCols)) *
               tensor->nb[1];
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_spiral_buffer_type::get_alloc_size_n(ggml_backend_buffer_type_t buft,
                                                         ggml_tensor ** tensors, int n_tensors) {
    size_t total = 0;
    for (int i = 0; i < n_tensors; ++i) {
        if (tensors[i]->data == nullptr && tensors[i]->view_src == nullptr) {
            total += GGML_PAD(get_alloc_size(buft, tensors[i]), get_alignment(buft));
        }
    }
    return total;
}

bool ggml_backend_spiral_buffer_type::is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

ggml_backend_buffer_t ggml_backend_spiral_buffer_type::alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {
    auto * self = static_cast<ggml_backend_spiral_buffer_type *>(buft->context);
    auto * buffer = new ggml_backend_spiral_buffer(self, size);
    return buffer->base;
}

ggml_backend_buffer_t ggml_backend_spiral_buffer_type::alloc_buffer_n(
    ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(n_tensors);

    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor * tensor = tensors[i];
        if (tensor->data != nullptr) {
            continue;
        }
        if (tensor->view_src != nullptr) {
            if (ggml_backend_view_init(tensor) != GGML_STATUS_SUCCESS) {
                goto fail;
            }
            continue;
        }

        const size_t size = GGML_PAD(get_alloc_size(buft, tensor), get_alignment(buft));
        ggml_backend_buffer_t buffer = alloc_buffer(buft, size);
        if (buffer == nullptr) {
            goto fail;
        }
        if (ggml_backend_tensor_alloc(buffer, tensor,
                                      ggml_backend_buffer_get_base(buffer)) !=
            GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            goto fail;
        }
        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return nullptr;
    }
    if (buffers.size() == 1) {
        return buffers[0];
    }
    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());

fail:
    for (ggml_backend_buffer_t buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    return nullptr;
}

ggml_backend_spiral_buffer::ggml_backend_spiral_buffer(
    ggml_backend_spiral_buffer_type * buft, size_t size) :
    base(ggml_backend_buffer_init(&buft->base, spiral_buffer_iface, this, size)),
    buft(buft) {
    base->usage = GGML_BACKEND_BUFFER_USAGE_ANY;
    placeholder.assign(size == 0 ? 1 : size, 0);
}

void ggml_backend_spiral_buffer::free_buffer(ggml_backend_buffer_t buffer) {
    delete static_cast<ggml_backend_spiral_buffer *>(buffer->context);
}

void * ggml_backend_spiral_buffer::get_base(ggml_backend_buffer_t buffer) {
    auto * self = static_cast<ggml_backend_spiral_buffer *>(buffer->context);
    return self->placeholder.empty() ? static_cast<void *>(self) : self->placeholder.data();
}

void ggml_backend_spiral_buffer::set_tensor(ggml_backend_buffer_t buffer,
                                            ggml_tensor * tensor, const void * data,
                                            size_t offset, size_t size) {
    auto * self = static_cast<ggml_backend_spiral_buffer *>(buffer->context);
    if (offset != 0 || size != ggml_nbytes(tensor)) {
        GGML_ABORT("%s: partial Spiral weight writes are not supported", __func__);
    }
    if (!ggml_is_contiguous(tensor)) {
        GGML_ABORT("%s: non-contiguous Spiral weights are not supported", __func__);
    }

    const int32_t format = spiral_format(tensor->type);
    const int32_t weights_rows = static_cast<int32_t>(tensor->ne[0]);
    const int32_t weights_cols = static_cast<int32_t>(tensor->ne[1]);
    if (format == 0 || weights_rows <= 0 || weights_cols <= 0) {
        GGML_ABORT("%s: unsupported Spiral weight type/shape", __func__);
    }
    if (weights_rows % 256 != 0) {
        GGML_ABORT("%s: weights_rows must be a multiple of 256", __func__);
    }

    const int32_t padded_cols = spiral_ceil_to(weights_cols, kSpiralBandCols);
    const int64_t raw_stride = tensor->nb[1];
    const size_t required =
        static_cast<size_t>(padded_cols) * static_cast<size_t>(raw_stride);
    if (self->placeholder.size() < required) {
        self->placeholder.resize(required);
    }
    // The placeholder is zero-initialised at construction; only refresh the
    // band padding so the canonical tail stays deterministic on rewrite.
    if (padded_cols != weights_cols) {
        std::memset(self->placeholder.data() +
                        static_cast<size_t>(weights_cols) * raw_stride,
                    0,
                    static_cast<size_t>(padded_cols - weights_cols) * raw_stride);
    }
    std::memcpy(self->placeholder.data(), data,
                static_cast<size_t>(weights_cols) * raw_stride);

    self->raw_stride = raw_stride;
    self->format = format;
    self->weights_rows = weights_rows;
    self->weights_cols = weights_cols;
    self->padded_cols = padded_cols;
    // A process-wide, monotonically increasing epoch keeps the pair cache
    // valid even if a freed buffer address is reused by a new tensor.
    static std::atomic<uint64_t> next_epoch{1};
    self->epoch = next_epoch.fetch_add(1);
    self->prepared.clear();
    self->shapes.clear();
    self->prepared_epoch = 0;
    if (!self->ensure_prepared()) {
        GGML_ABORT("%s: failed to prepare Spiral weights", __func__);
    }
}

bool ggml_backend_spiral_buffer::ensure_prepared() {
    if (!prepared.empty() && prepared_epoch == epoch) {
        return true;
    }
    if (format == 0 || weights_rows <= 0 || padded_cols <= 0 ||
        raw_stride <= 0) {
        return false;
    }
    if (!buft->dev->open()) {
        return false;
    }

    const int32_t max_chunk_cols =
        design::kMaxBandsPerChunk * design::kBandCols;
    const double prepare_start = spiral_now_us();
    prepared.clear();
    shapes.clear();
    try {
        for (int32_t start = 0; start < padded_cols; start += max_chunk_cols) {
            const int32_t chunk_cols =
                std::min(max_chunk_cols, padded_cols - start);
            spiral::MatmulShape shape;
            shape.format = format;
            shape.weights_rows = weights_rows;
            shape.weights_cols = chunk_cols;
            shape.inputs_vecs = kSpiralInputsVecs;
            prepared.push_back(buft->dev->program->prepare(
                shape, placeholder.data(), raw_stride, start));
            shapes.push_back(shape);
        }
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("spiral: weight preparation failed: %s\n", error.what());
        prepared.clear();
        shapes.clear();
        return false;
    }
    prepared_epoch = epoch;
    buft->dev->prepare_us += spiral_now_us() - prepare_start;
    ++buft->dev->prepare_tensors;
    return true;
}

bool ggml_backend_spiral_buffer::ensure_prenorm_prepared(const void * weight,
                                                         float eps) {
    if (!prenorm_prepared.empty() && prenorm_epoch == epoch &&
        prenorm_weight == weight && prenorm_eps == eps) {
        return true;
    }
    if (weight == nullptr || !ensure_prepared() || prepared.empty()) {
        return false;
    }
    const double prepare_start = spiral_now_us();
    prenorm_prepared.clear();
    try {
        for (const auto & base : prepared) {
            spiral::PrenormSpec spec;
            spec.weight = static_cast<const float *>(weight);
            spec.k = weights_rows;
            spec.eps = eps;
            prenorm_prepared.push_back(
                buft->dev->program->prepare_prenorm(base, spec));
        }
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("spiral: prenorm preparation failed: %s\n",
                       error.what());
        prenorm_prepared.clear();
        return false;
    }
    prenorm_weight = weight;
    prenorm_eps = eps;
    prenorm_epoch = epoch;
    buft->dev->prepare_us += spiral_now_us() - prepare_start;
    return true;
}

void ggml_backend_spiral_buffer::get_tensor(ggml_backend_buffer_t buffer,
                                            const ggml_tensor * tensor, void * data,
                                            size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
    GGML_ABORT("%s: Spiral weights cannot be read back", __func__);
}

void ggml_backend_spiral_buffer::memset_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor * tensor, uint8_t value,
                                               size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    GGML_UNUSED(value);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
    GGML_ABORT("%s: Spiral weight memset is not supported", __func__);
}

void ggml_backend_spiral_buffer::clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

const ggml_backend_device_i spiral_device_i = {
    /* .get_name             = */ ggml_backend_spiral_device::get_name,
    /* .get_description      = */ ggml_backend_spiral_device::get_description,
    /* .get_memory           = */ ggml_backend_spiral_device::get_memory,
    /* .get_type             = */ ggml_backend_spiral_device::get_type,
    /* .get_props            = */ ggml_backend_spiral_device::get_props,
    /* .init_backend         = */ ggml_backend_spiral_device::init_backend,
    /* .get_buffer_type      = */ ggml_backend_spiral_device::get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_spiral_device::supports_op,
    /* .supports_buft        = */ ggml_backend_spiral_device::supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

ggml_backend_spiral_device::ggml_backend_spiral_device(
    ggml_backend_reg_t reg, int index, std::string name, std::string description) :
    base{ spiral_device_i, reg, this },
    index(index),
    name(std::move(name)),
    description(std::move(description)),
    weights_buffer_type(this, this->name + "-weights") {
    extra_bufts[0] = &weights_buffer_type.base;
    extra_bufts[1] = nullptr;
}

bool ggml_backend_spiral_device::open() {
    if (program) {
        return true;
    }
    try {
        program = std::make_unique<spiral::Program>();
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to open Spiral program: %s\n", __func__, e.what());
        return false;
    }
}

const char * ggml_backend_spiral_device::get_name(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_spiral_device *>(dev->context)->name.c_str();
}

const char * ggml_backend_spiral_device::get_description(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_spiral_device *>(dev->context)->description.c_str();
}

void ggml_backend_spiral_device::get_memory(ggml_backend_dev_t dev,
                                            size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 0;
    *total = 0;
#if defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        *free = static_cast<size_t>(info.freeram) * info.mem_unit;
        *total = static_cast<size_t>(info.totalram) * info.mem_unit;
    }
#endif
}

enum ggml_backend_dev_type ggml_backend_spiral_device::get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_IGPU;
}

void ggml_backend_spiral_device::get_props(ggml_backend_dev_t dev,
                                           ggml_backend_dev_props * props) {
    props->name = get_name(dev);
    props->description = get_description(dev);
    get_memory(dev, &props->memory_free, &props->memory_total);
    props->type = get_type(dev);
    props->device_id = nullptr;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

ggml_backend_t ggml_backend_spiral_device::init_backend(ggml_backend_dev_t dev,
                                                        const char * params) {
    GGML_UNUSED(params);
    return ggml_backend_spiral_init(
        static_cast<ggml_backend_spiral_device *>(dev->context)->index);
}

ggml_backend_buffer_type_t ggml_backend_spiral_device::get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

bool spiral_supports_mul_mat(const ggml_backend_spiral_device * self,
                             const ggml_tensor * op) {
    const auto * src0 = op->src[0];
    const auto * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr || op->src[2] != nullptr ||
        op->src[3] != nullptr) {
        return false;
    }
    if (src0->buffer == nullptr ||
        src0->buffer->buft != &self->weights_buffer_type.base) {
        return false;
    }
    if (src0->view_src != nullptr || !ggml_is_contiguous(src0) ||
        src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q6_K) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (src1->ne[0] != src0->ne[0] || op->ne[0] != src0->ne[1]) {
        return false;
    }
    if (src0->ne[0] <= 0 || src0->ne[0] % 256 != 0 || src0->ne[1] <= 0) {
        return false;
    }
    return true;
}

bool spiral_cpu_compatible(const ggml_backend_spiral_device * self,
                           const ggml_tensor * op) {
    auto host_tensor = [&](const ggml_tensor * t) {
        if (t == nullptr || t->buffer == nullptr) {
            return true;
        }
        if (t->buffer->buft == &self->weights_buffer_type.base) {
            return false;
        }
        return ggml_backend_buft_is_host(t->buffer->buft);
    };
    if (!host_tensor(op)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (!host_tensor(op->src[i])) {
            return false;
        }
    }
    return true;
}

bool ggml_backend_spiral_device::supports_op(ggml_backend_dev_t dev,
                                             const ggml_tensor * op) {
    static const bool matmul_only = std::getenv("SPIRAL_MATMUL_ONLY") != nullptr;
    auto * self = static_cast<ggml_backend_spiral_device *>(dev->context);
    auto host_owned = [&](const ggml_tensor * t) {
        if (t == nullptr || t->buffer == nullptr) {
            return true;
        }
        if (t->buffer->buft == &self->weights_buffer_type.base) {
            return false;
        }
        return ggml_backend_buft_is_host(t->buffer->buft);
    };
    switch (op->op) {
        case GGML_OP_NONE:
            return true;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            // A view has to stay with its source.  Claim it only when both
            // the view and its source are host-accessible; a view of another
            // device's buffer must run on that device.
            if (matmul_only) {
                return false;
            }
            return host_owned(op) && host_owned(op->src[0]);
        case GGML_OP_MUL_MAT:
            // Returning true for a CPU-buffered weight would keep it on the
            // CPU buffer and prevent llama.cpp from selecting the Spiral
            // weights buft.  Only claim MUL_MAT when it is already in the
            // Spiral weight buffer.
            return spiral_supports_mul_mat(self, op);
        default:
            if (matmul_only) {
                return false;
            }
            return spiral_cpu_compatible(self, op);
    }
}

bool ggml_backend_spiral_device::supports_buft(ggml_backend_dev_t dev,
                                               ggml_backend_buffer_type_t buft) {
    auto * self = static_cast<ggml_backend_spiral_device *>(dev->context);
    return buft == &self->weights_buffer_type.base || ggml_backend_buft_is_host(buft);
}

ggml_backend_spiral_device & spiral_device(ggml_backend_reg_t reg) {
    static ggml_backend_spiral_device device(reg, 0, "Spiral0",
                                             "AMD XDNA2 / AIE2P NPU");
    return device;
}

struct spiral_prenorm_run {
    const ggml_tensor * input = nullptr;
    const ggml_tensor * weight = nullptr;
    float eps = 0.0f;
};

bool spiral_run_mul_mat(ggml_backend_spiral * backend, ggml_tensor * node,
                        ggml_tensor * out = nullptr,
                        float soft_cap = 0.0f,
                        const spiral_prenorm_run * prenorm = nullptr) {
    const auto * src0 = node->src[0];
    const auto * src1 = node->src[1];
    if (src0 == nullptr || src1 == nullptr || src0->buffer == nullptr ||
        src0->buffer->buft != &backend->dev->weights_buffer_type.base) {
        GGML_LOG_ERROR("spiral: MUL_MAT %s has no Spiral weight buffer\n", node->name);
        return false;
    }
    if (out == nullptr) {
        out = node;
    }
    auto * weight_buffer =
        static_cast<ggml_backend_spiral_buffer *>(src0->buffer->context);
    if (!weight_buffer->ensure_prepared() || weight_buffer->prepared.empty()) {
        GGML_LOG_ERROR("spiral: MUL_MAT %s has no prepared weights\n", node->name);
        return false;
    }
    const ggml_tensor * input = prenorm != nullptr ? prenorm->input : src1;
    if (input == nullptr) {
        GGML_LOG_ERROR("spiral: MUL_MAT %s has no input\n", node->name);
        return false;
    }
    if (prenorm != nullptr) {
        if (prenorm->weight == nullptr ||
            !weight_buffer->ensure_prenorm_prepared(prenorm->weight->data,
                                                    prenorm->eps)) {
            GGML_LOG_ERROR("spiral: MUL_MAT %s prenorm preparation failed\n",
                           node->name);
            return false;
        }
    }
    if (prenorm != nullptr && backend->profile_nodes) {
        GGML_LOG_INFO("spiral prenorm node %s\n", node->name);
    }
    const auto & prepared_chunks =
        prenorm != nullptr ? weight_buffer->prenorm_prepared
                           : weight_buffer->prepared;

    const int32_t weights_rows = weight_buffer->shapes[0].weights_rows;
    const int32_t weights_cols = weight_buffer->weights_cols;
    const int64_t rows = input->ne[1] * input->ne[2] * input->ne[3];
    if (rows <= 0) {
        // A graph with no output rows (for example a warm-up batch that
        // requests no logits) still visits the matmul nodes.  There is nothing
        // to compute; do not treat it as a shape error.
        return true;
    }
    if (input->ne[0] != weights_rows ||
        node->ne[0] != weights_cols || node->ne[1] != input->ne[1] ||
        node->ne[2] != input->ne[2] || node->ne[3] != input->ne[3]) {
        GGML_LOG_ERROR("spiral: MUL_MAT %s shape mismatch K=%lld N=%lld rows=%lld node=[%lld,%lld,%lld,%lld]\n",
                       node->name, (long long) input->ne[0], (long long) weights_cols,
                       (long long) rows, (long long) node->ne[0], (long long) node->ne[1],
                       (long long) node->ne[2], (long long) node->ne[3]);
        return false;
    }
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) ||
        out->ne[0] != node->ne[0] || out->ne[1] != node->ne[1] ||
        out->ne[2] != node->ne[2] || out->ne[3] != node->ne[3]) {
        GGML_LOG_ERROR("spiral: MUL_MAT %s bad output node %s\n",
                       node->name, out->name);
        return false;
    }

    backend->input_staging.resize(
        static_cast<size_t>(kSpiralInputsVecs) * weights_rows);
    const auto max_chunk_cols =
        design::kMaxBandsPerChunk * design::kBandCols;
    backend->output_staging.resize(
        static_cast<size_t>(kSpiralInputsVecs) * max_chunk_cols);

    const double node_start = spiral_now_us();
    double node_npu_us = 0.0;
    double node_txn_us = 0.0;
    double node_input_us = 0.0;
    double node_kernel_us = 0.0;
    double node_start_us = 0.0;
    double node_wait_us = 0.0;
    double node_sync_us = 0.0;
    double node_unpack_us = 0.0;
    double node_pack_us = 0.0;
    double node_gather_us = 0.0;
    int node_col_chunks = 0;

    const auto * x = static_cast<const float *>(input->data);
    auto * y = static_cast<float *>(out->data);

    struct PendingCall {
        spiral::RunTicket ticket;
        int64_t first = 0;
        int32_t chunk_rows = 0;
        int32_t chunk_cols = 0;
        int32_t col_start = 0;
    };
    const int32_t pipeline_slots =
        std::max(1, backend->dev->program->pipeline_slots());
    std::vector<PendingCall> pending;
    pending.reserve(static_cast<size_t>(pipeline_slots));

    auto finish_one = [&]() {
        PendingCall call = pending.front();
        pending.erase(pending.begin());
        backend->dev->program->finish(call.ticket);
        const spiral::RunTimings & timings = call.ticket.timings;
        node_npu_us += timings.txn_us + timings.input_us + timings.kernel_us +
                       timings.unpack_us;
        node_txn_us += timings.txn_us;
        node_input_us += timings.input_us;
        node_kernel_us += timings.kernel_us;
        node_start_us += timings.start_us;
        node_wait_us += timings.wait_us;
        node_sync_us += timings.sync_us;
        node_unpack_us += timings.unpack_us;
        backend->stats_npu_txn_us += timings.txn_us;
        backend->stats_npu_input_us += timings.input_us;
        backend->stats_npu_kernel_us += timings.kernel_us;
        backend->stats_npu_unpack_us += timings.unpack_us;

        const int32_t copy_cols =
            std::min(call.chunk_cols, weights_cols - call.col_start);
        if (copy_cols > 0) {
            const double gather_start = spiral_now_us();
            for (int32_t r = 0; r < call.chunk_rows; ++r) {
                ggml_bf16_to_fp32_row(
                    reinterpret_cast<const ggml_bf16_t *>(
                        backend->output_staging.data() +
                        static_cast<size_t>(r) * call.chunk_cols),
                    y + (call.first + r) * weights_cols + call.col_start,
                    copy_cols);
            }
            node_gather_us += spiral_now_us() - gather_start;
        }
    };

    for (int64_t first = 0; first < rows; first += kSpiralInputsVecs) {
        const int32_t chunk_rows = static_cast<int32_t>(
            std::min<int64_t>(kSpiralInputsVecs, rows - first));
        const double pack_start = spiral_now_us();
        std::fill(backend->input_staging.begin(),
                  backend->input_staging.end(), 0);
        for (int32_t r = 0; r < chunk_rows; ++r) {
            ggml_fp32_to_bf16_row(
                x + (first + r) * weights_rows,
                reinterpret_cast<ggml_bf16_t *>(backend->input_staging.data()) +
                    static_cast<size_t>(r) * weights_rows,
                weights_rows);
        }
        node_pack_us += spiral_now_us() - pack_start;

        int32_t col_start = 0;
        for (size_t chunk = 0; chunk < prepared_chunks.size(); ++chunk) {
            const auto & shape = weight_buffer->shapes[chunk];
            const int32_t chunk_cols = shape.weights_cols;
            spiral::MatmulShape run_shape = shape;
            if (prenorm != nullptr) {
                run_shape.prenorm = true;
            }
            if (soft_cap != 0.0f) {
                run_shape.soft_cap = soft_cap;
                run_shape.epilogue = spiral::kEpilogueSoftcap;
            }
            if (static_cast<int32_t>(pending.size()) >= pipeline_slots) {
                finish_one();
            }
            PendingCall call;
            call.first = first;
            call.chunk_rows = chunk_rows;
            call.chunk_cols = chunk_cols;
            call.col_start = col_start;
            call.ticket = backend->dev->program->submit(
                run_shape, prepared_chunks[chunk],
                backend->input_staging.data(),
                backend->output_staging.data(), chunk_cols, 0);
            pending.push_back(call);
            ++node_col_chunks;
            col_start += chunk_cols;
        }
    }
    while (!pending.empty()) {
        finish_one();
    }
    const int64_t row_chunks =
        (rows + kSpiralInputsVecs - 1) / kSpiralInputsVecs;
    backend->stats_mul_mats++;
    backend->stats_row_chunks += row_chunks;
    backend->stats_col_chunks += node_col_chunks;
    backend->stats_npu_us += node_npu_us;
    backend->stats_pack_us += node_pack_us;
    backend->stats_gather_us += node_gather_us;
    if (backend->profile_nodes) {
        GGML_LOG_INFO("spiral node %-32s fmt=%d K=%-6d N=%-7d rows=%-3lld "
                      "chunks=%-3d npu=%8.1f us (txn=%.1f input=%.1f "
                      "kernel=%.1f start=%.1f wait=%.1f sync=%.1f "
                      "unpack=%.1f) pack=%7.1f gather=%7.1f "
                      "total=%7.1f\n",
                      node->name, weight_buffer->format, weights_rows,
                      weights_cols, (long long) rows, node_col_chunks,
                      node_npu_us, node_txn_us, node_input_us,
                      node_kernel_us, node_start_us, node_wait_us,
                      node_sync_us, node_unpack_us, node_pack_us,
                      node_gather_us, spiral_now_us() - node_start);
    }
    return true;
}

ggml_backend_spiral_buffer * spiral_weight_buffer_of(
    const ggml_backend_spiral_device * dev, const ggml_tensor * node) {
    if (node == nullptr || node->op != GGML_OP_MUL_MAT) {
        return nullptr;
    }
    const ggml_tensor * src0 = node->src[0];
    if (src0 == nullptr || src0->buffer == nullptr ||
        src0->buffer->buft != &dev->weights_buffer_type.base) {
        return nullptr;
    }
    return static_cast<ggml_backend_spiral_buffer *>(src0->buffer->context);
}

bool spiral_same_input(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    if (a->data != b->data || a->type != b->type) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (a->ne[d] != b->ne[d] || a->nb[d] != b->nb[d]) {
            return false;
        }
    }
    return true;
}

constexpr int32_t kSpiralMaxChain = 4;

// How many consecutive supported MUL_MATs can run as one wide call?
//
// The nodes must read the same input data (same pointer and layout) and share
// K and quant format; the combined padded columns must fit one band chunk.
// FFN gate+up is the important case.  Nodes separated by other ops are not
// chained because the scheduler's compute-buffer liveness assumes strict
// graph order, and computing a later matmul early can clobber a gap tensor.
int32_t spiral_chain_length(const ggml_backend_spiral_device * dev,
                            ggml_tensor ** nodes, int32_t count) {
    static const bool chains_enabled =
        std::getenv("SPIRAL_NO_PAIR") == nullptr;
    if (!chains_enabled || count <= 0 ||
        !spiral_supports_mul_mat(dev, nodes[0])) {
        return 0;
    }
    const ggml_tensor * first_input = nodes[0]->src[1];
    const auto * first_buffer = spiral_weight_buffer_of(dev, nodes[0]);
    if (first_input == nullptr || first_buffer == nullptr) {
        return 0;
    }
    const int32_t max_chunk_cols =
        design::kMaxBandsPerChunk * design::kBandCols;
    int32_t total_cols = 0;
    int32_t length = 0;
    for (int32_t j = 0; j < count && j < kSpiralMaxChain; ++j) {
        const ggml_tensor * node = nodes[j];
        if (!spiral_supports_mul_mat(dev, node)) {
            break;
        }
        const auto * buffer = spiral_weight_buffer_of(dev, node);
        if (buffer == nullptr ||
            !spiral_same_input(first_input, node->src[1]) ||
            buffer->format != first_buffer->format ||
            buffer->weights_rows != first_buffer->weights_rows ||
            buffer->padded_cols <= 0 ||
            total_cols + buffer->padded_cols > max_chunk_cols) {
            break;
        }
        total_cols += buffer->padded_cols;
        ++length;
    }
    return length >= 2 ? length : 0;
}

spiral_weight_chain * spiral_prepare_chain(
    ggml_backend_spiral * backend,
    ggml_backend_spiral_buffer * const * buffers, int32_t count,
    const spiral_prenorm_run * prenorm = nullptr) {
    auto build_variant = [&](spiral_weight_chain & chain) -> bool {
        if (prenorm == nullptr || prenorm->weight == nullptr) {
            return true;
        }
        if (!chain.prenorm_prepared.empty() &&
            chain.prenorm_weight == prenorm->weight->data &&
            chain.prenorm_eps == prenorm->eps) {
            return true;
        }
        chain.prenorm_prepared.clear();
        chain.prenorm_weight = nullptr;
        try {
            spiral::PrenormSpec spec;
            spec.weight = static_cast<const float *>(prenorm->weight->data);
            spec.k = chain.buffers[0]->weights_rows;
            spec.eps = prenorm->eps;
            chain.prenorm_prepared.push_back(
                backend->dev->program->prepare_prenorm(chain.prepared[0],
                                                       spec));
        } catch (const std::exception & error) {
            GGML_LOG_ERROR("spiral: chain prenorm preparation failed: %s\n",
                           error.what());
            chain.prenorm_prepared.clear();
            return false;
        }
        chain.prenorm_weight = prenorm->weight->data;
        chain.prenorm_eps = prenorm->eps;
        return true;
    };
    for (auto & chain : backend->weight_chains) {
        if (static_cast<int32_t>(chain.buffers.size()) != count) {
            continue;
        }
        bool match = true;
        for (int32_t i = 0; i < count; ++i) {
            if (chain.buffers[i] != buffers[i] ||
                chain.epochs[i] != buffers[i]->epoch) {
                match = false;
                break;
            }
        }
        if (match) {
            return build_variant(chain) ? &chain : nullptr;
        }
    }

    std::vector<spiral::WeightSegment> segments;
    segments.reserve(count);
    int32_t total_cols = 0;
    for (int32_t i = 0; i < count; ++i) {
        auto * buffer = buffers[i];
        if (buffer == nullptr || !buffer->ensure_prepared()) {
            return nullptr;
        }
        if (buffer->format != buffers[0]->format ||
            buffer->weights_rows != buffers[0]->weights_rows) {
            return nullptr;
        }
        segments.push_back({buffer->placeholder.data(), buffer->raw_stride, 0,
                            buffer->padded_cols});
        total_cols += buffer->padded_cols;
    }
    const int32_t max_chunk_cols =
        design::kMaxBandsPerChunk * design::kBandCols;
    if (total_cols > max_chunk_cols) {
        return nullptr;
    }

    spiral::MatmulShape shape;
    shape.format = buffers[0]->format;
    shape.weights_rows = buffers[0]->weights_rows;
    shape.weights_cols = total_cols;
    shape.inputs_vecs = kSpiralInputsVecs;

    spiral_weight_chain chain;
    chain.padded_cols = total_cols;
    for (int32_t i = 0; i < count; ++i) {
        chain.buffers.push_back(buffers[i]);
        chain.epochs.push_back(buffers[i]->epoch);
    }
    const double prepare_start = spiral_now_us();
    try {
        chain.prepared.push_back(backend->dev->program->prepare(
            shape, segments.data(), count));
        chain.shapes.push_back(shape);
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("spiral: combined weight preparation failed: %s\n",
                       error.what());
        return nullptr;
    }
    if (!build_variant(chain)) {
        return nullptr;
    }
    backend->dev->prepare_us += spiral_now_us() - prepare_start;
    backend->weight_chains.push_back(std::move(chain));
    return &backend->weight_chains.back();
}

bool spiral_run_chain(ggml_backend_spiral * backend, ggml_tensor ** nodes,
                      int32_t count, spiral_weight_chain & chain,
                      const spiral_prenorm_run * prenorm = nullptr) {
    if (static_cast<int32_t>(chain.buffers.size()) != count ||
        chain.prepared.size() != 1 || chain.shapes.size() != 1) {
        return false;
    }
    const auto * src1 = nodes[0]->src[1];
    const auto * input = prenorm != nullptr ? prenorm->input : src1;
    if (src1 == nullptr || input == nullptr) {
        return false;
    }
    if (prenorm != nullptr && chain.prenorm_prepared.size() != 1) {
        return false;
    }
    const int32_t weights_rows = chain.buffers[0]->weights_rows;
    if (input->ne[0] != weights_rows) {
        return false;
    }
    int32_t offset = 0;
    for (int32_t i = 0; i < count; ++i) {
        const auto * buffer = chain.buffers[i];
        if (!spiral_same_input(src1, nodes[i]->src[1]) ||
            nodes[i]->ne[0] != buffer->weights_cols ||
            nodes[i]->ne[1] != input->ne[1] ||
            nodes[i]->ne[2] != input->ne[2] ||
            nodes[i]->ne[3] != input->ne[3]) {
            return false;
        }
        offset += buffer->padded_cols;
    }
    if (offset != chain.padded_cols) {
        return false;
    }

    const int64_t rows = input->ne[1] * input->ne[2] * input->ne[3];
    if (rows <= 0) {
        return true;  // zero-row graph; nothing to compute
    }

    backend->input_staging.resize(
        static_cast<size_t>(kSpiralInputsVecs) * weights_rows);
    backend->output_staging.resize(
        static_cast<size_t>(kSpiralInputsVecs) * chain.padded_cols);

    const double node_start = spiral_now_us();
    double node_npu_us = 0.0;
    double node_pack_us = 0.0;
    double node_gather_us = 0.0;
    int node_col_chunks = 0;

    const auto * x = static_cast<const float *>(input->data);
    struct PendingCall {
        spiral::RunTicket ticket;
        int64_t first = 0;
        int32_t chunk_rows = 0;
    };
    const int32_t pipeline_slots =
        std::max(1, backend->dev->program->pipeline_slots());
    std::vector<PendingCall> pending;
    pending.reserve(static_cast<size_t>(pipeline_slots));

    auto finish_one = [&]() {
        PendingCall call = pending.front();
        pending.erase(pending.begin());
        backend->dev->program->finish(call.ticket);
        const spiral::RunTimings & timings = call.ticket.timings;
        node_npu_us += timings.txn_us + timings.input_us + timings.kernel_us +
                       timings.unpack_us;
        backend->stats_npu_txn_us += timings.txn_us;
        backend->stats_npu_input_us += timings.input_us;
        backend->stats_npu_kernel_us += timings.kernel_us;
        backend->stats_npu_unpack_us += timings.unpack_us;

        const double gather_start = spiral_now_us();
        int32_t col_offset = 0;
        for (int32_t i = 0; i < count; ++i) {
            const auto * buffer = chain.buffers[i];
            auto * y = static_cast<float *>(nodes[i]->data);
            for (int32_t r = 0; r < call.chunk_rows; ++r) {
                const auto * row = reinterpret_cast<const ggml_bf16_t *>(
                    backend->output_staging.data() +
                    static_cast<size_t>(r) * chain.padded_cols + col_offset);
                ggml_bf16_to_fp32_row(
                    row, y + (call.first + r) * buffer->weights_cols,
                    buffer->weights_cols);
            }
            col_offset += buffer->padded_cols;
        }
        node_gather_us += spiral_now_us() - gather_start;
    };

    for (int64_t first = 0; first < rows; first += kSpiralInputsVecs) {
        const int32_t chunk_rows = static_cast<int32_t>(
            std::min<int64_t>(kSpiralInputsVecs, rows - first));
        const double pack_start = spiral_now_us();
        std::fill(backend->input_staging.begin(),
                  backend->input_staging.end(), 0);
        for (int32_t r = 0; r < chunk_rows; ++r) {
            ggml_fp32_to_bf16_row(
                x + (first + r) * weights_rows,
                reinterpret_cast<ggml_bf16_t *>(backend->input_staging.data()) +
                    static_cast<size_t>(r) * weights_rows,
                weights_rows);
        }
        node_pack_us += spiral_now_us() - pack_start;

        if (static_cast<int32_t>(pending.size()) >= pipeline_slots) {
            finish_one();
        }
        PendingCall call;
        call.first = first;
        call.chunk_rows = chunk_rows;
        spiral::MatmulShape run_shape = chain.shapes[0];
        if (prenorm != nullptr) {
            run_shape.prenorm = true;
        }
        call.ticket = backend->dev->program->submit(
            run_shape,
            prenorm != nullptr ? chain.prenorm_prepared[0] : chain.prepared[0],
            backend->input_staging.data(), backend->output_staging.data(),
            chain.padded_cols, 0);
        pending.push_back(call);
        ++node_col_chunks;
    }
    while (!pending.empty()) {
        finish_one();
    }
    const int64_t row_chunks =
        (rows + kSpiralInputsVecs - 1) / kSpiralInputsVecs;
    backend->stats_chains++;
    backend->stats_mul_mats += static_cast<uint64_t>(count);
    backend->stats_row_chunks += static_cast<uint64_t>(count) * row_chunks;
    backend->stats_col_chunks += node_col_chunks;
    backend->stats_npu_us += node_npu_us;
    backend->stats_pack_us += node_pack_us;
    backend->stats_gather_us += node_gather_us;
    if (backend->profile_nodes) {
        std::string names;
        for (int32_t i = 0; i < count; ++i) {
            if (i != 0) {
                names += '+';
            }
            names += nodes[i]->name;
        }
        GGML_LOG_INFO("spiral chain %-40s K=%-6d N=%-7d segs=%-2d rows=%-3lld "
                      "chunks=%-3d npu=%8.1f us pack=%7.1f gather=%7.1f "
                      "total=%7.1f\n",
                      names.c_str(), weights_rows, chain.padded_cols, count,
                      (long long) rows, node_col_chunks, node_npu_us,
                      node_pack_us, node_gather_us,
                      spiral_now_us() - node_start);
    }
    return true;
}

// Run a combined gate+up call with the GEGLU epilogue and write the activation
// directly into the GLU node.  This is the host-mediated form of the FFN
// fusion: the gate+up launch and the down launch stay separate, but the CPU
// GELU is replaced by the device epilogue.  The caller invokes this at the GLU
// node's graph position, so the GLU output is written when its lifetime
// starts.
bool spiral_run_geglu_pair(ggml_backend_spiral * backend, ggml_tensor * gate,
                           ggml_tensor * up, ggml_tensor * glu,
                           spiral_weight_chain & chain,
                           const spiral_prenorm_run * prenorm = nullptr) {
    if (chain.buffers.size() != 2 || chain.prepared.size() != 1 ||
        chain.shapes.size() != 1) {
        return false;
    }
    if (prenorm != nullptr && chain.prenorm_prepared.size() != 1) {
        return false;
    }
    const auto * src1 = gate->src[1];
    const auto * input = prenorm != nullptr ? prenorm->input : src1;
    if (src1 == nullptr || input == nullptr || up->src[1] != src1) {
        return false;
    }
    const int32_t weights_rows = chain.buffers[0]->weights_rows;
    const int32_t gate_padded = chain.buffers[0]->padded_cols;
    const int32_t up_padded = chain.buffers[1]->padded_cols;
    const int32_t gate_cols = gate->ne[0];
    if (input->ne[0] != weights_rows || gate_padded != up_padded ||
        chain.padded_cols != gate_padded + up_padded ||
        gate->ne[0] != up->ne[0] || glu->ne[0] != gate_cols ||
        glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
        return false;
    }
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (glu->ne[d] != gate->ne[d]) {
            return false;
        }
    }

    const int64_t rows = input->ne[1] * input->ne[2] * input->ne[3];
    if (rows <= 0) {
        return true;  // zero-row graph; nothing to compute
    }

    backend->input_staging.resize(
        static_cast<size_t>(kSpiralInputsVecs) * weights_rows);
    backend->output_staging.resize(
        static_cast<size_t>(kSpiralInputsVecs) * gate_padded);

    const double node_start = spiral_now_us();
    double node_npu_us = 0.0;
    double node_pack_us = 0.0;
    double node_gather_us = 0.0;
    int node_col_chunks = 0;

    spiral::MatmulShape run_shape = chain.shapes[0];
    run_shape.epilogue = spiral::kEpilogueGeglu;
    run_shape.soft_cap = 0.0f;
    run_shape.prenorm = prenorm != nullptr;
    const auto & chain_prepared =
        prenorm != nullptr ? chain.prenorm_prepared[0] : chain.prepared[0];

    // Diagnostic: run the same gate+up pair without the epilogue and compare
    // the device GEGLU against a host GELU of that result.
    static const bool geglu_check =
        std::getenv("SPIRAL_GEGLU_CHECK") != nullptr;
    spiral::MatmulShape plain_shape = chain.shapes[0];
    plain_shape.prenorm = prenorm != nullptr;
    std::vector<uint16_t> check_staging;
    if (geglu_check) {
        check_staging.resize(
            static_cast<size_t>(kSpiralInputsVecs) * chain.padded_cols);
    }

    const auto * x = static_cast<const float *>(input->data);
    auto * y = static_cast<float *>(glu->data);

    auto gather = [&](int64_t first, int32_t chunk_rows) {
        const double gather_start = spiral_now_us();
        for (int32_t r = 0; r < chunk_rows; ++r) {
            const auto * row = reinterpret_cast<const ggml_bf16_t *>(
                backend->output_staging.data() +
                static_cast<size_t>(r) * gate_padded);
            ggml_bf16_to_fp32_row(
                row, y + (first + r) * gate_cols, gate_cols);
        }
        node_gather_us += spiral_now_us() - gather_start;

        if (geglu_check) {
            auto bf = [](uint16_t bits) {
                const uint32_t wide = static_cast<uint32_t>(bits) << 16;
                float value = 0.0f;
                std::memcpy(&value, &wide, sizeof(value));
                return value;
            };
            auto gelu = [](float x) {
                constexpr float kC = 0.7978845608028654f;
                constexpr float kA = 0.044715f;
                return 0.5f * x *
                       (1.0f + std::tanh(kC * x * (1.0f + kA * x * x)));
            };
            double max_abs = 0.0;
            double max_ref = 0.0;
            for (int32_t r = 0; r < chunk_rows; ++r) {
                const auto * base = reinterpret_cast<const uint16_t *>(
                    check_staging.data() +
                    static_cast<size_t>(r) * chain.padded_cols);
                for (int32_t j = 0; j < gate_cols; ++j) {
                    const float ref =
                        gelu(bf(base[j])) * bf(base[gate_padded + j]);
                    const float got =
                        y[static_cast<int64_t>(first + r) * gate_cols + j];
                    max_abs = std::max(
                        max_abs, static_cast<double>(std::fabs(ref - got)));
                    max_ref = std::max(
                        max_ref, static_cast<double>(std::fabs(ref)));
                }
            }
            GGML_LOG_INFO(
                "spiral geglu check %-28s first=%lld rows=%d "
                "max_abs=%.6g rel=%.3g\n",
                glu->name, (long long) first, chunk_rows, max_abs,
                max_abs / std::max(1e-6, max_ref));
        }
    };

    for (int64_t first = 0; first < rows; first += kSpiralInputsVecs) {
        const int32_t chunk_rows = static_cast<int32_t>(
            std::min<int64_t>(kSpiralInputsVecs, rows - first));
        const double pack_start = spiral_now_us();
        std::fill(backend->input_staging.begin(),
                  backend->input_staging.end(), 0);
        for (int32_t r = 0; r < chunk_rows; ++r) {
            ggml_fp32_to_bf16_row(
                x + (first + r) * weights_rows,
                reinterpret_cast<ggml_bf16_t *>(backend->input_staging.data()) +
                    static_cast<size_t>(r) * weights_rows,
                weights_rows);
        }
        node_pack_us += spiral_now_us() - pack_start;

        if (geglu_check) {
            backend->dev->program->run(
                plain_shape, chain_prepared, backend->input_staging.data(),
                check_staging.data(), chain.padded_cols, 0);
        }

        // One row chunk at a time: the shared output staging and the
        // optional host reference make cross-chunk pipelining unsafe.  Decode
        // has one chunk; prefill can pipeline once the staging is per-slot.
        spiral::RunTimings timings;
        backend->dev->program->run(
            run_shape, chain_prepared, backend->input_staging.data(),
            backend->output_staging.data(), gate_padded, 0, &timings);
        node_npu_us += timings.txn_us + timings.input_us + timings.kernel_us +
                       timings.unpack_us;
        backend->stats_npu_txn_us += timings.txn_us;
        backend->stats_npu_input_us += timings.input_us;
        backend->stats_npu_kernel_us += timings.kernel_us;
        backend->stats_npu_unpack_us += timings.unpack_us;
        gather(first, chunk_rows);
        ++node_col_chunks;
    }

    const int64_t row_chunks =
        (rows + kSpiralInputsVecs - 1) / kSpiralInputsVecs;
    backend->stats_chains++;
    backend->stats_mul_mats += 2;
    backend->stats_row_chunks += 2 * static_cast<uint64_t>(row_chunks);
    backend->stats_col_chunks += node_col_chunks;
    backend->stats_npu_us += node_npu_us;
    backend->stats_pack_us += node_pack_us;
    backend->stats_gather_us += node_gather_us;
    if (backend->profile_nodes) {
        GGML_LOG_INFO(
            "spiral geglu %-32s K=%-6d N=%-7d rows=%-3lld chunks=%-3d "
            "npu=%8.1f us pack=%7.1f gather=%7.1f total=%7.1f\n",
            glu->name, weights_rows, gate_cols, (long long) rows,
            node_col_chunks, node_npu_us, node_pack_us, node_gather_us,
            spiral_now_us() - node_start);
    }
    return true;
}

spiral_ffn_chain * spiral_prepare_ffn_chain(
    ggml_backend_spiral * backend, ggml_backend_spiral_buffer * gate,
    ggml_backend_spiral_buffer * up, ggml_backend_spiral_buffer * down,
    const spiral_prenorm_run * prenorm = nullptr) {
    auto build_prenorm = [&](spiral_ffn_chain & cached,
                             spiral_weight_chain * gate_chain) -> bool {
        if (prenorm == nullptr || prenorm->weight == nullptr) {
            return true;
        }
        if (cached.has_prenorm &&
            cached.prenorm_weight == prenorm->weight->data &&
            cached.prenorm_eps == prenorm->eps) {
            return true;
        }
        if (gate_chain == nullptr || gate_chain->prenorm_prepared.empty() ||
            down->prepared.empty() || down->shapes.empty()) {
            return false;
        }
        spiral::ChainOp ops[2];
        ops[0].shape = gate_chain->shapes[0];
        ops[0].shape.epilogue = spiral::kEpilogueGeglu;
        ops[0].shape.soft_cap = 0.0f;
        ops[0].shape.prenorm = true;
        ops[0].weights = &gate_chain->prenorm_prepared[0];
        ops[1].shape = down->shapes[0];
        ops[1].shape.epilogue = spiral::kEpilogueNone;
        ops[1].shape.soft_cap = 0.0f;
        ops[1].weights = &down->prepared[0];
        try {
            cached.prenorm_chain = backend->dev->program->prepare_chain(ops, 2);
        } catch (const std::exception & error) {
            GGML_LOG_ERROR("spiral: FFN prenorm chain preparation failed: %s\n",
                           error.what());
            cached.has_prenorm = false;
            return false;
        }
        cached.prenorm_weight = prenorm->weight->data;
        cached.prenorm_eps = prenorm->eps;
        cached.has_prenorm = true;
        return true;
    };

    ggml_backend_spiral_buffer * pair[2] = {gate, up};
    for (auto & cached : backend->ffn_chains) {
        if (cached.gate == gate && cached.up == up && cached.down == down &&
            cached.gate_epoch == gate->epoch &&
            cached.up_epoch == up->epoch &&
            cached.down_epoch == down->epoch) {
            if (prenorm != nullptr) {
                spiral_weight_chain * gate_chain =
                    spiral_prepare_chain(backend, pair, 2, prenorm);
                if (!build_prenorm(cached, gate_chain)) {
                    return nullptr;
                }
            }
            return &cached;
        }
    }
    if (!gate->ensure_prepared() || !up->ensure_prepared() ||
        !down->ensure_prepared()) {
        return nullptr;
    }
    spiral_weight_chain * gate_chain =
        spiral_prepare_chain(backend, pair, 2, prenorm);
    if (gate_chain == nullptr || gate_chain->prepared.empty() ||
        gate_chain->shapes.empty() || down->prepared.empty() ||
        down->shapes.empty()) {
        return nullptr;
    }

    spiral::ChainOp ops[2];
    ops[0].shape = gate_chain->shapes[0];
    ops[0].shape.epilogue = spiral::kEpilogueGeglu;
    ops[0].shape.soft_cap = 0.0f;
    ops[0].weights = &gate_chain->prepared[0];
    ops[1].shape = down->shapes[0];
    ops[1].shape.epilogue = spiral::kEpilogueNone;
    ops[1].shape.soft_cap = 0.0f;
    ops[1].weights = &down->prepared[0];

    spiral_ffn_chain cached;
    cached.gate = gate;
    cached.up = up;
    cached.down = down;
    cached.gate_epoch = gate->epoch;
    cached.up_epoch = up->epoch;
    cached.down_epoch = down->epoch;
    const double prepare_start = spiral_now_us();
    try {
        cached.chain = backend->dev->program->prepare_chain(ops, 2);
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("spiral: FFN chain preparation failed: %s\n",
                       error.what());
        return nullptr;
    }
    if (!build_prenorm(cached, gate_chain)) {
        return nullptr;
    }
    backend->dev->prepare_us += spiral_now_us() - prepare_start;
    backend->ffn_chains.push_back(std::move(cached));
    return &backend->ffn_chains.back();
}

bool spiral_run_ffn_chain(ggml_backend_spiral * backend,
                          spiral_ffn_chain & chain, ggml_tensor * gate,
                          ggml_tensor * down,
                          const spiral_prenorm_run * prenorm = nullptr) {
    const auto * src1 = gate->src[1];
    const auto * input = prenorm != nullptr ? prenorm->input : src1;
    if (src1 == nullptr || input == nullptr) {
        return false;
    }
    if (prenorm != nullptr && !chain.has_prenorm) {
        return false;
    }
    const int32_t weights_rows = input->ne[0];
    const int32_t final_cols = down->ne[0];
    const int64_t rows = input->ne[1] * input->ne[2] * input->ne[3];
    if (rows <= 0) {
        return true;  // zero-row graph; nothing to compute
    }
    if (final_cols <= 0) {
        return false;
    }

    backend->input_staging.resize(
        static_cast<size_t>(kSpiralInputsVecs) * weights_rows);
    backend->output_staging.resize(
        static_cast<size_t>(kSpiralInputsVecs) * final_cols);

    const auto * x = static_cast<const float *>(input->data);
    auto * y = static_cast<float *>(down->data);
    const double node_start = spiral_now_us();
    double node_npu_us = 0.0;
    double node_pack_us = 0.0;
    double node_gather_us = 0.0;
    int32_t node_chunks = 0;

    for (int64_t first = 0; first < rows; first += kSpiralInputsVecs) {
        const int32_t chunk_rows = static_cast<int32_t>(
            std::min<int64_t>(kSpiralInputsVecs, rows - first));
        const double pack_start = spiral_now_us();
        std::fill(backend->input_staging.begin(),
                  backend->input_staging.end(), 0);
        for (int32_t r = 0; r < chunk_rows; ++r) {
            ggml_fp32_to_bf16_row(
                x + (first + r) * weights_rows,
                reinterpret_cast<ggml_bf16_t *>(backend->input_staging.data()) +
                    static_cast<size_t>(r) * weights_rows,
                weights_rows);
        }
        node_pack_us += spiral_now_us() - pack_start;

        spiral::RunTimings timings;
        backend->dev->program->run_chain(
            prenorm != nullptr ? chain.prenorm_chain : chain.chain,
            backend->input_staging.data(), backend->output_staging.data(),
            final_cols, 0, &timings);
        node_npu_us += timings.txn_us + timings.input_us + timings.kernel_us +
                       timings.unpack_us;
        backend->stats_npu_txn_us += timings.txn_us;
        backend->stats_npu_input_us += timings.input_us;
        backend->stats_npu_kernel_us += timings.kernel_us;
        backend->stats_npu_unpack_us += timings.unpack_us;

        const double gather_start = spiral_now_us();
        for (int32_t r = 0; r < chunk_rows; ++r) {
            const auto * row = reinterpret_cast<const ggml_bf16_t *>(
                backend->output_staging.data() +
                static_cast<size_t>(r) * final_cols);
            ggml_bf16_to_fp32_row(
                row, y + (first + r) * final_cols, final_cols);
        }
        node_gather_us += spiral_now_us() - gather_start;
        ++node_chunks;
    }

    backend->stats_ffn_chains++;
    backend->stats_chains++;
    backend->stats_mul_mats += 3;
    backend->stats_row_chunks +=
        3 * ((rows + kSpiralInputsVecs - 1) / kSpiralInputsVecs);
    backend->stats_col_chunks += node_chunks;
    backend->stats_npu_us += node_npu_us;
    backend->stats_pack_us += node_pack_us;
    backend->stats_gather_us += node_gather_us;
    if (backend->profile_nodes) {
        GGML_LOG_INFO(
            "spiral ffn chain %-28s K=%-6d N=%-7d rows=%-3lld chunks=%-3d "
            "npu=%8.1f us pack=%7.1f gather=%7.1f total=%7.1f\n",
            down->name, weights_rows, final_cols, (long long) rows,
            node_chunks, node_npu_us, node_pack_us, node_gather_us,
            spiral_now_us() - node_start);
    }
    return true;
}

ggml_cgraph spiral_graph_view(ggml_cgraph * cgraph, int i0, int i1) {
    ggml_cgraph view = *cgraph;
    view.nodes = cgraph->nodes + i0;
    view.n_nodes = i1 - i0;
    view.n_leafs = 0;
    view.leafs = nullptr;
    view.grads = nullptr;
    view.grad_accs = nullptr;
    return view;
}

// Match MUL_MAT -> SCALE(1/cap) -> UNARY(TANH) -> SCALE(cap), the final-logit
// softcap emitted by the Gemma-family graphs.  Only fuse when every
// intermediate has a single consumer and the two scales are reciprocal; a
// different pre-scale (Grok's attention multiplier) deliberately falls back.
bool spiral_softcap_pattern(const ggml_cgraph * cgraph, int32_t i, float * cap) {
    if (i + 3 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * mm = cgraph->nodes[i];
    const ggml_tensor * s1 = cgraph->nodes[i + 1];
    const ggml_tensor * t  = cgraph->nodes[i + 2];
    const ggml_tensor * s2 = cgraph->nodes[i + 3];
    if (mm->op != GGML_OP_MUL_MAT || mm->type != GGML_TYPE_F32) {
        return false;
    }
    if (s1->op != GGML_OP_SCALE || s1->src[0] != mm || s1->src[1] != nullptr) {
        return false;
    }
    if (t->op != GGML_OP_UNARY || ggml_get_unary_op(t) != GGML_UNARY_OP_TANH ||
        t->src[0] != s1 || t->src[1] != nullptr) {
        return false;
    }
    if (s2->op != GGML_OP_SCALE || s2->src[0] != t || s2->src[1] != nullptr) {
        return false;
    }
    if (s1->type != GGML_TYPE_F32 || t->type != GGML_TYPE_F32 ||
        s2->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(s1) || !ggml_is_contiguous(t) ||
        !ggml_is_contiguous(s2)) {
        return false;
    }
    if (s1->view_src != nullptr || t->view_src != nullptr ||
        s2->view_src != nullptr) {
        return false;
    }
    for (int32_t d = 0; d < GGML_MAX_DIMS; ++d) {
        if (s1->ne[d] != mm->ne[d] || t->ne[d] != mm->ne[d] ||
            s2->ne[d] != mm->ne[d]) {
            return false;
        }
    }

    float p1[2] = {};
    float p2[2] = {};
    std::memcpy(p1, s1->op_params, sizeof(p1));
    std::memcpy(p2, s2->op_params, sizeof(p2));
    if (p1[1] != 0.0f || p2[1] != 0.0f) {
        return false;
    }
    if (!(p2[0] > 0.0f) || std::fabs(p1[0] * p2[0] - 1.0f) > 1e-3f) {
        return false;
    }

    for (int32_t j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        for (int32_t k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * src = n->src[k];
            if (src == mm && n != s1) {
                return false;
            }
            if (src == s1 && n != t) {
                return false;
            }
            if (src == t && n != s2) {
                return false;
            }
        }
    }

    *cap = p2[0];
    return true;
}

bool spiral_geglu_pattern(const ggml_backend_spiral_device * dev,
                          const ggml_cgraph * cgraph, int32_t i,
                          ggml_tensor ** glu_out) {
    if (i + 3 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * gate = cgraph->nodes[i];
    ggml_tensor * up = cgraph->nodes[i + 1];
    ggml_tensor * glu = cgraph->nodes[i + 2];
    ggml_tensor * down = cgraph->nodes[i + 3];
    if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT) {
        return false;
    }
    // Reuse the gate+up pair checks: same input, K, format, and band budget.
    if (spiral_chain_length(dev, cgraph->nodes + i, 2) != 2) {
        return false;
    }
    const auto * gate_buffer = spiral_weight_buffer_of(dev, gate);
    const auto * up_buffer = spiral_weight_buffer_of(dev, up);
    if (gate_buffer == nullptr || up_buffer == nullptr ||
        gate_buffer->padded_cols != up_buffer->padded_cols) {
        return false;
    }
    if (glu->op != GGML_OP_GLU ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_GEGLU) {
        return false;
    }
    if (glu->src[0] != gate || glu->src[1] != up) {
        return false;
    }
    if (glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) ||
        glu->view_src != nullptr) {
        return false;
    }
    if (gate->ne[0] != up->ne[0] || glu->ne[0] != gate->ne[0]) {
        return false;
    }
    for (int32_t d = 1; d < GGML_MAX_DIMS; ++d) {
        if (glu->ne[d] != gate->ne[d]) {
            return false;
        }
    }
    if (down->op != GGML_OP_MUL_MAT || down->src[1] != glu ||
        !spiral_supports_mul_mat(dev, down)) {
        return false;
    }
    // The gate and up results are not materialised, so each must feed only the
    // GLU; the GLU must feed only the down matmul.
    for (int32_t j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        for (int32_t k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * src = n->src[k];
            if ((src == gate || src == up) && n != glu) {
                return false;
            }
            if (src == glu && n != down) {
                return false;
            }
        }
    }
    *glu_out = glu;
    return true;
}

int32_t spiral_max_prenorm_k() {
    return (design::kInputGroupElems * 2 - design::kPrenormWeightOffset) / 4;
}

// Match the forced prelude: `MUL_MAT` over `MUL(RMS_NORM(x), gain)` or bare
// `RMS_NORM(x)`.  Only used with SPIRAL_PRENORM=1 for the end-to-end A/B.
bool spiral_prenorm_source(const ggml_backend_spiral_device * dev,
                           const ggml_tensor * mm,
                           spiral_prenorm_run * run) {
    if (mm == nullptr || mm->op != GGML_OP_MUL_MAT || dev == nullptr ||
        run == nullptr) {
        return false;
    }
    const ggml_tensor * src1 = mm->src[1];
    if (src1 == nullptr) {
        return false;
    }
    const ggml_tensor * norm = nullptr;
    const ggml_tensor * gain = nullptr;
    if (src1->op == GGML_OP_RMS_NORM) {
        norm = src1;
    } else if (src1->op == GGML_OP_MUL) {
        norm = src1->src[0];
        gain = src1->src[1];
    }
    if (norm == nullptr || norm->op != GGML_OP_RMS_NORM) {
        return false;
    }
    if (gain != nullptr) {
        if (gain->type != GGML_TYPE_F32 || !ggml_is_contiguous(gain) ||
            gain->view_src != nullptr || gain->ne[1] != 1 ||
            gain->ne[2] != 1 || gain->ne[3] != 1) {
            return false;
        }
    }
    const ggml_tensor * x = norm->src[0];
    if (x == nullptr || x->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(x) || x->view_src != nullptr) {
        return false;
    }
    if (norm->type != GGML_TYPE_F32 || !ggml_is_contiguous(norm) ||
        src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1)) {
        return false;
    }
    if (gain != nullptr && gain->ne[0] != x->ne[0]) {
        return false;
    }
    if (x->ne[0] <= 0 || x->ne[0] % 256 != 0 ||
        x->ne[0] > spiral_max_prenorm_k()) {
        return false;
    }
    if (!spiral_supports_mul_mat(dev, mm) || mm->src[0]->ne[0] != x->ne[0]) {
        return false;
    }
    // The prelude needs the gain; a bare norm would also need an all-ones
    // config weight, which the experiment does not build yet.
    if (gain == nullptr) {
        return false;
    }
    run->input = x;
    run->weight = gain;
    run->eps = 0.0f;
    std::memcpy(&run->eps, norm->op_params, sizeof(float));
    return true;
}

// One pass over the graph builds the skip plan.  A norm/gain node is in the
// plan only if every consumer is a prenorm matmul, so skipping it cannot strand
// another node.
void spiral_build_prenorm_plan(const ggml_backend_spiral_device * dev,
                               const ggml_cgraph * cgraph,
                               std::unordered_map<const ggml_tensor *, bool> & plan) {
    plan.clear();
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> norm_gain;
    std::unordered_map<const ggml_tensor *, int32_t> uses;
    for (int32_t i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (t->op != GGML_OP_RMS_NORM && t->op != GGML_OP_MUL) {
            continue;
        }
        bool ok = t->view_src == nullptr &&
                  (t->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
        if (t->op == GGML_OP_MUL &&
            (t->src[0] == nullptr || t->src[0]->op != GGML_OP_RMS_NORM ||
             t->src[1] == nullptr)) {
            ok = false;
        }
        plan.emplace(t, ok);
    }
    for (int32_t j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        for (int32_t k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * src = n->src[k];
            if (src == nullptr) {
                continue;
            }
            auto node = plan.find(src);
            if (node == plan.end()) {
                continue;
            }
            ++uses[src];
            if (!node->second) {
                continue;
            }
            if (src->op == GGML_OP_MUL) {
                spiral_prenorm_run run;
                node->second = n->op == GGML_OP_MUL_MAT &&
                               n->src[1] == src &&
                               spiral_prenorm_source(dev, n, &run);
            } else if (n->op == GGML_OP_MUL && n->src[0] == src) {
                if (norm_gain.count(src) != 0) {
                    node->second = false;  // more than one gain consumer
                } else {
                    norm_gain.emplace(src, n);
                }
            } else {
                spiral_prenorm_run run;
                node->second = n->op == GGML_OP_MUL_MAT &&
                               n->src[1] == src &&
                               spiral_prenorm_source(dev, n, &run);
            }
        }
    }
    // One prelude per fused call.  A norm/gain with several matmul consumers
    // is only worth taking when they form one wide chain; otherwise each
    // consumer would repeat the sum and scale, which loses to the host norm.
    for (auto & [t, ok] : plan) {
        if (!ok) {
            continue;
        }
        std::vector<int32_t> consumers;
        for (int32_t j = 0; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n->op != GGML_OP_MUL_MAT) {
                continue;
            }
            for (int32_t k = 0; k < GGML_MAX_SRC; ++k) {
                if (n->src[k] == t) {
                    consumers.push_back(j);
                    break;
                }
            }
        }
        if (consumers.empty()) {
            continue;
        }
        if (consumers.size() == 1) {
            const int32_t single_remaining = cgraph->n_nodes - consumers[0];
            if (spiral_chain_length(
                    dev, cgraph->nodes + consumers[0],
                    std::min(single_remaining, kSpiralMaxChain)) < 2) {
                ok = false;
            }
            continue;
        }
        bool contiguous = true;
        for (size_t c = 1; c < consumers.size(); ++c) {
            contiguous = contiguous && consumers[c] == consumers[0] + c;
        }
        const int32_t remaining = cgraph->n_nodes - consumers[0];
        const int32_t length = spiral_chain_length(
            dev, cgraph->nodes + consumers[0],
            std::min(remaining, kSpiralMaxChain));
        if (!contiguous ||
            length < static_cast<int32_t>(consumers.size())) {
            ok = false;
        }
    }
    for (const auto & [norm, gain] : norm_gain) {
        auto norm_node = plan.find(norm);
        auto gain_node = plan.find(gain);
        if (norm_node != plan.end()) {
            norm_node->second =
                norm_node->second && gain_node != plan.end() &&
                gain_node->second;
        }
    }
    for (auto & [t, ok] : plan) {
        if (ok && uses[t] == 0) {
            ok = false;
        }
    }
}

enum ggml_status spiral_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * self = static_cast<ggml_backend_spiral *>(backend->context);
    const double graph_start = spiral_now_us();
    const uint64_t calls0 = self->stats_mul_mats;
    const uint64_t cpu_nodes0 = self->stats_cpu_nodes;
    const uint64_t cpu_batches0 = self->stats_cpu_batches;
    const double cpu0 = self->stats_cpu_us;
    const double npu0 = self->stats_npu_us;
    const double pack0 = self->stats_pack_us;
    const double gather0 = self->stats_gather_us;

    // Calls whose device work was started early.  The host output is still
    // unpacked when the graph reaches the node, so no host tensor is written
    // out of order; only the NPU work is moved ahead of the intervening CPU
    // ops.  This is safe because the input data is already live and the
    // weights are immutable.
    struct DeferredCall {
        ggml_tensor * node = nullptr;
        spiral::RunTicket ticket;
        int32_t chunk_rows = 0;
        int32_t weights_rows = 0;
        int32_t weights_cols = 0;
    };
    std::vector<DeferredCall> deferred;

    auto deferred_contains = [&](const ggml_tensor * node) {
        for (const auto & call : deferred) {
            if (call.node == node) {
                return true;
            }
        }
        return false;
    };

    auto single_call_buffer = [&](ggml_tensor * node)
        -> ggml_backend_spiral_buffer * {
        if (!spiral_supports_mul_mat(self->dev, node)) {
            return nullptr;
        }
        auto * weight_buffer = spiral_weight_buffer_of(self->dev, node);
        if (weight_buffer == nullptr ||
            weight_buffer->prepared.size() != 1 ||
            weight_buffer->shapes.size() != 1) {
            return nullptr;
        }
        const auto * src1 = node->src[1];
        if (src1 == nullptr) {
            return nullptr;
        }
        const int64_t rows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        if (rows <= 0 || rows > kSpiralInputsVecs) {
            return nullptr;
        }
        return weight_buffer;
    };

    auto submit_deferred = [&](ggml_tensor * node,
                               ggml_backend_spiral_buffer * weight_buffer) {
        const auto * src1 = node->src[1];
        const int64_t rows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        const int32_t chunk_rows = static_cast<int32_t>(rows);
        const auto & shape = weight_buffer->shapes[0];
        self->input_staging.resize(
            static_cast<size_t>(kSpiralInputsVecs) * shape.weights_rows);
        // Keep the output staging area at its maximum size: a deferred
        // ticket points into it, and shrinking/growing it past capacity
        // would invalidate that pointer.
        self->output_staging.resize(
            static_cast<size_t>(kSpiralInputsVecs) *
            design::kMaxBandsPerChunk * design::kBandCols);
        const auto * x = static_cast<const float *>(src1->data);
        std::fill(self->input_staging.begin(),
                  self->input_staging.end(), 0);
        for (int32_t r = 0; r < chunk_rows; ++r) {
            ggml_fp32_to_bf16_row(
                x + static_cast<int64_t>(r) * shape.weights_rows,
                reinterpret_cast<ggml_bf16_t *>(
                    self->input_staging.data()) +
                    static_cast<size_t>(r) * shape.weights_rows,
                shape.weights_rows);
        }
        DeferredCall call;
        call.node = node;
        call.chunk_rows = chunk_rows;
        call.weights_rows = shape.weights_rows;
        call.weights_cols = weight_buffer->weights_cols;
        call.ticket = self->dev->program->submit(
            shape, weight_buffer->prepared[0], self->input_staging.data(),
            self->output_staging.data(), shape.weights_cols, 0);
        deferred.push_back(call);
        self->stats_mul_mats++;
        self->stats_row_chunks++;
        self->stats_col_chunks++;
    };

    // Start the next same-input matmul before the intervening CPU work.
    // Cross-node look-ahead is opt-in: on Gemma it is within noise of the
    // plain pipeline, and it adds slot/ordering risk.
    static const bool lookahead_enabled =
        std::getenv("SPIRAL_LOOKAHEAD") != nullptr;
    auto look_ahead = [&](int32_t index) {
        const int32_t slots = self->dev->program->pipeline_slots();
        if (!lookahead_enabled || slots <= 1 ||
            static_cast<int32_t>(deferred.size()) >= slots) {
            return;
        }
        ggml_tensor * base = cgraph->nodes[index];
        if (single_call_buffer(base) == nullptr) {
            return;
        }
        if (spiral_chain_length(
                self->dev, cgraph->nodes + index,
                std::min(cgraph->n_nodes - index, kSpiralMaxChain)) >= 2) {
            return;
        }
        const auto * src1 = base->src[1];
        for (int32_t j = index + 1;
             j < cgraph->n_nodes && j <= index + 8; ++j) {
            ggml_tensor * next = cgraph->nodes[j];
            if (!spiral_supports_mul_mat(self->dev, next)) {
                if (!spiral_cpu_compatible(self->dev, next)) {
                    return;
                }
                continue;
            }
            if (deferred_contains(next)) {
                continue;
            }
            auto * weight_buffer = single_call_buffer(next);
            if (weight_buffer == nullptr ||
                !spiral_same_input(src1, next->src[1])) {
                return;
            }
            // Leave chained calls to the chain path so gate+up is not
            // computed twice or skipped with its slot still busy.
            const int32_t chain_len = spiral_chain_length(
                self->dev, cgraph->nodes + j,
                std::min(cgraph->n_nodes - j, kSpiralMaxChain));
            if (chain_len >= 2) {
                return;
            }
            for (int32_t t = std::max(index, j - kSpiralMaxChain + 1);
                 t < j; ++t) {
                const int32_t earlier = spiral_chain_length(
                    self->dev, cgraph->nodes + t,
                    std::min(cgraph->n_nodes - t, kSpiralMaxChain));
                if (earlier >= 2 && t + earlier > j) {
                    return;
                }
            }
            submit_deferred(next, weight_buffer);
            return;
        }
    };

    int i = 0;
    ggml_tensor * geglu_gate = nullptr;
    ggml_tensor * geglu_up = nullptr;
    ggml_tensor * geglu_out = nullptr;
    spiral_weight_chain * geglu_chain = nullptr;
    spiral_prenorm_run geglu_prenorm_run;
    bool geglu_has_prenorm = false;
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_glu = nullptr;
    ggml_tensor * ffn_down = nullptr;
    spiral_ffn_chain * ffn_chain = nullptr;
    spiral_prenorm_run ffn_prenorm_run;
    bool ffn_has_prenorm = false;
    // Dynamic per-graph switch: production builds select the prelude per
    // pattern.  SPIRAL_NO_PRENORM=1 disables it.
    static const bool prenorm_enabled =
        design::kPrenormAvailable != 0 &&
        std::getenv("SPIRAL_NO_PRENORM") == nullptr;
    std::unordered_map<const ggml_tensor *, bool> prenorm_plan;
    if (prenorm_enabled) {
        spiral_build_prenorm_plan(self->dev, cgraph, prenorm_plan);
    }
    static const bool geglu_fuse =
        std::getenv("SPIRAL_NO_GEGLU_FUSE") == nullptr;
    static const bool ffn_chain_enabled =
        std::getenv("SPIRAL_NO_FFN_CHAIN") == nullptr;
    while (i < cgraph->n_nodes) {
        ggml_tensor * node = cgraph->nodes[i];

        // If this node's device work was started early, unpack it here, at
        // its original graph position.
        bool finished_early = false;
        for (auto it = deferred.begin(); it != deferred.end(); ++it) {
            if (it->node != node) {
                continue;
            }
            DeferredCall call = *it;
            deferred.erase(it);
            self->dev->program->finish(call.ticket);
            const spiral::RunTimings & timings = call.ticket.timings;
            const double node_us = timings.txn_us + timings.input_us +
                                   timings.kernel_us + timings.unpack_us;
            self->stats_npu_txn_us += timings.txn_us;
            self->stats_npu_input_us += timings.input_us;
            self->stats_npu_kernel_us += timings.kernel_us;
            self->stats_npu_unpack_us += timings.unpack_us;
            self->stats_npu_us += node_us;
            auto * y = static_cast<float *>(node->data);
            for (int32_t r = 0; r < call.chunk_rows; ++r) {
                const auto * row = reinterpret_cast<const ggml_bf16_t *>(
                    self->output_staging.data() +
                    static_cast<size_t>(r) * call.ticket.outputs_cols);
                ggml_bf16_to_fp32_row(
                    row, y + static_cast<int64_t>(r) * call.weights_cols,
                    call.weights_cols);
            }
            if (self->profile_nodes) {
                GGML_LOG_INFO(
                    "spiral node %-32s K=%-6d N=%-7d rows=%-3d early "
                    "kernel=%8.1f wait=%8.1f total=%7.1f\n",
                    node->name, call.weights_rows, call.weights_cols,
                    call.chunk_rows, timings.kernel_us, timings.wait_us,
                    node_us);
            }
            look_ahead(i - 1);
            ++i;
            finished_early = true;
            break;
        }
        if (finished_early) {
            continue;
        }

        // The whole FFN chain was detected at the gate node; run it now, at
        // the down node, so the down output is written when its lifetime
        // starts.
        if (ffn_chain != nullptr) {
            if (node != ffn_down) {
                GGML_LOG_ERROR(
                    "spiral: expected FFN down node %s, got %s (%s)\n",
                    ffn_down->name, node->name, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
            }
            if (!spiral_run_ffn_chain(
                    self, *ffn_chain, ffn_gate, ffn_down,
                    ffn_has_prenorm ? &ffn_prenorm_run : nullptr)) {
                GGML_LOG_ERROR("spiral: FFN chain for %s failed\n",
                               ffn_down->name);
                return GGML_STATUS_FAILED;
            }
            ffn_gate = nullptr;
            ffn_up = nullptr;
            ffn_glu = nullptr;
            ffn_down = nullptr;
            ffn_chain = nullptr;
            ffn_has_prenorm = false;
            ++i;
            continue;
        }

        if (geglu_out != nullptr) {
            if (node != geglu_out) {
                GGML_LOG_ERROR(
                    "spiral: expected GEGLU node %s, got %s (%s)\n",
                    geglu_out->name, node->name, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
            }
            if (!spiral_run_geglu_pair(
                    self, geglu_gate, geglu_up, geglu_out, *geglu_chain,
                    geglu_has_prenorm ? &geglu_prenorm_run : nullptr)) {
                GGML_LOG_ERROR("spiral: GEGLU pair for %s failed\n",
                               geglu_out->name);
                return GGML_STATUS_FAILED;
            }
            self->stats_geglu_fused++;
            geglu_gate = nullptr;
            geglu_up = nullptr;
            geglu_out = nullptr;
            geglu_chain = nullptr;
            geglu_has_prenorm = false;
            ++i;
            continue;
        }

        if (prenorm_enabled && !spiral_supports_mul_mat(self->dev, node)) {
            auto plan = prenorm_plan.find(node);
            if (plan != prenorm_plan.end() && plan->second) {
                ++i;
                continue;
            }
        }

        if (spiral_supports_mul_mat(self->dev, node)) {
            spiral_prenorm_run prenorm_run;
            bool has_prenorm = false;
            if (prenorm_enabled) {
                const ggml_tensor * key = node->src[1];
                auto plan = key != nullptr ? prenorm_plan.find(key)
                                           : prenorm_plan.end();
                has_prenorm = plan != prenorm_plan.end() && plan->second &&
                              spiral_prenorm_source(self->dev, node,
                                                    &prenorm_run);
            }
            static const bool softcap_fuse =
                std::getenv("SPIRAL_NO_SOFTCAP_FUSE") == nullptr;
            float soft_cap = 0.0f;
            if (!has_prenorm && softcap_fuse &&
                spiral_softcap_pattern(cgraph, i, &soft_cap)) {
                if (!spiral_run_mul_mat(self, node, cgraph->nodes[i + 3],
                                        soft_cap)) {
                    return GGML_STATUS_FAILED;
                }
                self->stats_softcap_fused++;
                i += 4;
                continue;
            }
            if (ffn_chain_enabled) {
                ggml_tensor * glu = nullptr;
                if (spiral_geglu_pattern(self->dev, cgraph, i, &glu)) {
                    auto * b0 = spiral_weight_buffer_of(self->dev, node);
                    auto * b1 = spiral_weight_buffer_of(
                        self->dev, cgraph->nodes[i + 1]);
                    auto * bd = spiral_weight_buffer_of(
                        self->dev, cgraph->nodes[i + 3]);
                    if (b0 != nullptr && b1 != nullptr && bd != nullptr) {
                        spiral_ffn_chain * chain =
                            spiral_prepare_ffn_chain(
                                self, b0, b1, bd,
                                has_prenorm ? &prenorm_run : nullptr);
                        if (chain != nullptr) {
                            ffn_gate = node;
                            ffn_up = cgraph->nodes[i + 1];
                            ffn_glu = glu;
                            ffn_down = cgraph->nodes[i + 3];
                            ffn_chain = chain;
                            ffn_prenorm_run = prenorm_run;
                            ffn_has_prenorm = has_prenorm;
                            if (has_prenorm) {
                                self->stats_prenorm_fused++;
                                self->stats_prenorm_chains++;
                            }
                            i += 3;
                            continue;
                        }
                    }
                }
            }
            if (geglu_fuse) {
                ggml_tensor * glu = nullptr;
                if (spiral_geglu_pattern(self->dev, cgraph, i, &glu)) {
                    auto * b0 = spiral_weight_buffer_of(self->dev, node);
                    auto * b1 = spiral_weight_buffer_of(
                        self->dev, cgraph->nodes[i + 1]);
                    ggml_backend_spiral_buffer * pair[2] = {b0, b1};
                    spiral_weight_chain * chain =
                        (b0 != nullptr && b1 != nullptr)
                            ? spiral_prepare_chain(
                                  self, pair, 2,
                                  has_prenorm ? &prenorm_run : nullptr)
                            : nullptr;
                    if (chain != nullptr) {
                        geglu_gate = node;
                        geglu_up = cgraph->nodes[i + 1];
                        geglu_out = glu;
                        geglu_chain = chain;
                        geglu_prenorm_run = prenorm_run;
                        geglu_has_prenorm = has_prenorm;
                        if (has_prenorm) {
                            self->stats_prenorm_fused++;
                            self->stats_prenorm_chains++;
                        }
                        i += 2;
                        continue;
                    }
                }
            }
            const int32_t remaining = cgraph->n_nodes - i;
            const int32_t length = spiral_chain_length(
                self->dev, cgraph->nodes + i,
                std::min(remaining, kSpiralMaxChain));
            if (length >= 2) {
                ggml_backend_spiral_buffer * buffers[kSpiralMaxChain] = {};
                bool have_buffers = true;
                for (int32_t j = 0; j < length; ++j) {
                    buffers[j] = spiral_weight_buffer_of(
                        self->dev, cgraph->nodes[i + j]);
                    have_buffers = have_buffers && buffers[j] != nullptr;
                }
                spiral_weight_chain * chain =
                    have_buffers
                        ? spiral_prepare_chain(
                              self, buffers, length,
                              has_prenorm ? &prenorm_run : nullptr)
                        : nullptr;
                if (chain != nullptr &&
                    spiral_run_chain(self, cgraph->nodes + i, length,
                                     *chain,
                                     has_prenorm ? &prenorm_run : nullptr)) {
                    if (has_prenorm) {
                        self->stats_prenorm_fused++;
                        self->stats_prenorm_chains++;
                    }
                    i += length;
                    continue;
                }
            }
            if (!spiral_run_mul_mat(self, node, nullptr, 0.0f,
                                    has_prenorm ? &prenorm_run : nullptr)) {
                return GGML_STATUS_FAILED;
            }
            if (has_prenorm) {
                self->stats_prenorm_fused++;
            }
            look_ahead(i);
            ++i;
            continue;
        }

        int j = i;
        while (j < cgraph->n_nodes &&
               !spiral_supports_mul_mat(self->dev, cgraph->nodes[j])) {
            if (prenorm_enabled && j > i) {
                auto plan = prenorm_plan.find(cgraph->nodes[j]);
                if (plan != prenorm_plan.end() && plan->second) {
                    break;
                }
            }
            if (!spiral_cpu_compatible(self->dev, cgraph->nodes[j])) {
                GGML_LOG_ERROR("spiral: no CPU fallback for %s (%s)\n",
                               ggml_op_name(cgraph->nodes[j]->op),
                               cgraph->nodes[j]->name);
                return GGML_STATUS_FAILED;
            }
            ++j;
        }
        if (j == i || self->cpu == nullptr) {
            return GGML_STATUS_FAILED;
        }

        if (self->profile_cpu) {
            for (int32_t k = i; k < j; ++k) {
                struct ggml_cgraph one = spiral_graph_view(cgraph, k, k + 1);
                const double node_start = spiral_now_us();
                if (ggml_backend_graph_compute(self->cpu, &one) !=
                    GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                const ggml_tensor * t = cgraph->nodes[k];
                GGML_LOG_INFO(
                    "spiral cpu %-28s %-12s ne=[%lld,%lld,%lld,%lld] "
                    "type=%d %8.1f us\n",
                    t->name, ggml_op_name(t->op), (long long) t->ne[0],
                    (long long) t->ne[1], (long long) t->ne[2],
                    (long long) t->ne[3], (int) t->type,
                    spiral_now_us() - node_start);
            }
            self->stats_cpu_us += 0;
        } else {
            struct ggml_cgraph view = spiral_graph_view(cgraph, i, j);
            const double cpu_start = spiral_now_us();
            if (ggml_backend_graph_compute(self->cpu, &view) !=
                GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            self->stats_cpu_us += spiral_now_us() - cpu_start;
        }
        self->stats_cpu_nodes += static_cast<uint64_t>(j - i);
        self->stats_cpu_batches++;
        i = j;
    }
    // Safety net: a deferred call should always be reached, but never leave a
    // pipeline slot busy across graphs if the look-ahead ever misses one.
    for (auto & call : deferred) {
        self->dev->program->finish(call.ticket);
        const spiral::RunTimings & timings = call.ticket.timings;
        self->stats_npu_txn_us += timings.txn_us;
        self->stats_npu_input_us += timings.input_us;
        self->stats_npu_kernel_us += timings.kernel_us;
        self->stats_npu_unpack_us += timings.unpack_us;
        self->stats_npu_us += timings.txn_us + timings.input_us +
                              timings.kernel_us + timings.unpack_us;
        auto * y = static_cast<float *>(call.node->data);
        for (int32_t r = 0; r < call.chunk_rows; ++r) {
            const auto * row = reinterpret_cast<const ggml_bf16_t *>(
                self->output_staging.data() +
                static_cast<size_t>(r) * call.ticket.outputs_cols);
            ggml_bf16_to_fp32_row(
                row, y + static_cast<int64_t>(r) * call.weights_cols,
                call.weights_cols);
        }
    }
    deferred.clear();
    const double graph_us = spiral_now_us() - graph_start;
    self->stats_wall_us += graph_us;
    self->stats_graphs++;
    if (self->profile_graph) {
        GGML_LOG_INFO("spiral graph %llu: wall=%.3f ms cpu=%.3f npu=%.3f "
                      "pack=%.3f gather=%.3f calls=%llu cpu_nodes=%llu cpu_batches=%llu\n",
                      (unsigned long long) self->stats_graphs,
                      graph_us / 1000.0,
                      (self->stats_cpu_us - cpu0) / 1000.0,
                      (self->stats_npu_us - npu0) / 1000.0,
                      (self->stats_pack_us - pack0) / 1000.0,
                      (self->stats_gather_us - gather0) / 1000.0,
                      (unsigned long long) (self->stats_mul_mats - calls0),
                      (unsigned long long) (self->stats_cpu_nodes - cpu_nodes0),
                      (unsigned long long) (self->stats_cpu_batches - cpu_batches0));
    }
    return GGML_STATUS_SUCCESS;
}

void spiral_backend_free(ggml_backend_t backend) {
    // backend points at the embedded `base` member of ggml_backend_spiral.
    auto * self = static_cast<ggml_backend_spiral *>(backend->context);
    if (self->profile) {
        GGML_LOG_INFO(
            "spiral profile: graphs=%llu wall=%.3f ms\n"
            "  cpu fallback: %.3f ms  nodes=%llu batches=%llu\n"
            "  npu mulmat:   %.3f ms  calls=%llu chains=%llu fused_softcap=%llu fused_geglu=%llu ffn_chains=%llu prenorm=%llu prenorm_chains=%llu row_chunks=%llu col_chunks=%llu max_slots=%d\n"
            "  npu stages:   txn=%.3f input=%.3f kernel=%.3f unpack=%.3f ms\n"
            "  host:         pack=%.3f gather=%.3f ms\n"
            "  weight prep:  %.3f ms  tensors=%llu\n",
            (unsigned long long) self->stats_graphs,
            self->stats_wall_us / 1000.0,
            self->stats_cpu_us / 1000.0,
            (unsigned long long) self->stats_cpu_nodes,
            (unsigned long long) self->stats_cpu_batches,
            self->stats_npu_us / 1000.0,
            (unsigned long long) self->stats_mul_mats,
            (unsigned long long) self->stats_chains,
            (unsigned long long) self->stats_softcap_fused,
            (unsigned long long) self->stats_geglu_fused,
            (unsigned long long) self->stats_ffn_chains,
            (unsigned long long) self->stats_prenorm_fused,
            (unsigned long long) self->stats_prenorm_chains,
            (unsigned long long) self->stats_row_chunks,
            (unsigned long long) self->stats_col_chunks,
            self->dev->program ? self->dev->program->max_outstanding() : 0,
            self->stats_npu_txn_us / 1000.0,
            self->stats_npu_input_us / 1000.0,
            self->stats_npu_kernel_us / 1000.0,
            self->stats_npu_unpack_us / 1000.0,
            self->stats_pack_us / 1000.0,
            self->stats_gather_us / 1000.0,
            self->dev->prepare_us / 1000.0,
            (unsigned long long) self->dev->prepare_tensors);
    }
    if (self->cpu != nullptr) {
        ggml_backend_free(self->cpu);
    }
    delete self;
}

const char * spiral_backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return kSpiralBackendName;
}

const ggml_backend_i spiral_backend_i = {
    /* .get_name                = */ spiral_backend_get_name,
    /* .free                    = */ spiral_backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ spiral_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

ggml_backend_buffer_type_t * spiral_get_extra_bufts(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_spiral_device *>(dev->context)->extra_bufts;
}

const char * spiral_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return kSpiralBackendName;
}

size_t spiral_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

ggml_backend_dev_t spiral_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    return &spiral_device(reg).base;
}

void * spiral_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return reinterpret_cast<void *>(spiral_get_extra_bufts);
    }
    return nullptr;
}

const ggml_backend_reg_i spiral_reg_i = {
    /* .get_name         = */ spiral_reg_get_name,
    /* .get_device_count = */ spiral_reg_get_device_count,
    /* .get_device       = */ spiral_reg_get_device,
    /* .get_proc_address = */ spiral_reg_get_proc_address,
};

}  // namespace

ggml_backend_t ggml_backend_spiral_init(int index) {
    auto & dev = spiral_device(ggml_backend_spiral_reg());
    if (index != 0 || !dev.open()) {
        return nullptr;
    }
    auto * backend = new ggml_backend_spiral(&dev);
    backend->cpu = ggml_backend_cpu_init();
    if (backend->cpu == nullptr) {
        delete backend;
        return nullptr;
    }
    // The fallback runs many small graph views.  Without a persistent CPU
    // threadpool, spawning several threads per view costs more than the work.
    // One thread is fastest by default; SPIRAL_CPU_THREADS can override it.
    int cpu_threads = 1;
    if (const char * env = std::getenv("SPIRAL_CPU_THREADS")) {
        const int configured = std::atoi(env);
        if (configured > 0) {
            cpu_threads = configured;
        }
    }
    ggml_backend_cpu_set_n_threads(backend->cpu, cpu_threads);
    backend->base.iface = spiral_backend_i;
    return &backend->base;
}

bool ggml_backend_is_spiral(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, spiral_guid());
}

ggml_backend_buffer_type_t ggml_backend_spiral_weights_buffer_type(void) {
    return &spiral_device(ggml_backend_spiral_reg()).weights_buffer_type.base;
}

ggml_backend_reg_t ggml_backend_spiral_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ spiral_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_spiral_reg)
