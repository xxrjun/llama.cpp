#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-topology.h"
#include "llama-kv-transfer.h"
#include "ggml-rpc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]\n", argv[0]);
    LOG("\n");
}

// Helper function to parse endpoint from device string (e.g., "RPC@ip:port=/path" -> "ip:port")
static std::string parse_rpc_endpoint(const std::string & device_str) {
    std::string result = device_str;

    // Remove "RPC@" prefix if present
    size_t at_pos = result.find('@');
    if (at_pos != std::string::npos) {
        result = result.substr(at_pos + 1);
    }

    // Remove "=/path" suffix if present (model path specification)
    size_t eq_pos = result.find('=');
    if (eq_pos != std::string::npos) {
        result = result.substr(0, eq_pos);
    }

    return result;
}

// Helper function to get model path for a specific device
// NOTE: Currently returns the same path for all devices.
// Model weights are transferred over RPC regardless of path.
// TODO: Future GPU pre-loading feature would allow RPC server to use its own local model.
static std::string get_model_path_for_device(
    const std::string & device,
    const common_params & params) {
    GGML_UNUSED(device);
    return params.model.path;
}

// Helper function to find or register RPC server device
static ggml_backend_dev_t register_rpc_device(const std::string & endpoint) {
    // First, check if the RPC device already exists (from topology query)
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);

        // Check if this RPC device matches our endpoint
        if (name && strncmp(name, "RPC", 3) == 0) {
            if (desc && strstr(desc, endpoint.c_str()) != nullptr) {
                LOG_INF("Reusing existing RPC device: %s\n", name);
                return dev;
            }
        }
    }

    // Not found, need to register
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        LOG_ERR("ERROR: RPC backend not available. Build with -DGGML_RPC=ON\n");
        return nullptr;
    }

    // Get the add_server function
    typedef ggml_backend_reg_t (*add_server_fn)(const char *);
    add_server_fn add_rpc_server = (add_server_fn)ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!add_rpc_server) {
        LOG_ERR("ERROR: Could not find ggml_backend_rpc_add_server function\n");
        return nullptr;
    }

    // Register the RPC server
    ggml_backend_reg_t server_reg = add_rpc_server(endpoint.c_str());
    if (!server_reg) {
        LOG_ERR("ERROR: Failed to register RPC server at %s\n", endpoint.c_str());
        return nullptr;
    }

    // Register it with the backend system
    ggml_backend_register(server_reg);

    // Find the newly registered device
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);

        if (name && strncmp(name, "RPC", 3) == 0) {
            if (desc && strstr(desc, endpoint.c_str()) != nullptr) {
                return dev;
            }
        }
    }

    return nullptr;
}

// Helper function to get local device (Metal, CUDA, or CPU)
static ggml_backend_dev_t get_local_device(const std::string & device_hint) {
    // Try exact match first
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(device_hint.c_str());
    if (dev) {
        return dev;
    }

    // Try to find by type
    if (device_hint == "Metal" || device_hint == "metal") {
        // Look for Metal device
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            dev = ggml_backend_dev_get(i);
            const char * name = ggml_backend_dev_name(dev);
            if (name && strstr(name, "Metal") != nullptr) {
                return dev;
            }
        }
    }

    if (device_hint == "CUDA" || device_hint == "cuda" || device_hint.find("CUDA") == 0) {
        // Look for CUDA device
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            dev = ggml_backend_dev_get(i);
            const char * name = ggml_backend_dev_name(dev);
            if (name && strstr(name, "CUDA") != nullptr) {
                return dev;
            }
        }
    }

    // Fallback: return first GPU device (excluding RPC)
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            const char * reg_name = ggml_backend_reg_name(reg);
            // Skip RPC devices
            if (reg_name && strcmp(reg_name, "RPC") != 0) {
                return dev;
            }
        }
    }

    return nullptr;
}

// Helper function to validate model compatibility across tiers
static bool validate_model_compatibility(
    const std::string & path1,
    const std::string & path2,
    const std::string & tier1_name,
    const std::string & tier2_name) {

    if (path1 == path2) {
        LOG_INF("Using same model file on both tiers: %s\n", path1.c_str());
        return true;  // Same file, definitely compatible
    }

    LOG_INF("Validating model compatibility:\n");
    LOG_INF("  %s: %s\n", tier1_name.c_str(), path1.c_str());
    LOG_INF("  %s: %s\n", tier2_name.c_str(), path2.c_str());

    // Load model metadata without full weights
    llama_model_params params1;
    params1 = llama_model_default_params();
    params1.use_mmap = false;

    llama_model_params params2;
    params2 = llama_model_default_params();
    params2.use_mmap = false;

    LOG_INF("Loading model metadata...\n");

    llama_model * model1 = llama_model_load_from_file(path1.c_str(), params1);
    llama_model * model2 = llama_model_load_from_file(path2.c_str(), params2);

    if (!model1) {
        LOG_ERR("ERROR: Failed to load model from %s\n", path1.c_str());
        return false;
    }

    if (!model2) {
        LOG_ERR("ERROR: Failed to load model from %s\n", path2.c_str());
        llama_model_free(model1);
        return false;
    }

    // Get model hyperparameters
    int32_t n_layer1 = llama_model_n_layer(model1);
    int32_t n_embd1 = llama_model_n_embd(model1);
    int32_t n_head1 = llama_model_n_head(model1);

    int32_t n_layer2 = llama_model_n_layer(model2);
    int32_t n_embd2 = llama_model_n_embd(model2);
    int32_t n_head2 = llama_model_n_head(model2);

    bool compatible = true;

    // Compare critical hyperparameters
    if (n_layer1 != n_layer2) {
        LOG_ERR("ERROR: Layer count mismatch: %d vs %d\n", n_layer1, n_layer2);
        compatible = false;
    }

    if (n_embd1 != n_embd2) {
        LOG_ERR("ERROR: Embedding dimension mismatch: %d vs %d\n", n_embd1, n_embd2);
        compatible = false;
    }

    if (n_head1 != n_head2) {
        LOG_ERR("ERROR: Head count mismatch: %d vs %d\n", n_head1, n_head2);
        compatible = false;
    }

    // Free models
    llama_model_free(model1);
    llama_model_free(model2);

    if (!compatible) {
        LOG_ERR("\n");
        LOG_ERR("CRITICAL: Models are NOT compatible for disaggregation!\n");
        LOG_ERR("          Both tiers must use identical model architectures\n");
        LOG_ERR("          (same layers, dimensions, heads, etc.)\n");
        return false;
    }

    LOG_INF("Models are compatible (n_layer=%d, n_embd=%d, n_head=%d)\n",
            n_layer1, n_embd1, n_head1);
    return true;
}

// Structure to hold disaggregated contexts
struct disagg_context {
    llama_model * model_prefill = nullptr;
    llama_model * model_decode = nullptr;
    llama_context * ctx_prefill = nullptr;
    llama_context * ctx_decode = nullptr;
    ggml_backend_t backend_prefill = nullptr;
    ggml_backend_t backend_decode = nullptr;
    ggml_backend_dev_t dev_prefill = nullptr;
    ggml_backend_dev_t dev_decode = nullptr;
    bool is_disaggregated = false;
    bool use_network_kv_transfer = false;
    std::string rpc_endpoint;

    // Device arrays for model loading (null-terminated)
    std::vector<ggml_backend_dev_t> prefill_devices;
    std::vector<ggml_backend_dev_t> decode_devices;

    // KV transfer buffer
    std::vector<uint8_t> kv_buffer;
    llama_kv_transfer_stats kv_stats;

    // Cumulative checksum for streaming mode
    uint32_t cumulative_checksum = 0;
};

// Network KV transfer function using RPC
static bool transfer_kv_via_rpc(
    disagg_context & disagg,
    llama_context * ctx_src,
    llama_seq_id seq_id,
    bool use_streaming,
    int32_t stream_cadence,
    int32_t n_layers) {

    const std::string & endpoint = disagg.rpc_endpoint;

    auto t_start = std::chrono::high_resolution_clock::now();

    if (use_streaming && stream_cadence > 0 && n_layers > 0) {
        // Layer streaming mode: send KV in chunks
        int32_t layer = 0;
        size_t total_sent = 0;

        // Reset cumulative checksum for this transfer
        disagg.cumulative_checksum = 0;

        // Calculate total size for all layers
        size_t total_kv_size = 0;
        for (int32_t l = 0; l < n_layers; l += stream_cadence) {
            int32_t layer_end = std::min(l + stream_cadence, n_layers);
            size_t chunk_size = llama_kv_stream_get_size(ctx_src, seq_id, l, layer_end, LLAMA_KV_COMPRESSION_NONE);
            total_kv_size += chunk_size;
        }

        if (total_kv_size == 0) {
            LOG_ERR("Failed to calculate total KV size for seq %d\n", seq_id);
            return false;
        }

        // Initialize transfer on remote
        if (!ggml_backend_rpc_kv_transfer_init(endpoint.c_str(), seq_id, total_kv_size, LLAMA_KV_COMPRESSION_NONE)) {
            LOG_ERR("Failed to initialize RPC KV transfer for seq %d\n", seq_id);
            return false;
        }

        while (layer < n_layers) {
            int32_t layer_end = std::min(layer + stream_cadence, n_layers);

            // Get required buffer size for this layer range
            size_t required_size = llama_kv_stream_get_size(
                ctx_src, seq_id, layer, layer_end, LLAMA_KV_COMPRESSION_NONE);

            if (required_size == 0) {
                LOG_WRN("Empty KV chunk for layers [%d, %d), seq %d - skipping\n", layer, layer_end, seq_id);
                layer = layer_end;
                continue;
            }

            // Resize buffer if needed
            if (disagg.kv_buffer.size() < required_size) {
                disagg.kv_buffer.resize(required_size);
            }

            // Serialize layer range
            size_t chunk_size = llama_kv_stream_serialize(
                ctx_src, seq_id, layer, layer_end,
                LLAMA_KV_COMPRESSION_NONE,
                disagg.kv_buffer.data(), disagg.kv_buffer.size());

            if (chunk_size == 0) {
                LOG_ERR("Failed to serialize KV layers [%d, %d) for seq %d\n", layer, layer_end, seq_id);
                return false;
            }

            // Send chunk over RPC
            if (!ggml_backend_rpc_kv_transfer_send(
                    endpoint.c_str(), seq_id, total_sent,
                    disagg.kv_buffer.data(), chunk_size)) {
                LOG_ERR("Failed to send KV chunk for seq %d, layers [%d, %d)\n", seq_id, layer, layer_end);
                return false;
            }

            // Accumulate checksum for all chunks (XOR of individual checksums)
            disagg.cumulative_checksum ^= llama_kv_crc32(disagg.kv_buffer.data(), chunk_size);

            total_sent += chunk_size;
            disagg.kv_stats.frame_count++;  // Count each chunk as a frame
            layer = layer_end;
        }

        // Commit the transfer with cumulative checksum (covers all chunks)
        if (!ggml_backend_rpc_kv_transfer_commit(endpoint.c_str(), seq_id, disagg.cumulative_checksum)) {
            LOG_ERR("Failed to commit RPC KV transfer for seq %d\n", seq_id);
            disagg.kv_stats.checksum_failures++;
            return false;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // Update stats: add bytes and time, but frame_count was already incremented per chunk
        disagg.kv_stats.bytes_sent += total_sent;
        disagg.kv_stats.transfer_time_ms += time_ms;
        if (disagg.kv_stats.transfer_time_ms > 0.0) {
            disagg.kv_stats.bandwidth_mbps = (disagg.kv_stats.bytes_sent * 8.0 / 1000000.0) / (disagg.kv_stats.transfer_time_ms / 1000.0);
        }

    } else {
        // Full KV transfer mode (no streaming)
        size_t kv_size = llama_kv_frame_serialize(ctx_src, seq_id, LLAMA_KV_COMPRESSION_NONE, nullptr, 0);
        if (kv_size == 0) {
            LOG_ERR("Failed to get KV frame size for seq %d\n", seq_id);
            return false;
        }

        // Resize buffer if needed
        if (disagg.kv_buffer.size() < kv_size) {
            disagg.kv_buffer.resize(kv_size);
        }

        // Serialize complete frame
        size_t actual_size = llama_kv_frame_serialize(
            ctx_src, seq_id, LLAMA_KV_COMPRESSION_NONE,
            disagg.kv_buffer.data(), disagg.kv_buffer.size());

        if (actual_size == 0) {
            LOG_ERR("Failed to serialize KV cache for seq %d\n", seq_id);
            return false;
        }

        // Initialize transfer
        if (!ggml_backend_rpc_kv_transfer_init(endpoint.c_str(), seq_id, actual_size, LLAMA_KV_COMPRESSION_NONE)) {
            LOG_ERR("Failed to initialize RPC KV transfer for seq %d\n", seq_id);
            return false;
        }

        // Send full KV frame
        if (!ggml_backend_rpc_kv_transfer_send(endpoint.c_str(), seq_id, 0, disagg.kv_buffer.data(), actual_size)) {
            LOG_ERR("Failed to send KV frame for seq %d\n", seq_id);
            return false;
        }

        // Commit with checksum (calculate on payload, excluding header for consistency)
        // Note: For full frame mode, the checksum is on the serialized data after the header
        uint32_t checksum = llama_kv_crc32(disagg.kv_buffer.data() + sizeof(llama_kv_frame_header),
                                            actual_size - sizeof(llama_kv_frame_header));
        if (!ggml_backend_rpc_kv_transfer_commit(endpoint.c_str(), seq_id, checksum)) {
            LOG_ERR("Failed to commit RPC KV transfer for seq %d\n", seq_id);
            disagg.kv_stats.checksum_failures++;
            return false;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        llama_kv_transfer_stats_update(&disagg.kv_stats, actual_size, time_ms, true);
    }

    return true;
}

// In-memory KV transfer (fallback when RPC transfer is not available or for local disaggregation)
static bool transfer_kv_in_memory(
    disagg_context & disagg,
    llama_context * ctx_src,
    llama_context * ctx_dst,
    llama_seq_id seq_id,
    bool use_streaming,
    int32_t stream_cadence,
    int32_t n_layers) {

    auto t_start = std::chrono::high_resolution_clock::now();

    if (use_streaming && stream_cadence > 0 && n_layers > 0) {
        // Layer streaming mode
        int32_t layer = 0;
        size_t total_transferred = 0;

        while (layer < n_layers) {
            int32_t layer_end = std::min(layer + stream_cadence, n_layers);

            // Get size for this layer range
            size_t chunk_size = llama_kv_stream_get_size(ctx_src, seq_id, layer, layer_end, LLAMA_KV_COMPRESSION_NONE);
            if (chunk_size == 0) {
                LOG_WRN("Empty KV chunk for layers [%d, %d), seq %d\n", layer, layer_end, seq_id);
                layer = layer_end;
                continue;
            }

            if (disagg.kv_buffer.size() < chunk_size) {
                disagg.kv_buffer.resize(chunk_size);
            }

            // Serialize layer range
            size_t actual_size = llama_kv_stream_serialize(
                ctx_src, seq_id, layer, layer_end,
                LLAMA_KV_COMPRESSION_NONE,
                disagg.kv_buffer.data(), disagg.kv_buffer.size());

            if (actual_size == 0) {
                LOG_ERR("Failed to serialize KV layers [%d, %d) for seq %d\n", layer, layer_end, seq_id);
                return false;
            }

            // Deserialize to destination
            bool checksum_ok = false;
            size_t imported = llama_kv_stream_deserialize(
                ctx_dst, seq_id, layer, layer_end,
                disagg.kv_buffer.data(), actual_size, &checksum_ok);

            if (imported == 0) {
                LOG_ERR("Failed to import KV layers [%d, %d) for seq %d\n", layer, layer_end, seq_id);
                return false;
            }

            if (!checksum_ok) {
                LOG_WRN("Checksum mismatch for KV layers [%d, %d), seq %d\n", layer, layer_end, seq_id);
                disagg.kv_stats.checksum_failures++;
            }

            total_transferred += actual_size;
            layer = layer_end;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        llama_kv_transfer_stats_update(&disagg.kv_stats, total_transferred, time_ms, true);

    } else {
        // Full KV transfer mode
        size_t kv_size = llama_kv_frame_serialize(ctx_src, seq_id, LLAMA_KV_COMPRESSION_NONE, nullptr, 0);
        if (kv_size == 0) {
            LOG_ERR("Failed to get KV size for seq %d\n", seq_id);
            return false;
        }

        if (disagg.kv_buffer.size() < kv_size) {
            disagg.kv_buffer.resize(kv_size);
        }

        size_t actual_size = llama_kv_frame_serialize(
            ctx_src, seq_id, LLAMA_KV_COMPRESSION_NONE,
            disagg.kv_buffer.data(), disagg.kv_buffer.size());

        if (actual_size == 0) {
            LOG_ERR("Failed to serialize KV cache for seq %d\n", seq_id);
            return false;
        }

        bool checksum_ok = false;
        size_t imported = llama_kv_frame_deserialize(
            ctx_dst, seq_id, disagg.kv_buffer.data(), actual_size, &checksum_ok);

        if (imported == 0) {
            LOG_ERR("Failed to import KV cache for seq %d\n", seq_id);
            return false;
        }

        if (!checksum_ok) {
            LOG_WRN("Checksum mismatch for seq %d\n", seq_id);
            disagg.kv_stats.checksum_failures++;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        llama_kv_transfer_stats_update(&disagg.kv_stats, actual_size, time_ms, true);
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    common_init();

    int is_pp_shared   = params.is_pp_shared;
    int is_tg_separate = params.is_tg_separate;

    std::vector<int> n_pp = params.n_pp;
    std::vector<int> n_tg = params.n_tg;
    std::vector<int> n_pl = params.n_pl;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // Display topology if requested
    if (params.topology) {
        llama_topology_info topology;
        llama_topology_init_simple(&topology, params.prefill_devices, params.decode_devices);
        llama_topology_print(&topology);
        llama_topology_free(&topology);
    }

    // Check if disaggregated mode
    // Disaggregation is enabled if:
    // 1. --disagg flag is set (explicit), OR
    // 2. Both --prefill-devices and --decode-devices are specified (implicit)
    disagg_context disagg;
    disagg.is_disaggregated = params.disagg || (!params.prefill_devices.empty() && !params.decode_devices.empty());

    if (disagg.is_disaggregated) {
        llama_kv_transfer_stats_init(&disagg.kv_stats);
    }

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;

    if (!disagg.is_disaggregated) {
        // Monolithic mode - single model and context
        llama_model_params model_params = common_model_params_to_llama(params);
        model = llama_model_load_from_file(params.model.path.c_str(), model_params);

        if (model == NULL) {
            fprintf(stderr , "%s: error: unable to load model\n" , __func__);
            return 1;
        }

        llama_context_params ctx_params = common_context_params_to_llama(params);
        ctx_params.n_seq_max = n_pl.empty() ? 1 : *std::max_element(n_pl.begin(), n_pl.end());

        ctx = llama_init_from_model(model, ctx_params);

        if (ctx == NULL) {
            fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
            return 1;
        }
    } else {
        // Disaggregated mode - separate models and contexts for prefill and decode
        LOG_INF("========================================\n");
        LOG_INF("Disaggregated Mode Initialization\n");
        LOG_INF("========================================\n");
        LOG_INF("Prefill device: %s\n", params.prefill_devices[0].c_str());
        LOG_INF("Decode device:  %s\n", params.decode_devices[0].c_str());
        LOG_INF("\n");

        // Parse and store RPC endpoint
        std::string prefill_device = params.prefill_devices[0];
        bool prefill_is_rpc = (prefill_device.find("RPC") != std::string::npos);

        if (prefill_is_rpc) {
            disagg.rpc_endpoint = parse_rpc_endpoint(prefill_device);
            disagg.use_network_kv_transfer = true;
            LOG_INF("RPC endpoint: %s\n", disagg.rpc_endpoint.c_str());
        }

        // Get device-specific model paths
        std::string prefill_model_path = get_model_path_for_device(params.prefill_devices[0], params);
        std::string decode_model_path = get_model_path_for_device(params.decode_devices[0], params);

        LOG_INF("Model paths:\n");
        LOG_INF("  Prefill: %s\n", prefill_model_path.c_str());
        LOG_INF("  Decode:  %s\n", decode_model_path.c_str());
        LOG_INF("\n");

        //===========================================
        // PHASE 1 & 2: Device Selection & Registration
        //===========================================
        LOG_INF("Registering devices...\n");

        // Register RPC device if prefill uses RPC
        if (prefill_is_rpc) {
            LOG_INF("Registering RPC server: %s\n", disagg.rpc_endpoint.c_str());
            disagg.dev_prefill = register_rpc_device(disagg.rpc_endpoint);
            if (!disagg.dev_prefill) {
                LOG_ERR("ERROR: Failed to register RPC device at %s\n", disagg.rpc_endpoint.c_str());
                return 1;
            }
            LOG_INF("RPC device registered: %s\n", ggml_backend_dev_name(disagg.dev_prefill));
        } else {
            // Local device for prefill
            disagg.dev_prefill = get_local_device(prefill_device);
            if (!disagg.dev_prefill) {
                LOG_ERR("ERROR: Failed to find local device: %s\n", prefill_device.c_str());
                return 1;
            }
            LOG_INF("Prefill device: %s\n", ggml_backend_dev_name(disagg.dev_prefill));
        }

        // Get local device for decode
        disagg.dev_decode = get_local_device(params.decode_devices[0]);
        if (!disagg.dev_decode) {
            LOG_ERR("ERROR: Failed to find decode device: %s\n", params.decode_devices[0].c_str());
            return 1;
        }
        LOG_INF("Decode device: %s\n", ggml_backend_dev_name(disagg.dev_decode));

        // Build device arrays (null-terminated)
        disagg.prefill_devices = { disagg.dev_prefill, nullptr };
        disagg.decode_devices = { disagg.dev_decode, nullptr };

        LOG_INF("\n");

        //===========================================
        // Load prefill model with ONLY the prefill device
        //===========================================
        LOG_INF("Loading prefill model on %s...\n", ggml_backend_dev_name(disagg.dev_prefill));

        llama_model_params model_params_prefill = common_model_params_to_llama(params);
        model_params_prefill.devices = disagg.prefill_devices.data();  // Force device selection

        disagg.model_prefill = llama_model_load_from_file(prefill_model_path.c_str(), model_params_prefill);
        if (!disagg.model_prefill) {
            fprintf(stderr, "ERROR: Failed to load model for prefill from: %s\n", prefill_model_path.c_str());
            return 1;
        }
        LOG_INF("Model loaded on prefill tier\n");

        // Create context for prefill
        llama_context_params ctx_params_prefill = common_context_params_to_llama(params);
        ctx_params_prefill.n_seq_max = n_pl.empty() ? 1 : *std::max_element(n_pl.begin(), n_pl.end());

        disagg.ctx_prefill = llama_init_from_model(disagg.model_prefill, ctx_params_prefill);
        if (!disagg.ctx_prefill) {
            fprintf(stderr, "ERROR: Failed to create prefill context\n");
            return 1;
        }
        LOG_INF("Prefill context created\n\n");

        //===========================================
        // Load decode model with ONLY the decode device
        //===========================================
        LOG_INF("Loading decode model on %s...\n", ggml_backend_dev_name(disagg.dev_decode));

        llama_model_params model_params_decode = common_model_params_to_llama(params);
        model_params_decode.devices = disagg.decode_devices.data();  // Force device selection

        disagg.model_decode = llama_model_load_from_file(decode_model_path.c_str(), model_params_decode);
        if (!disagg.model_decode) {
            fprintf(stderr, "ERROR: Failed to load model for decode from: %s\n", decode_model_path.c_str());
            return 1;
        }
        LOG_INF("Model loaded on decode tier\n");

        // Create context for decode
        llama_context_params ctx_params_decode = common_context_params_to_llama(params);
        ctx_params_decode.n_seq_max = n_pl.empty() ? 1 : *std::max_element(n_pl.begin(), n_pl.end());

        disagg.ctx_decode = llama_init_from_model(disagg.model_decode, ctx_params_decode);
        if (!disagg.ctx_decode) {
            fprintf(stderr, "ERROR: Failed to create decode context\n");
            return 1;
        }
        LOG_INF("Decode context created\n\n");

        //===========================================
        // Configuration Summary
        //===========================================
        LOG_INF("========================================\n");
        LOG_INF("Disaggregated Configuration\n");
        LOG_INF("========================================\n");
        LOG_INF("Prefill:\n");
        LOG_INF("  Device: %s\n", ggml_backend_dev_name(disagg.dev_prefill));
        LOG_INF("  Model:  %s\n", prefill_model_path.c_str());
        LOG_INF("Decode:\n");
        LOG_INF("  Device: %s\n", ggml_backend_dev_name(disagg.dev_decode));
        LOG_INF("  Model:  %s\n", decode_model_path.c_str());
        LOG_INF("KV Transfer:\n");
        if (disagg.use_network_kv_transfer) {
            LOG_INF("  Mode:     Network (RPC)\n");
            LOG_INF("  Endpoint: %s\n", disagg.rpc_endpoint.c_str());
        } else {
            LOG_INF("  Mode:     In-memory\n");
        }
        if (params.kv_stream) {
            LOG_INF("  Streaming: Enabled (cadence=%d)\n", params.kv_stream_every);
        } else {
            LOG_INF("  Streaming: Disabled\n");
        }
        LOG_INF("========================================\n\n");

        // Use prefill context as the primary one for warmup
        ctx = disagg.ctx_prefill;
        model = disagg.model_prefill;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t       n_layer = llama_model_n_layer(model);

    const auto get_token_rand = [n_vocab]() -> llama_token {
        return std::rand() % n_vocab;
    };

    auto * mem = llama_get_memory(ctx);

    const int32_t n_kv_max = llama_n_ctx(ctx);

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [](llama_context * ctx, llama_batch & batch, int32_t n_batch, bool synchronize) {
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_ERR("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
                return false;
            }

            if (synchronize) {
                llama_synchronize(ctx);
            }
        }

        return true;
    };

    llama_context_params ctx_params = common_context_params_to_llama(params);

    // warm up
    {
        for (int i = 0; i < 16; ++i) {
            common_batch_add(batch, get_token_rand(), i, { 0 }, false);
        }

        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    if (!params.batched_bench_output_jsonl) {
        LOG("\n");
        if (disagg.is_disaggregated) {
            LOG("%s: DISAGGREGATED MODE - prefill=%s, decode=%s\n", __func__,
                params.prefill_devices[0].c_str(), params.decode_devices[0].c_str());
            LOG("%s: NOTE: T_tot = T_PP + T_TG + T_KV (total time includes KV transfer)\n", __func__);
        }
        LOG("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, is_pp_shared = %d, is_tg_separate = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), is_pp_shared, is_tg_separate, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
        LOG("\n");
        if (disagg.is_disaggregated) {
            LOG("|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s |\n",
                "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T_tot s", "S t/s", "KV_MB", "T_KV s");
            LOG("|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n",
                "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------", "--------", "--------");
        } else {
            LOG("|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s |\n",
                "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s");
            LOG("|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n",
                "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------");
        }
    }

    for (        int i_pp = 0; i_pp < (int) n_pp.size(); ++i_pp) {
        for (    int i_tg = 0; i_tg < (int) n_tg.size(); ++i_tg) {
            for (int i_pl = 0; i_pl < (int) n_pl.size(); ++i_pl) {
                const int pp = n_pp[i_pp];
                const int tg = n_tg[i_tg];
                const int pl = n_pl[i_pl];

                const int n_ctx_req = is_pp_shared ? (params.kv_unified ? pp : pl*pp) + pl*tg : pl*(pp + tg);

                if (n_ctx_req > n_kv_max) {
                    continue;
                }

                // Reset KV transfer stats for this iteration
                if (disagg.is_disaggregated) {
                    llama_kv_transfer_stats_init(&disagg.kv_stats);
                }

                common_batch_clear(batch);

                for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                    for (int i = 0; i < pp; ++i) {
                        common_batch_add(batch, get_token_rand(), i, { j }, i == pp - 1);
                    }
                }

                // Clear memory on appropriate context
                if (disagg.is_disaggregated) {
                    llama_memory_clear(llama_get_memory(disagg.ctx_prefill), false);
                    llama_memory_clear(llama_get_memory(disagg.ctx_decode), false);
                } else {
                    llama_memory_clear(mem, false);
                }

                const auto t_pp_start = ggml_time_us();

                // PREFILL PHASE
                if (disagg.is_disaggregated) {
                    // Run prefill on prefill device (RPC or local)
                    if (!decode_helper(disagg.ctx_prefill, batch, ctx_params.n_batch, false)) {
                        LOG_ERR("%s: prefill llama_decode() failed\n", __func__);
                        return 1;
                    }
                    llama_synchronize(disagg.ctx_prefill);
                } else {
                    // Monolithic mode
                    if (!decode_helper(ctx, batch, ctx_params.n_batch, false)) {
                        LOG_ERR("%s: llama_decode() failed\n", __func__);
                        return 1;
                    }
                    llama_synchronize(ctx);
                }

                const auto t_pp_end = ggml_time_us();

                // KV CACHE TRANSFER
                double kv_transfer_time_s = 0.0;
                double kv_transfer_mb = 0.0;

                if (disagg.is_disaggregated) {
                    // Transfer KV cache from prefill to decode context
                    for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                        bool success;

                        if (disagg.use_network_kv_transfer) {
                            // Phase 1: Network KV transfer via RPC
                            success = transfer_kv_via_rpc(
                                disagg, disagg.ctx_prefill, j,
                                params.kv_stream, params.kv_stream_every, n_layer);
                        } else {
                            // Fallback: In-memory transfer (for local disaggregation testing)
                            success = transfer_kv_in_memory(
                                disagg, disagg.ctx_prefill, disagg.ctx_decode, j,
                                params.kv_stream, params.kv_stream_every, n_layer);
                        }

                        if (!success) {
                            LOG_ERR("KV transfer failed for seq %d\n", j);
                            return 1;
                        }
                    }

                    kv_transfer_time_s = disagg.kv_stats.transfer_time_ms / 1000.0;
                    kv_transfer_mb = disagg.kv_stats.bytes_sent / (1024.0 * 1024.0);
                }

                // SEQUENCE COPY (if pp_shared)
                if (is_pp_shared) {
                    llama_memory_t mem_target = disagg.is_disaggregated ?
                                                llama_get_memory(disagg.ctx_decode) : mem;

                    for (int32_t i = 1; i < pl; ++i) {
                        llama_memory_seq_cp(mem_target, 0, i, -1, -1);
                    }

                    if (!params.kv_unified) {
                        // run one dummy token to apply the memory copy
                        llama_context * ctx_target = disagg.is_disaggregated ? disagg.ctx_decode : ctx;

                        common_batch_clear(batch);
                        common_batch_add(batch, get_token_rand(), pp + 0, { 0 }, true);
                        if (!decode_helper(ctx_target, batch, ctx_params.n_batch, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            return 1;
                        }
                        llama_memory_seq_rm(mem_target, 0, pp, -1);
                    }
                }

                const auto t_tg_start = ggml_time_us();

                // DECODE PHASE
                llama_context * ctx_decode_target = disagg.is_disaggregated ? disagg.ctx_decode : ctx;

                if (is_tg_separate) {
                    // decode pattern: 0 0 0 ... 1 1 1 ... 2 2 2 ... 3 3 3 ...
                    for (int j = 0; j < pl; ++j) {
                        for (int i = 0; i < tg; ++i) {
                            common_batch_clear(batch);
                            common_batch_add(batch, get_token_rand(), pp + i, { j }, true);

                            if (!decode_helper(ctx_decode_target, batch, ctx_params.n_batch, true)) {
                                LOG_ERR("%s: llama_decode() failed\n", __func__);
                                return 1;
                            }
                        }
                    }
                } else {
                    // decode pattern: 0123 0123 0123 ...
                    for (int i = 0; i < tg; ++i) {
                        common_batch_clear(batch);

                        for (int j = 0; j < pl; ++j) {
                            common_batch_add(batch, get_token_rand(), pp + i, { j }, true);
                        }

                        if (!decode_helper(ctx_decode_target, batch, ctx_params.n_batch, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            return 1;
                        }
                    }
                }

                const auto t_tg_end = ggml_time_us();

                const int32_t n_kv = n_ctx_req;

                const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
                const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;
                // Total time includes KV transfer time for disaggregated mode
                const float t_kv = disagg.is_disaggregated ? (float)kv_transfer_time_s : 0.0f;
                const float t    = t_pp + t_tg + t_kv;

                const float speed_pp = is_pp_shared ? pp / t_pp : pl*pp / t_pp;
                const float speed_tg = pl*tg / t_tg;
                // Speed calculation uses total time including KV transfer
                const float speed    = ((is_pp_shared ? pp : pl*pp) + pl*tg) / t;

                if(params.batched_bench_output_jsonl) {
                    if (disagg.is_disaggregated) {
                        LOG(
                            "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"is_pp_shared\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                            "\"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"t\": %f, \"speed\": %f, "
                            "\"kv_transfer_mb\": %f, \"kv_transfer_s\": %f, \"kv_frames\": %d}\n",
                            n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), params.is_pp_shared, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                            pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed,
                            kv_transfer_mb, kv_transfer_time_s, disagg.kv_stats.frame_count
                        );
                    } else {
                        LOG(
                            "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"is_pp_shared\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                            "\"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"t\": %f, \"speed\": %f}\n",
                            n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), params.is_pp_shared, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                            pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed
                        );
                    }
                } else {
                    if (disagg.is_disaggregated) {
                        LOG("|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f | %8.2f | %8.3f |\n",
                            pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed, kv_transfer_mb, kv_transfer_time_s);
                    } else {
                        LOG("|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n",
                            pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed);
                    }
                }
            }
        }
    }

    LOG("\n");

    if (disagg.is_disaggregated) {
        LOG("========================================\n");
        LOG("KV Cache Transfer Statistics\n");
        LOG("========================================\n");
        LOG("Transfer mode:        %s\n", disagg.use_network_kv_transfer ? "NETWORK (RPC)" : "IN-MEMORY");
        LOG("Layer streaming:      %s\n", params.kv_stream ? "ENABLED" : "DISABLED");
        if (params.kv_stream) {
            LOG("Stream every N layers: %d\n", params.kv_stream_every);
        }
        LOG("Total KV transferred: %.2f MB\n", disagg.kv_stats.bytes_sent / (1024.0 * 1024.0));
        LOG("Total transfer time:  %.3f s (included in T_tot)\n", disagg.kv_stats.transfer_time_ms / 1000.0);
        LOG("Average bandwidth:    %.2f MB/s\n", (disagg.kv_stats.bytes_sent / (1024.0 * 1024.0)) / (disagg.kv_stats.transfer_time_ms / 1000.0));
        LOG("Total frames:         %d\n", disagg.kv_stats.frame_count);
        LOG("Checksum failures:    %d\n", disagg.kv_stats.checksum_failures);
        LOG("----------------------------------------\n");
        LOG("Prefill device:       %s\n", disagg.dev_prefill ? ggml_backend_dev_name(disagg.dev_prefill) : "default");
        LOG("Decode device:        %s\n", disagg.dev_decode ? ggml_backend_dev_name(disagg.dev_decode) : "default");
        if (!disagg.rpc_endpoint.empty()) {
            LOG("RPC endpoint:         %s\n", disagg.rpc_endpoint.c_str());
        }
        LOG("========================================\n\n");

        LOG("Prefill context performance:\n");
        llama_perf_context_print(disagg.ctx_prefill);
        LOG("\n");

        LOG("Decode context performance:\n");
        llama_perf_context_print(disagg.ctx_decode);
    } else {
        llama_perf_context_print(ctx);
    }

    llama_batch_free(batch);

    if (disagg.is_disaggregated) {
        llama_free(disagg.ctx_prefill);
        llama_free(disagg.ctx_decode);
        llama_model_free(disagg.model_prefill);
        llama_model_free(disagg.model_decode);
        // Note: RPC device is managed by ggml backend registry, no explicit free needed
    } else {
        llama_free(ctx);
        llama_model_free(model);
    }

    llama_backend_free();

    LOG("\n");

    return 0;
}
