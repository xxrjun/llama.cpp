#include "llama-topology.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// Forward declaration
static std::string parse_device_string(const std::string & device_str, std::string & endpoint);

void llama_topology_init_simple(
    struct llama_topology_info * topology,
    const std::vector<std::string> & prefill_devices,
    const std::vector<std::string> & decode_devices) {

    if (!topology) {
        return;
    }

    topology->devices.clear();
    topology->has_prefill_tier = !prefill_devices.empty();
    topology->has_decode_tier = !decode_devices.empty();

    // Enumerate all available backend devices
    size_t num_backends = ggml_backend_dev_count();

    // Process prefill devices
    for (const auto & device_str : prefill_devices) {
        llama_device_info dev_info;
        dev_info.role = LLAMA_DEVICE_ROLE_PREFILL;

        // Parse device string (e.g., "RPC@192.168.1.10:50052", "CUDA0", "Metal")
        dev_info.type = parse_device_string(device_str, dev_info.endpoint);
        dev_info.name = device_str;
        dev_info.is_remote = (dev_info.type == "RPC");
        dev_info.memory_bytes = 0;  // Unknown for remote devices
        dev_info.backend_dev = nullptr;

        // Try to match with actual backend device or connect to RPC
        if (!dev_info.is_remote) {
            // Local device - enumerate backends
            for (size_t i = 0; i < num_backends; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (dev) {
                    std::string backend_name = ggml_backend_dev_name(dev);
                    if (backend_name.find(dev_info.type) != std::string::npos) {
                        dev_info.backend_dev = dev;
                        size_t free_mem = 0, total_mem = 0;
                        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
                        dev_info.memory_bytes = total_mem;
                        break;
                    }
                }
            }
        } else {
            // Remote RPC device - try to connect and query
            fprintf(stderr, "Querying RPC device at %s...\n", dev_info.endpoint.c_str());
            ggml_backend_t rpc_backend = ggml_backend_rpc_init(dev_info.endpoint.c_str(), 0);
            if (rpc_backend) {
                ggml_backend_dev_t dev = ggml_backend_get_device(rpc_backend);
                if (dev) {
                    dev_info.name = ggml_backend_dev_name(dev);
                    dev_info.type = "CUDA"; // Most RPC servers are CUDA
                    size_t free_mem = 0, total_mem = 0;
                    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
                    dev_info.memory_bytes = total_mem;
                    fprintf(stderr, "  Device: %s, Memory: %.2f GiB\n",
                            dev_info.name.c_str(),
                            total_mem / (1024.0 * 1024.0 * 1024.0));
                }
                ggml_backend_free(rpc_backend);
            } else {
                fprintf(stderr, "  Warning: Could not connect to RPC server\n");
            }
        }

        topology->devices.push_back(dev_info);
    }

    // Process decode devices
    for (const auto & device_str : decode_devices) {
        llama_device_info dev_info;
        dev_info.role = LLAMA_DEVICE_ROLE_DECODE;

        dev_info.type = parse_device_string(device_str, dev_info.endpoint);
        dev_info.name = device_str;
        dev_info.is_remote = (dev_info.type == "RPC");
        dev_info.memory_bytes = 0;
        dev_info.backend_dev = nullptr;

        // Try to match with actual backend device or connect to RPC
        if (!dev_info.is_remote) {
            // Local device - enumerate backends
            for (size_t i = 0; i < num_backends; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (dev) {
                    std::string backend_name = ggml_backend_dev_name(dev);
                    if (backend_name.find(dev_info.type) != std::string::npos) {
                        dev_info.backend_dev = dev;
                        size_t free_mem = 0, total_mem = 0;
                        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
                        dev_info.memory_bytes = total_mem;
                        break;
                    }
                }
            }
        } else {
            // Remote RPC device - try to connect and query
            fprintf(stderr, "Querying RPC device at %s...\n", dev_info.endpoint.c_str());
            ggml_backend_t rpc_backend = ggml_backend_rpc_init(dev_info.endpoint.c_str(), 0);
            if (rpc_backend) {
                ggml_backend_dev_t dev = ggml_backend_get_device(rpc_backend);
                if (dev) {
                    dev_info.name = ggml_backend_dev_name(dev);
                    dev_info.type = "CUDA"; // Most RPC servers are CUDA
                    size_t free_mem = 0, total_mem = 0;
                    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
                    dev_info.memory_bytes = total_mem;
                    fprintf(stderr, "  Device: %s, Memory: %.2f GiB\n",
                            dev_info.name.c_str(),
                            total_mem / (1024.0 * 1024.0 * 1024.0));
                }
                ggml_backend_free(rpc_backend);
            } else {
                fprintf(stderr, "  Warning: Could not connect to RPC server\n");
            }
        }

        topology->devices.push_back(dev_info);
    }

    // If no specific devices configured, enumerate all local devices
    if (topology->devices.empty()) {
        for (size_t i = 0; i < num_backends; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev) {
                llama_device_info dev_info;
                dev_info.name = ggml_backend_dev_name(dev);
                dev_info.type = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ? "GPU" :
                               ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ? "CPU" : "Other";
                dev_info.role = LLAMA_DEVICE_ROLE_LOCAL;
                dev_info.is_remote = false;
                dev_info.endpoint = "";
                size_t free_mem = 0, total_mem = 0;
                ggml_backend_dev_memory(dev, &free_mem, &total_mem);
                dev_info.memory_bytes = total_mem;
                dev_info.backend_dev = dev;

                topology->devices.push_back(dev_info);
            }
        }
    }
}

void llama_topology_print(const struct llama_topology_info * topology) {
    if (!topology) {
        return;
    }

    printf("\n");
    printf("========================================\n");
    printf("Topology Information\n");
    printf("========================================\n");

    printf("\nDevices:\n");
    printf("%-30s %-10s %-10s %-12s %s\n", "Name", "Type", "Role", "Memory", "Endpoint");
    printf("%-30s %-10s %-10s %-12s %s\n", "----", "----", "----", "------", "--------");

    for (const auto & dev : topology->devices) {
        const char * role_str = "local";
        if (dev.role == LLAMA_DEVICE_ROLE_PREFILL) {
            role_str = "prefill";
        } else if (dev.role == LLAMA_DEVICE_ROLE_DECODE) {
            role_str = "decode";
        }

        char memory_str[32] = "N/A";
        if (dev.memory_bytes > 0) {
            snprintf(memory_str, sizeof(memory_str), "%.2f GB",
                    dev.memory_bytes / (1024.0 * 1024.0 * 1024.0));
        }

        printf("%-30s %-10s %-10s %-12s %s\n",
               dev.name.c_str(),
               dev.type.c_str(),
               role_str,
               memory_str,
               dev.is_remote ? dev.endpoint.c_str() : "-");
    }

    printf("\n");

    if (topology->has_prefill_tier && topology->has_decode_tier) {
        printf("Mode: Disaggregated (prefill + decode)\n");
    } else if (topology->has_prefill_tier) {
        printf("Mode: Prefill-only\n");
    } else if (topology->has_decode_tier) {
        printf("Mode: Decode-only\n");
    } else {
        printf("Mode: Monolithic (no disaggregation)\n");
    }

    printf("========================================\n\n");
}

char * llama_topology_to_json(const struct llama_topology_info * topology) {
    if (!topology) {
        return nullptr;
    }

    // Build JSON string manually (simple implementation without json library)
    std::string json = "{\n";
    json += "  \"has_prefill_tier\": " + std::string(topology->has_prefill_tier ? "true" : "false") + ",\n";
    json += "  \"has_decode_tier\": " + std::string(topology->has_decode_tier ? "true" : "false") + ",\n";
    json += "  \"devices\": [\n";

    for (size_t i = 0; i < topology->devices.size(); i++) {
        const auto & dev = topology->devices[i];

        json += "    {\n";
        json += "      \"name\": \"" + dev.name + "\",\n";
        json += "      \"type\": \"" + dev.type + "\",\n";

        std::string role_str = "local";
        if (dev.role == LLAMA_DEVICE_ROLE_PREFILL) {
            role_str = "prefill";
        } else if (dev.role == LLAMA_DEVICE_ROLE_DECODE) {
            role_str = "decode";
        }
        json += "      \"role\": \"" + role_str + "\",\n";

        json += "      \"is_remote\": " + std::string(dev.is_remote ? "true" : "false") + ",\n";
        json += "      \"endpoint\": \"" + dev.endpoint + "\",\n";
        json += "      \"memory_bytes\": " + std::to_string(dev.memory_bytes) + "\n";
        json += "    }";

        if (i < topology->devices.size() - 1) {
            json += ",";
        }
        json += "\n";
    }

    json += "  ]\n";
    json += "}\n";

    // Allocate and copy JSON string
    char * result = (char *)malloc(json.size() + 1);
    if (result) {
        strcpy(result, json.c_str());
    }

    return result;
}

void llama_topology_free(struct llama_topology_info * topology) {
    if (topology) {
        topology->devices.clear();
    }
}

// Helper: parse device string like "RPC@192.168.1.10:50052" or "CUDA0" or "Metal"
static std::string parse_device_string(const std::string & device_str, std::string & endpoint) {
    endpoint = "";

    // Check for RPC format: RPC@endpoint
    size_t at_pos = device_str.find('@');
    if (at_pos != std::string::npos) {
        std::string prefix = device_str.substr(0, at_pos);
        if (prefix == "RPC") {
            endpoint = device_str.substr(at_pos + 1);
            return "RPC";
        }
    }

    // Otherwise, extract device type (e.g., "CUDA", "Metal", "CPU")
    std::string type = device_str;

    // Remove trailing digits (e.g., "CUDA0" -> "CUDA")
    while (!type.empty() && std::isdigit(type.back())) {
        type.pop_back();
    }

    return type;
}
