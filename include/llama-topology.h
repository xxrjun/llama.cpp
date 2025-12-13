#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <map>
#include <cstdint>

// Forward declaration
struct common_params;

// Topology display for prefill-decode disaggregation
// Shows local and remote devices, their roles, and network bandwidth

enum llama_device_role {
    LLAMA_DEVICE_ROLE_LOCAL   = 0,  // Default local device
    LLAMA_DEVICE_ROLE_PREFILL = 1,  // Prefill tier device
    LLAMA_DEVICE_ROLE_DECODE  = 2,  // Decode tier device
};

struct llama_device_info {
    std::string name;                     // Device name (e.g., "CUDA0", "Metal", "RPC@ip:port")
    std::string type;                     // Device type (e.g., "CUDA", "Metal", "CPU", "RPC")
    llama_device_role role;               // Device role
    bool is_remote;                       // Is this a remote RPC device?
    std::string endpoint;                 // RPC endpoint if remote (e.g., "192.168.1.10:50052")
    size_t memory_bytes;                  // Available memory in bytes
    ggml_backend_dev_t backend_dev;       // Backend device handle (if available)
};

struct llama_topology_info {
    std::vector<llama_device_info> devices;  // All devices
    bool has_prefill_tier;                   // Has dedicated prefill tier
    bool has_decode_tier;                    // Has dedicated decode tier
};

#ifdef __cplusplus
extern "C" {
#endif

// Gather topology information from parameters (simplified version)
void llama_topology_init_simple(
    struct llama_topology_info * topology,
    const std::vector<std::string> & prefill_devices,
    const std::vector<std::string> & decode_devices);

// Print topology to stdout
void llama_topology_print(const struct llama_topology_info * topology);

// Generate JSON representation of topology
// Returns JSON string (caller must free)
char * llama_topology_to_json(const struct llama_topology_info * topology);

// Free topology info
void llama_topology_free(struct llama_topology_info * topology);

#ifdef __cplusplus
}
#endif
