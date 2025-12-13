#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    3
#define RPC_PROTO_MINOR_VERSION    6
#define RPC_PROTO_PATCH_VERSION    0
#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

// TODO: GPU Pre-loading Feature (Not Yet Implemented)
// The intended feature would allow pre-loading the model into GPU memory at server startup,
// so client connections can reuse existing GPU buffers without data transfer.
// This requires new RPC protocol commands for buffer sharing.
//
// GGML_BACKEND_API int ggml_backend_rpc_preload_model_to_gpu(const char * model_path);
// GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_rpc_get_preloaded_buffer(const char * endpoint, const char * tensor_name);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// KV cache transfer functions for prefill-decode disaggregation
GGML_BACKEND_API bool ggml_backend_rpc_kv_transfer_init(const char * endpoint, int32_t seq_id, uint64_t total_size, uint32_t compression);
GGML_BACKEND_API bool ggml_backend_rpc_kv_transfer_send(const char * endpoint, int32_t seq_id, uint64_t frame_offset, const uint8_t * data, uint64_t chunk_size);
GGML_BACKEND_API bool ggml_backend_rpc_kv_transfer_commit(const char * endpoint, int32_t seq_id, uint32_t checksum);

#ifdef  __cplusplus
}
#endif
