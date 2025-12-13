#pragma once

#include "llama.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

// KV cache transfer protocol for prefill-decode disaggregation
// Supports framing, compression, checksums, and streaming

// Frame magic number: "KVFR" (0x5246564B in little-endian)
#define LLAMA_KV_FRAME_MAGIC 0x5246564B

// Frame version
// Version 1: Full KV cache transfer (original)
// Version 2: Supports layer-specific streaming with layer_start/layer_end
#define LLAMA_KV_FRAME_VERSION 2

// Compression types
enum llama_kv_compression_type {
    LLAMA_KV_COMPRESSION_NONE = 0,
    LLAMA_KV_COMPRESSION_ZSTD = 1,
};

// Frame header structure (48 bytes, aligned)
struct llama_kv_frame_header {
    uint32_t magic;              // Magic number (0x5246564B)
    uint32_t version;            // Protocol version (1 or 2)
    llama_seq_id seq_id;         // Sequence ID
    uint32_t compression;        // Compression type (llama_kv_compression_type)
    uint64_t uncompressed_size;  // Original KV data size
    uint64_t compressed_size;    // Compressed size (equals uncompressed if no compression)
    uint32_t checksum;           // CRC32 checksum of payload
    uint32_t _padding;           // Padding for alignment
    int32_t layer_start;         // First layer in frame (version 2+), -1 for full KV (version 1)
    int32_t layer_end;           // Last layer + 1 in frame (version 2+), -1 for full KV (version 1)
};

static_assert(sizeof(llama_kv_frame_header) == 48, "Frame header must be 48 bytes");

// KV transfer statistics (per frame or per request)
struct llama_kv_transfer_stats {
    size_t bytes_sent;
    size_t bytes_received;
    double transfer_time_ms;
    double bandwidth_mbps;
    int32_t frame_count;
    int32_t checksum_failures;
    int32_t retries;
};

#ifdef __cplusplus
extern "C" {
#endif

// Calculate CRC32 checksum
uint32_t llama_kv_crc32(const uint8_t * data, size_t size);

// Serialize KV cache for a sequence into a frame
// Returns the size of the serialized frame (including header), or 0 on error
// If dst is NULL, returns the required size
size_t llama_kv_frame_serialize(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    enum llama_kv_compression_type compression,
    uint8_t * dst,
    size_t dst_size);

// Deserialize KV cache from a frame
// Returns the number of bytes consumed, or 0 on error
// Validates magic, version, and checksum
size_t llama_kv_frame_deserialize(
    struct llama_context * ctx,
    llama_seq_id dest_seq_id,
    const uint8_t * src,
    size_t src_size,
    bool * out_checksum_ok);

// Streaming: get size needed for KV cache export at current layer
// Used with --kv-stream to emit incremental frames
size_t llama_kv_stream_get_size(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    int32_t layer_start,
    int32_t layer_end,
    enum llama_kv_compression_type compression);

// Streaming: serialize KV cache for a range of layers
// layer_start and layer_end define the range [start, end)
size_t llama_kv_stream_serialize(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    int32_t layer_start,
    int32_t layer_end,
    enum llama_kv_compression_type compression,
    uint8_t * dst,
    size_t dst_size);

// Streaming: deserialize partial KV cache for a range of layers
size_t llama_kv_stream_deserialize(
    struct llama_context * ctx,
    llama_seq_id dest_seq_id,
    int32_t layer_start,
    int32_t layer_end,
    const uint8_t * src,
    size_t src_size,
    bool * out_checksum_ok);

// Initialize transfer statistics
void llama_kv_transfer_stats_init(struct llama_kv_transfer_stats * stats);

// Update transfer statistics after sending/receiving a frame
void llama_kv_transfer_stats_update(
    struct llama_kv_transfer_stats * stats,
    size_t bytes_transferred,
    double time_ms,
    bool is_send);

// Print transfer statistics (for logging)
void llama_kv_transfer_stats_print(const struct llama_kv_transfer_stats * stats, bool is_send);

#ifdef __cplusplus
}
#endif
