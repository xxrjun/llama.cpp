#include "llama-kv-transfer.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-io.h"
#include <cstring>
#include <vector>
#include <chrono>
#include <cstdio>

// CRC32 implementation (simple table-based)
static uint32_t crc32_table[256];
static bool crc32_table_initialized = false;

static void llama_kv_crc32_init() {
    if (crc32_table_initialized) {
        return;
    }
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = true;
}

uint32_t llama_kv_crc32(const uint8_t * data, size_t size) {
    llama_kv_crc32_init();
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < size; i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

// Compression helpers (stub for now - would need zstd library)
static size_t llama_kv_compress_zstd(const uint8_t * src, size_t src_size, uint8_t * dst, size_t dst_size) {
    // TODO: Implement zstd compression when ZSTD is available
    // For now, just copy (no compression)
    if (dst_size < src_size) {
        return 0;
    }
    memcpy(dst, src, src_size);
    return src_size;
}

static size_t llama_kv_decompress_zstd(const uint8_t * src, size_t src_size, uint8_t * dst, size_t dst_size) {
    // TODO: Implement zstd decompression when ZSTD is available
    // For now, just copy (no compression)
    if (dst_size < src_size) {
        return 0;
    }
    memcpy(dst, src, src_size);
    return src_size;
}

size_t llama_kv_frame_serialize(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    enum llama_kv_compression_type compression,
    uint8_t * dst,
    size_t dst_size) {

    if (!ctx) {
        return 0;
    }

    // Get the size needed for the KV state
    size_t kv_size = llama_state_seq_get_size(ctx, seq_id);
    if (kv_size == 0) {
        return 0;
    }

    // Calculate total frame size
    size_t frame_size = sizeof(llama_kv_frame_header) + kv_size;

    // If dst is NULL, just return the required size
    if (!dst) {
        return frame_size;
    }

    // Check if dst has enough space
    if (dst_size < frame_size) {
        return 0;
    }

    // Allocate temporary buffer for KV data
    std::vector<uint8_t> kv_buffer(kv_size);
    size_t actual_kv_size = llama_state_seq_get_data(ctx, kv_buffer.data(), kv_size, seq_id);
    if (actual_kv_size == 0 || actual_kv_size > kv_size) {
        return 0;
    }

    // Prepare frame header
    llama_kv_frame_header header;
    header.magic = LLAMA_KV_FRAME_MAGIC;
    header.version = LLAMA_KV_FRAME_VERSION;
    header.seq_id = seq_id;
    header.compression = compression;
    header.uncompressed_size = actual_kv_size;
    header._padding = 0;

    // Compress if requested
    uint8_t * payload_dst = dst + sizeof(llama_kv_frame_header);
    size_t payload_size = dst_size - sizeof(llama_kv_frame_header);
    size_t compressed_size = actual_kv_size;

    if (compression == LLAMA_KV_COMPRESSION_ZSTD) {
        compressed_size = llama_kv_compress_zstd(kv_buffer.data(), actual_kv_size, payload_dst, payload_size);
        if (compressed_size == 0) {
            return 0;
        }
    } else {
        // No compression
        if (payload_size < actual_kv_size) {
            return 0;
        }
        memcpy(payload_dst, kv_buffer.data(), actual_kv_size);
    }

    header.compressed_size = compressed_size;
    header.checksum = llama_kv_crc32(payload_dst, compressed_size);

    // Write header
    memcpy(dst, &header, sizeof(header));

    return sizeof(header) + compressed_size;
}

size_t llama_kv_frame_deserialize(
    struct llama_context * ctx,
    llama_seq_id dest_seq_id,
    const uint8_t * src,
    size_t src_size,
    bool * out_checksum_ok) {

    if (!ctx || !src || src_size < sizeof(llama_kv_frame_header)) {
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    // Read header
    llama_kv_frame_header header;
    memcpy(&header, src, sizeof(header));

    // Validate magic and version
    if (header.magic != LLAMA_KV_FRAME_MAGIC) {
        fprintf(stderr, "KV frame deserialize: invalid magic 0x%08X (expected 0x%08X)\n",
                header.magic, LLAMA_KV_FRAME_MAGIC);
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    if (header.version != LLAMA_KV_FRAME_VERSION) {
        fprintf(stderr, "KV frame deserialize: version mismatch %u (expected %u)\n",
                header.version, LLAMA_KV_FRAME_VERSION);
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    // Check frame size
    size_t frame_size = sizeof(header) + header.compressed_size;
    if (src_size < frame_size) {
        fprintf(stderr, "KV frame deserialize: insufficient data (%zu < %zu)\n", src_size, frame_size);
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    // Validate checksum
    const uint8_t * payload = src + sizeof(header);
    uint32_t calculated_checksum = llama_kv_crc32(payload, header.compressed_size);
    bool checksum_ok = (calculated_checksum == header.checksum);

    if (out_checksum_ok) {
        *out_checksum_ok = checksum_ok;
    }

    if (!checksum_ok) {
        fprintf(stderr, "KV frame deserialize: checksum mismatch (calculated 0x%08X, expected 0x%08X)\n",
                calculated_checksum, header.checksum);
        return 0;
    }

    // Decompress if needed
    std::vector<uint8_t> kv_buffer;
    const uint8_t * kv_data = payload;
    size_t kv_size = header.compressed_size;

    if (header.compression == LLAMA_KV_COMPRESSION_ZSTD) {
        kv_buffer.resize(header.uncompressed_size);
        size_t decompressed_size = llama_kv_decompress_zstd(payload, header.compressed_size,
                                                             kv_buffer.data(), header.uncompressed_size);
        if (decompressed_size == 0 || decompressed_size != header.uncompressed_size) {
            fprintf(stderr, "KV frame deserialize: decompression failed\n");
            return 0;
        }
        kv_data = kv_buffer.data();
        kv_size = decompressed_size;
    }

    // Import KV state
    size_t imported_size = llama_state_seq_set_data(ctx, kv_data, kv_size, dest_seq_id);
    if (imported_size == 0) {
        fprintf(stderr, "KV frame deserialize: failed to import KV state\n");
        return 0;
    }

    return frame_size;
}

// Streaming functions with layer-specific KV access
size_t llama_kv_stream_get_size(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    int32_t layer_start,
    int32_t layer_end,
    enum llama_kv_compression_type compression) {

    if (!ctx) {
        return 0;
    }

    // Get KV cache from context
    auto * mem = llama_get_memory(ctx);
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);

    if (!kv) {
        // Fallback to full KV size if cast fails
        return sizeof(llama_kv_frame_header) + llama_state_seq_get_size(ctx, seq_id);
    }

    // Calculate size for layer range for the specific sequence
    size_t layer_data_size = kv->layer_range_size_for_seq(seq_id, layer_start, layer_end);

    if (layer_data_size == 0) {
        return 0;
    }

    // Account for compression (simplified - actual compressed size may vary)
    size_t kv_size = layer_data_size;
    if (compression == LLAMA_KV_COMPRESSION_ZSTD) {
        // Estimate: assume 50% compression ratio
        kv_size = layer_data_size / 2;
    }

    return sizeof(llama_kv_frame_header) + kv_size;
}

size_t llama_kv_stream_serialize(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    int32_t layer_start,
    int32_t layer_end,
    enum llama_kv_compression_type compression,
    uint8_t * dst,
    size_t dst_size) {

    if (!ctx) {
        return 0;
    }

    // Get KV cache from context
    auto * mem = llama_get_memory(ctx);
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);

    if (!kv) {
        // Fallback to full KV serialization
        fprintf(stderr, "Warning: Layer-specific streaming not available, using full KV\n");
        return llama_kv_frame_serialize(ctx, seq_id, compression, dst, dst_size);
    }

    // Build cell ranges for the specific sequence
    llama_kv_cache::cell_ranges_t cr = kv->build_cell_ranges_for_seq(seq_id);

    // Check if any cells were found
    uint32_t cell_count = 0;
    for (const auto & range : cr.data) {
        cell_count += range.second - range.first;
    }

    if (cell_count == 0) {
        fprintf(stderr, "Warning: No cells found for seq_id %d\n", seq_id);
        return 0;
    }

    // Calculate size for layer range for this sequence
    size_t layer_data_size = kv->layer_range_size_for_seq(seq_id, layer_start, layer_end);
    if (layer_data_size == 0) {
        fprintf(stderr, "Warning: Empty layer range [%d, %d) for seq %d\n", layer_start, layer_end, seq_id);
        return 0;
    }

    // Calculate total frame size
    size_t frame_size = sizeof(llama_kv_frame_header) + layer_data_size;

    // If dst is NULL, return required size
    if (!dst) {
        return frame_size;
    }

    if (dst_size < frame_size) {
        fprintf(stderr, "Warning: Buffer too small (%zu < %zu)\n", dst_size, frame_size);
        return 0;
    }

    // Serialize layer-specific KV data to temporary buffer
    std::vector<uint8_t> kv_buffer(layer_data_size);

    // Use a buffer writer to serialize layer data
    llama_io_write_buffer io_write(kv_buffer.data(), layer_data_size);

    // Write layer-specific data using the proper cell ranges
    kv->state_write_data_layers(io_write, cr, layer_start, layer_end);

    size_t actual_kv_size = io_write.n_bytes();

    // Prepare frame header
    llama_kv_frame_header header;
    header.magic = LLAMA_KV_FRAME_MAGIC;
    header.version = LLAMA_KV_FRAME_VERSION;
    header.seq_id = seq_id;
    header.compression = compression;
    header.uncompressed_size = actual_kv_size;
    header.compressed_size = actual_kv_size;  // No compression for now
    header.layer_start = layer_start;
    header.layer_end = layer_end;
    header._padding = 0;

    // Calculate checksum
    header.checksum = llama_kv_crc32(kv_buffer.data(), actual_kv_size);

    // Write header
    memcpy(dst, &header, sizeof(header));

    // Write KV data
    memcpy(dst + sizeof(header), kv_buffer.data(), actual_kv_size);

    return sizeof(header) + actual_kv_size;
}

size_t llama_kv_stream_deserialize(
    struct llama_context * ctx,
    llama_seq_id dest_seq_id,
    int32_t layer_start,
    int32_t layer_end,
    const uint8_t * src,
    size_t src_size,
    bool * out_checksum_ok) {

    if (!ctx || !src || src_size < sizeof(llama_kv_frame_header)) {
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    // Read header
    llama_kv_frame_header header;
    memcpy(&header, src, sizeof(header));

    // Validate magic and version
    if (header.magic != LLAMA_KV_FRAME_MAGIC) {
        fprintf(stderr, "KV stream deserialize: invalid magic 0x%08X\n", header.magic);
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    if (header.version != LLAMA_KV_FRAME_VERSION && header.version != 1) {
        fprintf(stderr, "KV stream deserialize: unsupported version %u\n", header.version);
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    // Validate layer range
    if (header.version >= 2) {
        if (header.layer_start != layer_start || header.layer_end != layer_end) {
            fprintf(stderr, "KV stream deserialize: layer range mismatch\n");
            if (out_checksum_ok) *out_checksum_ok = false;
            return 0;
        }
    }

    // Extract payload
    const uint8_t * payload = src + sizeof(header);
    size_t payload_size = header.compressed_size;

    if (src_size < sizeof(header) + payload_size) {
        fprintf(stderr, "KV stream deserialize: insufficient data\n");
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    // Verify checksum
    uint32_t checksum = llama_kv_crc32(payload, payload_size);
    if (checksum != header.checksum) {
        fprintf(stderr, "KV stream deserialize: checksum mismatch (0x%08X != 0x%08X)\n",
                checksum, header.checksum);
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    if (out_checksum_ok) *out_checksum_ok = true;

    // Get KV cache from context
    auto * mem = llama_get_memory(ctx);
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);

    if (!kv) {
        // Fallback to full KV deserialization
        fprintf(stderr, "Warning: Layer-specific streaming not available, using full KV\n");
        return llama_kv_frame_deserialize(ctx, dest_seq_id, src, src_size, out_checksum_ok);
    }

    // Deserialize layer-specific KV data
    llama_io_read_buffer io_read(payload, payload_size);

    // Build cell ranges for the destination sequence (not the full cache)
    llama_kv_cache::cell_ranges_t cr = kv->build_cell_ranges_for_seq(dest_seq_id);

    // If no cells found, we may need to allocate space for this sequence
    if (cr.data.empty()) {
        // Fallback: use full range and hope for the best
        // This can happen when importing to an empty cache
        fprintf(stderr, "Warning: No existing cells for seq %d, using full cache range\n", dest_seq_id);
        cr.strm = 0;
        cr.data.push_back({0, static_cast<uint32_t>(kv->get_size())});
    }

    // Read layer-specific data
    if (!kv->state_read_data_layers(io_read, cr, layer_start, layer_end)) {
        fprintf(stderr, "KV stream deserialize: failed to read layer data\n");
        if (out_checksum_ok) *out_checksum_ok = false;
        return 0;
    }

    return sizeof(header) + payload_size;
}

// Transfer statistics functions
void llama_kv_transfer_stats_init(struct llama_kv_transfer_stats * stats) {
    if (stats) {
        memset(stats, 0, sizeof(*stats));
    }
}

void llama_kv_transfer_stats_update(
    struct llama_kv_transfer_stats * stats,
    size_t bytes_transferred,
    double time_ms,
    bool is_send) {

    if (!stats) {
        return;
    }

    if (is_send) {
        stats->bytes_sent += bytes_transferred;
    } else {
        stats->bytes_received += bytes_transferred;
    }

    stats->transfer_time_ms += time_ms;
    stats->frame_count++;

    // Calculate bandwidth
    if (stats->transfer_time_ms > 0.0) {
        double total_bytes = stats->bytes_sent + stats->bytes_received;
        stats->bandwidth_mbps = (total_bytes * 8.0 / 1000000.0) / (stats->transfer_time_ms / 1000.0);
    }
}

void llama_kv_transfer_stats_print(const struct llama_kv_transfer_stats * stats, bool is_send) {
    if (!stats) {
        return;
    }

    size_t bytes = is_send ? stats->bytes_sent : stats->bytes_received;
    double mb = bytes / (1024.0 * 1024.0);

    fprintf(stderr, "KV transfer stats (%s): %.2f MB, %.2f ms, %.2f MB/s, %d frames",
            is_send ? "send" : "recv",
            mb,
            stats->transfer_time_ms,
            stats->bandwidth_mbps,
            stats->frame_count);

    if (stats->checksum_failures > 0 || stats->retries > 0) {
        fprintf(stderr, ", %d checksum failures, %d retries",
                stats->checksum_failures, stats->retries);
    }

    fprintf(stderr, "\n");
}
