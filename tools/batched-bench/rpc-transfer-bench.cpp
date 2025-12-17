#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "llama-kv-transfer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct options {
    std::string endpoint;
    size_t total_bytes   = 1024ull * 1024ull * 1024ull; // default: 1 GiB
    size_t chunk_bytes   = 8ull * 1024ull * 1024ull;    // default: 8 MiB
    int repeat           = 3;
    bool checksum        = true;
};

void print_usage(const char * prog) {
    std::printf("Synthetic RPC KV transfer benchmark\n");
    std::printf("Usage: %s --endpoint HOST:PORT [--bytes N] [--chunk N] [--repeat N] [--no-checksum]\n", prog);
    std::printf("  --endpoint    Target RPC endpoint (e.g., 10.0.0.10:50052)\n");
    std::printf("  --bytes       Total bytes per iteration (default 1GiB). Supports k/m/g suffix.\n");
    std::printf("  --chunk       Send chunk size (default 8MiB). Supports k/m/g suffix.\n");
    std::printf("  --repeat      Number of iterations (default 3)\n");
    std::printf("  --no-checksum Skip CRC32 calculation; commit uses 0\n");
}

bool parse_size(const char * text, size_t & out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }

    std::string s(text);
    char suffix = 0;
    if (std::isalpha(static_cast<unsigned char>(s.back()))) {
        suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(s.back())));
        s.pop_back();
    }

    char * end = nullptr;
    unsigned long long base = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') {
        return false;
    }

    size_t mult = 1;
    if (suffix == 'k') {
        mult = 1024ull;
    } else if (suffix == 'm') {
        mult = 1024ull * 1024ull;
    } else if (suffix == 'g') {
        mult = 1024ull * 1024ull * 1024ull;
    }

    out = static_cast<size_t>(base * mult);
    return true;
}

bool parse_args(int argc, char ** argv, options & opt) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--endpoint") == 0 && i + 1 < argc) {
            opt.endpoint = argv[++i];
        } else if (std::strcmp(arg, "--bytes") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], opt.total_bytes)) {
                return false;
            }
        } else if (std::strcmp(arg, "--chunk") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], opt.chunk_bytes)) {
                return false;
            }
        } else if (std::strcmp(arg, "--repeat") == 0 && i + 1 < argc) {
            opt.repeat = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--no-checksum") == 0) {
            opt.checksum = false;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }

    if (opt.endpoint.empty() || opt.total_bytes == 0 || opt.chunk_bytes == 0 || opt.repeat <= 0) {
        return false;
    }

    if (opt.chunk_bytes > opt.total_bytes) {
        opt.chunk_bytes = opt.total_bytes;
    }

    return true;
}

double to_mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

} // namespace

int main(int argc, char ** argv) {
    options opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        std::fprintf(stderr, "RPC backend not available (build with -DGGML_RPC=ON)\n");
        return 1;
    }

    std::printf("Endpoint     : %s\n", opt.endpoint.c_str());
    std::printf("Total bytes  : %.2f MiB\n", to_mib(opt.total_bytes));
    std::printf("Chunk bytes  : %.2f MiB\n", to_mib(opt.chunk_bytes));
    std::printf("Repeat       : %d\n", opt.repeat);
    std::printf("Checksum     : %s\n", opt.checksum ? "on" : "off");
    std::printf("----------------------------------------\n");

    std::vector<uint8_t> buffer(opt.total_bytes, 0u);

    for (int iter = 0; iter < opt.repeat; ++iter) {
        const int32_t seq_id = 1000 + iter; // arbitrary, unique per run

        auto t0 = std::chrono::steady_clock::now();
        if (!ggml_backend_rpc_kv_transfer_init(opt.endpoint.c_str(), seq_id, opt.total_bytes, LLAMA_KV_COMPRESSION_NONE)) {
            std::fprintf(stderr, "init failed on iter %d\n", iter);
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();

        size_t offset = 0;
        while (offset < opt.total_bytes) {
            const size_t chunk = std::min(opt.chunk_bytes, opt.total_bytes - offset);
            if (!ggml_backend_rpc_kv_transfer_send(opt.endpoint.c_str(), seq_id, offset, buffer.data() + offset, chunk)) {
                std::fprintf(stderr, "send failed at offset %zu on iter %d\n", offset, iter);
                return 1;
            }
            offset += chunk;
        }
        auto t2 = std::chrono::steady_clock::now();

        uint32_t checksum = 0;
        if (opt.checksum) {
            checksum = llama_kv_crc32(buffer.data(), buffer.size());
        }

        if (!ggml_backend_rpc_kv_transfer_commit(opt.endpoint.c_str(), seq_id, checksum)) {
            std::fprintf(stderr, "commit failed on iter %d\n", iter);
            return 1;
        }
        auto t3 = std::chrono::steady_clock::now();

        const double init_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double send_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
        const double commit_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        const double total_ms  = std::chrono::duration<double, std::milli>(t3 - t0).count();

        const double send_bw_mb_s  = to_mib(opt.total_bytes) / (send_ms / 1000.0);
        const double total_bw_mb_s = to_mib(opt.total_bytes) / (total_ms / 1000.0);

        std::printf("iter %d | init %.2f ms | send %.2f ms | commit %.2f ms | total %.2f ms | send BW %.2f MiB/s | total BW %.2f MiB/s\n",
                    iter + 1, init_ms, send_ms, commit_ms, total_ms, send_bw_mb_s, total_bw_mb_s);
    }

    return 0;
}
