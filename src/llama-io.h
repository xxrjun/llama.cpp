#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <cstring>
#include <stdexcept>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual const uint8_t * read(size_t size) = 0;
    virtual void read_to(void * dst, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
// Buffer-based implementations
class llama_io_write_buffer : public llama_io_write_i {
public:
    llama_io_write_buffer(uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override;

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_buffer : public llama_io_read_i {
public:
    llama_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};