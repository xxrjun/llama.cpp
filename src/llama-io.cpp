#include "llama-io.h"
#include "ggml-backend.h"

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read_to(&str_size, sizeof(str_size));

    str.assign((const char *) read(str_size), str_size);
}
void llama_io_write_buffer::write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) {
    if (size > buf_size) {
        throw std::runtime_error("unexpectedly reached end of buffer");
    }
    ggml_backend_tensor_get(tensor, ptr, offset, size);
    ptr += size;
    size_written += size;
    buf_size -= size;
}