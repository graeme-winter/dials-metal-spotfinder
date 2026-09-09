#include "decompress.hh"

#include <bitshuffle.h>
#include <lz4.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace decompress {
namespace {

// Both HDF5 compression filters used here prefix their output with the same
// 12-byte header, and both count in bytes rather than elements: eight bytes of
// uncompressed size, then four of block size, big-endian.
constexpr std::size_t kFilterHeader = 12;

struct FilterHeader {
  std::uint64_t uncompressed_size = 0;
  std::uint32_t block_size = 0;
};

bool read_filter_header(std::span<const std::uint8_t> data,
                        FilterHeader *header) {
  if (data.size() < kFilterHeader)
    return false;
  std::uint64_t size = 0;
  for (int i = 0; i < 8; i++)
    size = (size << 8) | data[i];
  std::uint32_t block = 0;
  for (int i = 8; i < 12; i++)
    block = (block << 8) | data[i];
  header->uncompressed_size = size;
  header->block_size = block;
  return true;
}

std::string fail(const char *what, std::int64_t code) {
  return std::string(what) + " failed with code " + std::to_string(code);
}

// The HDF5 LZ4 filter's framing: after the 12-byte header, each block is a
// four-byte big-endian compressed length followed by that many bytes.
void lz4_blocks(std::span<const std::uint8_t> in, std::span<std::uint8_t> out,
                std::size_t block_size) {
  std::size_t read = 0;
  std::size_t written = 0;

  while (written < out.size()) {
    if (read + 4 > in.size())
      throw std::runtime_error("lz4 data truncated");
    std::uint32_t compressed = 0;
    for (int i = 0; i < 4; i++)
      compressed = (compressed << 8) | in[read + i];
    read += 4;
    if (compressed == 0 || read + compressed > in.size()) {
      throw std::runtime_error("lz4 block size out of range");
    }

    const std::size_t remaining = out.size() - written;
    const std::size_t expected =
        remaining < block_size ? remaining : block_size;
    const int produced = LZ4_decompress_safe(
        reinterpret_cast<const char *>(in.data() + read),
        reinterpret_cast<char *>(out.data() + written),
        static_cast<int>(compressed), static_cast<int>(expected));
    if (produced < 0)
      throw std::runtime_error(fail("lz4 decompression", produced));
    read += compressed;
    written += static_cast<std::size_t>(produced);
  }
}

} // namespace

std::size_t frame_bytes(std::size_t height, std::size_t width,
                        unsigned bit_depth) {
  return height * width * bit_depth / 8;
}

void image(std::span<const std::uint8_t> data, std::string_view algorithm,
           unsigned bit_depth, std::size_t height, std::size_t width,
           std::span<std::uint8_t> out) {
  const std::size_t expected = frame_bytes(height, width, bit_depth);
  const std::size_t element = bit_depth / 8;
  if (element == 0 || out.size() < expected) {
    throw std::runtime_error("output buffer too small for the frame");
  }

  if (algorithm.empty()) {
    if (data.size() != expected) {
      throw std::runtime_error("uncompressed frame is " +
                               std::to_string(data.size()) +
                               " bytes, expected " + std::to_string(expected));
    }
    std::memcpy(out.data(), data.data(), expected);
    return;
  }

  // The filter header is the cheapest possible check that the pointer, the
  // dimensions and the bit depth all agree, and it has to be read anyway for
  // the block size.
  FilterHeader header;
  if (!read_filter_header(data, &header)) {
    throw std::runtime_error("too short for a filter header");
  }
  if (header.uncompressed_size != expected) {
    throw std::runtime_error(
        "filter header says " + std::to_string(header.uncompressed_size) +
        " bytes uncompressed, expected " + std::to_string(expected));
  }

  const std::span<const std::uint8_t> payload = data.subspan(kFilterHeader);

  if (algorithm == "bslz4") {
    // bitshuffle counts in elements where the header counts in bytes.
    const std::int64_t consumed =
        bshuf_decompress_lz4(payload.data(), out.data(), expected / element,
                             element, header.block_size / element);
    if (consumed < 0) {
      throw std::runtime_error(fail("bitshuffle decompression", consumed));
    }
    return;
  }

  if (algorithm == "lz4") {
    lz4_blocks(payload, out.subspan(0, expected),
               header.block_size > 0 ? header.block_size : expected);
    return;
  }

  throw std::runtime_error("unknown compression algorithm " +
                           std::string(algorithm));
}

} // namespace decompress
