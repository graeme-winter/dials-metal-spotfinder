// Decompression of one image.
//
// Chunks arrive as they were written: bitshuffle+LZ4, plain LZ4, or not
// compressed at all. Kept apart from the reader so that a frame takes the same
// path through the analysis however it arrived, and so that the check on the
// filter header lives in one place.

#ifndef SPOTFINDER_DECOMPRESS_HH
#define SPOTFINDER_DECOMPRESS_HH

#include <cstdint>
#include <span>
#include <string_view>

namespace decompress {

// How many bytes `image` will write, given the frame geometry.
std::size_t frame_bytes(std::size_t height, std::size_t width,
                        unsigned bit_depth);

// Decompress one image into `out`, which must be at least frame_bytes() long.
// `data` is the chunk exactly as HDF5 stored it, filter header and all, and
// `algorithm` is what the dataset's filter pipeline said it is: "bslz4", "lz4",
// or empty for a chunk stored uncompressed.
//
// Throws std::runtime_error if the data does not describe the frame it should
// -- which is the check worth having, since a wrong pointer or a wrong bit
// depth usually decompresses to something rather than failing outright.
void image(std::span<const std::uint8_t> data, std::string_view algorithm,
           unsigned bit_depth, std::size_t height, std::size_t width,
           std::span<std::uint8_t> out);

} // namespace decompress

#endif // SPOTFINDER_DECOMPRESS_HH
