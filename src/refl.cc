#include "refl.hh"

#include <bit>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace refl {
namespace {

// DIALS' shoebox mask codes, from dials/model/data/mask_code.h. The spot finder
// marks its own pixels valid and foreground and leaves the rest of the box at
// zero, which is what makes centroid_valid() a centroid of the signal pixels.
constexpr std::uint8_t kValid = 1 << 0;
constexpr std::uint8_t kForeground = 1 << 2;

// dials::af::Flags::Strong. dials.find_spots sets this on every row it writes
// and nothing else, so a strong spot list is recognisable as one.
constexpr std::uint64_t kStrong = 1 << 5;

// Shoebox<> is Shoebox<ProfileFloatType>, and dials/config.h makes that float
// rather than double. Getting this wrong writes a blob of the right length that
// decodes to nonsense, which is why it is named here rather than assumed.
static_assert(sizeof(float) == 4, "float must be four bytes");

// msgpack's bin type tops out at a 32-bit length.
constexpr std::uint64_t kBinLimit = 0xffffffffull;

// A buffered writer. The whole file is streamed rather than assembled, because
// the shoebox column is the largest thing here by two orders of magnitude and
// holding it twice -- once as spots, once as bytes -- is not worth the
// simplicity.
class Out {
public:
  explicit Out(const std::string &path) : path_(path) {
    file_ = std::fopen(path.c_str(), "wb");
    if (file_ == nullptr)
      throw std::runtime_error("cannot write " + path);
    buffer_.reserve(kFlushAt + 64);
  }

  ~Out() {
    if (file_ != nullptr)
      std::fclose(file_);
  }

  Out(const Out &) = delete;
  Out &operator=(const Out &) = delete;

  void bytes(const void *data, std::size_t n) {
    const char *at = static_cast<const char *>(data);
    buffer_.insert(buffer_.end(), at, at + n);
    if (buffer_.size() >= kFlushAt)
      flush();
  }

  void byte(std::uint8_t value) { bytes(&value, 1); }

  void close() {
    flush();
    if (std::fclose(file_) != 0) {
      file_ = nullptr;
      throw std::runtime_error("cannot finish writing " + path_);
    }
    file_ = nullptr;
  }

private:
  static constexpr std::size_t kFlushAt = 1u << 20;

  void flush() {
    if (buffer_.empty())
      return;
    const std::size_t written =
        std::fwrite(buffer_.data(), 1, buffer_.size(), file_);
    if (written != buffer_.size())
      throw std::runtime_error("short write to " + path_);
    buffer_.clear();
  }

  std::string path_;
  std::FILE *file_ = nullptr;
  std::vector<char> buffer_;
};

// msgpack is big-endian in its own headers and byte-for-byte in its payloads,
// which is the whole of what follows.
void be(Out &out, std::uint64_t value, int width) {
  std::uint8_t encoded[8];
  for (int i = 0; i < width; i++)
    encoded[i] = static_cast<std::uint8_t>(value >> (8 * (width - 1 - i)));
  out.bytes(encoded, static_cast<std::size_t>(width));
}

void put_uint(Out &out, std::uint64_t value) {
  if (value < 0x80) {
    out.byte(static_cast<std::uint8_t>(value)); // positive fixint
  } else if (value <= 0xff) {
    out.byte(0xcc);
    be(out, value, 1);
  } else if (value <= 0xffff) {
    out.byte(0xcd);
    be(out, value, 2);
  } else if (value <= 0xffffffff) {
    out.byte(0xce);
    be(out, value, 4);
  } else {
    out.byte(0xcf);
    be(out, value, 8);
  }
}

void put_str(Out &out, const std::string &text) {
  const std::size_t n = text.size();
  if (n < 32) {
    out.byte(static_cast<std::uint8_t>(0xa0 | n));
  } else if (n <= 0xff) {
    out.byte(0xd9);
    be(out, n, 1);
  } else if (n <= 0xffff) {
    out.byte(0xda);
    be(out, n, 2);
  } else {
    out.byte(0xdb);
    be(out, n, 4);
  }
  out.bytes(text.data(), n);
}

void put_array(Out &out, std::size_t n) {
  if (n < 16) {
    out.byte(static_cast<std::uint8_t>(0x90 | n));
  } else if (n <= 0xffff) {
    out.byte(0xdc);
    be(out, n, 2);
  } else {
    out.byte(0xdd);
    be(out, n, 4);
  }
}

void put_map(Out &out, std::size_t n) {
  if (n < 16) {
    out.byte(static_cast<std::uint8_t>(0x80 | n));
  } else if (n <= 0xffff) {
    out.byte(0xde);
    be(out, n, 2);
  } else {
    out.byte(0xdf);
    be(out, n, 4);
  }
}

void put_bin_header(Out &out, std::uint64_t n) {
  if (n > kBinLimit) {
    throw std::runtime_error(
        "a reflection table column would be over 4 GB, which msgpack's binary "
        "type cannot describe; write without shoeboxes");
  }
  if (n <= 0xff) {
    out.byte(0xc4);
    be(out, n, 1);
  } else if (n <= 0xffff) {
    out.byte(0xc5);
    be(out, n, 2);
  } else {
    out.byte(0xc6);
    be(out, n, 4);
  }
}

// The header a column shares: [ type, [ nrows, bin(bytes) ] ]. The caller then
// writes exactly `bytes` bytes.
void open_column(Out &out, const std::string &name, const std::string &type,
                 std::size_t rows, std::uint64_t bytes) {
  put_str(out, name);
  put_array(out, 2);
  put_str(out, type);
  put_array(out, 2);
  put_uint(out, rows);
  put_bin_header(out, bytes);
}

// Little-endian in the payload, which is to say native: these are memory
// images. write() has already refused to run anywhere that is not true.
template <typename T> void raw(Out &out, T value) {
  out.bytes(&value, sizeof(T));
}

std::uint64_t elements(const dials_spots::Spot &spot) {
  return static_cast<std::uint64_t>(spot.bbox[1] - spot.bbox[0]) *
         static_cast<std::uint64_t>(spot.bbox[3] - spot.bbox[2]) *
         static_cast<std::uint64_t>(spot.bbox[5] - spot.bbox[4]);
}

// panel, six bbox bounds, a version byte, then float data, byte mask and float
// background over the box.
std::uint64_t shoebox_bytes(const dials_spots::Spot &spot) {
  return 4 + 6 * 4 + 1 + elements(spot) * (4 + 1 + 4);
}

std::uint64_t
shoebox_column_bytes(const std::vector<dials_spots::Spot> &spots) {
  std::uint64_t total = 0;
  for (const dials_spots::Spot &spot : spots)
    total += shoebox_bytes(spot);
  return total;
}

void check_platform() {
  static_assert(sizeof(std::size_t) == 8,
                "a reflection table's std::size_t columns are eight bytes "
                "wide, so this needs a 64-bit build");
  if constexpr (std::endian::native != std::endian::little) {
    throw std::runtime_error(
        "a reflection table's columns are raw little-endian memory images and "
        "this is a big-endian machine; DIALS could not read what we wrote");
  }
}

} // namespace

void write(const std::string &path, const std::vector<dials_spots::Spot> &spots,
           const std::vector<dials_spots::Pixel> &pixels, std::size_t width,
           const Options &options) {
  check_platform();
  if (width == 0)
    throw std::runtime_error("refl: a frame width of zero");

  const std::size_t rows = spots.size();
  Out out(path);

  put_array(out, 3);
  put_str(out, "dials::af::reflection_table");
  put_uint(out, 2); // the format version DIALS writes; it reads 1 or 2

  put_map(out, 3);

  // Which experiment each id names. DIALS keeps the table and the experiment
  // list tied together by these strings, so an identifier from the .expt is
  // carried through rather than invented.
  put_str(out, "identifiers");
  if (options.identifier.empty()) {
    put_map(out, 0);
  } else {
    put_map(out, 1);
    put_uint(out, static_cast<std::uint64_t>(options.id));
    put_str(out, options.identifier);
  }

  put_str(out, "nrows");
  put_uint(out, rows);

  put_str(out, "data");
  put_map(out, options.shoeboxes ? 10 : 9);

  // Alphabetical, which is the order DIALS' own std::map gives it. Nothing
  // reads them positionally; it just makes two files easy to compare.
  open_column(out, "bbox", "int6", rows, static_cast<std::uint64_t>(rows) * 24);
  for (const dials_spots::Spot &spot : spots) {
    for (int i = 0; i < 6; i++)
      raw<std::int32_t>(out, spot.bbox[i]);
  }

  open_column(out, "flags", "std::size_t", rows,
              static_cast<std::uint64_t>(rows) * 8);
  for (std::size_t i = 0; i < rows; i++)
    raw<std::uint64_t>(out, kStrong);

  open_column(out, "id", "int", rows, static_cast<std::uint64_t>(rows) * 4);
  for (std::size_t i = 0; i < rows; i++)
    raw<std::int32_t>(out, options.id);

  open_column(out, "intensity.sum.value", "double", rows,
              static_cast<std::uint64_t>(rows) * 8);
  for (const dials_spots::Spot &spot : spots)
    raw<double>(out, spot.intensity);

  open_column(out, "intensity.sum.variance", "double", rows,
              static_cast<std::uint64_t>(rows) * 8);
  for (const dials_spots::Spot &spot : spots)
    raw<double>(out, spot.intensity_variance);

  open_column(out, "n_signal", "int", rows,
              static_cast<std::uint64_t>(rows) * 4);
  for (const dials_spots::Spot &spot : spots)
    raw<std::int32_t>(out, static_cast<std::int32_t>(spot.n_signal));

  open_column(out, "panel", "std::size_t", rows,
              static_cast<std::uint64_t>(rows) * 8);
  for (std::size_t i = 0; i < rows; i++)
    raw<std::uint64_t>(out, static_cast<std::uint64_t>(options.panel));

  if (options.shoeboxes) {
    open_column(out, "shoebox", "Shoebox<>", rows, shoebox_column_bytes(spots));
    std::vector<float> data;
    std::vector<std::uint8_t> mask;
    for (const dials_spots::Spot &spot : spots) {
      const std::size_t xsize =
          static_cast<std::size_t>(spot.bbox[1] - spot.bbox[0]);
      const std::size_t ysize =
          static_cast<std::size_t>(spot.bbox[3] - spot.bbox[2]);
      const std::size_t zsize =
          static_cast<std::size_t>(spot.bbox[5] - spot.bbox[4]);
      const std::size_t n = xsize * ysize * zsize;

      data.assign(n, 0.0f);
      mask.assign(n, 0);
      for (std::uint32_t k = 0; k < spot.n_signal; k++) {
        const dials_spots::Pixel &pixel = pixels[spot.first + k];
        const std::size_t x = static_cast<std::size_t>(pixel.index % width) -
                              static_cast<std::size_t>(spot.bbox[0]);
        const std::size_t y = static_cast<std::size_t>(pixel.index / width) -
                              static_cast<std::size_t>(spot.bbox[2]);
        const std::size_t z = static_cast<std::size_t>(
            static_cast<std::int64_t>(pixel.frame) - spot.bbox[4]);
        // c_grid<3>(zsize, ysize, xsize): z slowest, x fastest.
        const std::size_t at = (z * ysize + y) * xsize + x;
        data[at] = static_cast<float>(pixel.value);
        mask[at] = kValid | kForeground;
      }

      raw<std::uint32_t>(out, static_cast<std::uint32_t>(options.panel));
      for (int i = 0; i < 6; i++)
        raw<std::int32_t>(out, spot.bbox[i]);
      out.byte(2); // version 2: a uint8 mask rather than the original int one
      out.bytes(data.data(), n * sizeof(float));
      out.bytes(mask.data(), n);
      // The background, which the spot finder never fills. Reusing the data
      // buffer would write the pixels twice; this is a zeroed one.
      data.assign(n, 0.0f);
      out.bytes(data.data(), n * sizeof(float));
    }
  }

  open_column(out, "xyzobs.px.value", "vec3<double>", rows,
              static_cast<std::uint64_t>(rows) * 24);
  for (const dials_spots::Spot &spot : spots) {
    for (int axis = 0; axis < 3; axis++)
      raw<double>(out, spot.position[axis]);
  }

  open_column(out, "xyzobs.px.variance", "vec3<double>", rows,
              static_cast<std::uint64_t>(rows) * 24);
  for (const dials_spots::Spot &spot : spots) {
    for (int axis = 0; axis < 3; axis++)
      raw<double>(out, spot.variance[axis]);
  }

  out.close();
}

} // namespace refl
