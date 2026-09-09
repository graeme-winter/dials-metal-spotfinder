// Where frames come from.
//
// One source: an NXmx HDF5 file, whose /entry/data/data is a virtual dataset
// over a set of data files. That mapping is unpacked so that each frame can be
// read with H5Dread_chunk out of the file that holds it, which means the
// compressed bytes reach the spot finder without HDF5's filter pipeline ever
// running, and the bitshuffle plugin does not have to be installed.
//
// The Series / Reader split is not speculative generality: it is what keeps
// hdf5.h out of find_spots.cc. Everything HDF5 knows about lives behind these
// two classes, and a Reader is per thread because the HDF5 C library cannot be
// entered from two threads at once.

#ifndef SPOTFINDER_SERIES_HH
#define SPOTFINDER_SERIES_HH

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace series {

// One image, and the buffer its compressed bytes live in.
//
// `data` points into `storage`, so a Frame must be filled in place and not
// moved -- moving a std::string can relocate a small buffer and leave the span
// dangling. Reader::read fills one by pointer for that reason, which also lets
// a worker keep one Frame, and its capacity, for a whole run.
struct Frame {
  std::int64_t number = -1; // the frame's index in the dataset

  std::uint64_t height = 0; // slow
  std::uint64_t width = 0;  // fast
  unsigned bit_depth = 0;   // 16 or 32

  // "bslz4", "lz4", or empty for a chunk that was stored uncompressed. Points
  // at storage owned by the Reader, which outlives the Frame.
  std::string_view algorithm;

  std::span<const std::uint8_t> data;
  std::string storage;
};

// One per worker thread: an HDF5 file handle cannot be shared, and neither can
// the library itself.
class Reader {
public:
  virtual ~Reader() = default;

  // False if the frame is not there -- an unallocated chunk, which is how a
  // frame the writer never received appears. Throws std::runtime_error for
  // anything that is actually wrong.
  virtual bool read(const std::string &key, Frame *frame) = 0;
};

struct Info {
  std::string name;         // the master file's stem
  std::uint64_t images = 0; // 0 if not known
  std::uint64_t height = 0;
  std::uint64_t width = 0;
};

class Series {
public:
  virtual ~Series() = default;

  // False if the file is not readable yet, so that the caller can poll: a
  // master file that is still being written is a legitimate thing to be handed.
  virtual bool try_open(Info *info) = 0;

  // Keys that can be read now. The caller keeps track of what it has already
  // dispatched, so returning the same keys again is harmless.
  virtual std::vector<std::string> ready() = 0;

  // True once no further keys can appear.
  virtual bool finished() = 0;

  virtual std::string describe() const = 0;
  virtual std::unique_ptr<Reader> reader() = 0;
};

std::unique_ptr<Series> nxmx(std::string master);

} // namespace series

#endif // SPOTFINDER_SERIES_HH
