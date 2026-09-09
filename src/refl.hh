// Writing a DIALS reflection table.
//
// A .refl file is a msgpack document and nothing else -- no header, no
// compression, no pickle:
//
//   [ "dials::af::reflection_table", 2,
//     { "identifiers" : { id : identifier, ... },
//       "nrows"       : N,
//       "data"        : { name : [ type, [ N, <binary> ] ], ... } } ]
//
// Each column is a type name and a raw little-endian dump of N elements of that
// type, which is why this needs no msgpack library and no DIALS: the encoding
// is a few length-prefixed headers around memory that is already in the right
// layout. The format is dials/array_family/reflection_table_msgpack_adapter.h,
// and version 2 is what DIALS writes; it reads 1 or 2.
//
// The raw dump is also the format's one real constraint. It is a memory image,
// so it is neither portable across endianness nor across the width of
// std::size_t -- DIALS has the same property and the same silence about it.
// This refuses to write on a machine where the result would not read back.
//
// What is written is what dials.find_spots writes, column for column:
//
//   bbox                    int6            the bounding box
//   flags                   std::size_t     Strong, and nothing else
//   id                      int             which experiment
//   intensity.sum.value     double          summed over the signal pixels
//   intensity.sum.variance  double          equal to it, with no background
//   n_signal                int             pixels in the spot
//   panel                   std::size_t     0; one panel only
//   shoebox                 Shoebox<>       optional, and large
//   xyzobs.px.value         vec3<double>    the centroid
//   xyzobs.px.variance      vec3<double>    its standard error, squared
//
// The shoebox column is what dials.find_spots calls output.shoeboxes=True, and
// it is the expensive one: a dense block of float data, byte mask and float
// background per spot, where every other column is a few bytes a row. It
// carries no information the pixel list does not, since the spot finder only
// ever fills the signal pixels and leaves the background at zero, so it is
// worth writing for the image viewer and worth leaving out for indexing.

#ifndef SPOTFINDER_REFL_HH
#define SPOTFINDER_REFL_HH

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dials_spots.hh"

namespace refl {

struct Options {
  std::size_t panel = 0; // one panel; a segmented detector is not handled
  int id = 0;            // the experiment every spot belongs to

  // The experiment identifier from the .expt, so that the table and the
  // experiment list agree about what they describe. Empty writes an empty
  // identifier map, which DIALS accepts.
  std::string identifier;

  bool shoeboxes = true;
};

// Write `spots` to `path`. `pixels` and `width` are needed for the shoeboxes
// and for turning a row-major index back into a column and a row; they must be
// the vector and the frame width the Labeller was given.
//
// Throws std::runtime_error if the file cannot be written, and refuses outright
// on a big-endian machine or one where std::size_t is not eight bytes.
void write(const std::string &path, const std::vector<dials_spots::Spot> &spots,
           const std::vector<dials_spots::Pixel> &pixels, std::size_t width,
           const Options &options);

} // namespace refl

#endif // SPOTFINDER_REFL_HH
