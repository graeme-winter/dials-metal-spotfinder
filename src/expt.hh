// Reading the few facts a spot list needs out of a dxtbx experiment list.
//
// An .expt is JSON, and dials.index wants the .expt and the .refl to agree
// about three things: which images the scan covers, so that a spot's z is the
// array index DIALS expects rather than the detector's own numbering; how big a
// panel is, so that a mismatched frame is caught here rather than as a
// nonsensical lattice later; and the experiment identifier, which is the string
// DIALS uses to tie a table to an experiment.
//
// Nothing else is taken from it, and nothing is written back. This deliberately
// does not try to be dxtbx: the beam, the goniometer and the detector's
// hierarchy are dials.import's business, and reimplementing them here would be
// a second model to keep in step with the first.
//
// The JSON reader is a few hundred lines because pulling three numbers out of a
// document with a regular expression is the kind of thing that works until an
// experiment list gains a field.

#ifndef SPOTFINDER_EXPT_HH
#define SPOTFINDER_EXPT_HH

#include <cstddef>
#include <cstdint>
#include <string>

namespace expt {

struct Info {
  std::size_t experiments = 0;

  // The identifier of the first experiment. Empty if dials.import did not set
  // one, which is not an error.
  std::string identifier;

  // The scan, as image numbers, inclusive and counting from one, exactly as
  // dxtbx stores image_range. The array index of image n is n - 1, and that
  // index is what a reflection's z is measured in.
  bool has_scan = false;
  std::int64_t first_image = 0;
  std::int64_t last_image = 0;

  // The first detector's panels. A segmented detector is reported rather than
  // refused here; the caller decides.
  bool has_detector = false;
  std::size_t panels = 0;
  std::size_t image_fast = 0; // pixels across a row
  std::size_t image_slow = 0; // rows

  std::int64_t images() const {
    return has_scan ? last_image - first_image + 1 : 0;
  }
};

// Throws std::runtime_error if the file cannot be read or is not an experiment
// list. A missing scan or detector is not an error: a still has no scan.
Info read(const std::string &path);

// One line for the startup banner.
std::string describe(const Info &info);

} // namespace expt

#endif // SPOTFINDER_EXPT_HH
