// Grouping signal pixels into spots the way DIALS does it.
//
// src/spots.{hh,cc} groups one frame at a time, four-connected, which is what a
// per-frame report needs. A rotation series is not a stack of independent
// frames: a reflection is swept through the Ewald sphere over one or two
// degrees and lands on several of them, so DIALS groups in three dimensions --
// six-connected, x, y and z -- and a spot that spans frames 12 to 15 is one
// reflection with one centroid rather than four.
//
// This is that grouping, plus what DIALS computes from it: the centroid, its
// variance, the summed intensity, the bounding box, and the two filters
// dials.find_spots applies by default. It is a transcription of
// PixelListLabeller::labels_3d, PixelListShoeboxCreator,
// Shoebox::centroid_valid, Shoebox::summed_intensity and
// PeakCentroidDistanceFilter, so that what comes out can be written into a
// reflection table and handed to dials.index.
//
// Two things are deliberately different from DIALS.
//
//   * It is streaming. DIALS holds every signal pixel of the whole sweep in
//     memory and labels the lot at the end, which at a few hundred thousand
//     pixels a frame is gigabytes. Only frames f - 1 and f can be connected, so
//     a component with no pixel on the frame just added can never grow again:
//     it is emitted and its pixels are dropped. What is held is the components
//     that touch the current frame, which is a couple of frames' worth.
//
//   * Rows come out in the order their components closed -- by last frame, then
//     by first pixel -- where DIALS emits them in first-pixel order over the
//     whole sweep. Nothing downstream depends on row order, and both are
//     deterministic.
//
// Frames must be added in ascending order, each exactly once. A gap in the
// numbering is not an error: no reflection can bridge a frame that was never
// written, so everything open is closed at the gap.

#ifndef SPOTFINDER_DIALS_SPOTS_HH
#define SPOTFINDER_DIALS_SPOTS_HH

#include <cstddef>
#include <cstdint>
#include <vector>

#include "signal_pixel.hh"

namespace dials_spots {

// One signal pixel of an accepted spot, kept so that a shoebox can be written
// without going back to the images. Twelve bytes: the frame, the row-major
// position within it, and the raw value.
struct Pixel {
  std::int32_t frame;
  std::uint32_t index;
  std::uint32_t value;
};

// One accepted spot, in DIALS' terms and DIALS' units.
struct Spot {
  // x0, x1, y0, y1, z0, z1, upper bounds exclusive, as int6 in a reflection
  // table. z is the frame, so a spot on one frame alone has z1 == z0 + 1.
  std::int32_t bbox[6] = {0, 0, 0, 0, 0, 0};

  // xyzobs.px.value: the intensity-weighted centroid, in pixels and frames,
  // with the centre of pixel (0, 0) of frame 0 at (0.5, 0.5, 0.5).
  double position[3] = {0.0, 0.0, 0.0};

  // xyzobs.px.variance: the unbiased standard error on the mean, squared, plus
  // 1/12. The constant is DIALS', and deliberate -- see the comment on
  // mean_sq_error in dials/algorithms/image/centroid/centroid_points.h.
  double variance[3] = {0.0, 0.0, 0.0};

  // intensity.sum.value and .variance. No background is subtracted at this
  // stage, so both are the sum of the signal pixels: DIALS' Summation over a
  // shoebox whose background array is all zero.
  double intensity = 0.0;
  double intensity_variance = 0.0;

  std::uint32_t n_signal = 0; // pixels in the spot
  std::uint64_t first = 0;    // where its pixels start in pixels()
};

struct Options {
  // dials.find_spots defaults. min_spot_size is what phil resolves Auto to for
  // a pixel array detector, which is every detector this can read; the other
  // two are the phil defaults outright.
  std::size_t min_spot_size = 3;
  std::size_t max_spot_size = 1000;

  // Maximum distance between the brightest pixel and the centroid, in pixels.
  // Zero or less turns the filter off, which is what max_separation=None does.
  double max_separation = 2.0;

  // Group each frame on its own, as DIALS does for a still or under
  // spotfinder.force_2d=True. A rotation series wants this off.
  bool two_d = false;
};

// What became of every group, for the report. The three rejection counts
// account for the difference between groups and accepted.
struct Counts {
  std::uint64_t signal_pixels = 0; // handed in
  std::uint64_t groups = 0;        // connected components found
  std::uint64_t too_small = 0;     // fewer than min_spot_size pixels
  std::uint64_t too_large = 0;     // more than max_spot_size
  std::uint64_t separated = 0;     // peak too far from the centroid
  std::uint64_t accepted = 0;
};

class Labeller {
public:
  Labeller(std::size_t height, std::size_t width, Options options);

  // `signal` must be ascending by index, which is what both finders produce.
  // Throws std::runtime_error if a frame arrives out of order or twice.
  void add(std::int64_t frame, const std::vector<SignalPixel> &signal);

  // Close everything still open. Call once, after the last frame.
  void finish();

  const std::vector<Spot> &spots() const { return spots_; }
  const std::vector<Pixel> &pixels() const { return kept_; }
  const Counts &counts() const { return counts_; }

  // Pixels currently held for components that might still grow, for a caller
  // that wants to report the high water mark.
  std::size_t live() const { return live_.size(); }

private:
  std::uint32_t find(std::uint32_t i);
  void merge(std::uint32_t a, std::uint32_t b);

  // Emit every component with no pixel on `keep_frame`, and compact the rest to
  // the front. `keep_frame` below every frame present closes the lot.
  void flush(std::int64_t keep_frame);
  void emit(const std::uint32_t *members, std::size_t n);

  std::size_t height_;
  std::size_t width_;
  Options options_;

  std::vector<Pixel> live_;
  std::vector<std::uint32_t> parent_;
  std::size_t previous_ = 0; // where the last frame's pixels start in live_
  std::int64_t last_frame_ = 0;
  bool started_ = false;

  std::vector<Spot> spots_;
  std::vector<Pixel> kept_;
  Counts counts_;

  // Reused by flush() so that a sweep allocates these once rather than per
  // frame. They are scratch, not state.
  std::vector<std::uint32_t> root_;
  std::vector<std::uint32_t> label_;
  std::vector<std::uint32_t> offset_;
  std::vector<std::uint32_t> cursor_;
  std::vector<std::uint32_t> member_;
  std::vector<char> open_;
};

} // namespace dials_spots

#endif // SPOTFINDER_DIALS_SPOTS_HH
