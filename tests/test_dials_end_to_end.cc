// Frames in, reflection table out, with no detector and no HDF5.
//
// The other tests take the three stages apart: the threshold in
// test_dext_squares.cc, the grouping in test_dials_spots.cc, the file in
// test_refl.cc. This one puts them together in the order
// dials-metal-find-spots does -- dext over each frame, the frames handed to the
// labeller in order, the spots written out -- over frames with reflections
// planted at known positions and a rocking curve across three frames each.
//
// What it checks is that a reflection planted at (x, y) comes back at (x, y),
// which is the only claim that matters and the one that no amount of testing
// the parts separately makes. It is also the test to reach for after changing
// any of them, since a mistake in the coordinate conventions -- fast against
// slow, pixel corners against pixel centres, frames against array indices --
// shows up here as a systematic offset and nowhere else as anything at all.
//
// Writes the table to the path given as the first argument, so that
// tests/check_refl.py can be run over it; defaults to a temporary name that it
// then removes.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "dext.hh"
#include "dials_spots.hh"
#include "refl.hh"
#include "signal_pixel.hh"

namespace {

int failures = 0;

void check(const std::string &what, unsigned long long actual,
           unsigned long long expected) {
  if (actual == expected)
    return;
  std::printf("FAIL %s: %llu, expected %llu\n", what.c_str(), actual, expected);
  failures++;
}

void near(const std::string &what, double actual, double expected,
          double tolerance) {
  if (std::fabs(actual - expected) <= tolerance)
    return;
  std::printf("FAIL %s: %.3f, expected %.3f give or take %.3f\n", what.c_str(),
              actual, expected, tolerance);
  failures++;
}

struct Planted {
  std::size_t slow = 0; // row, which is y
  std::size_t fast = 0; // column, which is x
  double peak = 0.0;
};

// A Gaussian a pixel and a half wide, scaled by the rocking curve, on a flat
// background. Deliberately not Poisson: this test is about where a spot lands,
// and a noisy frame would make the tolerances a statement about the noise.
std::vector<std::uint16_t> frame(std::size_t height, std::size_t width,
                                 const std::vector<Planted> &spots,
                                 double scale) {
  std::vector<std::uint16_t> pixels(height * width, 2);
  for (const Planted &spot : spots) {
    for (int di = -3; di <= 3; di++) {
      for (int dj = -3; dj <= 3; dj++) {
        const long i = static_cast<long>(spot.slow) + di;
        const long j = static_cast<long>(spot.fast) + dj;
        if (i < 0 || j < 0 || i >= static_cast<long>(height) ||
            j >= static_cast<long>(width))
          continue;
        const double radius = static_cast<double>(di * di + dj * dj);
        const double value =
            spot.peak * scale * std::exp(-radius / (2.0 * 1.5 * 1.5));
        const std::size_t k =
            static_cast<std::size_t>(i) * width + static_cast<std::size_t>(j);
        const double total = pixels[k] + value;
        pixels[k] =
            static_cast<std::uint16_t>(total > 65533.0 ? 65533.0 : total);
      }
    }
  }
  return pixels;
}

} // namespace

int main(int argc, char **argv) {
  const std::size_t height = 240;
  const std::size_t width = 320;

  // Well separated, so that grouping cannot join two of them, and away from the
  // borders except for one that is deliberately close to a corner. Peak counts
  // spanning what a real frame carries -- a few hundred to a couple of thousand
  // in the strongest pixel of a spot -- rather than the top of the 16-bit
  // range, because a spot of tens of thousands of counts a pixel is not what
  // this has to handle and planting one would be exercising the window
  // arithmetic outside its domain. tests/test_dext_squares.cc is where that
  // domain is pinned down.
  const std::vector<Planted> planted{{40, 50, 250.0},    {40, 150, 500.0},
                                     {40, 250, 900.0},   {120, 50, 1200.0},
                                     {120, 150, 1600.0}, {120, 250, 2000.0},
                                     {200, 50, 2500.0},  {200, 150, 700.0},
                                     {200, 250, 350.0},  {5, 5, 1400.0}};

  // Five frames, with each reflection rocking through frames 1, 2 and 3: a
  // three-frame spot, which is what a real sweep gives and what the
  // two-dimensional grouping would report as three.
  const double curve[5] = {0.0, 0.35, 1.0, 0.35, 0.0};

  dials_spots::Options options; // the dials.find_spots defaults
  dials_spots::Labeller labeller(height, width, options);

  dext_scratch<std::uint16_t> scratch;
  std::vector<SignalPixel> signal;
  std::uint64_t pixels_found = 0;

  for (std::int64_t z = 0; z < 5; z++) {
    const std::vector<std::uint16_t> image =
        frame(height, width, planted, curve[z]);
    check("the threshold accepts the frame",
          dext<std::uint16_t>(image.data(), signal, height, width, scratch), 0);
    pixels_found += signal.size();
    labeller.add(z, signal);
  }
  labeller.finish();

  const std::vector<dials_spots::Spot> &spots = labeller.spots();
  std::printf("  %llu signal pixels over five frames, %zu spots kept of %llu "
              "groups\n",
              static_cast<unsigned long long>(pixels_found), spots.size(),
              static_cast<unsigned long long>(labeller.counts().groups));

  check("one spot per reflection planted", spots.size(), planted.size());

  // Match by position rather than by order: the emission order follows the
  // frame a component closed on, which is not the order they were planted in.
  for (const Planted &want : planted) {
    const dials_spots::Spot *best = nullptr;
    double closest = 0.0;
    for (const dials_spots::Spot &spot : spots) {
      const double dx =
          spot.position[0] - (static_cast<double>(want.fast) + 0.5);
      const double dy =
          spot.position[1] - (static_cast<double>(want.slow) + 0.5);
      const double distance = std::sqrt(dx * dx + dy * dy);
      if (best == nullptr || distance < closest) {
        best = &spot;
        closest = distance;
      }
    }
    const std::string where = "the reflection at (" +
                              std::to_string(want.slow) + ", " +
                              std::to_string(want.fast) + ")";
    if (best == nullptr) {
      std::printf("FAIL %s: nothing was found at all\n", where.c_str());
      failures++;
      continue;
    }
    // Half a pixel: the profile is symmetric, so the centroid should land on
    // the pixel it was planted in. A systematic error in the conventions -- a
    // transposed frame, a missing half pixel -- is larger than this.
    near(where + " in x", best->position[0],
         static_cast<double>(want.fast) + 0.5, 0.5);
    near(where + " in y", best->position[1],
         static_cast<double>(want.slow) + 0.5, 0.5);
    // The rocking curve is symmetric about frame 2, whose pixel centre is 2.5.
    near(where + " in z", best->position[2], 2.5, 0.3);
    check(where + " spans three frames",
          static_cast<unsigned>(best->bbox[5] - best->bbox[4]), 3);
    // The centroid has to sit inside its own bounding box, which is a weaker
    // claim than the ones above and would catch a bbox built from the wrong
    // axis.
    check(where + " has its centroid inside its box",
          (best->position[0] >= best->bbox[0] &&
           best->position[0] <= best->bbox[1] &&
           best->position[1] >= best->bbox[2] &&
           best->position[1] <= best->bbox[3])
              ? 1
              : 0,
          1);
    check(where + " is brighter than its own pixel count",
          best->intensity > best->n_signal ? 1 : 0, 1);
  }

  // The weakest planted reflection is at risk of falling under the threshold
  // rather than of being mislocated, so it is worth saying that it was found at
  // all: 250 counts in its brightest pixel on a background of two is near the
  // bottom of what the dispersion test will call signal.
  {
    double weakest = 0.0;
    for (const dials_spots::Spot &spot : spots) {
      if (weakest == 0.0 || spot.intensity < weakest)
        weakest = spot.intensity;
    }
    std::printf("  the weakest spot found summed to %.0f counts\n", weakest);
    check("even the weakest planted reflection survives the filters",
          weakest > 0.0 ? 1 : 0, 1);
  }

  // And the table itself, which check_refl.py can then be pointed at.
  const std::string path = argc > 1 ? argv[1] : "test_dials_end_to_end.refl";
  refl::Options writing;
  writing.identifier = "end-to-end";
  refl::write(path, spots, labeller.pixels(), width, writing);
  std::printf("  wrote %s\n", path.c_str());
  if (argc <= 1)
    std::remove(path.c_str());

  std::printf("%s: frames to a reflection table, %d failures\n",
              failures == 0 ? "PASS" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
