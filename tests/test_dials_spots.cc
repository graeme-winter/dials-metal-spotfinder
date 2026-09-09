// The three-dimensional grouping, and the numbers DIALS gets from it.
//
// Small cases, written out by hand, with the expected centroids and variances
// worked through in the comments rather than copied from a run of this code --
// a test that records what the implementation does is not a test of whether it
// is right.
//
// The cases that matter: a spot that spans frames is one spot; a frame that is
// missing breaks the connection; the last pixel of a row does not touch the
// first pixel of the next; a component that is still growing is not emitted
// early; and the two filters dials.find_spots applies by default do what they
// say.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "dials_spots.hh"
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

void close(const std::string &what, double actual, double expected) {
  // Everything here is arithmetic on small integers, so agreement should be to
  // the last few bits; the tolerance is against the order of the summation
  // rather than against any real uncertainty.
  if (std::fabs(actual - expected) < 1e-9)
    return;
  std::printf("FAIL %s: %.12f, expected %.12f\n", what.c_str(), actual,
              expected);
  failures++;
}

std::vector<SignalPixel>
pixels(const std::vector<std::pair<std::uint32_t, std::uint32_t>> &at) {
  std::vector<SignalPixel> result;
  for (const auto &pair : at)
    result.push_back({pair.first, pair.second, 0.0F, 0, 0});
  return result;
}

// Every pixel at one value, which is what most of the connectivity cases want.
std::vector<SignalPixel> flat(const std::vector<std::uint32_t> &indices,
                              std::uint32_t value) {
  std::vector<SignalPixel> result;
  for (const std::uint32_t index : indices)
    result.push_back({index, value, 0.0F, 0, 0});
  return result;
}

dials_spots::Options plain() {
  dials_spots::Options options;
  options.max_separation = 0.0; // one thing at a time
  return options;
}

} // namespace

int main() {
  const std::size_t height = 8;
  const std::size_t width = 10;

  // -------------------------------------------------------------------------
  // Three pixels in a row on one frame, with values 1, 2, 1 at x = 0, 1, 2.
  //
  //   sum = 4, sum of squares = 6
  //   x: (0.5 + 2 * 1.5 + 2.5) / 4 = 1.5
  //      delta^2 = 1 * 1 + 2 * 0 + 1 * 1 = 2
  //      unbiased variance = 2 * 4 / (16 - 6) = 0.8
  //      standard error squared = 0.8 / 4 + 1/12 = 0.283333...
  //   y: every pixel on row 0, so 0.5 with no spread: 0 + 1/12
  //   z: one frame, so 0.5 by definition, and 1/12
  // -------------------------------------------------------------------------
  {
    dials_spots::Labeller labeller(height, width, plain());
    labeller.add(0, pixels({{0, 1}, {1, 2}, {2, 1}}));
    labeller.finish();
    const std::vector<dials_spots::Spot> &spots = labeller.spots();
    check("one spot from three pixels in a row", spots.size(), 1);
    if (spots.size() == 1) {
      const dials_spots::Spot &spot = spots[0];
      check("its pixel count", spot.n_signal, 3);
      close("its intensity", spot.intensity, 4.0);
      close("its variance, which is the intensity", spot.intensity_variance,
            4.0);
      close("centroid x", spot.position[0], 1.5);
      close("centroid y", spot.position[1], 0.5);
      close("centroid z", spot.position[2], 0.5);
      close("variance x", spot.variance[0], 0.8 / 4.0 + 1.0 / 12.0);
      close("variance y", spot.variance[1], 1.0 / 12.0);
      close("variance z", spot.variance[2], 1.0 / 12.0);
      check("bbox x0", static_cast<unsigned>(spot.bbox[0]), 0);
      check("bbox x1", static_cast<unsigned>(spot.bbox[1]), 3);
      check("bbox y0", static_cast<unsigned>(spot.bbox[2]), 0);
      check("bbox y1", static_cast<unsigned>(spot.bbox[3]), 1);
      check("bbox z0", static_cast<unsigned>(spot.bbox[4]), 0);
      check("bbox z1", static_cast<unsigned>(spot.bbox[5]), 1);
      check("pixels kept for the shoebox", labeller.pixels().size(), 3);
    }
    check("groups seen", labeller.counts().groups, 1);
    check("nothing rejected", labeller.counts().too_small, 0);
  }

  // -------------------------------------------------------------------------
  // The same three pixels on three consecutive frames: one spot in 3D, three
  // in 2D. This is the whole reason this file exists.
  //
  // In 3D, with all nine pixels at value 1: sum = 9, sum of squares = 9,
  // so sum^2 - sum_sq = 72 and the unbiased variance in x is
  // (4 * 1 + 0 + 4 * 1) ... per frame there are 3 pixels at x = 0.5, 1.5, 2.5,
  // over three frames, so delta_x^2 = 6 * 1 = 6 and variance = 6 * 9 / 72 =
  // 0.75. In z the same shape: 6 * 9 / 72 = 0.75.
  // -------------------------------------------------------------------------
  {
    dials_spots::Labeller labeller(height, width, plain());
    for (std::int64_t frame = 0; frame < 3; frame++)
      labeller.add(frame, flat({0, 1, 2}, 1));
    labeller.finish();
    check("three frames of one streak are one spot", labeller.spots().size(),
          1);
    if (labeller.spots().size() == 1) {
      const dials_spots::Spot &spot = labeller.spots()[0];
      check("nine pixels", spot.n_signal, 9);
      close("intensity", spot.intensity, 9.0);
      close("centroid x", spot.position[0], 1.5);
      close("centroid z", spot.position[2], 1.5);
      close("variance x", spot.variance[0], 0.75 / 9.0 + 1.0 / 12.0);
      close("variance z", spot.variance[2], 0.75 / 9.0 + 1.0 / 12.0);
      check("bbox z0", static_cast<unsigned>(spot.bbox[4]), 0);
      check("bbox z1", static_cast<unsigned>(spot.bbox[5]), 3);
    }
  }
  {
    dials_spots::Options options = plain();
    options.two_d = true;
    dials_spots::Labeller labeller(height, width, options);
    for (std::int64_t frame = 0; frame < 3; frame++)
      labeller.add(frame, flat({0, 1, 2}, 1));
    labeller.finish();
    check("and three spots in two dimensions", labeller.spots().size(), 3);
    if (labeller.spots().size() == 3) {
      close("the second one is on frame 1", labeller.spots()[1].position[2],
            1.5);
      check("with a one-frame box",
            static_cast<unsigned>(labeller.spots()[1].bbox[5] -
                                  labeller.spots()[1].bbox[4]),
            1);
    }
  }

  // -------------------------------------------------------------------------
  // A frame in the middle that was never written. Nothing can be connected
  // across it, so the streak is two spots and not one.
  // -------------------------------------------------------------------------
  {
    dials_spots::Labeller labeller(height, width, plain());
    labeller.add(0, flat({0, 1, 2}, 1));
    labeller.add(1, flat({0, 1, 2}, 1));
    labeller.add(3, flat({0, 1, 2}, 1));
    labeller.add(4, flat({0, 1, 2}, 1));
    labeller.finish();
    check("a missing frame breaks the connection", labeller.spots().size(), 2);
    if (labeller.spots().size() == 2) {
      check("the first pair", labeller.spots()[0].n_signal, 6);
      check("the second pair", labeller.spots()[1].n_signal, 6);
      check("and they are on different frames",
            static_cast<unsigned>(labeller.spots()[1].bbox[4]), 3);
    }
  }

  // -------------------------------------------------------------------------
  // Connectivity edges, all on one frame. Index 9 is the last pixel of row 0
  // and index 10 the first of row 1, which are adjacent in the flat index and
  // on opposite sides of the detector.
  // -------------------------------------------------------------------------
  {
    dials_spots::Options options = plain();
    options.min_spot_size = 1;
    dials_spots::Labeller labeller(height, width, options);
    labeller.add(0, flat({9, 10}, 1));
    labeller.finish();
    check("a row wrap is not an edge", labeller.spots().size(), 2);
  }
  {
    dials_spots::Options options = plain();
    options.min_spot_size = 1;
    dials_spots::Labeller labeller(height, width, options);
    labeller.add(0, flat({0, 10}, 1)); // (0,0) and (1,0)
    labeller.finish();
    check("the pixel above row one is an edge", labeller.spots().size(), 1);
  }
  {
    // Diagonal only: six-connected, so these are two spots, where a
    // twenty-six-connected grouping would call them one.
    dials_spots::Options options = plain();
    options.min_spot_size = 1;
    dials_spots::Labeller labeller(height, width, options);
    labeller.add(0, flat({0, 11}, 1)); // (0,0) and (1,1)
    labeller.finish();
    check("a diagonal is not an edge", labeller.spots().size(), 2);
  }
  {
    // The same, across frames: (0, y, x) and (1, y+1, x) do not touch.
    dials_spots::Options options = plain();
    options.min_spot_size = 1;
    dials_spots::Labeller labeller(height, width, options);
    labeller.add(0, flat({0}, 1));
    labeller.add(1, flat({10}, 1));
    labeller.finish();
    check("a diagonal in z is not an edge either", labeller.spots().size(), 2);
  }

  // -------------------------------------------------------------------------
  // Nothing is emitted before it is finished. After frame 0 the streak might
  // still grow, so the table is empty until finish().
  // -------------------------------------------------------------------------
  {
    dials_spots::Labeller labeller(height, width, plain());
    labeller.add(0, flat({0, 1, 2}, 1));
    check("an open component is not emitted yet", labeller.spots().size(), 0);
    check("and its pixels are still held", labeller.live(), 3);
    labeller.add(1, flat({40, 41, 42}, 1)); // somewhere else: frame 0 closes
    check("a component with no pixel on the new frame is emitted",
          labeller.spots().size(), 1);
    check("and only the new frame is held", labeller.live(), 3);
    labeller.finish();
    check("then the rest", labeller.spots().size(), 2);
    check("nothing left held", labeller.live(), 0);
  }

  // -------------------------------------------------------------------------
  // The size filters, which are dials.find_spots' min_spot_size and
  // max_spot_size.
  // -------------------------------------------------------------------------
  {
    dials_spots::Labeller labeller(height, width, plain());
    labeller.add(0, flat({0, 1}, 1));       // two: too small
    labeller.add(1, flat({40, 41, 42}, 1)); // three: kept
    labeller.finish();
    check("two pixels are noise", labeller.spots().size(), 1);
    check("counted as too small", labeller.counts().too_small, 1);
    check("groups seen", labeller.counts().groups, 2);
    check("signal pixels handed in", labeller.counts().signal_pixels, 5);
  }
  {
    dials_spots::Options options = plain();
    options.max_spot_size = 5;
    dials_spots::Labeller labeller(height, width, options);
    labeller.add(0, flat({0, 1, 2, 3, 4, 5}, 1));
    labeller.finish();
    check("six pixels is too large for a limit of five",
          labeller.spots().size(), 0);
    check("counted as too large", labeller.counts().too_large, 1);
  }

  // -------------------------------------------------------------------------
  // The peak-to-centroid filter, which is on by default at two pixels. A long
  // faint tail with one bright pixel at the end pulls the two apart.
  //
  // Values 100, 1, 1, 1, 1, 1 at x = 0..5: sum = 105, weighted x =
  // 100 * 0.5 + 1.5 + 2.5 + 3.5 + 4.5 + 5.5 = 67.5, so the centroid is at
  // 0.642857 and the peak at 0.5 -- 0.14 apart, which passes. Turn it round,
  // with the bright pixel at x = 5, and the centroid is at 5.357 with the peak
  // at 5.5: also fine. So the case has to be a bright pixel away from a heavy
  // cluster: 1 at x = 0 and 100 each at x = 4, 5 gives a centroid at
  // (0.5 + 450 + 550) / 201 = 4.98 and a peak at 4.5. Still under two.
  //
  // The filter is hard to trip with a real spot, which is the point of it: it
  // catches shoeboxes where the brightest pixel is nowhere near the centre of
  // mass, as a merged pair of reflections or a cosmic ray track gives. A track
  // does it -- one bright pixel and a long faint line away from it.
  // -------------------------------------------------------------------------
  {
    dials_spots::Options options;
    options.max_separation = 2.0;
    dials_spots::Labeller labeller(height, width, options);
    // x = 0 bright, then a faint tail out to x = 9 on row 0.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> at{{0, 1000}};
    for (std::uint32_t x = 1; x < 10; x++)
      at.push_back({x, 1});
    labeller.add(0, pixels(at));
    labeller.finish();
    // sum = 1009, weighted x = 500 + (1.5 + ... + 9.5) = 500 + 49.5 = 549.5,
    // centroid 0.5446, peak 0.5: 0.045 apart. Not rejected.
    check("a bright head with a faint tail is kept", labeller.spots().size(),
          1);
    check("nothing rejected by separation", labeller.counts().separated, 0);
  }
  {
    dials_spots::Options options;
    options.max_separation = 2.0;
    dials_spots::Labeller labeller(height, width, options);
    // Two equal lumps four pixels apart, joined by a single faint pixel: the
    // centroid lands between them and the peak is in one of them.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> at{
        {0, 100}, {1, 100}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 90}, {7, 90}};
    labeller.add(0, pixels(at));
    labeller.finish();
    // sum = 384; weighted x = 100*0.5 + 100*1.5 + 2.5 + 3.5 + 4.5 + 5.5 +
    // 90*6.5 + 90*7.5 = 50 + 150 + 16 + 585 + 675 = 1476, centroid 3.844.
    // The peak is the first of the 100s, at 0.5, which is 3.34 away.
    check("a merged pair is rejected by separation", labeller.spots().size(),
          0);
    check("and counted", labeller.counts().separated, 1);
    check("having been a group", labeller.counts().groups, 1);
  }

  // -------------------------------------------------------------------------
  // A single pixel, where DIALS' unbiased variance divides by zero and its own
  // code catches the assertion. Reproduced: no variance, and 1/12 all round.
  // -------------------------------------------------------------------------
  {
    dials_spots::Options options = plain();
    options.min_spot_size = 1;
    dials_spots::Labeller labeller(height, width, options);
    labeller.add(0, flat({35}, 700));
    labeller.finish();
    check("a single pixel can be a spot when asked for",
          labeller.spots().size(), 1);
    if (labeller.spots().size() == 1) {
      const dials_spots::Spot &spot = labeller.spots()[0];
      close("centroid x", spot.position[0], 5.5); // 35 = row 3, column 5
      close("centroid y", spot.position[1], 3.5);
      close("variance x is the fallback", spot.variance[0], 1.0 / 12.0);
      close("variance y is the fallback", spot.variance[1], 1.0 / 12.0);
      close("intensity", spot.intensity, 700.0);
    }
  }

  // -------------------------------------------------------------------------
  // Frames out of order, and the same frame twice, are refused rather than
  // silently mis-grouped: the whole streaming argument rests on the order.
  // -------------------------------------------------------------------------
  {
    dials_spots::Labeller labeller(height, width, plain());
    labeller.add(5, flat({0, 1, 2}, 1));
    bool threw = false;
    try {
      labeller.add(4, flat({0}, 1));
    } catch (const std::exception &) {
      threw = true;
    }
    check("a frame out of order throws", threw ? 1 : 0, 1);
    threw = false;
    try {
      labeller.add(5, flat({0}, 1));
    } catch (const std::exception &) {
      threw = true;
    }
    check("the same frame twice throws", threw ? 1 : 0, 1);
  }
  {
    dials_spots::Labeller labeller(height, width, plain());
    bool threw = false;
    try {
      labeller.add(0, flat({2, 1}, 1)); // descending
    } catch (const std::exception &) {
      threw = true;
    }
    check("pixels out of index order throw", threw ? 1 : 0, 1);
  }
  {
    dials_spots::Labeller labeller(height, width, plain());
    bool threw = false;
    try {
      labeller.add(0, flat({static_cast<std::uint32_t>(height * width)}, 1));
    } catch (const std::exception &) {
      threw = true;
    }
    check("a pixel off the frame throws", threw ? 1 : 0, 1);
  }

  // -------------------------------------------------------------------------
  // A long streak through many frames, to exercise the compaction: the pixels
  // of an open component are carried forward frame after frame, and the
  // component's root has to survive being renumbered every time.
  // -------------------------------------------------------------------------
  {
    dials_spots::Labeller labeller(height, width, plain());
    const int frames = 200;
    for (std::int64_t frame = 0; frame < frames; frame++) {
      // The streak, plus a three-pixel spot that closes on the next frame, so
      // that there is always something to flush as well as something to keep.
      labeller.add(frame, flat({44, 45, 46, 70, 71, 72}, 3));
    }
    labeller.finish();
    // Two components: the streak at 44-46 through every frame, and the one at
    // 70-72, which is also connected in z. So two, each 3 * 200 pixels.
    check("two streaks", labeller.spots().size(), 2);
    if (labeller.spots().size() == 2) {
      check("the first is whole", labeller.spots()[0].n_signal,
            static_cast<unsigned>(3 * frames));
      check("and spans every frame",
            static_cast<unsigned>(labeller.spots()[0].bbox[5] -
                                  labeller.spots()[0].bbox[4]),
            static_cast<unsigned>(frames));
    }
    check("all the pixels were kept", labeller.pixels().size(),
          static_cast<unsigned>(6 * frames));
  }

  std::printf("%s: DIALS-compatible grouping, %d failures\n",
              failures == 0 ? "PASS" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
