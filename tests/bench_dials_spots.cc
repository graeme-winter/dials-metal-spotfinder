// What the grouping costs, against how many signal pixels a frame has and how
// many frames a reflection spans.
//
// This exists because I talked myself into rewriting the grouping to be faster,
// having never measured what density a real frame has. It is worth writing the
// arithmetic down. A dials.find_spots run on a real sweep reports:
//
//     Extracted 194150 spots
//     Removed 48006 spots with size < 3 pixels
//     Calculated 146143 spot centroids
//
// 194150 connected components over the whole sweep. Over 600 frames that is 324
// a frame; over 3600, 54. At twenty-five pixels a spot it is somewhere between
// one and six thousand signal pixels a frame -- and at that density the
// grouping takes between 0.05 and 0.6 ms of a frame whose threshold takes 5 ms
// on a GPU and 450 on one CPU core. One per cent of it at the very worst.
//
// The rewrite was real: accumulate each component's moments as pixels arrive
// instead of keeping the pixels and relabelling them every frame, so that a
// frame costs what its own pixels cost rather than what is open. It measured
// 2.2x faster at 66,000 signal pixels a frame with reflections spanning eight
// frames, and it was depth-independent where this is not. It was also a
// component pool with a free list and a two-generation recycler, and it bought
// nothing at all at the density the instrument produces. So it is not here, and
// this is, so that the next person to have the idea can start from the numbers.
//
// Run it if the density assumption ever changes -- a much finer slicing, an ice
// ring, a threshold set too low, a detector with ten times the pixels. The
// shape to watch for is the third column falling as the span grows, which is
// this design paying for what is open rather than for what arrived.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "dials_spots.hh"
#include "signal_pixel.hh"

namespace {

using Clock = std::chrono::steady_clock;

// A real detector, so that the indices are spread over as many pixels as they
// would be: the grouping walks them with monotonic pointers, and how far apart
// they are matters.
constexpr std::size_t kHeight = 4362;
constexpr std::size_t kWidth = 4148;
constexpr int kFrames = 40;

struct Result {
  std::size_t pixels_a_frame = 0;
  double milliseconds = 0.0;
  std::size_t live = 0;
  std::size_t spots = 0;
};

// Blobs of nineteen pixels -- a plausible reflection footprint -- each spanning
// `span` consecutive frames, scattered over the detector.
//
// The values follow a profile rather than being flat, so that the brightest
// pixel is at the middle of the blob and at the middle of its rocking curve.
// Flat blobs put the peak in a corner, the peak-centroid filter then rejects
// almost everything, and the timing measures the filter instead of the emit
// path it is supposed to include.
Result run(std::size_t pixels_a_frame, int span, int repeats) {
  const std::size_t blobs = pixels_a_frame * kFrames / (19 * span);
  std::mt19937 rng(7);
  std::uniform_int_distribution<std::size_t> slow(8, kHeight - 9);
  std::uniform_int_distribution<std::size_t> fast(8, kWidth - 9);

  std::vector<std::vector<SignalPixel>> frames(kFrames);
  for (std::size_t blob = 0; blob < blobs; blob++) {
    const std::size_t y = slow(rng), x = fast(rng);
    const int first = static_cast<int>(blob % (kFrames - span));
    for (int step = 0; step < span; step++) {
      // Symmetric about the middle of the span, so the z centroid lands on the
      // brightest frame.
      const double from_middle = static_cast<double>(step) - (span - 1) / 2.0;
      const double rocking = std::exp(-from_middle * from_middle / 4.0);
      for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
          if (dy * dy + dx * dx > 5)
            continue;
          const double radial =
              std::exp(-static_cast<double>(dy * dy + dx * dx) / 4.0);
          const std::uint32_t value =
              50 + static_cast<std::uint32_t>(1500.0 * rocking * radial);
          frames[first + step].push_back(
              {static_cast<std::uint32_t>((y + dy) * kWidth + x + dx), value,
               0.0F, 0, 0});
        }
      }
    }
  }

  // Ascending by index and no duplicates, which is what both finders emit and
  // what the grouping requires.
  Result result;
  for (std::vector<SignalPixel> &frame : frames) {
    std::sort(frame.begin(), frame.end(),
              [](const SignalPixel &a, const SignalPixel &b) {
                return a.index < b.index;
              });
    frame.erase(std::unique(frame.begin(), frame.end(),
                            [](const SignalPixel &a, const SignalPixel &b) {
                              return a.index == b.index;
                            }),
                frame.end());
    result.pixels_a_frame += frame.size();
  }
  result.pixels_a_frame /= kFrames;

  // Best of `repeats`, not the mean: this is a timing on a shared machine, and
  // the fastest run is the one least interfered with. Taking means here had me
  // reading a 20% variance as a real difference between two implementations.
  double best = 0.0;
  for (int repeat = 0; repeat < repeats; repeat++) {
    dials_spots::Labeller labeller(kHeight, kWidth, dials_spots::Options{});
    std::size_t high = 0;
    const Clock::time_point began = Clock::now();
    for (int frame = 0; frame < kFrames; frame++) {
      labeller.add(frame, frames[frame]);
      high = labeller.live() > high ? labeller.live() : high;
    }
    labeller.finish();
    const double seconds =
        std::chrono::duration<double>(Clock::now() - began).count();
    if (repeat == 0 || seconds < best)
      best = seconds;
    result.live = high;
    result.spots = labeller.spots().size();
  }
  result.milliseconds = 1e3 * best / kFrames;
  return result;
}

} // namespace

int main(int argc, char **argv) {
  const int repeats = argc > 1 ? std::atoi(argv[1]) : 5;

  std::printf("%zu x %zu, %d frames, best of %d\n\n", kHeight, kWidth, kFrames,
              repeats);
  std::printf("%12s %6s %10s %9s %12s %8s\n", "pixels/frame", "span",
              "ms/frame", "Mpixel/s", "live pixels", "spots");

  for (const std::size_t density : {1000u, 6000u, 30000u, 130000u}) {
    for (const int span : {1, 3, 10}) {
      const Result result = run(density, span, repeats);
      std::printf("%12zu %6d %10.3f %9.1f %12zu %8zu\n", result.pixels_a_frame,
                  span, result.milliseconds,
                  result.pixels_a_frame / result.milliseconds / 1e3,
                  result.live, result.spots);
    }
  }

  std::printf("\nA real sweep sits in the first two rows: 194150 components "
              "over 600 to 3600\nframes is 54 to 324 a frame. Against a "
              "threshold taking 5 ms a frame on a\nGPU, the grouping there "
              "costs one per cent of it at the very worst.\n");
  return 0;
}
