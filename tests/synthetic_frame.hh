// Synthetic frames for the device tests and the benchmark.
//
// Shared so that the two cannot drift: a benchmark that times a different
// frame from the one the correctness test compares is timing something nobody
// has checked.
//
// Sparse planted spots on a Poisson background, not diffraction. This tests
// the plumbing rather than the science, which is the same caveat the whole
// signal calculation carries.

#ifndef SPOTFINDER_TESTS_SYNTHETIC_FRAME_HH
#define SPOTFINDER_TESTS_SYNTHETIC_FRAME_HH

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace synthetic {

// A deterministic generator, so a failure is reproducible from the seed alone.
class Random {
public:
  explicit Random(std::uint64_t seed)
      : state_(seed * 6364136223846793005ull + 1) {}

  std::uint32_t next() {
    state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(state_ >> 33);
  }

  double uniform() { return static_cast<double>(next()) / 4294967296.0; }

  // Knuth, which is fine at these means and is worth the few lines: the
  // dispersion test is a test for variance above Poisson, so a background drawn
  // from anything wider flags the whole frame as signal and the comparison
  // stops testing the finder and starts testing the sort. The first version of
  // this test used a scaled uniform and found 509,855 signal pixels in a
  // 512 x 1030 frame -- more than the packed list can even hold.
  std::uint16_t counts(double mean) {
    const double limit = std::exp(-mean);
    double product = uniform();
    std::uint16_t k = 0;
    while (product > limit && k < 1000) {
      product *= uniform();
      k++;
    }
    return k;
  }

private:
  std::uint64_t state_;
};

struct Frame {
  std::string name;
  std::size_t height = 0;
  std::size_t width = 0;
  std::vector<std::uint16_t> pixels;
};

inline void plant(Frame *frame, std::size_t i, std::size_t j,
                  std::uint16_t peak) {
  // A three-by-three spot with a shoulder, clipped at the border rather than
  // skipped: a spot that runs off the edge is the case the halo gets wrong.
  static const int weight[5][5] = {{0, 1, 2, 1, 0},
                                   {1, 4, 8, 4, 1},
                                   {2, 8, 16, 8, 2},
                                   {1, 4, 8, 4, 1},
                                   {0, 1, 2, 1, 0}};
  for (int di = -2; di <= 2; di++) {
    for (int dj = -2; dj <= 2; dj++) {
      const long _i = static_cast<long>(i) + di;
      const long _j = static_cast<long>(j) + dj;
      if (_i < 0 || _j < 0 || _i >= static_cast<long>(frame->height) ||
          _j >= static_cast<long>(frame->width))
        continue;
      const std::size_t k = static_cast<std::size_t>(_i) * frame->width +
                            static_cast<std::size_t>(_j);
      const int w = weight[di + 2][dj + 2];
      const std::uint32_t value =
          frame->pixels[k] + static_cast<std::uint32_t>(peak) *
                                 static_cast<std::uint32_t>(w) / 16u;
      // Never up into the masked sentinels by accident: that is a separate case
      // and it is planted deliberately below.
      frame->pixels[k] =
          static_cast<std::uint16_t>(value > 0xfffdu ? 0xfffdu : value);
    }
  }
}

inline Frame make_frame(const std::string &name, std::size_t height,
                        std::size_t width, std::uint64_t seed,
                        double background, bool with_mask) {
  Frame frame;
  frame.name = name;
  frame.height = height;
  frame.width = width;
  frame.pixels.assign(height * width, 0);

  Random random(seed);
  for (std::size_t k = 0; k < height * width; k++)
    frame.pixels[k] = random.counts(background);

  // How much is planted has to scale with the frame. The packed list reuses a
  // buffer of one frame's pixels, so it holds one entry per eight of them, and
  // a small frame with a generous scatter of spots on it overflows that and
  // comes back as -2 -- which is the device behaving correctly and the test
  // being wrong. An earlier version of this planted 31 spots on a 32 x 32
  // frame and asked for 399 entries in a buffer that holds 128.
  const std::size_t area = height * width;

  if (height >= 48 && width >= 48) {
    // Spots on every border and in both far corners, where the tile halo and
    // the summed-area clamp have to agree about a window that runs off the
    // frame.
    plant(&frame, 0, 0, 4000);
    plant(&frame, 0, width / 2, 3000);
    plant(&frame, height - 1, width - 1, 4000);
    plant(&frame, height / 2, 0, 3500);
    plant(&frame, height / 2, width - 1, 3500);
    plant(&frame, 1, 1, 2500);
    plant(&frame, height - 2, 2, 2500);

    // And a scatter through the middle, plus two that overlap, which is the
    // case where the second-pass window erodes away most of its background.
    const std::size_t scattered = area / 3000;
    for (std::size_t n = 0; n < scattered; n++) {
      const std::size_t i = 6 + random.next() % (height - 13);
      const std::size_t j = 6 + random.next() % (width - 13);
      plant(&frame, i, j,
            static_cast<std::uint16_t>(800 + random.next() % 6000));
    }
    plant(&frame, height / 3, width / 3, 5000);
    plant(&frame, height / 3, width / 3 + 2, 5000);
  } else {
    // Too small to hold much, so plant only what the frame is here to test:
    // opposite corners, both of which run their window off two edges at once.
    plant(&frame, 0, 0, 4000);
    plant(&frame, height - 1, width - 1, 4000);
  }

  if (with_mask) {
    // A dead module and a scatter of hot pixels, at both sentinels: 0xfffe is
    // the threshold and 0xffff is above it, and a backend that tested for
    // equality rather than for >= would pass on one and fail on the other.
    for (std::size_t i = height / 4; i < height / 4 + 5 && i < height; i++)
      for (std::size_t j = 0; j < width; j++)
        frame.pixels[i * width + j] = 0xfffe;
    for (int n = 0; n < 20; n++) {
      const std::size_t k = random.next() % (height * width);
      frame.pixels[k] = 0xffff;
    }
  }
  return frame;
}

} // namespace synthetic

#endif // SPOTFINDER_TESTS_SYNTHETIC_FRAME_HH
