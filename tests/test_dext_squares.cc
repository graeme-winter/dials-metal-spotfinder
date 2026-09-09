// Where the 32-bit window sums stop being exact, and how far that is from any
// frame this will see.
//
// The dispersion test needs the count, the sum and the sum of squares over a 7
// x 7 window, and all three are accumulated in accumulator_t<T> -- 32 bits for
// 16-bit pixels. The summed-area tables wrap over a frame and that is harmless,
// because the identity holds in Z / 2^32 as much as in Z, so differencing a
// wrapped table returns the true window value as long as the window value
// itself fits.
//
// For the count and the sum it always fits: 49 and 49 * 65535. For the sum of
// squares there is a limit, and this test is where it is written down:
//
//     one pixel alone            65535^2 = 4.295e9, just inside 2^32
//     two pixels alike           46341 counts each
//     the whole window uniform   9360 counts a pixel
//
// Nothing near that is a reflection. A spot with a thousand counts in its
// brightest pixel puts about 7e6 into the window that contains it, three orders
// below the limit; the busiest window this test can construct out of
// reflections -- two 2500-count spots two pixels apart, so that both cores and
// their sum land in one window -- reaches 1.4e8, which is still a factor of 30
// short. Getting *to* the limit takes a flat plateau of nine thousand counts a
// pixel across the whole window, which is not what diffraction looks like. The
// factor is measured here rather than asserted from the arithmetic, so that it
// is a number on the screen and not a claim in a comment.
//
// So this is not a latent bug and it is not to be "fixed" by widening the
// accumulator. Metal has no 64-bit integer arithmetic to widen it with, the
// tile variant's three threadgroup tables would grow by a third of the 32 KB an
// Apple threadgroup gets, and the 32-bit form is what has been run on real
// data.
//
// What the test is for is the assumption. Two changes would quietly eat the
// headroom -- a larger kernel, and anything that stopped excluding the masked
// pixels, since those are the only ones in a real frame that carry values near
// the top of the range. Both would fail here.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "dext.hh"
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

const std::size_t kHeight = 120;
const std::size_t kWidth = 160;
const std::int32_t kKernel = 3; // dext's, and the 7 x 7 window it makes

// The largest sum of squares any 7 x 7 window of this frame reaches, in 64 bits
// and clamped at the frame's edges exactly as dext clamps. Masked pixels
// contribute nothing, which is what dext does with them and is the whole reason
// a saturated spot cannot reach the limit.
std::uint64_t largest_window_square(const std::vector<std::uint16_t> &frame) {
  const std::uint16_t masked = 0xfffe; // numeric_limits<T>::max() - 1
  std::uint64_t largest = 0;
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(kHeight); i++) {
    for (std::int32_t j = 0; j < static_cast<std::int32_t>(kWidth); j++) {
      std::uint64_t total = 0;
      for (std::int32_t di = -kKernel; di <= kKernel; di++) {
        for (std::int32_t dj = -kKernel; dj <= kKernel; dj++) {
          const std::int32_t y = i + di;
          const std::int32_t x = j + dj;
          if (y < 0 || x < 0 || y >= static_cast<std::int32_t>(kHeight) ||
              x >= static_cast<std::int32_t>(kWidth))
            continue;
          const std::uint16_t pixel =
              frame[static_cast<std::size_t>(y) * kWidth +
                    static_cast<std::size_t>(x)];
          if (pixel >= masked)
            continue;
          total += static_cast<std::uint64_t>(pixel) * pixel;
        }
      }
      largest = total > largest ? total : largest;
    }
  }
  return largest;
}

// A Gaussian spot a pixel and a half wide, which is what the profile of a
// reflection on a pixel array detector looks like closely enough for this.
void plant(std::vector<std::uint16_t> *frame, std::size_t slow,
           std::size_t fast, double peak) {
  for (int di = -3; di <= 3; di++) {
    for (int dj = -3; dj <= 3; dj++) {
      const long i = static_cast<long>(slow) + di;
      const long j = static_cast<long>(fast) + dj;
      if (i < 0 || j < 0 || i >= static_cast<long>(kHeight) ||
          j >= static_cast<long>(kWidth))
        continue;
      const double radius = static_cast<double>(di * di + dj * dj);
      const double value = peak * std::exp(-radius / (2.0 * 1.5 * 1.5));
      const std::size_t k =
          static_cast<std::size_t>(i) * kWidth + static_cast<std::size_t>(j);
      const double total = frame->at(k) + value;
      (*frame)[k] =
          static_cast<std::uint16_t>(total > 65533.0 ? 65533.0 : total);
    }
  }
}

} // namespace

int main() {
  // The three ways to reach 2^32, as arithmetic rather than as prose. If the
  // kernel size ever changes, the third of these changes with it.
  const std::uint64_t limit = 1ull << 32;
  const std::uint64_t window = static_cast<std::uint64_t>(2 * kKernel + 1) *
                               static_cast<std::uint64_t>(2 * kKernel + 1);
  const std::uint64_t brightest = 65533; // the largest unmasked value
  check("one pixel at the top of the trusted range stays inside 2^32",
        brightest * brightest < limit ? 1 : 0, 1);
  const std::uint64_t uniform = static_cast<std::uint64_t>(
      std::sqrt(static_cast<double>(limit / window)));
  std::printf("  a uniform %llu-pixel window reaches 2^32 at %llu counts a "
              "pixel\n",
              static_cast<unsigned long long>(window),
              static_cast<unsigned long long>(uniform));
  check("which is far above anything a detector records for a reflection",
        uniform > 5000 ? 1 : 0, 1);

  // A frame of reflections at the brightnesses a real one carries: a few
  // hundred to a couple of thousand counts in the strongest pixel.
  {
    std::vector<std::uint16_t> frame(kHeight * kWidth, 2);
    const double peaks[] = {250.0,  400.0,  700.0,  1000.0, 1400.0,
                            1800.0, 2200.0, 2500.0, 3000.0};
    std::size_t at = 0;
    for (const double peak : peaks) {
      plant(&frame, 10 + 12 * at, 10 + 17 * at, peak);
      at++;
    }
    // Two spots overlapping, which puts two bright cores in one window and is
    // the worst case a frame of reflections offers.
    plant(&frame, 60, 100, 2500.0);
    plant(&frame, 60, 102, 2500.0);

    const std::uint64_t largest = largest_window_square(frame);
    std::printf("  the busiest window of a realistic frame reaches %llu, which "
                "is 2^32 / %.0f\n",
                static_cast<unsigned long long>(largest),
                static_cast<double>(limit) / static_cast<double>(largest));
    // An order of magnitude, with the factor itself printed above: the busiest
    // realistic window measures about 30x short of the limit, and the bar is
    // set at 10 so that this fails on a change that eats most of the margin
    // rather than only on one that eats all of it. This is the assumption the
    // 32-bit accumulator rests on, and the number to look at if the kernel size
    // or the masking ever changes.
    check("a realistic frame stays an order of magnitude clear of the limit",
          largest * 10 < limit ? 1 : 0, 1);

    // And the finder finds them, which is the other half: headroom is only
    // interesting if the spots are there to lose.
    std::vector<SignalPixel> found;
    check("the frame is accepted",
          dext<std::uint16_t>(frame.data(), found, kHeight, kWidth), 0);
    check("and it has signal on it", found.empty() ? 0 : 1, 1);
  }

  // Saturated pixels are excluded rather than accumulated, which is why the top
  // of the 16-bit range never reaches the window sums at all. A frame that is
  // nothing but the two sentinels has to sum to zero.
  {
    std::vector<std::uint16_t> frame(kHeight * kWidth, 0xffff);
    for (std::size_t k = 0; k < frame.size(); k += 2)
      frame[k] = 0xfffe;
    check("both masked sentinels contribute nothing to a window",
          largest_window_square(frame), 0);

    std::vector<SignalPixel> found;
    check("a wholly masked frame is accepted",
          dext<std::uint16_t>(frame.data(), found, kHeight, kWidth), 0);
    check("and yields no signal", found.size(), 0);
  }

  std::printf("%s: the 16-bit window sums, %d failures\n",
              failures == 0 ? "PASS" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
