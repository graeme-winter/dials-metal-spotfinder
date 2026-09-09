// signal_order::by_index against std::sort, and how much faster it is.
//
// The correctness bar is exact agreement with std::sort on the index, over
// inputs chosen to break a bucket sort if it is going to be broken: counts
// either side of the threshold where it switches strategy, indices at 0 and at
// the last pixel of the frame, everything in one bucket, duplicates, and the
// input that broke an earlier version which trusted a caller-supplied bound.
//
// The timing at the end is not a pass or a fail. It is the number that
// justifies this file existing rather than a call to std::sort, so it is
// printed.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "signal_order.hh"

namespace {

int failures = 0;

void check(bool condition, const std::string &what) {
  if (condition) {
    std::printf("  ok   %s\n", what.c_str());
    return;
  }
  std::printf("  FAIL %s\n", what.c_str());
  failures++;
}

SignalPixel make(std::uint32_t index) {
  SignalPixel pixel;
  pixel.index = index;
  // Carried through so that a sort which reorders the index and not the rest of
  // the record is caught: the value is derived from the index.
  pixel.value = index * 7u + 1u;
  pixel.background = static_cast<float>(index % 1000u) + 0.5f;
  pixel.population = static_cast<std::uint16_t>(index % 121u);
  pixel.reserved = 0;
  return pixel;
}

// The emit order a device actually produces: runs of up to 32 consecutive
// columns of one row, each run ascending, the runs in group-completion order.
// Modelled rather than assumed -- by_index must not depend on it -- but it is
// the realistic case to measure.
std::vector<SignalPixel> emit_order(std::size_t height, std::size_t width,
                                    std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::size_t> row(0, height - 1);
  std::uniform_int_distribution<std::size_t> tile(0, width / 32 - 1);
  std::uniform_int_distribution<std::size_t> run(1, 12);

  std::vector<bool> taken(height * width, false);
  std::vector<SignalPixel> out;
  out.reserve(n);
  while (out.size() < n) {
    const std::size_t r = row(rng);
    const std::size_t t = tile(rng);
    const std::size_t length = run(rng);
    for (std::size_t c = 0; c < length && out.size() < n; c++) {
      const std::size_t k = r * width + t * 32 + c;
      if (k >= height * width || taken[k])
        continue;
      taken[k] = true;
      out.push_back(make(static_cast<std::uint32_t>(k)));
    }
  }
  return out;
}

bool same(const std::vector<SignalPixel> &a,
          const std::vector<SignalPixel> &b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t k = 0; k < a.size(); k++) {
    if (a[k].index != b[k].index || a[k].value != b[k].value ||
        a[k].population != b[k].population ||
        a[k].background != b[k].background)
      return false;
  }
  return true;
}

// Sorts a copy with std::sort and requires by_index to match it exactly.
void agrees(std::vector<SignalPixel> pixels, const std::string &what) {
  std::vector<SignalPixel> reference = pixels;
  std::stable_sort(reference.begin(), reference.end(),
                   [](const SignalPixel &a, const SignalPixel &b) {
                     return a.index < b.index;
                   });
  signal_order::scratch work;
  signal_order::by_index(pixels, work);
  check(same(pixels, reference), what);
}

} // namespace

int main() {
  const std::size_t height = 4362;
  const std::size_t width = 4148;
  const std::size_t frame = height * width;

  std::printf("signal_order::by_index against std::sort\n");

  // Nothing to do, and nothing to crash on.
  agrees({}, "empty");
  agrees({make(17)}, "one pixel");
  agrees({make(9), make(4)}, "two pixels, reversed");

  // Either side of the threshold where the strategy changes, so both branches
  // are exercised and the boundary itself is not a special case.
  for (std::size_t n : {size_t(1023), size_t(1024), size_t(1025)}) {
    agrees(emit_order(height, width, n, 3u),
           std::to_string(n) + " pixels, across the small/bucket threshold");
  }

  agrees(emit_order(height, width, 126002, 7u),
         "126002 pixels, a real frame's worth");
  agrees(emit_order(height, width, 600000, 11u),
         "600000 pixels, a very dense frame");

  // The extremes of the index range, which is where an off-by-one in the
  // bucket count would land.
  {
    std::vector<SignalPixel> pixels;
    for (std::size_t n = 0; n < 2000; n++)
      pixels.push_back(
          make(static_cast<std::uint32_t>(n % 2 ? frame - 1 - n / 2 : n / 2)));
    agrees(pixels, "indices at both ends of the frame");
  }

  // All in one bucket: the per-bucket sort does all the work and the histogram
  // none of it.
  {
    std::vector<SignalPixel> pixels;
    for (std::uint32_t n = 0; n < 5000; n++)
      pixels.push_back(make(4999u - n));
    agrees(pixels, "5000 consecutive indices, reversed");
  }

  // Duplicates. The finder does not emit them; tolerating them is cheaper than
  // a trap for whatever later might.
  {
    std::vector<SignalPixel> pixels;
    for (std::uint32_t n = 0; n < 3000; n++)
      pixels.push_back(make(n % 40u));
    agrees(pixels, "3000 pixels over 40 distinct indices");
  }

  // What used to be the frame_pixels footgun: there is no bound to understate
  // any more, but the case that broke it is kept as a regression.
  agrees(emit_order(height, width, 4000, 5u),
         "4000 pixels, once broken by an understated frame bound");

  // What this is for.
  std::printf("\nat %zu x %zu:\n", height, width);
  for (std::size_t n : {size_t(12000), size_t(126002), size_t(600000)}) {
    const std::vector<SignalPixel> source = emit_order(height, width, n, 7u);
    signal_order::scratch work;

    double mine = 1e9;
    double theirs = 1e9;
    for (int r = 0; r < 5; r++) {
      std::vector<SignalPixel> pixels = source;
      auto started = std::chrono::steady_clock::now();
      signal_order::by_index(pixels, work);
      mine = std::min(mine, std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count());

      pixels = source;
      started = std::chrono::steady_clock::now();
      std::sort(pixels.begin(), pixels.end(),
                [](const SignalPixel &a, const SignalPixel &b) {
                  return a.index < b.index;
                });
      theirs = std::min(theirs, std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count());
    }
    std::printf("  %7zu pixels: by_index %6.2f ms, std::sort %6.2f ms, "
                "%.1fx\n",
                n, mine, theirs, theirs / mine);
  }

  if (failures != 0) {
    std::printf("\n%d checks failed\n", failures);
    return 1;
  }
  std::printf("\nall checks passed\n");
  return 0;
}
