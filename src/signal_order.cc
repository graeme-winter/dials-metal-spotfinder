#include <algorithm>

#include "signal_order.hh"

namespace signal_order {
namespace {

bool ascending(const SignalPixel &a, const SignalPixel &b) {
  return a.index < b.index;
}

// Below this, the bucket arrays cost more than they save and std::sort is the
// right answer. A dark frame produces tens of signal pixels and should not
// touch a 30,000-entry histogram to order them.
constexpr std::size_t kSmall = 1024;

} // namespace

void by_index(std::vector<SignalPixel> &pixels, scratch &work) {
  const std::size_t n = pixels.size();
  if (n < 2)
    return;
  if (n < kSmall) {
    std::sort(pixels.begin(), pixels.end(), ascending);
    return;
  }

  // The bucket width comes from the largest index actually present, not from
  // the frame size. An extra pass over the list is a fraction of what follows,
  // and it removes the only way a caller could get this wrong: an understated
  // bound would have indices landing outside the histogram, which is a heap
  // overflow rather than a slow sort. The first version of this took the frame
  // size as an argument and tests/test_signal_order.cc caught it.
  std::uint32_t largest = 0;
  for (const SignalPixel &pixel : pixels)
    largest = std::max(largest, pixel.index);

  // About four pixels a bucket: enough that the per-bucket sort is a handful of
  // comparisons, few enough that the histogram stays in cache. Derived from the
  // count, so a sparse frame does not pay for a histogram sized for a dense
  // one.
  unsigned shift = 0;
  while ((largest >> shift) > n / 4 + 1)
    shift++;
  const std::size_t buckets = (largest >> shift) + 1;

  // start[b] is where bucket b begins; one extra entry so that start[b + 1] is
  // always the end.
  work.start.assign(buckets + 1, 0u);
  for (const SignalPixel &pixel : pixels)
    work.start[(pixel.index >> shift) + 1]++;
  for (std::size_t b = 1; b <= buckets; b++)
    work.start[b] += work.start[b - 1];

  // A second cursor array rather than consuming start[], which is needed again
  // below to find each bucket's extent.
  work.at.assign(work.start.begin(), work.start.end() - 1);
  work.staged.resize(n);
  for (const SignalPixel &pixel : pixels)
    work.staged[work.at[pixel.index >> shift]++] = pixel;
  pixels.swap(work.staged);

  // The buckets are in order already; only their contents are not.
  for (std::size_t b = 0; b < buckets; b++) {
    const std::uint32_t from = work.start[b];
    const std::uint32_t to = work.start[b + 1];
    if (to - from > 1) {
      std::sort(pixels.begin() + static_cast<std::ptrdiff_t>(from),
                pixels.begin() + static_cast<std::ptrdiff_t>(to), ascending);
    }
  }
}

void by_index(std::vector<SignalPixel> &pixels) {
  scratch work;
  by_index(pixels, work);
}

} // namespace signal_order
