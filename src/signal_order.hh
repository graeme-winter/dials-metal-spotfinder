// Putting a device's signal pixels into ascending index order.
//
// Every device backend needs this and neither should have its own: the emit is
// compacted per SIMD group, so the list comes back in whatever order the groups
// finished, and `spots::merge_labelled` needs it ascending -- its
// left-neighbour test is only O(1) because `k - 1` can only be the entry
// immediately before.
//
// std::sort was doing this and it was the single most expensive thing in
// gpu::find(). Measured at 4362 x 4148 with 126,002 signal pixels, which is a
// real frame's worth:
//
//     std::sort               8.78 ms
//     this                    1.61 ms
//
// and the gap widens with density, because a comparison sort cannot use any of
// the structure and this is O(n): at 600,000 pixels std::sort takes 49 ms
// against 9.6. A frame that dense is unusual but it is exactly the frame where
// spending 49 ms on a sort would be worst.
//
// The structure it uses: the flat index is bounded by the frame, so one pass
// over the high bits of the index puts every pixel in a bucket of consecutive
// indices, and the buckets are then in order by construction. Each bucket holds
// about four pixels, so finishing them off is a handful of comparisons apiece.
// No division -- the bucket is a shift.
//
// It does not depend on the emit order being anything in particular, which is
// deliberate: the order is a property of the threadgroup shape and the
// scheduler, and a sort that broke when either changed would break quietly.

#ifndef SPOTFINDER_SIGNAL_ORDER_HH
#define SPOTFINDER_SIGNAL_ORDER_HH

#include <cstddef>
#include <cstdint>
#include <vector>

#include "signal_pixel.hh"

namespace signal_order {

// Working buffers. Hold one per thread and reuse it across frames: after the
// first frame of a given size nothing is allocated. Passing it explicitly keeps
// the routine free of hidden state, as dext_scratch does.
struct scratch {
  std::vector<SignalPixel> staged;
  std::vector<std::uint32_t> start;
  std::vector<std::uint32_t> at;
};

// Sorts `pixels` ascending by `index`, in place.
//
// The bucket width is taken from the largest index present rather than from the
// frame size, so there is no bound to pass and no way to pass a wrong one. The
// first version of this took height * width as an argument; an understated
// value put indices outside the histogram, which is a heap overflow and not a
// slow sort. tests/test_signal_order.cc caught it, and the argument went.
//
// Duplicated indices are tolerated and end up adjacent. The finder does not
// produce them: a pixel is emitted at most once. Relying on that here would buy
// nothing and would be a trap for anything that later did.
void by_index(std::vector<SignalPixel> &pixels, scratch &work);

// Convenience form, allocating and discarding a scratch. Use the form above in
// a streaming loop.
void by_index(std::vector<SignalPixel> &pixels);

} // namespace signal_order

#endif // SPOTFINDER_SIGNAL_ORDER_HH
