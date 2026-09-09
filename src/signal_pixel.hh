// What the signal calculation emits, instead of a mask image.
//
// A frame is sparse: on a typical still, something like one per cent of pixels
// survive the finder. Writing a full mask and then scanning it to find them
// costs a pass over the whole frame at both ends, and on the device it costs a
// download of the entire frame -- 72 MB for a 16M detector at 16 bits -- to
// recover a hundredth of it. Emitting the survivors directly is smaller, and it
// carries the local background out with them, which the mask never could.
//
// Sixteen bytes, so a frame's worth of these fits in the buffer the mask used
// to occupy as long as fewer than one pixel in eight is signal at 16 bits, or
// one in four at 32. Both are far beyond anything a real frame produces; the
// device path checks the bound rather than trusting it.

#ifndef SPOTFINDER_SIGNAL_PIXEL_HH
#define SPOTFINDER_SIGNAL_PIXEL_HH

#include <cstdint>

struct SignalPixel {
  std::uint32_t index;      // row-major position in the frame
  std::uint32_t value;      // the raw pixel
  float background;         // the masked local mean at that pixel
  std::uint16_t population; // pixels that mean was averaged over
  std::uint16_t reserved;   // pack to 128 bits
};

static_assert(sizeof(SignalPixel) == 16, "SignalPixel must pack to 128 bits");

#endif // SPOTFINDER_SIGNAL_PIXEL_HH
