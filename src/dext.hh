#ifndef SPOTFINDER_DEXT_HH
#define SPOTFINDER_DEXT_HH

#include <cstddef>
#include <cstdint>
#include <vector>

#include "signal_pixel.hh"

// ---------------------------------------------------------------------------
// Type traits.
//
//   T is the pixel type (std::uint16_t or std::uint32_t)
//   W is the "wide" accumulator type used for the summed-area tables
//   R is the real type used for the dispersion / Poisson tests
// ---------------------------------------------------------------------------

template <typename T> struct accumulator;

template <> struct accumulator<std::uint16_t> {
  using type = std::uint32_t;
};
template <> struct accumulator<std::uint32_t> {
  using type = std::uint64_t;
};

template <typename T> using accumulator_t = typename accumulator<T>::type;

template <typename T> struct real_type;

template <> struct real_type<std::uint16_t> {
  using type = float;
};
template <> struct real_type<std::uint32_t> {
  using type = double;
};

template <typename T> using real_t = typename real_type<T>::type;

// ---------------------------------------------------------------------------
// Working buffers.
//
// Three summed-area tables and two byte masks, all height * width elements.
// Hold of these per thread and reuse it across frames: after the first call of
// a given size no allocation takes place.  Passing it explicitly keeps the
// routine free of hidden state.
// ---------------------------------------------------------------------------

template <typename T> struct dext_scratch {
  using W = accumulator_t<T>;

  std::vector<W> m_sat, i_sat, i2_sat;
  std::vector<std::uint8_t> scr0, scr1;

  // Grow to hold n elements.  Never shrinks, so a mixed stream of frame sizes
  // settles at the largest.
  void resize(std::size_t n) {
    if (m_sat.size() < n) {
      m_sat.resize(n);
      i_sat.resize(n);
      i2_sat.resize(n);
      scr0.resize(n);
      scr1.resize(n);
    }
  }
};

// Bytes a scratch will occupy for a given frame, for budgeting or logging.
template <typename T>
constexpr std::size_t dext_scratch_bytes(std::size_t height,
                                         std::size_t width) {
  return height * width *
         (3 * sizeof(accumulator_t<T>) + 2 * sizeof(std::uint8_t));
}

// ---------------------------------------------------------------------------
// Extended dispersion spot finder.
//
// image_in    height * width pixels, row major, read only and left untouched
// signal_out  cleared on entry, then filled with one SignalPixel per surviving
//             pixel, ascending by index. Reuse the same vector across frames
//             and it stops allocating after the first.
// scratch     working buffers, resized on entry if needed
//
// Pixels at or above numeric_limits<T>::max() - 1 are treated as masked.
//
// Returns 0 on success, -1 if image_in is null, if height or width is zero, or
// if height * width exceeds INT32_MAX (internal indices are int32_t).
//
// The frame is not written to. An earlier form of this wrote a mask image,
// which the caller then had to scan to find the survivors; emitting them
// directly removes that pass, carries the local background out with them, and
// leaves the raw frame intact for anything downstream that wants it.
// ---------------------------------------------------------------------------

template <typename T>
int dext(const T *image_in, std::vector<SignalPixel> &signal_out,
         std::size_t height, std::size_t width, dext_scratch<T> &scratch);

// Convenience form: allocates and discards a scratch on every call. Use the
// form above in a streaming loop.
template <typename T>
int dext(const T *image_in, std::vector<SignalPixel> &signal_out,
         std::size_t height, std::size_t width) {
  dext_scratch<T> scratch;
  return dext(image_in, signal_out, height, width, scratch);
}

extern template int dext<std::uint16_t>(const std::uint16_t *,
                                        std::vector<SignalPixel> &, std::size_t,
                                        std::size_t,
                                        dext_scratch<std::uint16_t> &);
extern template int dext<std::uint32_t>(const std::uint32_t *,
                                        std::vector<SignalPixel> &, std::size_t,
                                        std::size_t,
                                        dext_scratch<std::uint32_t> &);

#endif // SPOTFINDER_DEXT_HH
