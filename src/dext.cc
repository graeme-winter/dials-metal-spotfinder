#include <cmath>
#include <limits>

#include "dext.hh"

#pragma region DISPERSION EXTENDED

template <typename T>
int dext(const T *image_in, std::vector<SignalPixel> &signal_out,
         std::size_t height, std::size_t width, dext_scratch<T> &scratch) {
  using W = accumulator_t<T>; // wide accumulator
  using R = real_t<T>; // float for std::uint16_t, double for std::uint32_t

  if (image_in == nullptr || height == 0 || width == 0)
    return -1;

  // Every internal index is std::int32_t, so the frame must fit that.
  const std::size_t i32max =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  if (height > i32max / width)
    return -1;

  const std::size_t n = height * width;
  if (n > i32max)
    return -1;

  scratch.resize(n);
  // Cleared rather than appended to, so a caller can hand back the same vector
  // frame after frame and keep its capacity.
  signal_out.clear();

  // Signed extents so every index expression below stays in std::int32_t and no
  // signed/unsigned comparison warnings are emitted.
  const std::int32_t ny = static_cast<std::int32_t>(height);
  const std::int32_t nx = static_cast<std::int32_t>(width);

  std::int32_t knl = 3;

  const R sigma_b = R(6.0), sigma_s = R(3.0);
  const T masked = static_cast<T>(std::numeric_limits<T>::max() - 1);

#pragma region SAT ACCUMULATE INITAL THRESHOLD

  // Index through pointers throughout.  Pointer arithmetic takes a signed
  // ptrdiff_t, so the std::int32_t loop counters need no cast and no
  // -Wsign-conversion warning is produced.  The scratch owns the storage.
  W *const m_sat = scratch.m_sat.data();
  W *const i_sat = scratch.i_sat.data();
  W *const i2_sat = scratch.i2_sat.data();
  std::uint8_t *const scr0 = scratch.scr0.data();
  std::uint8_t *const scr1 = scratch.scr1.data();

  for (std::int32_t i = 0, k = 0; i < ny; i++) {
    W _m = 0, _i = 0, _i2 = 0;
    for (std::int32_t j = 0; j < nx; j++, k++) {
      const bool valid = image_in[k] < masked;
      const W p = valid ? static_cast<W>(image_in[k]) : W(0);
      _m += valid ? W(1) : W(0);
      _i += p;
      _i2 += p * p; // W arithmetic: no promotion to int
      m_sat[k] = i > 0 ? _m + m_sat[k - nx] : _m;
      i_sat[k] = i > 0 ? _i + i_sat[k - nx] : _i;
      i2_sat[k] = i > 0 ? _i2 + i2_sat[k - nx] : _i2;
    }
  }

#pragma region INITIAL THRESHOLD

  for (std::int32_t i = 0, k = 0; i < ny; i++) {
    for (std::int32_t j = 0; j < nx; j++, k++) {
      std::int32_t j0 = j - knl - 1;
      std::int32_t j1 = j < (nx - knl) ? j + knl : nx - 1;
      std::int32_t i0 = i - knl - 1;
      std::int32_t i1 = i < (ny - knl) ? i + knl : ny - 1;

      std::int32_t a = i1 * nx + j1;
      std::int32_t b = i0 * nx + j1;
      std::int32_t c = i1 * nx + j0;
      std::int32_t d = i0 * nx + j0;

      W m_sum = m_sat[a], i_sum = i_sat[a], i2_sum = i2_sat[a];

      if (j0 >= 0 && i0 >= 0) {
        m_sum += m_sat[d] - m_sat[b] - m_sat[c];
        i_sum += i_sat[d] - i_sat[b] - i_sat[c];
        i2_sum += i2_sat[d] - i2_sat[b] - i2_sat[c];
      } else if (j0 >= 0) {
        m_sum -= m_sat[c];
        i_sum -= i_sat[c];
        i2_sum -= i2_sat[c];
      } else if (i0 >= 0) {
        m_sum -= m_sat[b];
        i_sum -= i_sat[b];
        i2_sum -= i2_sat[b];
      }

      bool signal = false;

      if (m_sum >= 2) {
        // n * sum(i^2) - sum(i)^2 - (n - 1) * sum(i)
        //     > sigma_b * sum(i) * sqrt(2 * (n - 1))
        // Evaluated entirely in R: the products overflow W for bright windows.
        const R m = static_cast<R>(m_sum);
        const R s = static_cast<R>(i_sum);
        const R s2 = static_cast<R>(i2_sum);

        signal = (m * s2 - s * s - s * (m - R(1))) >
                 (s * sigma_b * std::sqrt(R(2) * (m - R(1))));
      }

      scr0[k] = (signal || image_in[k] >= masked) ? 1 : 0;
    }
  }

#pragma region ERODE

  const std::int32_t er = knl - 1;

  for (std::int32_t i = 0, k = 0; i < ny; i++) {
    for (std::int32_t j = 0; j < nx; j++, k++) {
      if (image_in[k] >= masked) {
        scr1[k] = 1;
        continue;
      }
      std::uint8_t m = scr0[k];
      if (m > 0) {
        for (std::int32_t _i = i - er; _i <= i + er && m; _i++) {
          if (_i < 0 || _i >= ny)
            continue;
          for (std::int32_t _j = j - er; _j <= j + er; _j++) {
            if (_j < 0 || _j >= nx)
              continue;
            if (!scr0[_i * nx + _j]) {
              m = 0;
              break;
            }
          }
        }
      }
      scr1[k] = m;
    }
  }

#pragma region RECOMPUTE MASKED SAT

  // i2_sat is dead from here on, but it is not released: the scratch is meant
  // to be reused across frames, and freeing it here would mean reallocating it
  // on the next one.  It is not the peak either way - the peak is the first
  // pass, where all three tables are live.

  for (std::int32_t i = 0, k = 0; i < ny; i++) {
    W _m = 0, _i = 0;
    for (std::int32_t j = 0; j < nx; j++, k++) {
      const bool valid = (image_in[k] < masked) && !scr1[k];
      const W p = valid ? static_cast<W>(image_in[k]) : W(0);
      _m += valid ? W(1) : W(0);
      _i += p;
      m_sat[k] = i > 0 ? _m + m_sat[k - nx] : _m;
      i_sat[k] = i > 0 ? _i + i_sat[k - nx] : _i;
    }
  }

#pragma region RECOMPUTE SAT AND FINAL THRESHOLD

  knl += 2;

  for (std::int32_t i = 0, k = 0; i < ny; i++) {
    for (std::int32_t j = 0; j < nx; j++, k++) {
      std::int32_t j0 = j - knl - 1;
      std::int32_t j1 = j < (nx - knl) ? j + knl : nx - 1;
      std::int32_t i0 = i - knl - 1;
      std::int32_t i1 = i < (ny - knl) ? i + knl : ny - 1;

      std::int32_t a = i1 * nx + j1;
      std::int32_t b = i0 * nx + j1;
      std::int32_t c = i1 * nx + j0;
      std::int32_t d = i0 * nx + j0;

      W m_sum = m_sat[a], i_sum = i_sat[a];

      if (j0 >= 0 && i0 >= 0) {
        m_sum += m_sat[d] - m_sat[b] - m_sat[c];
        i_sum += i_sat[d] - i_sat[b] - i_sat[c];
      } else if (j0 >= 0) {
        m_sum -= m_sat[c];
        i_sum -= i_sat[c];
      } else if (i0 >= 0) {
        m_sum -= m_sat[b];
        i_sum -= i_sat[b];
      }

      bool signal = false;
      const T p = image_in[k] >= masked ? T(0) : image_in[k];

      // Lifted out of the test below so that it is still in scope at the emit.
      // It is still computed only at candidates: hoisting it above the p > 0
      // and erosion tests would put a division on every pixel of the frame, a
      // cost worth paying only when the background is wanted everywhere.
      R mean = R(0);

      // The original also tested (m_sum >= 0), which is vacuous for an
      // unsigned accumulator.
      if (p > 0 && scr1[k]) {
        mean =
            m_sum >= 2 ? static_cast<R>(i_sum) / static_cast<R>(m_sum) : R(0);
        signal = static_cast<R>(p) >= (mean + sigma_s * std::sqrt(mean));
      }

      // Appended in scan order, so the list arrives ascending by index, which
      // is what the grouping needs and what the device path has to sort to
      // achieve.
      if (signal) {
        SignalPixel pixel;
        pixel.index = static_cast<std::uint32_t>(k);
        pixel.value = static_cast<std::uint32_t>(p);
        pixel.background = static_cast<float>(mean);
        pixel.population = static_cast<std::uint16_t>(m_sum);
        pixel.reserved = 0;
        signal_out.push_back(pixel);
      }
    }
  }

  return 0;
}

template int dext<std::uint16_t>(const std::uint16_t *,
                                 std::vector<SignalPixel> &, std::size_t,
                                 std::size_t, dext_scratch<std::uint16_t> &);
template int dext<std::uint32_t>(const std::uint32_t *,
                                 std::vector<SignalPixel> &, std::size_t,
                                 std::size_t, dext_scratch<std::uint32_t> &);
