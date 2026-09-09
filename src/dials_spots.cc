#include "dials_spots.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace dials_spots {
namespace {

constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

// The 1/12 DIALS adds to every standard error. It is the variance of a uniform
// distribution over one pixel, and DIALS' own comment says it is there to keep
// refinement from weighting itself onto a handful of very strong reflections
// rather than because the statistics call for it. Reproduced rather than
// improved: a different weight is a different refinement.
constexpr double kPixelVariance = 1.0 / 12.0;

} // namespace

Labeller::Labeller(std::size_t height, std::size_t width, Options options)
    : height_(height), width_(width), options_(options) {
  if (height == 0 || width == 0)
    throw std::runtime_error("dials_spots: a frame with no pixels in it");
  if (options_.max_spot_size < options_.min_spot_size)
    throw std::runtime_error("dials_spots: max_spot_size is below "
                             "min_spot_size, so nothing can be accepted");
  if (options_.min_spot_size == 0)
    throw std::runtime_error("dials_spots: min_spot_size must be at least 1");
}

// Path halving. The trees never get deep -- a union is between neighbouring
// pixels of the same frame or of consecutive ones -- and flush() flattens every
// surviving component to a star, so this stays close to O(1).
std::uint32_t Labeller::find(std::uint32_t i) {
  while (parent_[i] != i) {
    parent_[i] = parent_[parent_[i]];
    i = parent_[i];
  }
  return i;
}

void Labeller::merge(std::uint32_t a, std::uint32_t b) {
  const std::uint32_t ra = find(a);
  const std::uint32_t rb = find(b);
  if (ra == rb)
    return;
  // Toward the lower index, so a component's root is its first pixel and the
  // flattening in flush() has nothing to undo.
  if (ra < rb)
    parent_[rb] = ra;
  else
    parent_[ra] = rb;
}

void Labeller::add(std::int64_t frame, const std::vector<SignalPixel> &signal) {
  if (started_ && frame <= last_frame_) {
    throw std::runtime_error("dials_spots: frame " + std::to_string(frame) +
                             " arrived after frame " +
                             std::to_string(last_frame_) +
                             "; frames must be added in ascending order");
  }

  // A gap in the numbering. Nothing can be connected across a frame that is not
  // there, so everything open is closed before the new frame is added: no
  // pixel in live_ carries the frame number about to be used.
  if (started_ && frame != last_frame_ + 1)
    flush(frame);

  const std::size_t pixels = height_ * width_;
  const std::size_t base = live_.size();
  const std::uint32_t columns = static_cast<std::uint32_t>(width_);

  if (base + signal.size() > static_cast<std::size_t>(kNone)) {
    throw std::runtime_error(
        "dials_spots: more than 4 billion pixels are held open at once");
  }

  live_.reserve(base + signal.size());
  parent_.reserve(base + signal.size());
  for (const SignalPixel &pixel : signal) {
    if (pixel.index >= pixels) {
      throw std::runtime_error("dials_spots: signal pixel " +
                               std::to_string(pixel.index) +
                               " is off the end of the frame");
    }
    if (live_.size() > base && pixel.index <= live_.back().index) {
      throw std::runtime_error(
          "dials_spots: signal pixels are not ascending by index");
    }
    Pixel held;
    held.frame = static_cast<std::int32_t>(frame);
    held.index = pixel.index;
    held.value = pixel.value;
    parent_.push_back(static_cast<std::uint32_t>(live_.size()));
    live_.push_back(held);
  }
  counts_.signal_pixels += signal.size();

  const std::size_t end = live_.size();

  // The pixel to the left, within the same row. Without the column test the
  // last pixel of one row joins the first pixel of the next; DIALS gets the
  // same effect by comparing whole (frame, y, x) coordinates.
  for (std::size_t i = base + 1; i < end; i++) {
    if (live_[i].index % columns != 0 &&
        live_[i].index == live_[i - 1].index + 1)
      merge(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(i - 1));
  }

  // The pixel above, in the same frame. Both this list and its own targets
  // ascend, so one pointer walks the frame once rather than searching per
  // pixel.
  for (std::size_t i = base, up = base; i < end; i++) {
    if (live_[i].index < columns)
      continue;
    const std::uint32_t target = live_[i].index - columns;
    while (up < i && live_[up].index < target)
      up++;
    if (up < i && live_[up].index == target)
      merge(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(up));
  }

  // The same pixel on the previous frame, which is what makes this three
  // dimensional. Only consecutive frames, and only when the previous frame is
  // actually the one before this.
  if (!options_.two_d && started_ && last_frame_ + 1 == frame &&
      previous_ < base) {
    for (std::size_t i = base, back = previous_; i < end; i++) {
      const std::uint32_t target = live_[i].index;
      while (back < base && live_[back].index < target)
        back++;
      if (back < base && live_[back].index == target)
        merge(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(back));
    }
  }

  started_ = true;
  last_frame_ = frame;

  // In two dimensions every component is finished as soon as its frame is, so
  // keep a frame number that cannot be present.
  flush(options_.two_d ? frame + 1 : frame);
}

void Labeller::finish() {
  if (!started_)
    return;
  flush(last_frame_ + 1);
}

void Labeller::flush(std::int64_t keep_frame) {
  const std::size_t n = live_.size();
  if (n == 0) {
    previous_ = 0;
    return;
  }

  // Roots first, then a compact label per component numbered in order of first
  // appearance, which makes the emitted rows follow their first pixel.
  root_.resize(n);
  label_.assign(n, kNone);
  std::uint32_t components = 0;
  for (std::size_t i = 0; i < n; i++) {
    const std::uint32_t r = find(static_cast<std::uint32_t>(i));
    if (label_[r] == kNone)
      label_[r] = components++;
    root_[i] = label_[r];
  }

  // Counting sort of the pixels by component, so each component's members are
  // contiguous and still ascending by (frame, index) within it.
  offset_.assign(static_cast<std::size_t>(components) + 1, 0);
  for (std::size_t i = 0; i < n; i++)
    offset_[root_[i] + 1]++;
  for (std::size_t c = 0; c < components; c++)
    offset_[c + 1] += offset_[c];
  cursor_.assign(offset_.begin(), offset_.end() - 1);
  member_.resize(n);
  for (std::size_t i = 0; i < n; i++)
    member_[cursor_[root_[i]]++] = static_cast<std::uint32_t>(i);

  // A component that has a pixel on the frame just added can still grow.
  open_.assign(components, 0);
  for (std::size_t i = 0; i < n; i++) {
    if (live_[i].frame == keep_frame)
      open_[root_[i]] = 1;
  }

  for (std::uint32_t c = 0; c < components; c++) {
    if (!open_[c])
      emit(&member_[offset_[c]], offset_[c + 1] - offset_[c]);
  }

  // Compact, keeping global (frame, index) order: add() needs the previous
  // frame to be a contiguous ascending suffix. Every kept component is
  // re-rooted at its own first surviving pixel, so the trees start flat again.
  cursor_.assign(components, kNone);
  std::size_t write = 0;
  for (std::size_t i = 0; i < n; i++) {
    const std::uint32_t c = root_[i];
    if (!open_[c])
      continue;
    if (cursor_[c] == kNone)
      cursor_[c] = static_cast<std::uint32_t>(write);
    live_[write] = live_[i];
    parent_[write] = cursor_[c];
    write++;
  }
  live_.resize(write);
  parent_.resize(write);

  // Where the kept frame's pixels begin. They are the tail, since they are the
  // largest frame present and the order is preserved.
  previous_ = write;
  while (previous_ > 0 && live_[previous_ - 1].frame == keep_frame)
    previous_--;
}

void Labeller::emit(const std::uint32_t *members, std::size_t n) {
  counts_.groups++;

  if (n < options_.min_spot_size) {
    counts_.too_small++;
    return;
  }
  if (n > options_.max_spot_size) {
    counts_.too_large++;
    return;
  }

  Spot spot;
  spot.n_signal = static_cast<std::uint32_t>(n);

  // The bounding box, and the sums the centroid needs. Coordinates are pixel
  // centres, so a pixel at column i contributes at i + 0.5 -- DIALS generates
  // them that way in CentroidMaskedImage3d and the halves are what put a
  // single-pixel spot in the middle of its pixel rather than at its corner.
  double sum = 0.0, sum_sq = 0.0;
  double weighted[3] = {0.0, 0.0, 0.0};
  std::int32_t x0 = 0, x1 = 0, y0 = 0, y1 = 0, z0 = 0, z1 = 0;
  std::uint32_t peak_value = 0;
  double peak[3] = {0.0, 0.0, 0.0};

  for (std::size_t k = 0; k < n; k++) {
    const Pixel &pixel = live_[members[k]];
    const std::int32_t x = static_cast<std::int32_t>(pixel.index % width_);
    const std::int32_t y = static_cast<std::int32_t>(pixel.index / width_);
    const std::int32_t z = pixel.frame;
    if (k == 0) {
      x0 = x;
      x1 = x + 1;
      y0 = y;
      y1 = y + 1;
      z0 = z;
      z1 = z + 1;
    } else {
      x0 = x < x0 ? x : x0;
      x1 = x + 1 > x1 ? x + 1 : x1;
      y0 = y < y0 ? y : y0;
      y1 = y + 1 > y1 ? y + 1 : y1;
      z0 = z < z0 ? z : z0;
      z1 = z + 1 > z1 ? z + 1 : z1;
    }

    const double value = static_cast<double>(pixel.value);
    const double position[3] = {static_cast<double>(x) + 0.5,
                                static_cast<double>(y) + 0.5,
                                static_cast<double>(z) + 0.5};
    sum += value;
    sum_sq += value * value;
    for (int axis = 0; axis < 3; axis++)
      weighted[axis] += value * position[axis];

    // The brightest pixel, ties to the lowest (z, y, x). DIALS takes
    // af::max_index over the shoebox in that order, and the members arrive in
    // it, so a strictly-greater test matches.
    if (pixel.value > peak_value) {
      peak_value = pixel.value;
      peak[0] = position[0];
      peak[1] = position[1];
      peak[2] = position[2];
    }
  }

  spot.bbox[0] = x0;
  spot.bbox[1] = x1;
  spot.bbox[2] = y0;
  spot.bbox[3] = y1;
  spot.bbox[4] = z0;
  spot.bbox[5] = z1;

  // A shoebox whose background array is all zero, which is what the spot finder
  // produces: DIALS' Summation then reports the sum of the foreground and a
  // variance equal to it.
  spot.intensity = sum;
  spot.intensity_variance = sum;

  if (sum > 0.0) {
    for (int axis = 0; axis < 3; axis++)
      spot.position[axis] = weighted[axis] / sum;

    double delta_sq[3] = {0.0, 0.0, 0.0};
    for (std::size_t k = 0; k < n; k++) {
      const Pixel &pixel = live_[members[k]];
      const double value = static_cast<double>(pixel.value);
      const double position[3] = {
          static_cast<double>(pixel.index % width_) + 0.5,
          static_cast<double>(pixel.index / width_) + 0.5,
          static_cast<double>(pixel.frame) + 0.5};
      for (int axis = 0; axis < 3; axis++) {
        const double delta = position[axis] - spot.position[axis];
        delta_sq[axis] += value * delta * delta;
      }
    }

    // DIALS' unbiased variance, which fails its own assertion when the weights
    // carry no more information than one observation -- a single pixel, where
    // sum^2 == sum of squares. It catches that and reports 1/12 with no
    // variance, so the same case has to give the same answer here.
    if (sum * sum > sum_sq) {
      for (int axis = 0; axis < 3; axis++) {
        const double variance = delta_sq[axis] * sum / (sum * sum - sum_sq);
        spot.variance[axis] = variance / sum + kPixelVariance;
      }
    } else {
      for (int axis = 0; axis < 3; axis++)
        spot.variance[axis] = kPixelVariance;
    }

    // One frame, one answer: with a single-image bounding box the weighted mean
    // is already the middle of the frame, and DIALS says so explicitly rather
    // than relying on the arithmetic.
    if (spot.bbox[5] == spot.bbox[4] + 1)
      spot.position[2] = static_cast<double>(spot.bbox[4]) + 0.5;
  } else {
    // Unreachable from either finder, which only emits pixels above a positive
    // local mean, but DIALS has a defined answer for it and so should this.
    spot.position[0] = (spot.bbox[1] + spot.bbox[0]) / 2.0;
    spot.position[1] = (spot.bbox[3] + spot.bbox[2]) / 2.0;
    spot.position[2] = (spot.bbox[5] + spot.bbox[4]) / 2.0;
  }

  if (options_.max_separation > 0.0) {
    const double dx = peak[0] - spot.position[0];
    const double dy = peak[1] - spot.position[1];
    const double dz = peak[2] - spot.position[2];
    if (std::sqrt(dx * dx + dy * dy + dz * dz) > options_.max_separation) {
      counts_.separated++;
      return;
    }
  }

  spot.first = kept_.size();
  for (std::size_t k = 0; k < n; k++)
    kept_.push_back(live_[members[k]]);
  spots_.push_back(spot);
  counts_.accepted++;
}

} // namespace dials_spots
