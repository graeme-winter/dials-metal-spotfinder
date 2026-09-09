// The extended dispersion signal calculation on the device.
//
// A templated transcription of the three kernels from spot_finder.cu, with the
// pixel type, the accumulator type and the real type taken from the same traits
// the CPU uses -- accumulator_t and real_t from dext.hh -- so that the 16-bit
// instantiation is what that file already did and the 32-bit one widens by
// construction rather than by hand.
//
// The output contract is dext()'s: a packed list of the surviving pixels, one
// SignalPixel each, ascending by index. stage0 and stage1 still pass 1/0 masks
// between themselves; only the last stage's output changed. Pixels at
// or above max() - 1 are masked, which is 0xfffe and 0xfffffffe.
//
// Device buffers and a stream are held per thread and reused between frames, in
// the same lifetime the original's worker() gave them.

#include "dext_gpu.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "dext.hh"
#include "dext_gpu_internal.hh"
#include "signal_order.hh"

namespace gpu {
namespace {

// A compile-time constant, so no <limits> is needed in device code.
template <typename T> __device__ __host__ constexpr T masked_value() {
  return static_cast<T>(~T(0) - 1);
}

// sqrtf for float, sqrt for double, chosen by the real type the traits give.
__device__ inline float root(float value) { return sqrtf(value); }
__device__ inline double root(double value) { return sqrt(value); }

void ok(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA: ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

// First stage: the dispersion test over a 7 x 7 window, summed directly rather
// than through an integral image. Writes 1 where the window looks like signal,
// and 1 for a masked pixel so that the erode below cannot grow through it.
template <typename T>
__global__ void stage0(const T *image_in, T *mask_out, int height, int width) {
  using W = accumulator_t<T>;
  using R = real_t<T>;

  const int knl = 3;
  const R sigma_b = R(6.0);
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= height || j >= width)
    return;
  const int k = i * width + j;
  const T masked = masked_value<T>();

  W m_sum = 0;
  W i_sum = 0;
  W i2_sum = 0;
  for (int di = -knl; di <= knl; di++) {
    for (int dj = -knl; dj <= knl; dj++) {
      const int _i = i + di;
      const int _j = j + dj;
      if (_i >= 0 && _i < height && _j >= 0 && _j < width) {
        const T pixel = image_in[_i * width + _j];
        const W valid = pixel >= masked ? W(0) : W(1);
        const W p = valid * static_cast<W>(pixel);
        m_sum += valid;
        i_sum += p;
        i2_sum += p * p;
      }
    }
  }

  bool signal = false;
  if (m_sum >= 2) {
    // n * sum(i^2) - sum(i)^2 - (n - 1) * sum(i)
    //     > sigma_b * sum(i) * sqrt(2 * (n - 1))
    const R m = static_cast<R>(m_sum);
    const R s = static_cast<R>(i_sum);
    const R s2 = static_cast<R>(i2_sum);
    signal = (m * s2 - s * s - s * (m - R(1))) >
             (s * sigma_b * root(R(2) * (m - R(1))));
  }

  mask_out[k] = (signal || image_in[k] >= masked) ? T(1) : T(0);
}

// The same test through summed-area tables in shared memory, over the block's
// tile plus its halo. Three tables, since the dispersion test needs the count,
// the sum and the sum of squares: 38 x 38 of W, so 17 KB at 16 bits and 35 KB
// at 32.
//
// Whether this beats the direct sum above is a hardware question and the answer
// is not obvious on a discrete card, where bandwidth is plentiful and shared
// memory is 48 KB or more. It won on Apple silicon at stage2 by 1.76x, which is
// why it exists here; SPOTFINDER_GPU_STAGE0 selects it and
// tests/bench_dext_gpu.cc settles it.
//
// The deliberate overflow of p * p survives the change of method. The
// summed-area identity holds in any commutative ring, Z / 2^32 and Z / 2^64
// included, so differencing a wrapped table gives exactly the value the direct
// sum reaches with the same wrapping. That is already why dext.cc, which
// differences a whole-frame table that has certainly wrapped, agrees with the
// direct kernel above.
template <typename T>
__global__ void stage0_tile(const T *image_in, T *mask_out, int height,
                            int width) {
  using W = accumulator_t<T>;
  using R = real_t<T>;

  const int knl = 3;
  const R sigma_b = R(6.0);
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int n = threadIdx.x + blockDim.x * threadIdx.y;
  const int N = 32 + 3 + 3;
  const T masked = masked_value<T>();

  __shared__ W M[N][N];
  __shared__ W I[N][N];
  __shared__ W I2[N][N];

  // Rows first: each of the first N threads runs along one row of the halo.
  if (n < N) {
    W tm = 0;
    W ti = 0;
    W ti2 = 0;
    const int _i = n - knl + blockDim.y * blockIdx.y;
    for (int m = 0; m < N; m++) {
      const int _j = m - knl + blockDim.x * blockIdx.x;
      if (_i >= 0 && _j >= 0 && _i < height && _j < width) {
        const T pixel = image_in[_i * width + _j];
        const W valid = pixel >= masked ? W(0) : W(1);
        const W p = valid * static_cast<W>(pixel);
        tm += valid;
        ti += p;
        ti2 += p * p;
      }
      M[n][m] = tm;
      I[n][m] = ti;
      I2[n][m] = ti2;
    }
  }

  __syncthreads();

  // Then down the columns, which completes the tables.
  if (n < N) {
    W tm = 0;
    W ti = 0;
    W ti2 = 0;
    for (int m = 0; m < N; m++) {
      tm += M[m][n];
      ti += I[m][n];
      ti2 += I2[m][n];
      M[m][n] = tm;
      I[m][n] = ti;
      I2[m][n] = ti2;
    }
  }

  __syncthreads();

  // Safe here and not before: no warp-wide operation follows, unlike stage2,
  // but every thread still has to reach both barriers.
  if (i >= height || j >= width)
    return;
  const int k = i * width + j;

  const int j0 = static_cast<int>(threadIdx.x) - 1;
  const int j1 = static_cast<int>(threadIdx.x) + 2 * knl;
  const int i0 = static_cast<int>(threadIdx.y) - 1;
  const int i1 = static_cast<int>(threadIdx.y) + 2 * knl;

  W m_sum = M[i1][j1];
  W i_sum = I[i1][j1];
  W i2_sum = I2[i1][j1];

  if (i0 >= 0 && j0 >= 0) {
    m_sum += M[i0][j0] - M[i1][j0] - M[i0][j1];
    i_sum += I[i0][j0] - I[i1][j0] - I[i0][j1];
    i2_sum += I2[i0][j0] - I2[i1][j0] - I2[i0][j1];
  } else if (j0 >= 0) {
    m_sum -= M[i1][j0];
    i_sum -= I[i1][j0];
    i2_sum -= I2[i1][j0];
  } else if (i0 >= 0) {
    m_sum -= M[i0][j1];
    i_sum -= I[i0][j1];
    i2_sum -= I2[i0][j1];
  }

  bool signal = false;
  if (m_sum >= 2) {
    const R m = static_cast<R>(m_sum);
    const R s = static_cast<R>(i_sum);
    const R s2 = static_cast<R>(i2_sum);
    signal = (m * s2 - s * s - s * (m - R(1))) >
             (s * sigma_b * root(R(2) * (m - R(1))));
  }

  mask_out[k] = (signal || image_in[k] >= masked) ? T(1) : T(0);
}

// Second stage: erode the dispersion map, so that a pixel survives only if its
// whole 5 x 5 neighbourhood did.
template <typename T>
__global__ void stage1(const T *image_in, const T *mask_in, T *mask_out,
                       int height, int width) {
  const int knl = 2;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= height || j >= width)
    return;

  const int k = i * width + j;
  if (image_in[k] >= masked_value<T>()) {
    mask_out[k] = T(1);
    return;
  }
  if (mask_in[k] == 0) {
    mask_out[k] = T(0);
    return;
  }

  for (int di = -knl; di <= knl; di++) {
    for (int dj = -knl; dj <= knl; dj++) {
      const int _i = i + di;
      const int _j = j + dj;
      if (_i >= 0 && _i < height && _j >= 0 && _j < width) {
        if (!mask_in[_i * width + _j]) {
          mask_out[k] = T(0);
          return;
        }
      }
    }
  }
  mask_out[k] = T(1);
}

// The Poisson test and the compacted write, shared by both stage2 kernels so
// that a change to either cannot land in one and not the other.
//
// `in_range` rather than an early return in the caller: every thread in the
// warp has to reach __ballot_sync, and a thread that has returned cannot take
// part in it -- the result would be silently wrong rather than a clean failure.
template <typename T>
__device__ inline void
emit_signal(const T *image_in, const T *mask_in, SignalPixel *out,
            unsigned *counter, unsigned capacity, int k, bool in_range,
            accumulator_t<T> m_sum, accumulator_t<T> i_sum) {
  using R = real_t<T>;
  const R sigma_s = R(3.0);
  const T masked = masked_value<T>();

  bool signal = false;
  T p = T(0);
  R mean = R(0);

  if (in_range) {
    p = image_in[k] >= masked ? T(0) : image_in[k];
    if (p > 0 && mask_in[k]) {
      mean = m_sum >= 2 ? static_cast<R>(i_sum) / static_cast<R>(m_sum) : R(0);
      signal = static_cast<R>(p) >= (mean + sigma_s * root(mean));
    }
  }

  // One atomic per warp rather than one per pixel: the warp votes, the leader
  // claims that many slots, and each emitting lane takes the slot at its rank
  // among the set bits below it. With a 32 x 32 block the lane is threadIdx.x.
  const unsigned lane = threadIdx.x;
  const unsigned ballot = __ballot_sync(0xffffffffu, signal);
  const unsigned rank = __popc(ballot & ((1u << lane) - 1u));

  unsigned base = 0;
  if (lane == 0 && ballot != 0u)
    base = atomicAdd(counter, __popc(ballot));
  base = __shfl_sync(0xffffffffu, base, 0);

  // Past the end is dropped rather than written. The counter still counts it,
  // so the host sees the overflow and refuses the frame instead of analysing a
  // truncated list.
  if (signal && base + rank < capacity) {
    SignalPixel pixel;
    pixel.index = static_cast<unsigned>(k);
    pixel.value = static_cast<unsigned>(p);
    pixel.background = static_cast<float>(mean);
    pixel.population = static_cast<unsigned short>(m_sum);
    pixel.reserved = 0;
    out[base + rank] = pixel;
  }
}

// Third stage: the Poisson test against a local mean taken over an 11 x 11
// window, built as a summed-area table in shared memory across the block's tile
// plus its halo. The accumulators follow the traits, so this is 14 KB of shared
// memory for 16-bit pixels and 28 KB for 32-bit; if the wider one costs too
// much occupancy, summing directly as stage0 does is the alternative.
template <typename T>
__global__ void stage2(const T *image_in, const T *mask_in, SignalPixel *out,
                       unsigned *counter, unsigned capacity, int height,
                       int width) {
  using W = accumulator_t<T>;
  using R = real_t<T>;

  const int knl = 5;
  const R sigma_s = R(3.0);
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int n = threadIdx.x + blockDim.x * threadIdx.y;
  const int k = i * width + j;
  const int N = 32 + 5 + 5;
  const T masked = masked_value<T>();

  __shared__ W M[N][N];
  __shared__ W I[N][N];

  // Rows first: each of the first N threads runs along one row of the halo,
  // accumulating as it goes.
  if (n < N) {
    W tm = 0;
    W ti = 0;
    const int _i = n - knl + blockDim.y * blockIdx.y;
    for (int m = 0; m < N; m++) {
      const int _j = m - knl + blockDim.x * blockIdx.x;
      if (_i >= 0 && _j >= 0 && _i < height && _j < width) {
        const int _k = _i * width + _j;
        const T pixel = image_in[_k];
        const W valid = (pixel >= masked) || mask_in[_k] ? W(0) : W(1);
        tm += valid;
        ti += valid * static_cast<W>(pixel);
      }
      M[n][m] = tm;
      I[n][m] = ti;
    }
  }

  __syncthreads();

  // Then down the columns, which completes the summed-area table.
  if (n < N) {
    W tm = 0;
    W ti = 0;
    for (int m = 0; m < N; m++) {
      tm += M[m][n];
      ti += I[m][n];
      M[m][n] = tm;
      I[m][n] = ti;
    }
  }

  __syncthreads();

  // Not an early return. Every thread in the warp has to reach the ballot
  // below, and a thread that has returned cannot take part in __ballot_sync --
  // the result would be silently wrong rather than a clean failure.
  const bool in_range = i < height && j < width;

  const int j0 = static_cast<int>(threadIdx.x) - 1;
  const int j1 = static_cast<int>(threadIdx.x) + 2 * knl;
  const int i0 = static_cast<int>(threadIdx.y) - 1;
  const int i1 = static_cast<int>(threadIdx.y) + 2 * knl;

  W m_sum = M[i1][j1];
  W i_sum = I[i1][j1];

  if (i0 >= 0 && j0 >= 0) {
    m_sum += M[i0][j0] - M[i1][j0] - M[i0][j1];
    i_sum += I[i0][j0] - I[i1][j0] - I[i0][j1];
  } else if (j0 >= 0) {
    m_sum -= M[i1][j0];
    i_sum -= I[i1][j0];
  } else if (i0 >= 0) {
    m_sum -= M[i0][j1];
    i_sum -= I[i0][j1];
  }

  emit_signal<T>(image_in, mask_in, out, counter, capacity, k, in_range, m_sum,
                 i_sum);
}

// The same test with the window summed straight, 121 pixels a thread against
// the table's two. Slower on Apple silicon by 1.76x; whether that holds on a
// discrete card, where bandwidth is plentiful, is what SPOTFINDER_GPU_STAGE2
// and tests/bench_dext_gpu.cc are for.
//
// It is also the second, independent way of summing this window, which is what
// makes tests/test_dext_gpu.cc a cross-check on the tile's index arithmetic
// rather than only on the kernel as a whole.
template <typename T>
__global__ void stage2_direct(const T *image_in, const T *mask_in,
                              SignalPixel *out, unsigned *counter,
                              unsigned capacity, int height, int width) {
  using W = accumulator_t<T>;

  const int knl = 5;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int k = i * width + j;
  const T masked = masked_value<T>();

  // Not an early return, for the reason emit_signal documents.
  const bool in_range = i < height && j < width;

  W m_sum = 0;
  W i_sum = 0;

  if (in_range) {
    // Clipped once per axis rather than tested once per pixel.
    const int i_from = i - knl > 0 ? i - knl : 0;
    const int i_to = i + knl < height - 1 ? i + knl : height - 1;
    const int j_from = j - knl > 0 ? j - knl : 0;
    const int j_to = j + knl < width - 1 ? j + knl : width - 1;

    for (int _i = i_from; _i <= i_to; _i++) {
      const int row = _i * width;
      for (int _j = j_from; _j <= j_to; _j++) {
        const int _k = row + _j;
        const T pixel = image_in[_k];
        const W valid = (pixel >= masked) || mask_in[_k] ? W(0) : W(1);
        m_sum += valid;
        i_sum += valid * static_cast<W>(pixel);
      }
    }
  }

  emit_signal<T>(image_in, mask_in, out, counter, capacity, k, in_range, m_sum,
                 i_sum);
}

// Leaves the stream idle however the scope is left.
//
// Without this, an exception thrown after the first enqueue -- a launch
// failure, a timeout, an illegal address reported by a later call -- returns
// from find() with kernels still running and a copy still in flight. The worker
// counts a failure and takes the next frame, which decompresses into the same
// host buffer the outstanding device-to-host copy is still writing to, and
// enqueues fresh work over device buffers the previous kernels are still
// reading. The damage surfaces somewhere else entirely, which is what an
// intermittent fault under load looks like.
class StreamGuard {
public:
  explicit StreamGuard(cudaStream_t stream) : stream_(stream) {}
  ~StreamGuard() {
    if (stream_ != nullptr)
      cudaStreamSynchronize(stream_);
  }
  StreamGuard(const StreamGuard &) = delete;
  StreamGuard &operator=(const StreamGuard &) = delete;

private:
  cudaStream_t stream_;
};

// The device side of one worker thread: three buffers and a stream, allocated
// on the first frame and reused after that. Freed when the thread ends.
class Workspace {
public:
  ~Workspace() {
    // Nothing may be in flight when the buffers go: a thread that left find()
    // through an exception can still have work queued.
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
    for (cudaEvent_t &mark : marks_) {
      if (mark != nullptr)
        cudaEventDestroy(mark);
    }
    cudaFree(counter_);
    cudaFree(in_);
    cudaFree(out_);
    cudaFree(tmp_);
  }

  unsigned *counter() const { return counter_; }

  // Five events: one before the upload and one after each of the copy and the
  // three kernels. Created on first use so that a
  // run which never profiles never allocates them. Recording an event costs
  // essentially nothing and, unlike Metal's per-command-buffer timestamps, does
  // not change the schedule -- so the CUDA split can be trusted as a split and
  // its sum compared against the frame.
  cudaEvent_t *marks() {
    if (marks_[0] == nullptr) {
      for (cudaEvent_t &mark : marks_)
        ok(cudaEventCreate(&mark), "cudaEventCreate");
    }
    return marks_;
  }

  void reserve(std::size_t bytes) {
    if (stream_ == nullptr)
      ok(cudaStreamCreate(&stream_), "cudaStreamCreate");
    if (counter_ == nullptr)
      ok(cudaMalloc(&counter_, sizeof(unsigned)), "cudaMalloc for the counter");
    if (bytes <= capacity_)
      return;
    // As in the destructor: a larger frame must not free buffers that
    // outstanding work is still reading.
    cudaStreamSynchronize(stream_);
    cudaFree(in_);
    cudaFree(out_);
    cudaFree(tmp_);
    in_ = out_ = tmp_ = nullptr;
    capacity_ = 0;
    ok(cudaMalloc(&in_, bytes), "cudaMalloc for the frame");
    ok(cudaMalloc(&out_, bytes), "cudaMalloc for the mask");
    ok(cudaMalloc(&tmp_, bytes), "cudaMalloc for the working mask");
    capacity_ = bytes;
  }

  cudaStream_t stream() const { return stream_; }
  void *in() const { return in_; }
  void *out() const { return out_; }
  void *tmp() const { return tmp_; }

private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t marks_[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  unsigned *counter_ = nullptr;
  void *in_ = nullptr;
  void *out_ = nullptr;
  void *tmp_ = nullptr;
  std::size_t capacity_ = 0;
};

Workspace &workspace() {
  static thread_local Workspace workspace;
  return workspace;
}

} // namespace

const char *backend() { return "CUDA"; }

bool available() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

std::size_t memory_free() {
  std::size_t free = 0;
  std::size_t total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess)
    return 0;
  return free;
}

void *host_alloc(std::size_t bytes) {
  void *pointer = nullptr;
  if (cudaMallocHost(&pointer, bytes) != cudaSuccess)
    return nullptr;
  return pointer;
}

void host_free(void *pointer) { cudaFreeHost(pointer); }

template <typename T>
int find(const T *image_in, std::vector<SignalPixel> &signal_out,
         std::size_t height, std::size_t width) {
  signal_out.clear();
  if (image_in == nullptr || height == 0 || width == 0)
    return -1;
  // The kernels index in int, as the original's did.
  if (height > 0x7fffffffu / width)
    return -1;

  const std::size_t bytes = height * width * sizeof(T);
  Workspace &space = workspace();
  space.reserve(bytes);
  const cudaStream_t stream = space.stream();

  // Armed before the first enqueue, so that no exit path leaves work in flight.
  const StreamGuard idle(stream);

  const dim3 block(32, 32);
  const dim3 grid(static_cast<unsigned>((width + 31) / 32),
                  static_cast<unsigned>((height + 31) / 32));
  const int rows = static_cast<int>(height);
  const int columns = static_cast<int>(width);

  T *const in = static_cast<T *>(space.in());
  T *const tmp = static_cast<T *>(space.tmp());
  // stage0's mask is dead once stage1 has read it, so the same block carries
  // the packed list. Sixteen bytes an entry against sizeof(T) a pixel is where
  // the one-in-eight bound at 16 bits comes from.
  T *const scratch = static_cast<T *>(space.out());
  SignalPixel *const out = static_cast<SignalPixel *>(space.out());
  const unsigned capacity = static_cast<unsigned>(bytes / sizeof(SignalPixel));
  unsigned *const counter = space.counter();

  const Window window0 = stage0_window();
  const Window window2 = stage2_window();
  internal::announce_windows(window0, window2);

  const bool timing = profile_stages();
  cudaEvent_t *const marks = timing ? space.marks() : nullptr;

  // The upload is bracketed too. It is 36 MB at 16M pixels and a PCIe 3.0 x16
  // link takes about 2.7 ms over it, which is a fifth of the frame -- and it
  // sat unexplained between the kernel sum and the wall clock until it was
  // measured. Metal has no equivalent: there the kernels read the buffer the
  // frame was decompressed into.
  if (timing)
    ok(cudaEventRecord(marks[0], stream), "cudaEventRecord");
  ok(cudaMemcpyAsync(in, image_in, bytes, cudaMemcpyHostToDevice, stream),
     "copying a frame to the device");
  ok(cudaMemsetAsync(counter, 0, sizeof(unsigned), stream),
     "clearing the signal counter");

  if (timing)
    ok(cudaEventRecord(marks[1], stream), "cudaEventRecord");
  if (window0 == Window::Tile)
    stage0_tile<T><<<grid, block, 0, stream>>>(in, scratch, rows, columns);
  else
    stage0<T><<<grid, block, 0, stream>>>(in, scratch, rows, columns);

  if (timing)
    ok(cudaEventRecord(marks[2], stream), "cudaEventRecord");
  stage1<T><<<grid, block, 0, stream>>>(in, scratch, tmp, rows, columns);

  if (timing)
    ok(cudaEventRecord(marks[3], stream), "cudaEventRecord");
  if (window2 == Window::Tile)
    stage2<T><<<grid, block, 0, stream>>>(in, tmp, out, counter, capacity, rows,
                                          columns);
  else
    stage2_direct<T><<<grid, block, 0, stream>>>(in, tmp, out, counter,
                                                 capacity, rows, columns);
  if (timing)
    ok(cudaEventRecord(marks[4], stream), "cudaEventRecord");
  ok(cudaGetLastError(), "launching the signal calculation");

  unsigned found = 0;
  ok(cudaMemcpyAsync(&found, counter, sizeof(unsigned), cudaMemcpyDeviceToHost,
                     stream),
     "reading the signal count back");
  ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  StageTimes times;
  if (timing) {
    float elapsed = 0.0f;
    ok(cudaEventElapsedTime(&elapsed, marks[0], marks[1]), "upload time");
    times.upload = static_cast<double>(elapsed);
    ok(cudaEventElapsedTime(&elapsed, marks[1], marks[2]), "stage0 time");
    times.stage0 = static_cast<double>(elapsed);
    ok(cudaEventElapsedTime(&elapsed, marks[2], marks[3]), "stage1 time");
    times.stage1 = static_cast<double>(elapsed);
    ok(cudaEventElapsedTime(&elapsed, marks[3], marks[4]), "stage2 time");
    times.stage2 = static_cast<double>(elapsed);
  }

  if (found > capacity)
    return -2;
  if (found == 0) {
    if (timing)
      internal::set_stage_times(times);
    return 0;
  }

  const auto mark = [timing] {
    return timing ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
  };
  const auto since = [timing](std::chrono::steady_clock::time_point from) {
    if (!timing)
      return 0.0;
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - from)
        .count();
  };
  const auto copy_started = mark();

  // Only the part that was written: at a typical signal fraction this is a
  // small fraction of the frame, which is the point of compacting on the
  // device.
  signal_out.resize(found);
  ok(cudaMemcpyAsync(signal_out.data(), out, found * sizeof(SignalPixel),
                     cudaMemcpyDeviceToHost, stream),
     "copying the signal pixels back");
  ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  // The warps emit in whatever order they finish; the grouping needs ascending
  // index, and relies on k - 1 being the entry immediately before.
  //
  // Was std::sort here too. The measurement that replaced it was taken on
  // Metal, but the sort is host code and the frames are the same size, so this
  // gets the same several milliseconds a frame back.
  const double copy_ms = since(copy_started);

  static thread_local signal_order::scratch order;
  const auto sort_started = mark();
  signal_order::by_index(signal_out, order);

  if (timing) {
    times.copy = copy_ms;
    times.sort = since(sort_started);
    internal::set_stage_times(times);
  }
  return 0;
}

template int find<std::uint16_t>(const std::uint16_t *,
                                 std::vector<SignalPixel> &, std::size_t,
                                 std::size_t);
template int find<std::uint32_t>(const std::uint32_t *,
                                 std::vector<SignalPixel> &, std::size_t,
                                 std::size_t);

} // namespace gpu
