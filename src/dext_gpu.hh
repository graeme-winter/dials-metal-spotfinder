// The GPU signal calculation, built only with -DSPOTFINDER_CUDA=ON or
// -DSPOTFINDER_METAL=ON.
//
// gpu::find mirrors dext() from dext.hh exactly -- same arguments, same output
// contract, same masked-pixel rule -- so signal.cc has one call site with a
// branch, and the two implementations can be run over the same frames and
// compared. Working buffers and a command stream are held per thread inside the
// implementation and reused between frames.
//
// There is one header for both backends because there is one interface: CUDA
// implements it in dext_cuda.cu and Metal in dext_metal.cc, exactly one of the
// two is compiled in, and nothing above this line knows which. Nothing here
// exposes a CUDA or a Metal header, so signal.cc needs no toolkit include path.

#ifndef SPOTFINDER_DEXT_GPU_HH
#define SPOTFINDER_DEXT_GPU_HH

#include <cstddef>
#include <cstdint>
#include <vector>

#include "signal_pixel.hh"

namespace gpu {

// "CUDA" or "Metal": which implementation was linked in. Reported by --version,
// because "with GPU support" does not say enough to reproduce a result.
const char *backend();

// ---------------------------------------------------------------------------
// How each stage sums its window.
//
// stage0 needs the count, sum and sum of squares over a 7 x 7 window; stage2
// the masked count and sum over an 11 x 11 one. Each can sum its window's
// pixels directly, or build summed-area tables in the block's shared memory
// over the block's tile plus its halo. The two compute the same thing -- the
// summed-area identity holds in Z / 2^32, so even the deliberately overflowing
// sum of squares comes out the same -- and which is faster is a question about
// the hardware.
//
// Backend-agnostic because the question is: on Apple silicon the tile wins
// stage2 by 1.76x, which is what raised it for stage0 and then for CUDA. It is
// not obvious that the answer carries across: a discrete card has several times
// the memory bandwidth and far more shared memory to spend, so the per-pixel
// load traffic the tile saves is worth less there and the occupancy it costs is
// worth less too. Measure with tests/bench_dext_gpu.cc on the machine in hand.
// ---------------------------------------------------------------------------

enum class Window {
  Direct, // the window's pixels summed straight
  Tile    // summed-area tables in shared memory
};

const char *name(Window window);

// Defaults come from SPOTFINDER_GPU_STAGE0 and SPOTFINDER_GPU_STAGE2, each
// taking "tile" or "direct". An
// unrecognised value is a warning on stderr and the default, not a failure:
// this changes how long a run takes and not what it reports, so refusing to
// start over a typo would be the wrong trade.
//
// The setters exist for a process that wants to time the combinations. Read
// once per frame rather than cached, so a switch takes effect on the next
// frame; not intended to be called while other threads are inside find().

// stage0's 7 x 7 dispersion test.
Window stage0_window();
void stage0_window(Window window);

// stage2's 11 x 11 Poisson test.
Window stage2_window();
void stage2_window(Window window);

// ---------------------------------------------------------------------------
// Where a frame's time went, in milliseconds, for the last frame this thread
// analysed while profiling was on.
// ---------------------------------------------------------------------------

struct StageTimes {
  // Getting the frame to where the kernels can read it. On CUDA this is the
  // host-to-device copy and it is not small: 36 MB at 16M pixels, which a PCIe
  // 3.0 x16 link delivers in about 2.7 ms, or a fifth of the frame. On Metal it
  // is zero, because the kernels read the buffer the frame was decompressed
  // into -- unless host_alloc failed and find() had to stage a copy, in which
  // case this is that copy and it is worth noticing.
  //
  // Itemised because it was not, and 2.7 ms of a 12.1 ms CUDA frame sat
  // unexplained between the kernel sum and the wall clock.
  double upload = 0.0;

  // Device time, from the device's own clock.
  double stage0 = 0.0; // the 7 x 7 dispersion test
  double stage1 = 0.0; // the 5 x 5 erode
  double stage2 = 0.0; // the 11 x 11 Poisson test and the emit

  // Host wall clock for the rest of find(), which is not device work and was
  // for a while being compared against device numbers as though it were. Kept
  // in the same struct so that mistake is harder to make again.
  double copy = 0.0; // reading the packed list back
  double sort = 0.0; // putting it into ascending index order
};

// Off by default, and a measurement mode rather than a cheap counter.
//
// How the split is obtained differs by backend, and so does how much to trust
// its sum. CUDA records events around each kernel in the one stream, which
// costs almost nothing and does not change the schedule. Metal has no
// per-dispatch timestamp without counter sample buffers, so it submits each
// stage in its own command buffer, which serialises whatever was overlapping --
// there, read the shares and not the sum, and compare against an unprofiled
// frame before trusting a split.
void profile_stages(bool on);
bool profile_stages();

// The last profiled frame's split, for the calling thread.
StageTimes last_stage_times();

// Whether find() names its window configuration on stderr when it changes. On
// by default: a frame rate that cannot be attributed to a configuration is not
// a measurement of anything. A benchmark that labels every row itself should
// turn it off, or its own output gets a line of duplicate commentary between
// each pair of numbers.
void report_windows(bool on);

// False if the binary has GPU support but the machine has no usable device.
bool available();

// Free device memory in bytes, for reporting before a run commits to it. On a
// unified-memory device this is what the process may still allocate before the
// system starts paying for it, not a separate pool.
std::size_t memory_free();

// Memory the device can reach without a copy: page-locked on CUDA, so that the
// frame copies are asynchronous, and a shared-storage buffer on Metal, where
// there is no copy at all. Returns nullptr on failure, in which case the caller
// should fall back to ordinary memory rather than fail -- a slower frame is
// better than no run.
void *host_alloc(std::size_t bytes);
void host_free(void *pointer);

// As dext(), including the return code: 0 on success, -1 for a bad argument.
//
// `signal_out` is cleared and refilled, ascending by index. The device emits in
// whatever order the warps reach the end, so this sorts before returning: the
// grouping downstream depends on the order, and getting it from the device
// would cost more than sorting a list this short on the host.
//
// Returns -2 if the frame produced more signal pixels than the device buffer
// holds -- one pixel in eight at 16 bits, one in four at 32. A frame that dense
// is not a frame of spots, and truncating it silently would be worse than
// refusing it.
//
// Throws std::runtime_error if the backend cannot do this pixel type: Metal has
// no double precision, so its 32-bit instantiation refuses rather than quietly
// running the dispersion test in float and disagreeing with the CPU.
template <typename T>
int find(const T *image_in, std::vector<SignalPixel> &signal_out,
         std::size_t height, std::size_t width);

extern template int find<std::uint16_t>(const std::uint16_t *,
                                        std::vector<SignalPixel> &, std::size_t,
                                        std::size_t);
extern template int find<std::uint32_t>(const std::uint32_t *,
                                        std::vector<SignalPixel> &, std::size_t,
                                        std::size_t);

} // namespace gpu

#endif // SPOTFINDER_DEXT_GPU_HH
