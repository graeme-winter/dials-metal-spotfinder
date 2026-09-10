// The parts of gpu:: that are the same whichever backend is linked in.
//
// Which window each stage uses, whether to profile, and where the last frame's
// split went are all questions about configuration rather than about a device,
// so there is one implementation and both dext_cuda.cu and dext_metal.cc use
// it. The alternative was the same forty lines twice, which is how "tile" ends
// up spelled two ways.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "dext_gpu_internal.hh"

namespace gpu {
namespace {

// stage2 is tile on both devices and by a wide margin: per-pixel load traffic
// costs more than the barrier and the threadgroup memory save.
//
// stage0 is where they disagree, so it is not a constant. Measured per frame at
// 4362 x 4148, stage2 tile throughout:
//
//     stage0      Apple silicon    workstation CUDA
//     direct           4.86 ms         12.10 ms
//     tile             4.09 ms         12.37 ms
//
// 16% on Metal for tile, 2% on CUDA for direct. The Metal figure was 3.6% when
// it was first measured and grew when the benchmark stopped timing a staging
// copy the tool does not make; the CUDA column has not been re-measured since
// that fix and is worth re-running before it is trusted at 2%.
//
// So the default is asked of the backend rather than fixed here. This file
// still owns the policy -- the environment overrides the default, an
// unrecognised value warns and is ignored -- because that is the part that
// would otherwise end up written twice and spelled two ways.
std::atomic<Window> chosen0{Window::Direct};
std::atomic<Window> chosen2{Window::Tile};
std::once_flag chosen_once;

std::atomic<bool> profiling{false};
std::atomic<bool> reporting{true};

// -1 for "nothing announced yet", so the first frame always reports.
std::atomic<int> announced{-1};

StageTimes &thread_times() {
  static thread_local StageTimes times;
  return times;
}

// SPOTFINDER_GPU_<name>, taking "tile" or "direct".
void resolve_one(const char *name_suffix, std::atomic<Window> &into) {
  char variable[64];
  std::snprintf(variable, sizeof(variable), "SPOTFINDER_GPU_%s", name_suffix);
  const char *const asked = std::getenv(variable);
  if (asked == nullptr)
    return;

  if (std::strcmp(asked, "direct") == 0) {
    into.store(Window::Direct, std::memory_order_relaxed);
    return;
  }
  if (std::strcmp(asked, "tile") == 0) {
    into.store(Window::Tile, std::memory_order_relaxed);
    return;
  }
  // Not fatal: this changes how long a run takes, not what it reports, and
  // refusing to start a shift because of a typo would be the wrong trade.
  std::fprintf(stderr,
               "warning: %s=%s is not 'tile' or 'direct'; leaving it alone\n",
               variable, asked);
}

void resolve_windows() {
  chosen0.store(internal::default_stage0_window(), std::memory_order_relaxed);
  chosen2.store(internal::default_stage2_window(), std::memory_order_relaxed);
  resolve_one("STAGE0", chosen0);
  resolve_one("STAGE2", chosen2);
}

} // namespace

const char *name(Window window) {
  return window == Window::Tile ? "tile" : "direct";
}

Window stage0_window() {
  std::call_once(chosen_once, resolve_windows);
  return chosen0.load(std::memory_order_relaxed);
}

Window stage2_window() {
  std::call_once(chosen_once, resolve_windows);
  return chosen2.load(std::memory_order_relaxed);
}

// The call_once marks the environment as already consulted, so a later find()
// does not read it back over an explicit choice.
void stage0_window(Window window) {
  std::call_once(chosen_once, resolve_windows);
  chosen0.store(window, std::memory_order_relaxed);
}

void stage2_window(Window window) {
  std::call_once(chosen_once, resolve_windows);
  chosen2.store(window, std::memory_order_relaxed);
}

void profile_stages(bool on) { profiling.store(on, std::memory_order_relaxed); }

bool profile_stages() { return profiling.load(std::memory_order_relaxed); }

StageTimes last_stage_times() { return thread_times(); }

void report_windows(bool on) { reporting.store(on, std::memory_order_relaxed); }

namespace internal {

void set_stage_times(const StageTimes &times) { thread_times() = times; }

void announce_windows(Window stage0, Window stage2) {
  if (!reporting.load(std::memory_order_relaxed))
    return;
  // Two bits, so a change to either is a change here.
  const int now = static_cast<int>(stage0) * 2 + static_cast<int>(stage2);
  // Relaxed, and a benign race: two threads switching at once might both
  // report, which is a duplicate line and not a wrong one.
  if (announced.exchange(now, std::memory_order_relaxed) == now)
    return;
  std::fprintf(stderr, "%s: stage0 %s window, stage2 %s window\n", backend(),
               name(stage0), name(stage2));
}

} // namespace internal
} // namespace gpu
