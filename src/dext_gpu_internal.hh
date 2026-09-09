// What a backend calls into the shared part of gpu:: that a caller should not.
//
// dext_gpu.hh is the interface signal.cc uses; this is the other direction, and
// it is separate so that including the first does not offer the second.

#ifndef SPOTFINDER_DEXT_GPU_INTERNAL_HH
#define SPOTFINDER_DEXT_GPU_INTERNAL_HH

#include "dext_gpu.hh"

namespace gpu {
namespace internal {

// Records the split for the calling thread, which last_stage_times() returns.
void set_stage_times(const StageTimes &times);

// Reports the configuration on stderr whenever it changes. Called once per
// frame by find(), which is cheap because it is one relaxed exchange when
// nothing has changed.
//
// Whenever it changes and not once per process: the first version used
// call_once, which announced whichever configuration ran first and then said
// nothing when the benchmark switched, so the benchmark's own output attributed
// every one of its numbers to the first configuration it happened to run.
void announce_windows(Window stage0, Window stage2);

} // namespace internal
} // namespace gpu

#endif // SPOTFINDER_DEXT_GPU_INTERNAL_HH
