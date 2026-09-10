// What a backend and the shared part of gpu:: say to each other, and a caller
// should not.
//
// dext_gpu.hh is the interface find_spots.cc uses; this is everything behind
// it, and it is separate so that including the first does not offer the
// second. Both directions live here: what a backend records into the shared
// part, and the two things the shared part has to ask of the backend.

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

// Which window each stage uses when nothing has asked for one, which is a
// question about the device and so is answered by the backend. Every other part
// of the choice -- the environment override, the warning on a value that is
// neither, the announcing -- stays in dext_gpu.cc, so there is still one
// implementation of the policy and only the measured preference is per backend.
//
// Both are consulted once, on the first call to stage0_window() or
// stage2_window(), so a backend may look at the device it found.
Window default_stage0_window();
Window default_stage2_window();

} // namespace internal
} // namespace gpu

#endif // SPOTFINDER_DEXT_GPU_INTERNAL_HH
