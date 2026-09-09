// dext() against gpu::find(), over the same frames.
//
// The two are meant to compute the same thing -- same windows, same masked
// pixel rule, same output contract -- so the useful test is not "does the
// device find spots" but "does it find exactly the ones the CPU finds, with the
// same background and the same population". Anything less than exact equality
// here is a real difference in the calculation, not a tolerance: the windowed
// sums are integers on both sides and the two floating-point operations that
// follow, one divide and one square root, are both correctly rounded.
//
// That is why the shader is compiled with -fno-fast-math. If this test starts
// failing on the background field by a bit or two, that flag is the first thing
// to check.
//
// Frames are chosen for the edges rather than for realism: sizes that are not
// multiples of the 32 x 32 threadgroup, spots planted hard against every
// border, and masked regions, because that is where a tile with a halo and a
// summed-area table with a clamp disagree if they are going to.
//
// Exits 77, which CTest reads as a skip, when the build has a GPU backend but
// the machine has no device.

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dext.hh"
#include "dext_gpu.hh"
#include "synthetic_frame.hh"

namespace {

using synthetic::Frame;
using synthetic::make_frame;

// Reported by field, because which field differs says what is wrong: index
// means the emit or the sort, value means the masking, background and
// population mean the second-pass window.
bool compare(const Frame &frame, const std::vector<SignalPixel> &cpu,
             const std::vector<SignalPixel> &gpu) {
  if (cpu.size() != gpu.size()) {
    std::printf("  FAIL %s: cpu found %zu signal pixels, gpu found %zu\n",
                frame.name.c_str(), cpu.size(), gpu.size());
    // Still worth saying where they first part company.
    const std::size_t n = cpu.size() < gpu.size() ? cpu.size() : gpu.size();
    for (std::size_t k = 0; k < n; k++) {
      if (cpu[k].index != gpu[k].index) {
        std::printf("       first differing index at entry %zu: cpu %u, gpu %u "
                    "(row %u, column %u)\n",
                    k, cpu[k].index, gpu[k].index,
                    cpu[k].index / static_cast<unsigned>(frame.width),
                    cpu[k].index % static_cast<unsigned>(frame.width));
        break;
      }
    }
    return false;
  }

  for (std::size_t k = 0; k < cpu.size(); k++) {
    const SignalPixel &a = cpu[k];
    const SignalPixel &b = gpu[k];
    if (a.index == b.index && a.value == b.value &&
        a.population == b.population && a.background == b.background)
      continue;
    std::printf("  FAIL %s: entry %zu differs\n", frame.name.c_str(), k);
    std::printf("       cpu index %u value %u background %.9g population %u\n",
                a.index, a.value, static_cast<double>(a.background),
                a.population);
    std::printf("       gpu index %u value %u background %.9g population %u\n",
                b.index, b.value, static_cast<double>(b.background),
                b.population);
    std::printf("       row %u, column %u of %zu x %zu\n",
                a.index / static_cast<unsigned>(frame.width),
                a.index % static_cast<unsigned>(frame.width), frame.height,
                frame.width);
    return false;
  }
  return true;
}

bool run(const Frame &frame) {
  std::vector<SignalPixel> cpu;
  std::vector<SignalPixel> device;

  // What the device buffer holds: it is the frame's bytes reused, sixteen to an
  // entry. Checked here so that a future edit to the plantings fails as "the
  // test frame is too dense" rather than as an unexplained -2 from find().
  const std::size_t capacity = frame.height * frame.width * 2 / 16;

  dext_scratch<std::uint16_t> scratch;
  const int cpu_status = dext<std::uint16_t>(
      frame.pixels.data(), cpu, frame.height, frame.width, scratch);
  if (cpu_status != 0) {
    std::printf("  FAIL %s: dext returned %d\n", frame.name.c_str(),
                cpu_status);
    return false;
  }
  if (cpu.size() > capacity) {
    std::printf("  FAIL %s: %zu signal pixels will not fit the device buffer's "
                "%zu; plant fewer spots\n",
                frame.name.c_str(), cpu.size(), capacity);
    return false;
  }

  const int gpu_status = gpu::find<std::uint16_t>(frame.pixels.data(), device,
                                                  frame.height, frame.width);
  if (gpu_status != 0) {
    std::printf("  FAIL %s: gpu::find returned %d\n", frame.name.c_str(),
                gpu_status);
    return false;
  }

  if (!compare(frame, cpu, device))
    return false;
  std::printf("  ok   %-22s %zu x %zu, %zu signal pixels\n", frame.name.c_str(),
              frame.height, frame.width, cpu.size());
  return true;
}

} // namespace

namespace {

// Every frame under one configuration of the backend. Returns the number that
// disagreed with the CPU.
int run_all(const std::vector<Frame> &frames) {
  int failures = 0;
  for (const Frame &frame : frames) {
    if (!run(frame))
      failures++;
  }
  return failures;
}

} // namespace

int main() {
  if (!gpu::available()) {
    std::printf("no %s device available; skipping\n", gpu::backend());
    return 77;
  }

  // Each section below is headed with its own configuration, so find()'s
  // running commentary on stderr would only duplicate it.
  gpu::report_windows(false);

  std::printf("comparing dext() against the %s backend\n", gpu::backend());

  std::vector<Frame> frames;
  // Exactly one threadgroup, so nothing is clipped.
  frames.push_back(make_frame("exact tile", 32, 32, 1, 12.0, false));
  // Smaller than a threadgroup in both directions.
  frames.push_back(make_frame("under one tile", 19, 27, 2, 8.0, false));
  // Not a multiple of 32 either way, which leaves a partial group on two edges.
  frames.push_back(make_frame("ragged", 200, 301, 3, 15.0, false));
  frames.push_back(make_frame("ragged, masked", 200, 301, 4, 15.0, true));
  // A dark frame: almost nothing should survive, and the two should agree that
  // almost nothing did.
  frames.push_back(make_frame("dark", 128, 128, 5, 1.0, false));
  // Something the size of a real detector module, over several tiles.
  frames.push_back(make_frame("module", 512, 1030, 6, 20.0, true));

  // A frame that is all mask: every pixel is a sentinel, so the finder must
  // come back empty. Built here rather than in the table because plant() would
  // undo it.
  {
    Frame frame;
    frame.name = "entirely masked";
    frame.height = 64;
    frame.width = 64;
    frame.pixels.assign(frame.height * frame.width, 0xffff);
    frames.push_back(frame);
  }

  int failures = 0;

  // Every combination of window, against the same CPU results. stage0 and
  // stage2 each have two, they are four ways of summing the same two windows,
  // and none of them is allowed to drift from the rest -- so testing whichever
  // pair the environment happened to select would leave most of the code
  // untested, and the fastest pair is the one that ends up in production.
  //
  // The cross product rather than one stage at a time: stage1 reads what stage0
  // wrote and stage2 reads what stage1 wrote, so an interaction is possible
  // even though neither variant changes what it writes.
  const gpu::Window windows[] = {gpu::Window::Direct, gpu::Window::Tile};
  for (const gpu::Window window0 : windows) {
    for (const gpu::Window window2 : windows) {
      gpu::stage0_window(window0);
      gpu::stage2_window(window2);
      std::printf("\nstage0 %s, stage2 %s:\n", gpu::name(window0),
                  gpu::name(window2));
      failures += run_all(frames);
    }
  }
  const std::size_t checks = frames.size() * 4;

  if (failures != 0) {
    std::printf("\n%d of %zu comparisons disagree\n", failures, checks);
    return 1;
  }
  std::printf("\nall %zu comparisons agree\n", checks);
  return 0;
}
