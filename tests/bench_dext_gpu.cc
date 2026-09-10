// Time the two stage2 window strategies against each other, and against the
// CPU.
//
//     ./bench_dext_gpu [height width [repeats [threads]]]
//
// Defaults to 4362 x 4148, the real detector geometry the signal calculation
// has been run at, 20 repeats, and one thread.
//
// With threads > 1 it runs that many workers over their own frames at once and
// reports aggregate throughput, which is the number -j actually
// delivers. Single-frame latency and aggregate throughput are different
// questions and the second is the one that decides whether an acquisition keeps
// up: a variant that does more load traffic per pixel loses more as workers
// pile up, because they compete for the same memory. That is not a hypothesis
// -- it is why stage2's direct window is 1.27x slower than the tile at one
// thread and 2.08x slower across a real run.
//
// Why this exists: stage0 and stage2 can each get their window from a
// summed-area table in threadgroup memory or by summing its pixels directly.
// The table does far less load traffic; the direct sum needs no threadgroup
// memory, no barrier, and keeps every thread busy. Which matters more is not
// something to reason about -- reasoning got it backwards once already -- so
// all four combinations are timed on the machine that will run them.
//
// What is timed is whole frames through gpu::find, not stage2 in isolation.
// That is deliberate: frames per second is the number that decides whether an
// acquisition keeps up, and isolating stage2 would need counter sample buffers
// and would answer a question nobody asked.
//
// Correctness is not assumed. The two variants are compared against each other
// before anything is timed, and the run stops if they disagree: a faster
// variant that computes something else is not faster.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "dext.hh"
#include "dext_gpu.hh"
#include "synthetic_frame.hh"

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}

// Median rather than mean, and the best and worst kept: the first frame pays
// for the pipeline build and the buffer allocation, and a laptop under thermal
// pressure produces outliers that a mean would quietly fold in.
struct Timing {
  double best = 0.0;
  double median = 0.0;
  double worst = 0.0;
};

Timing summarise(std::vector<double> samples) {
  Timing timing;
  if (samples.empty())
    return timing;
  std::sort(samples.begin(), samples.end());
  timing.best = samples.front();
  timing.median = samples[samples.size() / 2];
  timing.worst = samples.back();
  return timing;
}

void report(const char *what, const Timing &timing, std::size_t pixels) {
  const double rate = timing.median > 0.0 ? 1000.0 / timing.median : 0.0;
  const double throughput =
      timing.median > 0.0
          ? static_cast<double>(pixels) / (timing.median / 1000.0) / 1e9
          : 0.0;
  std::printf("  %-28s %8.2f ms  %7.1f frame/s  %5.2f Gpixel/s  "
              "(best %.2f, worst %.2f)\n",
              what, timing.median, rate, throughput, timing.best, timing.worst);
}

bool identical(const std::vector<SignalPixel> &a,
               const std::vector<SignalPixel> &b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t k = 0; k < a.size(); k++) {
    if (a[k].index != b[k].index || a[k].value != b[k].value ||
        a[k].population != b[k].population ||
        a[k].background != b[k].background)
      return false;
  }
  return true;
}

// The frame in the memory the tool hands find(), which is memory the device can
// read without a copy: gpu::host_alloc, as src/find_spots.cc's FrameBuffer
// does. Ordinary memory works and is copied into a staging buffer once per
// frame instead -- correct, but half a millisecond of a four millisecond frame
// that production does not spend, which is exactly the sort of thing a
// benchmark must not invent. time_threaded() always did this; the single-frame
// paths did not, and reported a copy nobody pays.
class Shared {
public:
  explicit Shared(const synthetic::Frame &frame)
      : fallback_(frame.pixels.data()),
        bytes_(frame.pixels.size() * sizeof(std::uint16_t)) {
    buffer_ = gpu::host_alloc(bytes_);
    if (buffer_ != nullptr)
      std::memcpy(buffer_, frame.pixels.data(), bytes_);
  }

  ~Shared() {
    if (buffer_ != nullptr)
      gpu::host_free(buffer_);
  }

  Shared(const Shared &) = delete;
  Shared &operator=(const Shared &) = delete;

  const std::uint16_t *pixels() const {
    return buffer_ != nullptr ? static_cast<const std::uint16_t *>(buffer_)
                              : fallback_;
  }

  bool shared() const { return buffer_ != nullptr; }

private:
  const std::uint16_t *fallback_;
  std::size_t bytes_;
  void *buffer_ = nullptr;
};

// One configuration, timed. Returns the median frame time in milliseconds.
Timing time_windows(gpu::Window window0, gpu::Window window2,
                    const synthetic::Frame &frame, const std::uint16_t *pixels,
                    int repeats, std::vector<SignalPixel> *out) {
  gpu::stage0_window(window0);
  gpu::stage2_window(window2);
  std::vector<SignalPixel> signal;

  // One untimed frame first: it pays for the pipelines, the buffers and the
  // first touch of the shared allocation, none of which a steady stream does.
  if (gpu::find<std::uint16_t>(pixels, signal, frame.height, frame.width) !=
      0) {
    std::printf("  find() failed\n");
    return Timing();
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  for (int n = 0; n < repeats; n++) {
    const Clock::time_point started = Clock::now();
    const int status =
        gpu::find<std::uint16_t>(pixels, signal, frame.height, frame.width);
    const Clock::time_point finished = Clock::now();
    if (status != 0) {
      std::printf("  find() returned %d on repeat %d\n", status, n);
      return Timing();
    }
    samples.push_back(milliseconds(started, finished));
  }

  if (out != nullptr)
    *out = signal;
  return summarise(samples);
}

// The same configuration under concurrency: `threads` workers, each with its
// own frame buffer, all analysing at once. Returns aggregate frames per second.
//
// Each worker gets its frame from gpu::host_alloc, which is what signal.cc
// does, so the no-copy path is exercised rather than the staging fallback --
// otherwise this would be measuring a memcpy that production does not do.
double time_threaded(gpu::Window window0, gpu::Window window2,
                     const synthetic::Frame &frame, int repeats, int threads) {
  gpu::stage0_window(window0);
  gpu::stage2_window(window2);

  const std::size_t bytes = frame.pixels.size() * sizeof(std::uint16_t);
  std::atomic<int> failed{0};

  const auto worker = [&]() {
    void *const buffer = gpu::host_alloc(bytes);
    std::uint16_t *const pixels =
        buffer != nullptr ? static_cast<std::uint16_t *>(buffer) : nullptr;
    if (pixels != nullptr)
      std::memcpy(pixels, frame.pixels.data(), bytes);
    const std::uint16_t *const source =
        pixels != nullptr ? pixels : frame.pixels.data();

    std::vector<SignalPixel> signal;
    // Warm this thread's workspace before the timed section, so the first
    // frame's allocations are not counted against everyone else.
    if (gpu::find<std::uint16_t>(source, signal, frame.height, frame.width) !=
        0)
      failed++;

    for (int n = 0; n < repeats; n++) {
      if (gpu::find<std::uint16_t>(source, signal, frame.height, frame.width) !=
          0) {
        failed++;
        break;
      }
    }
    if (buffer != nullptr)
      gpu::host_free(buffer);
  };

  // No barrier between each worker's warm-up frame and its timed loop. One
  // would need every thread to arrive before any could start, which is not what
  // a worker pool does either; the warm-up sits inside the timed window and
  // shows up as a small fixed cost that every configuration pays alike.
  const Clock::time_point started = Clock::now();
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; t++)
    pool.emplace_back(worker);
  for (std::thread &thread : pool)
    thread.join();
  const double elapsed = milliseconds(started, Clock::now());

  if (failed.load() != 0) {
    std::printf("  %d worker frames failed\n", failed.load());
    return 0.0;
  }
  // The warm-up frames are counted, because they are inside the timed window:
  // threads * (repeats + 1) frames were actually analysed.
  const double frames = static_cast<double>(threads) * (repeats + 1);
  return elapsed > 0.0 ? frames / (elapsed / 1000.0) : 0.0;
}

// Where a frame's time goes, per stage, for one variant. Reported separately
// from the totals above and never added to them: profiling submits each stage
// in its own command buffer, which serialises what Metal may have been
// overlapping, so the sum here is expected to exceed the unprofiled frame time.
// That gap is information too, and it is printed rather than hidden.
void profile(gpu::Window window0, gpu::Window window2,
             const synthetic::Frame &frame, const std::uint16_t *pixels,
             int repeats, double unprofiled) {
  gpu::stage0_window(window0);
  gpu::stage2_window(window2);
  gpu::profile_stages(true);

  std::vector<SignalPixel> signal;
  std::vector<double> up, zero, one, two, copied, sorted;
  for (int n = 0; n < repeats; n++) {
    if (gpu::find<std::uint16_t>(pixels, signal, frame.height, frame.width) !=
        0) {
      gpu::profile_stages(false);
      std::printf("  profiling: find() failed\n");
      return;
    }
    // The first frame pays for the pipelines and the buffers.
    if (n == 0)
      continue;
    const gpu::StageTimes times = gpu::last_stage_times();
    up.push_back(times.upload);
    zero.push_back(times.stage0);
    one.push_back(times.stage1);
    two.push_back(times.stage2);
    copied.push_back(times.copy);
    sorted.push_back(times.sort);
  }
  gpu::profile_stages(false);

  if (zero.empty()) {
    std::printf("  not enough repeats to profile\n");
    return;
  }

  const Timing tx = summarise(up);
  const Timing s0 = summarise(zero);
  const Timing s1 = summarise(one);
  const Timing s2 = summarise(two);
  const Timing cp = summarise(copied);
  const Timing st = summarise(sorted);

  const double gpu_total = s0.median + s1.median + s2.median;
  const double host_total = cp.median + st.median;
  const double accounted = tx.median + gpu_total + host_total;

  // GPU and host reported apart, and never added into one column. The first
  // version of this printed a GPU sum next to find()'s wall clock as though the
  // two were comparable; they are not, and the difference between them is
  // exactly the host work now itemised below.
  std::printf("\nwhere a frame's time goes, stage0 %s and stage2 %s:\n",
              gpu::name(window0), gpu::name(window2));
  // First, because on CUDA it is a fifth of the frame and on Metal it is zero,
  // which is the difference between the two backends in one line.
  std::printf("  getting the frame to the device:\n");
  std::printf("    %-26s %8.2f ms\n",
              tx.median > 0.0 ? "copied" : "nothing to copy", tx.median);

  // How much the sum is worth depends on how it was obtained, and that differs
  // by backend: CUDA records events in the one stream and does not perturb the
  // schedule, Metal submits each stage separately and gives up the overlap.
  // Saying so here rather than leaving it in a header nobody has open.
  const bool serialised = std::strcmp(gpu::backend(), "Metal") == 0;
  std::printf("  gpu, per stage%s:\n",
              serialised ? " (submitted alone, so read the shares not the sum)"
                         : "");
  const double shares[3] = {s0.median, s1.median, s2.median};
  const char *labels[3] = {"stage0  7x7 dispersion", "stage1  5x5 erode",
                           "stage2  11x11 Poisson"};
  for (int n = 0; n < 3; n++) {
    const double share = gpu_total > 0.0 ? 100.0 * shares[n] / gpu_total : 0.0;
    std::printf("    %-26s %8.2f ms  %5.1f%%\n", labels[n], shares[n], share);
  }
  std::printf("    %-26s %8.2f ms\n", "sum", gpu_total);

  std::printf("  host, wall clock:\n");
  std::printf("    %-26s %8.2f ms\n", "read the packed list", cp.median);
  std::printf("    %-26s %8.2f ms\n", "order it by index", st.median);
  std::printf("    %-26s %8.2f ms\n", "sum", host_total);

  // Everything against the frame, so that anything still missing is visible
  // rather than left as a subtraction for the reader. It was 2.7 ms of a
  // 12.1 ms CUDA frame once, and it turned out to be the upload now shown
  // above.
  std::printf("  a whole unprofiled frame took %.2f ms; the above accounts "
              "for %.2f (%.0f%%)\n",
              unprofiled, accounted,
              unprofiled > 0.0 ? 100.0 * accounted / unprofiled : 0.0);
  // Only meaningful where the split did not change the schedule; on Metal the
  // sum is expected to exceed the frame and the difference says nothing.
  const double missing = serialised ? 0.0 : unprofiled - accounted;
  if (missing > 0.2) {
    std::printf("  %.2f ms is unaccounted for -- launch latency, or something "
                "worth finding\n",
                missing);
  }
}

} // namespace

int main(int argc, char **argv) {
  std::size_t height = 4362;
  std::size_t width = 4148;
  int repeats = 20;
  int threads = 1;

  if (argc >= 3 && argc <= 5) {
    height = static_cast<std::size_t>(std::atol(argv[1]));
    width = static_cast<std::size_t>(std::atol(argv[2]));
    if (argc >= 4)
      repeats = std::atoi(argv[3]);
    if (argc == 5)
      threads = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::printf("usage: bench_dext_gpu [height width [repeats [threads]]]\n");
    return 2;
  }

  if (height == 0 || width == 0 || repeats < 1 || threads < 1) {
    std::printf("height, width, repeats and threads must all be positive\n");
    return 2;
  }

  if (!gpu::available()) {
    std::printf("no %s device available\n", gpu::backend());
    return 77;
  }

  // Every row below names its own configuration, so find()'s running
  // commentary would only interleave duplicate lines between the numbers.
  gpu::report_windows(false);

  const std::size_t pixels = height * width;
  std::printf("%zu x %zu, %.1f Mpixel, %d repeats, %s backend\n", height, width,
              static_cast<double>(pixels) / 1e6, repeats, gpu::backend());

  // The frame the finder is asked about is the same one the correctness test
  // compares, so a number here is a number about checked code.
  const synthetic::Frame frame =
      synthetic::make_frame("bench", height, width, 11, 18.0, true);

  // One shared allocation for every timed path below, so that all of them
  // measure the frame the tool actually gives the device.
  const Shared shared(frame);
  if (!shared.shared()) {
    std::printf("\nhost_alloc failed, so the frame is in ordinary memory and "
                "every frame below pays for a staging copy that the tool does "
                "not. Read the timings with that in mind.\n");
  }

  const gpu::Window windows[2] = {gpu::Window::Direct, gpu::Window::Tile};

  // Every combination, and the results of every combination compared: a faster
  // configuration that computes something else is not faster.
  Timing timings[2][2];
  std::vector<SignalPixel> reference;
  bool disagreed = false;

  std::printf("\none frame at a time, by window:\n");
  for (int a = 0; a < 2; a++) {
    for (int b = 0; b < 2; b++) {
      std::vector<SignalPixel> signal;
      timings[a][b] = time_windows(windows[a], windows[b], frame,
                                   shared.pixels(), repeats, &signal);
      if (reference.empty())
        reference = signal;
      else if (!identical(reference, signal))
        disagreed = true;
    }
  }

  if (disagreed) {
    std::printf("\nthe configurations disagree about what they found. The "
                "timings are meaningless; run ctest -R dext_gpu.\n");
    return 1;
  }

  for (int a = 0; a < 2; a++) {
    for (int b = 0; b < 2; b++) {
      char label[64];
      std::snprintf(label, sizeof(label), "stage0 %-6s stage2 %-6s",
                    gpu::name(windows[a]), gpu::name(windows[b]));
      report(label, timings[a][b], pixels);
    }
  }

  // Which won, for the profile and the summary below.
  int best_a = 0;
  int best_b = 0;
  for (int a = 0; a < 2; a++) {
    for (int b = 0; b < 2; b++) {
      if (timings[a][b].median > 0.0 &&
          (timings[best_a][best_b].median <= 0.0 ||
           timings[a][b].median < timings[best_a][best_b].median)) {
        best_a = a;
        best_b = b;
      }
    }
  }

  // Aggregate throughput, which is what -j delivers and is a different question
  // from the latencies above. Only run when asked: it takes threads * repeats
  // frames per configuration.
  if (threads > 1) {
    std::printf("\n%d workers at once, aggregate:\n", threads);
    for (int a = 0; a < 2; a++) {
      for (int b = 0; b < 2; b++) {
        const double rate =
            time_threaded(windows[a], windows[b], frame, repeats, threads);
        const double alone =
            timings[a][b].median > 0.0 ? 1000.0 / timings[a][b].median : 0.0;
        std::printf("  stage0 %-6s stage2 %-6s %7.1f frame/s  %5.2f Gpixel/s  "
                    "(%.1fx one worker)\n",
                    gpu::name(windows[a]), gpu::name(windows[b]), rate,
                    rate * static_cast<double>(pixels) / 1e9,
                    alone > 0.0 ? rate / alone : 0.0);
      }
    }
    std::printf("  a configuration that scales worse than another is competing "
                "for memory, not for compute\n");
  }

  // The CPU, for scale. Fewer repeats because it is much slower and the point
  // is the order of magnitude, not a tight figure.
  const int cpu_repeats = repeats > 4 ? 4 : repeats;
  std::vector<SignalPixel> cpu_signal;
  dext_scratch<std::uint16_t> scratch;
  dext<std::uint16_t>(frame.pixels.data(), cpu_signal, height, width, scratch);

  std::vector<double> cpu_samples;
  for (int n = 0; n < cpu_repeats; n++) {
    const Clock::time_point started = Clock::now();
    dext<std::uint16_t>(frame.pixels.data(), cpu_signal, height, width,
                        scratch);
    cpu_samples.push_back(milliseconds(started, Clock::now()));
  }

  std::printf("\ncpu, one thread, for scale:\n");
  const Timing cpu = summarise(cpu_samples);
  report("dext()", cpu, pixels);

  // Checked before it is claimed.
  if (!identical(cpu_signal, reference)) {
    std::printf("\nthe cpu found %zu signal pixels and the device %zu. The "
                "timings above are of something unverified; run "
                "ctest -R dext_gpu.\n",
                cpu_signal.size(), reference.size());
    return 1;
  }
  std::printf("\n%zu signal pixels, and every implementation agrees.\n",
              reference.size());
  // Which matters for reading the profile below. Two of its lines -- copying
  // the packed list back and ordering it -- are proportional to this count,
  // and the planted frame is far denser than diffraction: a real sweep of this
  // detector reported 194150 connected components over 1800 frames, which is
  // of the order of a thousand signal pixels a frame, not a hundred thousand.
  // The device stages barely notice, since they look at every pixel either
  // way; the host lines are inflated by whatever that ratio is.
  std::printf("  that is %.3f%% of the frame. Real diffraction runs nearer "
              "0.01%%, so the two host\n  lines in the profile below are "
              "weighted about %.0fx too heavily for a real frame.\n",
              100.0 * static_cast<double>(reference.size()) /
                  static_cast<double>(pixels),
              static_cast<double>(reference.size()) /
                  (0.0001 * static_cast<double>(pixels)));

  profile(windows[best_a], windows[best_b], frame, shared.pixels(),
          repeats > 6 ? 6 : repeats, timings[best_a][best_b].median);

  std::printf("\nfastest here: stage0 %s, stage2 %s at %.2f ms\n",
              gpu::name(windows[best_a]), gpu::name(windows[best_b]),
              timings[best_a][best_b].median);
  if (cpu.median > 0.0 && timings[best_a][best_b].median > 0.0) {
    std::printf("that is %.1fx one CPU thread.\n",
                cpu.median / timings[best_a][best_b].median);
  }
  return 0;
}
