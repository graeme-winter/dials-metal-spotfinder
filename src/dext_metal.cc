// The Metal backend for gpu::find, for Apple silicon.
//
// The same three stages as dext_cuda.cu, dispatched through metal-cpp. What is
// different is not the calculation -- src/dext_metal.metal is a transcription
// of the kernels, window for window -- but the memory model underneath it:
//
//   * The device shares memory with the host, so there is no upload and no
//     download. gpu::host_alloc hands back the contents of a shared MTLBuffer,
//     the frame is decompressed straight into it, and the kernels read it where
//     it lies. The packed list comes back the same way. On CUDA the equivalent
//     call returns page-locked memory so the copies can at least be
//     asynchronous; here the copies are gone, which is most of why this is
//     worth doing on an M-series machine at all.
//
//   * There is no cudaStream_t. A command queue per worker thread is the
//     closest thing, and is what this holds: command buffers within one queue
//     are ordered, so a shared queue would serialise the workers against each
//     other. Command buffers are cheap and made per frame.
//
//   * The three dispatches go into one compute encoder. Metal's default
//     dispatch type is serial, which means the encoder inserts the barriers
//     between them, so stage0 -> stage1 -> stage2 is ordered without saying so.
//
// Sixteen bit only. dext.hh gives real_t<std::uint32_t> = double and Metal has
// no double precision at all, so a 32-bit path would run a different
// calculation from the CPU's and disagree with it. find<std::uint32_t> throws
// rather than pretending; see the note in dext_metal.metal.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "dext_gpu.hh"
#include "dext_gpu_internal.hh"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "signal_order.hh"

// The compiled shader, embedded by CMake so that the binary carries its own
// kernels and there is nothing to install beside it or find at runtime.
extern "C" const unsigned char spotfinder_dext_metallib[];
extern "C" const unsigned long spotfinder_dext_metallib_size;

namespace gpu {
namespace {

// Metal reports failures as an NS::Error rather than a return code, and the
// localised description is the only part worth showing.
[[noreturn]] void fail(const char *what, NS::Error *error) {
  std::string message = std::string("Metal: ") + what;
  if (error != nullptr && error->localizedDescription() != nullptr)
    message += std::string(": ") + error->localizedDescription()->utf8String();
  throw std::runtime_error(message);
}

NS::String *literal(const char *text) {
  return NS::String::string(text, NS::UTF8StringEncoding);
}

// metal-cpp returns autoreleased objects from most methods, exactly as the
// Objective-C API does, and without a pool in scope they accumulate for the
// life of the process. One frame's command buffer and encoder is not much; ten
// thousand frames of them is.
class Pool {
public:
  Pool() : pool_(NS::AutoreleasePool::alloc()->init()) {}
  ~Pool() { pool_->release(); }
  Pool(const Pool &) = delete;
  Pool &operator=(const Pool &) = delete;

private:
  NS::AutoreleasePool *pool_;
};

// The device and the three pipeline states, built once for the process. Nothing
// here is per thread: MTLDevice and MTLComputePipelineState are immutable once
// created and may be used from any thread, and compiling the library on every
// worker would be the same work done -j times.
class Device {
public:
  static Device &instance() {
    static Device device;
    return device;
  }

  MTL::Device *device() const { return device_; }
  bool usable() const { return device_ != nullptr; }

  // Built on first use rather than in the constructor, so that available() can
  // answer without paying for a library compile, and so that a failure to build
  // the pipelines is reported from find() where there is somewhere to put it.
  void ensure_pipelines() {
    std::call_once(once_, [this] { build(); });
    if (stage0_direct_ == nullptr || stage0_tile_ == nullptr ||
        stage2_tile_ == nullptr || stage2_direct_ == nullptr)
      throw std::runtime_error("Metal: the signal kernels are not available");
  }

  // Every variant is built, always. They are a few kilobytes of pipeline state
  // each and building only the ones asked for would mean a process could not
  // time the combinations against each other.
  MTL::ComputePipelineState *stage0(Window window) const {
    return window == Window::Tile ? stage0_tile_ : stage0_direct_;
  }
  MTL::ComputePipelineState *stage1() const { return stage1_; }
  MTL::ComputePipelineState *stage2(Window window) const {
    return window == Window::Tile ? stage2_tile_ : stage2_direct_;
  }

private:
  Device() : device_(MTL::CreateSystemDefaultDevice()) {}

  MTL::ComputePipelineState *pipeline(MTL::Library *library, const char *name) {
    MTL::Function *function = library->newFunction(literal(name));
    if (function == nullptr)
      throw std::runtime_error(std::string("Metal: no kernel named ") + name);
    NS::Error *error = nullptr;
    MTL::ComputePipelineState *state =
        device_->newComputePipelineState(function, &error);
    function->release();
    if (state == nullptr)
      fail(name, error);

    // stage2's tile is the threadgroup plus its halo, so the kernel is only
    // correct at 32 x 32. The limit is per pipeline, not per device -- it falls
    // with register pressure -- so it is checked here rather than assumed.
    const NS::UInteger limit = state->maxTotalThreadsPerThreadgroup();
    if (limit < 1024) {
      state->release();
      throw std::runtime_error(
          std::string("Metal: ") + name + " will only take " +
          std::to_string(limit) +
          " threads per threadgroup, and the kernels need 1024");
    }
    return state;
  }

  void build() {
    if (device_ == nullptr)
      return;
    Pool pool;

    dispatch_data_t data = dispatch_data_create(
        spotfinder_dext_metallib, spotfinder_dext_metallib_size, nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error *error = nullptr;
    MTL::Library *library = device_->newLibrary(data, &error);
    dispatch_release(data);
    if (library == nullptr)
      fail("loading the embedded shader library", error);

    stage0_direct_ = pipeline(library, "dext_stage0_u16");
    stage0_tile_ = pipeline(library, "dext_stage0_tile_u16");
    stage1_ = pipeline(library, "dext_stage1_u16");
    stage2_tile_ = pipeline(library, "dext_stage2_u16");
    stage2_direct_ = pipeline(library, "dext_stage2_direct_u16");
    library->release();
  }

  MTL::Device *device_ = nullptr;
  std::once_flag once_;
  MTL::ComputePipelineState *stage0_direct_ = nullptr;
  MTL::ComputePipelineState *stage0_tile_ = nullptr;
  MTL::ComputePipelineState *stage1_ = nullptr;
  MTL::ComputePipelineState *stage2_tile_ = nullptr;
  MTL::ComputePipelineState *stage2_direct_ = nullptr;
};

// host_alloc hands out the contents pointer of a shared buffer and the caller
// keeps only that, so find() needs a way back from the pointer to the buffer it
// belongs to. There is one entry per worker thread, looked up once per frame,
// so a mutex around a map costs nothing measurable.
//
// The alternative -- newBuffer(bytesNoCopy:) over an ordinary allocation --
// needs the pointer page-aligned and the length a page multiple, and would
// still need this map to avoid wrapping the same allocation every frame.
class Registry {
public:
  static Registry &instance() {
    static Registry registry;
    return registry;
  }

  void *add(MTL::Buffer *buffer) {
    void *pointer = buffer->contents();
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_[pointer] = buffer;
    return pointer;
  }

  // Null if the caller allocated the frame some other way, which happens when
  // host_alloc failed and signal.cc fell back to new[]. find() copies in that
  // case rather than refusing.
  MTL::Buffer *lookup(const void *pointer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = buffers_.find(const_cast<void *>(pointer));
    return found == buffers_.end() ? nullptr : found->second;
  }

  void remove(void *pointer) {
    MTL::Buffer *buffer = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto found = buffers_.find(pointer);
      if (found == buffers_.end())
        return;
      buffer = found->second;
      buffers_.erase(found);
    }
    buffer->release();
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<void *, MTL::Buffer *> buffers_;
};

// Leaves the queue idle however the scope is left, for the reason StreamGuard
// exists in dext_cuda.cu: without it, an exception thrown after the commit
// returns from find() with the kernels still running over buffers the worker is
// about to reuse, and the damage surfaces somewhere else entirely.
class CommandGuard {
public:
  explicit CommandGuard(MTL::CommandBuffer *buffer) : buffer_(buffer) {}
  ~CommandGuard() {
    if (buffer_ != nullptr)
      buffer_->waitUntilCompleted();
  }
  void disarm() { buffer_ = nullptr; }
  CommandGuard(const CommandGuard &) = delete;
  CommandGuard &operator=(const CommandGuard &) = delete;

private:
  MTL::CommandBuffer *buffer_;
};

// The device side of one worker thread: a queue and three buffers, allocated on
// the first frame and reused after that. Freed when the thread ends.
//
// One queue per thread and not one shared: command buffers committed to a queue
// execute in commit order, so a shared queue would make -j workers take turns.
// macOS allows on the order of 64 queues per device, which is well above any
// sensible -j.
class Workspace {
public:
  ~Workspace() {
    // As in dext_cuda.cu: nothing may be in flight when the buffers go, and a
    // thread that left find() through an exception can still have work queued.
    if (queue_ != nullptr) {
      release_buffers();
      queue_->release();
    }
  }

  void reserve(std::size_t bytes) {
    MTL::Device *const device = Device::instance().device();
    if (queue_ == nullptr) {
      queue_ = device->newCommandQueue();
      if (queue_ == nullptr)
        throw std::runtime_error("Metal: could not create a command queue");
    }
    if (counter_ == nullptr) {
      counter_ = device->newBuffer(sizeof(std::uint32_t),
                                   MTL::ResourceStorageModeShared);
      if (counter_ == nullptr)
        throw std::runtime_error(
            "Metal: could not allocate the signal counter");
    }
    if (bytes <= capacity_)
      return;

    release_buffers();
    // tmp_ is never read by the host, so it is private; out_ carries the packed
    // list back and has to be shared. On unified memory the difference is
    // whether the host may look, not where the pages live.
    tmp_ = device->newBuffer(bytes, MTL::ResourceStorageModePrivate);
    out_ = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (tmp_ == nullptr || out_ == nullptr)
      throw std::runtime_error("Metal: could not allocate the working images");
    capacity_ = bytes;
  }

  // Only used when host_alloc failed and the frame is in ordinary memory.
  MTL::Buffer *staging(std::size_t bytes) {
    if (staging_ != nullptr && staging_->length() >= bytes)
      return staging_;
    if (staging_ != nullptr)
      staging_->release();
    staging_ = Device::instance().device()->newBuffer(
        bytes, MTL::ResourceStorageModeShared);
    if (staging_ == nullptr)
      throw std::runtime_error("Metal: could not allocate a staging buffer");
    return staging_;
  }

  MTL::CommandQueue *queue() const { return queue_; }
  MTL::Buffer *tmp() const { return tmp_; }
  MTL::Buffer *out() const { return out_; }
  MTL::Buffer *counter() const { return counter_; }

private:
  void release_buffers() {
    if (tmp_ != nullptr)
      tmp_->release();
    if (out_ != nullptr)
      out_->release();
    if (staging_ != nullptr)
      staging_->release();
    tmp_ = out_ = staging_ = nullptr;
    capacity_ = 0;
  }

  MTL::CommandQueue *queue_ = nullptr;
  MTL::Buffer *counter_ = nullptr;
  MTL::Buffer *tmp_ = nullptr;
  MTL::Buffer *out_ = nullptr;
  MTL::Buffer *staging_ = nullptr;
  std::size_t capacity_ = 0;
};

Workspace &workspace() {
  static thread_local Workspace space;
  return space;
}

} // namespace

const char *backend() { return "Metal"; }

bool available() { return Device::instance().usable(); }

std::size_t memory_free() {
  MTL::Device *const device = Device::instance().device();
  if (device == nullptr)
    return 0;
  // Unified memory, so this is not a separate pool: it is what the process may
  // still allocate before the system starts paying for it in swap. Reported
  // rather than enforced, as on CUDA.
  const std::size_t budget = device->recommendedMaxWorkingSetSize();
  const std::size_t used = device->currentAllocatedSize();
  return budget > used ? budget - used : 0;
}

void *host_alloc(std::size_t bytes) {
  MTL::Device *const device = Device::instance().device();
  if (device == nullptr || bytes == 0)
    return nullptr;
  MTL::Buffer *const buffer =
      device->newBuffer(bytes, MTL::ResourceStorageModeShared);
  if (buffer == nullptr)
    return nullptr;
  return Registry::instance().add(buffer);
}

void host_free(void *pointer) {
  if (pointer != nullptr)
    Registry::instance().remove(pointer);
}

template <typename T>
int find(const T *image_in, std::vector<SignalPixel> &signal_out,
         std::size_t height, std::size_t width) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4, "16 or 32 bit pixels only");

  if constexpr (sizeof(T) != 2) {
    // Not -1: a bad argument and an unsupported pixel type are different
    // things, and the second wants an explanation. dext.hh gives real_t =
    // double here, and Metal has no double precision, so the honest answers are
    // the CPU path or a deliberate decision to run the dispersion test in
    // float -- not a silent one taken here.
    (void)image_in;
    (void)height;
    (void)width;
    signal_out.clear();
    throw std::runtime_error(
        "the Metal signal calculation supports 16-bit data only, because "
        "Apple GPUs have no double precision; run 32-bit frames without -gpu");
  } else {
    signal_out.clear();
    if (image_in == nullptr || height == 0 || width == 0)
      return -1;
    // The kernels index in int, as the CUDA ones do.
    if (height > 0x7fffffffu / width)
      return -1;

    Device &backend_device = Device::instance();
    if (!backend_device.usable())
      throw std::runtime_error("Metal: no usable device");
    backend_device.ensure_pipelines();

    Pool pool;

    const std::size_t bytes = height * width * sizeof(T);
    Workspace &space = workspace();
    space.reserve(bytes);

    const bool timing = profile_stages();
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

    // The frame is already in a shared buffer if signal.cc got it from
    // host_alloc, which is the whole point; if it is not, it is copied once
    // into a staging buffer rather than refused.
    //
    // So `upload` is normally zero here, and that zero is the headline
    // difference from CUDA, where the same field is a fifth of the frame. A
    // non-zero value means host_alloc failed and every frame is paying for a
    // copy that should not exist -- worth seeing rather than absorbing.
    double upload_ms = 0.0;
    MTL::Buffer *in = Registry::instance().lookup(image_in);
    if (in == nullptr) {
      const auto upload_started = mark();
      in = space.staging(bytes);
      std::memcpy(in->contents(), image_in, bytes);
      upload_ms = since(upload_started);
    }

    // stage0's mask is dead once stage1 has read it, so the same buffer carries
    // the packed list. Sixteen bytes an entry against two a pixel is where the
    // one-in-eight bound comes from.
    MTL::Buffer *const scratch = space.out();
    MTL::Buffer *const tmp = space.tmp();
    MTL::Buffer *const counter = space.counter();
    const std::uint32_t capacity =
        static_cast<std::uint32_t>(bytes / sizeof(SignalPixel));
    const int rows = static_cast<int>(height);
    const int columns = static_cast<int>(width);

    // Shared storage, and nothing is in flight, so this is a plain store rather
    // than the memset-on-the-queue CUDA needs.
    *static_cast<std::uint32_t *>(counter->contents()) = 0;

    const Window window0 = stage0_window();
    const Window window2 = stage2_window();

    internal::announce_windows(window0, window2);

    const MTL::Size group(32, 32, 1);
    const MTL::Size grid((width + 31) / 32, (height + 31) / 32, 1);

    // Encoding each stage, so that the shape of the submission -- one command
    // buffer, or one per stage for profiling -- is the only thing that varies
    // and the dispatches themselves cannot drift between the two.
    //
    // dispatchThreadgroups and not dispatchThreads: the second clips the last
    // group to the grid, and stage2's tile assumes a full 32 x 32 group. The
    // kernels do their own bounds tests, as the CUDA ones do.
    const auto encode_stage0 = [&](MTL::ComputeCommandEncoder *encoder) {
      encoder->setComputePipelineState(backend_device.stage0(window0));
      encoder->setBuffer(in, 0, 0);
      encoder->setBuffer(scratch, 0, 1);
      encoder->setBytes(&rows, sizeof(rows), 2);
      encoder->setBytes(&columns, sizeof(columns), 3);
      encoder->dispatchThreadgroups(grid, group);
    };
    const auto encode_stage1 = [&](MTL::ComputeCommandEncoder *encoder) {
      encoder->setComputePipelineState(backend_device.stage1());
      encoder->setBuffer(in, 0, 0);
      encoder->setBuffer(scratch, 0, 1);
      encoder->setBuffer(tmp, 0, 2);
      encoder->setBytes(&rows, sizeof(rows), 3);
      encoder->setBytes(&columns, sizeof(columns), 4);
      encoder->dispatchThreadgroups(grid, group);
    };
    const auto encode_stage2 = [&](MTL::ComputeCommandEncoder *encoder) {
      encoder->setComputePipelineState(backend_device.stage2(window2));
      encoder->setBuffer(in, 0, 0);
      encoder->setBuffer(tmp, 0, 1);
      encoder->setBuffer(scratch, 0, 2);
      encoder->setBuffer(counter, 0, 3);
      encoder->setBytes(&capacity, sizeof(capacity), 4);
      encoder->setBytes(&rows, sizeof(rows), 5);
      encoder->setBytes(&columns, sizeof(columns), 6);
      encoder->dispatchThreadgroups(grid, group);
    };

    // Submit one stage in its own command buffer and return the GPU time it
    // took. Only reached when profiling: three submissions and three waits per
    // frame is not what a production frame should pay, and the serialisation
    // changes the very thing being measured, so what this gives is where the
    // time goes and not how long a frame takes.
    const auto run_alone = [&](const char *what, const auto &encode) -> double {
      MTL::CommandBuffer *const buffer = space.queue()->commandBuffer();
      if (buffer == nullptr)
        throw std::runtime_error("Metal: could not create a command buffer");
      CommandGuard guard(buffer);
      MTL::ComputeCommandEncoder *const encoder =
          buffer->computeCommandEncoder();
      if (encoder == nullptr)
        throw std::runtime_error("Metal: could not create a compute encoder");
      encode(encoder);
      encoder->endEncoding();
      buffer->commit();
      buffer->waitUntilCompleted();
      guard.disarm();
      if (buffer->status() != MTL::CommandBufferStatusCompleted)
        fail(what, buffer->error());
      return (buffer->GPUEndTime() - buffer->GPUStartTime()) * 1000.0;
    };

    if (timing) {
      StageTimes times;
      times.upload = upload_ms;
      times.stage0 = run_alone("stage0", encode_stage0);
      times.stage1 = run_alone("stage1", encode_stage1);
      times.stage2 = run_alone("stage2", encode_stage2);
      internal::set_stage_times(times);
    } else {
      MTL::CommandBuffer *const commands = space.queue()->commandBuffer();
      if (commands == nullptr)
        throw std::runtime_error("Metal: could not create a command buffer");

      // Armed before the first dispatch is encoded, so that no exit path
      // leaves work in flight.
      CommandGuard idle(commands);

      // One encoder for all three: the default dispatch type is serial, so
      // Metal puts the barriers between them and stage1 cannot start before
      // stage0 has finished writing the mask it reads.
      MTL::ComputeCommandEncoder *const encoder =
          commands->computeCommandEncoder();
      if (encoder == nullptr)
        throw std::runtime_error("Metal: could not create a compute encoder");

      encode_stage0(encoder);
      encode_stage1(encoder);
      encode_stage2(encoder);

      encoder->endEncoding();
      commands->commit();
      commands->waitUntilCompleted();
      idle.disarm();

      if (commands->status() != MTL::CommandBufferStatusCompleted)
        fail("running the signal calculation", commands->error());
    }
    const std::uint32_t found =
        *static_cast<const std::uint32_t *>(counter->contents());
    if (found > capacity)
      return -2;
    if (found == 0)
      return 0;

    // Shared storage again, so this is a copy out of the buffer and not a
    // download. Only the part that was written.
    //
    // `timing`, `mark` and `since` come from the top of the function, where the
    // staging copy needed them too.
    const auto copy_started = mark();
    signal_out.resize(found);
    std::memcpy(signal_out.data(), scratch->contents(),
                found * sizeof(SignalPixel));
    const double copy_ms = since(copy_started);

    // The SIMD groups emit in whatever order they finish; the grouping needs
    // ascending index, and relies on k - 1 being the entry immediately before.
    //
    // This was std::sort and it was the most expensive thing in the function:
    // 5.8 ms of a 6.7 ms frame at 126,002 signal pixels, against 1.7 for the
    // bucket pass. Reused across frames, so after the first it allocates
    // nothing.
    static thread_local signal_order::scratch order;
    const auto sort_started = mark();
    signal_order::by_index(signal_out, order);

    if (timing) {
      StageTimes times = last_stage_times();
      times.upload = upload_ms;
      times.copy = copy_ms;
      times.sort = since(sort_started);
      internal::set_stage_times(times);
    }
    return 0;
  }
}

template int find<std::uint16_t>(const std::uint16_t *,
                                 std::vector<SignalPixel> &, std::size_t,
                                 std::size_t);
template int find<std::uint32_t>(const std::uint32_t *,
                                 std::vector<SignalPixel> &, std::size_t,
                                 std::size_t);

} // namespace gpu
