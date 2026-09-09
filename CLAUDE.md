# CLAUDE.md

Notes for working in this repository. `README.md` is the document for people
building and running the tool; this one records the conventions, the invariants
that are expensive to rediscover, and what is not yet proven.

## What this is

One C++20 tool: `dials-metal-find-spots` reads an NXmx HDF5 series, runs the
DIALS extended dispersion threshold over each frame on the CPU or on a GPU,
groups the surviving pixels six-connected in three dimensions, and writes a
DIALS reflection table for `dials.index`.

Extracted from redhorn, a detector-stream relay and analyser that had this as
one of its tools; the object store, the wire format and the rest of that
machinery stayed behind. Worth knowing only for one reason: if something here
looks like the remains of a larger design -- the `Series` / `Reader` split with
a single implementation, say -- check this file before removing it, because some
of those seams are load-bearing for a reason that is not obvious when there is
one implementation of them.

## Layout

| Path | What it is |
| --- | --- |
| `src/find_spots.cc` | `main`: option parsing, the thread pool, the chunked ordering |
| `src/series.hh` | `Series`, `Reader`, `Frame`: what the tool reads, above HDF5 |
| `src/nxmx.cc` | VDS unpacking and `H5Dread_chunk`; the only file including hdf5.h |
| `src/decompress.{hh,cc}` | bslz4, lz4 and uncompressed chunks, via bitshuffle |
| `src/dext.{hh,cc}` | Extended dispersion threshold, CPU. **Given code** |
| `src/dext_gpu.hh` | The device interface. One header, two implementations, one linked |
| `src/dext_gpu.cc` | The backend-agnostic half: window choice, profiling, splits |
| `src/dext_gpu_internal.hh` | What a backend calls into that, and a caller does not |
| `src/dext_cuda.cu` | The same on CUDA, typed from dext.hh's traits |
| `src/dext_metal.{cc,metal}` | The same on Apple silicon, 16-bit only, via metal-cpp |
| `src/signal_pixel.hh` | What the threshold emits: 16 bytes a surviving pixel |
| `src/signal_order.{hh,cc}` | Signal pixels into index order, O(n). Host, shared |
| `src/dials_spots.{hh,cc}` | Six-connected grouping in 3D, DIALS' centroids, streaming |
| `src/refl.{hh,cc}` | A DIALS reflection table: msgpack, written by hand |
| `src/expt.{hh,cc}` | The scan range, panel size and identifier out of an .expt |
| `src/queue.hh` | Bounded blocking queue, the backpressure between threads |
| `tests/make_test_nxmx.py` | A synthetic series with reflections planted across frames |
| `tests/check_spots.py` | That series' manifest against the table it produced |
| `tests/check_refl.py` | Validates a .refl, and diffs two of them by centroid |
| `tests/synthetic_frame.hh` | The frames the device test and the benchmark share |
| `third_party/bitshuffle` | Submodule; lz4 comes with it |

## Commands

```sh
git submodule update --init
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
tests/regression.sh
```

`tests/regression.sh` is the one command to run before and after any change. It
generates the fixtures, runs `ctest`, and then checks the whole path from HDF5 to
a reflection table. No detector, no server, and it says so and skips where there
is no device.

**There are no recorded baselines here, deliberately.** The fixture plants
reflections at known positions, so the expectation is the plant rather than a
previous run: `tests/check_spots.py` compares the table against the manifest.
That is worth more than a baseline, because a baseline records what the code did
and this records what it should do -- and it means no `--record` step that can be
run without reading the diff.

## Before changing anything

**When a performance or behaviour problem is reported, check the invocation
before touching the code.** This is a standing instruction from the person who
owns this repository, and it was earned in redhorn: a report of poor threading
turned out to be `-t 16` where `-j 16` was meant -- `-t` is the idle timeout.
Two clues were on screen and missed. The startup line said four threads while
the report described sixteen, and `-t 1` and `-t 16` gave identical timings,
which no real change in thread count can do. A rewrite of the HDF5 read path was
written, shipped and reverted before the flag was noticed.

Read the command as given, compare every flag against `usage()`, and compare
what the program printed about itself against what the report claims. If a change
in a parameter produces no change at all, doubt the parameter before the code.

**And when a measurement rules a cause out, believe it.** From the same episode:
a route that touched no HDF5 took the same wall clock, which excluded HDF5 as
the bottleneck before any code was written. That was said twice and overridden
twice.

## The threshold

**The finder emits a packed list, not a mask.** `dext` and `gpu::find` fill a
`std::vector<SignalPixel>` -- index, value, background, population, 16 bytes --
and leave the frame untouched. A frame is sparse, so a mask spends a pass
writing zeros and another scanning for the survivors, and on the device it made
`gpu::find` download the whole frame to recover about one per cent of it. The
list also carries the local background out with the pixels, which a mask never
could.

* `dext.cc` and `dext.hh` are **given code**. Do not restructure them, and do not
  re-derive `accumulator_t` / `real_t`: the device kernels take their types from
  those traits so the implementations stay in step by construction.
* **Ascending by index is part of the contract.** The CPU emits in scan order and
  gets it free; the device sorts before returning. The grouping depends on it --
  its left-neighbour test is only O(1) because `k - 1` can only be the entry
  immediately before.
* Overflow of `p * p` in the 7 x 7 accumulator is **deliberately not guarded**:
  the data are sparse, values above about 1000 are rare, and both
  implementations behave alike, which is what matters.

  Worth having the numbers to hand, because this looks like a bug until they are
  written down, and I reasoned my way from a synthetic frame to the wrong
  conclusion about it once already. The count and the sum are exact whatever the
  table does, since their window values are bounded by 49 and 49 * 65535 and the
  summed-area identity holds in Z / 2^32. The sum of squares has a limit: one
  pixel at 65533 is 4.29e9, just inside; two pixels need 46341 each; a uniform
  window needs 9362 a pixel. A spot with a thousand counts in its brightest
  pixel puts 7e6 in its window, and the busiest window that can be built out of
  reflections -- two 2500-count spots two pixels apart -- reaches 1.4e8, still a
  factor of 30 short. Saturated pixels never enter the sums at all, being masked
  at `>= max() - 1`, which is what keeps the top of the range out of it.

  `tests/test_dext_squares.cc` measures that factor rather than asserting it, so
  a change to the kernel size or to the masking fails there. **Do not widen the
  accumulator.** Metal has no 64-bit integer arithmetic, the tile variant's three
  threadgroup tables would grow by a third of the 32 KB an Apple threadgroup
  gets, and the 32-bit form is what has been run on real data.
* **Capacity is checked, not assumed.** The packed list reuses stage0's mask
  buffer, which holds one entry per eight pixels at 16 bits. A frame denser than
  that is refused with -2 rather than truncated.
* Only 16 and 32 bit are instantiated. 8-bit is refused with a message rather
  than promoted.
* `-gpu` fails if no backend was built in or no device is present, rather than
  falling back: an explicit request answered by the other implementation would
  misattribute the results.

## The device backends

* **There is one GPU interface and two implementations of it**, `dext_gpu.hh`
  with `dext_cuda.cu` and `dext_metal.cc` behind it. Exactly one is compiled in;
  configuring both is a hard error in `CMakeLists.txt`, because the linker would
  otherwise pick one and `-gpu` would run a device nobody chose. `SPOTFINDER_GPU`
  at the call sites means "there is a backend"; `SPOTFINDER_CUDA` and
  `SPOTFINDER_METAL` mean which, and stay in `CMakeLists.txt` and in the
  backend's own file.
* **The Metal backend is 16-bit only, and that is a decision rather than a gap.**
  `real_t<std::uint32_t>` is `double` and Apple GPUs have no double precision at
  all -- not slow, absent. A 32-bit path would have to run the dispersion test in
  float, which is a different calculation from the CPU's, and it would disagree
  in a way that reads as a bug. `find<std::uint32_t>` throws and says so. Do not
  "fix" this by narrowing the traits.
* **`src/dext_metal.metal` is compiled with `-fno-fast-math`.** Fast math is on
  by default in the Metal compiler and the entire claim this backend makes is
  that it agrees with `dext.cc` bit for bit. `precise::sqrt` is used for the same
  reason. If `tests/test_dext_gpu.cc` starts failing on the `background` field by
  a bit or two, that flag is the first thing to check.
* **The shader is compiled at build time and embedded**, not compiled from source
  at run time. `newLibraryWithSource` needs the Metal compiler present on the
  machine that runs the binary -- a beamline machine -- and a shader that first
  fails to compile at the first frame of an acquisition is a bad trade for the
  second it saves. It also means `--version` cannot disagree with its own
  kernels.
* **metal-cpp returns autoreleased objects**, exactly as the Objective-C API
  does. `find()` holds an `NS::AutoreleasePool` for its whole body. Without it
  the command buffer and encoder of every frame accumulate for the life of the
  process, which at a few thousand frames is a leak that looks like a slow one.
* **One command queue per worker thread, not one shared.** Command buffers
  committed to a queue execute in commit order, so a shared queue would make `-j`
  workers take turns on the GPU. Same reasoning as the per-thread
  `cudaStream_t`; macOS allows on the order of 64 queues, well above any sensible
  `-j`.
* **`stage2` is hard-wired to a 32 x 32 threadgroup** in both backends: the tile
  it builds is the group plus a five-pixel halo, and 32 + 5 + 5 is where `kTile`
  comes from. The Metal side checks `maxTotalThreadsPerThreadgroup` on the
  pipeline rather than assuming it, because that limit is per pipeline and falls
  with register pressure.
* **`dispatchThreadgroups`, never `dispatchThreads`.** The second clips the last
  threadgroup to the grid, and `stage2`'s tile assumes a full group. The kernels
  do their own bounds tests, as the CUDA ones do.
* **The device must not return early before the ballot.** `stage2` computes an
  `in_range` predicate rather than returning, because a thread that has returned
  cannot take part in `__ballot_sync` -- or in Metal's
  `simd_prefix_exclusive_sum` -- and the result would be silently wrong rather
  than a clean failure.
* **Everything after the window sums lives in `emit_signal`**, shared by both
  stage2 variants, so a change to the Poisson test or to the compacted emit
  cannot land in one and not the other. `tests/test_dext_gpu.cc` runs its whole
  comparison under both variants for the same reason.
* **Both backends have all four window kernels**, chosen at run time by
  `SPOTFINDER_GPU_STAGE0` and `SPOTFINDER_GPU_STAGE2`, and the two devices do not
  agree about `stage0` -- which is why it is a run-time choice and not a decision
  in the source. The table is in the README. Keep the losing variants: each is a
  second, independent way of summing the same window, and the device test running
  all four combinations against the CPU is a real cross-check on the tiles' index
  arithmetic that nothing else provides.
* **The window choice, the profiling flag and the last frame's split live in
  `src/dext_gpu.cc`, not in either backend.** None of them is about a device:
  they are configuration, and two copies is how "tile" ends up spelled two ways.
* **The per-stage split is trustworthy on CUDA and only indicative on Metal.**
  CUDA records events around each kernel in the one stream, which costs nothing
  and does not change the schedule. Metal has no per-dispatch timestamp without
  `MTLCounterSampleBuffer`, so it submits each stage in its own command buffer and
  gives up whatever was overlapping -- there, read the shares and not the sum.
* **GPU time and host wall clock are reported apart and never added.** An earlier
  benchmark printed a GPU sum next to `find()`'s wall clock as though they were
  comparable, which made a frame look 3 ms slower than its stages -- and the
  explanation was the host-side sort, not anything on the device.
* **`signal_order::by_index` takes no frame bound, and that is deliberate.** The
  first version took `height * width` to size the histogram; an understated value
  put indices outside it, which is a heap overflow and not a slow sort. It now
  derives the bucket width from the largest index present.

## The DIALS side

* **Grouping is six-connected in x, y and z, not per frame.** A reflection is
  swept over a degree or two and lands on several frames; grouped per frame it is
  three or four spots with three or four centroids and nothing downstream can
  reassemble them. `--2d` is for stills, and is what
  `spotfinder.force_2d=True` means in DIALS. `tests/regression.sh` checks that
  `--2d` gives exactly three times as many spots on a fixture whose reflections
  span three frames each -- which is the check that would catch the 3D grouping
  quietly degrading, since a degraded run still writes a plausible file.
* **The grouping is streaming, and that is why frames must arrive in order.**
  Only frames `f - 1` and `f` can be connected, so a component with no pixel on
  the frame just added can never grow: it is emitted and its pixels dropped.
  DIALS holds the whole sweep's pixel list and labels it at the end.
* **Frames reach the grouping in order by chunking, not by resequencing.**
  Dispatch a few frames per thread, wait for all of them, sort, group. A reorder
  buffer keyed on a next-expected frame number would have to know that a frame it
  is waiting for will never arrive, and an unallocated chunk is exactly that. The
  barrier costs a fraction of one frame's latency per chunk, and the table comes
  out byte-identical whatever `-j` and `--chunk` are -- which the regression
  script checks, because it is the property that makes the chunking safe.
* **A gap in the frame numbering closes everything open**, rather than being an
  error.
* **z is an array index, not an image number.** The array index of image *n* is
  *n* - 1, an NXmx series numbers its frames from zero, and a sliced import
  starts higher -- which is why `-e` sets `--z-offset` from the scan's
  `image_range`. Getting this wrong indexes nothing and neither file says why.
* **What DIALS computes from the grouping is reproduced, not improved.** The
  centroid is `Shoebox::centroid_valid`, the variance its unbiased standard error
  plus `1/12`, the intensity `Summation` over a shoebox with a zero background,
  and both default filters are applied -- `min_spot_size` (3 for a pixel array
  detector, which is what phil resolves `Auto` to), `max_spot_size` and
  `max_separation`. The `1/12` is a weighting decision of DIALS' own, and a
  different weight is a different refinement.
* All of that reduces to closed forms over the pixel list, because the finder
  fills only the signal pixels of a shoebox and leaves its background at zero.
  **Nothing goes back to the images a second time**, including for the
  shoeboxes.
* **`Shoebox<>` is `Shoebox<float>`**, from `ProfileFloatType` in
  `dials/config.h`. A blob of the right length in double decodes to nonsense.
* **The reflection table format is a memory image behind msgpack headers**, so it
  is portable across neither endianness nor the width of `std::size_t`. DIALS has
  the same property and does not say so; `refl::write` refuses rather than
  writing something that will not read back. A column type name DIALS does not
  know makes the whole file unreadable, so the names are checked against the list
  in `reflection_table_msgpack_adapter.h`.
* **One panel, one experiment.** A segmented detector would need a panel number
  per spot and several experiments would need the `id` column to mean something.
  Each is refused rather than guessed at.
* **The `.expt` is read and never written.** The beam, the goniometer and the
  detector hierarchy are `dials.import`'s business; a second model of them here
  would be a second thing to keep in step with dxtbx.
* **Where this will not match `dials.find_spots`** is in the README, with the
  reasons. The pixel mask is the one real omission: DIALS masks from the
  detector's own mask where this recognises only the sentinel values in the data.
  Closing it needs no kernel change -- read the mask and stamp the sentinel into
  the frame before the threshold.

## HDF5

* **HDF5 is not thread-safe unless it was built that way**, and vcpkg's default
  port is not. The library keeps global state, so per-thread file handles do not
  help: two threads inside it at once crash. Every HDF5 call in `nxmx.cc`,
  including the closes in `Handle`, is taken under one recursive mutex. Do not
  remove it on the grounds that each thread has its own handle. Nothing is lost:
  a thread-safe build takes a single global lock across its whole public API, so
  reads were already serialised there, and decompression and the threshold --
  where the time goes -- still run in parallel.
* **`H5Dread_chunk` has two shapes.** HDF5 2.0 added a buffer-size argument and
  pointed the old name at the new function through its versioned-API macros, so
  no preprocessor macro distinguishes them and `H5_VERSION_GE` is a guess --
  vcpkg ships 2.0, the distributions mostly do not. A compile probe in
  `CMakeLists.txt` defines `SPOTFINDER_H5DREAD_CHUNK_TAKES_SIZE`, and
  `read_chunk` in `nxmx.cc` has both calls.
* **`-x` unpacks the VDS rather than trusting a layout.** The mapping names its
  own source files, datasets and offsets, and the source datasets state their own
  chunk shape, type size and filters. Do not hardcode `prefix_NNNN.h5`, 1000
  frames per file, or a filter identifier.
* An unallocated chunk is a frame the writer never received: `read` returns false
  and it is counted, not thresholded. Reading it normally would give a frame of
  fill value, which looks like saturated pixels.
* **Reading via `pread` at the chunk addresses was tried in redhorn and reverted,
  and it was never needed.** It was aimed at a bottleneck that did not exist, and
  on real files the addresses from `H5Dget_chunk_info_by_coord` did not land where
  the bytes were. Do not retry it without a measurement that says HDF5 reads are
  the problem.
* The filter header is 12 bytes and counts **bytes**; bitshuffle counts
  **elements**. `decompress.cc` converts, and checks the header's uncompressed
  size against the frame it is supposed to be -- which is the cheapest possible
  check that the pointer, the dimensions and the bit depth all agree. A wrong
  pointer or bit depth usually decompresses to *something*, and a frame of
  plausible garbage is much worse to debug than an exception.
* **A corrupted bslz4 block length is caught by bitshuffle**, which returns a
  negative code rather than reading off the end. `bshuf_decompress_lz4` takes no
  input length, which looks like a memory-safety hole; it was tested in redhorn
  by overwriting a block length with 0x00FFFFFF in a real packet and the result
  was a clean error. LZ4 decoding is bounded by the output buffer, which is what
  saves it.

## Design rules to preserve

* **A `series::Frame` must not be moved.** Its `data` span points into its own
  `storage`, and moving a `std::string` can relocate a small buffer. That is why
  `Reader::read` fills one by pointer, which also lets a worker keep one Frame
  and its capacity for a whole run.
* **The `Series` / `Reader` split is what keeps `hdf5.h` out of
  `find_spots.cc`.** It has one implementation and it is not speculative
  generality: it is information hiding, and `Reader` is per thread because the
  library cannot be entered twice at once.
* **The frame buffer comes from `gpu::host_alloc` when a device is in use.** On
  Metal that is a shared `MTLBuffer`, so the frame is decompressed into memory
  the kernels read and there is no copy at all; on CUDA it is page-locked, so the
  copies are asynchronous. If it fails, fall back to ordinary memory rather than
  failing the run.
* **Bounded queue**, so that a slow stage blocks rather than growing a backlog
  until the machine runs out of memory.
* **Every dispatched key must be accounted for**, whatever became of it: the
  chunk barrier waits on a count, and a worker that returns early without
  balancing it hangs the run. That is what `Chunk::abandon` and the `Leaving`
  guard in `find_spots.cc` are for.

## Failure modes

**Exceptions for anything a caller cannot handle where it stands; a return value
only where a non-zero or false result is an expected outcome rather than a
failure.** Every return code in the tree is of the second kind:

* An unallocated HDF5 chunk means a frame the writer never received.
* `Series::try_open` returning false means not ready yet, which is why it is
  polled.
* `Queue::pop` returning false means closed and drained.
* `dext` and `gpu::find` return -1 for a rejected argument and -2 for a frame
  denser than the packed list can hold. That is the given interface, and
  `find_spots.cc` turns both into exceptions at the boundary.

So a `bool` or a `-1` in this tree is a statement about the domain. If a new one
is an error in disguise, throw instead.

## Conventions

* **A frame's shape is `height` and `width`, in that order, slow then fast**, and
  typed `std::size_t` everywhere a buffer is described. Not `NY`/`NX`, not
  `ny`/`nx`, not `int`. The exception is deliberate: `series::Frame` and
  `series::Info` keep `std::uint64_t`, because those are values read from a
  dataset rather than dimensions of memory, and the conversion happens once with
  an explicit cast. `dext.cc` keeps its signed `int32_t` index locals, which are
  internal and documented there.
* **dxtbx writes `image_size` fast then slow**, which is the opposite way round
  from every frame dimension here. `expt.cc` converts at the read and
  `tests/test_expt.cc` checks it, because getting it the wrong way round on a
  square detector is invisible.
* **Fixed-width and size types are `std::`-qualified**, everywhere. The one
  exception is `nxmx.cc`, where bare `size_t` sits in the signature of an HDF5
  callback and in arguments passed straight to the C API: the types are HDF5's
  there, and writing them its way keeps the call sites checkable against the
  headers.
* `.hh` / `.cc`, exceptions for errors, no `using namespace`. Comments explain
  *why*, not what the line does, and are worth writing where a reader would
  otherwise reasonably suspect a mistake. Prefer purpose-fit code over
  generality nobody has asked for.

## Style

**Run the formatters before handing anything back.** This is not optional and it
is not a matter of taste: unformatted code makes every later diff noisier than it
needs to be.

```sh
clang-format -i src/*.cc src/*.hh tests/*.cc tests/*.hh
black tests/*.py
```

clang-format uses its own defaults, pinned as `BasedOnStyle: LLVM` in
`.clang-format` so that an editor or a different version cannot quietly impose
something else. black is used with its defaults. `clang-format --dry-run
--Werror` and `black --check` both pass on a clean tree.

## Not yet proven

Worth knowing before trusting any of it:

* **The Metal backend has never been compiled**, on any machine. It is a
  transcription of working kernels. The likeliest first failures are the
  attribute list on `dext_stage2_u16` and the `static_assert` on
  `sizeof(SignalPixel)` in MSL. **Run `ctest -R dext_gpu` before believing any
  number this backend produces.**
* **`stage0_tile` and `stage2_direct` in `dext_cuda.cu` have never been
  compiled**, let alone run, and neither has the 32-bit instantiation. What to
  expect from the last: a 28 KB shared tile in `stage2` capping occupancy, the
  dispersion test in double at 1/32 rate on consumer cards, and the
  `0xfffffffe` sentinel exercised for the first time.
* **The output has never been through `dials.index`, and nothing here has seen
  real diffraction.** `tests/regression.sh` takes planted reflections from HDF5
  through to a table and recovers every centroid to better than a pixel, which
  tests the conventions and the arithmetic and is not the same claim.
* **The reflection table format was written from the DIALS source, not from a
  file DIALS wrote.** The two have never met.
* The CUDA numbers in the README come from one workstation card, where `stage0`
  tile against direct is a 2% difference -- close enough to the noise that a
  different card could rank them the other way.
* **16 workers gave only 1.3x one worker on CUDA**, at 106 frame/s aggregate.
  That is not PCIe: 106 frames a second is 3.9 GB/s against a link that delivers
  13. So the kernels saturate the card at one worker already, and `-j` above
  about 2 buys little on the device even though it still helps the read and the
  decompression.
* The threshold has been run over real detector geometry (400 frames at
  4362 x 4148) but on synthetic frames: sparse planted pixels, not diffraction.
