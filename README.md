# dials-metal-spotfinder

The DIALS extended dispersion spot finder, on the GPU, from an NXmx HDF5 series
to a reflection table `dials.index` will take.

```sh
dials.import /data/ins10_1_master.h5
dials-metal-find-spots -gpu -j 8 -e imported.expt /data/ins10_1_master.h5
dials.index imported.expt strong.refl
```

The threshold is DIALS'. `src/dext.{hh,cc}` is a transcription of
`DispersionExtendedThreshold` -- kernel (3,3), sigma_b 6, sigma_s 3, threshold 0,
min_count 2, gain 1, and the same treatment of the frame border -- and
`src/dext_metal.{cc,metal}` and `src/dext_cuda.cu` are transcriptions of that,
window for window. What this adds is everything DIALS does afterwards: grouping
six-connected in three dimensions, centroids, and the file format.

It reads frames out of the HDF5 without going through HDF5's filter pipeline,
decompresses them with bitshuffle, and runs the threshold on the CPU or on a
device. Nothing else: there is no experiment model here, no indexing, and no
detector-specific correction.

**Sixteen-bit data on the device.** `real_t<std::uint32_t>` is `double` and
Apple GPUs have no double precision at all, so a 32-bit path under Metal would
be a different calculation from the CPU's. It refuses rather than quietly
running in float. The CPU path takes both.

## Build

HDF5 is the only external dependency; bitshuffle is a submodule, and lz4 comes
with it.

```sh
git submodule update --init
cmake -S . -B build
cmake --build build -j
```

HDF5 must be 1.10.3 or newer, for `H5Dread_chunk`. C++20 is required, for
`std::span` and `std::endian`. On Debian and Ubuntu that is `libhdf5-dev`; on
macOS, `brew install hdf5`; `vcpkg install` works too, and `vcpkg.json` asks for
nothing else.

Boost is not needed, and neither is a msgpack library: the grouping owns its
union-find and the reflection table is written by hand.

```sh
cmake -S . -B build -DSPOTFINDER_METAL=ON -DMETAL_CPP_DIR=/path/to/metal-cpp
cmake -S . -B build -DSPOTFINDER_CUDA=ON
cmake -S . -B build -DSPOTFINDER_AVX2=OFF     # portable SSE2 bitshuffle
cmake -S . -B build -DSPOTFINDER_SANITIZE=address,undefined
```

Configuring both backends at once is a hard error rather than a race at the
linker. Each is off by default and everything it needs is looked for only when
it is asked for, so a machine with neither still configures.

## Running

```
dials-metal-find-spots [-j threads] [-gpu] [-e imported.expt] [-o strong.refl]
                       [options] master.nxs

  master.nxs         an NXmx HDF5 master file, or -x master.nxs
  -e imported.expt   what dials.import wrote, for the scan range, the panel
                     size and the experiment identifier
  -o file            where to write the reflection table (strong.refl)
  -j threads         frames read and thresholded at once (default 4)
  -gpu               run the threshold on the GPU
  --no-shoeboxes     leave out the pixel data, which is most of the file
  --min-spot-size N  contiguous pixels a spot needs (3)
  --max-spot-size N  and the most it may have (1000)
  --max-separation D peak to centroid, in pixels; 0 turns it off (2)
  --2d               group each frame on its own, as for stills
  --z-offset N       array index of image number 0; from -e when given
  --chunk N          frames held between groupings (a few per thread)
  -t timeout         seconds to wait for the file to appear (default 60)
  -p poll-ms         interval between checks while waiting (default 200)
  --version          what this binary is, and what it was built with
```

`--version` says which backend was built in, because that is not visible from
the outside and the CPU and the two devices are not the same program:

```
$ dials-metal-find-spots --version
dials-metal-find-spots 0.1.0 (Metal)
```

`-gpu` fails, rather than falling back, if the build has no backend or no device
is present: it is an explicit request, and quietly answering with the other
implementation would misreport where the results came from.

The tool waits for a master file that does not exist yet, up to `-t`, so it can
be pointed at a series that is still being written -- but every frame of an
open master file is readable as soon as the file is, so it reads the lot and
stops rather than following an acquisition.

### The experiment list

The `.expt` is read and never written. The beam, the goniometer and the detector
are `dials.import`'s business, and a second model of them here would be a second
thing to keep in step with dxtbx. Three things are taken from it:

* **The scan's image range**, because a reflection's `z` is an array index, not
  an image number. The array index of image *n* is *n* - 1, an NXmx series
  numbers its frames from zero, and a sliced import starts higher -- so `-e`
  sets `--z-offset` from `image_range` and says so when it is not zero. Getting
  this wrong indexes nothing and neither file says why.
* **The panel size**, so that a series which is not the one the experiment
  describes is refused here rather than surfacing as a nonsensical lattice
  later.
* **The experiment identifier**, which is the string DIALS uses to tie a table
  to an experiment.

It is optional. Without it, `z` is the frame's own index and the table carries
no identifier, which DIALS accepts.

## Three dimensions, not one frame at a time

A reflection is swept through the Ewald sphere over a degree or two and lands on
several frames. Grouped per frame it comes back as three or four spots with
three or four centroids, and nothing downstream can put them together again. So
the grouping is DIALS': six-connected in x, y and z, a transcription of
`PixelListLabeller::labels_3d`. `--2d` is for stills, and is what
`spotfinder.force_2d=True` means over there.

It is streaming, which DIALS is not. Only frames *f* - 1 and *f* can be
connected, so a component with no pixel on the frame just added can never grow
again: it is written out and its pixels are dropped. DIALS holds every signal
pixel of the whole sweep and labels the lot at the end, which at a few hundred
thousand pixels a frame is gigabytes.

That is also why frames have to reach the grouping in order, and why they are
collected a chunk at a time: dispatch a few frames per thread, wait for them,
sort, group. The barrier costs a fraction of one frame's latency per chunk, and
the alternative -- a reorder buffer that has to know a frame it is waiting for
will never arrive, which is exactly what an unallocated chunk is -- is a great
deal more machinery for that fraction. `--chunk` is the memory knob.

A gap in the frame numbering is not an error. An unallocated chunk is a frame the
writer never received; nothing can be connected across a frame that does not
exist, so everything open closes at the gap and the reflections that would have
spanned it come back as two spots each.

Because frames reach the grouping in order however they were scheduled, the
table does not depend on `-j` or `--chunk`: one thread and eight produce
byte-identical files. `tests/regression.sh` checks that.

## What is in the table

The columns `dials.find_spots` writes, with the same names and the same types:
`bbox` as `int6`, `flags` carrying `Strong` and nothing else, `id`,
`intensity.sum.value` and `.variance`, `n_signal`, `panel`, `shoebox`,
`xyzobs.px.value` and `.variance`.

A `.refl` is a msgpack document around raw column dumps and nothing else -- no
header, no compression, no pickle:

```
[ "dials::af::reflection_table", 2,
  { "identifiers" : { id : identifier, ... },
    "nrows"       : N,
    "data"        : { name : [ type, [ N, <binary> ] ], ... } } ]
```

which is why this needs neither DIALS nor a msgpack library. The reference is
`dials/array_family/reflection_table_msgpack_adapter.h`. Two things about it
are worth knowing: a column type name DIALS does not recognise makes the whole
file unreadable rather than partly readable, and the dumps are memory images, so
the format is portable across neither endianness nor the width of
`std::size_t`. DIALS has the same property and is silent about it; this refuses
to write where the result would not read back.

The centroid, its variance and the intensity are DIALS' own -- `centroid_valid`,
its unbiased standard error plus the 1/12 DIALS adds, and `Summation` over a
shoebox whose background is all zero. All of it reduces to closed forms over the
pixel list, because the spot finder fills only the signal pixels of a shoebox,
so nothing goes back to the images a second time -- the shoeboxes included.

The shoeboxes are nearly the whole file: a dense block of float data, byte mask
and float background per spot, against a few bytes a row for everything else.
They carry nothing the pixel list does not. Worth having for
`dials.image_viewer`; `--no-shoeboxes` for indexing.

One panel, one experiment. A segmented detector would need a panel number per
spot, and several experiments would need the `id` column to mean something; each
is refused rather than guessed at.

## Where this will not match dials.find_spots

Not "it is broken" -- the list is short and each item is attributable:

* `dext` evaluates the dispersion test in float where DIALS uses double, so a
  spot at the threshold can fall either side of it.
* DIALS masks pixels using the detector's own pixel mask, which `dials.import`
  reads from `/entry/instrument/detector/pixel_mask`. This recognises only the
  sentinel values in the data, at or above `max() - 1`. On an Eiger the module
  gaps and the known bad pixels are written as `0xffff`, so that covers most of
  it -- but a bad pixel recorded with a plausible value is masked over there and
  not here. **This is the one real omission**, and closing it needs no change to
  the kernels: read the mask and stamp the sentinel into the frame before the
  threshold.
* DIALS sums a shoebox in float (`ProfileFloatType`), this in double, which
  shows up only above 2^24 counts in one spot.
* Rows come out as their components close rather than in first-pixel order over
  the whole sweep. Nothing depends on row order.

A per-cent tail of unmatched weak spots is that list. Wholesale disagreement is
a bug.

## Checking it

```sh
tests/check_refl.py strong.refl                  # well formed, and what is in it
tests/check_refl.py strong.refl dials.refl       # how two spot lists differ
tests/regression.sh                              # everything, no detector needed
```

`check_refl.py` decodes a table the way DIALS' own adapter reads it and checks
every column's type and length, which is worth doing before `dials.index` says
something less specific. Given two files it matches spots by centroid and
reports what matched, how far apart, and where the unmatched ones are -- bunched
at a detector edge or in a block of frames says something quite different from
scattered and weak. It needs nothing but python3, so it runs outside a DIALS
environment.

`tests/regression.sh` generates a synthetic NXmx series with reflections planted
at known positions, three frames each, and checks the whole path: the table
decodes, every reflection comes back as one spot where it was planted, `--2d`
gives exactly three times as many, a missing frame splits the reflections that
spanned it and no others, one thread and eight agree byte for byte, and a
mismatched experiment list is refused. Where there is a device it also compares
the device against the CPU, byte for byte; where there is not, it says so and
skips.

```sh
python3 tests/make_test_nxmx.py /tmp/series 12
./build/dials-metal-find-spots -e /tmp/series/series.expt /tmp/series/series.nxs
python3 tests/check_spots.py /tmp/series/manifest.json strong.refl
```

is that fixture by hand, and is the quickest way to see whether a change has
moved anything.

## Tests

```sh
ctest --test-dir build --output-on-failure
```

Everything under `ctest` is self-contained -- no detector, no server, no
fixtures to generate first:

| | |
| --- | --- |
| `dext_squares` | where the 32-bit window sums stop being exact, and the margin |
| `signal_order` | the O(n) index sort against `std::sort` |
| `dials_spots` | the grouping, its centroids and its filters, by hand |
| `refl` | the reflection table, decoded again by a second implementation |
| `expt` | the experiment list reader, including the awkward shapes |
| `dials_end_to_end` | the threshold, the grouping and the table together |
| `dext_gpu` | the device against the CPU, with no tolerance at all |

`dext_gpu` is the claim the device backends make. It runs both finders over
synthetic frames chosen for the edges -- sizes that are not multiples of the
32 x 32 threadgroup, spots planted hard against every border, both masked
sentinels -- under both window variants, and compares every field of every
signal pixel. It skips rather than fails on a machine with no device.
`dials_end_to_end` is the one that says the coordinate conventions are right: a
mistake in fast against slow, pixel corners against centres, or frames against
array indices shows up there as a systematic offset and nowhere else as
anything at all.

## The threshold

Each frame goes through the extended dispersion spot finder: a dispersion test
over a 7 x 7 window, an erode, then a Poisson test against a local mean over
11 x 11. It emits the surviving pixels as a packed list rather than a mask --
index, value, local background, window population, 16 bytes each -- which is
smaller on a sparse frame, removes a scan of the mask at both ends, and on the
device removes the download of a whole frame to recover about one per cent of
it.

Pixels at or above `numeric_limits<T>::max() - 1` are treated as masked. Only 16
and 32 bit are supported; 8-bit data is refused rather than promoted.

The window sums are accumulated in 32 bits at 16 bits a pixel, and the tables
wrap over a frame without that mattering: the summed-area identity holds in
Z / 2^32, so differencing a wrapped table gives the true window value provided
that value fits. For the count and the sum it always does. The sum of squares
has a limit -- a uniform 7 x 7 window reaches 2^32 at 9362 counts a pixel -- and
real frames are nowhere near it: the busiest window that can be built out of
reflections, two 2500-count spots two pixels apart, is a factor of 30 short, and
saturated pixels are masked before they can contribute at all.
`ctest -R dext_squares` measures that margin rather than assuming it, so a
change to the kernel size or to the masking shows up as a failure there.

## Reading NXmx

`/entry/data/data` in the master file is a virtual dataset over the data files.
A virtual dataset cannot be chunk-read, so the mapping is unpacked instead:
`H5Pget_virtual_filename`, `H5Pget_virtual_dsetname` and the two selections
give, for each block, which data file holds which frames and where they start in
it. Frames are then read with `H5Dread_chunk` in the worker threads, so the
compressed bytes reach the threshold without HDF5's filter pipeline ever
running. The bitshuffle plugin does not have to be installed, and nothing here
assumes a layout the mapping does not itself state -- not `prefix_NNNN.h5`, not
1000 frames a file, not a filter identifier.

Every HDF5 call is made under one lock, because the C library keeps global state
and is not thread-safe unless it was built that way -- vcpkg's default port is
not, and giving each thread its own file handle does not help. That costs
nothing: a thread-safe HDF5 takes a single global lock across its whole public
API anyway, so those builds were already serialising reads, and what runs in
parallel either way is the decompression and the threshold, which is where the
time goes.

A frame the writer never received is an unallocated chunk, which reads back as
the fill value. Those are reported as "never written" and not thresholded,
rather than analysed as a frame of saturated pixels.

## Building a GPU backend

### Metal

```sh
cmake -S . -B build -DSPOTFINDER_METAL=ON -DMETAL_CPP_DIR=/path/to/metal-cpp
```

metal-cpp is Apple's header-only C++ wrapper over the Metal API, distributed as
a zip from <https://developer.apple.com/metal/cpp/> rather than through any
package manager, so it is pointed at rather than found. Everything else comes
from the Xcode command line tools.

The build compiles `src/dext_metal.metal` with `xcrun -sdk macosx metal
-fno-fast-math` and embeds the resulting `.metallib` into the binary, so the
tool carries its own kernels and cannot get out of step with a shader file
installed beside it. `-fno-fast-math` is not optional: the Metal compiler
enables fast math by default, and the claim this backend makes is that it agrees
with `dext.cc` bit for bit.

The device shares memory with the host, so there is no upload and no download.
`gpu::host_alloc` hands back the contents of a shared `MTLBuffer`, the frame is
decompressed straight into it, and the kernels read it where it lies; the packed
list of signal pixels comes back out of memory the CPU already reads. At 16M
pixels that is 36 MB of traffic per frame that does not happen.

### CUDA

```sh
cmake -S . -B build -DSPOTFINDER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;90"
```

Architectures come from `CMAKE_CUDA_ARCHITECTURES`, since the right answer is
whichever cards this will run on; left unset it uses `native` on CMake 3.24 and
later. There, `host_alloc` returns page-locked memory so the copies to and from
the device are asynchronous rather than running at half rate -- and the copy to
the device is not small: 36 MB at 16M pixels is about 2.7 ms over PCIe 3.0 x16,
a fifth of a frame.

### Two ways to sum a window

`stage0` needs the count, sum and sum of squares over a 7 x 7 window; `stage2`
needs the masked local mean over an 11 x 11 one. Each can sum its window's
pixels directly or build summed-area tables in threadgroup memory, and each
choice is independent:

```sh
SPOTFINDER_GPU_STAGE0=tile   dials-metal-find-spots -gpu series.nxs
SPOTFINDER_GPU_STAGE2=direct dials-metal-find-spots -gpu series.nxs
```

`tile` touches each pixel about twice but fills its tables with a few dozen of
the threadgroup's 1024 threads, in two serial passes, with the rest waiting at a
barrier -- and it holds threadgroup memory. `direct` sums every pixel's window
itself, 49 loads a pixel at `stage0` and 121 at `stage2`, with no barrier and
all 1024 threads working throughout.

On paper the table loses. In fact it wins comfortably at `stage2`: per-pixel load
traffic costs more than the barrier saves. Measured per frame at 4362 x 4148:

| stage0 | stage2 | M4 | workstation CUDA |
| --- | --- | --- | --- |
| `direct` | `tile` | 5.32 ms | **12.10 ms** |
| `tile` | `tile` | **5.13 ms** | 12.37 ms |
| `direct` | `direct` | 9.34 ms | 16.06 ms |
| `tile` | `direct` | -- | 15.01 ms |

`stage2`'s tile wins on both, and by a lot. `stage0`'s helps on Apple silicon
and is slightly worse on the CUDA card, so the shipped defaults -- `direct` for
`stage0`, `tile` for `stage2` -- are right on both as it happens. Measure rather
than trusting the table:

```sh
build/bench_dext_gpu                 # 4362 x 4148, 20 repeats, one worker
build/bench_dext_gpu 4362 4148 20 16 # and aggregate across 16 workers
```

It times all four combinations, reports the CPU for scale, refuses to draw a
conclusion if two variants disagree about what they found, and reports aggregate
throughput as well as single-frame latency -- which is a different question: a
configuration doing more load traffic per pixel loses more as workers pile up,
because they compete for the same memory.

## Putting the signal pixels in order

The device emits its signal pixels compacted per SIMD group, so they arrive in
whatever order the groups finished, and the grouping needs them ascending by
index. `std::sort` was doing that and it was the most expensive thing in
`gpu::find()`: 5.8 ms of a frame at 126,002 signal pixels, against 1.7 for
`signal_order::by_index`, and the gap widens with density -- at 600,000 pixels
it is 33 ms against 9. The flat index is bounded, so one pass over its high bits
puts every pixel in a bucket of consecutive indices; the buckets are then in
order by construction and each holds about four pixels. O(n) rather than
O(n log n), and it uses nothing about the emit order, which is a property of the
threadgroup shape and the scheduler and would break quietly if relied on.

## Status

* **The Metal backend has never been compiled on any machine.** It is a
  transcription of working CUDA kernels. Run `ctest -R dext_gpu` before
  believing any number it produces.
* `dext_cuda.cu` is verified for 16-bit data; its 32-bit instantiation, and its
  `stage0_tile` and `stage2_direct` kernels, have not been run.
* **The output has never been through `dials.index`, and nothing here has seen
  real diffraction.** The whole path is tested against reflections planted at
  known positions in synthetic HDF5, which recovers every centroid to better
  than a pixel; that tests the conventions and the arithmetic and is not the
  same claim. The first real run should be `dials.find_spots` and this over the
  same images, compared with `tests/check_refl.py`.
* **The reflection table format was written from the DIALS source, not from a
  file DIALS wrote.** If DIALS refuses the file, the column type names and
  `Shoebox<>` being `float` rather than `double` are where to look first.
* The pixel mask is not read; see the list of differences above.

## Licence

BSD 3-Clause, in `LICENSE`. bitshuffle is MIT and lz4, which comes with it, is
BSD 2-Clause. HDF5 has a BSD-style licence of its own.
