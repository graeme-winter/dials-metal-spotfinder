// The extended dispersion signal calculation on an Apple GPU.
//
// A transcription of the three kernels in dext_cuda.cu, which are themselves a
// transcription of dext.cc. All three compute the same thing over the same
// windows, with the same masked-pixel rule and the same output contract, so a
// series can be run through any of them and the results diffed.
//
// SIXTEEN BIT ONLY, deliberately. dext.hh's traits give real_t<uint32_t> =
// double, and Metal has no double precision at all -- not slow, absent. A
// 32-bit path here would have to run the dispersion test in float, which is a
// different calculation from the one the CPU does, and it would disagree with
// it in a way that looks like a bug rather than a decision. The host side
// refuses 32-bit frames instead. So the traits are written out below as three
// typedefs rather than mirrored as templates: there is one instantiation, and
// a template with one instantiation is a comment that pretends to be code.
//
//   T  the pixel type      dext.hh: std::uint16_t
//   W  wide accumulator    dext.hh: accumulator_t<std::uint16_t> = std::uint32_t
//   R  the real type       dext.hh: real_t<std::uint16_t>        = float
//
// Compiled with -fno-fast-math, and using precise::sqrt below, because the
// point of this file is that it agrees with dext.cc bit for bit. Fast math is
// on by default in the Metal compiler and would quietly cost that.

#include <metal_stdlib>

using namespace metal;

typedef ushort T;
typedef uint W;
typedef float R;

constant T kMasked = T(0xfffe); // numeric_limits<T>::max() - 1, as dext.hh has it

// Must match signal_pixel.hh exactly: 4 + 4 + 4 + 2 + 2, four-byte aligned.
struct SignalPixel {
  uint index;       // row-major position in the frame
  uint value;       // the raw pixel
  float background; // the masked local mean at that pixel
  ushort population; // pixels that mean was averaged over
  ushort reserved;
};

static_assert(sizeof(SignalPixel) == 16, "SignalPixel must pack to 128 bits");

// stage2 is hard-wired to a 32 x 32 threadgroup: the tile it builds is the
// group plus a five-pixel halo on each side, which is the 11 x 11 window's
// reach, and the two are the same number. The host checks that the pipeline
// will actually take 1024 threads rather than assuming it.
constant int kGroup = 32;
constant int kTile = kGroup + 5 + 5;  // stage2's 11 x 11 reach
constant int kTile0 = kGroup + 3 + 3; // stage0's 7 x 7 reach

// ---------------------------------------------------------------------------
// First stage: the dispersion test over a 7 x 7 window, summed directly rather
// than through an integral image. Writes 1 where the window looks like signal,
// and 1 for a masked pixel so that the erode below cannot grow through it.
//
// There are two of these, as there are for stage2, differing only in how the
// window is summed:
//
//   dext_stage0_u16       the 49 pixels summed straight. What CUDA does.
//   dext_stage0_tile_u16  three summed-area tables in threadgroup memory.
//
// Selected at run time with SPOTFINDER_GPU_STAGE0=direct or =tile. stage2's
// measurement is the reason this exists: the tile beat direct summing there by
// 1.76x, stage0 is half the frame's GPU time, and it is doing 49 loads a pixel.
//
// The window sum of p * p is accumulated in W and is deliberately not guarded
// against overflow, as in every other implementation: the data are sparse,
// values above about 1000 are rare, and what matters is that they all wrap the
// same way. They do, and not by luck. The summed-area identity holds in any
// commutative ring, Z / 2^32 included, so differencing a wrapped table gives
// exactly the same value as summing the window directly with the same wrapping.
// That is already why dext.cc, which differences a whole-frame table that has
// certainly wrapped, agrees with the direct sum here.
// ---------------------------------------------------------------------------

kernel void dext_stage0_u16(device const T *image_in [[buffer(0)]],
                            device T *mask_out [[buffer(1)]],
                            constant int &height [[buffer(2)]],
                            constant int &width [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const int knl = 3;
  const R sigma_b = R(6.0);
  const int i = int(gid.y);
  const int j = int(gid.x);
  if (i >= height || j >= width)
    return;
  const int k = i * width + j;

  W m_sum = 0;
  W i_sum = 0;
  W i2_sum = 0;
  for (int di = -knl; di <= knl; di++) {
    for (int dj = -knl; dj <= knl; dj++) {
      const int _i = i + di;
      const int _j = j + dj;
      if (_i >= 0 && _i < height && _j >= 0 && _j < width) {
        const T pixel = image_in[_i * width + _j];
        const W valid = pixel >= kMasked ? W(0) : W(1);
        const W p = valid * W(pixel);
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
    const R m = R(m_sum);
    const R s = R(i_sum);
    const R s2 = R(i2_sum);
    signal = (m * s2 - s * s - s * (m - R(1))) >
             (s * sigma_b * precise::sqrt(R(2) * (m - R(1))));
  }

  mask_out[k] = (signal || image_in[k] >= kMasked) ? T(1) : T(0);
}

// The tile variant. Three tables rather than stage2's two -- the dispersion
// test needs the count, the sum and the sum of squares -- so 38 x 38 x 4 x 3,
// which is 16.9 KB of the 32 KB an Apple threadgroup has. That is more than
// stage2's 14 KB and it may cost a resident threadgroup; if it does, the fix is
// to build the count and sum tables, hold their two window results in
// registers, barrier, and reuse the same memory for the squares. Measure before
// writing that: it costs an extra fill pass and a barrier to save 5.6 KB.
//
// No SIMD group operations here, unlike stage2, so the out-of-range threads may
// simply return -- but only after both barriers, which every thread has to
// reach.
// ---------------------------------------------------------------------------

kernel void dext_stage0_tile_u16(device const T *image_in [[buffer(0)]],
                                 device T *mask_out [[buffer(1)]],
                                 constant int &height [[buffer(2)]],
                                 constant int &width [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]],
                                 uint2 tid [[thread_position_in_threadgroup]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint local [[thread_index_in_threadgroup]]) {
  const int knl = 3;
  const R sigma_b = R(6.0);
  const int i = int(gid.y);
  const int j = int(gid.x);
  const int n = int(local);
  const int N = kTile0;

  threadgroup W M[kTile0][kTile0];
  threadgroup W I[kTile0][kTile0];
  threadgroup W I2[kTile0][kTile0];

  // Rows first: each of the first N threads runs along one row of the halo.
  if (n < N) {
    W tm = 0;
    W ti = 0;
    W ti2 = 0;
    const int _i = n - knl + kGroup * int(group.y);
    for (int m = 0; m < N; m++) {
      const int _j = m - knl + kGroup * int(group.x);
      if (_i >= 0 && _j >= 0 && _i < height && _j < width) {
        const T pixel = image_in[_i * width + _j];
        const W valid = pixel >= kMasked ? W(0) : W(1);
        const W p = valid * W(pixel);
        tm += valid;
        ti += p;
        ti2 += p * p;
      }
      M[n][m] = tm;
      I[n][m] = ti;
      I2[n][m] = ti2;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

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

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Safe here and not before: both barriers are behind us.
  if (i >= height || j >= width)
    return;
  const int k = i * width + j;

  const int j0 = int(tid.x) - 1;
  const int j1 = int(tid.x) + 2 * knl;
  const int i0 = int(tid.y) - 1;
  const int i1 = int(tid.y) + 2 * knl;

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
    const R m = R(m_sum);
    const R s = R(i_sum);
    const R s2 = R(i2_sum);
    signal = (m * s2 - s * s - s * (m - R(1))) >
             (s * sigma_b * precise::sqrt(R(2) * (m - R(1))));
  }

  mask_out[k] = (signal || image_in[k] >= kMasked) ? T(1) : T(0);
}

// ---------------------------------------------------------------------------
// Second stage: erode the dispersion map, so that a pixel survives only if its
// whole 5 x 5 neighbourhood did.
// ---------------------------------------------------------------------------

kernel void dext_stage1_u16(device const T *image_in [[buffer(0)]],
                            device const T *mask_in [[buffer(1)]],
                            device T *mask_out [[buffer(2)]],
                            constant int &height [[buffer(3)]],
                            constant int &width [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const int knl = 2;
  const int i = int(gid.y);
  const int j = int(gid.x);
  if (i >= height || j >= width)
    return;

  const int k = i * width + j;
  if (image_in[k] >= kMasked) {
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

// ---------------------------------------------------------------------------
// Third stage: the Poisson test against a local mean taken over an 11 x 11
// window, and then the emit.
//
// There are two of these, which compute exactly the same thing and differ only
// in how they get m_sum and i_sum for the window:
//
//   dext_stage2_u16         a summed-area table in threadgroup memory, over
//                           the group's tile plus its halo. What CUDA does.
//   dext_stage2_direct_u16  the 121 pixels summed straight, as stage0 does
//                           with its 49.
//
// Both are kept because which is faster is a question about the hardware and
// not about the algorithm, and on Apple silicon the answer is not the same as
// on NVIDIA. Pick between them at run time with SPOTFINDER_GPU_STAGE2=tile or
// =direct; the host reports which it used. Neither may drift from the other:
// tests/test_dext_gpu.cc runs both and requires them identical.
//
// Everything after the window sums is shared, in emit_signal below, so that a
// change to the test or to the emit cannot land in one variant and not the
// other.
// ---------------------------------------------------------------------------

// The Poisson test and the compacted write. Called with the window already
// summed, however that was done.
//
// `in_range` rather than an early return in the caller: every thread in the
// SIMD group has to reach the prefix sum, and a thread that has returned does
// not take part in it -- the result would be silently wrong rather than a clean
// failure. This is the same rule __ballot_sync imposes on the CUDA version.
static void emit_signal(device const T *image_in, device const T *mask_in,
                        device SignalPixel *out, device atomic_uint *counter,
                        uint capacity, int k, bool in_range, W m_sum, W i_sum,
                        uint lane) {
  const R sigma_s = R(3.0);

  bool signal = false;
  T p = T(0);
  R mean = R(0);

  if (in_range) {
    p = image_in[k] >= kMasked ? T(0) : image_in[k];
    if (p > 0 && mask_in[k]) {
      mean = m_sum >= 2 ? R(i_sum) / R(m_sum) : R(0);
      signal = R(p) >= (mean + sigma_s * precise::sqrt(mean));
    }
  }

  // One atomic per SIMD group rather than one per pixel. CUDA does this with
  // __ballot_sync and __popc; the prefix sum says the same thing in one call
  // and does not assume the group is 32 wide, which is worth having even where
  // it is, because the assumption would be silent if it ever stopped holding.
  const uint want = signal ? 1u : 0u;
  const uint rank = simd_prefix_exclusive_sum(want);
  const uint total = simd_sum(want);

  uint base = 0;
  if (lane == 0 && total != 0u)
    base = atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
  base = simd_broadcast_first(base);

  // Past the end is dropped rather than written. The counter still counts it,
  // so the host sees the overflow and refuses the frame instead of analysing a
  // truncated list.
  if (signal && base + rank < capacity) {
    SignalPixel pixel;
    pixel.index = uint(k);
    pixel.value = uint(p);
    pixel.background = float(mean);
    pixel.population = ushort(m_sum);
    pixel.reserved = 0;
    out[base + rank] = pixel;
  }
}

// ---------------------------------------------------------------------------
// The summed-area variant. Two 42 x 42 tables of W, so 14 KB, against the
// 32 KB an Apple threadgroup may have.
//
// Only 42 of the 1024 threads fill the tables, in two serial passes of 42, and
// the other 982 wait at the barrier for both. That is the cost this variant
// pays for its 2-loads-per-pixel window, and it is why the direct one below is
// worth having: 84 serial steps at 4% occupancy against 121 steps at 100%.
//
// The tile is indexed in threadgroup coordinates: row t is global row
// (threadgroup origin + t - 5), so a thread at threadgroup row ty reads the
// window it wants at t = ty - 1 and t = ty + 10, which is global i - 6 and
// i + 5 -- the same i0 and i1 dext.cc computes with knl = 5. Rows and columns
// off the frame contribute nothing while the running sum carries on, which
// gives the same clamped total dext.cc gets from pinning its lookup to ny - 1.
// ---------------------------------------------------------------------------

kernel void dext_stage2_u16(device const T *image_in [[buffer(0)]],
                            device const T *mask_in [[buffer(1)]],
                            device SignalPixel *out [[buffer(2)]],
                            device atomic_uint *counter [[buffer(3)]],
                            constant uint &capacity [[buffer(4)]],
                            constant int &height [[buffer(5)]],
                            constant int &width [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint local [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const int knl = 5;
  const int i = int(gid.y);
  const int j = int(gid.x);
  const int k = i * width + j;
  const int n = int(local);
  const int N = kTile;

  threadgroup W M[kTile][kTile];
  threadgroup W I[kTile][kTile];

  // Rows first: each of the first N threads runs along one row of the halo,
  // accumulating as it goes.
  if (n < N) {
    W tm = 0;
    W ti = 0;
    const int _i = n - knl + kGroup * int(group.y);
    for (int m = 0; m < N; m++) {
      const int _j = m - knl + kGroup * int(group.x);
      if (_i >= 0 && _j >= 0 && _i < height && _j < width) {
        const int _k = _i * width + _j;
        const T pixel = image_in[_k];
        const W valid = (pixel >= kMasked) || mask_in[_k] ? W(0) : W(1);
        tm += valid;
        ti += valid * W(pixel);
      }
      M[n][m] = tm;
      I[n][m] = ti;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

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

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool in_range = i < height && j < width;

  const int j0 = int(tid.x) - 1;
  const int j1 = int(tid.x) + 2 * knl;
  const int i0 = int(tid.y) - 1;
  const int i1 = int(tid.y) + 2 * knl;

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

  emit_signal(image_in, mask_in, out, counter, capacity, k, in_range, m_sum,
              i_sum, lane);
}

// ---------------------------------------------------------------------------
// The direct variant. No threadgroup memory, no barrier, and every thread does
// its own 11 x 11 window, which is 121 loads of the frame and 121 of the mask
// against the table's two apiece.
//
// That sounds worse and may well not be. The table's loads are cheap but its
// fill is done by 42 threads while 982 wait, twice; this does eleven times the
// load traffic but does all of it in parallel, and the traffic lands in cache
// that the neighbouring threads of the group are hitting anyway -- the windows
// of adjacent pixels overlap in 110 of their 121 pixels. Dropping the 14 KB
// also lifts whatever occupancy limit that tile was imposing.
//
// The window is the same one: centred, clipped at the frame edge rather than
// wrapped, and masked and eroded pixels excluded, which is what the table's
// clamped lookups and its `valid` amount to.
// ---------------------------------------------------------------------------

kernel void dext_stage2_direct_u16(device const T *image_in [[buffer(0)]],
                                   device const T *mask_in [[buffer(1)]],
                                   device SignalPixel *out [[buffer(2)]],
                                   device atomic_uint *counter [[buffer(3)]],
                                   constant uint &capacity [[buffer(4)]],
                                   constant int &height [[buffer(5)]],
                                   constant int &width [[buffer(6)]],
                                   uint2 gid [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  const int knl = 5;
  const int i = int(gid.y);
  const int j = int(gid.x);
  const int k = i * width + j;

  // Not a return, for the reason emit_signal documents.
  const bool in_range = i < height && j < width;

  W m_sum = 0;
  W i_sum = 0;

  if (in_range) {
    // Clipped once per axis instead of tested once per pixel: the bounds do not
    // depend on the inner index, and 121 branches a thread is worth removing.
    const int i_from = max(i - knl, 0);
    const int i_to = min(i + knl, height - 1);
    const int j_from = max(j - knl, 0);
    const int j_to = min(j + knl, width - 1);

    for (int _i = i_from; _i <= i_to; _i++) {
      const int row = _i * width;
      for (int _j = j_from; _j <= j_to; _j++) {
        const int _k = row + _j;
        const T pixel = image_in[_k];
        const W valid = (pixel >= kMasked) || mask_in[_k] ? W(0) : W(1);
        m_sum += valid;
        i_sum += valid * W(pixel);
      }
    }
  }

  emit_signal(image_in, mask_in, out, counter, capacity, k, in_range, m_sum,
              i_sum, lane);
}
