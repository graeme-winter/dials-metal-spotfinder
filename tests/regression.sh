#!/bin/bash
#
# Everything that can be checked without a detector, in one command.
#
#   tests/regression.sh [build-dir]
#
# A GPU is not needed: the CPU/device comparison is a ctest that skips itself
# when the build has a backend but the machine has no device, and the one check
# here that needs a device says so and is skipped. So this runs the same
# everywhere and does more where there is something to do it with.
#
# The unit tests prove pieces in isolation. What this adds is the whole path,
# from HDF5 on disk to a reflection table, over frames with reflections planted
# at known positions -- which is where a mistake in the reading path shows up
# and nowhere else does. There are no recorded baselines: what is checked is
# what has to hold whatever the numbers are, and the planted positions are the
# expectation.
#
# Needs h5py, hdf5plugin and numpy for the fixture generator.

set -u

BUILD="${1:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "${HERE}")"
BUILD="${BUILD:-${ROOT}/build}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

FIND="${BUILD}/dials-metal-find-spots"
if [ ! -x "${FIND}" ]; then
    echo "no ${FIND}: build first, or pass the build directory" >&2
    exit 2
fi

PASS=0
FAIL=0
SKIP=0

pass() {
    echo "  ok    $1"
    PASS=$((PASS + 1))
}

fail() {
    echo "  FAIL  $1"
    FAIL=$((FAIL + 1))
}

skip() {
    echo "  skip  $1"
    SKIP=$((SKIP + 1))
}

echo "== unit tests"
if (cd "${BUILD}" && ctest --output-on-failure > "${WORK}/ctest.log" 2>&1); then
    pass "ctest"
else
    fail "ctest"
    tail -20 "${WORK}/ctest.log"
fi

echo "== fixtures"
if ! python3 "${HERE}/make_test_nxmx.py" "${WORK}/series" 12 \
        > "${WORK}/plant.log" 2>&1; then
    echo "the fixture generator failed; are h5py, hdf5plugin and numpy installed?" >&2
    cat "${WORK}/plant.log" >&2
    exit 2
fi
python3 "${HERE}/make_test_nxmx.py" "${WORK}/holey" 12 --holey > /dev/null || exit 2
cat "${WORK}/plant.log"

MASTER="${WORK}/series/series.nxs"
MANIFEST="${WORK}/series/manifest.json"
EXPT="${WORK}/series/series.expt"

echo "== from HDF5 to a reflection table"
# The whole path, and the only check here that exercises the reading of a real
# virtual dataset: the spots have to come back where they were planted, one per
# reflection rather than one per frame it touches.
"${FIND}" -j 4 -e "${EXPT}" -o "${WORK}/strong.refl" "${MASTER}" \
    > /dev/null 2> "${WORK}/find.err"
if [ "$?" != "0" ]; then
    fail "the spot finder exited non-zero"
    tail -3 "${WORK}/find.err"
elif python3 "${HERE}/check_refl.py" "${WORK}/strong.refl" \
        > "${WORK}/refl.log" 2>&1; then
    pass "the reflection table is well formed"
else
    fail "the reflection table is malformed"
    cat "${WORK}/refl.log"
fi

if python3 "${HERE}/check_spots.py" "${MANIFEST}" "${WORK}/strong.refl" \
        > "${WORK}/spots.log" 2>&1; then
    pass "every planted reflection is one spot, where it was planted"
else
    fail "the spots are not where they were planted"
    cat "${WORK}/spots.log"
fi

# The identifier from the .expt has to reach the table: that string is what ties
# the two together for dials.index.
if python3 "${HERE}/check_refl.py" "${WORK}/strong.refl" 2>&1 |
        grep -q "$(python3 -c "
import json,sys
print(json.load(open('${MANIFEST}'))['identifier'])")"; then
    pass "the experiment identifier is carried into the table"
else
    fail "the experiment identifier did not reach the table"
fi

echo "== the grouping is three dimensional"
# Each reflection spans three frames, so grouping per frame must give exactly
# three times as many spots. This is the check that would catch the 3D grouping
# quietly degrading to per-frame -- which would still produce a plausible file.
"${FIND}" -j 2 --2d -o "${WORK}/2d.refl" "${MASTER}" \
    > /dev/null 2> "${WORK}/2d.err"
three_d=$(python3 -c "
import sys
sys.path.insert(0, '${HERE}')
from check_refl import Table
print(Table('${WORK}/strong.refl').rows)")
two_d=$(python3 -c "
import sys
sys.path.insert(0, '${HERE}')
from check_refl import Table
print(Table('${WORK}/2d.refl').rows)")
if [ "${two_d}" = "$((three_d * 3))" ]; then
    pass "--2d gives ${two_d} spots against ${three_d}, which is three frames each"
else
    fail "--2d gives ${two_d} spots and 3D gives ${three_d}, expected 3x"
fi

echo "== a frame that was never written"
# An unallocated chunk is a frame the writer never received. Nothing can be
# connected across it, so the reflections that spanned it break in two, and the
# manifest works out how many that leaves.
"${FIND}" -j 3 -t 5 -o "${WORK}/holey.refl" "${WORK}/holey/series.nxs" \
    > /dev/null 2> "${WORK}/holey.err"
if [ "$?" != "0" ]; then
    fail "the holey series exited non-zero"
    tail -3 "${WORK}/holey.err"
elif ! grep -q "1 never written" "${WORK}/holey.err"; then
    fail "the missing frame was not reported as never written"
    tail -3 "${WORK}/holey.err"
elif python3 "${HERE}/check_spots.py" "${WORK}/holey/manifest.json" \
        "${WORK}/holey.refl" > "${WORK}/holey.log" 2>&1; then
    pass "a missing frame splits the reflections that spanned it, and no others"
else
    fail "the holey series does not match its manifest"
    cat "${WORK}/holey.log"
fi

echo "== the result does not depend on how it was scheduled"
# Frames reach the grouping in order whatever -j and --chunk are, so the table
# has to be byte-identical. If this ever fails, the ordering has broken and the
# centroids of anything spanning a chunk boundary are wrong.
"${FIND}" -j 1 -e "${EXPT}" -o "${WORK}/j1.refl" "${MASTER}" > /dev/null 2>&1
"${FIND}" -j 8 --chunk 3 -e "${EXPT}" -o "${WORK}/j8.refl" "${MASTER}" \
    > /dev/null 2>&1
if cmp -s "${WORK}/j1.refl" "${WORK}/j8.refl" &&
        cmp -s "${WORK}/j1.refl" "${WORK}/strong.refl"; then
    pass "one thread and eight give the same table, byte for byte"
else
    fail "the table depends on the thread or chunk count"
fi

echo "== the shoeboxes are optional"
"${FIND}" -j 2 --no-shoeboxes -e "${EXPT}" -o "${WORK}/thin.refl" "${MASTER}" \
    > /dev/null 2> "${WORK}/thin.err"
thin=$(wc -c < "${WORK}/thin.refl")
fat=$(wc -c < "${WORK}/strong.refl")
if [ "${thin}" -lt "${fat}" ] &&
        python3 "${HERE}/check_spots.py" "${MANIFEST}" "${WORK}/thin.refl" \
            > "${WORK}/thin.log" 2>&1; then
    pass "--no-shoeboxes is ${thin} bytes against ${fat}, with the same spots"
else
    fail "--no-shoeboxes wrote ${thin} bytes against ${fat}"
    cat "${WORK}/thin.log"
fi

echo "== the experiment list is checked, not trusted"
cat > "${WORK}/wrong.expt" <<'JSON'
{
  "__id__": "ExperimentList",
  "experiment": [{"identifier": "wrong", "detector": 0, "scan": 0}],
  "detector": [{"panels": [{"image_size": [7, 9], "type": "SENSOR_PAD"}]}],
  "scan": [{"image_range": [1, 12]}]
}
JSON
if "${FIND}" -t 5 -e "${WORK}/wrong.expt" -o "${WORK}/never.refl" "${MASTER}" \
        > /dev/null 2> "${WORK}/wrong.err"; then
    fail "a mismatched experiment list was accepted"
elif grep -q "not the same images" "${WORK}/wrong.err"; then
    pass "an experiment list of the wrong size is refused"
else
    fail "the mismatched experiment list failed for the wrong reason"
    tail -3 "${WORK}/wrong.err"
fi

echo "== the device, if there is one"
# Same series, same threshold, on the GPU: the two implementations agree bit for
# bit by design, so the tables have to be identical. -gpu fails rather than
# falling back when there is no device, which is what tells this to skip.
if "${FIND}" -gpu -j 2 -e "${EXPT}" -o "${WORK}/gpu.refl" "${MASTER}" \
        > /dev/null 2> "${WORK}/gpu.err"; then
    if cmp -s "${WORK}/gpu.refl" "${WORK}/strong.refl"; then
        pass "the device and the CPU give the same table, byte for byte"
    else
        fail "the device and the CPU disagree"
        python3 "${HERE}/check_refl.py" "${WORK}/strong.refl" \
            "${WORK}/gpu.refl" 2>&1 | tail -8
    fi
else
    skip "no GPU in this build or on this machine: $(tail -1 "${WORK}/gpu.err")"
fi

echo
echo "${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
[ "${FAIL}" = "0" ]
