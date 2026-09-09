#!/usr/bin/env python3
"""Check a reflection table against what tests/make_test_nxmx.py planted.

    check_spots.py <directory>/manifest.json <strong.refl>

This is the only test here that goes all the way from HDF5 on disk to a
reflection table, so it is where a mistake in the reading path shows up: a
mis-unpacked virtual dataset, a chunk read at the wrong offset, a frame number
that is not the array index it claims to be. Every one of those puts the spots
somewhere other than where they were planted, or loses them.

What it checks:

  * one row per planted reflection -- not one per frame the reflection touches,
    which is what grouping per frame would give and is three times too many
  * every centroid within a pixel of where it was planted
  * every z centroid at the middle of its own rocking curve
  * every bounding box spanning exactly the frames the reflection was on
  * nothing else in the table

Reuses the msgpack decoder in check_refl.py, which is beside it.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_refl import Table  # noqa: E402

# A pixel in x and y. The profile is symmetric, so the centroid should land on
# the pixel it was planted in; anything approaching a pixel is a systematic
# error rather than noise. Half a frame in z, where the curve is symmetric about
# its middle frame.
TOLERANCE = 1.0
Z_TOLERANCE = 0.5


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2

    with open(argv[1]) as handle:
        manifest = json.load(handle)
    table = Table(argv[2])

    wrong = table.check()
    for complaint in wrong:
        print(f"BAD  {complaint}")

    expected = manifest["expected"]
    centroids = table.centroids()
    boxes = table.bboxes()
    print(f"{len(expected)} spots planted, {table.rows} in {argv[2]}")

    failures = list(wrong)
    if manifest["missing_frame"] is not None:
        print(f"  frame {manifest['missing_frame']} was never written")

    taken = [False] * len(centroids)
    for spot in expected:
        want_x = spot["fast"] + 0.5
        want_y = spot["slow"] + 0.5
        frames = spot["frames"]
        want_z = (frames[0] + frames[-1] + 1) / 2.0

        best = None
        best_distance = None
        for index, (x, y, z) in enumerate(centroids):
            if taken[index]:
                continue
            distance = ((x - want_x) ** 2 + (y - want_y) ** 2) ** 0.5
            if best_distance is None or distance < best_distance:
                best, best_distance = index, distance

        where = (
            f"the reflection at ({spot['slow']}, {spot['fast']}) on frames "
            f"{frames[0]}-{frames[-1]}"
        )
        if best is None:
            failures.append(f"{where}: nothing left to match it to")
            continue
        if best_distance > TOLERANCE:
            x, y, z = centroids[best]
            failures.append(
                f"{where}: nearest centroid is ({x:.2f}, {y:.2f}), "
                f"{best_distance:.2f} px away"
            )
            continue

        taken[best] = True
        x, y, z = centroids[best]
        if abs(z - want_z) > Z_TOLERANCE:
            failures.append(f"{where}: z is {z:.2f}, expected {want_z:.2f}")
        x0, x1, y0, y1, z0, z1 = boxes[best]
        if (z0, z1) != (frames[0], frames[-1] + 1):
            failures.append(
                f"{where}: bounding box spans frames {z0}-{z1}, "
                f"expected {frames[0]}-{frames[-1] + 1}"
            )
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            failures.append(f"{where}: its centroid is outside its own box")

    extra = [index for index, used in enumerate(taken) if not used]
    if extra:
        where = ", ".join(
            f"({centroids[index][1]:.1f}, {centroids[index][0]:.1f}) "
            f"on frame {centroids[index][2]:.1f}"
            for index in extra[:5]
        )
        failures.append(
            f"{len(extra)} spots that were not planted: {where}"
            + (" ..." if len(extra) > 5 else "")
        )

    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        print(f"FAILED: {len(failures)} problems")
        return 1
    print("PASS: every planted reflection is one spot, where it was planted")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
