#!/usr/bin/env python3
"""Write an NXmx series with reflections planted at known positions.

    make_test_nxmx.py <directory> [frames] [--holey]

Produces, in <directory>:

    series.nxs        the master file, whose /entry/data/data is a virtual
                      dataset over the two data files
    series_0000.h5    bitshuffle-compressed uint16 frames, one chunk each
    series_0001.h5
    series.expt       a minimal experiment list matching the series
    manifest.json     what was planted, for tests/check_spots.py

Each reflection spans three consecutive frames with a symmetric rocking curve,
so the answer to "how many spots are in this series" is known and is not the
number of frames it touches. That is the property worth testing: grouped per
frame these would come back as three times too many spots, each with the wrong
centroid.

With --holey one frame is never written, which leaves an unallocated chunk --
the way a frame the writer never received appears. Nothing can be connected
across it, so the reflections that would have spanned it break in two, and the
manifest says so.

The .expt is hand-written rather than produced by dials.import: it carries the
image range, the panel size and an identifier, which is all the finder reads,
and writing it here means the -e path can be exercised without DIALS installed.
It is not a substitute for a real one -- there is no beam, no goniometer and no
detector position in it, so it will not index anything.

Needs h5py, hdf5plugin and numpy.
"""

from __future__ import annotations

import json
import os
import sys

import h5py
import hdf5plugin
import numpy as np

HEIGHT = 256
WIDTH = 384
BACKGROUND = 2.0
SIGMA = 1.5
SPAN = 3  # frames a reflection is swept across
CURVE = (0.35, 1.0, 0.35)


def planted_spots(frames: int) -> list[dict]:
    """A grid of reflections, each starting on a frame that lets it complete.

    Spaced by more than twice the profile width so that grouping cannot join
    two, and kept off the very edge so that every one has its whole profile on
    the detector -- the border cases belong in the unit tests, where the
    expected answer can be written down.
    """
    spots = []
    peaks = (250.0, 400.0, 700.0, 1000.0, 1500.0, 2000.0, 2500.0)
    index = 0
    for slow in range(20, HEIGHT - 20, 40):
        for fast in range(20, WIDTH - 20, 48):
            first = 1 + (index % max(1, frames - SPAN - 1))
            spots.append(
                {
                    "slow": slow,
                    "fast": fast,
                    "peak": peaks[index % len(peaks)],
                    "first_frame": first,
                    "frames": SPAN,
                }
            )
            index += 1
    return spots


def make_frames(frames: int, spots: list[dict]) -> np.ndarray:
    stack = np.full((frames, HEIGHT, WIDTH), BACKGROUND, dtype=np.float64)
    rows = np.arange(HEIGHT)[:, None]
    columns = np.arange(WIDTH)[None, :]
    for spot in spots:
        profile = np.exp(
            -((rows - spot["slow"]) ** 2 + (columns - spot["fast"]) ** 2)
            / (2.0 * SIGMA * SIGMA)
        )
        # Truncated at three sigma, as a real profile effectively is, so that
        # the planted spot has a definite extent rather than a floor of
        # fractional counts everywhere.
        profile[profile < np.exp(-4.5)] = 0.0
        for step in range(spot["frames"]):
            frame = spot["first_frame"] + step
            if frame >= frames:
                continue
            stack[frame] += spot["peak"] * CURVE[step] * profile
    return np.clip(np.rint(stack), 0, 65533).astype(np.uint16)


def write_data_file(path: str, stack: np.ndarray, skip: int | None) -> None:
    """One chunk per frame, bitshuffle compressed, written frame by frame.

    Frame by frame rather than in one write so that a skipped frame leaves its
    chunk unallocated rather than filled: an unallocated chunk is what the
    reader has to recognise, and writing zeros would not test it.
    """
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "data",
            shape=stack.shape,
            dtype=np.uint16,
            chunks=(1, stack.shape[1], stack.shape[2]),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        for index in range(stack.shape[0]):
            if index == skip:
                continue
            dataset[index] = stack[index]


def write_master(path: str, files: list[str], frames_each: list[int]) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        data = entry.create_group("data")
        data.attrs["NX_class"] = np.bytes_("NXdata")

        total = sum(frames_each)
        layout = h5py.VirtualLayout(shape=(total, HEIGHT, WIDTH), dtype=np.uint16)
        at = 0
        for name, count in zip(files, frames_each):
            source = h5py.VirtualSource(
                name, "data", shape=(count, HEIGHT, WIDTH), dtype=np.uint16
            )
            layout[at : at + count] = source
            at += count
        data.create_virtual_dataset("data", layout)


def write_expt(path: str, frames: int) -> str:
    identifier = "1c0ffee0-0000-4000-8000-000000000001"
    document = {
        "__id__": "ExperimentList",
        "experiment": [
            {
                "__id__": "Experiment",
                "identifier": identifier,
                "detector": 0,
                "scan": 0,
            }
        ],
        "detector": [
            {
                "panels": [
                    {
                        "name": "Panel",
                        "type": "SENSOR_PAD",
                        # dxtbx writes image_size fast then slow.
                        "image_size": [WIDTH, HEIGHT],
                        "pixel_size": [0.075, 0.075],
                        "trusted_range": [0.0, 65535.0],
                    }
                ]
            }
        ],
        "scan": [{"__id__": "Scan", "image_range": [1, frames]}],
    }
    with open(path, "w") as handle:
        json.dump(document, handle, indent=2)
    return identifier


def main(argv: list[str]) -> int:
    args = [argument for argument in argv[1:] if not argument.startswith("--")]
    holey = "--holey" in argv
    if not args:
        print(__doc__)
        return 2
    directory = args[0]
    frames = int(args[1]) if len(args) > 1 else 12
    if frames < SPAN + 2:
        print(f"at least {SPAN + 2} frames, so that a reflection can complete")
        return 2

    os.makedirs(directory, exist_ok=True)
    spots = planted_spots(frames)
    stack = make_frames(frames, spots)

    # Two data files, so that the virtual dataset has more than one mapping and
    # the unpacking is exercised rather than assumed.
    split = frames // 2
    counts = [split, frames - split]
    names = ["series_0000.h5", "series_0001.h5"]

    # The missing frame, if asked for: the middle of the second file, so that
    # the block offset arithmetic is in play as well.
    missing = split + counts[1] // 2 if holey else None

    at = 0
    for name, count in zip(names, counts):
        skip = None
        if missing is not None and at <= missing < at + count:
            skip = missing - at
        write_data_file(os.path.join(directory, name), stack[at : at + count], skip)
        at += count

    write_master(os.path.join(directory, "series.nxs"), names, counts)
    identifier = write_expt(os.path.join(directory, "series.expt"), frames)

    # What the finder should report. A reflection that spans the missing frame
    # is split in two, and either half may be too short to survive
    # min_spot_size, so the manifest records the frames each half covers and
    # lets the checker work out what to expect.
    expected = []
    for spot in spots:
        covered = [
            spot["first_frame"] + step
            for step in range(spot["frames"])
            if spot["first_frame"] + step != missing
        ]
        runs = []
        for frame in covered:
            if runs and frame == runs[-1][-1] + 1:
                runs[-1].append(frame)
            else:
                runs.append([frame])
        for run in runs:
            expected.append(
                {
                    "slow": spot["slow"],
                    "fast": spot["fast"],
                    "peak": spot["peak"],
                    "frames": run,
                }
            )

    manifest = {
        "height": HEIGHT,
        "width": WIDTH,
        "frames": frames,
        "missing_frame": missing,
        "identifier": identifier,
        "planted": spots,
        "expected": expected,
    }
    with open(os.path.join(directory, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(
        f"{directory}: {frames} frames of {HEIGHT} x {WIDTH}, "
        f"{len(spots)} reflections over {SPAN} frames each, "
        f"{len(expected)} spots expected"
        + (f", frame {missing} never written" if missing is not None else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
