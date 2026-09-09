#!/usr/bin/env python3
"""Read a DIALS .refl file, and compare two of them.

    check_refl.py strong.refl                 # what is in it, and is it well formed
    check_refl.py strong.refl dials.refl      # how do the two spot lists differ

The first form is the one to run on the output of dials-metal-find-spots before
handing it to dials.index: it decodes the file the way DIALS' own msgpack
adapter does, checks every column's type name and length, and prints the spot
count and the ranges of the columns that matter. A file that fails here would
fail inside DIALS with a much less specific message.

The second form is the check that matters for the science. Match the two lists
by centroid -- nearest within a couple of pixels and a frame or two -- and
report what matched, what did not, and how far apart the ones that did are.
Neither file needs to have come from here: comparing a run of dials.find_spots
against a run of this on the same images is exactly the intended use, and it is
the only way to see whether the two finders agree about a real sweep.

Do not expect them to agree exactly. Both implement the same extended
dispersion test, but this evaluates it in float where DIALS uses double, and
DIALS masks pixels from the detector's own pixel mask where this recognises only
the sentinel values in the data. Spots near the threshold can therefore
fall either side of it. Wholesale disagreement is a bug; a per-cent tail of
unmatched weak spots is the arithmetic.

No dependencies: the msgpack subset a reflection table uses is decoded here, so
this runs under any python3 rather than needing the DIALS environment.
"""

from __future__ import annotations

import struct
import sys

# The column types DIALS knows, and how wide one element of each is. A type
# name outside this set makes the file unreadable to DIALS, which is why the
# check is against the list rather than against whatever is in the file.
ELEMENT_SIZE = {
    "bool": 1,
    "int": 4,
    "std::size_t": 8,
    "double": 8,
    "vec2<double>": 16,
    "vec3<double>": 24,
    "mat3<double>": 72,
    "int6": 24,
    "cctbx::miller::index<>": 12,
}

# Shoebox and std::string columns are not fixed width, so they are checked
# differently or not at all.
VARIABLE = {"Shoebox<>", "std::string"}


class Unpacker:
    """Just the msgpack a reflection table is made of."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.at = 0

    def byte(self) -> int:
        value = self.data[self.at]
        self.at += 1
        return value

    def take(self, n: int) -> bytes:
        if self.at + n > len(self.data):
            raise ValueError(f"a value of {n} bytes runs off the end")
        value = self.data[self.at : self.at + n]
        self.at += n
        return value

    def be(self, n: int) -> int:
        return int.from_bytes(self.take(n), "big")

    def value(self):
        tag = self.byte()
        if tag < 0x80:
            return tag
        if tag >= 0xE0:
            return tag - 0x100  # negative fixint
        if 0xA0 <= tag <= 0xBF:
            return self.take(tag & 0x1F).decode("utf-8")
        if 0x90 <= tag <= 0x9F:
            return [self.value() for _ in range(tag & 0x0F)]
        if 0x80 <= tag <= 0x8F:
            return {self.value(): self.value() for _ in range(tag & 0x0F)}
        if tag == 0xC0:
            return None
        if tag == 0xC2:
            return False
        if tag == 0xC3:
            return True
        if tag in (0xC4, 0xC5, 0xC6):
            return self.take(self.be({0xC4: 1, 0xC5: 2, 0xC6: 4}[tag]))
        if tag in (0xCA, 0xCB):
            width = 4 if tag == 0xCA else 8
            return struct.unpack("<f" if width == 4 else "<d", self.take(width))[0]
        if tag in (0xCC, 0xCD, 0xCE, 0xCF):
            return self.be({0xCC: 1, 0xCD: 2, 0xCE: 4, 0xCF: 8}[tag])
        if tag in (0xD0, 0xD1, 0xD2, 0xD3):
            width = {0xD0: 1, 0xD1: 2, 0xD2: 4, 0xD3: 8}[tag]
            return int.from_bytes(self.take(width), "big", signed=True)
        if tag in (0xD9, 0xDA, 0xDB):
            return self.take(self.be({0xD9: 1, 0xDA: 2, 0xDB: 4}[tag])).decode("utf-8")
        if tag in (0xDC, 0xDD):
            return [self.value() for _ in range(self.be(2 if tag == 0xDC else 4))]
        if tag in (0xDE, 0xDF):
            n = self.be(2 if tag == 0xDE else 4)
            return {self.value(): self.value() for _ in range(n)}
        raise ValueError(f"msgpack tag {tag:#02x} at byte {self.at - 1}")


class Table:
    def __init__(self, path: str) -> None:
        with open(path, "rb") as handle:
            raw = handle.read()
        if raw[:2] == b"\x1f\x8b":
            import gzip

            raw = gzip.decompress(raw)
        unpacker = Unpacker(raw)
        document = unpacker.value()
        if unpacker.at != len(raw):
            raise ValueError(
                f"{unpacker.at} of {len(raw)} bytes consumed; there is more in "
                "here than a reflection table"
            )
        if not isinstance(document, list) or len(document) != 3:
            raise ValueError("not an array of three: this is not a reflection table")
        if document[0] != "dials::af::reflection_table":
            raise ValueError(f"the file says it is a {document[0]!r}")
        if document[1] not in (1, 2):
            raise ValueError(f"format version {document[1]}, and DIALS reads 1 or 2")

        self.path = path
        self.version = document[1]
        header = document[2]
        unknown = set(header) - {"nrows", "identifiers", "data"}
        if unknown:
            raise ValueError(f"DIALS refuses unknown header keys: {sorted(unknown)}")
        if "nrows" not in header:
            raise ValueError("no nrows, which DIALS requires")
        self.rows = header["nrows"]
        self.identifiers = header.get("identifiers", {})
        self.columns = header.get("data", {})

    def check(self) -> list[str]:
        """Everything wrong with the file, as a list of complaints."""
        wrong = []
        for name, column in sorted(self.columns.items()):
            if not isinstance(column, list) or len(column) != 2:
                wrong.append(f"{name}: not a [type, data] pair")
                continue
            kind, data = column
            if kind not in ELEMENT_SIZE and kind not in VARIABLE:
                wrong.append(f"{name}: type {kind!r} is not one DIALS knows")
                continue
            if not isinstance(data, list) or len(data) != 2:
                wrong.append(f"{name}: not a [size, binary] pair")
                continue
            rows, blob = data
            if rows != self.rows:
                wrong.append(f"{name}: {rows} rows where the table says {self.rows}")
            if not isinstance(blob, bytes):
                wrong.append(f"{name}: the data is not a binary")
                continue
            if kind in ELEMENT_SIZE:
                wanted = rows * ELEMENT_SIZE[kind]
                if len(blob) != wanted:
                    wrong.append(
                        f"{name}: {len(blob)} bytes where {rows} x {kind} is {wanted}"
                    )
        if "shoebox" in self.columns:
            wrong.extend(self.check_shoeboxes())
        return wrong

    def raw(self, name: str) -> bytes:
        return self.columns[name][1][1]

    def doubles(self, name: str, stride: int, offset: int = 0) -> list[float]:
        blob = self.raw(name)
        return [
            struct.unpack_from("<d", blob, row * stride + offset)[0]
            for row in range(self.rows)
        ]

    def ints(self, name: str, stride: int, offset: int = 0, signed=True) -> list[int]:
        blob = self.raw(name)
        code = "<i" if signed else "<Q"
        return [
            struct.unpack_from(code, blob, row * stride + offset)[0]
            for row in range(self.rows)
        ]

    def centroids(self) -> list[tuple[float, float, float]]:
        x = self.doubles("xyzobs.px.value", 24, 0)
        y = self.doubles("xyzobs.px.value", 24, 8)
        z = self.doubles("xyzobs.px.value", 24, 16)
        return list(zip(x, y, z))

    def bboxes(self) -> list[tuple[int, ...]]:
        blob = self.raw("bbox")
        return [struct.unpack_from("<6i", blob, row * 24) for row in range(self.rows)]

    def check_shoeboxes(self) -> list[str]:
        """Walk the shoebox blob, which is the one column with no fixed stride."""
        blob = self.raw("shoebox")
        at = 0
        wrong = []
        for row in range(self.rows):
            if at + 29 > len(blob):
                wrong.append(f"shoebox: row {row} runs off the end of the column")
                return wrong
            panel, x0, x1, y0, y1, z0, z1 = struct.unpack_from("<I6i", blob, at)
            at += 28
            if x1 < x0 or y1 < y0 or z1 < z0:
                wrong.append(f"shoebox {row}: an inverted bounding box")
                return wrong
            version = blob[at]
            at += 1
            if version == 0:
                continue
            if version not in (1, 2):
                wrong.append(f"shoebox {row}: data version {version}")
                return wrong
            n = (x1 - x0) * (y1 - y0) * (z1 - z0)
            mask_width = 1 if version == 2 else 4
            at += n * 4 + n * mask_width + n * 4
            del panel
        if at != len(blob):
            wrong.append(
                f"shoebox: {at} of {len(blob)} bytes walked, so the blob and the "
                "row count disagree"
            )
        return wrong


def describe(table: Table) -> None:
    print(f"{table.path}: {table.rows} reflections, format version {table.version}")
    if table.identifiers:
        for key, value in sorted(table.identifiers.items()):
            print(f"  experiment {key}: {value}")
    else:
        print("  no experiment identifiers")
    print(f"  columns: {', '.join(sorted(table.columns))}")

    wrong = table.check()
    for complaint in wrong:
        print(f"  BAD  {complaint}")
    if not table.rows:
        print("  no rows, so nothing to summarise")
        return None if wrong else True

    if "xyzobs.px.value" in table.columns:
        xyz = table.centroids()
        for axis, name in enumerate("xyz"):
            values = [point[axis] for point in xyz]
            print(
                f"  {name}: {min(values):.2f} to {max(values):.2f}"
                + (
                    f", over {len(set(int(v) for v in values))} frames"
                    if name == "z"
                    else ""
                )
            )
    if "intensity.sum.value" in table.columns:
        intensity = table.doubles("intensity.sum.value", 8)
        intensity.sort()
        print(
            f"  intensity: {intensity[0]:.0f} to {intensity[-1]:.0f}, "
            f"median {intensity[len(intensity) // 2]:.0f}"
        )
    if "n_signal" in table.columns:
        counts = sorted(table.ints("n_signal", 4))
        print(
            f"  pixels a spot: {counts[0]} to {counts[-1]}, "
            f"median {counts[len(counts) // 2]}"
        )
    if "flags" in table.columns:
        flags = table.ints("flags", 8, signed=False)
        strong = sum(1 for value in flags if value & (1 << 5))
        print(f"  flagged strong: {strong} of {len(flags)}")
    if "shoebox" in table.columns:
        print(f"  shoeboxes: {len(table.raw('shoebox')) / 1e6:.1f} MB of pixels")
    return not wrong


def compare(first: Table, second: Table, radius=2.0, frames=2.0) -> bool:
    """Match by centroid and report the difference."""
    a = first.centroids()
    b = second.centroids()

    # Bucket the second list so this does not go quadratic on a real sweep.
    cell = max(radius, 1.0)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, (x, y, z) in enumerate(b):
        key = (int(x // cell), int(y // cell), int(z // frames))
        buckets.setdefault(key, []).append(index)

    taken = [False] * len(b)
    pairs = []
    for index, (x, y, z) in enumerate(a):
        best = None
        best_distance = None
        key = (int(x // cell), int(y // cell), int(z // frames))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in buckets.get(
                        (key[0] + dx, key[1] + dy, key[2] + dz), ()
                    ):
                        if taken[other]:
                            continue
                        ox, oy, oz = b[other]
                        if abs(oz - z) > frames:
                            continue
                        distance = ((ox - x) ** 2 + (oy - y) ** 2) ** 0.5
                        if distance > radius:
                            continue
                        if best_distance is None or distance < best_distance:
                            best, best_distance = other, distance
        if best is not None:
            taken[best] = True
            pairs.append((index, best, best_distance))

    print(
        f"matched {len(pairs)} spots: {len(a) - len(pairs)} only in "
        f"{first.path}, {len(b) - len(pairs)} only in {second.path}"
    )
    if not pairs:
        print(
            "nothing matched at all, which is a difference in kind rather than "
            "in detail: check that both files cover the same images and that z "
            "means the same thing in each"
        )
        return False

    distances = sorted(pair[2] for pair in pairs)
    print(
        f"  centroid: median {distances[len(distances) // 2]:.3f} px, "
        f"90th centile {distances[int(0.9 * len(distances))]:.3f}, "
        f"worst {distances[-1]:.3f}"
    )

    if (
        "intensity.sum.value" in first.columns
        and "intensity.sum.value" in second.columns
    ):
        ia = first.doubles("intensity.sum.value", 8)
        ib = second.doubles("intensity.sum.value", 8)
        ratios = sorted(ib[j] / ia[i] for i, j, _ in pairs if ia[i] > 0 and ib[j] > 0)
        if ratios:
            print(
                f"  intensity ratio: median {ratios[len(ratios) // 2]:.4f}, "
                f"{ratios[0]:.3f} to {ratios[-1]:.3f}"
            )

    # Where the unmatched ones are says what is wrong. Weak spots scattered
    # through the frame are the threshold; a band at one edge or a block of
    # frames is a geometry or an indexing mistake.
    if len(a) - len(pairs):
        unmatched = [a[i] for i in set(range(len(a))) - {pair[0] for pair in pairs}]
        zs = sorted(int(point[2]) for point in unmatched)
        print(
            f"  unmatched in {first.path} run from frame {zs[0]} to {zs[-1]}"
            f" over {len(set(zs))} frames"
        )
        if "n_signal" in first.columns:
            counts = first.ints("n_signal", 4)
            only = [counts[i] for i in set(range(len(a))) - {pair[0] for pair in pairs}]
            only.sort()
            print(
                f"  and hold {only[0]} to {only[-1]} pixels, median "
                f"{only[len(only) // 2]}"
            )
    return len(pairs) == len(a) == len(b)


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 2
    try:
        tables = [Table(path) for path in argv[1:]]
    except (OSError, ValueError) as error:
        print(f"FAILED: {error}")
        return 1

    good = True
    for table in tables:
        if describe(table) is False:
            good = False
        print()
    if len(tables) == 2:
        compare(tables[0], tables[1])
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
