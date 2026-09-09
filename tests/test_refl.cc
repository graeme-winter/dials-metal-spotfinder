// What comes out of refl::write, read back with a msgpack decoder written here
// rather than with the encoder's own helpers.
//
// The point of the duplication is that this test fails if the encoder is wrong,
// where a round trip through shared code would agree with itself whatever it
// did. The decoder below is deliberately dumb: it knows the handful of msgpack
// types a reflection table uses and rejects everything else.
//
// tests/check_refl.py does the same job again from Python, and is the one to
// run against a file dials.find_spots wrote.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "dials_spots.hh"
#include "refl.hh"
#include "signal_pixel.hh"

namespace {

int failures = 0;

void check(const std::string &what, unsigned long long actual,
           unsigned long long expected) {
  if (actual == expected)
    return;
  std::printf("FAIL %s: %llu, expected %llu\n", what.c_str(), actual, expected);
  failures++;
}

void check_text(const std::string &what, const std::string &actual,
                const std::string &expected) {
  if (actual == expected)
    return;
  std::printf("FAIL %s: \"%s\", expected \"%s\"\n", what.c_str(),
              actual.c_str(), expected.c_str());
  failures++;
}

void close(const std::string &what, double actual, double expected) {
  if (std::fabs(actual - expected) < 1e-9)
    return;
  std::printf("FAIL %s: %.12f, expected %.12f\n", what.c_str(), actual,
              expected);
  failures++;
}

// ---------------------------------------------------------------------------
// A msgpack reader that handles what a reflection table contains and nothing
// else.
// ---------------------------------------------------------------------------

class Reader {
public:
  explicit Reader(std::string bytes) : bytes_(std::move(bytes)) {}

  bool ok() const { return ok_; }
  std::size_t at() const { return at_; }
  std::size_t size() const { return bytes_.size(); }

  std::uint8_t head() { return static_cast<std::uint8_t>(byte()); }

  std::size_t array() {
    const std::uint8_t tag = head();
    if ((tag & 0xf0) == 0x90)
      return tag & 0x0f;
    if (tag == 0xdc)
      return be(2);
    if (tag == 0xdd)
      return be(4);
    return bad("not an array");
  }

  std::size_t map() {
    const std::uint8_t tag = head();
    if ((tag & 0xf0) == 0x80)
      return tag & 0x0f;
    if (tag == 0xde)
      return be(2);
    if (tag == 0xdf)
      return be(4);
    return bad("not a map");
  }

  std::string text() {
    const std::uint8_t tag = head();
    std::size_t n = 0;
    if ((tag & 0xe0) == 0xa0)
      n = tag & 0x1f;
    else if (tag == 0xd9)
      n = be(1);
    else if (tag == 0xda)
      n = be(2);
    else if (tag == 0xdb)
      n = be(4);
    else
      return (bad("not a string"), std::string());
    if (at_ + n > bytes_.size())
      return (bad("a string past the end"), std::string());
    const std::string result = bytes_.substr(at_, n);
    at_ += n;
    return result;
  }

  std::uint64_t integer() {
    const std::uint8_t tag = head();
    if (tag < 0x80)
      return tag;
    if (tag == 0xcc)
      return be(1);
    if (tag == 0xcd)
      return be(2);
    if (tag == 0xce)
      return be(4);
    if (tag == 0xcf)
      return be(8);
    return bad("not an unsigned integer");
  }

  // The payload is left where it is and its extent returned: these are the
  // column dumps and they are large.
  std::size_t binary(std::size_t *offset) {
    const std::uint8_t tag = head();
    std::size_t n = 0;
    if (tag == 0xc4)
      n = be(1);
    else if (tag == 0xc5)
      n = be(2);
    else if (tag == 0xc6)
      n = be(4);
    else
      return bad("not a binary");
    if (at_ + n > bytes_.size())
      return bad("a binary past the end");
    *offset = at_;
    at_ += n;
    return n;
  }

  template <typename T> T value(std::size_t offset) const {
    T result;
    std::memcpy(&result, bytes_.data() + offset, sizeof(T));
    return result;
  }

private:
  char byte() {
    if (at_ >= bytes_.size()) {
      bad("the document ends early");
      return 0;
    }
    return bytes_[at_++];
  }

  std::uint64_t be(int width) {
    std::uint64_t result = 0;
    for (int i = 0; i < width; i++)
      result = (result << 8) | static_cast<std::uint8_t>(byte());
    return result;
  }

  std::size_t bad(const char *what) {
    if (ok_) {
      std::printf("FAIL msgpack: %s at byte %zu\n", what, at_);
      failures++;
      ok_ = false;
    }
    return 0;
  }

  std::string bytes_;
  std::size_t at_ = 0;
  bool ok_ = true;
};

std::string slurp(const std::string &path) {
  std::FILE *file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    std::printf("FAIL cannot read back %s\n", path.c_str());
    failures++;
    return std::string();
  }
  std::string text;
  char block[65536];
  for (;;) {
    const std::size_t got = std::fread(block, 1, sizeof(block), file);
    text.append(block, got);
    if (got < sizeof(block))
      break;
  }
  std::fclose(file);
  return text;
}

struct Column {
  std::string type;
  std::size_t rows = 0;
  std::size_t offset = 0;
  std::size_t bytes = 0;
};

// The whole file, as far as this test cares: the row count, the identifiers,
// and where each column's dump starts.
struct Table {
  std::uint64_t rows = 0;
  std::vector<std::pair<std::uint64_t, std::string>> identifiers;
  std::vector<std::pair<std::string, Column>> columns;

  const Column *column(const std::string &name) const {
    for (const auto &entry : columns) {
      if (entry.first == name)
        return &entry.second;
    }
    return nullptr;
  }
};

Table decode(Reader &reader) {
  Table table;
  check("the document is an array of three", reader.array(), 3);
  check_text("the file type", reader.text(), "dials::af::reflection_table");
  check("the format version", reader.integer(), 2);

  const std::size_t header = reader.map();
  check("three things in the header", header, 3);
  bool seen_rows = false;
  for (std::size_t i = 0; i < header && reader.ok(); i++) {
    const std::string name = reader.text();
    if (name == "nrows") {
      table.rows = reader.integer();
      seen_rows = true;
    } else if (name == "identifiers") {
      const std::size_t n = reader.map();
      for (std::size_t j = 0; j < n; j++) {
        const std::uint64_t id = reader.integer();
        table.identifiers.emplace_back(id, reader.text());
      }
    } else if (name == "data") {
      const std::size_t n = reader.map();
      for (std::size_t j = 0; j < n && reader.ok(); j++) {
        const std::string column = reader.text();
        check("a column is a pair", reader.array(), 2);
        Column found;
        found.type = reader.text();
        check("its data is a pair", reader.array(), 2);
        found.rows = reader.integer();
        found.bytes = reader.binary(&found.offset);
        table.columns.emplace_back(column, found);
      }
    } else {
      std::printf("FAIL an unknown header key \"%s\"; DIALS refuses these\n",
                  name.c_str());
      failures++;
      return table;
    }
  }
  check("nrows was there", seen_rows ? 1 : 0, 1);
  check("the whole document was consumed", reader.at(), reader.size());
  return table;
}

// The columns dials.find_spots writes, and the type name each one must carry.
// Getting a type name wrong is not a soft failure: DIALS throws "unexpected
// column type" and the file is unreadable.
void check_columns(const Table &table, bool shoeboxes) {
  const std::pair<const char *, const char *> expected[] = {
      {"bbox", "int6"},
      {"flags", "std::size_t"},
      {"id", "int"},
      {"intensity.sum.value", "double"},
      {"intensity.sum.variance", "double"},
      {"n_signal", "int"},
      {"panel", "std::size_t"},
      {"xyzobs.px.value", "vec3<double>"},
      {"xyzobs.px.variance", "vec3<double>"}};

  for (const auto &want : expected) {
    const Column *column = table.column(want.first);
    if (column == nullptr) {
      std::printf("FAIL no %s column\n", want.first);
      failures++;
      continue;
    }
    check_text(std::string("the type of ") + want.first, column->type,
               want.second);
    check(std::string("the row count of ") + want.first, column->rows,
          table.rows);
  }

  const Column *shoebox = table.column("shoebox");
  check("the shoebox column is there when asked for",
        (shoebox != nullptr) ? 1 : 0, shoeboxes ? 1 : 0);
  if (shoebox != nullptr)
    check_text("the type of shoebox", shoebox->type, "Shoebox<>");
  check("the column count", table.columns.size(), shoeboxes ? 10 : 9);
}

} // namespace

int main() {
  const std::size_t height = 8;
  const std::size_t width = 10;

  // Two spots: three pixels in a row on frame 0, and a two-frame pair. Both
  // are cases whose numbers tests/test_dials_spots.cc has already pinned down,
  // so what is being checked here is only how they are written.
  dials_spots::Options options;
  options.min_spot_size = 1;
  options.max_separation = 0.0;
  dials_spots::Labeller labeller(height, width, options);
  labeller.add(0, {{0, 1, 0.0F, 0, 0}, {1, 2, 0.0F, 0, 0}, {2, 1, 0.0F, 0, 0}});
  labeller.add(1, {{25, 7, 0.0F, 0, 0}});
  labeller.add(2, {{25, 3, 0.0F, 0, 0}});
  labeller.finish();
  check("two spots to write", labeller.spots().size(), 2);

  const std::string path = "test_refl.refl";

  {
    refl::Options writing;
    writing.identifier = "b3a1f0c2";
    refl::write(path, labeller.spots(), labeller.pixels(), width, writing);

    Reader reader(slurp(path));
    const Table table = decode(reader);
    check("two rows", table.rows, 2);
    check_columns(table, true);
    check("one identifier", table.identifiers.size(), 1);
    if (table.identifiers.size() == 1) {
      check("under experiment zero", table.identifiers[0].first, 0);
      check_text("and it is the one we were given", table.identifiers[0].second,
                 "b3a1f0c2");
    }

    // Every row of every fixed column, against the spots themselves.
    const Column *bbox = table.column("bbox");
    const Column *value = table.column("xyzobs.px.value");
    const Column *variance = table.column("xyzobs.px.variance");
    const Column *intensity = table.column("intensity.sum.value");
    const Column *intensity_variance = table.column("intensity.sum.variance");
    const Column *n_signal = table.column("n_signal");
    const Column *flags = table.column("flags");
    const Column *id = table.column("id");
    const Column *panel = table.column("panel");
    if (bbox != nullptr && value != nullptr && variance != nullptr &&
        intensity != nullptr && intensity_variance != nullptr &&
        n_signal != nullptr && flags != nullptr && id != nullptr &&
        panel != nullptr && reader.ok()) {
      check("bbox is six ints a row", bbox->bytes, table.rows * 24);
      check("a centroid is three doubles", value->bytes, table.rows * 24);
      for (std::size_t row = 0; row < labeller.spots().size(); row++) {
        const dials_spots::Spot &spot = labeller.spots()[row];
        for (int i = 0; i < 6; i++) {
          check("bbox element",
                static_cast<std::uint32_t>(reader.value<std::int32_t>(
                    bbox->offset + row * 24 + static_cast<std::size_t>(i) * 4)),
                static_cast<std::uint32_t>(spot.bbox[i]));
        }
        for (int axis = 0; axis < 3; axis++) {
          close("centroid",
                reader.value<double>(value->offset + row * 24 +
                                     static_cast<std::size_t>(axis) * 8),
                spot.position[axis]);
          close("centroid variance",
                reader.value<double>(variance->offset + row * 24 +
                                     static_cast<std::size_t>(axis) * 8),
                spot.variance[axis]);
        }
        close("intensity", reader.value<double>(intensity->offset + row * 8),
              spot.intensity);
        close("intensity variance",
              reader.value<double>(intensity_variance->offset + row * 8),
              spot.intensity_variance);
        check("n_signal",
              static_cast<std::uint32_t>(
                  reader.value<std::int32_t>(n_signal->offset + row * 4)),
              spot.n_signal);
        // dials::af::Flags::Strong, and nothing else set.
        check("flags", reader.value<std::uint64_t>(flags->offset + row * 8),
              32);
        check("id",
              static_cast<std::uint32_t>(
                  reader.value<std::int32_t>(id->offset + row * 4)),
              0);
        check("panel", reader.value<std::uint64_t>(panel->offset + row * 8), 0);
      }

      // The shoebox blob of the second spot: one pixel on each of two frames,
      // so a 1 x 1 x 2 box, values 7 then 3, both valid and foreground, with
      // the background left at zero.
      const Column *shoebox = table.column("shoebox");
      if (shoebox != nullptr) {
        // The first spot is 3 x 1 x 1: 4 + 24 + 1 + 3 * 9 = 56 bytes.
        const std::size_t first = shoebox->offset;
        check("the first shoebox has panel zero",
              reader.value<std::uint32_t>(first), 0);
        check("and version 2 data",
              static_cast<std::uint8_t>(reader.value<char>(first + 28)), 2);
        close("its first pixel", reader.value<float>(first + 29), 1.0);
        close("its second", reader.value<float>(first + 33), 2.0);
        close("its third", reader.value<float>(first + 37), 1.0);
        // Valid | Foreground, from dials/model/data/mask_code.h.
        check("the mask of its first pixel",
              static_cast<std::uint8_t>(reader.value<char>(first + 41)), 5);
        close("the background it was given", reader.value<float>(first + 44),
              0.0);
        check("the whole column is the two blobs", shoebox->bytes,
              56 + (4 + 24 + 1 + 2 * 9));

        const std::size_t second = first + 56;
        check("the second box starts on frame 1",
              static_cast<std::uint32_t>(
                  reader.value<std::int32_t>(second + 4 + 16)),
              1);
        check("and ends after frame 2",
              static_cast<std::uint32_t>(
                  reader.value<std::int32_t>(second + 4 + 20)),
              3);
        close("its frame 1 pixel", reader.value<float>(second + 29), 7.0);
        close("its frame 2 pixel", reader.value<float>(second + 33), 3.0);
      }
    }
  }

  // Without shoeboxes, which is what indexing wants: nine columns, and much
  // smaller.
  {
    refl::Options writing;
    writing.shoeboxes = false;
    refl::write(path, labeller.spots(), labeller.pixels(), width, writing);
    Reader reader(slurp(path));
    const Table table = decode(reader);
    check("still two rows", table.rows, 2);
    check_columns(table, false);
    check("and no identifiers when none was given", table.identifiers.size(),
          0);
  }

  // An empty table. dials.find_spots writes nothing at all in this case and
  // says so; the file is still well formed here, which is what a caller that
  // writes unconditionally needs.
  {
    refl::Options writing;
    refl::write(path, {}, {}, width, writing);
    Reader reader(slurp(path));
    const Table table = decode(reader);
    check("no rows", table.rows, 0);
    check_columns(table, true);
  }

  std::remove(path.c_str());

  std::printf("%s: the reflection table writer, %d failures\n",
              failures == 0 ? "PASS" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
