#include "expt.hh"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace expt {
namespace {

// ---------------------------------------------------------------------------
// Just enough JSON.
//
// Objects keep their members in a vector rather than a map: an experiment list
// has a handful of keys per object, a linear scan of them is faster than a
// tree, and the order a file was written in is preserved for anything that
// wants to report it.
// ---------------------------------------------------------------------------

struct Value {
  enum class Kind { Null, Boolean, Number, String, Array, Object };

  Kind kind = Kind::Null;
  bool boolean = false;
  double number = 0.0;
  std::string text;
  std::vector<Value> items;
  std::vector<std::pair<std::string, Value>> members;

  const Value *find(const char *name) const {
    if (kind != Kind::Object)
      return nullptr;
    for (const auto &member : members) {
      if (member.first == name)
        return &member.second;
    }
    return nullptr;
  }
};

class Parser {
public:
  Parser(const std::string &text, const std::string &path)
      : text_(text), path_(path) {}

  Value document() {
    Value value = parse();
    skip_space();
    if (at_ != text_.size())
      fail("trailing text after the document");
    return value;
  }

private:
  [[noreturn]] void fail(const std::string &what) const {
    throw std::runtime_error(path_ + ": " + what + " at byte " +
                             std::to_string(at_));
  }

  void skip_space() {
    while (at_ < text_.size()) {
      const char c = text_[at_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        at_++;
      else
        break;
    }
  }

  char peek() {
    skip_space();
    if (at_ >= text_.size())
      fail("the document ends early");
    return text_[at_];
  }

  void expect(char c) {
    if (peek() != c)
      fail(std::string("expected '") + c + "'");
    at_++;
  }

  bool literal(const char *word) {
    const std::size_t n = std::strlen(word);
    if (text_.compare(at_, n, word) != 0)
      return false;
    at_ += n;
    return true;
  }

  Value parse() {
    // Depth is bounded so that a pathological or truncated file cannot recurse
    // this into the stack guard. An experiment list nests about five deep.
    if (depth_ > 64)
      fail("nested more than 64 deep");

    const char c = peek();
    if (c == '{')
      return object();
    if (c == '[')
      return array();
    if (c == '"') {
      Value value;
      value.kind = Value::Kind::String;
      value.text = string();
      return value;
    }
    if (c == 't' || c == 'f') {
      Value value;
      value.kind = Value::Kind::Boolean;
      if (literal("true"))
        value.boolean = true;
      else if (literal("false"))
        value.boolean = false;
      else
        fail("not a boolean");
      return value;
    }
    if (c == 'n') {
      if (!literal("null"))
        fail("not null");
      return Value();
    }
    // Python's json writes NaN and Infinity, which are not JSON, and dxtbx
    // files have been seen carrying them. Accepted as numbers rather than
    // rejected: nothing here reads a field that would hold one.
    if (c == 'N') {
      if (!literal("NaN"))
        fail("not a number");
      Value value;
      value.kind = Value::Kind::Number;
      return value;
    }
    if (c == 'I' || (c == '-' && text_.compare(at_, 9, "-Infinity") == 0)) {
      if (!literal("Infinity") && !literal("-Infinity"))
        fail("not a number");
      Value value;
      value.kind = Value::Kind::Number;
      return value;
    }
    return numeric();
  }

  Value object() {
    Value value;
    value.kind = Value::Kind::Object;
    expect('{');
    if (peek() == '}') {
      at_++;
      return value;
    }
    depth_++;
    for (;;) {
      std::string name = string();
      expect(':');
      value.members.emplace_back(std::move(name), parse());
      const char c = peek();
      at_++;
      if (c == '}')
        break;
      if (c != ',')
        fail("expected ',' or '}'");
    }
    depth_--;
    return value;
  }

  Value array() {
    Value value;
    value.kind = Value::Kind::Array;
    expect('[');
    if (peek() == ']') {
      at_++;
      return value;
    }
    depth_++;
    for (;;) {
      value.items.push_back(parse());
      const char c = peek();
      at_++;
      if (c == ']')
        break;
      if (c != ',')
        fail("expected ',' or ']'");
    }
    depth_--;
    return value;
  }

  std::string string() {
    if (peek() != '"')
      fail("expected a string");
    at_++;
    std::string result;
    while (at_ < text_.size()) {
      const char c = text_[at_++];
      if (c == '"')
        return result;
      if (c != '\\') {
        result.push_back(c);
        continue;
      }
      if (at_ >= text_.size())
        fail("an escape at the end of the document");
      const char escape = text_[at_++];
      switch (escape) {
      case '"':
        result.push_back('"');
        break;
      case '\\':
        result.push_back('\\');
        break;
      case '/':
        result.push_back('/');
        break;
      case 'b':
        result.push_back('\b');
        break;
      case 'f':
        result.push_back('\f');
        break;
      case 'n':
        result.push_back('\n');
        break;
      case 'r':
        result.push_back('\r');
        break;
      case 't':
        result.push_back('\t');
        break;
      case 'u': {
        // Enough of \u to get through a filename with an accent in it. A
        // surrogate pair is left as the two code points it was written as,
        // since nothing here interprets the strings it reads.
        if (at_ + 4 > text_.size())
          fail("a truncated \\u escape");
        unsigned code = 0;
        for (int i = 0; i < 4; i++) {
          const char digit = text_[at_++];
          code <<= 4;
          if (digit >= '0' && digit <= '9')
            code |= static_cast<unsigned>(digit - '0');
          else if (digit >= 'a' && digit <= 'f')
            code |= static_cast<unsigned>(digit - 'a' + 10);
          else if (digit >= 'A' && digit <= 'F')
            code |= static_cast<unsigned>(digit - 'A' + 10);
          else
            fail("a \\u escape that is not hexadecimal");
        }
        if (code < 0x80) {
          result.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
          result.push_back(static_cast<char>(0xc0 | (code >> 6)));
          result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else {
          result.push_back(static_cast<char>(0xe0 | (code >> 12)));
          result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
          result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        }
        break;
      }
      default:
        fail("an unknown escape");
      }
    }
    fail("a string with no end");
  }

  Value numeric() {
    const std::size_t began = at_;
    if (at_ < text_.size() && (text_[at_] == '-' || text_[at_] == '+'))
      at_++;
    while (at_ < text_.size()) {
      const char c = text_[at_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
          c == '+' || c == '-')
        at_++;
      else
        break;
    }
    if (at_ == began)
      fail("expected a value");
    Value value;
    value.kind = Value::Kind::Number;
    value.number =
        std::strtod(text_.substr(began, at_ - began).c_str(), nullptr);
    return value;
  }

  const std::string &text_;
  const std::string &path_;
  std::size_t at_ = 0;
  int depth_ = 0;
};

std::string slurp(const std::string &path) {
  std::FILE *file = std::fopen(path.c_str(), "rb");
  if (file == nullptr)
    throw std::runtime_error("cannot read " + path);
  std::string text;
  char block[65536];
  for (;;) {
    const std::size_t got = std::fread(block, 1, sizeof(block), file);
    text.append(block, got);
    if (got < sizeof(block))
      break;
  }
  const bool bad = std::ferror(file) != 0;
  std::fclose(file);
  if (bad)
    throw std::runtime_error("cannot read " + path);
  return text;
}

// An index into one of the model lists, as an experiment records it. dxtbx
// writes these as integers, and null for a model the experiment does not have.
bool model_index(const Value &experiment, const char *name, std::size_t *out) {
  const Value *field = experiment.find(name);
  if (field == nullptr || field->kind != Value::Kind::Number)
    return false;
  if (field->number < 0)
    return false;
  *out = static_cast<std::size_t>(field->number);
  return true;
}

const Value *element(const Value &document, const char *list,
                     std::size_t index) {
  const Value *models = document.find(list);
  if (models == nullptr || models->kind != Value::Kind::Array)
    return nullptr;
  if (index >= models->items.size())
    return nullptr;
  return &models->items[index];
}

} // namespace

Info read(const std::string &path) {
  const std::string text = slurp(path);
  Parser parser(text, path);
  const Value document = parser.document();

  const Value *experiments = document.find("experiment");
  if (experiments == nullptr || experiments->kind != Value::Kind::Array) {
    throw std::runtime_error(path +
                             ": no experiment list in here; is this an .expt "
                             "written by dials.import?");
  }

  Info info;
  info.experiments = experiments->items.size();
  if (info.experiments == 0)
    return info;

  const Value &first = experiments->items[0];
  const Value *identifier = first.find("identifier");
  if (identifier != nullptr && identifier->kind == Value::Kind::String)
    info.identifier = identifier->text;

  std::size_t index = 0;
  if (model_index(first, "scan", &index)) {
    const Value *scan = element(document, "scan", index);
    const Value *range = scan != nullptr ? scan->find("image_range") : nullptr;
    if (range != nullptr && range->kind == Value::Kind::Array &&
        range->items.size() == 2) {
      info.has_scan = true;
      info.first_image = static_cast<std::int64_t>(range->items[0].number);
      info.last_image = static_cast<std::int64_t>(range->items[1].number);
    }
  }

  if (model_index(first, "detector", &index)) {
    const Value *detector = element(document, "detector", index);
    const Value *panels =
        detector != nullptr ? detector->find("panels") : nullptr;
    if (panels != nullptr && panels->kind == Value::Kind::Array &&
        !panels->items.empty()) {
      info.has_detector = true;
      info.panels = panels->items.size();
      const Value *size = panels->items[0].find("image_size");
      if (size != nullptr && size->kind == Value::Kind::Array &&
          size->items.size() == 2) {
        // dxtbx writes image_size fast then slow, which is the opposite way
        // round from every frame dimension here.
        info.image_fast = static_cast<std::size_t>(size->items[0].number);
        info.image_slow = static_cast<std::size_t>(size->items[1].number);
      }
    }
  }

  return info;
}

std::string describe(const Info &info) {
  std::string result = std::to_string(info.experiments) + " experiment";
  if (info.experiments != 1)
    result += "s";
  if (info.has_scan) {
    result += ", images " + std::to_string(info.first_image) + " to " +
              std::to_string(info.last_image);
  } else {
    result += ", no scan";
  }
  if (info.has_detector) {
    result += ", " + std::to_string(info.panels) + " panel";
    if (info.panels != 1)
      result += "s";
    if (info.image_slow > 0) {
      result += " of " + std::to_string(info.image_slow) + " x " +
                std::to_string(info.image_fast);
    }
  }
  if (!info.identifier.empty())
    result += ", identifier " + info.identifier;
  return result;
}

} // namespace expt
