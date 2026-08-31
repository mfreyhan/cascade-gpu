#pragma once

#include "core/Types.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace cascade {

// ===========================================================================
// Minimal TOML-subset configuration reader.
//
// Supported:  [section] headers, key = value pairs, # and ; comments,
//             quoted strings, bools, integers, reals, and [a, b, c] arrays.
// Keys are addressed as "section.key"; a top-level key has no prefix.
//
// A hand-written parser rather than a TOML library: the whole feature set a
// case file needs is above, and the solver stays dependency-free (which keeps
// the Vast.ai container setup to `git clone && cmake`).
// ===========================================================================
class Config {
 public:
  Config() = default;

  // Throws std::runtime_error on a missing file or a malformed line
  // (with the line number).
  static Config fromFile(const std::string& path);
  static Config fromString(const std::string& text, const std::string& originName = "<string>");

  bool has(const std::string& key) const;

  // Typed lookup with a default.
  Real getReal(const std::string& key, Real fallback) const;
  int getInt(const std::string& key, int fallback) const;
  bool getBool(const std::string& key, bool fallback) const;
  std::string getString(const std::string& key, const std::string& fallback) const;

  // Typed lookup that throws if the key is absent. Use for values that have no
  // sensible default (inlet stagnation conditions, mesh file, ...): a silent
  // default there produces a plausible-looking wrong answer.
  Real requireReal(const std::string& key) const;
  int requireInt(const std::string& key) const;
  std::string requireString(const std::string& key) const;

  std::vector<Real> getRealArray(const std::string& key) const;

  const std::map<std::string, std::string>& entries() const { return entries_; }

  std::string dump() const;

 private:
  const std::string* find(const std::string& key) const;
  [[noreturn]] void missing(const std::string& key) const;

  std::map<std::string, std::string> entries_;
  std::string origin_;
};

}  // namespace cascade
