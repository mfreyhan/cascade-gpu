#include "core/Config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace cascade {
namespace {

std::string trim(const std::string& s) {
  const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
  auto begin = std::find_if(s.begin(), s.end(), notSpace);
  auto end = std::find_if(s.rbegin(), s.rend(), notSpace).base();
  return begin < end ? std::string(begin, end) : std::string();
}

std::string stripComment(const std::string& line) {
  bool inQuotes = false;
  for (Size i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') inQuotes = !inQuotes;
    if (!inQuotes && (c == '#' || c == ';')) return line.substr(0, i);
  }
  return line;
}

std::string unquote(const std::string& s) {
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') return s.substr(1, s.size() - 2);
  return s;
}

std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

Config Config::fromString(const std::string& text, const std::string& originName) {
  Config cfg;
  cfg.origin_ = originName;

  std::istringstream stream(text);
  std::string line;
  std::string section;
  int lineNo = 0;

  while (std::getline(stream, line)) {
    ++lineNo;
    const std::string content = trim(stripComment(line));
    if (content.empty()) continue;

    if (content.front() == '[') {
      if (content.back() != ']') {
        throw std::runtime_error(originName + ":" + std::to_string(lineNo) +
                                 ": unterminated section header");
      }
      section = trim(content.substr(1, content.size() - 2));
      continue;
    }

    const auto eq = content.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error(originName + ":" + std::to_string(lineNo) +
                               ": expected 'key = value', got '" + content + "'");
    }

    const std::string key = trim(content.substr(0, eq));
    const std::string value = trim(content.substr(eq + 1));
    if (key.empty()) {
      throw std::runtime_error(originName + ":" + std::to_string(lineNo) + ": empty key");
    }

    cfg.entries_[section.empty() ? key : section + "." + key] = value;
  }

  return cfg;
}

Config Config::fromFile(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open config file: " + path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return fromString(buffer.str(), path);
}

const std::string* Config::find(const std::string& key) const {
  const auto it = entries_.find(key);
  return it == entries_.end() ? nullptr : &it->second;
}

void Config::missing(const std::string& key) const {
  throw std::runtime_error("required config key '" + key + "' not found in " +
                           (origin_.empty() ? std::string("<empty config>") : origin_));
}

bool Config::has(const std::string& key) const { return find(key) != nullptr; }

Real Config::getReal(const std::string& key, Real fallback) const {
  const std::string* v = find(key);
  return v ? static_cast<Real>(std::stod(*v)) : fallback;
}

int Config::getInt(const std::string& key, int fallback) const {
  const std::string* v = find(key);
  return v ? std::stoi(*v) : fallback;
}

bool Config::getBool(const std::string& key, bool fallback) const {
  const std::string* v = find(key);
  if (!v) return fallback;
  const std::string s = toLower(unquote(*v));
  if (s == "true" || s == "yes" || s == "on" || s == "1") return true;
  if (s == "false" || s == "no" || s == "off" || s == "0") return false;
  throw std::runtime_error("config key '" + key + "': '" + *v + "' is not a boolean");
}

std::string Config::getString(const std::string& key, const std::string& fallback) const {
  const std::string* v = find(key);
  return v ? unquote(*v) : fallback;
}

Real Config::requireReal(const std::string& key) const {
  const std::string* v = find(key);
  if (!v) missing(key);
  return static_cast<Real>(std::stod(*v));
}

int Config::requireInt(const std::string& key) const {
  const std::string* v = find(key);
  if (!v) missing(key);
  return std::stoi(*v);
}

std::string Config::requireString(const std::string& key) const {
  const std::string* v = find(key);
  if (!v) missing(key);
  return unquote(*v);
}

std::vector<Real> Config::getRealArray(const std::string& key) const {
  const std::string* v = find(key);
  if (!v) return {};
  std::string s = trim(*v);
  if (s.size() >= 2 && s.front() == '[' && s.back() == ']') s = s.substr(1, s.size() - 2);

  std::vector<Real> out;
  std::istringstream stream(s);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const std::string t = trim(token);
    if (!t.empty()) out.push_back(static_cast<Real>(std::stod(t)));
  }
  return out;
}

std::string Config::dump() const {
  std::ostringstream out;
  for (const auto& [key, value] : entries_) out << "  " << key << " = " << value << "\n";
  return out.str();
}

}  // namespace cascade
