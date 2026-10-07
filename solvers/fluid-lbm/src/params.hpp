// params.hpp -- reader for the shared case file (case/params.txt).
// Format: one "key value" pair per line, '#' starts a comment. The Fortran solver reads
// the same file, so both participants agree on geometry and material data.
#pragma once

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lbm {

class Params {
public:
  explicit Params(const std::string &filename)
  {
    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("cannot open parameter file " + filename);
    }
    std::string line;
    while (std::getline(in, line)) {
      const auto hash = line.find('#');
      if (hash != std::string::npos) {
        line.erase(hash);
      }
      std::istringstream ls(line);
      std::string key, value;
      if (ls >> key >> value) {
        values_[key] = value;
      }
    }
  }

  double real(const std::string &key) const { return std::stod(raw(key)); }
  int    integer(const std::string &key) const { return std::stoi(raw(key)); }
  bool   has(const std::string &key) const { return values_.count(key) > 0; }
  double real_or(const std::string &key, double fallback) const { return has(key) ? real(key) : fallback; }
  int    integer_or(const std::string &key, int fallback) const { return has(key) ? integer(key) : fallback; }

private:
  const std::string &raw(const std::string &key) const
  {
    auto it = values_.find(key);
    if (it == values_.end()) {
      throw std::runtime_error("missing parameter '" + key + "'");
    }
    return it->second;
  }
  std::map<std::string, std::string> values_;
};

} // namespace lbm
