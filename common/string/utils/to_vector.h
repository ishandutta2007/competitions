#pragma once

#include <string>
#include <vector>

inline std::vector<char> ToVector(const std::string& s) {
  return std::vector<char>(s.begin(), s.end());
}
