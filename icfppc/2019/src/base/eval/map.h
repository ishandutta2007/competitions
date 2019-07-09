#pragma once

#include "base/core/map.h"
#include "base/point.h"
#include <cstdint>
#include <string>
#include <vector>

namespace base {
namespace eval {
class Map : public core::Map {
 protected:
  static const uint8_t OBSTACLE = 1;
  static const uint8_t WRAPPED = 2;

 protected:
  std::vector<uint8_t> map;

 protected:
  void ResetSize();
  void InitMap(const std::string& map_encoded);

 public:
  void Init(const std::string& problem);

  bool Obstacle(const Point& p) const;
  void Wrap(const Point& p);
  void Drill(const Point& p);

  bool Wrapped() const;
};
}  // namespace eval
}  // namespace base