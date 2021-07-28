#pragma once

#include "common/geometry/d2/axis/rectangle.h"
#include "common/geometry/d2/point.h"
#include "common/geometry/d2/polygon.h"
#include "common/geometry/d2/segment.h"
#include "common/geometry/d2/stl_hash/segment.h"

#include <unordered_map>
#include <vector>

class Problem;

class Cache {
 protected:
  I2Polygon hole;
  I2ARectangle box;
  std::vector<std::vector<unsigned>> valid_points_map;
  std::vector<I2Point> valid_points;
  std::unordered_map<I2ClosedSegment, bool> valid_segments_map;
  std::vector<std::vector<int64_t>> figure_min_distance;
  std::vector<std::vector<int64_t>> figure_max_distance;

 public:
  void Init(const Problem& problem);

  const std::vector<I2Point>& GetValidPoints() const { return valid_points; }
  bool Test(const I2Point& p) const;
  bool Test(const I2ClosedSegment& s);
  bool TestI(const I2ClosedSegment& s);
};