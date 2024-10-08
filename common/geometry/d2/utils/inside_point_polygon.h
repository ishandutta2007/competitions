#pragma once

#include "common/geometry/d2/point.h"
#include "common/geometry/d2/polygon.h"
#include "common/geometry/d2/utils/locate_point_polygon.h"

namespace geometry {
namespace d2 {
template <class T>
constexpr bool Inside(const Point<T>& p, const Polygon<T>& plgn) {
  return location::Locate(p, plgn).type != location::Location::OUTSIDE;
}
}  // namespace d2
}  // namespace geometry
