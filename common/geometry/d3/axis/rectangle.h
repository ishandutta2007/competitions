#pragma once

#include "common/base.h"
#include "common/geometry/d3/point.h"

namespace geometry {
namespace d3 {
namespace axis {
// Rectangle with edges parallel to coordinate axis
template <class T>
class Rectangle {
 public:
  Point<T> p1, p2;

  constexpr Rectangle() : p1(), p2() {}

  constexpr Rectangle(const Point<T>& _p1, const Point<T>& _p2)
      : p1(_p1), p2(_p2) {}

  constexpr bool Inside(const Point<T>& p) const {
    return (p1.x <= p.x) && (p.x <= p2.x) && (p1.y <= p.y) && (p.y <= p2.y) &&
           (p1.z <= p.z) && (p.z <= p2.z);
  }

  constexpr bool StrictInside(const Point<T>& p) const {
    return (p1.x < p.x) && (p.x < p2.x) && (p1.y < p.y) && (p.y < p2.y) &&
           (p1.z < p.z) && (p.z < p2.z);
  }
};
}  // namespace axis
}  // namespace d3
}  // namespace geometry

using D3ARectagnle = geometry::d3::axis::Rectangle<double>;
using I3ARectagnle = geometry::d3::axis::Rectangle<int64_t>;
