#pragma once

#include "common/base.h"
#include "common/geometry/d2/point.h"

namespace geometry {
namespace d2 {
namespace axis {
// Rectangle with edges parallel to coordinate axis
template <class T>
class Rectangle {
 public:
  using TPoint = Point<T>;

 public:
  TPoint p1, p2;

 public:
  constexpr Rectangle() : p1(), p2() {}

  constexpr Rectangle(const TPoint& _p1, const TPoint& _p2)
      : p1(_p1), p2(_p2) {}

  constexpr T Area() const { return (p2.x - p1.x) * (p2.y - p1.y); }

  constexpr bool Inside(const TPoint& p) const {
    return (p1.x <= p.x) && (p.x <= p2.x) && (p1.y <= p.y) && (p.y <= p2.y);
  }

  constexpr T InsideIndexXY(const TPoint& p) const {
    return (p.x - p1.x) * (p2.y - p1.y + 1) + (p.y - p1.y);
  }

  constexpr T InsideIndexYX(const TPoint& p) const {
    return (p.y - p1.y) * (p2.x - p1.x + 1) + (p.x - p1.x);
  }

  constexpr bool StrictInside(const TPoint& p) const {
    return (p1.x < p.x) && (p.x < p2.x) && (p1.y < p.y) && (p.y < p2.y);
  }
};
}  // namespace axis
}  // namespace d2
}  // namespace geometry

using D2ARectangle = geometry::d2::axis::Rectangle<double>;
using I2ARectangle = geometry::d2::axis::Rectangle<int64_t>;
