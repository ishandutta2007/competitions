#pragma once

#include "common/base.h"
#include "common/geometry/d2/base.h"
#include "common/geometry/d2/distance.h"
#include "common/geometry/d2/point.h"

namespace geometry {
namespace d2 {
template <class T>
class Circle {
 public:
  Point<T> c;
  T r;

  constexpr Circle() : r() {}
  constexpr Circle(const Point<T>& _c, const T& _r) : c(_c), r(_r) {}

  constexpr bool Valid() const { return r >= 0; }

  constexpr bool Inside(const Point<T>& p) const {
    return SquaredDistance(p, c) <= r * r;
  }

  constexpr T operator()(const Point<T>& p) const { return Distance(p, c) - r; }
};
}  // namespace d2
}  // namespace geometry

using D2Circle = geometry::d2::Circle<double>;
