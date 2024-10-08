#pragma once

#include "common/base.h"
#include "common/geometry/d2/point.h"

namespace geometry {
namespace d2 {
template <class T1, class T2 = unsigned>
class WPoint : public Point<T1> {
 public:
  using TBase = Point<T1>;
  using TSelf = WPoint<T1, T2>;

  T2 w;

  constexpr WPoint() : w(T2(1)) {}

  constexpr WPoint(const T1& _x, const T1& _y, const T2& _w)
      : TBase(_x, _y), w(_w) {}

  constexpr explicit WPoint(const TBase& p) : TBase(p), w(1) {}

  constexpr WPoint(const TBase& p, const T2& _w) : TBase(p), w(_w) {}
};
}  // namespace d2
}  // namespace geometry

using D2WPoint = geometry::d2::WPoint<double>;
using I2WPoint = geometry::d2::WPoint<int64_t>;
