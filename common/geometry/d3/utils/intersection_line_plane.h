#pragma once

#include "common/geometry/d3/line.h"
#include "common/geometry/d3/plane.h"

template <class T>
constexpr T Intersection(const geometry::d3::Line<T>& l,
                         const geometry::d3::Plane<T>& p) {
  const T x0 = p(l(0)), x1 = p(l(1));
  return x0 / (x0 - x1);
}
