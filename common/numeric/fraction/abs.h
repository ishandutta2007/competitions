#pragma once

#include "common/numeric/fraction.h"
#include "common/numeric/utils/abs.h"

template <typename T>
constexpr numeric::Fraction<T> Abs(const numeric::Fraction<T>& val) {
  return {Abs(val.GetN()), val.GetD()};
}
