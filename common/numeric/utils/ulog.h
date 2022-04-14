#pragma once

template <class T>
inline unsigned ULogK(const T& x, const T& k) {
  T xx(x);
  unsigned l = 0;
  for (; xx; xx /= k) ++l;
  return (l > 0) ? l - 1 : 0;
}

template <class T>
inline unsigned ULog10(const T& x) {
  return ULogK(x, T(10));
}
