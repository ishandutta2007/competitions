#pragma once

#include <vector>

namespace modular {
namespace mstatic {
template <class TModular, bool keep_inverted = true>
class Factorial {
 protected:
  std::vector<TModular> vf, vfi;

 public:
  constexpr Factorial() {
    vf.push_back(TModular(1));
    if (keep_inverted) vfi.push_back(TModular(1));
  }

  constexpr void Adjust(unsigned n) {
    const unsigned k = vf.size();
    if (k > n) return;
    vf.resize(n + 1);
    for (unsigned i = k; i <= n; ++i) vf[i] = vf[i - 1] * TModular(i);
    if (keep_inverted) {
      vfi.resize(n + 1);
      vfi[n] = vf[n].Inverse();
      for (unsigned i = n; i > k; --i) vfi[i - 1] = vfi[i] * TModular(i);
    }
  }

  constexpr TModular Get(unsigned n) {
    Adjust(n);
    return vf[n];
  }

  constexpr TModular GetI(unsigned n) {
    Adjust(n);
    return (keep_inverted ? vfi[n] : vf[n].Inverse());
  }

  constexpr TModular Inverse(unsigned n) {
    Adjust(n);
    return vf[n - 1] * vfi[n];
  }

  constexpr TModular operator()(unsigned n) { return Get(n); }

  constexpr TModular BinomialCoefficient(unsigned n, unsigned k) {
    if (k > n) return 0;
    Adjust(n);
    return (keep_inverted ? vf[n] * vfi[k] * vfi[n - k]
                          : vf[n] / (vf[k] * vf[n - k]));
  }
};
}  // namespace mstatic
}  // namespace modular
