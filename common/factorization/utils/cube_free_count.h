#pragma once

#include "common/base.h"

#include <vector>

constexpr uint64_t CubeFreeCount(uint64_t n,
                                 const std::vector<uint64_t>& vprimes,
                                 size_t first_prime_to_check = 0) {
  uint64_t s = n;
  for (size_t i = first_prime_to_check; i < vprimes.size(); ++i) {
    const uint64_t p = vprimes[i];
    const uint64_t np = n / (p * p * p);
    if (np == 0) break;
    s -= CubeFreeCount(np, vprimes, i + 1);
  }
  return s;
}
