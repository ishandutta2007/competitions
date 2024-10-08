#pragma once

#include "common/base.h"
#include "common/graph/graph.h"

#include <stack>
#include <vector>

// Each value in the output vector is the mask for set of vertexes in component.
inline std::vector<uint64_t> ConnectedComponentsMask(const UndirectedGraph& g,
                                                     uint64_t mask = -1ull) {
  std::vector<uint64_t> components;
  const unsigned n = g.Size();
  assert(n <= 64);
  uint64_t visited = 0;
  std::stack<unsigned> s;
  for (unsigned i = 0; i < n; ++i) {
    const uint64_t ibit = (1ull << i);
    if (mask & ibit & ~visited) {
      visited |= ibit;
      uint64_t c = 0;
      for (s.push(i); !s.empty();) {
        const unsigned j = s.top();
        s.pop();
        c |= (1ull << j);
        for (unsigned k : g.Edges(j)) {
          uint64_t kbit = (1ull << k);
          if (mask & kbit & ~visited) {
            visited |= kbit;
            s.push(k);
          }
        }
      }
      components.push_back(c);
    }
  }
  return components;
}
