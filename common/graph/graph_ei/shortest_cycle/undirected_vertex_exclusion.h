#pragma once

#include "common/base.h"
#include "common/graph/graph_ei.h"
#include "common/heap/ukvm/dheap.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace graph {
namespace scycle {
// Find shortest cycle length by vertex exclusion.
// All edges cost are non-negative.
// Works for sparse graphs.
namespace hidden {
template <class TValue>
inline TValue VertexExclusionI(
    std::vector<std::unordered_map<unsigned, TValue>>& edges,
    const TValue& max_cost) {
  TValue best_cost = max_cost;
  std::vector<unsigned> vpc;
  vpc.reserve(edges.size());
  for (auto& m : edges) vpc.push_back(m.size());
  heap::ukvm::DHeap<4, unsigned> h(vpc, false);
  for (; !h.Empty();) {
    auto u = h.ExtractKey();
    for (auto it1 = edges[u].begin(); it1 != edges[u].end(); ++it1) {
      if (!(it1->second < best_cost)) continue;
      for (auto it2 = edges[u].begin(); it2 != it1; ++it2) {
        auto u1 = it1->first, u2 = it2->first;
        auto d12 = it1->second + it2->second;
        if (!(d12 < best_cost)) continue;
        auto itu12 = edges[u1].find(u2);
        if (itu12 == edges[u1].end()) {
          edges[u1][u2] = edges[u2][u1] = d12;
        } else {
          best_cost = std::min(best_cost, itu12->second + d12);
          if (itu12->second > d12) {
            edges[u1][u2] = edges[u2][u1] = d12;
          }
        }
      }
    }
    for (auto it1 : edges[u]) {
      auto u1 = it1.first;
      edges[u1].erase(u);
      h.Set(u1, edges[u1].size());
    }
  }
  return best_cost;
}
}  // namespace hidden

template <class TValue>
inline TValue VertexExclusion(const UndirectedGraphEI<TValue>& g,
                              TValue max_cost) {
  std::vector<std::unordered_map<unsigned, TValue>> edges(g.Size());
  for (unsigned u = 0; u < g.Size(); ++u) {
    auto& ue = g.Edges(u);
    auto& uei = g.EdgesInfo(u);
    assert(ue.size() == uei.size());
    for (unsigned j = 0; j < ue.size(); ++j) {
      auto v = ue[j];
      if (v == u) {
        max_cost = std::min(max_cost, uei[j]);
        continue;
      }
      auto it = edges[u].find(v);
      if (it == edges[u].end()) {
        edges[u][v] = uei[j];
      } else {
        max_cost = std::min(max_cost, it->second + uei[j]);
        it->second = std::min(it->second, uei[j]);
      }
    }
  }
  return hidden::VertexExclusionI(edges, max_cost);
}
}  // namespace scycle
}  // namespace graph
