#pragma once

#include "common/heap/ukvm/pairing.h"

#include <utility>
#include <vector>

namespace graph {
namespace mst {
// Prim's algorithm using pairing heap.
// https://en.wikipedia.org/wiki/Prim%27s_algorithm
// Time: O((V + E) log V)
template <class TGraph, class TEdgeCostFunction>
inline std::pair<unsigned, typename TEdgeCostFunction::TEdgeCost> Prim(
    const TGraph& g, const TEdgeCostFunction& f,
    typename TEdgeCostFunction::TEdgeCost max_cost, unsigned source = 0) {
  using TEdgeCost = typename TEdgeCostFunction::TEdgeCost;
  std::vector<unsigned> visited(g.Size(), 0);
  unsigned edges_added = -1u;
  TEdgeCost total_cost = TEdgeCost();
  heap::ukvm::Pairing<TEdgeCost> q(std::vector<TEdgeCost>(g.Size(), max_cost),
                                   true);
  for (q.AddNewKey(source, TEdgeCost()); !q.Empty();) {
    unsigned u = q.ExtractKey();
    visited[u] = true;
    ++edges_added;
    total_cost += q.Get(u);
    for (auto e : g.EdgesEI(u)) {
      if (!visited[e.to]) q.DecreaseValueIfLess(e.to, f(e.info));
    }
  }
  return {edges_added, total_cost};
}
}  // namespace mst
}  // namespace graph
