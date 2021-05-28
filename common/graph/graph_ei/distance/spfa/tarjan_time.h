#pragma once

#include "common/graph/graph_ei/distance/spfa/dllr.h"

#include <queue>
#include <stack>
#include <vector>

namespace graph {
namespace distance {
namespace spfa {
// Modification of Tarjan negative-cycle detection algorithm.
// Time: O(VE) worst case
template <class TGraph, class TEdgeCostFunction, class TEdgeCost>
inline std::vector<TEdgeCost> TarjanTime(const TGraph& graph,
                                         const TEdgeCostFunction& f,
                                         unsigned source,
                                         const TEdgeCost& max_cost) {
  unsigned gsize = graph.Size();
  std::vector<TEdgeCost> v(gsize, max_cost);
  v[source] = TEdgeCost();
  std::queue<unsigned> q;
  std::stack<unsigned> s;
  std::vector<unsigned> inq(gsize, 0);
  DLLR dllr(gsize);
  dllr.SetRoot(source);
  q.push(source);
  for (unsigned t = 0, te = 0; !q.empty(); ++t) {
    unsigned u = q.front();
    q.pop();
    if (inq[u] != t) continue;
    auto ucost = v[u];
    for (const auto& e : graph.EdgesEI(u)) {
      if (ucost + f(e.info) < v[e.to]) {
        unsigned u2 = e.to;
        auto n2 = dllr.Get(u2);
        if (n2->p) {
          // Disassemble subtree
          n2->p->nchilds -= 1;
          auto nl = n2->l;
          for (unsigned k = 1; k > 0; --k) {
            k += n2->nchilds;
            n2->Reset();
            unsigned u3 = dllr.Get(n2);
            if (u3 == u) return {};  // Negative cycle
            inq[u3] = 0;
            n2 = n2->r;
          }
          nl->r = n2;
          n2->l = nl;
        }
        v[u2] = ucost + f(e.info);
        dllr.SetParent(u2, u);
        inq[u2] = ++te;
        q.push(u2);
      }
    }
  }
  return v;
}
}  // namespace spfa
}  // namespace distance
}  // namespace graph
