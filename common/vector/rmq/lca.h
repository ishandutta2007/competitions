#pragma once

#include "common/base.h"
#include "common/binary_search_tree/base/deferred.h"
#include "common/binary_search_tree/base/node.h"
#include "common/binary_search_tree/base/subtree_data.h"
#include "common/graph/tree.h"
#include "common/graph/tree/lca.h"
#include "common/memory/contiguous_nodes_manager.h"
#include "common/vector/rmq/position_value.h"

#include <stack>
#include <vector>

namespace nvector {
namespace rmq {
// Convert RMQ problem to LCA problem for binary tree using Cartesian Tree.
// O(N) memory, O(N) preprocessing time, O(1) request time
template <class TTValue>
class LCA {
 public:
  using TValue = TTValue;
  using TPositionValue = PositionValue<TValue>;
  using TNode =
      bst::base::Node<TValue, bst::base::SubtreeData<std::tuple<>>,
                      bst::base::Deferred<std::tuple<>>, false, false>;

 protected:
  std::vector<TValue> vv;
  graph::LCA lca;

 public:
  LCA() {}
  LCA(const std::vector<TValue>& v) { Build(v); }

  void Build(const std::vector<TValue>& v) {
    assert(v.size() > 0);
    vv = v;
    size_t n = v.size();
    // Build Cartesian Tree
    memory::ContiguousNodesManager<TNode> m(n);
    TNode *proot = m.create(), *plast = proot, *p0 = proot;
    proot->data = v[0];
    std::stack<TNode*> s;
    for (size_t j = 1; j < n; ++j) {
      TNode* pj = m.create();
      pj->data = v[j];
      if (!(pj->data < plast->data)) {
        plast->right = pj;
        s.push(plast);
      } else if (pj->data < proot->data) {
        for (; !s.empty();) s.pop();
        pj->left = proot;
        proot = pj;
      } else {
        for (; pj->data < s.top()->data; s.pop()) plast = s.top();
        pj->left = plast;
        s.top()->right = pj;
      }
      plast = pj;
    }
    // Convert to GraphTree
    TreeGraph t(n);
    t.SetRoot(proot - p0);
    for (size_t j = 0; j < n; ++j) {
      auto p = m.at(j);
      if (p->left) t.AddEdge(j, p->left - p0);
      if (p->right) t.AddEdge(j, p->right - p0);
    }
    lca.Build(t);
  }

  TPositionValue Minimum(size_t b, size_t e) const {
    assert(b < e);
    unsigned m = lca.GetLCA(b, e - 1);
    return {m, vv[m]};
  }
};
}  // namespace rmq
}  // namespace nvector
