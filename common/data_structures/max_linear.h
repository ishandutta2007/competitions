#pragma once

#include "common/base.h"
#include "common/binary_search_tree/base/left.h"
#include "common/binary_search_tree/base/right.h"
#include "common/binary_search_tree/splay_tree.h"
#include "common/binary_search_tree/subtree_data/size.h"
#include "common/memory/nodes_manager.h"

#include <algorithm>

namespace ds {
template <class TTValue, class TTData>
class MaxLinear {
 public:
  using TValue = TTValue;
  using TData = TTData;

  class Linear {
   public:
    TValue a, b;
    TData data;

   public:
    constexpr TValue operator()(const TValue& x) const { return a * x + b; }
    constexpr const TData& Data() const { return data; }
  };

  class LinearExt {
   public:
    Linear l;
    double xl, xr;
  };

  using TTree =
      bst::SplayTree<true, LinearExt, std::tuple<bst::subtree_data::Size>,
                     std::tuple<>, TValue, memory::NodesManager>;
  using TNode = typename TTree::NodeType;

 public:
  TValue xl, xr;
  TTree tree;
  TNode* root = nullptr;

 public:
  MaxLinear(TValue _xl, TValue _xr) : xl(_xl), xr(_xr) {}

  void Add(const Linear& l) {
    if (!root) {
      root = tree.create_node(LinearExt{l, xl, xr}, l.a);
      return;
    }
    double lxl = xl, lxr = xr;
    TNode *pl, *pr;
    tree.split(root, l.a, pl, pr);
    for (; pr;) {
      pr = bst::base::left(pr);
      assert(pr);
      bst::base::splay(pr);
      assert(l.a <= pr->key);
      if (l.a == pr->key) {
        if (l.b <= pr->data.l.b) {
          lxr = xl;
          break;
        }
        pr = tree.remove_node(pr);
      } else {
        double x = (double(l.b) - double(pr->data.l.b)) /
                   (double(pr->data.l.a) - double(l.a));
        if (pr->data.xr <= x) {
          pr = tree.remove_node(pr);
        } else if (x <= pr->data.xl) {
          lxr = pr->data.xl;
          break;
        } else {
          pr->data.xl = lxr = x;
          break;
        }
      }
    }
    for (; pl;) {
      pl = bst::base::right(pl);
      assert(pl);
      bst::base::splay(pl);
      assert(pl->key < l.a);
      double x = (double(l.b) - double(pl->data.l.b)) /
                 (double(pl->data.l.a) - double(l.a));
      if (x <= pl->data.xl) {
        pl = tree.remove_node(pl);
      } else if (pl->data.xr <= x) {
        lxl = pl->data.xr;
        break;
      } else {
        pl->data.xr = lxl = x;
        break;
      }
    }
    lxl = std::max(lxl, double(xl));
    lxr = std::min(lxr, double(xr));
    if (lxl < lxr) {
      root = tree.join3(pl, tree.create_node(LinearExt{l, lxl, lxr}, l.a), pr);
    } else {
      root = tree.join(pl, pr);
    }
  }

  void Add(const TValue& a, const TValue& b, const TData& data) {
    Add({a, b, data});
  }

  bool Empty() const { return root == nullptr; }

  const Linear& Get(const TValue& x) const {
    assert(!Empty());
    assert((xl <= x) && (x <= xr));
    TNode* node = root;
    for (; node;) {
      if (x < node->data.xl)
        node = node->left;
      else if (node->data.xr < x)
        node = node->right;
      else
        break;
    }
    assert(node);
    return node->data.l;
  }
};
}  // namespace ds
