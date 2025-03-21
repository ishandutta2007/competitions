#pragma once

#include "common/base.h"
#include "common/binary_search_tree/ext/info/none.h"

namespace bst {
namespace info {
class Size : public None {
 public:
  using TSelf = Size;
  static const bool is_none = false;
  static const bool has_size = true;

  size_t size = 0;

  template <class TNode>
  void Update(TNode* node) {
    size = 1 + (node->l ? node->l->subtree_data.size : 0) +
           (node->r ? node->r->subtree_data.size : 0);
  }

  template <class TNode>
  void Insert(const TNode* node) {
    size += node->subtree_data.size;
  }

  template <class TNode>
  void Remove(const TNode* node) {
    size -= node->subtree_data.size;
  }
};
}  // namespace info
}  // namespace bst
