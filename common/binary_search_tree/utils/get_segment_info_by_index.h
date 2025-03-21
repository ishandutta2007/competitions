#pragma once

#include "common/base.h"

namespace bst {
template <class TTree>
inline typename TTree::TNode* GetSegmentInfoByIndex(
    typename TTree::TNode* root, size_t begin, size_t end,
    typename TTree::TInfo& output) {
  using TNode = typename TTree::TNode;
  TNode *l, *m, *r;
  TTree::SplitBySize(root, end, m, r);
  TTree::SplitBySize(m, begin, l, m);
  if (m) output = m->subtree_data;
  return TTree::Join(TTree::Join(l, m), r);
}
}  // namespace bst
