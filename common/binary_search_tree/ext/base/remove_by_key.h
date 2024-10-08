#pragma once

#include "common/binary_search_tree/ext/base/remove_by_key__default.h"
#include "common/binary_search_tree/ext/base/remove_by_key__skip_update.h"
#include "common/binary_search_tree/ext/base/remove_by_key__use_parent.h"
#include "common/template.h"

namespace bst {
namespace ext {
namespace base {
// TODO:
//   Add version for use_parent && support_remove.
namespace hidden {
template <class TNode, class TKey>
inline TNode* RemoveByKeyI(TNode* root, const TKey& key, TNode*& removed_node,
                           TNode*& first_changed_node, TMetaFalse, TMetaFalse) {
  return RemoveByKeyDefault(root, key, removed_node, first_changed_node);
}

template <class TNode, class TKey>
inline TNode* RemoveByKeyI(TNode* root, const TKey& key, TNode*& removed_node,
                           TNode*& first_changed_node, TMetaFalse, TMetaTrue) {
  return RemoveByKeyUseParent(root, key, removed_node, first_changed_node);
}

template <class TNode, class TKey>
inline TNode* RemoveByKeyI(TNode* root, const TKey& key, TNode*& removed_node,
                           TNode*& first_changed_node, TMetaTrue, TMetaFalse) {
  return RemoveByKeySkipUpdate(root, key, removed_node, first_changed_node);
}

template <class TNode, class TKey>
inline TNode* RemoveByKeyI(TNode* root, const TKey& key, TNode*& removed_node,
                           TNode*& first_changed_node, TMetaTrue, TMetaTrue) {
  return RemoveByKeySkipUpdate(root, key, removed_node, first_changed_node);
}
}  // namespace hidden

template <class TNode, class TKey>
inline TNode* RemoveByKey(TNode* root, const TKey& key, TNode*& removed_node,
                          TNode*& first_changed_node) {
  static_assert(TNode::use_key, "use_key should be true");
  if (!root) return root;
  return hidden::RemoveByKeyI(root, key, removed_node, first_changed_node,
                              TMetaBool<TNode::TInfo::is_none>(),
                              TMetaBool<TNode::use_parent>());
}
}  // namespace base
}  // namespace ext
}  // namespace bst
