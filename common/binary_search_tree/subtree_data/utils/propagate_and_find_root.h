#pragma once

#include "common/base.h"

namespace bst {
namespace subtree_data {

/**
 * @brief Updates subtree data from a node up to the root and returns the root.
 *
 * This function ensures that all subtree data aggregators are updated along the
 * path from the given node to the root, similar to propagate_to_root. However,
 * unlike propagate_to_root which only updates the data, this function also
 * returns a pointer to the root of the tree.
 *
 * This is useful when you need both to propagate subtree data changes and
 * obtain a reference to the root node in a single operation, particularly
 * useful in tree operations where the root might change due to rotations
 * or rebalancing.
 *
 * The updates are performed from the node up to the root to ensure that each
 * node's subtree data is updated with the latest changes from its children.
 *
 * @tparam TNode The BST node type.
 * @param node The starting node to update from.
 * @return Pointer to the root of the tree.
 */
template <typename TNode>
[[nodiscard]] constexpr TNode* propagate_and_find_root(TNode* node) {
  static_assert(TNode::has_parent, "Node must have parent pointer enabled");
  assert(node);

  // Update subtree data from node up to root
  node->update_subtree_data();
  while (node->parent) {
    node = node->parent;
    node->update_subtree_data();
  }
  return node;
}

}  // namespace subtree_data
}  // namespace bst
