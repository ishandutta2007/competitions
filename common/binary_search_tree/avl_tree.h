#pragma once

#include "common/base.h"
#include "common/binary_search_tree/base/deferred.h"
#include "common/binary_search_tree/base/node.h"
#include "common/binary_search_tree/base/rotate.h"
#include "common/binary_search_tree/base/self_balancing_tree.h"
#include "common/binary_search_tree/base/subtree_data.h"
#include "common/binary_search_tree/subtree_data/height.h"
#include "common/memory/contiguous_nodes_manager.h"
#include "common/templates/tuple.h"

#include <tuple>

namespace bst {

/**
 * @brief AVL tree implementation with automatic height balancing.
 *
 * This tree implements a self-balancing binary search tree where the heights
 * of the two child subtrees of any node differ by at most one. This ensures
 * that the tree maintains O(log n) height for all operations.
 *
 * Key features:
 * - Automatic height balancing after insertions and removals
 * - Guaranteed O(log n) time complexity for all operations
 * - Efficient join and split operations with balance maintenance
 * - Support for parent pointers, custom data, aggregators, and deferred
 * operations
 *
 * The tree uses height information stored in each node to maintain balance
 * through rotations. It automatically adds height tracking to the subtree data
 * if not already present.
 *
 * @tparam has_key Whether the tree uses keys for ordering
 * @tparam has_parent Whether nodes maintain parent pointers
 * @tparam Data The data type stored in each node
 * @tparam AggregatorsTuple Tuple of aggregator types for subtree data
 * @tparam DeferredTuple Tuple of deferred operation types
 * @tparam Key The key type used for ordering (if has_key is true)
 * @tparam NodesManager The node manager type for memory management
 */
template <bool has_key, bool has_parent, typename Data,
          typename AggregatorsTuple = std::tuple<subtree_data::Size>,
          typename DeferredTuple = std::tuple<>, typename Key = int64_t,
          template <class> class NodesManager = memory::ContiguousNodesManager>
class AVLTree
    : public base::SelfBalancingTree<
          NodesManager<base::Node<
              Data,
              base::SubtreeData<templates::PrependIfMissingT<
                  subtree_data::Height, AggregatorsTuple>>,
              base::Deferred<DeferredTuple>, has_parent, has_key, Key>>,
          AVLTree<has_key, has_parent, Data, AggregatorsTuple, DeferredTuple,
                  Key, NodesManager>> {
 public:
  using SubtreeDataType = base::SubtreeData<
      templates::PrependIfMissingT<subtree_data::Height, AggregatorsTuple>>;
  using DeferredType = base::Deferred<DeferredTuple>;
  using NodeType =
      base::Node<Data, SubtreeDataType, DeferredType, has_parent, has_key, Key>;
  using Self = AVLTree<has_key, has_parent, Data, AggregatorsTuple,
                       DeferredTuple, Key, NodesManager>;
  using SBTree = base::SelfBalancingTree<NodesManager<NodeType>, Self>;
  using Base = typename SBTree::Base;

  friend Base;
  friend SBTree;

 public:
  /**
   * @brief Constructs an AVL tree with the specified maximum number of nodes.
   *
   * @param max_nodes The maximum number of nodes to reserve
   */
  [[nodiscard]] explicit constexpr AVLTree(size_t max_nodes)
      : SBTree(max_nodes) {}

 protected:
  /**
   * @brief Returns the height of a node.
   *
   * @param node The node to get the height of
   * @return The height of the node, or 0 if the node is nullptr
   */
  [[nodiscard]] static constexpr int height(const NodeType* node) noexcept {
    return bst::subtree_data::height(node);
  }

  /**
   * @brief Calculates the balance factor of a node.
   *
   * The balance factor is the difference between the heights of the
   * left and right subtrees. A value outside [-1, 1] indicates that
   * the node is unbalanced.
   *
   * @param node The node to calculate the balance factor for
   * @return The balance factor (left height - right height)
   */
  [[nodiscard]] static constexpr int balance(NodeType* node) noexcept {
    return node ? height(node->left) - height(node->right) : 0;
  }

  /**
   * @brief Fixes the balance of a node using AVL rotations.
   *
   * This function performs the necessary rotations to restore the
   * AVL balance property. It handles both single and double rotations
   * depending on the balance factors of the node and its children.
   *
   * @param root The root of the subtree to balance
   * @return The new root of the balanced subtree
   */
  [[nodiscard]] static constexpr NodeType* fix_balance(NodeType* root) {
    const int b = balance(root);
    assert((b >= -2) && (b <= 2));
    if (b == 2) {
      root->left->apply_deferred();
      if (balance(root->left) == -1)
        base::rotate<false, true, NodeType>(root->left->right, root->left,
                                            root);
      NodeType* child = root->left;
      base::rotate<true, false, NodeType>(child, root, nullptr);
      return child;
    } else if (b == -2) {
      root->right->apply_deferred();
      if (balance(root->right) == 1)
        base::rotate<false, true, NodeType>(root->right->left, root->right,
                                            root);
      NodeType* child = root->right;
      base::rotate<true, false, NodeType>(child, root, nullptr);
      return child;
    }
    return root;
  }

  /**
   * @brief Joins three trees with left subtree potentially taller.
   *
   * This function is used when the left subtree is significantly taller
   * than the right subtree. It ensures that the resulting tree maintains
   * the AVL balance property.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param hr The height of the right tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_left_impl(NodeType* l,
                                                           NodeType* m1,
                                                           NodeType* r,
                                                           int hr) {
    if (height(l) <= hr + 1) return SBTree::join3_base_impl(l, m1, r);
    l->apply_deferred();
    l->set_right(join3_left_impl(l->right, m1, r, hr));
    l->update_subtree_data();
    return fix_balance(l);
  }

  /**
   * @brief Joins three trees with right subtree potentially taller.
   *
   * This function is used when the right subtree is significantly taller
   * than the left subtree. It ensures that the resulting tree maintains
   * the AVL balance property.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param hl The height of the left tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_right_impl(NodeType* l,
                                                            NodeType* m1,
                                                            NodeType* r,
                                                            int hl) {
    if (height(r) <= hl + 1) return SBTree::join3_base_impl(l, m1, r);
    r->apply_deferred();
    r->set_left(join3_right_impl(l, m1, r->left, hl));
    r->update_subtree_data();
    return fix_balance(r);
  }

  /**
   * @brief Implementation of three-way join operation for AVL trees.
   *
   * This function joins three trees while maintaining the AVL balance
   * property. It chooses between different joining strategies based on
   * the relative heights of the subtrees.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_impl(NodeType* l, NodeType* m1,
                                                      NodeType* r) {
    const auto hl = height(l), hr = height(r), hd = hl - hr;
    return (hd > 1)    ? join3_left_impl(l, m1, r, hr)
           : (hd < -1) ? join3_right_impl(l, m1, r, hl)
                       : SBTree::join3_base_impl(l, m1, r);
  }
};

}  // namespace bst
