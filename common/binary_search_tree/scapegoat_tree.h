#pragma once

#include "common/binary_search_tree/base/deferred.h"
#include "common/binary_search_tree/base/node.h"
#include "common/binary_search_tree/base/self_balancing_tree.h"
#include "common/binary_search_tree/base/subtree_data.h"
#include "common/binary_search_tree/base/to_vector.h"
#include "common/binary_search_tree/subtree_data/size.h"
#include "common/memory/contiguous_nodes_manager.h"

namespace bst {

/**
 * @brief Scapegoat tree implementation with size-based balancing.
 *
 * This tree implements a self-balancing binary search tree where the size
 * of any subtree is at most alpha times the size of its parent. This ensures
 * that the tree maintains O(log n) height for all operations in amortized time.
 *
 * Key features:
 * - Automatic size-based balancing after insertions and removals
 * - Amortized O(log n) time complexity for operations
 * - Support for parent pointers, custom data, aggregators, and deferred
 * operations
 *
 * Note: While most operations have amortized O(log n) complexity, some
 * operations can be slow (similar to Splay trees). The tree achieves its
 * amortized bounds by occasionally rebuilding entire subtrees to restore
 * balance.
 *
 * The tree uses size information stored in each node to maintain balance
 * through rebuilding. It automatically adds size tracking to the subtree data
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
class ScapegoatTree
    : public base::SelfBalancingTree<
          NodesManager<base::Node<
              Data,
              base::SubtreeData<templates::PrependIfMissingT<subtree_data::Size,
                                                             AggregatorsTuple>>,
              base::Deferred<DeferredTuple>, has_parent, has_key, Key>>,
          ScapegoatTree<has_key, has_parent, Data, AggregatorsTuple,
                        DeferredTuple, Key, NodesManager>> {
 public:
  // Height is less or equal to 2 * height(fully balanced tree).
  static constexpr double alpha = 0.7;

  using SubtreeDataType = base::SubtreeData<
      templates::PrependIfMissingT<subtree_data::Size, AggregatorsTuple>>;
  using DeferredType = base::Deferred<DeferredTuple>;
  using NodeType =
      base::Node<Data, SubtreeDataType, DeferredType, has_parent, has_key, Key>;
  using Self = ScapegoatTree<has_key, has_parent, Data, AggregatorsTuple,
                             DeferredTuple, Key, NodesManager>;
  using SBTree = base::SelfBalancingTree<NodesManager<NodeType>, Self>;
  using Base = typename SBTree::Base;

  friend Base;
  friend SBTree;

  // Note: While these operations are supported in the interface, they are not
  // meant to be used at scale because their amortized time complexity is O(N)
  // in the worst case.
  static constexpr bool support_join = false;
  static constexpr bool support_split = false;

 public:
  /**
   * @brief Constructs a Scapegoat tree with the specified maximum number of
   * nodes.
   *
   * @param max_nodes The maximum number of nodes to reserve
   */
  [[nodiscard]] explicit constexpr ScapegoatTree(size_t max_nodes)
      : SBTree(max_nodes) {}

 protected:
  /**
   * @brief Rebuilds a subtree to maintain balance.
   *
   * This function rebuilds a subtree from scratch to ensure it is balanced.
   * It collects all nodes in the subtree in inorder traversal and then
   * builds a new balanced tree from them.
   *
   * @param node The root of the subtree to rebuild
   * @return The new root of the rebuilt subtree
   */
  [[nodiscard]] static NodeType* rebuild_subtree(NodeType* node) {
    assert(node);
    thread_local std::vector<NodeType*> nodes;
    nodes.clear();
    nodes.reserve(subtree_data::size(node));
    base::collect_inorder<true>(node, nodes, [](NodeType* n) { return n; });
    return Base::template build_tree_impl<true>(nodes);
  }

  /**
   * @brief Fixes the balance of a node using size-based criteria.
   *
   * This function checks if the size of either child subtree exceeds
   * alpha times the size of the parent. If so, it rebuilds the subtree
   * to restore balance.
   *
   * @param node The root of the subtree to balance
   * @return The new root of the balanced subtree
   */
  [[nodiscard]] static constexpr NodeType* fix_balance(NodeType* node) {
    assert(node);
    const auto s = size_t(alpha * subtree_data::size(node));
    return ((subtree_data::size(node->left) > s) ||
            (subtree_data::size(node->right) > s))
               ? rebuild_subtree(node)
               : node;
  }

  /**
   * @brief Joins three trees with left subtree potentially larger.
   *
   * This function is used when the left subtree is significantly larger
   * than the right subtree. It ensures that the resulting tree maintains
   * the Scapegoat balance property.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param rsize The size of the right tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_left_impl(NodeType* l,
                                                           NodeType* m1,
                                                           NodeType* r,
                                                           size_t rsize) {
    if (subtree_data::size(l) < size_t(alpha / (1 - alpha) * (rsize + 1)))
      return SBTree::join3_base_impl(l, m1, r);
    l->apply_deferred();
    l->set_right(join3_left_impl(l->right, m1, r, rsize));
    l->update_subtree_data();
    return fix_balance(l);
  }

  /**
   * @brief Joins three trees with right subtree potentially larger.
   *
   * This function is used when the right subtree is significantly larger
   * than the left subtree. It ensures that the resulting tree maintains
   * the Scapegoat balance property.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param lsize The size of the left tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_right_impl(NodeType* l,
                                                            NodeType* m1,
                                                            NodeType* r,
                                                            size_t lsize) {
    if (subtree_data::size(r) < size_t(alpha / (1 - alpha) * (lsize + 1)))
      return SBTree::join3_base_impl(l, m1, r);
    r->apply_deferred();
    r->set_left(join3_right_impl(l, m1, r->left, lsize));
    r->update_subtree_data();
    return fix_balance(r);
  }

 public:
  /**
   * @brief Implementation of three-way join operation for Scapegoat trees.
   *
   * This function joins three trees while maintaining the Scapegoat balance
   * property. It chooses between different joining strategies based on
   * the relative sizes of the subtrees.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_impl(NodeType* l, NodeType* m1,
                                                      NodeType* r) {
    const size_t lsize = subtree_data::size(l), rsize = subtree_data::size(r);
    return lsize >= rsize ? join3_left_impl(l, m1, r, rsize)
                          : join3_right_impl(l, m1, r, lsize);
  }
};

}  // namespace bst
