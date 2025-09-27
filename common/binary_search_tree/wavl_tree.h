#pragma once

#include "common/base.h"
#include "common/binary_search_tree/base/deferred.h"
#include "common/binary_search_tree/base/node.h"
#include "common/binary_search_tree/base/rotate.h"
#include "common/binary_search_tree/base/self_balancing_tree.h"
#include "common/binary_search_tree/base/sibling.h"
#include "common/binary_search_tree/base/subtree_data.h"
#include "common/binary_search_tree/subtree_data/balance/wavl_rank.h"
#include "common/binary_search_tree/subtree_data/size.h"
#include "common/memory/contiguous_nodes_manager.h"
#include "common/templates/tuple.h"

#include <tuple>

namespace bst {

/**
 * @brief WAVL tree implementation with weak AVL balancing.
 *
 * This tree implements a self-balancing binary search tree where the rank
 * difference between a parent and its child is at least 1 and at most 2.
 * This ensures that the tree maintains O(log n) height for all operations.
 *
 * Key features:
 * - Automatic rank-based balancing after insertions and removals
 * - Guaranteed O(log n) time complexity for all operations
 * - Efficient join and split operations with balance maintenance
 * - Support for parent pointers, custom data, aggregators, and deferred
 * operations
 *
 * The tree uses rank information stored in each node to maintain balance
 * through rotations. It automatically adds rank tracking to the subtree data
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
class WAVLTree
    : public base::SelfBalancingTree<
          NodesManager<base::Node<
              Data,
              base::SubtreeData<templates::PrependT<subtree_data::WAVLRank,
                                                    AggregatorsTuple>>,
              base::Deferred<DeferredTuple>, has_parent, has_key, Key>>,
          WAVLTree<has_key, has_parent, Data, AggregatorsTuple, DeferredTuple,
                   Key, NodesManager>> {
 public:
  using SubtreeDataType = base::SubtreeData<
      templates::PrependT<subtree_data::WAVLRank, AggregatorsTuple>>;
  using DeferredType = base::Deferred<DeferredTuple>;
  using NodeType =
      base::Node<Data, SubtreeDataType, DeferredType, has_parent, has_key, Key>;
  using Self = WAVLTree<has_key, has_parent, Data, AggregatorsTuple,
                        DeferredTuple, Key, NodesManager>;
  using SBTree = base::SelfBalancingTree<NodesManager<NodeType>, Self>;
  using Base = typename SBTree::Base;

  friend Base;
  friend SBTree;

 public:
  /**
   * @brief Constructs a WAVL tree with the specified maximum number of nodes.
   *
   * @param max_nodes The maximum number of nodes to reserve
   */
  [[nodiscard]] explicit constexpr WAVLTree(size_t max_nodes)
      : SBTree(max_nodes) {}

 protected:
  /**
   * @brief Returns the rank of a node.
   *
   * @param node The node to get the rank of
   * @return The rank of the node, or -1 if the node is nullptr
   */
  [[nodiscard]] static constexpr int rank(const NodeType* node) noexcept {
    return node ? subtree_data::WAVLRank::get(node) : -1;
  }

  /**
   * @brief Calculates the rank difference between parent and child.
   *
   * The rank difference is the difference between the ranks of the
   * parent and child nodes. In a WAVL tree, this difference must be
   * at least 1 and at most 2.
   *
   * @param parent The parent node
   * @param child The child node
   * @return The rank difference (parent rank - child rank)
   */
  [[nodiscard]] static constexpr int rank_diff(const NodeType* parent,
                                               const NodeType* child) noexcept {
    return parent ? rank(parent) - rank(child) : 1;
  }

  /**
   * @brief Sets the rank of a node.
   *
   * @param node The node to set the rank for
   * @param rank The new rank value
   */
  static constexpr void set_rank(NodeType* node, int rank) noexcept {
    subtree_data::WAVLRank::set(node, rank);
  }

  /**
   * @brief Updates the rank of a node based on its children.
   *
   * The new rank is set to one more than the maximum rank of its children.
   *
   * @param node The node to update
   */
  static constexpr void update_rank(NodeType* node) noexcept {
    set_rank(node, std::max(rank(node->left), rank(node->right)) + 1);
  }

  /**
   * @brief Updates the rank of a node and all its descendants.
   *
   * This function recursively updates the ranks of all nodes in the subtree
   * rooted at the given node. It ensures that the rank of each node is
   * one more than the maximum rank of its children.
   *
   * @param root The root of the subtree to update
   */
  static constexpr void update_rank_subtree(NodeType* root) {
    if (!root) return;
    update_rank_subtree(root->left);
    update_rank_subtree(root->right);
    update_rank(root);
  }

  /**
   * @brief Increments the rank of a node.
   *
   * @param node The node to increment the rank of
   */
  static constexpr void inc_rank(NodeType* node) noexcept {
    subtree_data::WAVLRank::inc(node);
  }

  /**
   * @brief Decrements the rank of a node.
   *
   * @param node The node to decrement the rank of
   */
  static constexpr void dec_rank(NodeType* node) noexcept {
    subtree_data::WAVLRank::dec(node);
  }

  /**
   * @brief Builds a balanced WAVL tree from a sorted array of nodes.
   *
   * This function constructs a balanced WAVL tree from a sorted array of nodes.
   * It uses the middle element as the root and recursively builds the left and
   * right subtrees. After building the tree, it updates the ranks of all nodes
   * to maintain the WAVL balance property.
   *
   * @tparam update_leafs Whether to update subtree data for leaf nodes
   * @param vnodes The sorted array of nodes to build the tree from
   * @return The root of the constructed tree
   */
  template <bool update_leafs>
  [[nodiscard]] static constexpr NodeType* build_tree_impl(
      const std::vector<NodeType*>& vnodes) {
    NodeType* root = SBTree::template build_tree_impl<update_leafs>(vnodes);
    update_rank_subtree(root);
    return root;
  }

  /**
   * @brief Fixes balance after insertion using single rotation.
   *
   * @param child The child node to rotate up
   * @param parent The parent node to rotate down
   * @return The new root of the subtree
   */
  [[nodiscard]] static constexpr NodeType* fix_balance_insert_rotate1(
      NodeType* child, NodeType* parent) {
    dec_rank(parent);
    base::rotate<true, false, NodeType>(child, parent, nullptr);
    return child;
  }

  /**
   * @brief Fixes balance after insertion using double rotation.
   *
   * @param grandchild The grandchild node to rotate up
   * @param child The child node
   * @param parent The parent node
   * @return The new root of the subtree
   */
  [[nodiscard]] static constexpr NodeType* fix_balance_insert_rotate2(
      NodeType* grandchild, NodeType* child, NodeType* parent) {
    assert(rank_diff(child, grandchild) == 1);
    assert(rank_diff(child, base::sibling(grandchild, child)) == 2);
    dec_rank(parent);
    dec_rank(child);
    inc_rank(grandchild);
    base::rotate<false, true, NodeType>(grandchild, child, parent);
    base::rotate<true, false, NodeType>(grandchild, parent, nullptr);
    return grandchild;
  }

  /**
   * @brief Fixes balance after insertion.
   *
   * This function performs the necessary rotations to restore the
   * WAVL balance property after an insertion.
   *
   * @param root The root of the subtree to balance
   * @return The new root of the balanced subtree
   */
  [[nodiscard]] static constexpr NodeType* fix_balance_insert(NodeType* root) {
    if (rank_diff(root, root->right) == 0) {
      root->right->apply_deferred();
      if (rank_diff(root, root->left) == 1) {
        inc_rank(root);
      } else if (rank_diff(root->right, root->right->right) == 1) {
        assert(rank_diff(root->right, root->right->left) == 2);
        return fix_balance_insert_rotate1(root->right, root);
      } else {
        return fix_balance_insert_rotate2(root->right->left, root->right, root);
      }
    } else if (rank_diff(root, root->left) == 0) {
      root->left->apply_deferred();
      if (rank_diff(root, root->right) == 1) {
        inc_rank(root);
      } else if (rank_diff(root->left, root->left->left) == 1) {
        assert(rank_diff(root->left, root->left->right) == 2);
        return fix_balance_insert_rotate1(root->left, root);
      } else {
        return fix_balance_insert_rotate2(root->left->right, root->left, root);
      }
    }
    return root;
  }

  /**
   * @brief Fixes balance after removal using single rotation.
   *
   * @param child The child node to rotate up
   * @param parent The parent node to rotate down
   * @return The new root of the subtree
   */
  [[nodiscard]] static constexpr NodeType* fix_balance_remove_rotate1(
      NodeType* child, NodeType* parent) {
    dec_rank(parent);
    inc_rank(child);
    base::rotate<true, true, NodeType>(child, parent, nullptr);
    if (!parent->left && !parent->right) {
      set_rank(parent, 0);
    }
    return child;
  }

  /**
   * @brief Fixes balance after removal using double rotation.
   *
   * @param grandchild The grandchild node to rotate up
   * @param child The child node
   * @param parent The parent node
   * @return The new root of the subtree
   */
  [[nodiscard]] static constexpr NodeType* fix_balance_remove_rotate2(
      NodeType* grandchild, NodeType* child, NodeType* parent) {
    assert(grandchild);
    set_rank(parent, rank(parent) - 2);
    dec_rank(child);
    set_rank(grandchild, rank(grandchild) + 2);
    base::rotate<false, true, NodeType>(grandchild, child, parent);
    base::rotate<true, false, NodeType>(grandchild, parent, nullptr);
    return grandchild;
  }

  /**
   * @brief Fixes balance after removal.
   *
   * This function performs the necessary rotations to restore the
   * WAVL balance property after a removal.
   *
   * @param root The root of the subtree to balance
   * @return The new root of the balanced subtree
   */
  [[nodiscard]] static constexpr NodeType* fix_balance_remove(NodeType* root) {
    if (!root->left && !root->right) {
      set_rank(root, 0);
      return root;
    }
    NodeType* child = (rank_diff(root, root->left) > 2)    ? root->left
                      : (rank_diff(root, root->right) > 2) ? root->right
                                                           : root;
    if (child == root) return root;
    assert(rank_diff(root, child) == 3);
    NodeType* sibling = base::sibling(child, root);
    if (rank_diff(root, sibling) == 2) {
      dec_rank(root);
      return root;
    }
    assert((rank_diff(root, sibling) == 1));
    sibling->apply_deferred();
    if ((rank_diff(sibling, sibling->left) == 2) &&
        (rank_diff(sibling, sibling->right) == 2)) {
      dec_rank(sibling);
      dec_rank(root);
      return root;
    } else if (child == root->left) {
      if (rank_diff(sibling, sibling->right) == 1) {
        return fix_balance_remove_rotate1(sibling, root);
      } else {
        return fix_balance_remove_rotate2(sibling->left, sibling, root);
      }
    } else {
      if (rank_diff(sibling, sibling->left) == 1) {
        return fix_balance_remove_rotate1(sibling, root);
      } else {
        return fix_balance_remove_rotate2(sibling->right, sibling, root);
      }
    }
  }

  /**
   * @brief Base implementation of three-way join.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_base_impl(NodeType* l,
                                                           NodeType* m1,
                                                           NodeType* r) {
    SBTree::join3_base_impl(l, m1, r);
    update_rank(m1);
    return m1;
  }

  /**
   * @brief Joins three trees with left subtree potentially taller.
   *
   * This function is used when the left subtree is significantly taller
   * than the right subtree. It ensures that the resulting tree maintains
   * the WAVL balance property.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param hr The rank of the right tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_left_impl(NodeType* l,
                                                           NodeType* m1,
                                                           NodeType* r,
                                                           int hr) {
    if (rank(l) <= hr + 1) return join3_base_impl(l, m1, r);
    l->apply_deferred();
    l->set_right(join3_left_impl(l->right, m1, r, hr));
    l->update_subtree_data();
    assert(rank(l) >= rank(l->right));
    return fix_balance_insert(l);
  }

  /**
   * @brief Joins three trees with right subtree potentially taller.
   *
   * This function is used when the right subtree is significantly taller
   * than the left subtree. It ensures that the resulting tree maintains
   * the WAVL balance property.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param hl The rank of the left tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_right_impl(NodeType* l,
                                                            NodeType* m1,
                                                            NodeType* r,
                                                            int hl) {
    if (rank(r) <= hl + 1) return join3_base_impl(l, m1, r);
    r->apply_deferred();
    r->set_left(join3_right_impl(l, m1, r->left, hl));
    r->update_subtree_data();
    assert(rank(r) >= rank(r->left));
    return fix_balance_insert(r);
  }

  /**
   * @brief Implementation of three-way join operation for WAVL trees.
   *
   * This function joins three trees while maintaining the WAVL balance
   * property. It chooses between different joining strategies based on
   * the relative ranks of the subtrees.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @return The root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_impl(NodeType* l, NodeType* m1,
                                                      NodeType* r) {
    const auto hl = rank(l), hr = rank(r), hd = hl - hr;
    return (hd > 1)    ? join3_left_impl(l, m1, r, hr)
           : (hd < -1) ? join3_right_impl(l, m1, r, hl)
                       : join3_base_impl(l, m1, r);
  }
};

}  // namespace bst
