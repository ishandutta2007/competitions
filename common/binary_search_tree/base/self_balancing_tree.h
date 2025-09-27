#pragma once

#include "common/base.h"
#include "common/binary_search_tree/base/base_tree.h"

namespace bst {
namespace base {

/**
 * @brief Base class for self-balancing binary search tree implementations.
 *
 * This class extends the base Tree class with automatic balancing
 * functionality. It provides common implementations for balanced tree
 * operations that maintain the tree's balance properties:
 * - Automatic rebalancing after insertions and removals
 * - Support for join operations with balance maintenance
 * - Efficient node removal from the rightmost position
 *
 * The class handles all common balancing operations:
 * - Fixing balance after insertions and removals
 * - Maintaining balance during joins and splits
 * - Updating subtree data during rebalancing
 *
 * Derived classes are responsible for:
 * - Implementing specific balancing strategies (e.g., AVL rotations, WAVL
 * rebalancing)
 * - Defining balance criteria and invariants
 * - Providing balance-specific node information
 *
 * @tparam NodesManager The node manager type that handles node allocation and
 *         management
 * @tparam Derived The derived class type (for CRTP)
 */
template <class NodesManager, class Derived>
class SelfBalancingTree : public BaseTree<NodesManager, Derived> {
 public:
  using Base = BaseTree<NodesManager, Derived>;

  using typename Base::KeyType;
  using typename Base::NodeType;

  friend Base;

  /**
   * @brief Constructs a self-balancing tree with the specified maximum number
   * of nodes.
   *
   * @param max_nodes The maximum number of nodes to reserve
   */
  [[nodiscard]] explicit constexpr SelfBalancingTree(size_t max_nodes)
      : Base(max_nodes) {}

  /**
   * @brief Removes the rightmost node from the tree.
   *
   * This function removes the rightmost node from the tree while maintaining
   * balance. It's commonly used in join operations to extract a node for
   * combining two trees.
   *
   * @param root The root of the tree
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* remove_right(
      NodeType* root, NodeType*& removed_node) {
    assert(root);
    return Derived::template remove_right_impl<true>(root, removed_node);
  }

 protected:
  /**
   * @brief Fixes balance after insertion.
   *
   * By default, uses the same balancing logic as fix_balance.
   * Can be overridden by derived classes if insertion-specific
   * balancing is needed.
   *
   * @param node The node that needs balance fixing
   * @return The new root of the subtree after balancing
   */
  [[nodiscard]] static constexpr NodeType* fix_balance_insert(NodeType* node) {
    return Derived::fix_balance(node);
  }

  /**
   * @brief Fixes balance after removal.
   *
   * By default, uses the same balancing logic as fix_balance.
   * Can be overridden by derived classes if removal-specific
   * balancing is needed.
   *
   * @param node The node that needs balance fixing
   * @return The new root of the subtree after balancing
   */
  [[nodiscard]] static constexpr NodeType* fix_balance_remove(NodeType* node) {
    return Derived::fix_balance(node);
  }

  /**
   * @brief Base implementation of node insertion.
   *
   * This function provides the common insertion logic for balanced trees:
   * - Recursively finds the correct position for the new node
   * - Updates parent links and subtree data
   * - Calls fix_balance_insert to maintain balance
   *
   * @param root The root of the tree
   * @param node The node to insert
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* insert_impl_base(NodeType* root,
                                                            NodeType* node) {
    if (!root) return node;
    root->apply_deferred();
    if (root->key < node->key) {
      root->set_right(insert_impl_base(root->right, node));
    } else {
      root->set_left(insert_impl_base(root->left, node));
    }
    root->update_subtree_data();
    return Derived::fix_balance_insert(root);
  }

  /**
   * @brief Implementation of node insertion with optional update.
   *
   * This function wraps insert_impl_base with optional subtree data update
   * for the new node.
   *
   * @tparam update_required Whether to update subtree data for the new node
   * @param root The root of the tree
   * @param node The node to insert
   * @return Pointer to the new root of the tree
   */
  template <bool update_required>
  [[nodiscard]] static constexpr NodeType* insert_impl(NodeType* root,
                                                       NodeType* node) {
    if constexpr (update_required) node->update_subtree_data();
    return insert_impl_base(root, node);
  }

  /**
   * @brief Base implementation of node insertion at a specific index.
   *
   * This function provides the common insertion logic for balanced trees
   * when inserting at a specific inorder index:
   * - Recursively finds the correct position based on subtree sizes
   * - Updates parent links and subtree data
   * - Calls fix_balance_insert to maintain balance
   *
   * @param root The root of the tree
   * @param node The node to insert
   * @param index The zero-based index where to insert
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* insert_at_impl_base(NodeType* root,
                                                               NodeType* node,
                                                               size_t index) {
    if (!root) {
      assert(index == 0);
      return node;
    }
    root->apply_deferred();
    const size_t lsize = bst::subtree_data::size(root->left);
    if (index <= lsize) {
      root->set_left(insert_at_impl_base(root->left, node, index));
    } else {
      root->set_right(
          insert_at_impl_base(root->right, node, index - lsize - 1));
    }
    root->update_subtree_data();
    return Derived::fix_balance_insert(root);
  }

  /**
   * @brief Implementation of node insertion at index with optional update.
   *
   * This function wraps insert_at_impl_base with optional subtree data update
   * for the new node.
   *
   * @tparam update_required Whether to update subtree data for the new node
   * @param root The root of the tree
   * @param node The node to insert
   * @param index The zero-based index where to insert
   * @return Pointer to the new root of the tree
   */
  template <bool update_required>
  [[nodiscard]] static constexpr NodeType* insert_at_impl(NodeType* root,
                                                          NodeType* node,
                                                          size_t index) {
    if constexpr (update_required) node->update_subtree_data();
    return insert_at_impl_base(root, node, index);
  }

  /**
   * @brief Implementation of node removal by key.
   *
   * This function removes a node with the specified key while maintaining
   * balance:
   * - Recursively finds the node to remove
   * - Handles node removal and subtree reconnection
   * - Calls fix_balance_remove to maintain balance
   *
   * @tparam reset_links Whether to reset links of the removed node
   * @param root The root of the tree
   * @param key The key of the node to remove
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static constexpr NodeType* remove_impl(
      NodeType* root, const KeyType& key, NodeType*& removed_node) {
    if (!root) return root;
    root->apply_deferred();
    if (root->key < key) {
      root->set_right(remove_impl<reset_links>(root->right, key, removed_node));
      root->update_subtree_data();
      return Derived::fix_balance_remove(root);
    } else if (root->key > key) {
      root->set_left(remove_impl<reset_links>(root->left, key, removed_node));
      root->update_subtree_data();
      return Derived::fix_balance_remove(root);
    } else {
      removed_node = root;
      NodeType *left = root->left, *right = root->right;
      if (left) left->set_parent(nullptr);
      if (right) right->set_parent(nullptr);
      if (!left) {
        root = right;
      } else if (!right) {
        root = left;
      } else {
        left = Derived::template remove_right_impl<false>(left, root);
        assert(root);
        root->set_left(left);
        root->set_right(right);
        root->subtree_data.bti_copy(removed_node);
        root->set_parent(nullptr);
        root->update_subtree_data();
        root = Derived::fix_balance_remove(root);
      }
      if constexpr (reset_links)
        removed_node->reset_links_and_update_subtree_data();
      return root;
    }
  }

  /**
   * @brief Implementation of node removal by index.
   *
   * This function removes a node at the specified inorder index while
   * maintaining balance:
   * - Recursively finds the node to remove based on subtree sizes
   * - Handles node removal and subtree reconnection
   * - Calls fix_balance_remove to maintain balance
   *
   * @tparam reset_links Whether to reset links of the removed node
   * @param root The root of the tree
   * @param index The zero-based index of the node to remove
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static constexpr NodeType* remove_at_impl(
      NodeType* root, size_t index, NodeType*& removed_node) {
    assert(root);
    root->apply_deferred();
    const size_t left_size = bst::subtree_data::size(root->left);
    if (index < left_size) {
      root->set_left(
          remove_at_impl<reset_links>(root->left, index, removed_node));
      root->update_subtree_data();
      return Derived::fix_balance_remove(root);
    } else if (index == left_size) {
      removed_node = root;
      NodeType *left = root->left, *right = root->right;
      if (left) left->set_parent(nullptr);
      if (right) right->set_parent(nullptr);
      if (!left) {
        root = right;
      } else if (!right) {
        root = left;
      } else {
        left = Derived::template remove_right_impl<false>(left, root);
        assert(root);
        root->set_left(left);
        root->set_right(right);
        root->subtree_data.bti_copy(removed_node);
        root->set_parent(nullptr);
        root->update_subtree_data();
        root = Derived::fix_balance_remove(root);
      }
      if constexpr (reset_links)
        removed_node->reset_links_and_update_subtree_data();
      return root;
    } else {
      root->set_right(remove_at_impl<reset_links>(
          root->right, index - left_size - 1, removed_node));
      root->update_subtree_data();
      return Derived::fix_balance_remove(root);
    }
  }

  /**
   * @brief Implementation of node removal by node pointer.
   *
   * This function removes a specific node while maintaining balance:
   * - Handles node removal and subtree reconnection
   * - Updates parent links and subtree data
   * - Calls fix_balance_remove to maintain balance
   *
   * @tparam reset_links Whether to reset links of the removed node
   * @param node The node to remove
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static NodeType* remove_node_impl(NodeType* node) {
    static_assert(Base::has_parent, "has_parent should be true");
    NodeType *left = node->left, *right = node->right, *parent = node->parent;
    if (left) left->set_parent(nullptr);
    if (right) right->set_parent(nullptr);
    NodeType *root, *pc = node;
    if (!left) {
      root = right;
    } else if (!right) {
      root = left;
    } else {
      left = Derived::template remove_right_impl<false>(left, root);
      assert(root);
      root->set_left(left);
      root->set_right(right);
      root->subtree_data.bti_copy(node);
      root->update_subtree_data();
      root = Derived::fix_balance_remove(root);
    }
    if constexpr (reset_links) node->reset_links_and_update_subtree_data();
    while (parent) {
      if (parent->left == pc) {
        parent->set_left(root);
      } else {
        parent->set_right(root);
      }
      parent->update_subtree_data();
      pc = parent;
      parent = parent->parent;
      root = Derived::fix_balance_remove(pc);
    }
    if (root) root->set_parent(nullptr);
    return root;
  }

  /**
   * @brief Implementation of rightmost node removal.
   *
   * This function removes the rightmost node from the tree while maintaining
   * balance. It's used internally by join operations.
   *
   * @tparam reset_links Whether to reset links of the removed node
   * @param root The root of the tree
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static constexpr NodeType* remove_right_impl(
      NodeType* root, NodeType*& removed_node) {
    root->apply_deferred();
    if (root->right) {
      root->set_right(
          remove_right_impl<reset_links>(root->right, removed_node));
      root->update_subtree_data();
      return Derived::fix_balance_remove(root);
    } else {
      removed_node = root;
      NodeType* left = root->left;
      if (left) left->set_parent(nullptr);
      if constexpr (reset_links) root->reset_links_and_update_subtree_data();
      return left;
    }
  }

  /**
   * @brief Implementation of two-way join operation.
   *
   * This function joins two trees while maintaining balance:
   * - Removes the rightmost node from the left tree
   * - Uses that node to join the trees in a balanced way
   *
   * @param l The root of the left tree
   * @param r The root of the right tree
   * @return Pointer to the new root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join_impl(NodeType* l, NodeType* r) {
    if (!l) return r;
    if (!r) return l;
    NodeType* node = nullptr;
    l = Derived::template remove_right_impl<true>(l, node);
    assert(node);
    return Derived::join3_impl(l, node, r);
  }

  /**
   * @brief Base implementation of three-way join operation.
   *
   * This function joins three trees in a balanced way:
   * - The middle node becomes the root
   * - Left and right trees become its children
   * - Updates parent links and subtree data
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @return Pointer to the new root of the joined tree (m1)
   */
  static constexpr NodeType* join3_base_impl(NodeType* l, NodeType* m1,
                                             NodeType* r) {
    m1->set_left(l);
    m1->set_right(r);
    m1->update_subtree_data();
    return m1;
  }

  /**
   * @brief Implementation of split operation by key.
   *
   * This function splits the tree at a given key while maintaining balance:
   * - Recursively splits the tree based on key comparison
   * - Maintains parent links and subtree data
   * - Ensures both parts remain balanced
   *
   * @param root The root of the tree to split
   * @param key The key to split at
   * @param output_l Reference to store the left part (keys < key)
   * @param output_r Reference to store the right part (keys >= key)
   */
  static constexpr void split_impl(NodeType* root, const KeyType& key,
                                   NodeType*& output_l, NodeType*& output_r) {
    root->apply_deferred();
    NodeType *l = root->left, *r = root->right, *m = nullptr;
    root->set_left(nullptr);
    root->set_right(nullptr);
    if (l) l->set_parent(nullptr);
    if (r) r->set_parent(nullptr);
    if (root->key < key) {
      if (r) {
        split_impl(r, key, m, output_r);
      } else {
        output_r = nullptr;
      }
      output_l = Derived::join3_impl(l, root, m);
    } else {
      if (l) {
        split_impl(l, key, output_l, m);
      } else {
        output_l = nullptr;
      }
      output_r = Derived::join3_impl(m, root, r);
    }
  }

  /**
   * @brief Implementation of split operation by index.
   *
   * This function splits the tree at a given inorder index while maintaining
   * balance:
   * - Recursively splits the tree based on subtree sizes
   * - Maintains parent links and subtree data
   * - Ensures both parts remain balanced
   *
   * @param root The root of the tree to split
   * @param lsize The size of the left part
   * @param output_l Reference to store the left part
   * @param output_r Reference to store the right part
   */
  static constexpr void split_at_impl(NodeType* root, size_t lsize,
                                      NodeType*& output_l,
                                      NodeType*& output_r) {
    root->apply_deferred();
    NodeType *l = root->left, *r = root->right, *m = nullptr;
    root->set_left(nullptr);
    root->set_right(nullptr);
    if (l) l->set_parent(nullptr);
    if (r) r->set_parent(nullptr);
    const size_t left_size = bst::subtree_data::size(l);
    if (lsize < left_size) {
      split_at_impl(l, lsize, output_l, m);
      output_r = Derived::join3_impl(m, root, r);
    } else if (lsize == left_size) {
      output_l = l;
      output_r = Derived::join3_impl(nullptr, root, r);
    } else if (lsize == left_size + 1) {
      output_l = Derived::join3_impl(l, root, nullptr);
      output_r = r;
    } else {
      split_at_impl(r, lsize - left_size - 1, m, output_r);
      output_l = Derived::join3_impl(l, root, m);
    }
  }
};

}  // namespace base
}  // namespace bst
