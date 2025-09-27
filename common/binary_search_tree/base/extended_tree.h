#pragma once

#include "common/base.h"
#include "common/binary_search_tree/base/at.h"
#include "common/binary_search_tree/base/base_tree.h"
#include "common/binary_search_tree/base/find.h"

#include <vector>

namespace bst {
namespace base {

/**
 * @brief Extended base class for binary search tree operations.
 *
 * This class extends BaseTree with additional functionality:
 * - Tree building operations
 * - Insert and remove operations
 * - Join and split operations
 * - Order-based operations
 *
 * @tparam NodesManager The node manager type that handles node allocation and
 * management
 * @tparam Derived The derived class type (for CRTP)
 */
template <typename NodesManager, typename Derived>
class ExtendedTree : public BaseTree<NodesManager, Derived> {
 public:
  using Base = BaseTree<NodesManager, Derived>;
  using typename Base::DataType;
  using typename Base::KeyType;
  using typename Base::NodeType;
  using typename Base::SubtreeDataType;

  /**
   * @brief Operation support flags.
   *
   * These flags determine which operations are supported by the tree:
   * - support_insert: Whether insertion operations are supported
   * - support_remove: Whether removal operations are supported (requires parent
   * links)
   * - support_remove_by_node: Whether node-based removal is supported
   * - support_join3: Whether three-way join operations are supported
   * - support_join: Whether two-way join operations are supported
   * - support_split: Whether split operations are supported
   * - support_insert_at: Whether order-based insertion is supported
   */
  static constexpr bool support_insert = true;
  static constexpr bool support_remove = Base::has_parent;
  static constexpr bool support_remove_by_node =
      Base::has_parent && Derived::support_remove;
  static constexpr bool support_join3 = false;
  static constexpr bool support_join =
      Derived::support_join3 && Derived::support_remove_by_node;
  static constexpr bool support_split = Derived::support_join3;
  static constexpr bool support_insert_at =
      !Base::has_key && SubtreeDataType::has_size && Derived::support_join3 &&
      Derived::support_split;

 public:
  /**
   * @brief Constructs a tree with the specified maximum number of nodes.
   *
   * @param max_nodes The maximum number of nodes to reserve
   */
  explicit constexpr ExtendedTree(size_t max_nodes) : Base(max_nodes) {}

  /**
   * @brief Default constructor.
   *
   * Creates a tree with no initial capacity.
   */
  constexpr ExtendedTree() : Base() {}

  /**
   * @brief Finds a node with the given key in the tree.
   *
   * @param root The root of the tree to search in
   * @param key The key to search for
   * @return Pointer to the node with the given key, or nullptr if not found
   */
  [[nodiscard]] static constexpr NodeType* find(NodeType* root,
                                                const KeyType& key) {
    return bst::base::find(root, key);
  }

  /**
   * @brief Gets a node at the specified inorder index in the tree.
   *
   * @param root The root of the tree to search in
   * @param index The index of the node to retrieve
   * @return Pointer to the node at the given index, or nullptr if not found
   */
  [[nodiscard]] static constexpr NodeType* at(NodeType* root, size_t index) {
    return bst::base::at(root, index);
  }

 protected:
  /**
   * @brief Implementation of remove that can be overridden by derived classes.
   *
   * @param root The root of the tree
   * @param key The key of the node to remove
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  static NodeType* remove_impl(NodeType* root, const KeyType& key,
                               NodeType*& removed_node) {
    removed_node = Derived::find(root, key);
    return (removed_node
                ? Derived::template remove_node_impl<reset_links>(removed_node)
                : root);
  }

  /**
   * @brief Implementation of remove_at that can be overridden by derived
   * classes.
   *
   * @param root The root of the tree
   * @param index The zero-based index of the node to remove
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  static NodeType* remove_at_impl(NodeType* root, size_t index,
                                  NodeType*& removed_node) {
    removed_node = Derived::at(root, index);
    return (removed_node
                ? Derived::template remove_node_impl<reset_links>(removed_node)
                : root);
  }

  /**
   * @brief Implementation of join that can be overridden by derived classes.
   *
   * @param l The left tree
   * @param r The right tree
   * @return Pointer to the new root of the joined tree
   */
  static NodeType* join_impl(NodeType* l, NodeType* r) {
    static_assert(Derived::support_join, "Join should be supported");
    static_assert(Derived::support_remove_by_node,
                  "Remove by node should be supported");
    if (!l) return r;
    if (!r) return l;
    auto node = l;
    for (; node->right; node = node->right) node->apply_deferred();
    node->apply_deferred();
    l = Derived::remove_node_impl(node);
    return Derived::join3(l, node, r);
  }

  /**
   * @brief Implementation of join3 that can be overridden by derived classes.
   *
   * This base implementation always uses m1 as the root of the joined tree,
   * connecting l as the left subtree and r as the right subtree. Derived
   * classes may override this to implement different balancing strategies.
   *
   * @param l The left tree
   * @param m1 The middle node that will become the root
   * @param r The right tree
   * @return Pointer to the new root of the joined tree (always m1 in this
   * implementation)
   */
  static NodeType* join3_base_impl(NodeType* l, NodeType* m1, NodeType* r) {
    m1->set_left(l);
    m1->set_right(r);
    m1->update_subtree_data();
    return m1;
  }

  /**
   * @brief Implementation of split that can be overridden by derived classes.
   *
   * @param root The root of the tree to split
   * @param key The key to split at
   * @param output_l Reference to store the left part of the split
   * @param output_r Reference to store the right part of the split
   */
  static void split_impl(NodeType* root, const KeyType& key,
                         NodeType*& output_l, NodeType*& output_r) {
    static_assert(Base::has_key, "has_key should be true");
    static_assert(Derived::support_join3, "Join3 should be supported");
    static_assert(Derived::support_split, "Split should be supported");
    if (!root) {
      output_l = output_r = nullptr;
      return;
    }
    root->apply_deferred();
    NodeType *l = root->left, *r = root->right, *m = nullptr;
    root->set_left(nullptr);
    root->set_right(nullptr);
    if (l) l->set_parent(nullptr);
    if (r) r->set_parent(nullptr);
    if (root->key < key) {
      split_impl(r, key, m, output_r);
      output_l = Derived::join3(l, root, m);
    } else {
      split_impl(l, key, output_l, m);
      output_r = Derived::join3(m, root, r);
    }
  }
};

}  // namespace base
}  // namespace bst
