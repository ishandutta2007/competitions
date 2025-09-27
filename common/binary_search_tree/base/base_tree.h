#pragma once

#include "common/base.h"
#include "common/binary_search_tree/subtree_data/size.h"

#include <algorithm>
#include <vector>

namespace bst {
namespace base {

/**
 * @brief Base class for binary search tree implementations.
 *
 * This class provides core functionality and interface for binary search trees:
 * - Type definitions and feature flags for different BST implementations
 * - Node management (creation, release, and memory management)
 * - Base tree operations with comprehensive corner cases handling
 * - Tree building and manipulation operations
 *
 * The class handles all common corner cases and requirements checking:
 * - Verifies tree properties (root validity, key/size support)
 * - Handles null cases for all operations
 * - Manages node isolation and deferred computations
 * - Provides safe node creation and release
 *
 * Derived classes are responsible for:
 * - Implementing specific tree properties (e.g., splay, treap)
 * - Maintaining tree invariants during operations
 * - Updating parent links and subtree data
 * - Handling node connections and disconnections
 *
 * @tparam NodesManager The node manager type that handles node allocation and
 *         management
 * @tparam Derived The derived class type (for CRTP)
 */
template <typename NodesManager, typename Derived>
class BaseTree {
 public:
  /**
   * @brief Type aliases for better readability and type safety.
   */
  using NodesManagerType = NodesManager;
  using NodeType = typename NodesManagerType::NodeType;
  using DataType = typename NodeType::DataType;
  using KeyType = typename NodeType::KeyType;
  using SubtreeDataType = typename NodeType::SubtreeDataType;
  using DeferredType = typename NodeType::DeferredType;

  /**
   * @brief Node feature flags inherited from NodeType.
   */
  static constexpr bool has_key = NodeType::has_key;
  static constexpr bool has_parent = NodeType::has_parent;
  static constexpr bool has_size = SubtreeDataType::has_size;

  static constexpr bool is_persistent = false;

  /**
   * @brief Operation support flags.
   *
   * These flags determine which operations are supported by the tree:
   * - support_insert: Whether insertion operations are supported
   * - support_remove: Whether removal operations are supported
   * - support_remove_by_node: Whether node-based removal is supported
   * - support_join: Whether two-way join operations are supported
   * - support_split: Whether split operations are supported
   * - support_reverse_subtree: Whether subtree reversal is supported
   */
  static constexpr bool support_insert = true;
  static constexpr bool support_remove = true;
  static constexpr bool support_remove_by_node =
      Derived::support_remove && has_parent;
  static constexpr bool support_join = true;
  static constexpr bool support_split = true;
  static constexpr bool support_reverse_subtree = true;

 public:
  /**
   * @brief Constructs a tree with the specified maximum number of nodes.
   *
   * @param max_nodes The maximum number of nodes to reserve
   */
  [[nodiscard]] explicit constexpr BaseTree(size_t max_nodes)
      : nodes_manager_(max_nodes) {}

  /**
   * @brief Returns a pointer to the derived class instance.
   *
   * This helper is used to access derived class methods through CRTP.
   *
   * @return Pointer to the derived class instance
   */
  [[nodiscard]] constexpr Derived* derived() {
    return static_cast<Derived*>(this);
  }

  [[nodiscard]] constexpr const Derived* derived() const {
    return static_cast<const Derived*>(this);
  }

  /**
   * @brief Creates a new node with the given data.
   *
   * @param data The data to store in the new node
   * @return Pointer to the newly created node
   */
  constexpr NodeType* create_node(const DataType& data) {
    return create_node_impl<false>(data);
  }

  /**
   * @brief Creates a new node with the given data and key.
   *
   * @param data The data to store in the new node
   * @param key The key value for the node
   * @return Pointer to the newly created node
   */
  constexpr NodeType* create_node(const DataType& data, const KeyType& key) {
    return create_node_impl<false>(data, key);
  }

  /**
   * @brief Builds a tree from a vector of nodes.
   *
   * This function constructs a balanced binary search tree from a vector of
   * nodes. The nodes should be sorted by key if the tree uses keys.
   *
   * @param nodes Vector of nodes to build the tree from
   * @return Pointer to the root of the built tree
   */
  static constexpr NodeType* build_tree(const std::vector<NodeType*>& nodes) {
    return Derived::template build_tree_impl<false>(nodes);
  }

  /**
   * @brief Builds a tree from a vector of data elements.
   *
   * @param data Vector of data elements to build the tree from
   * @return Pointer to the root of the built tree
   */
  [[nodiscard]] constexpr NodeType* build(const std::vector<DataType>& data) {
    if (data.empty()) return nullptr;
    nodes_manager_.reserve_additional(data.size());
    std::vector<NodeType*> nodes(data.size());
    for (size_t i = 0; i < data.size(); ++i)
      nodes[i] = create_node_impl<true>(data[i]);
    return Derived::template build_tree_impl<true>(nodes);
  }

  /**
   * @brief Builds a tree from vectors of data and keys.
   *
   * @param data Vector of data elements to build the tree from
   * @param keys Vector of keys corresponding to the data elements
   * @return Pointer to the root of the built tree
   */
  [[nodiscard]] constexpr NodeType* build(const std::vector<DataType>& data,
                                          const std::vector<KeyType>& keys) {
    static_assert(has_key, "has_key should be true");
    assert(data.size() == keys.size());
    if (data.empty()) return nullptr;
    nodes_manager_.reserve_additional(data.size());
    std::vector<NodeType*> nodes(data.size());
    for (size_t i = 0; i < data.size(); ++i)
      nodes[i] = create_node_impl<true>(data[i], keys[i]);
    std::sort(
        nodes.begin(), nodes.end(),
        [](const NodeType* a, const NodeType* b) { return a->key < b->key; });
    return Derived::template build_tree_impl<true>(nodes);
  }

  /**
   * @brief Inserts a node into the tree.
   *
   * This function inserts a node into the tree while maintaining the binary
   * search tree property. The node must be isolated (no connections to other
   * nodes) and have no pending deferred computations.
   *
   * @param root The root of the tree
   * @param node The node to insert (must be isolated and have no pending
   * deferred computations)
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* insert(NodeType* root,
                                                  NodeType* node) {
    static_assert(has_key, "has_key should be true");
    assert(node && node->is_isolated() && !node->apply_required());
    return Derived::template insert_impl<false>(root, node);
  }

  /**
   * @brief Inserts a node at the specified inorder index.
   *
   * This function inserts a node at the given position in the inorder traversal
   * of the tree. The node must be isolated (no connections to other nodes) and
   * have no pending deferred computations.
   *
   * @param root The root of the tree
   * @param node The node to insert (must be isolated and have no pending
   * deferred computations)
   * @param index The zero-based index where to insert
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* insert_at(NodeType* root,
                                                     NodeType* node,
                                                     size_t index) {
    static_assert(has_size, "subtree data should contain size");
    assert(node && node->is_isolated() && !node->apply_required());
    assert(index <= bst::subtree_data::size(root));
    return Derived::template insert_at_impl<false>(root, node, index);
  }

  /**
   * @brief Creates and inserts a new node with the given data and key.
   *
   * This function creates a new node with the specified data and key, and
   * inserts it into the tree while maintaining the binary search tree property.
   * The node is created with deferred update enabled.
   *
   * @param root The root of the tree
   * @param data The data for the new node
   * @param key The key for the new node
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] constexpr NodeType* insert_new(NodeType* root,
                                               const DataType& data,
                                               const KeyType& key) {
    return Derived::template insert_impl<true>(
        root, create_node_impl<true>(data, key));
  }

  /**
   * @brief Creates and inserts a new node at the specified inorder index.
   *
   * This function creates a new node with the specified data and inserts it at
   * the given position in the inorder traversal of the tree. The node is
   * created with deferred update enabled.
   *
   * @param root The root of the tree
   * @param data The data for the new node
   * @param index The zero-based index where to insert
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] constexpr NodeType* insert_new_at(NodeType* root,
                                                  const DataType& data,
                                                  size_t index) {
    static_assert(has_size, "subtree data should contain size");
    assert(index <= bst::subtree_data::size(root));
    return Derived::template insert_at_impl<true>(
        root, create_node_impl<true>(data), index);
  }

  /**
   * @brief Removes a node with the given key from the tree.
   *
   * This function verifies that the tree supports key-based operations and
   * delegates the removal to the derived class implementation. The derived
   * class is responsible for maintaining the tree properties during removal.
   *
   * @param root The root of the tree
   * @param key The key of the node to remove
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* remove(NodeType* root,
                                                  const KeyType& key,
                                                  NodeType*& removed_node) {
    static_assert(has_key, "has_key should be true");
    return Derived::template remove_impl<true>(root, key, removed_node);
  }

  /**
   * @brief Removes a node at the given index from the tree.
   *
   * This function verifies that the tree supports size-based operations and
   * that the index is valid. It delegates the removal to the derived class
   * implementation. The derived class is responsible for maintaining the tree
   * properties during removal.
   *
   * @param root The root of the tree
   * @param index The position of the node to remove
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* remove_at(NodeType* root,
                                                     size_t index,
                                                     NodeType*& removed_node) {
    static_assert(has_size, "has_size should be true");
    assert(index < bst::subtree_data::size(root));
    return Derived::template remove_at_impl<true>(root, index, removed_node);
  }

  /**
   * @brief Removes a specific node from the tree.
   *
   * This function verifies that the tree supports parent pointers and that
   * the node exists. It delegates the removal to the derived class
   * implementation. The derived class is responsible for maintaining the tree
   * properties during removal.
   *
   * @param node The node to remove
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* remove_node(NodeType* node) {
    static_assert(has_parent, "has_parent should be true");
    assert(node);
    return Derived::template remove_node_impl<true>(node);
  }

  /**
   * @brief Releases a node back to the node manager.
   *
   * @param node The node to release
   */
  constexpr void release(NodeType* node) {
    assert(node);
    nodes_manager_.release(node);
  }

  /**
   * @brief Recursively releases all nodes in a tree.
   *
   * @param root The root of the tree to release
   */
  constexpr void release_tree(NodeType* root) {
    if (root) {
      release_tree(root->left);
      release_tree(root->right);
      release(root);
    }
  }

  /**
   * @brief Removes a node with the given key and releases it.
   *
   * @param root The root of the tree
   * @param key The key of the node to remove
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] constexpr NodeType* remove_and_release(NodeType* root,
                                                       const KeyType& key) {
    static_assert(has_key, "has_key should be true");
    NodeType* removed_node = nullptr;
    NodeType* new_root =
        Derived::template remove_impl<false>(root, key, removed_node);
    if (removed_node) release(removed_node);
    return new_root;
  }

  /**
   * @brief Removes a node at the specified inorder index and releases it.
   *
   * @param root The root of the tree
   * @param index The zero-based index of the node to remove
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] constexpr NodeType* remove_and_release_at(NodeType* root,
                                                          size_t index) {
    static_assert(has_size, "has_size should be true");
    assert(index < bst::subtree_data::size(root));
    NodeType* removed_node = nullptr;
    NodeType* new_root =
        Derived::template remove_at_impl<false>(root, index, removed_node);
    if (removed_node) release(removed_node);
    return new_root;
  }

  /**
   * @brief Removes a node and releases it back to the node manager.
   *
   * @param node The node to remove and release
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] constexpr NodeType* remove_and_release_node(NodeType* node) {
    static_assert(has_parent, "has_parent should be true");
    assert(node);
    NodeType* new_root = Derived::template remove_node_impl<false>(node);
    release(node);
    return new_root;
  }

  /**
   * @brief Joins two trees together.
   *
   * This function verifies that both trees are valid (their roots are actual
   * roots) and delegates the join operation to the derived class
   * implementation. The derived class is responsible for maintaining the tree
   * properties during the join operation.
   *
   * @param l The root of the left tree
   * @param r The root of the right tree
   * @return Pointer to the root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join(NodeType* l, NodeType* r) {
    if (!l) return r;
    if (!r) return l;
    assert(l->is_root() && r->is_root());
    return Derived::join_impl(l, r);
  }

  /**
   * @brief Joins three trees together.
   *
   * This function verifies that the left and right trees are valid (their roots
   * are actual roots) and that the middle tree is isolated. It delegates the
   * join operation to the derived class implementation. The derived class is
   * responsible for maintaining the tree properties during the join operation.
   *
   * @param l The root of the left tree
   * @param m1 The root of the middle tree (must be isolated node)
   * @param r The root of the right tree
   * @return Pointer to the root of the joined tree
   */
  static constexpr NodeType* join3(NodeType* l, NodeType* m1, NodeType* r) {
    assert((!l || l->is_root()) && (!r || r->is_root()));
    assert(m1 && m1->is_isolated());
    return Derived::join3_impl(l, m1, r);
  }

  /**
   * @brief Splits a tree at a given key.
   *
   * This function verifies that the tree supports key-based operations and
   * that the root is valid (either nullptr or a root of its tree). It handles
   * all corner cases:
   * - If root is nullptr, both output parts are set to nullptr
   * - If root is not nullptr, delegates to the derived class implementation
   *
   * The derived class implementation is responsible for:
   * - Maintaining the binary search tree property
   * - Updating parent links and subtree data
   * - Ensuring that output_l contains all nodes with keys < key
   * - Ensuring that output_r contains all nodes with keys >= key
   *
   * @param root The root of the tree to split
   * @param key The key to split at
   * @param output_l Reference to store the left part
   * @param output_r Reference to store the right part
   */
  static constexpr void split(NodeType* root, const KeyType& key,
                              NodeType*& output_l, NodeType*& output_r) {
    static_assert(has_key, "has_key should be true");
    assert(!root || root->is_root());
    if (!root) {
      output_l = output_r = nullptr;
    } else {
      Derived::split_impl(root, key, output_l, output_r);
    }
  }

  /**
   * @brief Splits a tree at a given inorder index.
   *
   * This function verifies that the tree supports size-based operations and
   * that the root is valid (either nullptr or a root of its tree). It handles
   * all corner cases:
   * - If root is nullptr, both output parts are set to nullptr
   * - If lsize is 0, output_l is nullptr and output_r is root
   * - If lsize >= tree size, output_l is root and output_r is nullptr
   * - Otherwise, delegates to the derived class implementation
   *
   * The derived class implementation is responsible for:
   * - Maintaining the binary search tree property
   * - Updating parent links and subtree data
   * - Ensuring that output_l contains exactly lsize nodes
   * - Ensuring that output_r contains the remaining nodes
   *
   * @param root The root of the tree to split
   * @param lsize The size of the left part
   * @param output_l Reference to store the left part
   * @param output_r Reference to store the right part
   */
  static constexpr void split_at(NodeType* root, size_t lsize,
                                 NodeType*& output_l, NodeType*& output_r) {
    static_assert(has_size, "has_size should be true");
    assert(!root || root->is_root());
    assert(lsize <= bst::subtree_data::size(root));
    if (!root) {
      output_l = output_r = nullptr;
    } else if (lsize == 0) {
      output_l = nullptr;
      output_r = root;
    } else if (lsize >= bst::subtree_data::size(root)) {
      output_l = root;
      output_r = nullptr;
    } else {
      Derived::split_at_impl(root, lsize, output_l, output_r);
    }
  }

 public:
  /**
   * @brief Clears all nodes from the node manager.
   */
  constexpr void clear() { nodes_manager_.clear(); }

  /**
   * @brief Initializes the node manager with a new capacity.
   *
   * @param max_nodes The maximum number of nodes to reserve
   */
  constexpr void init(size_t max_nodes) { nodes_manager_.init(max_nodes); }

  /**
   * @brief Returns the number of nodes currently in use.
   *
   * @return The number of used nodes
   */
  [[nodiscard]] constexpr size_t used() const noexcept {
    return nodes_manager_.used();
  }

  /**
   * @brief Gets a node at a specific index from the node manager.
   *
   * @param index The index of the node to retrieve
   * @return Pointer to the node at the given index
   */
  [[nodiscard]] constexpr NodeType* manager_at(size_t index) {
    static_assert(NodesManagerType::support_at,
                  "nodes manager should support at");
    return nodes_manager_.at(index);
  }

  /**
   * @brief Gets a const node at a specific index from the node manager.
   *
   * @param index The index of the node to retrieve
   * @return Const pointer to the node at the given index
   */
  [[nodiscard]] constexpr const NodeType* manager_at(size_t index) const {
    static_assert(NodesManagerType::support_at,
                  "nodes manager should support at");
    return nodes_manager_.at(index);
  }

  /**
   * @brief Gets the index of a node in the node manager.
   *
   * @param node The node to find the index for
   * @return The index of the node
   */
  [[nodiscard]] constexpr size_t manager_index(const NodeType* node) const {
    static_assert(NodesManagerType::support_index,
                  "nodes manager should support index");
    return nodes_manager_.index(node);
  }

 protected:
  /**
   * @brief Creates a new node with the given data.
   *
   * @tparam skip_update Whether to skip updating subtree data
   * @param data The data to store in the new node
   * @return Pointer to the newly created node
   */
  template <bool skip_update>
  [[nodiscard]] constexpr NodeType* create_node_impl(const DataType& data) {
    auto p = nodes_manager_.create();
    p->data = data;
    p->subtree_data.bti_reset();
    if constexpr (!skip_update) p->update_subtree_data();
    return p;
  }

  /**
   * @brief Creates a new node with the given data and key.
   *
   * @tparam skip_update Whether to skip updating subtree data
   * @param data The data to store in the new node
   * @param key The key value for the node
   * @return Pointer to the newly created node
   */
  template <bool skip_update>
  [[nodiscard]] constexpr NodeType* create_node_impl(const DataType& data,
                                                     const KeyType& key) {
    static_assert(has_key, "has_key should be true");
    auto p = nodes_manager_.create();
    p->data = data;
    p->key = key;
    p->subtree_data.bti_reset();
    if constexpr (!skip_update) p->update_subtree_data();
    return p;
  }

  /**
   * @brief Builds a balanced binary search tree from a range of nodes.
   *
   * @tparam update_leafs Whether to update subtree data for leaf nodes
   * @param nodes Vector of nodes to build the tree from
   * @param begin Start index of the range
   * @param end End index of the range (exclusive)
   * @return Pointer to the root of the built subtree
   */
  template <bool update_leafs>
  [[nodiscard]] static constexpr NodeType* build_tree_base_impl(
      const std::vector<NodeType*>& nodes, size_t begin, size_t end) {
    if (begin >= end) return nullptr;
    const size_t m = (begin + end) / 2;
    NodeType* root = nodes[m];
    root->set_left(build_tree_base_impl<update_leafs>(nodes, begin, m));
    root->set_right(build_tree_base_impl<update_leafs>(nodes, m + 1, end));
    if (update_leafs || (end - begin > 1)) root->update_subtree_data();
    return root;
  }

  /**
   * @brief Builds a balanced binary search tree from a vector of nodes.
   *
   * @tparam update_leafs Whether to update subtree data for leaf nodes
   * @param nodes Vector of nodes to build the tree from
   * @return Pointer to the root of the built tree
   */
  template <bool update_leafs>
  [[nodiscard]] static constexpr NodeType* build_tree_impl(
      const std::vector<NodeType*>& nodes) {
    NodeType* root = Derived::template build_tree_base_impl<update_leafs>(
        nodes, 0, nodes.size());
    if (root) root->set_parent(nullptr);
    return root;
  }

 protected:
  /**
   * @brief Node manager instance that handles node allocation and management.
   *
   * This member is protected to allow derived classes to access the node
   * manager for their specific implementations while keeping it hidden from
   * external users.
   */
  NodesManagerType nodes_manager_;
};

}  // namespace base
}  // namespace bst
