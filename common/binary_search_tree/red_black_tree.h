#pragma once

#include "common/base.h"
#include "common/binary_search_tree/base/at.h"
#include "common/binary_search_tree/base/at_with_path.h"
#include "common/binary_search_tree/base/base_tree.h"
#include "common/binary_search_tree/base/deferred.h"
#include "common/binary_search_tree/base/find.h"
#include "common/binary_search_tree/base/find_with_path.h"
#include "common/binary_search_tree/base/node.h"
#include "common/binary_search_tree/base/push_down.h"
#include "common/binary_search_tree/base/push_down_with_path.h"
#include "common/binary_search_tree/base/rotate.h"
#include "common/binary_search_tree/base/sibling.h"
#include "common/binary_search_tree/base/subtree_data.h"
#include "common/binary_search_tree/subtree_data/balance/rbt_color.h"
#include "common/binary_search_tree/subtree_data/size.h"
#include "common/binary_search_tree/subtree_data/utils/propagate_and_find_root.h"
#include "common/binary_search_tree/subtree_data/utils/propagate_to_root_with_path.h"
#include "common/memory/contiguous_nodes_manager.h"
#include "common/templates/tuple.h"

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

namespace bst {

/**
 * @brief Red-Black tree implementation with automatic color balancing.
 *
 * This tree implements a self-balancing binary search tree where each node
 * has an extra bit for denoting the color of the node, either red or black.
 * The tree maintains balance through a set of properties that ensure the
 * tree remains approximately balanced during insertions and deletions.
 *
 * Red-Black tree properties:
 * 1. Every node is either red or black
 * 2. The root is always black
 * 3. All leaves (nullptr) are considered black
 * 4. Red nodes cannot have red children (no two consecutive red nodes)
 * 5. Every path from root to leaves has the same number of black nodes
 *
 * Key features:
 * - Automatic color balancing after insertions and removals
 * - Guaranteed O(log n) time complexity for all operations
 * - Efficient join and split operations with balance maintenance
 * - Support for parent pointers, custom data, aggregators, and deferred
 *   operations
 * - Path-based operations for trees without parent links
 *
 * The tree uses color information stored in each node to maintain balance
 * through rotations and color changes. It automatically adds color tracking
 * to the subtree data if not already present.
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
class RedBlackTree
    : public base::BaseTree<
          NodesManager<base::Node<
              Data,
              base::SubtreeData<templates::PrependT<subtree_data::RBTColor,
                                                    AggregatorsTuple>>,
              base::Deferred<DeferredTuple>, has_parent, has_key, Key>>,
          RedBlackTree<has_key, has_parent, Data, AggregatorsTuple,
                       DeferredTuple, Key, NodesManager>> {
 public:
  using SubtreeDataType = base::SubtreeData<
      templates::PrependT<subtree_data::RBTColor, AggregatorsTuple>>;
  using DeferredType = base::Deferred<DeferredTuple>;
  using NodeType =
      base::Node<Data, SubtreeDataType, DeferredType, has_parent, has_key, Key>;
  using Self = RedBlackTree<has_key, has_parent, Data, AggregatorsTuple,
                            DeferredTuple, Key, NodesManager>;
  using Base = base::BaseTree<NodesManager<NodeType>, Self>;

  friend Base;

 public:
  /**
   * @brief Constructs a Red-Black tree with the specified maximum number of
   * nodes.
   *
   * @param max_nodes The maximum number of nodes to reserve
   */
  explicit RedBlackTree(size_t max_nodes) : Base(max_nodes) {}

  /**
   * @brief Removes the rightmost node from the tree.
   *
   * This function removes the rightmost node from the tree while maintaining
   * the Red-Black tree properties. It's commonly used in join operations to
   * extract a node for combining two trees.
   *
   * @param root The root of the tree
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* remove_right(
      NodeType* root, NodeType*& removed_node) {
    assert(root);
    return remove_right_impl<true>(root, removed_node);
  }

 protected:
  /**
   * @brief Checks if a node is black.
   *
   * A node is considered black if it is nullptr or if its color bit is set
   * to black. This follows the Red-Black tree property that all leaves
   * (nullptr) are considered black.
   *
   * @param node The node to check
   * @return true if the node is black, false otherwise
   */
  [[nodiscard]] static constexpr bool is_black(const NodeType* node) {
    return !node || subtree_data::RBTColor::get(node);
  }

  /**
   * @brief Checks if a node is red.
   *
   * A node is red if it is not black. This is a convenience function
   * that simply negates the result of is_black().
   *
   * @param node The node to check
   * @return true if the node is red, false otherwise
   */
  [[nodiscard]] static constexpr bool is_red(const NodeType* node) {
    return !is_black(node);
  }

  /**
   * @brief Calculates the black height of a subtree.
   *
   * The black height is the number of black nodes on any path from the root
   * to a leaf (nullptr). In a valid Red-Black tree, all paths from root to
   * leaves must have the same black height. This function calculates the
   * black height by traversing the leftmost path and counting black nodes.
   *
   * @param root The root of the subtree to calculate black height for
   * @return The black height of the subtree
   */
  [[nodiscard]] static constexpr int black_height(NodeType* root) {
    int h = 0;
    for (; root; root = root->left) {
      root->apply_deferred();
      if (is_black(root)) ++h;
    }
    return h;
  }

  /**
   * @brief Sets the color of a node.
   *
   * @param node The node to set the color for
   * @param black true to set the node to black, false to set it to red
   */
  static constexpr void set_color(NodeType* node, bool black) {
    assert(node);
    subtree_data::RBTColor::set(node, black);
  }

  /**
   * @brief Sets a node to black.
   *
   * @param node The node to set to black
   */
  static constexpr void set_black(NodeType* node) { set_color(node, true); }

  /**
   * @brief Sets a node to red.
   *
   * @param node The node to set to red
   */
  static constexpr void set_red(NodeType* node) { set_color(node, false); }

  /**
   * @brief Recursively fixes colors during tree construction.
   *
   * This function is used during tree construction to ensure that the
   * Red-Black tree properties are maintained. It sets colors based on
   * the height of each node in the tree:
   * - Nodes at non-zero height are set to black
   * - Leaf nodes (height 0) are set to red
   *
   * @param root The root of the subtree to fix colors for
   * @param height The height of the current node in the tree
   */
  static constexpr void build_tree_impl_fix_colors(NodeType* root,
                                                   size_t height) {
    assert(root || !height);
    if (!root) return;
    if (height) {
      set_black(root);
      build_tree_impl_fix_colors(root->left, height - 1);
      build_tree_impl_fix_colors(root->right, height - 1);
    } else {
      set_red(root);
      assert(!root->left && !root->right);
    }
  }

  /**
   * @brief Builds a Red-Black tree from a vector of nodes.
   *
   * This function constructs a balanced Red-Black tree from a vector of nodes.
   * It first builds a basic binary search tree using the base implementation,
   * then fixes the colors to maintain Red-Black tree properties.
   *
   * The color fixing process:
   * 1. Calculates the height of the tree based on the number of nodes
   * 2. Recursively sets colors based on node height
   * 3. Ensures the root is always black
   *
   * @tparam update_leafs Whether to update subtree data for leaf nodes
   * @param nodes Vector of nodes to build the tree from
   * @return Pointer to the root of the built Red-Black tree
   */
  template <bool update_leafs>
  [[nodiscard]] static constexpr NodeType* build_tree_impl(
      const std::vector<NodeType*>& nodes) {
    if (nodes.size() == 0) return nullptr;
    size_t h = 0;
    for (; nodes.size() >= (1ull << h);) ++h;
    NodeType* root = Base::template build_tree_impl<update_leafs>(nodes);
    build_tree_impl_fix_colors(root, h - 1);
    set_black(root);
    return root;
  }

  /**
   * @brief Internal implementation of insertion using path-based approach.
   *
   * This function handles the Red-Black tree insertion balancing when parent
   * links are not available. It uses a pre-computed path from root to the
   * inserted node to perform the necessary rotations and color changes.
   *
   * The balancing process:
   * 1. Traverses the path from the inserted node towards the root
   * 2. Handles cases where the parent is black (no balancing needed)
   * 3. Performs rotations and color changes when the parent is red
   * 4. Maintains Red-Black tree properties throughout the process
   *
   * @param nodes Vector containing the path from root to the inserted node
   * @param update_required Whether subtree data updates are required
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* insert_impl_internal_hpf(
      const std::vector<NodeType*>& nodes, bool update_required) {
    unsigned index = 0;
    while (index + 1 < nodes.size()) {
      NodeType* node = nodes[index];
      NodeType* parent = nodes[index + 1];
      if (is_black(parent)) {
        subtree_data::propagate_to_root_with_path(
            nodes, update_required ? index : index + 1);
        return nodes.back();
      }
      NodeType* grandparent = nodes[index + 2];
      NodeType* uncle = base::sibling(parent, grandparent);
      if (is_black(uncle)) {
        const bool rotate_required =
            ((grandparent->left == parent) != (parent->left == node));
        if (rotate_required) {
          base::rotate<false, false>(node, parent, grandparent);
          parent = node;
        } else if (update_required) {
          node->update_subtree_data();
        }
        base::rotate<true, false>(
            parent, grandparent,
            ((index + 3 < nodes.size()) ? nodes[index + 3] : nullptr));
        set_red(grandparent);
        set_black(parent);
        if (index + 3 < nodes.size()) {
          subtree_data::propagate_to_root_with_path(nodes, index + 3);
          return nodes.back();
        } else {
          return parent;
        }
      }
      if (update_required) node->update_subtree_data();
      set_black(parent);
      set_black(uncle);
      set_red(grandparent);
      parent->update_subtree_data();
      update_required = true;
      index += 2;
    }
    assert(index + 1 == nodes.size());
    NodeType* node = nodes.back();
    set_black(node);
    if (update_required) node->update_subtree_data();
    return node;
  }

  /**
   * @brief Internal implementation of insertion using parent-based approach.
   *
   * This function handles the Red-Black tree insertion balancing when parent
   * links are available. It traverses up the tree from the inserted node
   * using parent pointers to perform the necessary rotations and color changes.
   *
   * The balancing process:
   * 1. Traverses up the tree from the inserted node towards the root
   * 2. Handles cases where the parent is black (no balancing needed)
   * 3. Performs rotations and color changes when the parent is red
   * 4. Maintains Red-Black tree properties throughout the process
   *
   * @param node The inserted node to start balancing from
   * @param update_required Whether subtree data updates are required
   * @return Pointer to the new root of the tree
   */
  [[nodiscard]] static constexpr NodeType* insert_impl_internal_hpt(
      NodeType* node, bool update_required) {
    static_assert(has_parent, "has_parent should be true");
    while (NodeType* parent = node->parent) {
      if (is_black(parent))
        return subtree_data::propagate_and_find_root(update_required ? node
                                                                     : parent);
      NodeType* grandparent = parent->parent;
      NodeType* uncle = base::sibling(parent, grandparent);
      if (is_black(uncle)) {
        const bool rotate_required =
            ((grandparent->left == parent) != (parent->left == node));
        if (rotate_required) {
          base::rotate<false, false>(node, parent, grandparent);
          parent = node;
        } else if (update_required) {
          node->update_subtree_data();
        }
        base::rotate<true, false>(parent, grandparent, grandparent->parent);
        set_red(grandparent);
        set_black(parent);
        return parent->parent
                   ? subtree_data::propagate_and_find_root(parent->parent)
                   : parent;
      }
      if (update_required) node->update_subtree_data();
      set_black(parent);
      set_black(uncle);
      set_red(grandparent);
      parent->update_subtree_data();
      update_required = true;
      node = grandparent;
    }
    set_black(node);
    if (update_required) node->update_subtree_data();
    return node;
  }

  /**
   * @brief Inserts a node into the Red-Black tree.
   *
   * This function inserts a node into the tree while maintaining the Red-Black
   * tree properties. It handles both cases where parent links are available
   * and where they are not.
   *
   * @tparam update_required Whether to update subtree data for the new node
   * @param root The root of the tree
   * @param node The node to insert
   * @return Pointer to the new root of the tree
   */
  template <bool update_required>
  [[nodiscard]] static constexpr NodeType* insert_impl(NodeType* root,
                                                       NodeType* node)
    requires(has_parent)
  {
    if (!root) {
      set_black(node);
      if constexpr (update_required) node->update_subtree_data();
      return node;
    }

    while (true) {
      root->apply_deferred();
      if (root->key < node->key) {
        if (!root->right) {
          root->set_right(node);
          break;
        }
        root = root->right;
      } else {
        if (!root->left) {
          root->set_left(node);
          break;
        }
        root = root->left;
      }
    }
    set_red(node);
    return insert_impl_internal_hpt(node, update_required);
  }

  template <bool update_required>
  [[nodiscard]] static NodeType* insert_impl(NodeType* root, NodeType* node)
    requires(!has_parent)
  {
    if (!root) {
      set_black(node);
      if constexpr (update_required) node->update_subtree_data();
      return node;
    }

    thread_local std::vector<NodeType*> nodes;
    nodes.clear();
    while (true) {
      root->apply_deferred();
      nodes.push_back(root);
      if (root->key < node->key) {
        if (!root->right) {
          root->set_right(node);
          break;
        }
        root = root->right;
      } else {
        if (!root->left) {
          root->set_left(node);
          break;
        }
        root = root->left;
      }
    }
    set_red(node);
    nodes.push_back(node);
    std::reverse(nodes.begin(), nodes.end());
    return insert_impl_internal_hpf(nodes, update_required);
  }

  /**
   * @brief Inserts a node at a specific position in the Red-Black tree.
   *
   * This function inserts a node at the given inorder index while maintaining
   * the Red-Black tree properties. It uses subtree size information to
   * navigate to the correct position and then performs the necessary
   * balancing operations.
   *
   * The insertion process:
   * 1. Navigates to the correct position using subtree sizes
   * 2. Sets the new node to red initially
   * 3. Calls the appropriate internal balancing function
   * 4. Ensures the root is always black
   *
   * @tparam update_required Whether to update subtree data for the new node
   * @param root The root of the tree
   * @param node The node to insert
   * @param index The position where to insert the node (0-based)
   * @return Pointer to the new root of the tree
   */
  template <bool update_required>
  [[nodiscard]] static constexpr NodeType* insert_at_impl(NodeType* root,
                                                          NodeType* node,
                                                          size_t index)
    requires(has_parent)
  {
    if (!root) {
      assert(index == 0);
      set_black(node);
      if constexpr (update_required) node->update_subtree_data();
      return node;
    }

    while (true) {
      root->apply_deferred();
      const size_t lsize = bst::subtree_data::size(root->left);
      if (index <= lsize) {
        if (!root->left) {
          assert(index == 0);
          root->set_left(node);
          break;
        }
        root = root->left;
      } else {
        if (!root->right) {
          assert(index == lsize + 1);
          root->set_right(node);
          break;
        }
        root = root->right;
        index -= lsize + 1;
      }
    }
    set_red(node);
    return insert_impl_internal_hpt(node, update_required);
  }

  template <bool update_required>
  [[nodiscard]] static NodeType* insert_at_impl(NodeType* root, NodeType* node,
                                                size_t index)
    requires(!has_parent)
  {
    if (!root) {
      assert(index == 0);
      set_black(node);
      if constexpr (update_required) node->update_subtree_data();
      return node;
    }

    thread_local std::vector<NodeType*> nodes;
    nodes.clear();
    while (true) {
      root->apply_deferred();
      nodes.push_back(root);
      const size_t lsize = bst::subtree_data::size(root->left);
      if (index <= lsize) {
        if (!root->left) {
          assert(index == 0);
          root->set_left(node);
          break;
        }
        root = root->left;
      } else {
        if (!root->right) {
          assert(index == lsize + 1);
          root->set_right(node);
          break;
        }
        root = root->right;
        index -= lsize + 1;
      }
    }
    set_red(node);
    nodes.push_back(node);
    std::reverse(nodes.begin(), nodes.end());
    return insert_impl_internal_hpf(nodes, update_required);
  }

  /**
   * @brief Internal implementation of node removal using path-based approach.
   *
   * This function removes a node from the Red-Black tree while maintaining
   * the tree properties when parent links are not available. It uses a
   * pre-computed path from root to the node to perform the necessary
   * rotations and color changes.
   *
   * The removal process:
   * 1. Pushes down the node if it has two children using path-based approach
   * 2. Removes the node from the tree
   * 3. If the removed node was red, no further balancing is needed
   * 4. If the removed node was black, performs complex color fixing
   * 5. Uses rotations and color changes to restore balance
   * 6. Updates the path vector to reflect structural changes
   *
   * This function is the path-based counterpart to
   * remove_node_impl_internal_hpt and handles the same Red-Black tree balancing
   * cases but without relying on parent pointers.
   *
   * @tparam reset_links Whether to reset links in the removed node
   * @param path_to_node Vector containing the path from root to the node to
   * remove
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static constexpr NodeType* remove_node_impl_internal_hpf(
      std::vector<NodeType*>& path_to_node) {
    base::push_down_with_path(path_to_node);
    std::reverse(path_to_node.begin(), path_to_node.end());
    unsigned index = 1;  // point to parent
    NodeType* node = path_to_node[0];
    const bool is_node_red = is_red(node);

    // Drop node from tree
    NodeType* child = node->left ? node->left : node->right;
    NodeType* parent =
        (index < path_to_node.size()) ? path_to_node[index] : nullptr;
    if (parent) ((parent->left == node) ? parent->left : parent->right) = child;
    if (child) child->set_parent(parent);
    if constexpr (reset_links) node->reset_links_and_update_subtree_data();

    // Fix colors
    if (is_node_red) {
      subtree_data::propagate_to_root_with_path(path_to_node, index);
      return parent ? path_to_node.back() : child;
    }
    while (true) {
      if (is_red(child)) {
        set_black(child);
        subtree_data::propagate_to_root_with_path(path_to_node, index);
        return (parent ? path_to_node.back() : child);
      }
      if (!parent) return child;
      NodeType* sibling = base::sibling(child, parent);
      assert(sibling);
      sibling->apply_deferred();
      if (is_red(sibling)) {
        assert(is_black(parent));
        base::rotate<false, false>(sibling, parent,
                                   (index + 1 < path_to_node.size())
                                       ? path_to_node[index + 1]
                                       : nullptr);
        path_to_node[index] = sibling;
        path_to_node[--index] = parent;
        set_black(sibling);
        set_red(parent);
        sibling = base::sibling(child, parent);
        assert(sibling);
        sibling->apply_deferred();
      }
      assert(is_black(sibling));
      if (is_black(parent) && is_black(sibling->left) &&
          is_black(sibling->right)) {
        set_red(sibling);
        child = parent;
        child->update_subtree_data();
        ++index;
        parent = (index < path_to_node.size()) ? path_to_node[index] : nullptr;
        continue;
      }
      if (is_red(parent) && is_black(sibling->left) &&
          is_black(sibling->right)) {
        set_red(sibling);
        set_black(parent);
        subtree_data::propagate_to_root_with_path(path_to_node, index);
        return path_to_node.back();
      }
      if ((parent->left == child) && is_black(sibling->right)) {
        NodeType* left_child = sibling->left;
        assert(is_red(left_child));
        base::rotate<false, true>(left_child, sibling, parent);
        set_red(sibling);
        sibling = left_child;
        set_black(sibling);
      } else if ((parent->right == child) && is_black(sibling->left)) {
        NodeType* right_child = sibling->right;
        assert(is_red(right_child));
        base::rotate<false, true>(right_child, sibling, parent);
        set_red(sibling);
        sibling = right_child;
        set_black(sibling);
      }
      if (is_red(parent)) {
        set_red(sibling);
        set_black(parent);
      }
      set_black((parent->left == child) ? sibling->right : sibling->left);
      base::rotate<true, false>(sibling, parent,
                                (index + 1 < path_to_node.size())
                                    ? path_to_node[index + 1]
                                    : nullptr);
      subtree_data::propagate_to_root_with_path(path_to_node, index + 1);
      return (index + 1 < path_to_node.size()) ? path_to_node.back() : sibling;
    }
    assert(false);
    return nullptr;
  }

  /**
   * @brief Internal implementation of node removal using parent-based approach.
   *
   * This function removes a node from the Red-Black tree while maintaining
   * the tree properties. It handles the complex color balancing required
   * when removing a black node, which can violate the Red-Black tree
   * properties.
   *
   * The removal process:
   * 1. Pushes down the node if it has two children
   * 2. Removes the node from the tree
   * 3. If the removed node was red, no further balancing is needed
   * 4. If the removed node was black, performs complex color fixing
   * 5. Uses rotations and color changes to restore balance
   *
   * @tparam reset_links Whether to reset links in the removed node
   * @param node The node to remove
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static constexpr NodeType* remove_node_impl_internal_hpt(
      NodeType* node) {
    static_assert(has_parent, "has_parent should be true");

    base::push_down(node);
    const bool is_node_red = is_red(node);

    // Drop node from tree
    NodeType* child = node->left ? node->left : node->right;
    NodeType* parent = node->parent;
    if (parent) ((parent->left == node) ? parent->left : parent->right) = child;
    if (child) child->set_parent(parent);
    if constexpr (reset_links) node->reset_links_and_update_subtree_data();

    // Fix colors
    if (is_node_red)
      return (parent ? subtree_data::propagate_and_find_root(parent) : child);
    while (true) {
      if (is_red(child)) {
        set_black(child);
        return (parent ? subtree_data::propagate_and_find_root(parent) : child);
      }
      if (!parent) return child;
      NodeType* sibling = base::sibling(child, parent);
      assert(sibling);
      sibling->apply_deferred();
      if (is_red(sibling)) {
        assert(is_black(parent));
        base::rotate<false, false>(sibling, parent, parent->parent);
        set_black(sibling);
        set_red(parent);
        sibling = base::sibling(child, parent);
        assert(sibling);
        sibling->apply_deferred();
      }
      assert(is_black(sibling));
      if (is_black(parent) && is_black(sibling->left) &&
          is_black(sibling->right)) {
        set_red(sibling);
        child = parent;
        child->update_subtree_data();
        parent = child->parent;
        continue;
      }
      if (is_red(parent) && is_black(sibling->left) &&
          is_black(sibling->right)) {
        set_red(sibling);
        set_black(parent);
        return subtree_data::propagate_and_find_root(parent);
      }
      if ((parent->left == child) && is_black(sibling->right)) {
        assert(is_red(sibling->left));
        base::rotate<false, true>(sibling->left, sibling, parent);
        set_red(sibling);
        sibling = sibling->parent;
        set_black(sibling);
      } else if ((parent->right == child) && is_black(sibling->left)) {
        assert(is_red(sibling->right));
        base::rotate<false, true>(sibling->right, sibling, parent);
        set_red(sibling);
        sibling = sibling->parent;
        set_black(sibling);
      }
      if (is_red(parent)) {
        set_red(sibling);
        set_black(parent);
      }
      set_black((parent->left == child) ? sibling->right : sibling->left);
      base::rotate<false, false>(sibling, parent, parent->parent);
      return subtree_data::propagate_and_find_root(sibling);
    }
    assert(false);
    return nullptr;
  }

  /**
   * @brief Removes a node with the given key from the Red-Black tree.
   *
   * This function finds and removes a node with the given key while
   * maintaining the Red-Black tree properties. It handles both cases
   * where parent links are available and where they are not.
   *
   * @tparam reset_links Whether to reset links in the removed node
   * @param root The root of the tree
   * @param key The key of the node to remove
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static constexpr NodeType* remove_impl(NodeType* root,
                                                       const Key& key,
                                                       NodeType*& removed_node)
    requires(has_parent)
  {
    removed_node = base::find(root, key);
    return (removed_node
                ? remove_node_impl_internal_hpt<reset_links>(removed_node)
                : root);
  }

  template <bool reset_links>
  [[nodiscard]] static NodeType* remove_impl(NodeType* root, const Key& key,
                                             NodeType*& removed_node)
    requires(!has_parent)
  {
    thread_local std::vector<NodeType*> path_to_node;
    path_to_node.clear();
    removed_node = base::find_with_path(root, key, path_to_node);
    return (removed_node
                ? remove_node_impl_internal_hpf<reset_links>(path_to_node)
                : root);
  }

  /**
   * @brief Removes a node at the specified position from the Red-Black tree.
   *
   * This function removes a node at the given inorder index while
   * maintaining the Red-Black tree properties. It uses subtree size
   * information to locate the node and then performs the removal.
   *
   * @tparam reset_links Whether to reset links in the removed node
   * @param root The root of the tree
   * @param index The position of the node to remove (0-based)
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static constexpr NodeType* remove_at_impl(
      NodeType* root, size_t index, NodeType*& removed_node)
    requires(has_parent)
  {
    removed_node = base::at(root, index);
    return (removed_node
                ? remove_node_impl_internal_hpt<reset_links>(removed_node)
                : root);
  }

  template <bool reset_links>
  [[nodiscard]] static NodeType* remove_at_impl(NodeType* root, size_t index,
                                                NodeType*& removed_node)
    requires(!has_parent)
  {
    thread_local std::vector<NodeType*> path_to_node;
    path_to_node.clear();
    removed_node = base::at_with_path(root, index, path_to_node);
    return (removed_node
                ? remove_node_impl_internal_hpf<reset_links>(path_to_node)
                : root);
  }

  /**
   * @brief Removes a specific node from the Red-Black tree.
   *
   * This function removes the given node while maintaining the Red-Black
   * tree properties. It delegates to the parent-based implementation
   * since this function is only called when parent links are available.
   *
   * @tparam reset_links Whether to reset links in the removed node
   * @param node The node to remove
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static constexpr NodeType* remove_node_impl(NodeType* node) {
    return remove_node_impl_internal_hpt<reset_links>(node);
  }

  /**
   * @brief Removes the rightmost node from the Red-Black tree.
   *
   * This function finds and removes the rightmost node from the tree
   * while maintaining the Red-Black tree properties. It's commonly used
   * in join operations to extract a node for combining two trees.
   *
   * @tparam reset_links Whether to reset links in the removed node
   * @param root The root of the tree
   * @param removed_node Reference to store the removed node
   * @return Pointer to the new root of the tree
   */
  template <bool reset_links>
  [[nodiscard]] static constexpr NodeType* remove_right_impl(
      NodeType* root, NodeType*& removed_node)
    requires(has_parent)
  {
    for (root->apply_deferred(); root->right; root->apply_deferred())
      root = root->right;
    removed_node = root;
    return remove_node_impl_internal_hpt<reset_links>(removed_node);
  }

  template <bool reset_links>
  [[nodiscard]] static NodeType* remove_right_impl(NodeType* root,
                                                   NodeType*& removed_node)
    requires(!has_parent)
  {
    thread_local std::vector<NodeType*> path_to_node;
    path_to_node.clear();
    for (root->apply_deferred(); root->right; root->apply_deferred()) {
      path_to_node.push_back(root);
      root = root->right;
    }
    path_to_node.push_back(root);
    removed_node = root;
    return remove_node_impl_internal_hpf<reset_links>(path_to_node);
  }

  /**
   * @brief Implementation of two-way join operation.
   *
   * This function joins two trees while maintaining balance:
   * - Removes the rightmost node from the left tree
   * - Uses that node to join the trees in a balanced way
   * Code is similar to SelfBalancingTree::join_impl
   *
   * @param l The root of the left tree
   * @param r The root of the right tree
   * @return Pointer to the new root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join_impl(NodeType* l, NodeType* r) {
    if (!l) return r;
    if (!r) return l;
    NodeType* node = nullptr;
    l = remove_right_impl<true>(l, node);
    assert(node);
    return join3_impl(l, node, r);
  }

  /**
   * @brief Base implementation of three-way join operation.
   *
   * This function joins three trees in a balanced way:
   * - The middle node becomes the root
   * - Left and right trees become its children
   * - Updates parent links and subtree data
   * - Sets the middle node to red
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @return Pointer to the new root of the joined tree (m1)
   */
  [[nodiscard]] static constexpr NodeType* join3_base_impl(NodeType* l,
                                                           NodeType* m1,
                                                           NodeType* r) {
    m1->set_left(l);
    m1->set_right(r);
    m1->update_subtree_data();
    set_red(m1);
    return m1;
  }

  /**
   * @brief Joins three trees when the left tree has higher black height.
   *
   * This function handles the case where the left tree is taller than the
   * right tree. It recursively joins the middle node and right tree into
   * the right subtree of the left tree, then performs necessary rotations
   * and color changes to maintain Red-Black tree properties.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param bhd The difference in black heights (left - right)
   * @return Pointer to the new root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_left_impl(NodeType* l,
                                                           NodeType* m1,
                                                           NodeType* r,
                                                           int bhd) {
    if (is_black(l) && (bhd == 0)) return join3_base_impl(l, m1, r);
    l->apply_deferred();
    l->set_right(join3_left_impl(l->right, m1, r, bhd - (is_black(l) ? 1 : 0)));
    r = l->right;
    if (is_black(l) && is_red(r) && is_red(r->right)) {
      // Rotate left
      set_black(r->right);
      l->set_right(r->left);
      l->update_subtree_data();
      r->set_left(l);
      r->update_subtree_data();
      return r;
    } else {
      l->update_subtree_data();
      return l;
    }
  }

  /**
   * @brief Joins three trees when the right tree has higher black height.
   *
   * This function handles the case where the right tree is taller than the
   * left tree. It recursively joins the left tree and middle node into
   * the left subtree of the right tree, then performs necessary rotations
   * and color changes to maintain Red-Black tree properties.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param bhd The difference in black heights (right - left)
   * @return Pointer to the new root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_right_impl(NodeType* l,
                                                            NodeType* m1,
                                                            NodeType* r,
                                                            int bhd) {
    if (is_black(r) && (bhd == 0)) return join3_base_impl(l, m1, r);
    r->apply_deferred();
    r->set_left(join3_right_impl(l, m1, r->left, bhd - (is_black(r) ? 1 : 0)));
    l = r->left;
    if (is_black(r) && is_red(l) && is_red(l->left)) {
      // Rotate right
      set_black(l->left);
      r->set_left(l->right);
      r->update_subtree_data();
      l->set_right(r);
      l->update_subtree_data();
      return l;
    } else {
      r->update_subtree_data();
      return r;
    }
  }

  /**
   * @brief Joins three trees based on black height difference.
   *
   * This function chooses the appropriate joining strategy based on the
   * difference in black heights between the left and right trees. It
   * delegates to the specific implementation based on which tree is taller.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param bhd The difference in black heights (left - right)
   * @return Pointer to the new root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_impl_bhd(NodeType* l,
                                                          NodeType* m1,
                                                          NodeType* r,
                                                          int bhd) {
    return (bhd > 0)   ? join3_left_impl(l, m1, r, bhd)
           : (bhd < 0) ? join3_right_impl(l, m1, r, -bhd)
                       : join3_base_impl(l, m1, r);
  }

  /**
   * @brief Joins three trees while maintaining Red-Black tree properties.
   *
   * This function joins three trees (left, middle, right) into a single
   * Red-Black tree. It calculates the black height difference between
   * the left and right trees and uses the appropriate joining strategy.
   * The resulting root is always set to black to maintain the Red-Black
   * tree property that the root must be black.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @return Pointer to the new root of the joined tree
   */
  [[nodiscard]] static constexpr NodeType* join3_impl(NodeType* l, NodeType* m1,
                                                      NodeType* r) {
    auto root = join3_impl_bhd(l, m1, r, black_height(l) - black_height(r));
    set_black(root);
    root->set_parent(nullptr);
    return root;
  }

  /**
   * @brief Internal implementation of three-way join with black height
   * tracking.
   *
   * This function joins three trees and returns both the new root and the
   * black height of the resulting tree. It's used internally by split
   * operations to maintain black height information during the splitting
   * process.
   *
   * @param l The root of the left tree
   * @param m1 The middle node
   * @param r The root of the right tree
   * @param bhl The black height of the left tree
   * @param bhr The black height of the right tree
   * @return A pair containing the new root and the black height of the joined
   * tree
   */
  [[nodiscard]] static constexpr std::pair<NodeType*, int> join3_impl_internal(
      NodeType* l, NodeType* m1, NodeType* r, int bhl, int bhr) {
    auto root = join3_impl_bhd(l, m1, r, bhl - bhr);
    auto bh = std::max(bhl, bhr);
    if (is_red(root)) {
      set_black(root);
      ++bh;
    }
    return {root, bh};
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
  static constexpr void split_impl(NodeType* root, const Key& key,
                                   NodeType*& output_l, NodeType*& output_r) {
    output_l = nullptr;
    output_r = nullptr;
    int bhl = 0, bhr = 0;
    split_impl_internal(root, black_height(root), key, output_l, bhl, output_r,
                        bhr);
    if (output_l) output_l->set_parent(nullptr);
    if (output_r) output_r->set_parent(nullptr);
  }

  /**
   * @brief Internal implementation of split operation by key.
   *
   * This function recursively splits a Red-Black tree at a given key while
   * maintaining black height information. It disassembles the tree and
   * reassembles the parts using join operations to ensure both resulting
   * trees maintain Red-Black tree properties.
   *
   * @param root The root of the tree to split
   * @param bh The black height of the current subtree
   * @param key The key to split at
   * @param output_l Reference to store the left part (keys < key)
   * @param bhl Reference to store the black height of the left part
   * @param output_r Reference to store the right part (keys >= key)
   * @param bhr Reference to store the black height of the right part
   */
  static constexpr void split_impl_internal(NodeType* root, int bh,
                                            const Key& key, NodeType*& output_l,
                                            int& bhl, NodeType*& output_r,
                                            int& bhr) {
    root->apply_deferred();
    NodeType *l = root->left, *r = root->right, *m = nullptr;
    root->set_left(nullptr);
    root->set_right(nullptr);
    if (is_black(root)) --bh;
    int bhm = 0;
    if (root->key < key) {
      if (r) split_impl_internal(r, bh, key, m, bhm, output_r, bhr);
      std::tie(output_l, bhl) = join3_impl_internal(l, root, m, bh, bhm);
    } else {
      if (l) split_impl_internal(l, bh, key, output_l, bhl, m, bhm);
      std::tie(output_r, bhr) = join3_impl_internal(m, root, r, bhm, bh);
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
    int bhl = 0, bhr = 0;
    split_at_impl_internal(root, black_height(root), lsize, output_l, bhl,
                           output_r, bhr);
    if (output_l) output_l->set_parent(nullptr);
    if (output_r) output_r->set_parent(nullptr);
  }

  /**
   * @brief Internal implementation of split operation by index.
   *
   * This function recursively splits a Red-Black tree at a given inorder
   * index while maintaining black height information. It uses subtree
   * sizes to determine the split point and reassembles the parts using
   * join operations to ensure both resulting trees maintain Red-Black
   * tree properties.
   *
   * @param root The root of the tree to split
   * @param bh The black height of the current subtree
   * @param lsize The desired size of the left part
   * @param output_l Reference to store the left part
   * @param bhl Reference to store the black height of the left part
   * @param output_r Reference to store the right part
   * @param bhr Reference to store the black height of the right part
   */
  static constexpr void split_at_impl_internal(NodeType* root, int bh,
                                               size_t lsize,
                                               NodeType*& output_l, int& bhl,
                                               NodeType*& output_r, int& bhr) {
    root->apply_deferred();
    NodeType *l = root->left, *r = root->right, *m = nullptr;
    root->set_left(nullptr);
    root->set_right(nullptr);
    if (is_black(root)) --bh;
    int bhm = 0;
    const size_t left_size = bst::subtree_data::size(l);
    if (lsize < left_size) {
      split_at_impl_internal(l, bh, lsize, output_l, bhl, m, bhm);
      std::tie(output_r, bhr) = join3_impl_internal(m, root, r, bhm, bh);
    } else if (lsize == left_size) {
      output_l = l;
      bhl = bh;
      std::tie(output_r, bhr) = join3_impl_internal(nullptr, root, r, 0, bh);
    } else if (lsize == left_size + 1) {
      std::tie(output_l, bhl) = join3_impl_internal(l, root, nullptr, bh, 0);
      output_r = r;
      bhr = bh;
    } else {
      split_at_impl_internal(r, bh, lsize - left_size - 1, m, bhm, output_r,
                             bhr);
      std::tie(output_l, bhl) = join3_impl_internal(l, root, m, bh, bhm);
    }
  }
};

}  // namespace bst
