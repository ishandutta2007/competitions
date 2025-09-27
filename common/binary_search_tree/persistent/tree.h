#pragma once

#include "common/binary_search_tree/base/extended_tree.h"

namespace bst {
namespace persistent {

template <class TTNodesManager, class TTMe>
class Tree : public bst::base::ExtendedTree<TTNodesManager, TTMe> {
 public:
  using TNodesManager = TTNodesManager;
  using TNode = typename TNodesManager::NodeType;
  using TData = typename TNode::DataType;
  using TKey = typename TNode::KeyType;
  using TBase = bst::base::ExtendedTree<TTNodesManager, TTMe>;
  using TMe = TTMe;

  static constexpr bool is_persistent = true;

 public:
  Tree() : TBase() {}
  explicit Tree(size_t expected_nodes) : TBase(expected_nodes) {}

  TNode* insert_new(TNode* root, const TData& data, const TKey& key) {
    return TBase::derived()->insert(root, TBase::create_node(data, key));
  }

  TNode* PClone(TNode* node) {
    TNode* new_node = TBase::nodes_manager_.create();
    *new_node = *node;
    return new_node;
  }

  static void UpdatePForChildren(TNode* node) {
    if (node) {
      if (node->left) node->left->set_parent(node);
      if (node->right) node->right->set_parent(node);
    }
  }
};

}  // namespace persistent
}  // namespace bst
