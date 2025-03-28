#pragma once

#include "common/base.h"
#include "common/binary_search_tree/action/none.h"
#include "common/binary_search_tree/base/insert_by_key.h"
#include "common/binary_search_tree/base/swap.h"
#include "common/binary_search_tree/node.h"
#include "common/binary_search_tree/subtree_data/helpers/update_node_to_root.h"
#include "common/binary_search_tree/subtree_data/size.h"
#include "common/binary_search_tree/tree.h"
#include "common/memory/nodes_manager_fixed_size.h"
#include "common/template.h"

#include <stack>
#include <vector>

namespace bst {
// In this implementation delete operation is different from wiki Scapegoat
// tree. It removes node from tree (similar to other trees), not just mark for
// future deletion.
template <bool use_parent, class TData, class TInfo = subtree_data::Size,
          class TAction = action::None, class TKey = int64_t,
          template <class> class TTNodesManager = NodesManagerFixedSize>
class ScapegoatTree
    : public Tree<TTNodesManager<Node<TData, TInfo, TAction, true, use_parent,
                                      false, TKey>>,
                  ScapegoatTree<use_parent, TData, TInfo, TAction, TKey,
                                TTNodesManager>> {
 public:
  // Height is less or equal to 2 * height(fully balanced tree).
  static constexpr double alpha = 0.7;

  using TNode = Node<TData, TInfo, TAction, true, use_parent, false, TKey>;
  using TSelf =
      ScapegoatTree<use_parent, TData, TInfo, TAction, TKey, TTNodesManager>;
  using TTree = Tree<TTNodesManager<TNode>, TSelf>;
  friend class Tree<TTNodesManager<TNode>, TSelf>;

 public:
  explicit ScapegoatTree(size_t max_nodes) : TTree(max_nodes) {}

 protected:
  static void TraverseInorder(TNode* node, std::vector<TNode*>& output) {
    if (!node) return;
    node->ApplyAction();
    TraverseInorder(node->l, output);
    output.push_back(node);
    TraverseInorder(node->r, output);
  }

  static TNode* RebuildSubtree(TNode* node) {
    assert(node);
    std::vector<TNode*> nodes;
    TraverseInorder(node, nodes);
    return TTree::BuildTree(nodes);
  }

  static bool RebuildRequired(TNode* node) {
    assert(node);
    return ((node->l &&
             node->l->subtree_data.size > alpha * node->subtree_data.size) ||
            (node->r &&
             node->r->subtree_data.size > alpha * node->subtree_data.size));
  }

  static TNode* UpdateAndFixSubtree(TNode* node) {
    node->UpdateInfo();
    return ((node->l &&
             node->l->subtree_data.size > alpha * node->subtree_data.size) ||
            (node->r &&
             node->r->subtree_data.size > alpha * node->subtree_data.size))
               ? RebuildSubtree(node)
               : node;
  }

 protected:
  static inline TNode* InsertByKeyUseInsert(TNode* root, TNode* new_node) {
    for (TNode *node = root, *p = nullptr;;) {
      node->ApplyAction();
      if (RebuildRequired(node)) {
        bst::InsertByKeySkipUpdate(node, new_node);
        TNode* r = RebuildSubtree(node);
        (p ? ((node == p->l) ? p->l : p->r) : root) = r;
        r->set_parent(p);
        return root;
      }
      node->subtree_data.Insert(new_node);
      p = node;
      if (node->key < new_node->key) {
        if (node->r) {
          node = node->r;
        } else {
          node->set_right(new_node);
          break;
        }
      } else {
        if (node->l) {
          node = node->l;
        } else {
          node->set_left(new_node);
          break;
        }
      }
    }
    return root;
  }

  static inline TNode* InsertByKeyUseParent(TNode* root, TNode* new_node) {
    TNode* node = root;
    for (;;) {
      node->ApplyAction();
      if (RebuildRequired(node)) {
        bst::InsertByKeySkipUpdate(node, new_node);
        TNode* p = node->p;
        TNode* r = RebuildSubtree(node);
        (p ? ((node == p->l) ? p->l : p->r) : root) = r;
        r->set_parent(p);
        if (!p) return r;
        node = p;
        break;
      }
      if (node->key < new_node->key) {
        if (node->r) {
          node = node->r;
        } else {
          node->set_right(new_node);
          break;
        }
      } else {
        if (node->l) {
          node = node->l;
        } else {
          node->set_left(new_node);
          break;
        }
      }
    }
    subtree_data::propagate_to_root(node);
    return root;
  }

  static inline TNode* InsertByKeyDefault(TNode* root, TNode* new_node) {
    thread_local std::stack<TNode*> s;
    for (TNode* node = root;;) {
      node->ApplyAction();
      if (RebuildRequired(node)) {
        bst::InsertByKeySkipUpdate(node, new_node);
        TNode *r = RebuildSubtree(node), *p = s.empty() ? nullptr : s.top();
        (p ? ((node == p->l) ? p->l : p->r) : root) = r;
        r->set_parent(p);
        return root;
      }
      s.push(node);
      if (node->key < new_node->key) {
        if (node->r) {
          node = node->r;
        } else {
          node->set_right(new_node);
          break;
        }
      } else {
        if (node->l) {
          node = node->l;
        } else {
          node->set_left(new_node);
          break;
        }
      }
    }
    for (; !s.empty(); s.pop()) s.top()->UpdateInfo();
    return root;
  }

  static TNode* InsertByKeyI(TNode* root, TNode* node, MetaFalse, MetaFalse) {
    return InsertByKeyDefault(root, node);
  }

  static TNode* InsertByKeyI(TNode* root, TNode* node, MetaFalse, MetaTrue) {
    return InsertByKeyUseParent(root, node);
  }

  static TNode* InsertByKeyI(TNode* root, TNode* node, MetaTrue, MetaFalse) {
    return InsertByKeyUseInsert(root, node);
  }

  static TNode* InsertByKeyI(TNode* root, TNode* node, MetaTrue, MetaTrue) {
    return InsertByKeyUseInsert(root, node);
  }

 public:
  static TNode* InsertByKey(TNode* root, TNode* node) {
    if (!root) return node;
    return InsertByKeyI(root, node, MetaBool<TNode::support_insert>(),
                        MetaBool<TNode::use_parent>());
  }

 protected:
  static TNode* RemoveRootI(TNode* root) {
    if (!root->l) return root->r;
    if (!root->r) return root->l;
    thread_local std::stack<TNode*> s;
    TNode *l = root->l, *r = root->r, *new_root;
    if (l->subtree_data.size < r->subtree_data.size) {
      new_root = r;
      for (new_root->ApplyAction(); new_root->l; new_root->ApplyAction()) {
        s.push(new_root);
        new_root = new_root->l;
      }
      TNode* current = new_root->r;
      for (; !s.empty(); s.pop()) {
        s.top()->set_left(current);
        current = s.top();
        current->UpdateInfo();
        current = UpdateAndFixSubtree(current);
      }
      new_root->set_left(l);
      new_root->set_right(current);
    } else {
      new_root = l;
      for (new_root->ApplyAction(); new_root->r; new_root->ApplyAction()) {
        s.push(new_root);
        new_root = new_root->r;
      }
      TNode* current = new_root->l;
      for (; !s.empty(); s.pop()) {
        s.top()->set_right(current);
        current = s.top();
        current->UpdateInfo();
        current = UpdateAndFixSubtree(current);
      }
      new_root->set_left(current);
      new_root->set_right(r);
    }
    return UpdateAndFixSubtree(new_root);
  }

 public:
  static TNode* RemoveByKey(TNode* root, const TKey& key,
                            TNode*& removed_node) {
    if (!root) return root;
    root->ApplyAction();
    if (root->key < key) {
      root->set_right(RemoveByKey(root->r, key, removed_node));
    } else if (root->key > key) {
      root->set_left(RemoveByKey(root->l, key, removed_node));
    } else {
      removed_node = root;
      TNode* new_root = RemoveRootI(root);
      root->ResetLinksAndUpdateInfo();
      return new_root;
    }
    return UpdateAndFixSubtree(root);
  }

 protected:
  static TNode* RemoveByNodeI(TNode* node) {
    // Optional swap
    if (node->l && node->r) {
      TNode* temp = node->l;
      for (temp->ApplyAction(); temp->r;) {
        temp = temp->r;
        temp->ApplyAction();
      }
      SwapAuto(node, node->p, temp, temp->p);
    }

    // Drop node from tree
    TNode* child = node->l ? node->l : node->r;
    TNode* parent = node->p;
    if (child) child->p = parent;
    if (parent) {
      if (parent->l == node)
        parent->l = child;
      else
        parent->r = child;
    }
    node->ResetLinksAndUpdateInfo();
    for (TNode* node = parent; node; node = parent) {
      parent = node->p;
      child = UpdateAndFixSubtree(node);
      if (child != node) {
        child->set_parent(parent);
        if (parent) {
          if (parent->l == node)
            parent->l = child;
          else
            parent->r = child;
        }
      }
    }
    return child;
  }
};
}  // namespace bst
