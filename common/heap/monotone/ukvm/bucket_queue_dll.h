#pragma once

#include "common/base.h"
#include "common/heap/ukvm/data.h"
#include "common/node.h"
#include "common/nodes_manager.h"
#include "common/nodes_manager_fixed_size.h"

#include <vector>

namespace heap {
namespace monotone {
namespace ukvm {
// Bucket queue with double linked list
// P - max priority
// Memory  -- O(N + P)
// Add     -- O(1)
// DecV    -- O(1)
// IncV    -- O(1)
// Top     -- O(1 + P / N) amortized
// Pop     -- O(1 + P / N) amortized
class BucketQueueDLL {
 public:
  using TValue = unsigned;
  using TData = heap::ukvm::Data<TValue>;
  using TSelf = BucketQueueDLL;

  class TNode : public BaseNode {
   public:
    TNode *next = nullptr, *prev = nullptr;
  };

 protected:
  NodesManagerFixedSize<TNode> manager_key;
  NodesManager<TNode> manager_priority;
  std::vector<TNode*> priority_map;
  std::vector<unsigned> priorities;
  unsigned top_priority = 0;
  unsigned size = 0;

 protected:
  TNode* KNode(unsigned key) { return manager_key.NodeByRawIndex(key); }

  const TNode* KNode(unsigned key) const {
    return manager_key.NodeByRawIndex(key);
  }

  void ResetKeyMap(unsigned ukey_size) {
    manager_key.Reset(ukey_size);
    manager_priority.ResetNodes();
  }

 public:
  explicit BucketQueueDLL(unsigned ukey_size) {
    ResetKeyMap(ukey_size);
    priorities.resize(ukey_size);
  }

  BucketQueueDLL(const std::vector<unsigned>& v, bool skip_heap)
      : priorities(v) {
    ResetKeyMap(priorities.size());
    if (!skip_heap) {
      for (unsigned i = 0; i < priorities.size(); ++i) {
        unsigned p = priorities[i];
        AdjustQueueSize(p);
        auto knode = KNode(i), pnode = priority_map[p];
        knode->prev = pnode->prev;
        knode->prev->next = knode;
        pnode->prev = knode;
        knode->next = pnode;
      }
      size = priorities.size();
    }
  }

  bool Empty() const { return size == 0; }
  unsigned Size() const { return size; }
  unsigned UKeySize() const { return unsigned(priorities.size()); }

  bool InHeap(unsigned key) const { return KNode(key)->next; }

  unsigned Get(unsigned key) const { return priorities[key]; }

  const std::vector<TValue>& GetValues() const { return priorities; }

 public:
  void AddNewKey(unsigned key, unsigned priority, bool skip_heap = false) {
    assert(!InHeap(key));
    AddNewKeyI(key, priority, skip_heap);
  }

  void Set(unsigned key, unsigned new_priority) {
    if (InHeap(key))
      SetI(key, new_priority);
    else
      AddNewKeyI(key, new_priority, false);
  }

  void DecreaseValue(unsigned key, unsigned new_priority) {
    Set(key, new_priority);
  }

  void DecreaseValueIfLess(unsigned key, unsigned new_priority) {
    if (new_priority < priorities[key]) Set(key, new_priority);
  }

  void IncreaseValue(unsigned key, unsigned new_priority) {
    Set(key, new_priority);
  }

  void Add(const TData& x) { Set(x.key, x.value); }

  unsigned TopKey() {
    ShiftPriority();
    return manager_key.RawIndex(priority_map[top_priority]->next);
  }

  unsigned TopValue() {
    ShiftPriority();
    return top_priority;
  }

  TData Top() { return {TopKey(), TopValue()}; }

  void Pop() {
    ShiftPriority();
    RemoveNodeI(priority_map[top_priority]->next);
  }

  unsigned ExtractKey() {
    ShiftPriority();
    auto n = priority_map[top_priority]->next;
    unsigned key = manager_key.RawIndex(n);
    RemoveNodeI(n);
    return key;
  }

  unsigned ExtractValue() {
    Pop();
    return top_priority;
  }

  TData Extract() {
    TData t = Top();
    Pop();
    return t;
  }

  void DeleteKey(unsigned key) { RemoveNodeI(KNode(key)); }

 protected:
  void AdjustQueueSize(unsigned k) {
    unsigned s = priority_map.size();
    if (s <= k) {
      priority_map.resize(k + 1);
      for (; s <= k; ++s) {
        auto n = manager_priority.New();
        n->next = n->prev = n;
        priority_map[s] = n;
      }
    }
  }

  void ShiftPriority() {
    assert(!Empty());
    for (; priority_map[top_priority]->next == priority_map[top_priority];)
      ++top_priority;
  }

  void AddNewKeyI(unsigned key, unsigned priority, bool skip_heap) {
    priorities[key] = priority;
    if (!skip_heap) {
      AdjustQueueSize(priority);
      auto knode = KNode(key), pnode = priority_map[priority];
      knode->prev = pnode->prev;
      knode->prev->next = knode;
      pnode->prev = knode;
      knode->next = pnode;
      ++size;
    }
  }

  void SetI(unsigned key, unsigned new_priority) {
    if (priorities[key] != new_priority) {
      auto knode = KNode(key), pnode = priority_map[new_priority];
      knode->next->prev = knode->prev;
      knode->prev->next = knode->next;
      knode->prev = pnode->prev;
      knode->prev->next = knode;
      pnode->prev = knode;
      knode->next = pnode;
    }
  }

  void RemoveNodeI(TNode* node) {
    node->next->prev = node->prev;
    node->prev->next = node->next;
    node->prev = node->next = nullptr;
    --size;
  }
};
}  // namespace ukvm
}  // namespace monotone
}  // namespace heap