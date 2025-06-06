#pragma once

#include "common/base.h"
#include "common/heap/ext/fibonacci.h"
#include "common/heap/ukvm/data.h"
#include "common/memory/contiguous_nodes_manager.h"

#include <functional>
#include <vector>

namespace heap {
namespace ukvm {
// Memory  -- O(N)
// Add     -- O(1)
// DecV    -- O(1) amortized
// IncV    -- O(log N) amortized
// Top     -- O(1)
// Pop     -- O(log N) amortized
// Union   -- O(1)
template <class TTValue, class TCompare = std::less<TTValue>>
class Fibonacci : public heap::ext::Fibonacci<TTValue, TCompare,
                                              memory::ContiguousNodesManager> {
 public:
  using TValue = TTValue;
  using TData = Data<TValue>;
  using TBase =
      heap::ext::Fibonacci<TValue, TCompare, memory::ContiguousNodesManager>;
  using TSelf = Fibonacci<TValue, TCompare>;
  using TNode = typename TBase::Node;
  using TNodesManager = typename TBase::TNodesManager;

 protected:
  TNodesManager manager;

 protected:
  constexpr bool UnusedNode(const TNode* p) { return p->d == -1u; }

  constexpr void SetUnused(TNode* p) { p->d = -1u; }

  constexpr void CleanUnused(TNode* p) { p->d = 0; }

  constexpr TNode* GetNode(unsigned key) { return manager.at(key); }

 public:
  constexpr explicit Fibonacci(unsigned ukey_size)
      : TBase(manager), manager(ukey_size) {
    for (unsigned i = 0; i < ukey_size; ++i) SetUnused(manager.create());
  }

  constexpr Fibonacci(const std::vector<TValue>& v, bool skip_heap)
      : TBase(manager), manager(v.size()) {
    for (unsigned i = 0; i < v.size(); ++i) {
      auto p = manager.create();
      p->value = v[i];
      if (skip_heap) {
        SetUnused(p);
      } else {
        TBase::AddNode(p);
      }
    }
  }

  constexpr unsigned UKeySize() const { return manager.capacity(); }

  constexpr const TNode* GetNode(unsigned key) const { return manager.at(key); }

  constexpr bool InHeap(unsigned key) const {
    return !UnusedNode(GetNode(key));
  }

  constexpr const TValue& Get(unsigned key) const {
    return GetNode(key)->value;
  }

  constexpr std::vector<TValue> GetValues() const {
    const unsigned n = UKeySize();
    std::vector<TValue> v(n);
    for (unsigned i = 0; i < n; ++i) v[i] = Get(i);
    return v;
  }

  constexpr unsigned GetKey(const TNode* node) const {
    return manager.index(node);
  }

 protected:
  constexpr void AddNewKeyI(TNode* node, const TValue& new_value,
                            bool skip_heap) {
    node->value = new_value;
    if (!skip_heap) {
      CleanUnused(node);
      TBase::AddNode(node);
    }
  }

 public:
  constexpr void AddNewKey(unsigned key, const TValue& new_value,
                           bool skip_heap = false) {
    TNode* node = GetNode(key);
    assert(UnusedNode(node));
    AddNewKeyI(node, new_value, skip_heap);
  }

  constexpr void DecreaseValue(unsigned key, const TValue& new_value) {
    TNode* node = GetNode(key);
    if (UnusedNode(node)) {
      AddNewKeyI(node, new_value, false);
    } else {
      TBase::DecreaseValue(node, new_value);
    }
  }

  constexpr void DecreaseValueIfLess(unsigned key, const TValue& new_value) {
    TNode* node = GetNode(key);
    if (TBase::compare(new_value, node->value)) DecreaseValue(key, new_value);
  }

  constexpr void IncreaseValue(unsigned key, const TValue& new_value) {
    TNode* node = GetNode(key);
    assert(!UnusedNode(node));
    TBase::IncreaseValue(node, new_value);
  }

  constexpr void Set(unsigned key, const TValue& new_value) {
    TNode* node = GetNode(key);
    if (UnusedNode(node)) {
      AddNewKeyI(node, new_value, false);
    } else if (TBase::compare(new_value, node->value)) {
      TBase::DecreaseValue(node, new_value);
    } else {
      TBase::IncreaseValue(node, new_value);
    }
  }

  constexpr void Add(const TData& x) { Set(x.key, x.value); }

  constexpr unsigned TopKey() const { return GetKey(TBase::TopNode()); }

  constexpr const TValue& TopValue() const { return TBase::Top(); }

  constexpr TData Top() const {
    TNode* node = TBase::TopNode();
    return {GetKey(node), node->value};
  }

  constexpr void Delete(TNode* node) {
    TBase::RemoveNode(node);
    SetUnused(node);
  }

  constexpr void Pop() { Delete(TBase::TopNode()); }

  constexpr unsigned ExtractKey() {
    const unsigned t = TopKey();
    Pop();
    return t;
  }

  constexpr const TValue& ExtractValue() {
    const TValue& t = TopValue();
    Pop();
    return t;
  }

  constexpr TData Extract() {
    const TData t = Top();
    Pop();
    return t;
  }
};
}  // namespace ukvm
}  // namespace heap
