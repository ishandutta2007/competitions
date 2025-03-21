#pragma once

#include "common/base.h"
#include "common/binary_search_tree/action/add_each_key__sum_keys.h"
#include "common/binary_search_tree/subtree_data/size.h"
#include "common/binary_search_tree/subtree_data/sum_keys.h"

#include <string>
#include <vector>

class TesterBinarySearchTreeSplitJoin {
 public:
  using TKey = int64_t;
  using TInfo = bst::subtree_data::SumKeys<TKey, bst::subtree_data::Size>;
  using TAction = bst::action::AddEachKeySumKeys<TKey>;

 protected:
  const TKey max_key = (1ll << 30);
  std::vector<TKey> vkeys;
  std::vector<TKey> vshift;

 public:
  TesterBinarySearchTreeSplitJoin(size_t test_size);

 protected:
  template <class TTree, class TNode>
  void Rotate(TTree& tree, TNode*& root, TKey value) const;

  template <class TTree>
  size_t TestBase(const std::string& name) const;

 public:
  bool TestAll(bool small_test) const;
};
