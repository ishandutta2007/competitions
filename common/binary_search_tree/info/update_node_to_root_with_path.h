#pragma once

#include "common/base.h"
#include "common/template.h"

#include <vector>

namespace bst {
namespace info {
template <class TNode>
constexpr void UpdateNodeToRootWithPath(
    const std::vector<TNode*>& node_to_root_path, size_t start_from_index,
    TMetaFalse) {}

template <class TNode>
inline void UpdateNodeToRootWithPath(
    const std::vector<TNode*>& node_to_root_path, size_t start_from_index,
    TMetaTrue) {
  for (auto it = node_to_root_path.begin() + start_from_index;
       it != node_to_root_path.end(); ++it)
    (*it)->UpdateInfo();
}

template <class TNode>
inline void UpdateNodeToRootWithPath(
    const std::vector<TNode*>& node_to_root_path, size_t start_from_index = 0) {
  UpdateNodeToRootWithPath(node_to_root_path, start_from_index,
                           TMetaBool<!TNode::TInfo::is_none>());
}

template <class TNode>
inline void UpdateNodeToRootWithPath_DataUpdated(
    const std::vector<TNode*>& node_to_root_path, size_t start_from_index = 0) {
  UpdateNodeToRootWithPath(node_to_root_path, start_from_index,
                           TMetaBool<TNode::TInfo::use_data>());
}

template <class TNode>
inline void UpdateNodeToRootWithPath_KeysUpdated(
    const std::vector<TNode*>& node_to_root_path, size_t start_from_index = 0) {
  UpdateNodeToRootWithPath(node_to_root_path, start_from_index,
                           TMetaBool<TNode::TInfo::use_keys>());
}
}  // namespace info
}  // namespace bst
