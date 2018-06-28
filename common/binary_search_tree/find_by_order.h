#pragma once

namespace {

template<class TNode>
TNode* FindByOrder(TNode* root, unsigned order_index)
{
	if (!root) return 0;
	if (order_index >= root->info.size) return 0;
	for (TNode* node = root; node;)
	{
		node->ApplyAction();
		unsigned ls = (node->l ? node->l->info.size : 0);
		if (order_index < ls)
		{
			node = node->l;
		}
		else if (order_index == ls)
		{
			return node;
		}
		else
		{
			order_index -= 1 + ls;
			node = node->r;
		}
	}
	assert(false);
	return 0;
}

} // namespace
