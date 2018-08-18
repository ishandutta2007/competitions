#pragma once

#include "node.h"
#include "tree.h"
#include "action/apply_action.h"
#include "action/none.h"
#include "base/swap.h"
#include "info/size.h"

// In this implementation delete operation is different from wiki Scapegoat
// tree. It removes node from tree (similar to other trees), not just mark for
// future deletion.
template <
	bool _use_parent,
	class TTData,
	class TTInfo = BSTInfoSize,
	class TTAction = BSTActionNone,
	class TTKey = int64_t>
class ScapegoatTree : public BSTree<BSTNode<TTData, TTInfo, TTAction, true, _use_parent, false, TTKey>,
									ScapegoatTree<_use_parent, TTData, TTInfo, TTAction, TTKey>>
{
public:
	static const bool use_key = true;
	static const bool use_parent = _use_parent;
	static const bool use_height = false;
	static constexpr double alpha = 0.7; // Height is less or equal to 2 * height(fully balanced tree).

	using TData = TTData;
	using TInfo = TTInfo;
	using TAction = TTAction;
	using TKey = TTKey;
	using TNode = BSTNode<TData, TInfo, TAction, use_key, use_parent, use_height, TKey>;
	using TSelf = ScapegoatTree<use_parent, TData, TInfo, TAction, TKey>;
	using TTree = BSTree<TNode, TSelf>;
	friend class BSTree<TNode, TSelf>;

public:
	ScapegoatTree(unsigned max_nodes) : TTree(max_nodes) {}

protected:
	static void TraverseInorder(TNode* node, vector<TNode*>& output)
	{
		if (!node) return;
		node->ApplyAction();
		TraverseInorder(node->l, output);
		output.push_back(node);
		TraverseInorder(node->r, output);
	}

public:
	static TNode* RebuildSubtree(TNode* node)
	{
		assert(node);
		vector<TNode*> nodes;
		TraverseInorder(node, nodes);
		return TTree::BuildTree(nodes);
	}

	static TNode* UpdateAndFixSubtree(TNode* node)
	{
		node->UpdateInfo();
		return ((node->l && node->l->info.size > alpha * node->info.size)
			|| (node->r && node->r->info.size > alpha * node->info.size)) ?
			RebuildSubtree(node) : node;
	}

	static TNode* InsertByKey(TNode* root, TNode* node)
	{
		if (!root) return node;
		root->ApplyAction();
		if (root->key < node->key)
			root->SetR(InsertByKey(root->r, node));
		else
			root->SetL(InsertByKey(root->l, node));
		return UpdateAndFixSubtree(root);
	}

protected:
	static TNode* RemoveRootI(TNode* root)
	{
		if (!root->l) return root->r;
		if (!root->r) return root->l;
		static thread_local stack<TNode*> s;
		TNode* l = root->l, *r = root->r, *new_root;
		if (l->info.size < r->info.size)
		{
			new_root = r;
			for (new_root->ApplyAction(); new_root->l; new_root->ApplyAction())
			{
				s.push(new_root);
				new_root = new_root->l;
			}
			TNode* current = new_root->r;
			for (; !s.empty(); s.pop())
			{
				s.top()->SetL(current);
				current = s.top();
				current->UpdateInfo();
				current = UpdateAndFixSubtree(current);
			}
			new_root->SetL(l);
			new_root->SetR(current);
		}
		else
		{
			new_root = l;
			for (new_root->ApplyAction(); new_root->r; new_root->ApplyAction())
			{
				s.push(new_root);
				new_root = new_root->r;
			}
			TNode* current = new_root->l;
			for (; !s.empty(); s.pop())
			{
				s.top()->SetR(current);
				current = s.top();
				current->UpdateInfo();
				current = UpdateAndFixSubtree(current);
			}
			new_root->SetL(current);
			new_root->SetR(r);
		}
		return UpdateAndFixSubtree(new_root);
	}

public:
	static TNode* RemoveByKey(TNode* root, const TKey& key, TNode*& removed_node)
	{
		if (!root) return root;
		root->ApplyAction();
		if (root->key < key)
			root->SetR(RemoveByKey(root->r, key, removed_node));
		else if (root->key > key)
			root->SetL(RemoveByKey(root->l, key, removed_node));
		else
		{
			removed_node = root;
			TNode* new_root = RemoveRootI(root);
			root->ResetLinks();
			root->UpdateInfo();
			return new_root;
		}
		return UpdateAndFixSubtree(root);
	}

	static TNode* RemoveByNode(TNode* node)
	{
		static_assert(use_parent, "use_parent should be true");
		assert(node);
		ApplyActionRootToNode(node);

		// Optional swap
		if (node->l && node->r)
		{
			TNode* temp = node->l;
			for (temp->ApplyAction(); temp->r; )
			{
				temp = temp->r;
				temp->ApplyAction();
			}
			BSTSwapAuto(node, node->p, temp, temp->p);
		}

		// Drop node from tree
		TNode* child = node->l ? node->l : node->r;
		TNode* parent = node->p;
		if (child) child->p = parent;
		if (parent)
		{
			if (parent->l == node)
				parent->l = child;
			else
				parent->r = child;
		}
		node->ResetLinks();
		for (TNode * node = parent; node; node = parent)
		{
			parent = node->p;
			child = UpdateAndFixSubtree(node);
			if (child != node)
			{
				child->SetParentLink(parent);
				if (parent)
				{
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
