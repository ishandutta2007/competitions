#pragma once

#include "common/base.h"
#include "common/geometry/kdtree/base/dset.h"
#include "common/geometry/kdtree/base/tree.h"
#include "common/geometry/kdtree/info/update_node_to_root.h"
#include "common/template.h"

namespace geometry {
namespace kdtree {
// Implicit k-d tree with plane split by half on the longest dimension.
template <unsigned _dim, class TTPoint, class TTLData, class TTIData,
          class TTInfo, class TTAction>
class ISPTree
    : public base::Tree<_dim, TTPoint, TTLData, TTIData, TTInfo, TTAction> {
 public:
  static const unsigned dim = _dim;

  using TPoint = TTPoint;
  using TValue = typename TPoint::TType;
  using TLData = TTLData;
  using TIData = TTIData;
  using TBase = base::Tree<dim, TPoint, TLData, TIData, TTInfo, TTAction>;
  using TNode = typename TBase::TNode;

 protected:
  void SetI(const TPoint& pp, const TLData& ldata, TFakeFalse) {
    auto p = TBase::root;
    auto pb = TBase::sb, pe = TBase::se;
    for (p->ApplyAction(); !p->IsLeaf(); p->ApplyAction()) {
      if (pp[p->split_dim] < p->split_value) {
        pe[p->split_dim] = p->split_value;
        p = p->l;
      } else {
        pb[p->split_dim] = p->split_value;
        p = p->r;
      }
    }
    if (p->ldata == ldata) return;
    for (; TBase::PossibleToSplit(pb, pe);) {
      TBase::SplitNodeHLD(p, pb, pe);
      if (pp[p->split_dim] < p->split_value) {
        pe[p->split_dim] = p->split_value;
        p = p->l;
      } else {
        pb[p->split_dim] = p->split_value;
        p = p->r;
      }
    }
    p->ldata = ldata;
    p->UpdateLeafInfo(pb, pe);
    info::UpdateNodeToRoot(p->p);
  }

  void SetI(const TPoint& pp, const TLData& ldata, TFakeTrue) {
    auto p = TBase::root;
    for (p->ApplyAction(); !p->IsLeaf(); p->ApplyAction())
      p = (pp[p->split_dim] < p->split_value) ? p->l : p->r;
    if (p->ldata == ldata) return;
    for (; TBase::PossibleToSplit(p->idata.b, p->idata.e);) {
      TBase::SplitNodeHLD(p);
      p = (pp[p->split_dim] < p->split_value) ? p->l : p->r;
    }
    p->ldata = ldata;
    p->UpdateLeafInfo(p->idata.b, p->idata.e);
    info::UpdateNodeToRoot(p->p);
  }

 public:
  void Set(const TPoint& pp, const TLData& ldata) {
    SetI(pp, ldata, TFakeBool<TIData::support_box>());
  }

  void Add(const TPoint& pp, const TLData& ldata) { Set(pp, ldata + Get(pp)); }

 protected:
  void SetRIF(TNode* p, const TPoint& rb, const TPoint& re, const TPoint& pb,
              const TPoint& pe, const TLData& ldata, unsigned dd) {
    p->ApplyAction();
    if (dd == 0) return TBase::SetRIReplace(p, pb, pe, ldata);
    if (p->IsLeaf() && (p->ldata == ldata)) return;  // No changes
    if (p->IsLeaf()) TBase::SplitNodeHLD(p, rb, re);
    unsigned d = p->split_dim;
    const TValue& v = p->split_value;
    if (pb[d] >= v) {
      SetRIF(p->r, base::DSet(d, rb, v), re, pb, pe, ldata,
             dd - ((pb[d] == v) ? 1u : 0u));
    } else if (pe[d] <= v) {
      SetRIF(p->l, rb, base::DSet(d, re, v), pb, pe, ldata,
             dd - ((pe[d] == v) ? 1u : 0u));
    } else {
      SetRIF(p->l, rb, base::DSet(d, re, v), pb, base::DSet(d, pe, v), ldata,
             dd - ((re[d] == pe[d]) ? 0u : 1u));
      SetRIF(p->r, base::DSet(d, rb, v), re, base::DSet(d, pb, v), pe, ldata,
             dd - ((rb[d] == pb[d]) ? 0u : 1u));
    }
    p->UpdateInfo();
  }

  void SetRIT(TNode* p, const TPoint& pb, const TPoint& pe, const TLData& ldata,
              unsigned dd) {
    p->ApplyAction();
    if (dd == 0) return TBase::SetRIReplace(p, pb, pe, ldata);
    if (p->IsLeaf() && (p->ldata == ldata)) return;  // No changes
    if (p->IsLeaf()) TBase::SplitNodeHLD(p);
    unsigned d = p->split_dim;
    const TValue& v = p->split_value;
    if (pb[d] >= v) {
      SetRIT(p->r, pb, pe, ldata, dd - ((pb[d] == v) ? 1u : 0u));
    } else if (pe[d] <= v) {
      SetRIT(p->l, pb, pe, ldata, dd - ((pe[d] == v) ? 1u : 0u));
    } else {
      SetRIT(p->l, pb, base::DSet(d, pe, v), ldata,
             dd - ((p->idata.e[d] == pe[d]) ? 0u : 1u));
      SetRIT(p->r, base::DSet(d, pb, v), pe, ldata,
             dd - ((p->idata.b[d] == pb[d]) ? 0u : 1u));
    }
    p->UpdateInfo();
  }

  void SetRI(TNode* p, const TPoint& pb, const TPoint& pe, const TLData& ldata,
             unsigned dd, TFakeFalse) {
    SetRIF(p, TBase::sb, TBase::se, pb, pe, ldata, dd);
  }

  void SetRI(TNode* p, const TPoint& pb, const TPoint& pe, const TLData& ldata,
             unsigned dd, TFakeTrue) {
    SetRIT(p, pb, pe, ldata, dd);
  }

 public:
  void Set(const TPoint& pb, const TPoint& pe, const TLData& ldata) {
    unsigned dd = 0;
    for (unsigned i = 0; i < dim; ++i)
      dd += ((TBase::sb[i] == pb[i]) ? 0u : 1u) +
            ((TBase::se[i] == pe[i]) ? 0u : 1u);
    SetRI(TBase::root, pb, pe, ldata, dd, TFakeBool<TIData::support_box>());
  }
};
}  // namespace kdtree
}  // namespace geometry
