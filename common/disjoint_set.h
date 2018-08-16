#pragma once

#include "base.h"

class DisjointSet
{
protected:
    unsigned n;
    vector<unsigned> p;
    vector<unsigned> rank;
    vector<unsigned> vsize;
	unsigned unions;

public:
    DisjointSet(unsigned n = 0)
    {
        Init(n);
    }

	unsigned Size() const { return n; }
	unsigned GetUnions() const { return unions; }
	unsigned GetSetsCount() const { return n - unions; }

    void Init(unsigned n_)
    {
        n = n_;
        p.resize(n);
        for (unsigned i = 0; i < n; ++i) {
            p[i] = i;
        }
        rank.resize(n);
        fill(rank.begin(), rank.end(), 0);
        vsize.resize(n);
        fill(vsize.begin(), vsize.end(), 1);
        unions = 0;
    }

    void Union(unsigned i1, unsigned i2)
    {
        UnionI(Find(i1), Find(i2));
    }

	unsigned Find(unsigned x)
    {
        unsigned px = p[x], ppx = p[px];
        for (; px != ppx; )
        {
            p[x] = ppx;
            x = ppx;
            px = p[x];
            ppx = p[px];
        }
        return px;
    }

	unsigned GetSize(unsigned x)
    {
        return vsize[Find(x)];
    }

protected:
    void UnionI(unsigned i1, unsigned i2)
    {
        if (i1 == i2)
            return;
        ++unions;
        if (rank[i1] > rank[i2])
        {
            p[i2] = i1;
            vsize[i1] += vsize[i2];
        }
        else
        {
            p[i1] = i2;
            if (rank[i1] == rank[i2])
                ++rank[i1];
            vsize[i2] += vsize[i1];
        }
    }
};
