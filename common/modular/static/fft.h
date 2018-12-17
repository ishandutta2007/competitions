#pragma once

#include <algorithm>
#include <vector>

#include "../utils/primitive_root.h"
#include "../../base.h"
#include "../../factorization/factorization.h"

template <class TModular>
class ModularFFT
{
public:
    using TVector = std::vector<TModular>;

protected:
    TModular primitive = 0;
    unsigned maxn = 0;
    std::vector<TVector> roots;
    std::vector<std::vector<unsigned>> bit_rev_map;

protected:
    void Init(unsigned n)
    {
        uint64_t p = TModular::GetMod();
        TFactorization p1f = FastFactorize(p - 1);
        assert(p1f.size() > 0);
        primitive = FindSmallestPrimitive<typename TModular::TArithmetic>(p, p1f);
        unsigned max2p;
        if (n)
        {
            maxn = GetFFTN(n);
            for (max2p = 0; (1u << max2p) < maxn; ++max2p);
        }
        else
        {
            max2p = ((p1f[0].first == 2) ? p1f[0].second : 0u);
            maxn = 1u << max2p;
        }

        TVector base_roots(maxn);
        TModular nprimitive = primitive.PowU((p - 1) / maxn), r = 1;
        for (unsigned i = 0; i < maxn; ++i)
        {
            base_roots[i] = r;
            r *= nprimitive;
        }
        assert(r == 1);

        roots.clear();
        roots.reserve(max2p + 1);
        for (unsigned k = 1; k <= maxn; k *= 2)
		{
			TVector v(k);
			for (unsigned i = 0; i < k; ++i)
				v[i] = base_roots[i * (maxn / k)];
			roots.push_back(v);
		}

        bit_rev_map.clear();
        bit_rev_map.reserve(max2p + 1);
        bit_rev_map.push_back({0});
        for (unsigned b = 1; b < maxn; b *= 2)
        {
            std::vector<unsigned> v(2 * b);
            for (unsigned i = 0; i < 2 * b; ++i)
                v[i] = bit_rev_map.back()[i / 2] + ((i & 1) ? b : 0u);
            bit_rev_map.push_back(v);
        }
    }

    void FFTI_Adjust(TVector& output) const
	{
		std::reverse(output.begin() + 1, output.end());
		TModular invn = TModular(output.size()).Inverse();
		for (size_t i = 0; i < output.size(); ++i)
            output[i] *= invn;
	}

public:
    static unsigned GetFFTN(unsigned l) { unsigned n = 1; for (; n < l; n *= 2); return n; }

    ModularFFT(unsigned n = 0) { Init(n); }

    TVector FFT(unsigned n, const TVector& vx) const
	{
        assert((n > 0) && (maxn % n == 0));
        unsigned k = 0;
        for (; (1u << k) < n;) ++k;
        TVector output(n);
		for (unsigned i = 0; i < n; ++i)
			output[i] = (bit_rev_map[k][i] < vx.size()) ? vx[bit_rev_map[k][i]] : 0;

		TVector current(n);
		for (unsigned l = 1, kl = 1; l < n; l <<= 1, ++kl)
		{
			const TVector& current_roots = roots[kl];
			for (unsigned pdest = 0; pdest < n; pdest += l)
			{
				for (unsigned i = 0; i < l; ++i, ++pdest)
				{
                    TModular v = current_roots[i] * output[pdest + l];
                    current[pdest] = output[pdest] + v;
                    current[pdest + l] = output[pdest] - v;
				}
			}
			output.swap(current);
		}
        return output;
	}

    TVector FFTI(unsigned n, const TVector& vx) const
    {
        TVector output = FFT(n, vx);
        FFTI_Adjust(output);
        return output;
    }
};
