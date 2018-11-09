// https://www.hackerrank.com/challenges/coprime-power-sum

#include "common/modular/static/modular.h"
#include "common/polynomial/base_newton_polynomial.h"
#include "common/vector/read.h"
#include "common/stl/base.h"

#include <functional>

using TModular = Modular<>;
using TPolynom = BaseNewtonPolynomial<TModular>;

int main_coprime_power_sum()
{
    unsigned T, N, K, maxn = 50, max_cache_size = 1000;
    cin >> T;
    for (unsigned iT = 0; iT < T; ++iT)
    {
        uint64_t M;
        cin >> N >> K >> M;
        vector<uint64_t> vs = ReadVector<uint64_t>(N);
        sort(vs.begin(), vs.end());
        reverse(vs.begin(), vs.end());

        vector<TModular> vspk;
        for (uint64_t s : vs)
            vspk.push_back(TModular(s).PowU(K));

        uint64_t tail_cache_size = 1;
        unsigned l = unsigned(vs.size());
        for (; l && (tail_cache_size <= max_cache_size); )
            tail_cache_size *= vs[--l];
        if (tail_cache_size > max_cache_size)
            tail_cache_size /= vs[l++];
        vector<unsigned> vskipped(tail_cache_size, 0);
        if (tail_cache_size > 1)
        vskipped[0] = (l == vs.size()) ? 0 : 1;
        for (unsigned i = l; i < vs.size(); ++i)
        {
            for (uint64_t s = vs[i], ss = s; ss < tail_cache_size; ss += s)
                vskipped[ss] = 1;
        }
        uint64_t full_cache_size = tail_cache_size * (K + 2);
        vector<TModular> vcache(full_cache_size);
        for (uint64_t i = 1; i < full_cache_size; ++i)
            vcache[i] = (vskipped[i % tail_cache_size] ? vcache[i-1] : vcache[i-1] + TModular(i).PowU(K));
        vector<TPolynom> vpoly(tail_cache_size);
        vector<TModular> vpolytemp(K + 2);
        for (uint64_t i = 0; i < tail_cache_size; ++i)
        {
            for (unsigned j = 0; j < K + 2; ++j)
                vpolytemp[j] = vcache[i + j * tail_cache_size];
            vpoly[i].Interpolate(vpolytemp);
        }

        std::function<TModular(uint64_t, unsigned)> SolveR = [&](uint64_t rM, unsigned i) -> TModular
        {
            return (i >= l) ? 
                (rM < full_cache_size ? vcache[rM] : vpoly[rM % tail_cache_size].Apply(TModular(rM / tail_cache_size))) :
                rM < vs[i] ? SolveR(rM, i + 1) : SolveR(rM, i + 1) - SolveR(rM / vs[i], i + 1) * vspk[i];
        };

        cout << ((vs.back() == 1) ? uint64_t(0) : SolveR(M, 0).Get()) << endl;
    }
	return 0;
}
