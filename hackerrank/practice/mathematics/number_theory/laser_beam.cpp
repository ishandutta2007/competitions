// https://www.hackerrank.com/challenges/laser-beam

#include "common/factorization/mertens.h"
#include "common/modular/static/modular_io.h"
#include "common/numeric/utils/usqrt.h"
#include "common/stl/base.h"

using TModular = Modular<>;

int main_laser_beam()
{
    uint64_t N0 = 1;
    Mertens mertens(3000000);
    vector<TModular> vf(N0), vfs(N0);
    for (uint64_t i = 1; i < N0; ++i)
        vf[i] = 24 * i * i + 2;
    for (uint64_t i = 1; 2 * i < N0; ++i)
    {
        for (uint64_t j = 2 * i; j < N0; j += i)
            vf[j] -= vf[i];
        vfs[i] = vfs[i - 1] + vf[i];
    }

    auto Mobius = [&](uint64_t n) { TModular x; x.SetS(mertens.GetMobius(n)); return x; };
    auto Mertens = [&](uint64_t n) { TModular x; x.SetS(mertens.GetMertens(n)); return x; };
    auto G = [](uint64_t n) { TModular x(n); return x * x * 24 + 2; };
    auto GS = [](uint64_t n) { TModular x(2*n + 1); return x*x*x; };

    auto FS = [&](uint64_t n)
    {
        if (n < N0) return vfs[n];
        uint64_t v = USqrt(n), ml = n / (v + 1);
        TModular r = 0;
        for (uint64_t l = 1; l <= ml; ++l)
            r += G(l) * Mertens(n / l);
        for (uint64_t k = 1; k <= v; ++k)
            r += (GS(n / k) - GS(n / (k + 1))) * Mertens(k);
        return r;
    };
    auto F = [&](uint64_t n) { return (n < N0) ? vf[n] : FS(n) - FS(n-1); };

    unsigned T;
    cin >> T;
    for (unsigned iT = 0; iT < T; ++iT)
    {
        uint64_t N, M, D;
        cin >> N >> M >> D;
        uint64_t NM = N / M, maxk = 0;
        // if (D == 1)
        //     cout << FS(NM) << endl;
        // else
        {
            TModular r = 0;
            uint64_t l = min(USqrt(N), NM);
            for (uint64_t k = 1;; ++k)
            {
                uint64_t kd = k * D, i1 = N / kd, i0 = N / (kd + 1);
                i1 = min(i1, NM);
                if (i1 <= l) break;
                if (i0 >= i1) continue;
                r += FS(i1) - FS(i0);
                l = min(l, i0);
            }
            for (uint64_t i = 1; i <= l; ++i)
            {
                if (((N / i) % D) == 0)
                    r += F(i);
            }
            cout << r << endl;
        }
    }
    return 0;
}
