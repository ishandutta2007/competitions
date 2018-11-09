#pragma once

#include "../arithmetic.h"
#include "../utils/primitive_root.h"
#include "../../base.h"
#include "../../factorization/base.h"
#include "../../numeric/utils/usqrt.h"
#include <algorithm>
#include <unordered_map>

// Calculate discrete logarithm using "Baby-step giant-step" algorithm ( https://en.wikipedia.org/wiki/Baby-step_giant-step ).
// Default m = sqrt(P).
// Prebuild:
//   O(m) time, O(m) memory
// Log:
//   O(P / m) time
class PModularDiscreteLogSqrtMap
{
public:
    using TModularA = TModularArithmetic_P32U;

protected:
    uint64_t p, primitive;
    uint64_t m, ippm; 
    std::unordered_map<uint64_t, uint64_t> vmap;

public:
    void Build()
    {
        vmap.clear();
        uint64_t r = 1;
        for (uint64_t i = 0; i < m; ++i)
        {
            vmap[r] = i;
            r = TModularA::Mult(r, primitive, p);
        }
        ippm = TModularA::Inverse(TModularA::PowU(primitive, m, p), p);
    }

    PModularDiscreteLogSqrtMap(uint64_t prime, uint64_t pprimitive) : p(prime), primitive(pprimitive) { m = USqrt(p) + 1; Build(); }
    PModularDiscreteLogSqrtMap(uint64_t prime, uint64_t pprimitive, uint64_t size) : p(prime), primitive(pprimitive) { m = std::min(size, p-1); Build(); }
    PModularDiscreteLogSqrtMap(uint64_t prime, const TFactorization& p1_factorization) : PModularDiscreteLogSqrtMap(prime, FindSmallestPrimitive(prime, p1_factorization)) {}
    PModularDiscreteLogSqrtMap(uint64_t prime, const TFactorization& p1_factorization, uint64_t size) : PModularDiscreteLogSqrtMap(prime, FindSmallestPrimitive(prime, p1_factorization), size) {}

    uint64_t Log(uint64_t x) const
    {
        x = TModularA::ApplyU(x, p);
        assert(x);
        for (uint64_t i = 0; i * m <= p; ++i)
        {
            auto it = vmap.find(x);
            if (it != vmap.end())
                return i * m + it->second;
            x = TModularA::Mult(x, ippm, p);
        }
        assert(false);
        return 0;
    }
};
