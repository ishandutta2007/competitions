#pragma once

#include "../arithmetic.h"

// From chinese remainder theorem
template<class TModularAQ = TModularArithmetic_C32U>
static uint64_t MergeRemainders(uint64_t p, uint64_t pr, uint64_t q, uint64_t qr)
{
    return TModularAQ::Div(TModularAQ::SubSafe(qr, pr, q), p % q, q) * p + pr;
}
