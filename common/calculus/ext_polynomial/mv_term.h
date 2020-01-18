#pragma once

#include "common/calculus/ext_polynomial/mv_term_power.h"
#include "common/calculus/ext_polynomial/term.h"

namespace calculus {
namespace ext_polynomial {
template <class TValue, unsigned dim>
using MVTerm = Term<TValue, MVTermPower<TValue, dim>>;
}  // namespace ext_polynomial
}  // namespace calculus
