// https://www.hackerrank.com/challenges/super-humble-matrix

#include "common/modular/static/factorial.h"
#include "common/modular/static/modular_io.h"
#include "common/stl/base.h"

using TModular = Modular<>;
using TFactorial = ModularFactorial<TModular>;

int main_super_humble_matrix()
{
    TFactorial f;
    unsigned n, m;
    cin >> n >> m;
    if (n < m) swap(n, m);
    TModular r = 1;
    for (unsigned i = 1; i < m; ++i)
        r *= f(i) * f(i);
    r *= f(m).PowU(n - m + 1);
    cout << r << endl;
	return 0;
}
