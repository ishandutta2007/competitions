// https://www.hackerrank.com/challenges/assignment

#include "common/numeric/log_double.h"
#include "common/statistics/stat_m1.h"
#include "common/stl/base.h"

#include <iomanip>
#include <unordered_map>
#include <utility>

int main_assignment_problem()
{
    unsigned maxm = 251, maxm2 = maxm * maxm; //251;
    vector<LogDouble> v(maxm * maxm * maxm); // layers max_length m
    for (unsigned l = 1; l < maxm; ++l)
        v[1 * maxm2 + l * maxm + l] = 1.0;
    for (unsigned m = 1; m + 1 < maxm; ++m)
    {
        for (unsigned l = 1; l <= m; ++l)
        {
            unsigned ml_end = m - l + 2;
            for (unsigned ml = (m - 1) / l + 1; ml < ml_end; ++ml)
            {
                LogDouble c = v[l * maxm2 + ml * maxm + m];                
                // if (c.GetLog() < -0.5) { cout << m << "\t" << l << "\t" << ml << endl; continue; }// c == 0
                for (unsigned k = 1; m + k < maxm; ++k)
                    v[(l + 1) * maxm2 + max(k, ml) * maxm + (m + k)] += c;
            }
        }
    }

	cout << setprecision(3) << fixed;
	unsigned T, M, N;
	cin >> T;
	for (unsigned iT = 0; iT < T; ++iT)
	{
        StatM1<LogDouble> s;
        cin >> M >> N;
        LogDouble b = 1.0;
        for (unsigned l = 1; l <= min(M, N); ++l)
        {
            b *= double(N - l + 1) / double(l);
            for (unsigned ml = 1; ml + l - 1 <= M; ++ml)
                s.AddSample(double(ml), v[l* maxm2 + ml *maxm + M] * b);
        }
        cout << s.Mean().Get() << endl;
	}
	return 0;
}