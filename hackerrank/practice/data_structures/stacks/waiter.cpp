// https://www.hackerrank.com/challenges/waiter

#include "common/numeric/primes_list.h"
#include "common/stl_base.h"

#include <stack>

int main_waiter()
{
	PrimesList pl(100000);
	unsigned N, Q, x;
	cin >> N >> Q;
	stack<unsigned> sa, sb, st;
	for (unsigned in = 0; in < N; ++in)
	{
		cin >> x;
		sa.push(x);
	}
	for (unsigned iq = 0; iq < Q; ++iq)
	{
		unsigned p = unsigned(pl.vprimes[iq]);
		for (; !sa.empty(); sa.pop())
		{
			if (sa.top() % p)
				st.push(sa.top());
			else
				sb.push(sa.top());
		}
		for (; !sb.empty(); sb.pop())
			cout << sb.top() << endl;
		sa.swap(st);
	}
	for (; !sa.empty(); sa.pop())
		cout << sa.top() << endl;
	return 0;
}