#include "common/stl_full.h"

int main_base()
{
	cout << "Hello from code jam main." << endl;
	unsigned T;
	cin >> T;
	for (unsigned it = 1; it <= T; ++it)
	{
		cout << "Case #" << it << ": ";
		// ...
		cout << endl;
	}
	return 0;
}
