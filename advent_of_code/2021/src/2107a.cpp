#include "common/stl/base.h"
#include "common/vector/read_line.h"

int main_2107a() {
  auto v = nvector::ReadLine<int>(',');
  sort(v.begin(), v.end());
  int m = v[v.size() / 2], r = 0;
  for (auto x : v) r += abs(x - m);
  cout << r << endl;
  return 0;
}
