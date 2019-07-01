#include "base/ext/map.h"

#include "base/booster_type.h"
#include "base/decode.h"
#include "base/point.h"
#include "common/assert_exception.h"
#include "common/string/split.h"
#include "common/unsigned_set.h"
#include <algorithm>
#include <cassert>
#include <string>

namespace base {
namespace ext {
void Map::InitSize(int _xsize, int _ysize) {
  MapCore::InitSize(_xsize, ysize);
  obstacles.clear();
  unsigned size = Size();
  obstacles.resize(size);
  unwrapped.Resize(size);
}

void Map::InitMap(const std::string& desc) {
  int xs = 0, ys = 0;
  std::vector<Point> v;
  std::vector<std::vector<int>> vvy;
  for (const std::string& scontour : Split(desc, ';')) {
    v.clear();
    auto vst = Split(scontour, ',');
    Assert((vst.size() % 2) == 0);
    std::vector<std::string> vs;
    for (unsigned i = 0; i < vst.size(); i += 2)
      vs.emplace_back(vst[i] + "," + vst[i + 1]);
    for (const std::string& st : vs) {
      Point p = DecodePoint(st);
      xs = std::max(p.x, xs);
      ys = std::max(p.y, ys);
      v.emplace_back(p);
    }
    Assert(v.size() > 1);
    v.push_back(v[0]);
    if (vvy.size() < xsize) vvy.resize(xsize);
    for (size_t i = 1; i < v.size(); ++i) {
      if (v[i - 1].y == v[i].y) {
        int x1 = std::min(v[i - 1].x, v[i].x);
        int x2 = std::max(v[i - 1].x, v[i].x);
        for (int x = x1; x < x2; ++x) vvy[x].push_back(v[i].y);
      } else {
        Assert(v[i - 1].x == v[i].x);
      }
    }
  }
  InitSize(xs, ys);
  for (int x = 0; x < xsize; ++x) {
    auto& vy = vvy[x];
    std::sort(vy.begin(), vy.end());
    Assert((vy.size() % 2) == 0);
    vy.push_back(ysize);
    vy.push_back(ysize);
    int y = 0;
    for (unsigned i = 0; i < vy.size(); i += 2) {
      for (; y < vy[i]; ++y) obstacles[x * ysize + y] = true;
      y = vy[i + 1];
    }
  }
}

void Map::AddBooster(const Point& p, BoosterType type) {
  Assert(Inside(p));
  unsigned index = Index(p);
  Assert(boosters.find(index) == boosters.end());
  boosters[index] = type;
}

void Map::Init(const std::string& problem) {
  auto vs = Split(problem, '#');
  Assert(vs.size() == 4);
  InitMap(vs[2].empty() ? vs[0] : vs[0] + ";" + vs[2]);
  boosters.clear();
  beacons.clear();
  codex.clear();
  for (auto& boost_desc : Split(vs[3], ';')) {
    Assert(boost_desc.size() >= 2);
    Point p = DecodePoint(boost_desc.substr(1));
    switch (boost_desc[0]) {
      case 'B':
        AddBooster(p, BoosterType::EXTENSION);
        break;
      case 'F':
        AddBooster(p, BoosterType::FAST_WHEELS);
        break;
      case 'L':
        AddBooster(p, BoosterType::DRILL);
        break;
      case 'X':
        codex.insert(Index(p));
        break;
      case 'R':
        AddBooster(p, BoosterType::TELEPORT);
        break;
      case 'C':
        AddBooster(p, BoosterType::CLONING);
        break;
      default:
        Assert(false, "Unknown item the in problem description.");
    }
  }
}

BoosterType Map::PickupItem(const Point& p) {
  assert(Inside(p));
  auto it = boosters.find(Index(p));
  if (it == boosters.end()) return BoosterType::NONE;
  auto type = it->second;
  boosters.erase(it);
  return type;
}

bool Map::Obstacle(const Point& p) const {
  return !Inside(p) || obstacles[Index(p)];
}

void Map::Wrap(const Point& p) {
  assert(Inside(p));
  unwrapped.Remove(Index(p));
}

void Map::Drill(const Point& p) {
  assert(Inside(p));
  obstacles[Index(p)] = false;
}

void Map::AddBeacon(const Point& p) {
  assert(Inside(p));
  beacons.insert(Index(p));
}

bool Map::CheckBeacon(const Point& p) const {
  assert(Inside(p));
  return beacons.find(Index(p)) != beacons.end();
}

bool Map::CheckCodeX(const Point& p) const {
  assert(Inside(p));
  return codex.find(Index(p)) != codex.end();
}

bool Map::Wrapped() const { return unwrapped.Empty(); }
}  // namespace ext
}  // namespace base