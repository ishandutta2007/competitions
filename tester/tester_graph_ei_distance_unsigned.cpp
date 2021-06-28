#include "tester/tester_graph_ei_distance_unsigned.h"

#include "tester/graph_ei_distance_positive_cost.h"
#include "tester/graph_type.h"

#include "common/graph/graph_ei/create_hrandom_graph.h"
#include "common/graph/graph_ei/distance/bellman_ford.h"
#include "common/graph/graph_ei/distance/bellman_ford_yen.h"
#include "common/graph/graph_ei/distance/dial.h"
#include "common/graph/graph_ei/distance/floyd_warshall.h"
#include "common/graph/graph_ei/distance/johnson.h"
#include "common/graph/graph_ei/distance/spfa/goldberg_radzik.h"
#include "common/graph/graph_ei/distance/spfa/goldberg_radzik_lazy.h"
#include "common/graph/graph_ei/distance/spfa/goldberg_radzik_pcr.h"
#include "common/graph/graph_ei/distance/spfa/goldfarb_hao_kai.h"
#include "common/graph/graph_ei/distance/spfa/large_label_last.h"
#include "common/graph/graph_ei/distance/spfa/levit.h"
#include "common/graph/graph_ei/distance/spfa/pallottino.h"
#include "common/graph/graph_ei/distance/spfa/safe.h"
#include "common/graph/graph_ei/distance/spfa/sipitq.h"
#include "common/graph/graph_ei/distance/spfa/small_label_first.h"
#include "common/graph/graph_ei/distance/spfa/spfa.h"
#include "common/graph/graph_ei/distance/spfa/tarjan.h"
#include "common/graph/graph_ei/distance/spfa/tarjan_pcr.h"
#include "common/graph/graph_ei/distance/spfa/tarjan_pcr_time.h"
#include "common/graph/graph_ei/distance/spfa/tarjan_time.h"
#include "common/graph/graph_ei/distance/spfa/zero_degrees_only.h"
#include "common/graph/graph_ei/distance/spfa/zero_degrees_only_base.h"
#include "common/graph/graph_ei/distance/spfa/zero_degrees_only_time.h"
#include "common/graph/graph_ei/edge_cost_proxy.h"
#include "common/hash.h"
#include "common/heap/base/binary.h"
#include "common/heap/base/bucket_queue.h"
#include "common/heap/base/bucket_queue_ll.h"
#include "common/heap/base/dheap.h"
#include "common/heap/ext/dheap_ukey_pos_map.h"
#include "common/heap/monotone/base/bucket_queue.h"
#include "common/heap/monotone/base/bucket_queue_ll.h"
#include "common/heap/monotone/base/rolling_bucket_queue.h"
#include "common/heap/monotone/base/rolling_bucket_queue_ll.h"
#include "common/heap/monotone/base/two_layers_buckets.h"
#include "common/heap/monotone/base/two_layers_buckets_sqrt.h"
#include "common/heap/monotone/ukvm/bucket_queue.h"
#include "common/heap/monotone/ukvm/bucket_queue_dll.h"
#include "common/heap/monotone/ukvm/radix.h"
#include "common/heap/monotone/ukvm/radix_dll.h"
#include "common/heap/monotone/ukvm/rolling_bucket_queue.h"
#include "common/heap/monotone/ukvm/rolling_bucket_queue_dll.h"
#include "common/heap/monotone/ukvm/two_layers_buckets.h"
#include "common/heap/monotone/ukvm/two_layers_buckets_dll.h"
#include "common/heap/monotone/ukvm/two_layers_buckets_sqrt.h"
#include "common/heap/monotone/ukvm/two_layers_buckets_sqrt_dll.h"
#include "common/heap/ukvm/binomial.h"
#include "common/heap/ukvm/bucket_queue.h"
#include "common/heap/ukvm/bucket_queue_dll.h"
#include "common/heap/ukvm/complete_binary_tree.h"
#include "common/heap/ukvm/dheap.h"
#include "common/heap/ukvm/fibonacci.h"
#include "common/heap/ukvm/pairing.h"
#include "common/heap/ukvm/unordered_set.h"
#include "common/timer.h"

#include <functional>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

template <class TData>
using TBinaryHeap = heap::base::Binary<TData>;
template <class TData>
using TDHeap2 = heap::base::DHeap<2u, TData>;
template <class TData>
using TDHeap4 = heap::base::DHeap<4u, TData>;
template <class TData>
using TDHeap8 = heap::base::DHeap<8u, TData>;
template <class TData>
using TDHeap16 = heap::base::DHeap<16u, TData>;
template <bool multipass, bool auxiliary>
using TPairing =
    heap::ukvm::Pairing<unsigned, std::less<unsigned>, multipass, auxiliary>;

TesterGraphEIDistanceUnsigned::TesterGraphEIDistanceUnsigned(
    EGraphType _gtype, unsigned graph_size, unsigned edges_per_node,
    unsigned _max_edge_cost)
    : max_edge_cost(_max_edge_cost),
      gtype(_gtype),
      g(CreateHRandomGraph<unsigned, true>(graph_size, edges_per_node,
                                           max_edge_cost)) {}

size_t TesterGraphEIDistanceUnsigned::TestBellmanFord() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::BellmanFord(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [   BF  ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestBellmanFordYen() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::BellmanFordYen(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [   BFY ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestDial() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::Dial(g, edge_proxy, i, max_cost, max_edge_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [   Dial]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestFloydWarshall() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  auto vv = graph::distance::FloydWarshall(g, edge_proxy, max_cost);
  for (unsigned i = 0; i < vv.size(); ++i) {
    for (uint64_t d : vv[i]) h = HashCombine(h, d);
  }
  std::cout << "Test results  [A  FlWa]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestGoldbergRadzik() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::GoldbergRadzik(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  GR  ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestGoldbergRadzikLazy() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::GoldbergRadzikLazy(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  GRL ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestGoldbergRadzikPCR() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::GoldbergRadzikPCR(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  GRP ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestGoldfarbHaoKai() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::GoldfarbHaoKai(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  GHK ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestJohnson() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  auto vv = graph::distance::Johnson(g, edge_proxy, max_cost);
  for (unsigned i = 0; i < vv.size(); ++i) {
    for (uint64_t d : vv[i]) h = HashCombine(h, d);
  }
  std::cout << "Test results  [A  John]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestLevit() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::Levit(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  Levi]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestPallottino() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::Pallottino(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  Pall]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestSIPITQ() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::SIPITQ(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S   SPQ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestSPFA() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::SPFA(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  SPFA]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestSPFALLL() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::LargeLabelLast(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S   LLL]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestSPFASafe() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::Safe(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  Safe]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestSPFASLF() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::SmallLabelFirst(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S   SLF]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestTarjan() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::Tarjan(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  Ta  ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestTarjanPCR() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::TarjanPCR(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  TaP ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestTarjanPCRTime() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::TarjanPCRTime(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  TaPT]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestTarjanTime() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::TarjanTime(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  Ta T]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestZDO() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::ZeroDegreesOnly(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  ZDO ]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestZDOBase() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::ZeroDegreesOnlyBase(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  ZDOB]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

size_t TesterGraphEIDistanceUnsigned::TestZDOTime() const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = graph::distance::spfa::ZeroDegreesOnlyTime(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [S  ZDOT]: " << h << "\t" << t.GetMilliseconds()
            << std::endl;
  return h;
}

template <template <class TData> class THeap>
size_t TesterGraphEIDistanceUnsigned::TestDBH(const std::string& name) const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = DistanceFromSourcePositiveCost_DBH<THeap>(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [DB" << name << "]: " << h << "\t"
            << t.GetMilliseconds() << std::endl;
  return h;
}

template <class THeap>
size_t TesterGraphEIDistanceUnsigned::TestDBHP(const std::string& name) const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = DistanceFromSourcePositiveCost_DBHP<THeap>(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [DB" << name << "]: " << h << "\t"
            << t.GetMilliseconds() << std::endl;
  return h;
}

template <class THeap>
size_t TesterGraphEIDistanceUnsigned::TestDBHPW(const std::string& name) const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = DistanceFromSourcePositiveCost_DBHPW<THeap>(g, edge_proxy, i,
                                                    max_edge_cost, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [DB" << name << "]: " << h << "\t"
            << t.GetMilliseconds() << std::endl;
  return h;
}

template <class THeap>
size_t TesterGraphEIDistanceUnsigned::TestDEH(const std::string& name) const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = DistanceFromSourcePositiveCost_DEH<THeap>(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [DE" << name << "]: " << h << "\t"
            << t.GetMilliseconds() << std::endl;
  return h;
}

template <class THeap>
size_t TesterGraphEIDistanceUnsigned::TestDMH(const std::string& name) const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = DistanceFromSourcePositiveCost_DMH<THeap>(g, edge_proxy, i, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [DM" << name << "]: " << h << "\t"
            << t.GetMilliseconds() << std::endl;
  return h;
}

template <class THeap>
size_t TesterGraphEIDistanceUnsigned::TestDMHW(const std::string& name) const {
  Timer t;
  size_t h = 0;
  unsigned max_cost = -1u;
  std::vector<unsigned> v;
  for (unsigned i = 0; i < g.Size(); ++i) {
    v = DistanceFromSourcePositiveCost_DMHW<THeap>(g, edge_proxy, i,
                                                   max_edge_cost, max_cost);
    for (unsigned d : v) h = HashCombine(h, d);
  }
  std::cout << "Test results  [DM" << name << "]: " << h << "\t"
            << t.GetMilliseconds() << std::endl;
  return h;
}

bool TesterGraphEIDistanceUnsigned::TestAll() {
  switch (gtype) {
    case EGraphType::SMALL:
      std::cout << "Small " << max_edge_cost << ":" << std::endl;
      break;
    case EGraphType::SPARSE:
      std::cout << "Sparse " << max_edge_cost << ":" << std::endl;
      break;
    case EGraphType::DENSE:
      std::cout << "Dense " << max_edge_cost << ":" << std::endl;
      break;
    default:
      assert(false);
  }
  std::unordered_set<size_t> hs;
  hs.insert(TestDBH<TBinaryHeap>("  BH "));
  hs.insert(TestDBH<TDHeap2>(" DH1 "));
  hs.insert(TestDBH<TDHeap4>(" DH2 "));
  hs.insert(TestDBH<TDHeap8>(" DH3 "));
  hs.insert(TestDBH<TDHeap16>(" DH4 "));
  hs.insert(TestDBHP<heap::base::BucketQueue<unsigned>>(" BQ  "));
  hs.insert(TestDBHP<heap::base::BucketQueueLL<unsigned>>(" BQ L"));
  hs.insert(TestDBHP<heap::monotone::base::BucketQueue<unsigned>>("MBQ  "));
  hs.insert(TestDBHP<heap::monotone::base::BucketQueueLL<unsigned>>("MBQ L"));
  hs.insert(
      TestDBHPW<heap::monotone::base::RollingBucketQueue<unsigned>>("MRQ  "));
  hs.insert(
      TestDBHPW<heap::monotone::base::RollingBucketQueueLL<unsigned>>("MRQ L"));
  hs.insert(
      TestDBHP<heap::monotone::base::TwoLayersBuckets<4, unsigned>>("MBQ2 "));
  hs.insert(
      TestDBHP<heap::monotone::base::TwoLayersBuckets<8, unsigned>>("MBQ3 "));
  hs.insert(
      TestDBHP<heap::monotone::base::TwoLayersBuckets<16, unsigned>>("MBQ4 "));
  hs.insert(
      TestDBHP<heap::monotone::base::TwoLayersBuckets<32, unsigned>>("MBQ5 "));
  hs.insert(
      TestDBHPW<heap::monotone::base::TwoLayersBucketsSqrt<unsigned>>("MBQA "));
  hs.insert(TestDEH<heap::ext::DHeapUKeyPosMap<2, unsigned>>(" DH1 "));
  hs.insert(TestDEH<heap::ext::DHeapUKeyPosMap<4, unsigned>>(" DH2 "));
  hs.insert(TestDEH<heap::ext::DHeapUKeyPosMap<8, unsigned>>(" DH3 "));
  hs.insert(TestDEH<heap::ext::DHeapUKeyPosMap<16, unsigned>>(" DH4 "));
  hs.insert(TestDMH<heap::ukvm::DHeap<2, unsigned>>(" DH1 "));
  hs.insert(TestDMH<heap::ukvm::DHeap<4, unsigned>>(" DH2 "));
  hs.insert(TestDMH<heap::ukvm::DHeap<8, unsigned>>(" DH3 "));
  hs.insert(TestDMH<heap::ukvm::DHeap<16, unsigned>>(" DH4 "));
  hs.insert(TestDMH<heap::ukvm::CompleteBinaryTree<unsigned>>(" CBT "));
  hs.insert(TestDMH<heap::ukvm::BucketQueue>(" BQ  "));
  hs.insert(TestDMH<heap::ukvm::BucketQueueDLL>(" BQ L"));
  hs.insert(TestDMH<heap::ukvm::Binomial<unsigned>>(" BNML"));
  hs.insert(TestDMH<heap::ukvm::Fibonacci<unsigned>>(" FBNC"));
  hs.insert(TestDMH<TPairing<0, 0>>(" PR00"));
  hs.insert(TestDMH<TPairing<1, 0>>(" PR01"));
  hs.insert(TestDMH<TPairing<0, 1>>(" PR10"));
  hs.insert(TestDMH<TPairing<1, 1>>(" PR11"));
  if (gtype != EGraphType::SPARSE)
    hs.insert(TestDMH<heap::ukvm::UnorderedSet<unsigned>>(" USet"));
  hs.insert(TestDMH<heap::monotone::ukvm::BucketQueue>("MBQ  "));
  hs.insert(TestDMH<heap::monotone::ukvm::BucketQueueDLL>("MBQ L"));
  hs.insert(TestDMHW<heap::monotone::ukvm::RollingBucketQueue>("MRQ  "));
  hs.insert(TestDMHW<heap::monotone::ukvm::RollingBucketQueueDLL>("MRQ L"));
  hs.insert(TestDMH<heap::monotone::ukvm::TwoLayersBuckets<4>>("MBQ2 "));
  hs.insert(TestDMH<heap::monotone::ukvm::TwoLayersBuckets<8>>("MBQ3 "));
  hs.insert(TestDMH<heap::monotone::ukvm::TwoLayersBuckets<16>>("MBQ4 "));
  hs.insert(TestDMH<heap::monotone::ukvm::TwoLayersBuckets<32>>("MBQ5 "));
  hs.insert(TestDMHW<heap::monotone::ukvm::TwoLayersBucketsSqrt>("MBQA "));
  hs.insert(TestDMH<heap::monotone::ukvm::TwoLayersBucketsDLL<4>>("MBQ2L"));
  hs.insert(TestDMH<heap::monotone::ukvm::TwoLayersBucketsDLL<8>>("MBQ3L"));
  hs.insert(TestDMH<heap::monotone::ukvm::TwoLayersBucketsDLL<16>>("MBQ4L"));
  hs.insert(TestDMH<heap::monotone::ukvm::TwoLayersBucketsDLL<32>>("MBQ5L"));
  hs.insert(TestDMHW<heap::monotone::ukvm::TwoLayersBucketsSqrtDLL>("MBQAL"));
  hs.insert(TestDMH<heap::monotone::ukvm::Radix>("MRDX "));
  hs.insert(TestDMH<heap::monotone::ukvm::RadixDLL>("MRDXL"));
  hs.insert(TestDial());
  if (gtype == EGraphType::SMALL) hs.insert(TestBellmanFord());
  if (gtype == EGraphType::SMALL) hs.insert(TestBellmanFordYen());
  hs.insert(TestSPFA());
  if (gtype == EGraphType::SMALL) hs.insert(TestSPFASafe());
  if (gtype == EGraphType::SMALL) hs.insert(TestSIPITQ());
  if (gtype == EGraphType::SMALL) hs.insert(TestSPFALLL());
  if (gtype == EGraphType::SMALL) hs.insert(TestSPFASLF());
  if (gtype == EGraphType::SMALL) hs.insert(TestLevit());
  if (gtype == EGraphType::SMALL) hs.insert(TestPallottino());
  if (gtype == EGraphType::SMALL) hs.insert(TestGoldfarbHaoKai());
  if (gtype == EGraphType::SMALL) hs.insert(TestGoldbergRadzik());
  if (gtype == EGraphType::SMALL) hs.insert(TestGoldbergRadzikLazy());
  if (gtype == EGraphType::SMALL) hs.insert(TestGoldbergRadzikPCR());
  if (gtype == EGraphType::SMALL) hs.insert(TestTarjan());
  if (gtype == EGraphType::SMALL) hs.insert(TestTarjanTime());
  if (gtype == EGraphType::SMALL) hs.insert(TestTarjanPCR());
  if (gtype == EGraphType::SMALL) hs.insert(TestTarjanPCRTime());
  if (gtype == EGraphType::SMALL) hs.insert(TestZDOBase());
  if (gtype == EGraphType::SMALL) hs.insert(TestZDO());
  if (gtype == EGraphType::SMALL) hs.insert(TestZDOTime());
  if (gtype != EGraphType::SPARSE) hs.insert(TestFloydWarshall());
  hs.insert(TestJohnson());
  return hs.size() == 1;
}

bool TestGraphEIDistanceUnsigned(bool time_test) {
  if (time_test) {
    bool ok = true;
    for (unsigned max_cost : {10, 100, 1000}) {
      TesterGraphEIDistanceUnsigned t1(EGraphType::SPARSE, 5000, 4, max_cost),
          t2(EGraphType::DENSE, 1000, 500, max_cost);
      ok = t1.TestAll() && ok;
      ok = t2.TestAll() && ok;
    }
    return ok;
  } else {
    TesterGraphEIDistanceUnsigned t(EGraphType::SMALL, 100, 4, 10);
    return t.TestAll();
  }
}
