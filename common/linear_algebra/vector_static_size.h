#pragma once

#include <array>

template <class TTValue, unsigned _size>
class VectorStaticSize {
 public:
  static const unsigned size = _size;

  using TValue = TTValue;
  using TData = std::array<TValue, size>;
  using TSelf = VectorStaticSize<TValue, size>;

  using iterator = TValue*;
  using const_iterator = const TValue*;

 protected:
  TData data;

 public:
  void Fill(const TValue& v) { data.fill(v); }
  void Clear() { Fill(TValue(0)); }
  unsigned Size() const { return size; }
  const TData& GetData() const { return data; }

  VectorStaticSize() { Clear(); }
  VectorStaticSize(const TValue& v) { Fill(v); }
  TSelf& operator=(const TValue& v) {
    Fill(v);
    return *this;
  }

  TValue& operator()(unsigned i) { return data[i]; }
  const TValue& operator()(unsigned i) const { return data[i]; }
  iterator begin() { return &data.front(); }
  const_iterator begin() const { return &data.front(); }
  iterator end() { return begin() + size; }
  const_iterator end() const { return begin() + size; }
  iterator GetP(unsigned i) { return begin() + i; }
  const_iterator GetP(unsigned i) const { return begin() + i; }
  void swap(TSelf& r) { data.swap(r.data); }

  TSelf& operator+=(const TValue& v) {
    for (iterator p = begin(), pend = end(); p < pend;) *p++ += v;
    return *this;
  }

  TSelf& operator-=(const TValue& v) {
    for (iterator p = begin(), pend = end(); p < pend;) *p++ -= v;
    return *this;
  }

  TSelf& operator*=(const TValue& v) {
    for (iterator p = begin(), pend = end(); p < pend;) *p++ *= v;
    return *this;
  }

  TSelf& operator/=(const TValue& v) {
    for (iterator p = begin(), pend = end(); p < pend;) *p++ /= v;
    return *this;
  }

  TSelf operator+(const TValue& v) const {
    TSelf t(*this);
    t += v;
    return t;
  }

  TSelf operator-(const TValue& v) const {
    TSelf t(*this);
    t -= v;
    return t;
  }

  TSelf operator*(const TValue& v) const {
    TSelf t(*this);
    t *= v;
    return t;
  }

  TSelf operator/(const TValue& v) const {
    TSelf t(*this);
    t /= v;
    return t;
  }

  TSelf operator-() const {
    TSelf t(*this);
    for (iterator p = begin(), pend = end(); p < pend; ++p) *p = -*p;
    return t;
  }

  TSelf& operator+=(const TSelf& v) {
    const_iterator pv = v.begin();
    for (iterator p = begin(), pend = end(); p < pend;) *p++ += *pv++;
    return *this;
  }

  TSelf& operator-=(const TSelf& v) {
    const_iterator pv = v.begin();
    for (iterator p = begin(), pend = end(); p < pend;) *p++ -= *pv++;
    return *this;
  }

  TSelf operator+(const TSelf& v) const {
    TSelf t(*this);
    t += v;
    return t;
  }

  TSelf operator-(const TSelf& v) const {
    TSelf t(*this);
    t -= v;
    return t;
  }
};

template <unsigned size>
using TDVectorStaticSize = VectorStaticSize<double, size>;
