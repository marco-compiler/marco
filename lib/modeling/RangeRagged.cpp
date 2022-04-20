#include "marco/modeling/RangeRagged.h"
#include "marco/utils/IRange.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace llvm;
using namespace std;

namespace marco::modeling
{

  template<typename Ragged, typename Foo>
  bool visitorAny(const Ragged& a, const Ragged& b, Foo foo)
  {
    if (a.isRagged()) {
      if (b.isRagged()) {
        assert(a.asRagged().size() == b.asRagged().size());
        for (auto [aa, bb] : llvm::zip(a.asRagged(), b.asRagged())) {
          if (visitorAny(aa, bb, foo)) return true;
        }
      } else
        for (auto it : a.asRagged())
          if (visitorAny(it, b, foo)) return true;
    } else if (b.isRagged()) {
      for (auto it : b.asRagged())
        if (visitorAny(a, it, foo)) return true;
    } else {
      return foo(a.asValue(), b.asValue());
    }
    return false;
  }

  template<typename Ragged, typename Foo>
  bool visitorAll(const Ragged& a, const Ragged& b, Foo foo)
  {
    if (a.isRagged()) {
      if (b.isRagged()) {
        assert(a.asRagged().size() == b.asRagged().size());
        for (auto [aa, bb] : llvm::zip(a.asRagged(), b.asRagged())) {
          if (!visitorAll(aa, bb, foo)) return false;
        }
      } else
        for (auto it : a.asRagged())
          if (!visitorAll(it, b, foo)) return false;
    } else if (b.isRagged()) {
      for (auto it : b.asRagged())
        if (!visitorAll(a, it, foo)) return false;
    } else {
      return foo(a.asValue(), b.asValue());
    }

    return true;
  }

  template<typename Ragged, typename Foo>
  Ragged visitorConstruct(const Ragged& a, const Ragged& b, Foo foo)
  {
    if (a.isRagged()) {
      llvm::SmallVector<Ragged, 2> values;
      if (b.isRagged()) {
        assert(a.asRagged().size() == b.asRagged().size());
        for (auto [aa, bb] : llvm::zip(a.asRagged(), b.asRagged())) {
          values.push_back(visitorConstruct(aa, bb, foo));
        }
      } else
        for (auto it : a.asRagged())
          values.push_back(visitorConstruct(it, b, foo));
      return Ragged(values);
    } else if (b.isRagged()) {
      llvm::SmallVector<Ragged, 2> values;
      for (auto it : b.asRagged())
        values.push_back(visitorConstruct(a, it, foo));
      return Ragged(values);
    } else {
      return foo(a.asValue(), b.asValue());
    }
  }

  RaggedValue RaggedValue::operator*(const RaggedValue& other) const
  {
    return visitorConstruct(*this, other, [](RaggedValue::data_type a, RaggedValue::data_type b) {
      return a * b;
    });
  }
  RaggedValue RaggedValue::operator+(const RaggedValue& other) const
  {
    return visitorConstruct(*this, other, [](RaggedValue::data_type a, RaggedValue::data_type b) {
      return a + b;
    });
  }
  RaggedValue RaggedValue::operator-(const RaggedValue& other) const
  {
    return visitorConstruct(*this, other, [](RaggedValue::data_type a, RaggedValue::data_type b) {
      return a - b;
    });
  }
  RaggedValue RaggedValue::operator/(const RaggedValue& other) const
  {
    return visitorConstruct(*this, other, [](RaggedValue::data_type a, RaggedValue::data_type b) {
      return a / b;
    });
  }

  RaggedValue& RaggedValue::operator+=(const RaggedValue& other)
  {
    return *this = *this + other;
  }
  RaggedValue& RaggedValue::operator*=(const RaggedValue& other)
  {
    return *this = *this * other;
  }
  RaggedValue& RaggedValue::operator-=(const RaggedValue& other)
  {
    return *this = *this - other;
  }
  RaggedValue& RaggedValue::operator/=(const RaggedValue& other)
  {
    return *this = *this / other;
  }

  RaggedValue::data_type RaggedValue::min() const
  {
    if (!isRagged()) return asValue();
    auto v = asRagged();
    data_type m = v[0].min();
    for (auto i : irange((size_t) 1, v.size()))
      m = std::min(m, v[i].min());
    return m;
  }
  RaggedValue::data_type RaggedValue::max() const
  {
    if (!isRagged()) return asValue();
    auto v = asRagged();
    data_type m = v[0].max();
    for (auto i : irange((size_t) 1, v.size()))
      m = std::max(m, v[i].max());
    return m;
  }

  RaggedValue& RaggedValue::operator++()
  {
    if (isRagged())
      for (auto& it : *std::get<RaggedPtr>(value))
        ++it;
    else
      ++std::get<data_type>(value);
    return *this;
  }

  RaggedValue RaggedValue::operator++(int)
  {
    auto temp = *this;
    ++(*this);
    return temp;
  }

  bool RaggedValue::operator==(const RaggedValue& other) const
  {
    if (isRagged()) {
      if (other.isRagged()) {
        auto a = asRagged();
        auto b = other.asRagged();
        return a == b;
      }
      return false;
    }
    if (other.isRagged()) return false;
    return asValue() == other.asValue();
  }
  bool RaggedValue::operator!=(const RaggedValue& other) const
  {
    return !(*this == other);
  }
  bool RaggedValue::operator>(const RaggedValue& other) const
  {
    return visitorAll(*this, other, [](const RaggedValue::data_type& a, const RaggedValue::data_type& b) {
      return a > b;
    });
  }
  bool RaggedValue::operator<(const RaggedValue& other) const
  {
    return visitorAll(*this, other, [](const RaggedValue::data_type& a, const RaggedValue::data_type& b) {
      return a < b;
    });
  }

  bool RaggedValue::operator>=(const RaggedValue& other) const
  {
    return visitorAll(*this, other, [](const RaggedValue::data_type& a, const RaggedValue::data_type& b) {
      return a >= b;
    });
  }

  bool RaggedValue::operator<=(const RaggedValue& other) const
  {
    return visitorAll(*this, other, [](const RaggedValue::data_type& a, const RaggedValue::data_type& b) {
      return a <= b;
    });
  }

  RangeRagged::RangeRagged(const RaggedValue& a, const RaggedValue& b) : value(Range(0, 1))
  {

    if (a.isRagged()) {
      llvm::SmallVector<RangeRagged, 2> values;
      if (b.isRagged()) {
        assert(a.asRagged().size() == b.asRagged().size());
        for (auto [aa, bb] : llvm::zip(a.asRagged(), b.asRagged())) {
          values.push_back(RangeRagged(aa, bb));
        }
      } else
        for (auto it : a.asRagged())
          values.push_back(RangeRagged(it, b));
      value = std::make_unique<Ragged>(values);
    } else if (b.isRagged()) {
      llvm::SmallVector<RangeRagged, 2> values;
      for (auto it : b.asRagged())
        values.push_back(RangeRagged(a, it));
      value = std::make_unique<Ragged>(values);
    } else {
      value = Range(a.asValue(), b.asValue());
      return;
    }
    compact();
  }

  RaggedValue min(const RaggedValue& a, const RaggedValue& b)
  {
    assert(a.isRagged() == b.isRagged());

    if (a.isRagged()) {
      assert(a.asRagged().size() == b.asRagged().size());

      llvm::SmallVector<RaggedValue, 2> values;
      for (auto [m, M] : llvm::zip(a.asRagged(), b.asRagged())) {
        values.emplace_back(min(m, M));
      }
      return RaggedValue(values);
    }
    return std::min(a.asValue(), b.asValue());
  }

  RaggedValue max(const RaggedValue& a, const RaggedValue& b)
  {
    assert(a.isRagged() == b.isRagged());

    if (a.isRagged()) {
      assert(a.asRagged().size() == b.asRagged().size());

      llvm::SmallVector<RaggedValue, 2> values;
      for (auto [m, M] : llvm::zip(a.asRagged(), b.asRagged())) {
        values.emplace_back(max(m, M));
      }
      return RaggedValue(values);
    }
    return std::max(a.asValue(), b.asValue());
  }

  long minMin(const RangeRagged& interval)
  {
    if (interval.isRagged()) {
      auto r = interval.asRagged();
      auto min = minMin(*r.begin());
      for (auto it = next(r.begin()); it != r.end(); it++)
        min = std::min(min, minMin(*it));
      return min;
    }
    return interval.asValue().getBegin();
  }

  long maxMax(const RangeRagged& interval)
  {
    if (interval.isRagged()) {
      auto r = interval.asRagged();
      auto max = maxMax(*r.begin());
      for (auto it = next(r.begin()); it != r.end(); it++)
        max = std::max(max, maxMax(*it));
      return max;
    }
    return interval.asValue().getEnd();
  }

  long RangeRagged::min() const { return minMin(*this); }
  long RangeRagged::max() const { return maxMax(*this); }

  Range RangeRagged::getContainingRange() const
  {
    return Range(min(),max());
  }


  RaggedValue RangeRagged::getBegin() const
  {
    if (isRagged()) {
      llvm::SmallVector<RaggedValue, 2> values;
      for (auto r : asRagged())
        values.push_back(r.getBegin());
      return RaggedValue(values);
    }
    return asValue().getBegin();
  }
  RaggedValue RangeRagged::getEnd() const
  {
    if (isRagged()) {
      llvm::SmallVector<RaggedValue, 2> values;
      for (auto r : asRagged())
        values.push_back(r.getEnd());
      return RaggedValue(values);
    }
    return asValue().getEnd();
  }

  size_t RangeRagged::size() const
  {
    assert(!isRagged());

    return asValue().getEnd() - asValue().getBegin();
  }

  RaggedValue RangeRagged::getSize() const
  {
    if (isRagged()) {
      llvm::SmallVector<RaggedValue, 2> values;
      for (auto r : asRagged())
        values.push_back(r.getSize());
      return RaggedValue(values);
    }
    return size();
  }

  void RangeRagged::compact()
  {
    if (!isRagged()) return;

    auto r = asRagged();

    assert(r.size());

    if (r.size() == 1) {
      *this = r[0];
      compact();
      return;
    }

    if (std::equal(r.begin() + 1, r.end(), r.begin())) {
      // check if all ragged dimensions are equals
      // e.g. {3,3,3} -> 3
      // multidimensionally: [2]{2,3}{{3,3},{2,2,2}} == [2]{2,3}{3,2}
      *this = *r.begin();
      compact();
      return;
    }
  }

  bool RangeRagged::contains(const RangeRagged& other) const
  {
    // if(isRagged() && other.isRagged()){
    // 	auto C = asRagged();
    // 	auto c = other.asRagged();
    // 	assert(C.size()==c.size());
    // 	for(auto [it_C,it_c]: llvm::zip(C,c)){
    // 		if(!it_C.contains(it_c))return false;
    // 	}
    // 	return true;
    // }
    // return getBegin() <= other.getBegin() && getEnd() >= other.getEnd();

    return visitorAll(*this, other, [](const Range& a, const Range& b) {
      return a.contains(b);
    });
  }

  bool RangeRagged::isContained(const RangeRagged& other) const
  {
    return other.contains(*this);
  }

  bool RangeRagged::isFullyContained(const RangeRagged& other) const
  {
    if (isRagged() && other.isRagged()) {
      auto C = asRagged();
      auto c = other.asRagged();
      assert(C.size() == c.size());
      for (auto [it_C, it_c] : llvm::zip(C, c)) {
        if (!it_C.isFullyContained(it_c)) return false;
      }
      return true;
    }
    return getBegin() >= other.getBegin() && getEnd() <= other.getEnd() && getBegin() < getEnd();
  }

  // void RangeRagged::dump(llvm::raw_ostream& OS) const
  // {
  // 	OS << toString(*this);
  // }

  bool areDisjoint(const RangeRagged& left, const RangeRagged& right)
  {
    // has no meaning without the context of the previous dimensions.
    // assert( !(left.isRagged() && right.isRagged()) );

    return left.getBegin() >= right.getEnd() || left.getEnd() <= right.getBegin();
  }

  bool RangeRagged::overlaps(const RangeRagged& other) const
  {
    return visitorAny(*this, other, [](const Range& a, const Range& b) {
      return a.overlaps(b);
    });
  }

  RangeRagged RangeRagged::intersect(const RangeRagged& other) const
  {
    return visitorConstruct(*this, other, [](const Range& a, const Range& b) {
      return a.intersect(b);
    });
  }

  /// Check whether the range can be merged with another one.
  /// Two ranges can be merged if they overlap or if they are contiguous.
  bool RangeRagged::canBeMerged(const RangeRagged& other) const
  {
    return visitorAll(*this, other, [](const Range& a, const Range& b) {
      return a.canBeMerged(b);
    });
  }

  /// Create a range that is the resulting of merging this one with
  /// another one that can be merged.
  RangeRagged RangeRagged::merge(const RangeRagged& other) const
  {
    return visitorConstruct(*this, other, [](const Range& a, const Range& b) {
      return a.merge(b);
    });
  }

  /// Subtract a range from the current one.
  /// Multiple results are created if the removed range is fully contained
  /// and does not touch the borders.
  std::vector<RangeRagged> RangeRagged::subtract(const RangeRagged& other) const
  {
    std::vector<RangeRagged> results;

    if (!overlaps(other)) {
      results.push_back(*this);
    } else if (contains(other)) {
      if (getBegin() != other.getBegin()) {
        results.emplace_back(getBegin(), other.getBegin());
      }

      if (getEnd() != other.getEnd()) {
        results.emplace_back(other.getEnd(), getEnd());
      }
    } else if (!other.contains(*this)) {
      if (getBegin() <= other.getBegin()) {
        results.emplace_back(getBegin(), other.getBegin());
      } else {
        results.emplace_back(other.getEnd(), getEnd());
      }
    }

    return results;

    // return std::vector<RangeRagged>();
  }

  std::string toString(const RaggedValue &value)
  {
    if(value.isRagged())
    {
      std::string separator;
      std::string s = "{";
      for(auto r: value.asRagged())
      {
        s += separator + toString(r);
        separator=", ";
      }
      return s + "}";
    }
    return std::to_string(value.asValue());
  }

  std::ostream& operator<<(std::ostream& stream, const RaggedValue &value)
  {
    return stream<<toString(value);
  }


  std::string toString(const RangeRagged& value)
  {
    if (value.isRagged()) {
      std::string out = "{";
      std::string pre = "";
      for (auto r : value.asRagged()) {
        out += pre + toString(r);
        pre = ", ";
      }
      out += "}";
      return out;
    }

    auto min = value.asValue().getBegin();
    auto max = value.asValue().getEnd();

    return "[" + std::to_string(min) + "," + std::to_string(max) + "]";
  }



}// namespace marco::modeling