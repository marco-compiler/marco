#include "marco/modeling/MultidimensionalRangeRagged.h"
#include "marco/utils/IRange.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace llvm;
using namespace std;

namespace marco::modeling
{
  
  using MultiIterator = MultidimensionalRangeRagged::MultidimensionalRangeRaggedIterator;

  MultidimensionalRangeRagged::MultidimensionalRangeRagged(llvm::ArrayRef<RangeRagged> ranges)
      : ranges(ranges.begin(), ranges.end())
  {}

  MultidimensionalRangeRagged::MultidimensionalRangeRagged(const MultidimensionalRange& range)
  {
    for (size_t i : irange(range.rank()))
      ranges.emplace_back(range[i]);
  }
  // MultidimensionalRangeRagged::MultidimensionalRangeRagged(Point point)
  // {
  // 	for (const auto& index : point) {
  // 		ranges.emplace_back(index, index + 1);
  // 	}
  // }

  bool MultidimensionalRangeRagged::isRagged() const
  {
    for (auto i : ranges)
      if (i.isRagged())
        return true;
    return false;
  }

  Range getCurrentRaggedInterval(llvm::ArrayRef<size_t> partialIndex, const RangeRagged& interval, size_t ragged_depth = 0)
  {
    if (!interval.isRagged()) return interval.asValue();

    assert(partialIndex.size());
    size_t index = partialIndex[partialIndex.size() - ragged_depth];

    return getCurrentRaggedInterval(partialIndex, interval.asRagged()[index], ragged_depth - 1);
  }

  bool MultidimensionalRangeRagged::contains(Point point) const
  {
    llvm::ArrayRef<long> point_arr(point.begin(), point.end());
    assert(point_arr.size() == rank());// NOLINT

    assert(point_arr.size()==ranges.size());

    std::vector<size_t> path;
    size_t ragged_depth = 0;

    for(auto [p,ragged]:llvm::zip(point_arr,ranges))
    {
      ragged_depth = ragged_depth ? ragged_depth + 1 : (ragged.isRagged() ? 1 : 0);
      auto interval = getCurrentRaggedInterval(path, ragged, ragged_depth);
      if(!interval.contains(p))
        return false;

      path.push_back(p - interval.getBegin());
    }
    return true;
  }

  bool containsHelper(llvm::ArrayRef<RangeRagged> containing, llvm::ArrayRef<RangeRagged> contained, std::vector<size_t> path = {}, size_t ragged_depth = 0)
  {
    if (containing.empty()) return true;

    const auto& containingInterval = containing[0];
    const auto& containedInterval = contained[0];

    ragged_depth = ragged_depth ? ragged_depth + 1 : (containingInterval.isRagged() || containedInterval.isRagged() ? 1 : 0);

    auto a = getCurrentRaggedInterval(path, containingInterval, ragged_depth);
    auto b = getCurrentRaggedInterval(path, containedInterval, ragged_depth);

    if (!a.contains(b))
      return false;

    //iterate the contained dimensions, since they should be less than equal than the containing ones
    for (auto it : irange(b.size())) {
      auto p = path;
      p.push_back(it);
      if (!containsHelper(containing.slice(1), contained.slice(1), p, ragged_depth))
        return false;
    }
    return true;
  }

  bool MultidimensionalRangeRagged::contains(const MultidimensionalRangeRagged& other) const
  {
    assert(rank() == other.rank());

    if (isRagged() || other.isRagged())
      return containsHelper(ranges, other.ranges);

    for (size_t i : irange(rank())) {
      auto a = ranges[i].asValue();
      auto b = other.ranges[i].asValue();
      if (not a.contains(b.getBegin()) or not a.contains(b.getEnd() - 1))
        return false;
    }
    return true;
  }

  bool areDisjointHelper(llvm::ArrayRef<RangeRagged> nextIntervals, llvm::ArrayRef<RangeRagged> otherIntervals, std::vector<size_t> point = {}, size_t ragged_depth = 0)
  {
    if (nextIntervals.empty()) return false;

    const auto& interval = nextIntervals[0];
    const auto& other = otherIntervals[0];

    ragged_depth = ragged_depth ? ragged_depth + 1 : (interval.isRagged() || other.isRagged() ? 1 : 0);

    auto a = getCurrentRaggedInterval(point, interval, ragged_depth);
    auto b = getCurrentRaggedInterval(point, other, ragged_depth);

    if (areDisjoint(a, b))
      return true;

    for (auto it : irange(std::max(a.size(), b.size()))) {
      auto p = point;
      p.push_back(it);
      if (!areDisjointHelper(nextIntervals.slice(1), otherIntervals.slice(1), p, ragged_depth))
        return false;
    }
    return true;
  }

  bool areDisjoint(
      const MultidimensionalRangeRagged& left, const MultidimensionalRangeRagged& right)
  {
    assert(left.rank() == right.rank());// NOLINT

    if (left.isRagged() || right.isRagged())
      return areDisjointHelper(left.ranges, right.ranges);

    for (size_t a = 0; a < left.rank(); a++)
      if (areDisjoint(left[a].asValue(), right[a].asValue()))
        return true;

    return false;
  }

  RangeRagged intersection(const RangeRagged& a, const RangeRagged& b)
  {
    const RangeRagged *ragged, *normal;
    if (a.isRagged()) {
      if (b.isRagged()) {
        auto ar = a.asRagged();
        auto br = b.asRagged();
        assert(ar.size() == br.size());

        llvm::SmallVector<RangeRagged, 2> ranges;
        for (auto [it_a, it_b] : llvm::zip(ar, br)) {
          ranges.push_back(intersection(it_a, it_b));
        }
        return RangeRagged(ranges);
      } else {
        ragged = &a;
        normal = &b;
      }
    } else {
      if (b.isRagged()) {
        ragged = &b;
        normal = &a;
      } else {
        return Range(std::max(a.asValue().getBegin(), b.asValue().getBegin()),
                     std::min(a.asValue().getEnd(), b.asValue().getEnd()));
      }
    }

    llvm::SmallVector<RangeRagged, 2> ranges;
    for (auto it : ragged->asRagged()) {
      ranges.push_back(intersection(it, *normal));
    }
    return RangeRagged(ranges);
  }

  RangeRagged removeDimensions(const RangeRagged& pre, const RangeRagged& post, const RangeRagged& dim)
  {
    if (!dim.isRagged()) return dim;
    if (pre == post) return dim;

    SmallVector<RangeRagged, 2> toReturn;
    auto dims = dim.asRagged();

    if (!pre.isRagged()) {
      assert(!post.isRagged());

      auto a = pre.asValue();
      auto b = post.asValue();
      auto start = b.getBegin() - a.getBegin();

      for (auto d : dims.slice(start, b.size()))
        toReturn.push_back(d);
    } else {
      assert(post.isRagged());
      auto pre_r = pre.asRagged();
      auto post_r = post.asRagged();
      assert(pre_r.size() == post_r.size());
      assert(pre_r.size() == dims.size());
      for (auto i : irange(pre_r.size())) {
        toReturn.push_back(removeDimensions(pre_r[i], post_r[i], dims[i]));
      }
    }

    if (toReturn.size() == 1) return toReturn[0];

    return RangeRagged(toReturn);
  }

  MultidimensionalRangeRagged intersection(
      const MultidimensionalRangeRagged& left, const MultidimensionalRangeRagged& right)
  {
    assert(!areDisjoint(left, right));  // NOLINT
    assert(left.rank() == right.rank());// NOLINT

    if (left.isRagged() || right.isRagged()) {

      llvm::SmallVector<RangeRagged, 2> a = left.ranges;
      llvm::SmallVector<RangeRagged, 2> b = right.ranges;

      size_t rank = left.rank();

      for (auto dim : irange(rank)) {

        auto c = intersection(a[dim], b[dim]);

        for (auto next : irange(dim + 1, rank)) {
          a[next] = removeDimensions(a[dim], c, a[next]);
          b[next] = removeDimensions(b[dim], c, b[next]);
        }

        a[dim] = c;
      }
      return MultidimensionalRangeRagged(std::move(a));
    }

    SmallVector<RangeRagged, 2> toReturn;

    for (size_t a = 0; a < left.rank(); a++) {
      toReturn.emplace_back(Range(
          std::max(left[a].asValue().getBegin(), right[a].asValue().getBegin()),
          std::min(left[a].asValue().getEnd(), right[a].asValue().getEnd())));
    }

    return MultidimensionalRangeRagged(move(toReturn));
  }

  pair<bool, size_t> MultidimensionalRangeRagged::canBeMerged(
      const MultidimensionalRangeRagged& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

    bool found = false;
    size_t dimension = 0;

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (const RangeRagged& first = ranges[i], second = other.ranges[i]; first != second) {
        if (first.canBeMerged(other.ranges[i])) {
          if (found) {
            return std::make_pair(false, 0);
          }

          found = true;
          dimension = i;
        } else {
          return std::make_pair(false, 0);
        }
      }
    }

    return std::make_pair(found, dimension);
  }

  unsigned int MultidimensionalRangeRagged::rank() const
  {
    return ranges.size();
  }

  RangeRagged& MultidimensionalRangeRagged::operator[](size_t index)
  {
    return ranges[index];
  }
  const RangeRagged& MultidimensionalRangeRagged::operator[](size_t index) const
  {
    return ranges[index];
  }

  MultidimensionalRangeRagged MultidimensionalRangeRagged::merge(const MultidimensionalRangeRagged& other, size_t dimension) const
  {
    assert(rank() == other.rank());
    llvm::SmallVector<RangeRagged, 3> mergedRanges;

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (i == dimension) {
        RangeRagged merged = ranges[i].merge(other.ranges[i]);
        mergedRanges.push_back(std::move(merged));
      } else {
        assert(ranges[i] == other.ranges[i]);
        mergedRanges.push_back(ranges[i]);
      }
    }

    return MultidimensionalRangeRagged(mergedRanges);
  }

  void splitRaggedIntervals(
      llvm::SmallVector<llvm::SmallVector<RangeRagged, 2>>& results,
      llvm::ArrayRef<RangeRagged> ranges)
  {
    //assuming results already of correct size
    for (auto i : ranges) {
      if (i.isRagged()) {
        auto rag = i.asRagged();

        assert(results.size() == rag.size());

        for (size_t index : irange(results.size()))
          results[index].push_back(rag[index]);
      } else {
        for (auto& res : results)
          res.push_back(i);
      }
    }
  }

  void toMultiDimIntervalsHelper(
      llvm::SmallVector<llvm::SmallVector<RangeRagged, 2>>& results,
      llvm::ArrayRef<RangeRagged> a)
  {
    for (auto index : irange((size_t) 1, std::max((size_t)1,a.size()))) {

      if (a[index].isRagged()) {
        auto pre = a[index - 1].asValue();
        auto rag = a[index].asRagged();

        assert(pre.size() == rag.size());

        llvm::SmallVector<llvm::SmallVector<RangeRagged, 2>> tmp(rag.size());
        splitRaggedIntervals(tmp, a.slice(index));

        for (auto i : irange(pre.size())) {
          llvm::SmallVector<llvm::SmallVector<RangeRagged, 2>> tmp2;
          toMultiDimIntervalsHelper(tmp2, tmp[i]);

          llvm::SmallVector<RangeRagged, 2> r(a.begin(), a.begin() + (index - 1));

          r.push_back(RangeRagged(Range(pre.getBegin() + i, pre.getBegin() + i + 1)));

          for (auto postfix : tmp2) {
            llvm::SmallVector<RangeRagged, 2> rr = r;
            rr.insert(rr.end(), postfix.begin(), postfix.end());
            results.push_back(rr);
          }
        }
        return;
      }
    }

    if (a.size())
      results.push_back(llvm::SmallVector<RangeRagged, 2>(a.begin(), a.end()));
  }

  llvm::SmallVector<MultidimensionalRange, 3> MultidimensionalRangeRagged::toMultidimensionalRanges() const
  {
    llvm::SmallVector<llvm::SmallVector<RangeRagged, 2>> results;
    toMultiDimIntervalsHelper(results, ranges);

    llvm::SmallVector<MultidimensionalRange, 3> res;

    for (auto r : results) {
      llvm::SmallVector<Range, 2> ranges;

      for (auto i : r)
        ranges.push_back(i.asValue());

      res.emplace_back(ranges);
    }
    return res;
  }

  MultidimensionalRange MultidimensionalRangeRagged::toMultidimensionalRange() const
  {
    auto ranges = toMultidimensionalRanges();
    assert(ranges.size() == 1);
    return ranges[0];
  }

  MultidimensionalRange MultidimensionalRangeRagged::toMinContainingMultidimensionalRange() const
  {
    llvm::SmallVector<Range, 3> res;

    for(auto r: ranges)
    {
      res.push_back(r.getContainingRange());
    }
    return MultidimensionalRange(res);
  }

  unsigned int MultidimensionalRangeRagged::flatSize() const
  {
    if (ranges.empty())
      return 0;

    size_t size = 0;

    //todo: optimize?
    for (auto it : toMultidimensionalRanges())
      size += it.flatSize();

    return size;
  }


  bool MultidimensionalRangeRagged::overlaps(const MultidimensionalRangeRagged& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

    for (const auto& [x, y] : llvm::zip(ranges, other.ranges)) {
      if (!x.overlaps(y)) {
        return false;
      }
    }

    return true;
  }

  MultidimensionalRangeRagged MultidimensionalRangeRagged::intersect(const MultidimensionalRangeRagged& other) const
  {
    return intersection(*this, other);
  }

  std::vector<MultidimensionalRangeRagged> MultidimensionalRangeRagged::subtract(const MultidimensionalRangeRagged& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");
    std::vector<MultidimensionalRangeRagged> results;

    if (!overlaps(other)) {
      results.push_back(*this);
    } else {
      llvm::SmallVector<RangeRagged, 3> resultRanges(ranges.begin(), ranges.end());

      for (size_t i = 0, e = rank(); i < e; ++i) {
        const auto& x = ranges[i];
        const auto& y = other.ranges[i];
        assert(x.overlaps(y));

        for (const auto& subRange : x.subtract(y)) {
          resultRanges[i] = std::move(subRange);
          results.emplace_back(resultRanges);
        }

        resultRanges[i] = x.intersect(y);
        llvm::errs() << toString(resultRanges[i]) << "\n";
      }
    }

    return results;
  }

  bool MultidimensionalRangeRagged::operator==(const Point& other) const
  {
    if (rank() != other.rank())
      return false;

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] != other[i]) {
        return false;
      }
    }

    return true;
  }
  bool MultidimensionalRangeRagged::operator==(const MultidimensionalRangeRagged& other) const
  {
    if (rank() != other.rank()) {
      return false;
    }

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] != other.ranges[i]) {
        return false;
      }
    }

    return true;
  }
  bool MultidimensionalRangeRagged::operator!=(const Point& other) const
  {
    if (rank() != other.rank())
      return true;

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] != other[i]) {
        return true;
      }
    }

    return false;
  }
  bool MultidimensionalRangeRagged::operator!=(const MultidimensionalRangeRagged& other) const
  {
    if (rank() != other.rank()) {
      return true;
    }

    for (const auto& [lhs, rhs] : llvm::zip(ranges, other.ranges)) {
      if (lhs != rhs) {
        return true;
      }
    }

    return false;
  }

  bool MultidimensionalRangeRagged::operator>(const MultidimensionalRangeRagged& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] > other.ranges[i]) {
        return true;
      }

      if (ranges[i] < other.ranges[i]) {
        return false;
      }
    }

    return false;
  }
  bool MultidimensionalRangeRagged::operator<(const MultidimensionalRangeRagged& other) const
  {
    assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

    for (size_t i = 0, e = rank(); i < e; ++i) {
      if (ranges[i] < other.ranges[i]) {
        return true;
      }

      if (ranges[i] > other.ranges[i]) {
        return false;
      }
    }

    return false;
  }

  std::string toString(const MultidimensionalRangeRagged& value)
  {
    std::string out;

    for (auto i : irange(value.rank()))
      out += toString(value[i]);

    return out;
  }

  MultidimensionalRangeRagged::const_iterator MultidimensionalRangeRagged::begin() const
  {
    return MultiIterator(*this);
  }
  
  MultidimensionalRangeRagged::const_iterator MultidimensionalRangeRagged::end() const
  {
    return MultiIterator(*this, true);
  }
  



  //iterator 
  size_t getRaggedDepth(size_t index, size_t ragged_start)
  {
    return std::max(0UL, index - ragged_start + 1UL);
  }


  MultiIterator::MultidimensionalRangeRaggedIterator(const MultidimensionalRangeRagged &container)
    :container(&container),indexes(container.rank(),0)
  {
    calculateRaggedStart();
    updatePoint();
  }

  MultiIterator::MultidimensionalRangeRaggedIterator(const MultidimensionalRangeRagged &container, bool end)
    :container(&container),end(true)
  {

  }

  bool MultiIterator::operator==(const MultiIterator& it) const
  {
    assert(container==it.container);
    return ((end && it.end) || point == it.point);
  }

  bool MultiIterator::operator!=(const MultiIterator& it) const
  {
    return !(*this==it);
  }

  MultiIterator& MultiIterator::operator++()
  {
    fetchNext();
    return *this;
  }

  MultiIterator MultiIterator::operator++(int)
  {
    auto temp = *this;
    fetchNext();
    return temp;
  }

  MultiIterator::value_type MultiIterator::operator*() const
  {
    return Point(point);
  }
  
  void MultiIterator::calculateRaggedStart()
  {
    raggedStart = indexes.size();
    for(size_t i=0;i<indexes.size();++i)
    { 
      if((*container)[i].isRagged())
      {
        raggedStart=i;
        return;
      }
    }
  }

  void MultiIterator::updatePoint()
  {
    point.resize(indexes.size());
    size_t ragged_depth=0;

    for(size_t i=0;i<indexes.size();++i)
    { 
      auto ragged = (*container)[i];
      ragged_depth = ragged_depth ? ragged_depth + 1 : (ragged.isRagged() ? 1 : 0);
      auto range = getCurrentRaggedInterval(llvm::ArrayRef<size_t>(indexes).slice(0,i), ragged, ragged_depth);
      point[i] = range.getBegin() + indexes[i];
    }
  }

  void MultiIterator::fetchNext()
  {
    if(end)return;
    
    bool update = false;

    for(size_t c = 1; c<=indexes.size(); ++c)
    { 
      size_t i = indexes.size() - c;

      auto &index = indexes[i];
      auto &coord = point[i];

      ++index;
      ++coord;

      auto range = getCurrentRaggedInterval(llvm::ArrayRef<size_t>(indexes).slice(0,i), (*container)[i], getRaggedDepth(i,raggedStart));
      if(range.contains(coord))
      {
        if(update)
          updatePoint();
        return;
      }
      update = true;
      index = 0;
    }
    end = true;
  }

  RangeRagged getRangeFromDimensionSize(const Shape::DimensionSize &dimension)
  {
    if(dimension.isRagged())
    {
      llvm::SmallVector<RangeRagged,3> arr;

      for(const auto r : dimension.asRagged()){
        arr.push_back(getRangeFromDimensionSize(r));
      }
      return RangeRagged(arr);
    }
    
    return RangeRagged(0,dimension.getNumericValue());
  }
  
  MultidimensionalRangeRagged getMultidimensionalRangeRaggedFromShape(const Shape& shape)
  {
    llvm::SmallVector<RangeRagged,3> ranges;
    for(auto dim: shape.dimensions())
    {
      ranges.push_back(getRangeFromDimensionSize(dim));
    }

    return MultidimensionalRangeRagged(std::move(ranges));
  }

  IndexSet getIndexSetFromRaggedRange(const MultidimensionalRangeRagged& range)
  {
    IndexSet set;
    auto ranges = range.toMultidimensionalRanges();

    for(auto r:ranges)
      set += r;

    return set;
  }

  IndexSet getIndexSetFromShape(const Shape& shape)
  {
    return getIndexSetFromRaggedRange(getMultidimensionalRangeRaggedFromShape(shape));
  }

}// namespace marco::modeling