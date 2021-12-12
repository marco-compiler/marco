#include <list>
#include <marco/matching/MultidimensionalRange.h>

using namespace marco::matching;

MultidimensionalRange::MultidimensionalRange(llvm::ArrayRef<Range> ranges)
        : ranges(ranges.begin(), ranges.end())
{
}

bool MultidimensionalRange::operator==(const MultidimensionalRange& other) const
{
  if (rank() != other.rank())
    return false;

  for (const auto& [lhs, rhs] : llvm::zip(ranges, other.ranges))
    if (lhs != rhs)
      return false;

  return true;
}

bool MultidimensionalRange::operator!=(const MultidimensionalRange& other) const
{
  if (rank() != other.rank())
    return true;

  for (const auto& [lhs, rhs] : llvm::zip(ranges, other.ranges))
    if (lhs != rhs)
      return true;

  return false;
}

bool MultidimensionalRange::operator<(const MultidimensionalRange& other) const
{
  assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

  for (size_t i = 0, e = rank(); i < e; ++i)
  {
    if (ranges[i] < other.ranges[i])
      return true;

    if (ranges[i] > other.ranges[i])
      return false;
  }

  return false;
}

bool MultidimensionalRange::operator>(const MultidimensionalRange& other) const
{
  assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

  for (size_t i = 0, e = rank(); i < e; ++i)
  {
    if (ranges[i] > other.ranges[i])
      return true;

    if (ranges[i] < other.ranges[i])
      return false;
  }

  return false;
}

Range& MultidimensionalRange::operator[](size_t index)
{
  assert(index < ranges.size());
  return ranges[index];
}

const Range& MultidimensionalRange::operator[](size_t index) const
{
  assert(index < ranges.size());
  return ranges[index];
}

unsigned int MultidimensionalRange::rank() const
{
  return ranges.size();
}

void MultidimensionalRange::getSizes(llvm::SmallVectorImpl<size_t>& sizes) const
{
  for (size_t i = 0, e = rank(); i < e; ++i)
    sizes.push_back(ranges[i].size());
}

unsigned int MultidimensionalRange::flatSize() const
{
  unsigned int result = 1;

  for (unsigned int i = 0, r = rank(); i < r; ++i)
    result *= (*this)[i].getEnd() - (*this)[i].getBegin();

  return result;
}

bool MultidimensionalRange::contains(const Point& other) const
{
  for (const auto& [position, range] : llvm::zip(other, ranges))
    if (!range.contains(position))
      return false;

  return true;
}

bool MultidimensionalRange::contains(const MultidimensionalRange& other) const
{
  assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

  for (size_t i = 0, e = rank(); i < e; ++i)
  {
    if (!ranges[i].contains(other.ranges[i]))
      return false;
  }

  return true;
}

bool MultidimensionalRange::overlaps(const MultidimensionalRange& other) const
{
  assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

  for (const auto& [x, y] : llvm::zip(ranges, other.ranges))
    if (!x.overlaps(y))
      return false;

  return true;
}

MultidimensionalRange MultidimensionalRange::intersect(const MultidimensionalRange& other) const
{
  assert(overlaps(other));
  llvm::SmallVector<Range, 3> intersectionRanges;

  for (const auto& [x, y] : llvm::zip(ranges, other.ranges))
    intersectionRanges.push_back(x.intersect(y));

  return MultidimensionalRange(std::move(intersectionRanges));
}

std::pair<bool, size_t> MultidimensionalRange::canBeMerged(const MultidimensionalRange& other) const
{
  assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");

  bool found = false;
  size_t dimension = 0;

  for (size_t i = 0, e = rank(); i < e; ++i)
  {
    if (const Range& first = ranges[i], second = other.ranges[i]; first != second)
    {
      if (first.canBeMerged(other.ranges[i]))
      {
        if (found)
          return std::make_pair(false, 0);

        found = true;
        dimension = i;
      }
      else
      {
        return std::make_pair(false, 0);
      }
    }
  }

  return std::make_pair(found, dimension);
}

MultidimensionalRange MultidimensionalRange::merge(const MultidimensionalRange& other, size_t dimension) const
{
  assert(rank() == other.rank());
  llvm::SmallVector<Range, 3> mergedRanges;

  for (size_t i = 0, e = rank(); i < e; ++i)
  {
    if (i == dimension)
    {
      Range merged = ranges[i].merge(other.ranges[i]);
      mergedRanges.push_back(std::move(merged));
    }
    else
    {
      assert(ranges[i] == other.ranges[i]);
      mergedRanges.push_back(ranges[i]);
    }
  }

  return MultidimensionalRange(mergedRanges);
}

std::vector<MultidimensionalRange> MultidimensionalRange::subtract(const MultidimensionalRange& other) const
{
  assert(rank() == other.rank() && "Can't compare ranges defined on different hyper-spaces");
  std::vector<MultidimensionalRange> results;

  if (!overlaps(other))
  {
    results.push_back(*this);
  }
  else
  {
    llvm::SmallVector<Range, 3> resultRanges(ranges.begin(), ranges.end());

    for (size_t i = 0, e = rank(); i < e; ++i)
    {
      const auto& x = ranges[i];
      const auto& y = other.ranges[i];
      assert(x.overlaps(y));

      for (const auto& subRange : x.subtract(y))
      {
        resultRanges[i] = std::move(subRange);
        results.emplace_back(resultRanges);
      }

      resultRanges[i] = x.intersect(y);
    }
  }

  return results;
}

MultidimensionalRange::const_iterator MultidimensionalRange::begin() const
{
  return const_iterator(
          ranges, [](const Range& range) {
              return range.begin();
          });
}

MultidimensionalRange::const_iterator MultidimensionalRange::end() const
{
  return const_iterator(
          ranges, [](const Range& range) {
              return range.end();
          });
}

namespace marco::matching
{
  std::ostream& operator<<(std::ostream& stream, const MultidimensionalRange& obj)
  {
    stream << "[";

    for (size_t i = 0, e = obj.rank(); i < e; ++i)
    {
      if (i != 0)
        stream << ",";

      stream << obj[i];
    }

    stream << "]";
    return stream;
  }
}
