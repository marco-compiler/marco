#include <list>
#include <marco/matching/Range.h>

using namespace marco::matching;

Range::Range(Range::data_type begin, Range::data_type end)
		: _begin(begin), _end(end)
{
	assert(begin < end && "Range is not well-formed");
}

bool Range::operator==(const Range& other) const
{
	return getBegin() == other.getBegin() && getEnd() == other.getEnd();
}

bool Range::operator!=(const Range& other) const
{
	return getBegin() != other.getBegin() || getEnd() != other.getEnd();
}

bool Range::operator<(const Range& other) const
{
  if (getBegin() == other.getBegin())
    return getEnd() < other.getEnd();

  return getBegin() < other.getBegin();
}

bool Range::operator>(const Range& other) const
{
  if (getBegin() == other.getBegin())
    return getEnd() > other.getEnd();

  return getBegin() > other.getBegin();
}

Range::data_type Range::getBegin() const
{
	return _begin;
}

Range::data_type Range::getEnd() const
{
	return _end;
}

size_t Range::size() const
{
	return getEnd() - getBegin();
}

bool Range::contains(Range::data_type value) const
{
	return value >= getBegin() && value < getEnd();
}

bool Range::contains(const Range& other) const
{
  return getBegin() <= other.getBegin() && getEnd() >= other.getEnd();
}

bool Range::overlaps(const Range& other) const
{
  return (getBegin() <= other.getEnd() - 1) && (getEnd() - 1 >= other.getBegin());
}

Range Range::intersect(const Range& other) const
{
  assert(overlaps(other));

  if (contains(other))
    return other;

  if (other.contains(*this))
    return *this;

  if (getBegin() <= other.getBegin())
    return Range(other.getBegin(), getEnd());

  return Range(getBegin(), other.getEnd());
}

bool Range::canBeMerged(const Range& other) const
{
  return getBegin() == other.getEnd() || getEnd() == other.getBegin() || overlaps(other);
}

Range Range::merge(const Range& other) const
{
  assert(canBeMerged(other));

  if (overlaps(other))
    return Range(std::min(getBegin(), other.getBegin()), std::max(getEnd(), other.getEnd()));

  if (getBegin() == other.getEnd())
    return Range(other.getBegin(), getEnd());

  return Range(getBegin(), other.getEnd());
}

std::vector<Range> Range::subtract(const Range& other) const
{
  std::vector<Range> results;

  if (!overlaps(other))
  {
    results.push_back(*this);
  }
  else if (contains(other))
  {
    if (getBegin() != other.getBegin())
      results.emplace_back(getBegin(), other.getBegin());

    if (getEnd() != other.getEnd())
      results.emplace_back(other.getEnd(), getEnd());
  }
  else if (!other.contains(*this))
  {
    if (getBegin() <= other.getBegin())
      results.emplace_back(getBegin(), other.getBegin());
    else
      results.emplace_back(other.getEnd(), getEnd());
  }

  return results;
}

Range::iterator Range::begin()
{
	return iterator(getBegin(), getBegin(), getEnd());
}

Range::const_iterator Range::begin() const
{
	return const_iterator(getBegin(), getBegin(), getEnd());
}

Range::iterator Range::end()
{
	return iterator(getBegin(), getEnd(), getEnd());
}

Range::const_iterator Range::end() const
{
	return const_iterator(getBegin(), getEnd(), getEnd());
}

namespace marco::matching
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Range& range)
	{
		return stream << "[" << range.getBegin() << "," << range.getEnd() << ")";
	}

	std::ostream& operator<<(std::ostream& stream, const Range& range)
	{
		llvm::raw_os_ostream(stream) << range;
		return stream;
	}
}

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

bool MultidimensionalRange::contains(llvm::ArrayRef<Range::data_type> element) const
{
	for (const auto& [position, range] : llvm::zip(element, ranges))
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

MultidimensionalRange::iterator MultidimensionalRange::begin()
{
	return iterator(
			ranges, [](const Range& range) {
				return range.begin();
			});
}

MultidimensionalRange::const_iterator MultidimensionalRange::begin() const
{
	return const_iterator(
			ranges, [](const Range& range) {
				return range.begin();
			});
}

MultidimensionalRange::iterator MultidimensionalRange::end()
{
	return iterator(
			ranges, [](const Range& range) {
				return range.end();
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
	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const MultidimensionalRange& range)
	{
		stream << "[";

		for (size_t i = 0, e = range.rank(); i < e; ++i)
		{
			if (i != 0)
				stream << ",";

			stream << range[i];
		}

		stream << "]";
		return stream;
	}

	std::ostream& operator<<(
			std::ostream& stream, const MultidimensionalRange& range)
	{
		llvm::raw_os_ostream(stream) << range;
		return stream;
	}
}
