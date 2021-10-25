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

bool Range::intersects(Range other) const
{
	return getBegin() < other.getEnd() && getEnd() > other.getBegin();
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

Range MultidimensionalRange::operator[](size_t index) const
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

bool MultidimensionalRange::intersects(MultidimensionalRange other) const
{
	assert(rank() == other.rank() &&
				 "Can't compare ranges defined on different hyper-spaces");

	for (const auto& [x, y] : llvm::zip(ranges, other.ranges))
		if (!x.intersects(y))
			return false;

	return true;
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
