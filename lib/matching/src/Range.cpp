#include <marco/matching/Range.h>

using namespace marco::matching;

Range::Range(long begin, long end)
		: begin(begin), end(end)
{
}

long Range::getBegin() const
{
	return begin;
}

long Range::getEnd() const
{
	return end;
}

size_t Range::size() const
{
	return getEnd() - getBegin();
}

bool Range::intersects(Range other) const
{
	return getBegin() <= other.getEnd() && getEnd() >= other.getBegin();
}

MultidimensionalRange::MultidimensionalRange(llvm::ArrayRef<Range> ranges)
		: ranges(ranges.begin(), ranges.end())
{
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

	for (const auto& [x, y] : llvm::zip(*this, other))
		if (x.intersects(y))
			return true;

	return false;
}

MultidimensionalRange::iterator MultidimensionalRange::begin()
{
	return ranges.begin();
}

MultidimensionalRange::const_iterator MultidimensionalRange::begin() const
{
	return ranges.begin();
}

MultidimensionalRange::iterator MultidimensionalRange::end()
{
	return ranges.end();
}

MultidimensionalRange::const_iterator MultidimensionalRange::end() const
{
	return ranges.end();
}
