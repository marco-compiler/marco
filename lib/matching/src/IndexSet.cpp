#include <marco/matching/IndexSet.h>

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

RangeSet::RangeSet(llvm::ArrayRef<Range> ranges)
		: ranges(ranges.begin(), ranges.end())
{
}

Range RangeSet::operator[](size_t index) const
{
	assert(index < ranges.size());
	return ranges[index];
}

unsigned int RangeSet::rank()
{
	return ranges.size();
}
