#include <marco/matching/MCIS.h>

using namespace marco::matching;

static bool doRangesIntersect(llvm::ArrayRef<MultidimensionalRange> ranges)
{
	for (size_t i = 0; i < ranges.size(); ++i)
		for (size_t j = 0; j < ranges.size(); ++j)
			if (i != j && ranges[i].intersects(ranges[j]))
				return true;

	return false;
}

MCIS::MCIS(llvm::ArrayRef<MultidimensionalRange> ranges)
		: ranges(ranges.begin(), ranges.end())
{
	assert(!doRangesIntersect(this->ranges) && "Ranges must not intersect");
}

MultidimensionalRange MCIS::operator[](size_t index) const
{
	assert(index < ranges.size());
	return ranges[index];
}
