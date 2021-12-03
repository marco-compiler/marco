#include <marco/matching/MCIS.h>

using namespace marco::matching;

static bool doRangesIntersect(llvm::ArrayRef<MultidimensionalRange> ranges)
{
	for (size_t i = 0; i < ranges.size(); ++i)
		for (size_t j = i + 1; j < ranges.size(); ++j)
			if (ranges[i].intersects(ranges[j]))
				return true;

	return false;
}

MCIS::MCIS(llvm::ArrayRef<MultidimensionalRange> ranges)
		: ranges(ranges.begin(), ranges.end())
{
	assert(!doRangesIntersect(this->ranges) && "Ranges must not intersect");
  sort();
}

bool MCIS::operator==(const MCIS& other) const
{
  // TODO
}

bool MCIS::operator!=(const MCIS& other) const
{
  // TODO
}

MultidimensionalRange& MCIS::operator[](size_t index)
{
  assert(index < ranges.size());
  return ranges[index];
}

const MultidimensionalRange& MCIS::operator[](size_t index) const
{
	assert(index < ranges.size());
	return ranges[index];
}

bool MCIS::contains(llvm::ArrayRef<Range::data_type> element) const
{
  for (const auto& range : ranges)
    if (range.contains(element))
      return true;

  return false;
}

void MCIS::add(MultidimensionalRange range)
{
  auto hasCompatibleRank = [&](const MultidimensionalRange& range) {
    if (ranges.empty())
      return true;

    return ranges[0].rank() == range.rank();
  };

  assert(hasCompatibleRank(range) && "Incompatible range");

  assert(llvm::none_of(ranges, [&](const MultidimensionalRange& r) {
    return r.intersects(range);
  }) && "New range must not intersect the existing ones");

  ranges.push_back(std::move(range));
  sort();
}

void MCIS::sort()
{
  llvm::sort(this->ranges, [](const MultidimensionalRange& first, const MultidimensionalRange& second) {
    assert(first.rank() == second.rank());

    for (size_t i = 0, e = first.rank(); i < e; ++i)
      if (first[i].getBegin() < second[i].getBegin())
        return true;

    return false;
  });
}
