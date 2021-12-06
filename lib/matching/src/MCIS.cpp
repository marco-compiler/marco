#include <marco/matching/MCIS.h>

using namespace marco::matching;

template<typename It>
static bool doRangesIntersect(It begin, It end)
{
  for (It it1 = begin; it1 != end; ++it1)
  {
    for (It it2 = std::next(it1); it2 != end; ++it2)
      if (it1->intersects(*it2))
        return true;
  }

	return false;
}

MCIS::MCIS(llvm::ArrayRef<MultidimensionalRange> ranges)
		: ranges(ranges.begin(), ranges.end())
{
	assert(!doRangesIntersect(this->ranges.begin(), this->ranges.end()) && "Ranges must not intersect");
  sort();
  merge();
}

MultidimensionalRange& MCIS::operator[](size_t index)
{
  assert(index < ranges.size());
  return *(std::next(ranges.begin(), index));
}

const MultidimensionalRange& MCIS::operator[](size_t index) const
{
	assert(index < ranges.size());
  return *(std::next(ranges.begin(), index));
}

bool MCIS::contains(llvm::ArrayRef<Range::data_type> element) const
{
  for (const auto& range : ranges)
    if (range.contains(element))
      return true;

  return false;
}

bool MCIS::contains(const MultidimensionalRange& range) const
{
  for (const auto& current : ranges)
  {
    if (current.contains(range))
      return true;

    if (current > range)
      return false;
  }

  return false;
}

void MCIS::add(MultidimensionalRange range)
{
  auto hasCompatibleRank = [&](const MultidimensionalRange& range) {
    if (ranges.empty())
      return true;

    return ranges.front().rank() == range.rank();
  };

  assert(hasCompatibleRank(range) && "Incompatible range");

  assert(llvm::none_of(ranges, [&](const MultidimensionalRange& r) {
    return r.intersects(range);
  }) && "New range must not intersect the existing ones");

  auto it = std::find_if(ranges.begin(), ranges.end(), [&range](const MultidimensionalRange& r) {
    return r > range;
  });

  ranges.insert(it, std::move(range));
  merge();
}

void MCIS::sort()
{
  ranges.sort([](const MultidimensionalRange& first, const MultidimensionalRange& second) {
      return first < second;
  });
}

void MCIS::merge()
{
  using It = decltype(ranges)::iterator;

  auto findCandidates = [&](It begin, It end) -> std::tuple<It, It, size_t> {
    for (It it1 = begin; it1 != end; ++it1)
      for (It it2 = std::next(it1); it2 != end; ++it2)
        if (auto mergePossibility = it1->canBeMerged(*it2); mergePossibility.first)
          return std::make_tuple(it1, it2, mergePossibility.second);

    return std::make_tuple(end, end, 0);
  };

  auto candidates = findCandidates(ranges.begin(), ranges.end());

  while (std::get<0>(candidates) != ranges.end() && std::get<1>(candidates) != ranges.end())
  {
    auto& first = std::get<0>(candidates);
    auto& second = std::get<1>(candidates);
    size_t dimension = std::get<2>(candidates);

    *first = first->merge(*second, dimension);
    ranges.erase(second);
    candidates = findCandidates(ranges.begin(), ranges.end());
  }
}
