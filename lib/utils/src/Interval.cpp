#include "modelica/utils/Interval.hpp"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace modelica;
using namespace std;

bool MultiDimInterval::contains(llvm::ArrayRef<size_t> point) const
{
	assert(point.size() == dimensions());	 // NOLINT

	for (size_t t = 0; t < point.size(); t++)
		if (!intervals[t].contains(point[t]))
			return false;
	return true;
}

bool modelica::areDisjoint(const Interval& left, const Interval& right)
{
	if (left.min() >= right.max())
		return true;
	if (left.max() <= right.min())
		return true;

	return false;
}

bool modelica::areDisjoint(
		const MultiDimInterval& left, const MultiDimInterval& right)
{
	assert(left.dimensions() == right.dimensions());	// NOLINT

	for (size_t a = 0; a < left.dimensions(); a++)
		if (areDisjoint(left.at(a), right.at(a)))
			return true;

	return false;
}

MultiDimInterval modelica::intersection(
		const MultiDimInterval& left, const MultiDimInterval& right)
{
	assert(!areDisjoint(left, right));								// NOLINT
	assert(left.dimensions() == right.dimensions());	// NOLINT
	SmallVector<Interval, 2> toReturn;

	for (size_t a = 0; a < left.dimensions(); a++)
	{
		toReturn.emplace_back(
				max(left.at(a).min(), right.at(a).min()),
				min(left.at(a).max(), right.at(a).max()));
	}

	return MultiDimInterval(move(toReturn));
}

pair<bool, size_t> MultiDimInterval::isExpansionOf(
		const MultiDimInterval& other) const
{
	assert(other.dimensions() == dimensions());	 // NOLINT
	size_t missmatchedSizes = 0;
	size_t lastMissmached = 0;

	for (size_t index = 0; index < dimensions(); index++)
	{
		if (other.at(index) != at(index))
		{
			missmatchedSizes++;
			lastMissmached = index;
		}
	}
	if (missmatchedSizes != 1)
		return make_pair(false, 0);

	if (other.at(lastMissmached).min() == at(lastMissmached).max())
		return make_pair(true, lastMissmached);

	if (other.at(lastMissmached).max() == at(lastMissmached).min())
		return make_pair(true, lastMissmached);
	return make_pair(false, 0);
}

void MultiDimInterval::expand(const MultiDimInterval& other)
{
	auto [isExpansion, location] = isExpansionOf(other);
	assert(isExpansion);	// NOLINT

	const Interval& left = at(location);
	const Interval& right = other.at(location);

	intervals[location] =
			Interval(min(left.min(), right.min()), max(left.max(), right.max()));
}

int MultiDimInterval::confront(const MultiDimInterval& other) const
{
	assert(dimensions() == other.dimensions());	 // NOLINT

	for (size_t a = 0; a < dimensions(); a++)
	{
		if (at(a) < other.at(a))
			return -1;
		if (at(a) > other.at(a))
			return 1;
	}

	return 0;
}

SmallVector<MultiDimInterval, 3> MultiDimInterval::cutOnDimension(
		size_t dimension, llvm::ArrayRef<size_t> cutLines) const
{
	assert(intervals.size() > dimension);									// NOLINT
	assert(is_sorted(cutLines.begin(), cutLines.end()));	// NOLINT

	SmallVector<MultiDimInterval, 3> toReturn;
	const auto& cuttedDim = intervals[dimension];

	size_t left = cuttedDim.min();
	size_t right = cuttedDim.max();
	for (size_t cutPoint : cutLines)
	{
		if (cutPoint <= cuttedDim.min())
			continue;
		if (cutPoint >= cuttedDim.max())
			break;

		toReturn.emplace_back(replacedDimension(dimension, left, cutPoint));
		left = cutPoint;
	}
	toReturn.push_back(replacedDimension(dimension, left, right));
	return toReturn;
}

bool MultiDimInterval::isFullyContained(const MultiDimInterval& other) const
{
	assert(other.dimensions() == dimensions());	 // NOLINT
	for (size_t dim = 0; dim < dimensions(); dim++)
	{
		if (!at(dim).isFullyContained(other.at(dim)))
			return false;
	}
	return true;
}

void MultiDimInterval::dump(llvm::raw_ostream& OS) const
{
	OS << "[";
	for (const auto& i : *this)
		OS << i.min() << "," << i.max();

	OS << "]";
}
