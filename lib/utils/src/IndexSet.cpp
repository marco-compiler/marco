#include "modelica/utils/IndexSet.hpp"

#include <algorithm>

#include "llvm/Support/raw_ostream.h"
#include "modelica/utils/Interval.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

bool IndexSet::contains(ArrayRef<size_t> point) const
{
	return find_if(values.begin(), values.end(), [point](auto& child) {
					 return child.contains(point);
				 }) != values.end();
}

bool IndexSet::disjoint(const MultiDimInterval& other) const
{
	for (auto& el : *this)
		if (!areDisjoint(el, other))
			return false;

	return true;
}

void IndexSet::intersecate(const MultiDimInterval& other)
{
	IndexSet newValues;

	for (const auto& range : *this)
		if (!areDisjoint(range, other))
			newValues.unite(intersection(range, other));

	*this = move(newValues);
}

void IndexSet::intersecate(const IndexSet& other)
{
	IndexSet newValues;

	for (const auto& range : *this)
		for (const auto& range2 : other)
			if (!areDisjoint(range, range2))
				newValues.unite(intersection(range, range2));

	*this = move(newValues);
}

static SmallVector<MultiDimInterval, 3> cuttAll(
		const SmallVector<MultiDimInterval, 3>& arrayToCut,
		size_t dimension,
		ArrayRef<size_t> cutPoints)
{
	SmallVector<MultiDimInterval, 3> toFill;
	for (const auto& toCut : arrayToCut)
		toCut.cutOnDimension(dimension, cutPoints, back_inserter(toFill));

	return toFill;
}

IndexSet modelica::remove(
		const MultiDimInterval& left, const MultiDimInterval& right)
{
	if (areDisjoint(left, right))
		return IndexSet(left);

	IndexSet toReturn;

	SmallVector<MultiDimInterval, 3> cuttedLeft = { left };

	// for each dimension
	for (size_t dimension = 0; dimension < left.dimensions(); dimension++)
	{
		const auto& val = right.at(dimension);
		// we cut all the cubes we collected so far
		// on that dimension  in the lines implied by the cube to be removed
		cuttedLeft = cuttAll(cuttedLeft, dimension, { val.min(), val.max() });
	}

	// for each small cube obtained
	for (auto& subCube : cuttedLeft)
	{
		// if they are into inscribed in the right cube
		// we add it to the solution
		if (areDisjoint(subCube, right))
		{
			toReturn.unite(move(subCube));
		}
		else
		{
			// else we check that is fully inscribed
			// if it's not there is a bug because by design
			// we tried to divide the outside cubes from the inside cubes
			// and there should not be cube half inside and half outside
			assert(subCube.isFullyContained(right));	// NOLINT
		}
	}

	return toReturn;
}

void IndexSet::remove(const MultiDimInterval& other)
{
	IndexSet newValues;
	for (const auto& range : *this)
		newValues.unite(modelica::remove(range, other));

	*this = move(newValues);
}

void IndexSet::remove(const IndexSet& other)
{
	for (const auto& otherRange : other)
		remove(otherRange);
}

void IndexSet::compact()
{
	assert(!empty());	 // NOLINT
	auto last = values.end() - 1;

	const auto isExpansion = [&last](const auto& l) {
		return last->isExpansionOf(l).first;
	};

	auto expandable = find_if(values.begin(), last, isExpansion);

	while (expandable != last)
	{
		expandable->expand(*last);
		values.erase(last);
		last = values.end() - 1;
		expandable = find_if(values.begin(), last, isExpansion);
	}

	const auto confront = [](const auto& l, const auto& r) {
		return l.confront(r);
	};
	std::sort(values.begin(), values.end(), confront);
}

void IndexSet::unite(MultiDimInterval other)
{
	assert(disjoint(other));	// NOLINT
	values.emplace_back(std::move(other));
	compact();
}

void IndexSet::dump(llvm::raw_ostream& OS) const
{
	OS << "{ ";
	for (const auto& el : *this)
	{
		el.dump(OS);
		OS << ",";
	}

	OS << " }";
}
