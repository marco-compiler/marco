#include <marco/matching/Matching.h>

using namespace marco::matching;
using namespace marco::matching::detail;

LocalMatchingSolutions::LocalMatchingSolutions(
		llvm::ArrayRef<AccessFunction> accessFunctions,
		MultidimensionalRange equationRanges,
		MultidimensionalRange variableRanges)
		: accessFunctions(accessFunctions.begin(), accessFunctions.end()),
			equationRanges(std::move(equationRanges)),
			variableRanges(std::move(variableRanges))
{
	// Sort the access functions so that we prefer the ones referring to
	// induction variables. More in details, we prefer the ones covering the
	// most of them, as they are the ones that lead to the least amount of
	// groups.

	llvm::sort(this->accessFunctions, [&](const AccessFunction& lhs, const AccessFunction& rhs) -> bool {
		llvm::SmallVector<size_t, 3> lhsUsages;
		llvm::SmallVector<size_t, 3> rhsUsages;

		getInductionVariablesUsage(lhsUsages, lhs);
		getInductionVariablesUsage(rhsUsages, rhs);

		auto counter = [](const size_t& usage) {
			return usage == 1;
		};

		size_t lhsUniqueUsages = llvm::count_if(lhsUsages, counter);
		size_t rhsUniqueUsages = llvm::count_if(rhsUsages, counter);

		return lhsUniqueUsages >= rhsUniqueUsages;
	});

	// Determine the amount of solutions. Precomputing it allows the actual
	// solutions to be determined only when requested. This is useful when
	// the solutions set will be processed only if a certain amount of solutions
	// is found, as it may allow for skipping their computation.

	solutionsAmount = 0;
	llvm::SmallVector<size_t, 3> inductionsUsage;

	for (const auto& accessFunction : this->accessFunctions)
	{
		size_t count = 1;
		getInductionVariablesUsage(inductionsUsage, accessFunction);

		for (const auto& usage : llvm::enumerate(inductionsUsage))
			if (usage.value() == 0)
				count *= this->equationRanges[usage.index()].size();

		solutionsAmount += count;
	}
}

IncidenceMatrix& LocalMatchingSolutions::operator[](size_t index)
{
	assert(index < size());

	while (matrices.size() <= index)
		fetchNext();

	return matrices[index];
}

size_t LocalMatchingSolutions::size() const
{
	return solutionsAmount;
}

LocalMatchingSolutions::iterator LocalMatchingSolutions::begin()
{
	return iterator(*this, 0);
}

LocalMatchingSolutions::iterator LocalMatchingSolutions::end()
{
	return iterator(*this, size());
}

void LocalMatchingSolutions::fetchNext()
{
	if (rangeIt == nullptr || *rangeIt == *rangeEnd)
	{
		assert(currentAccessFunction < accessFunctions.size() &&
					 "No more access functions to be processed");

		llvm::SmallVector<size_t, 3> inductionsUsage;
		getInductionVariablesUsage(inductionsUsage, accessFunctions[currentAccessFunction]);

		groupSize = 1;

		reorderedRanges.clear();

		ordering.clear();
		ordering.insert(ordering.begin(), equationRanges.rank(), 0);

		// We need to reorder the variables iteration order so that the unused ones
		// are the last ones changing. In fact, the unused variables are the one
		// leading to repetitions among variable usages, and thus lead to a new
		// group for each time they change value.

		for (const auto& usage : llvm::enumerate(inductionsUsage))
		{
			if (usage.value() == 0)
			{
				ordering[usage.index()] = reorderedRanges.size();
				reorderedRanges.push_back(equationRanges[usage.index()]);
			}
		}

		for (const auto& usage : llvm::enumerate(inductionsUsage))
		{
			if (usage.value() != 0)
			{
				ordering[usage.index()] = reorderedRanges.size();
				reorderedRanges.push_back(equationRanges[usage.index()]);
				groupSize *= equationRanges[usage.index()].size();
			}
		}

		// Theoretically, it would be sufficient to store just the iterators of
		// the reordered multidimensional range. Anyway, their implementation may
		// rely on the range existence, and having it allocated on the stack may
		// lead to dangling pointers. Thus we also store the range inside the
		// class.

		range = std::make_unique<MultidimensionalRange>(reorderedRanges);
		rangeIt = std::make_unique<MultidimensionalRange::iterator>(range->begin());
		rangeEnd = std::make_unique<MultidimensionalRange::iterator>(range->end());
	}

	IncidenceMatrix matrix(equationRanges, variableRanges);
	llvm::SmallVector<long, 3> equationIndexes;
	llvm::SmallVector<long, 3> indexes;
	size_t counter = 0;

	while (counter++ != groupSize)
	{
		const auto& reorderedIndexes = **rangeIt;
		equationIndexes.clear();

		for (size_t i = 0, e = equationRanges.rank(); i < e; ++i)
			equationIndexes.push_back(reorderedIndexes[ordering[i]]);

		indexes.clear();
		indexes.insert(indexes.begin(), equationIndexes.begin(), equationIndexes.end());
		accessFunctions[currentAccessFunction].map(indexes, equationIndexes);
		matrix.set(indexes);

		if (counter == groupSize)
			matrices.push_back(std::move(matrix));

		++(*rangeIt);
	}
}

void LocalMatchingSolutions::getInductionVariablesUsage(
		llvm::SmallVectorImpl<size_t>& usages,
		const AccessFunction& accessFunction) const
{
	usages.clear();
	usages.insert(usages.begin(), equationRanges.rank(), 0);

	for (const auto& dimensionAccess : accessFunction)
		if (!dimensionAccess.isConstantAccess())
			++usages[dimensionAccess.getInductionVariableIndex()];
}

namespace marco::matching::detail
{
	LocalMatchingSolutions solveLocalMatchingProblem(
			const IncidenceMatrix& u,
			llvm::ArrayRef<AccessFunction> accessFunctions)
	{
		return LocalMatchingSolutions(
				std::move(accessFunctions),
				u.getEquationRanges(),
				u.getVariableRanges());
	}
}
