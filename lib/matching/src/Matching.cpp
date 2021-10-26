#include <marco/matching/Matching.h>

using namespace marco::matching;
using namespace marco::matching::detail;

LocalMatchingSolutions::LocalMatchingSolutions(
		llvm::ArrayRef<AccessFunction> accessFunctions,
		MultidimensionalRange equationRanges,
		MultidimensionalRange variableRanges)
		: accessFunctions(std::move(accessFunctions)),
			equationRanges(std::move(equationRanges)),
			variableRanges(std::move(variableRanges))
{
	solutionsCount = 0;
	llvm::SmallVector<size_t, 3> inductionsUsage;

	for (const auto& accessFunction : this->accessFunctions)
	{
		size_t count = 1;
		getInductionVariablesUsage(inductionsUsage, accessFunction);

		for (const auto& usage : llvm::enumerate(inductionsUsage))
			if (usage.value() == 0)
				count *= this->equationRanges[usage.index()].size();

		solutionsCount += count;
	}

	compute();
}

IncidenceMatrix& LocalMatchingSolutions::operator[](size_t index)
{
	assert(index < size());
	return matrices[index];
}

const IncidenceMatrix& LocalMatchingSolutions::operator[](size_t index) const
{
	assert(index < size());
	return matrices[index];
}

size_t LocalMatchingSolutions::size() const
{
	return solutionsCount;
}

LocalMatchingSolutions::iterator LocalMatchingSolutions::begin()
{
	return iterator(*this, 0);
}

LocalMatchingSolutions::const_iterator LocalMatchingSolutions::begin() const
{
	return const_iterator(*this, 0);
}

LocalMatchingSolutions::iterator LocalMatchingSolutions::end()
{
	return iterator(*this, size());
}

LocalMatchingSolutions::const_iterator LocalMatchingSolutions::end() const
{
	return const_iterator(*this, size());
}

void LocalMatchingSolutions::compute()
{
	llvm::SmallVector<size_t, 3> inductionsUsage;

	for (const auto& accessFunction : accessFunctions)
	{
		size_t groupSize = 1;

		getInductionVariablesUsage(inductionsUsage, accessFunction);

		llvm::SmallVector<Range, 3> reorderedRanges;
		llvm::SmallVector<size_t, 3> ordering(equationRanges.rank(), 0);

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

		MultidimensionalRange reorderedRange(reorderedRanges);
		IncidenceMatrix matrix(equationRanges, variableRanges);
		llvm::SmallVector<long, 3> equationIndexes;
		llvm::SmallVector<long, 3> indexes;
		size_t counter = 0;

		for (const auto& reorderedIndexes : reorderedRange)
		{
			equationIndexes.clear();

			for (size_t i = 0, e = equationRanges.rank(); i < e; ++i)
				equationIndexes.push_back(reorderedIndexes[ordering[i]]);

			indexes.clear();
			indexes.insert(indexes.begin(), equationIndexes.begin(), equationIndexes.end());
			accessFunction.map(indexes, equationIndexes);
			matrix.set(indexes);

			if (++counter == groupSize)
			{
				matrices.push_back(matrix);
				matrix.clear();
				counter = 0;
			}
		}
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
