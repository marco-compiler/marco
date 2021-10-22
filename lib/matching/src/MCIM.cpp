#include <marco/matching/MCIM.h>

using namespace marco::matching;

MCIMElement::MCIMElement(long delta, MCIS k)
		: delta(delta), k(std::move(k))
{
}

MCIM::MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
		: equationRanges(equationRanges),
			variableRanges(variableRanges),
			data(equationRanges.flatSize(), variableRanges.flatSize())
{
	assert(variableRanges.rank() <= equationRanges.rank());
}

void MCIM::apply(AccessFunction access)
{
	llvm::SmallVector<size_t, 6> indexes;

	for (const auto& equationIndexes : equationRanges)
	{
		assert(equationIndexes.size() == equationRanges.rank());
		assert(access.size() == variableRanges.rank());

		indexes.clear();
		indexes.insert(indexes.begin(), equationIndexes.begin(), equationIndexes.end());

		for (const auto& singleDimensionAccess : access)
			access.map(indexes, equationIndexes);

		assert(indexes.size() == equationRanges.rank() + variableRanges.rank());
		set(indexes);
	}
}

void MCIM::set(llvm::ArrayRef<size_t> indexes)
{
	bool separator = false;

	std::cout << "set (";

	for (const auto& index : indexes)
	{
		if (separator)
			std::cout << ",";

		separator = true;
		std::cout << index;
	}

	std::cout << ")" << std::endl;
}

/*
size_t MCIM::flatEquationIndex(llvm::ArrayRef<size_t> indexes) const
{

}

size_t MCIM::flatVariableIndex(llvm::ArrayRef<size_t> indexes) const
{

}
*/
