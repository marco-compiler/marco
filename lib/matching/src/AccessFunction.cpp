#include <marco/matching/AccessFunction.h>

using namespace marco::matching;

SingleDimensionAccess::SingleDimensionAccess(
		bool constantAccess,
		long position,
		unsigned int inductionVariableIndex)
		: constantAccess(constantAccess),
			position(position),
			inductionVariableIndex(inductionVariableIndex)
{
}

SingleDimensionAccess SingleDimensionAccess::constant(long position)
{
	return SingleDimensionAccess(true, position);
}

SingleDimensionAccess SingleDimensionAccess::relative(unsigned int inductionVariableIndex, long relativePosition)
{
	return SingleDimensionAccess(false, relativePosition, inductionVariableIndex);
}

size_t SingleDimensionAccess::operator()(llvm::ArrayRef<long> equationIndexes) const
{
	if (isConstantAccess())
		return getPosition();

	return equationIndexes[getInductionVariableIndex()] + getOffset();
}

bool SingleDimensionAccess::isConstantAccess() const
{
	return constantAccess;
}

size_t SingleDimensionAccess::getPosition() const
{
	assert(isConstantAccess());
	return position;
}

size_t SingleDimensionAccess::getOffset() const
{
	assert(!isConstantAccess());
	return position;
}

unsigned int SingleDimensionAccess::getInductionVariableIndex() const
{
	assert(!isConstantAccess());
	return inductionVariableIndex;
}

AccessFunction::AccessFunction(llvm::ArrayRef<SingleDimensionAccess> functions)
		: functions(functions.begin(), functions.end())
{
}

SingleDimensionAccess AccessFunction::operator[](size_t index) const
{
	assert(index < size());
	return functions[index];
}


llvm::ArrayRef<SingleDimensionAccess> AccessFunction::getDimensionAccesses() const
{
	return functions;
}

void AccessFunction::map(llvm::SmallVectorImpl<size_t>& results, llvm::ArrayRef<long> equationIndexes) const
{
	for (const auto& function : functions)
		results.push_back(function(equationIndexes));
}

size_t AccessFunction::size() const
{
	return functions.size();
}

AccessFunction::iterator AccessFunction::begin()
{
	return functions.begin();
}

AccessFunction::const_iterator AccessFunction::begin() const
{
	return functions.begin();
}

AccessFunction::iterator AccessFunction::end()
{
	return functions.end();
}

AccessFunction::const_iterator AccessFunction::end() const
{
	return functions.end();
}
