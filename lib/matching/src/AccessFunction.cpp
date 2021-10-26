#include <marco/matching/AccessFunction.h>

using namespace marco::matching;

DimensionAccess::DimensionAccess(
		bool constantAccess,
		long position,
		unsigned int inductionVariableIndex)
		: constantAccess(constantAccess),
			position(position),
			inductionVariableIndex(inductionVariableIndex)
{
}

DimensionAccess DimensionAccess::constant(long position)
{
	return DimensionAccess(true, position);
}

DimensionAccess DimensionAccess::relative(unsigned int inductionVariableIndex, long relativePosition)
{
	return DimensionAccess(false, relativePosition, inductionVariableIndex);
}

size_t DimensionAccess::operator()(llvm::ArrayRef<long> equationIndexes) const
{
	if (isConstantAccess())
		return getPosition();

	return equationIndexes[getInductionVariableIndex()] + getOffset();
}

bool DimensionAccess::isConstantAccess() const
{
	return constantAccess;
}

size_t DimensionAccess::getPosition() const
{
	assert(isConstantAccess());
	return position;
}

size_t DimensionAccess::getOffset() const
{
	assert(!isConstantAccess());
	return position;
}

unsigned int DimensionAccess::getInductionVariableIndex() const
{
	assert(!isConstantAccess());
	return inductionVariableIndex;
}

AccessFunction::AccessFunction(llvm::ArrayRef<DimensionAccess> functions)
		: functions(functions.begin(), functions.end())
{
}

DimensionAccess AccessFunction::operator[](size_t index) const
{
	assert(index < size());
	return functions[index];
}


llvm::ArrayRef<DimensionAccess> AccessFunction::getDimensionAccesses() const
{
	return functions;
}

void AccessFunction::map(llvm::SmallVectorImpl<long>& results, llvm::ArrayRef<long> equationIndexes) const
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
