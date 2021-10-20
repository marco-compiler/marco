#include <marco/matching/AccessFunction.h>

using namespace marco::matching;

SingleDimensionAccess::SingleDimensionAccess(
		bool constantAccess,
		int64_t position,
		unsigned int inductionVariableIndex)
		: constantAccess(constantAccess),
			position(position),
			inductionVariableIndex(inductionVariableIndex)
{
}

SingleDimensionAccess SingleDimensionAccess::constant(int64_t position)
{
	return SingleDimensionAccess(position, true);
}

SingleDimensionAccess SingleDimensionAccess::relative(unsigned int inductionVariableIndex, int64_t relativePosition)
{
	return SingleDimensionAccess(relativePosition, false, inductionVariableIndex);
}

bool SingleDimensionAccess::isConstantAccess() const
{
	return constantAccess;
}

int64_t SingleDimensionAccess::getPosition() const
{
	assert(isConstantAccess());
	return position;
}

int64_t SingleDimensionAccess::getOffset() const
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
