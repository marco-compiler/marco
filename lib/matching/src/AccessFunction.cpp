#include <marco/matching/AccessFunction.h>

using namespace marco::matching;

DimensionAccess::DimensionAccess(
		bool constantAccess,
    Point::data_type position,
		unsigned int inductionVariableIndex)
		: constantAccess(constantAccess),
			position(position),
			inductionVariableIndex(inductionVariableIndex)
{
}

DimensionAccess DimensionAccess::constant(Point::data_type position)
{
	return DimensionAccess(true, position);
}

DimensionAccess DimensionAccess::relative(unsigned int inductionVariableIndex, Point::data_type relativePosition)
{
	return DimensionAccess(false, relativePosition, inductionVariableIndex);
}

Point::data_type DimensionAccess::operator()(const Point& equationIndexes) const
{
	if (isConstantAccess())
		return getPosition();

	return equationIndexes[getInductionVariableIndex()] + getOffset();
}

bool DimensionAccess::isConstantAccess() const
{
	return constantAccess;
}

Point::data_type DimensionAccess::getPosition() const
{
	assert(isConstantAccess());
	return position;
}

Point::data_type DimensionAccess::getOffset() const
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

const DimensionAccess& AccessFunction::operator[](size_t index) const
{
	assert(index < size());
	return functions[index];
}

size_t AccessFunction::size() const
{
  return functions.size();
}

AccessFunction::const_iterator AccessFunction::begin() const
{
  return functions.begin();
}

AccessFunction::const_iterator AccessFunction::end() const
{
  return functions.end();
}

Point AccessFunction::map(const Point& equationIndexes) const
{
  llvm::SmallVector<Point::data_type, 3> results;

	for (const auto& function : functions)
		results.push_back(function(equationIndexes));

  return Point(std::move(results));
}
