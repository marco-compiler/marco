#include <marco/modeling/AccessFunction.h>

using namespace marco::modeling::internal;

namespace marco::modeling
{
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
    assert(position >= 0);
    return DimensionAccess(true, position);
  }

  DimensionAccess DimensionAccess::relative(unsigned int inductionVariableIndex, Point::data_type relativePosition)
  {
    return DimensionAccess(false, relativePosition, inductionVariableIndex);
  }

  Point::data_type DimensionAccess::operator()(const Point& equationIndexes) const
  {
    if (isConstantAccess()) {
      return getPosition();
    }

    return equationIndexes[getInductionVariableIndex()] + getOffset();
  }

  Range DimensionAccess::operator()(const MultidimensionalRange& range) const
  {
    if (isConstantAccess()) {
      return Range(getPosition(), getPosition() + 1);
    }

    auto accessedDimensionIndex = getInductionVariableIndex();
    assert(accessedDimensionIndex < range.rank());
    const auto& sourceRange = range[accessedDimensionIndex];
    return Range(sourceRange.getBegin() + getOffset(), sourceRange.getEnd() + getOffset());
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

  AccessFunction AccessFunction::combine(const AccessFunction& other) const
  {
    llvm::SmallVector<DimensionAccess, 3> accesses;

    for (const auto& function : other.functions)
      accesses.push_back(combine(function));

    return AccessFunction(std::move(accesses));
  }

  DimensionAccess AccessFunction::combine(const DimensionAccess& other) const
  {
    if (other.isConstantAccess())
      return other;

    unsigned int inductionVariableIndex = other.getInductionVariableIndex();
    assert(inductionVariableIndex < functions.size());

    const auto& mapped = functions[inductionVariableIndex];
    return DimensionAccess::relative(mapped.getInductionVariableIndex(), mapped.getOffset() + other.getOffset());
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

  bool AccessFunction::isIdentity() const
  {
    for (size_t i = 0; i < size(); ++i) {
      const auto& function = functions[i];

      if (function.isConstantAccess())
        return false;

      if (function.getInductionVariableIndex() != i)
        return false;

      if (function.getOffset() != 0)
        return false;
    }

    return true;
  }

  Point AccessFunction::map(const Point& equationIndexes) const
  {
    llvm::SmallVector<Point::data_type, 3> results;

    for (const auto& function: functions) {
      results.push_back(function(equationIndexes));
    }

    return Point(std::move(results));
  }

  MultidimensionalRange AccessFunction::map(const MultidimensionalRange& range) const
  {
    assert(functions.size() == range.rank());
    llvm::SmallVector<Range, 3> ranges;

    for (const auto& function : functions)
      ranges.push_back(function(range));

    return MultidimensionalRange(std::move(ranges));
  }

  AccessFunction AccessFunction::invert() const
  {
    llvm::SmallVector<DimensionAccess, 2> accesses;
    // TODO
    accesses.reserve(functions.size());

    for (size_t i = 0; i < functions.size(); ++i)
      if (!functions[i].isConstantAccess())
        accesses[functions[i].getInductionVariableIndex()] = DimensionAccess::relative(-1 * functions[i].getOffset(), i);

    return AccessFunction(std::move(accesses));
  }
}
