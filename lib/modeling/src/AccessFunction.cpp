#include <llvm/ADT/DenseMap.h>
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
    return map(equationIndexes);
  }

  Range DimensionAccess::operator()(const MultidimensionalRange& range) const
  {
    return map(range);
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

  Point::data_type DimensionAccess::map(const Point& equationIndexes) const
  {
    if (isConstantAccess()) {
      return getPosition();
    }

    return equationIndexes[getInductionVariableIndex()] + getOffset();
  }

  Range DimensionAccess::map(const MultidimensionalRange& range) const
  {
    if (isConstantAccess()) {
      return Range(getPosition(), getPosition() + 1);
    }

    auto accessedDimensionIndex = getInductionVariableIndex();
    assert(accessedDimensionIndex < range.rank());
    const auto& sourceRange = range[accessedDimensionIndex];
    return Range(sourceRange.getBegin() + getOffset(), sourceRange.getEnd() + getOffset());
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

  bool AccessFunction::isInvertible() const
  {
    llvm::SmallVector<bool, 3> usedDimensions(size(), false);

    for (const auto& function : functions)
      if (!function.isConstantAccess())
        usedDimensions[function.getInductionVariableIndex()] = true;

    return llvm::all_of(usedDimensions, [](const auto& used) {
      return used;
    });
  }

  AccessFunction AccessFunction::inverse() const
  {
    assert(isInvertible());

    llvm::SmallVector<DimensionAccess, 2> remapped;
    remapped.reserve(functions.size());

    llvm::SmallVector<size_t, 3> positionsMap;
    positionsMap.resize(functions.size());

    for (size_t i = 0; i < functions.size(); ++i)
    {
      auto inductionVar = functions[i].getInductionVariableIndex();
      remapped.push_back(DimensionAccess::relative(i, -1 * functions[i].getOffset()));
      positionsMap[inductionVar] = i;
    }

    llvm::SmallVector<DimensionAccess, 2> reordered;

    for (const auto& position : positionsMap)
      reordered.push_back(std::move(remapped[position]));

    return AccessFunction(std::move(reordered));
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

  MultidimensionalRange AccessFunction::inverseMap(const MultidimensionalRange& range) const
  {
    return inverse().map(range);
  }
}
