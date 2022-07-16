#include "marco/Modeling/AccessFunction.h"

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

  bool DimensionAccess::operator==(const DimensionAccess& other) const
  {
    if (isConstantAccess() && other.isConstantAccess()) {
      return getPosition() == other.getPosition();
    }

    if (!isConstantAccess() && !other.isConstantAccess()) {
      return getOffset() == other.getOffset();
    }

    return false;
  }

  bool DimensionAccess::operator!=(const DimensionAccess& other) const
  {
    if (isConstantAccess() && other.isConstantAccess()) {
      return getPosition() != other.getPosition();
    }

    if (!isConstantAccess() && !other.isConstantAccess()) {
      return getOffset() != other.getOffset();
    }

    return true;
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

  std::ostream& operator<<(std::ostream& stream, const DimensionAccess& obj)
  {
    stream << "[";

    if (obj.isConstantAccess()) {
      stream << obj.getPosition();
    } else {
      stream << "i" << obj.getInductionVariableIndex();

      if (auto offset = obj.getOffset(); offset > 0) {
        stream << " + " << offset;
      } else if (offset < 0) {
        stream << " - " << (-1 * offset);
      }
    }

    stream << "]";
    return stream;
  }

  AccessFunction::AccessFunction(llvm::ArrayRef<DimensionAccess> functions)
      : functions(functions.begin(), functions.end())
  {
  }

  AccessFunction AccessFunction::identity(size_t dimensionality)
  {
    llvm::SmallVector<DimensionAccess, 3> accesses;

    for (size_t i = 0; i < dimensionality; ++i) {
      accesses.push_back(DimensionAccess::relative(i, 0));
    }

    return AccessFunction(std::move(accesses));
  }

  bool AccessFunction::operator==(const AccessFunction& other) const
  {
    if (size() != other.size()) {
      return false;
    }

    for (const auto& [first, second] : llvm::zip(*this, other)) {
      if (first != second) {
        return false;
      }
    }

    return true;
  }

  bool AccessFunction::operator!=(const AccessFunction& other) const
  {
    if (size() != other.size()) {
      return true;
    }

    for (const auto& [first, second] : llvm::zip(*this, other)) {
      if (first != second) {
        return true;
      }
    }

    return false;
  }

  const DimensionAccess& AccessFunction::operator[](size_t index) const
  {
    assert(index < size());
    return functions[index];
  }

  AccessFunction AccessFunction::combine(const AccessFunction& other) const
  {
    llvm::SmallVector<DimensionAccess, 3> accesses;

    for (const auto& function : other.functions) {
      accesses.push_back(combine(function));
    }

    return AccessFunction(std::move(accesses));
  }

  DimensionAccess AccessFunction::combine(const DimensionAccess& other) const
  {
    if (other.isConstantAccess()) {
      return other;
    }

    unsigned int inductionVariableIndex = other.getInductionVariableIndex();
    assert(inductionVariableIndex < functions.size());

    const auto& mapped = functions[inductionVariableIndex];

    if (mapped.isConstantAccess()) {
      return DimensionAccess::constant(mapped.getPosition() + other.getOffset());
    }

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

      if (function.isConstantAccess()) {
        return false;
      }

      if (function.getInductionVariableIndex() != i) {
        return false;
      }

      if (function.getOffset() != 0) {
        return false;
      }
    }

    return true;
  }

  bool AccessFunction::isInvertible() const
  {
    llvm::SmallVector<bool, 3> usedDimensions(size(), false);

    for (const auto& function : functions) {
      if (!function.isConstantAccess()) {
        auto inductionVar = function.getInductionVariableIndex();

        if (inductionVar >= usedDimensions.size()) {
          return false;
        }

        usedDimensions[inductionVar] = true;
      }
    }

    return llvm::all_of(usedDimensions, [](const auto& used) {
      return used;
    });
  }

  AccessFunction AccessFunction::inverse() const
  {
    assert(isInvertible());

    std::vector<DimensionAccess> remapped;
    remapped.reserve(functions.size());

    std::vector<size_t> positionsMap;
    positionsMap.resize(functions.size());

    for (size_t i = 0; i < functions.size(); ++i) {
      auto inductionVar = functions[i].getInductionVariableIndex();
      remapped.push_back(DimensionAccess::relative(i, -1 * functions[i].getOffset()));
      positionsMap[inductionVar] = i;
    }

    std::vector<DimensionAccess> reordered;

    for (const auto& position : positionsMap) {
      reordered.push_back(std::move(remapped[position]));
    }

    return AccessFunction(std::move(reordered));
  }

  Point AccessFunction::map(const Point& equationIndexes) const
  {
    std::vector<Point::data_type> results;

    for (const auto& function: functions) {
      results.push_back(function(equationIndexes));
    }

    return Point(std::move(results));
  }

  MultidimensionalRange AccessFunction::map(const MultidimensionalRange& range) const
  {
    std::vector<Range> ranges;

    for (const auto& function : functions) {
      ranges.push_back(function(range));
    }

    return MultidimensionalRange(std::move(ranges));
  }

  IndexSet AccessFunction::map(const IndexSet& indexes) const
  {
    IndexSet result;

    for (const auto& range : llvm::make_range(indexes.rangesBegin(), indexes.rangesEnd())) {
      result += map(range);
    }

    return result;
  }

  MultidimensionalRange AccessFunction::inverseMap(const MultidimensionalRange& range) const
  {
    return inverse().map(range);
  }

  IndexSet AccessFunction::inverseMap(const IndexSet& indexes) const
  {
    return inverse().map(indexes);
  }

  IndexSet AccessFunction::inverseMap(const IndexSet& indices, const IndexSet& parentIndexes) const
  {
    if (isInvertible() && !indices.empty() && !parentIndexes.empty() && indices.rank() == parentIndexes.rank()) {
      auto mapped = inverseMap(indices);
      assert(map(mapped).contains(indices));
      return mapped;
    }

    // If the access function is not invertible, then not all the iteration variables are
    // used. This loss of information don't allow to reconstruct the equation ranges that
    // leads to the dependency loop. Thus, we need to iterate on all the original equation
    // points and determine which of them lead to a loop. This is highly expensive but also
    // inevitable, and confined only to very few cases within real scenarios.

    IndexSet result;

    for (const auto& range: llvm::make_range(parentIndexes.rangesBegin(), parentIndexes.rangesEnd())) {
      for (const auto& point: range) {
        if (indices.contains(map(point))) {
          result += point;
        }
      }
    }

    return result;
  }

  std::ostream& operator<<(std::ostream& stream, const AccessFunction& obj)
  {
    stream << "[";

    for (const auto& access : obj) {
      stream << access;
    }

    stream << "]";
    return stream;
  }
}
