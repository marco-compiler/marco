#include "marco/modeling/AccessFunction.h"

namespace marco::modeling
{
  DimensionAccess::DimensionAccess(
      bool constantAccess,
      Point::data_type position,
      unsigned int inductionVariableIndex,
      llvm::ArrayRef<Point::data_type> array)
      : constantAccess(constantAccess),
        position(position),
        inductionVariableIndex(inductionVariableIndex),
        array(array.begin(),array.end())
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

  DimensionAccess DimensionAccess::relativeToArray(unsigned int inductionVariableIndex, llvm::ArrayRef<Point::data_type> array)
  {
    assert(!array.empty());
    return DimensionAccess(false, 0, inductionVariableIndex, array);
  }

  bool DimensionAccess::operator==(const DimensionAccess& other) const
  {
    if (isConstantAccess() && other.isConstantAccess()) {
      return getPosition() == other.getPosition();
    }

    if (!isConstantAccess() && !other.isConstantAccess()) {
      return getOffset() == other.getOffset() && array==other.array;
    }

    return false;
  }

  bool DimensionAccess::operator!=(const DimensionAccess& other) const
  {
    if (isConstantAccess() && other.isConstantAccess()) {
      return getPosition() != other.getPosition();
    }

    if (!isConstantAccess() && !other.isConstantAccess()) {
      if(!array.empty())
        return array!=other.array;
      return getOffset() != other.getOffset();
    }

    return true;
  }

  Point::data_type DimensionAccess::operator()(const Point& equationIndexes) const
  {
    return map(equationIndexes);
  }

  IndexSet DimensionAccess::operator()(const MultidimensionalRange& range) const
  {
    return map(range);
  }

  bool DimensionAccess::isConstantAccess() const
  {
    return constantAccess;
  }

  bool DimensionAccess::hasArray() const
  {
    return !array.empty();
  }

  Point::data_type DimensionAccess::getPosition() const
  {
    assert(isConstantAccess());
    assert(array.empty());
    return position;
  }

  Point::data_type DimensionAccess::getOffset() const
  {
    assert(!isConstantAccess());
    assert(!hasArray());
    return position;
  }

  llvm::ArrayRef<Point::data_type> DimensionAccess::getArray() const
  {
    assert(hasArray());
    return array;
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
    if (!array.empty()) {
      auto index = equationIndexes[getInductionVariableIndex()]-1;
      return array[index];
    }

    return equationIndexes[getInductionVariableIndex()] + getOffset();
  }

  IndexSet DimensionAccess::map(const MultidimensionalRange& range) const
  {
    if (isConstantAccess()) {
      return IndexSet(MultidimensionalRange(Range(getPosition(), getPosition() + 1)));
    }
    auto accessedDimensionIndex = getInductionVariableIndex();
    assert(accessedDimensionIndex < range.rank());
    const auto& sourceRange = range[accessedDimensionIndex];

    if (!array.empty()) {
      IndexSet result;
      for (auto index: sourceRange)
        result += array[index - 1];
      
      return result;
    }

    return IndexSet(MultidimensionalRange(Range(sourceRange.getBegin() + getOffset(), sourceRange.getEnd() + getOffset())));
  }

  std::ostream& operator<<(std::ostream& stream, const DimensionAccess& obj)
  {
    stream << "[";

    if (obj.isConstantAccess()) {
      stream << obj.getPosition();
    } else if (obj.hasArray()){
      stream << "{";

      std::string separator;
      for (const auto val: obj.getArray()){
        stream << separator << val;
        separator = ", ";
      }
      stream << "}[ i" << obj.getInductionVariableIndex() << "]";

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

    for (size_t i = 0; i < dimensionality; ++i)
      accesses.push_back(DimensionAccess::relative(i, 0));

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
      if(other.hasArray())
      {
        std::vector<long> array;

        for(auto val: mapped.getArray())
          array.push_back( mapped.getPosition() + val );
          
        return DimensionAccess::relativeToArray(other.getInductionVariableIndex(), array);
      }

      return DimensionAccess::constant(mapped.getPosition() + other.getOffset());
    }

    if(mapped.hasArray())
    {
      if(other.hasArray())
      {
        std::vector<long> array;

        for(auto [a,b]: llvm::zip(mapped.getArray(),other.getArray()))
          array.push_back( a + b );
          
        return DimensionAccess::relativeToArray(mapped.getInductionVariableIndex(), array);
      }
      std::vector<long> array;

      for(auto val: mapped.getArray())
        array.push_back( other.getOffset() + val );
        
      return DimensionAccess::relativeToArray(mapped.getInductionVariableIndex(), array);
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
        usedDimensions[function.getInductionVariableIndex()] = true;
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
      
      if(functions[i].hasArray())
      {
        std::vector<long> array;

        for(auto val: functions[i].getArray())
          array.push_back( -1 * val );
          
        remapped.push_back(DimensionAccess::relativeToArray(i, array));
      }
      else
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

  IndexSet AccessFunction::map(const MultidimensionalRange& range) const
  {
    return map(IndexSet(range));
  }

  IndexSet AccessFunction::map(const IndexSet& indexes) const
  {
    IndexSet result;

    for (const auto& range : indexes) {
      result += map(range);
    }

    return result;
  }

  IndexSet AccessFunction::inverseMap(const MultidimensionalRange& range) const
  {
    return inverse().map(IndexSet(range));
  }

  IndexSet AccessFunction::inverseMap(const IndexSet& indexes) const
  {
    return inverse().map(indexes);
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
