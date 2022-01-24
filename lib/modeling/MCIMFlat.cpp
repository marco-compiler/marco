#include "marco/modeling/MCIMFlat.h"
#include <numeric>

namespace marco::modeling::internal
{
  template<typename T = Point::data_type>
  static void convertIndexesToZeroBased(
      llvm::ArrayRef<T> indexes,
      const MultidimensionalRange& base,
      llvm::SmallVectorImpl<std::make_unsigned_t<T>>& rescaled)
  {
    assert(base.rank() == indexes.size());

    for (unsigned int i = 0, e = base.rank(); i < e; ++i) {
      const auto& monoDimRange = base[i];
      T index = indexes[i] - monoDimRange.getBegin();
      assert(index >= 0 && index < monoDimRange.getEnd() - monoDimRange.getBegin());
      rescaled.push_back(std::move(index));
    }
  }

  template<typename T = Point::data_type>
  static void convertIndexesFromZeroBased(
      llvm::ArrayRef<std::make_unsigned_t<T>> indexes,
      const MultidimensionalRange& base,
      llvm::SmallVectorImpl<T>& rescaled)
  {
    assert(base.rank() == indexes.size());

    for (size_t i = 0, e = base.rank(); i < e; ++i) {
      const auto& monoDimRange = base[i];
      T index = indexes[i] + monoDimRange.getBegin();
      assert(index < monoDimRange.getEnd());
      rescaled.push_back(index);
    }
  }

  /// Get the index to be used to access a flattened array.
  /// If an array is declared as [a][b][c], then the access [i][j][k] corresponds
  /// to the access [k + c * (j + b * (i))] of the flattened array of size
  /// [a * b * c].
  ///
  /// @param indexes 		access with respect to the original dimensions
  /// @param dimensions 	original array dimensions
  /// @return flattened index
  template<typename T = Point::data_type>
  static std::make_unsigned_t<T> flattenIndexes(
      llvm::ArrayRef<std::make_unsigned_t<T>> indexes,
      llvm::ArrayRef<size_t> dimensions)
  {
    assert(dimensions.size() == indexes.size());
    std::make_unsigned_t<T> result = 0;

    for (auto index: llvm::enumerate(indexes)) {
      result += index.value();

      if (index.index() < indexes.size() - 1) {
        result *= dimensions[index.index() + 1];
      }
    }

    return result;
  }

  /// Convert a flattened index into the ones to be used to access the array
  /// in its non-flattened version.
  ///
  /// @param dimensions  original array dimensions
  /// @param index       flattened index
  /// @param results     where the non-flattened access indexes are saved
  template<typename T = Point::data_type>
  static void unflattenIndex(
      llvm::ArrayRef<size_t> dimensions,
      std::make_unsigned_t<T> index,
      llvm::SmallVectorImpl<std::make_unsigned_t<T>>& results)
  {
    assert(dimensions.size() != 0);

    size_t totalSize = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
    assert(index < totalSize && "Flattened index exceeds the flat array size");

    size_t size = 1;

    for (size_t i = 1, e = dimensions.size(); i < e; ++i) {
      size *= dimensions[i];
    }

    for (size_t i = 1, e = dimensions.size(); i < e; ++i) {
      results.push_back(index / size);
      index %= size;
      size /= dimensions[i];
    }

    results.push_back(index);

    assert(size == 1);
    assert(results.size() == dimensions.size());
  }

  static MCIS flattenMCIS(const MCIS& value, const MultidimensionalRange& range, llvm::ArrayRef<size_t> dimensions)
  {
    MCIS result;

    for (const auto& multiDimRange: value) {
      llvm::SmallVector<Point::data_type, 3> firstItemIndexes;
      llvm::SmallVector<Point::data_type, 3> lastItemIndexes;

      for (size_t i = 0, e = multiDimRange.rank(); i < e; ++i) {
        const auto& monoDimRange = multiDimRange[i];
        firstItemIndexes.push_back(monoDimRange.getBegin());
        lastItemIndexes.push_back(monoDimRange.getEnd() - 1);
      }

      llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> firstItemRescaled;
      convertIndexesToZeroBased<Point::data_type>(firstItemIndexes, range, firstItemRescaled);

      llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> lastItemRescaled;
      convertIndexesToZeroBased<Point::data_type>(lastItemIndexes, range, lastItemRescaled);

      auto firstItemFlattened = flattenIndexes(firstItemRescaled, dimensions);
      auto lastItemFlattened = flattenIndexes(lastItemRescaled, dimensions);

      result += MultidimensionalRange(Range(firstItemFlattened, lastItemFlattened + 1));
    }

    return result;
  }

  static MCIS unflattenMCIS(const MCIS& value, const MultidimensionalRange& range, llvm::ArrayRef<size_t> dimensions)
  {
    MCIS result;

    for (const auto& multiDimRange: value) {
      assert(multiDimRange.rank() == 1);
      auto& monoDimRange = multiDimRange[0];

      std::make_unsigned_t<Point::data_type> firstItemFlattened = monoDimRange.getBegin();
      std::make_unsigned_t<Point::data_type> lastItemFlattened = monoDimRange.getEnd() - 1;

      for (auto flattened = firstItemFlattened; flattened <= lastItemFlattened; ++flattened) {
        llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> rescaled;
        unflattenIndex(dimensions, flattened, rescaled);

        llvm::SmallVector<Point::data_type, 3> indexes;
        convertIndexesFromZeroBased(rescaled, range, indexes);

        llvm::SmallVector<Range, 3> ranges;

        for (const auto& index: indexes) {
          ranges.emplace_back(index, index + 1);
        }

        result += MultidimensionalRange(std::move(ranges));
      }
    }

    return result;
  }

  FlatMCIM::Delta::Delta(size_t key, size_t value) : value(value - key)
  {
  }

  bool FlatMCIM::Delta::operator==(const Delta& other) const
  {
    return value == other.value;
  }

  long FlatMCIM::Delta::getValue() const
  {
    return value;
  }

  FlatMCIM::Delta FlatMCIM::Delta::inverse() const
  {
    Delta result(*this);
    result.value *= -1;
    return result;
  }

  FlatMCIM::MCIMElement::MCIMElement(MCIS keys, Delta delta)
      : keys(std::move(keys)), delta(std::move(delta))
  {
  }

  const MCIS& FlatMCIM::MCIMElement::getKeys() const
  {
    return keys;
  }

  void FlatMCIM::MCIMElement::addKeys(MCIS newKeys)
  {
    keys += std::move(newKeys);
  }

  const FlatMCIM::Delta& FlatMCIM::MCIMElement::getDelta() const
  {
    return delta;
  }

  MCIS FlatMCIM::MCIMElement::getValues() const
  {
    MCIS result;

    for (const auto& keyRange: keys) {
      llvm::SmallVector<Range, 3> valueRanges;

      for (size_t i = 0, e = keyRange.rank(); i < e; ++i) {
        valueRanges.emplace_back(keyRange[i].getBegin() + delta.getValue(), keyRange[i].getEnd() + delta.getValue());
      }

      result += MultidimensionalRange(valueRanges);
    }

    return result;
  }

  FlatMCIM::MCIMElement FlatMCIM::MCIMElement::inverse() const
  {
    return FlatMCIM::MCIMElement(getValues(), delta.inverse());
  }

  FlatMCIM::FlatMCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
      : MCIM::Impl(Flat, std::move(equationRanges), std::move(variableRanges))
  {
    assert(getEquationRanges().rank() != getVariableRanges().rank());

    for (size_t i = 0, e = getEquationRanges().rank(); i < e; ++i) {
      equationDimensions.push_back(getEquationRanges()[i].size());
    }

    for (size_t i = 0, e = getVariableRanges().rank(); i < e; ++i) {
      variableDimensions.push_back(getVariableRanges()[i].size());
    }
  }

  bool FlatMCIM::operator==(const MCIM::Impl& rhs) const
  {
    if (auto other = rhs.dyn_cast<FlatMCIM>()) {
      if (groups.empty() && other->groups.empty()) {
        return true;
      }

      if (groups.size() != other->groups.size()) {
        return false;
      }

      for (const auto& group: other->groups) {
        auto groupIt = llvm::find_if(groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
        });

        if (groupIt == groups.end()) {
          return false;
        }

        if (group.getKeys() != groupIt->getKeys()) {
          return false;
        }
      }

      return true;
    }

    return MCIM::Impl::operator==(rhs);
  }

  bool FlatMCIM::operator!=(const MCIM::Impl& rhs) const
  {
    if (auto other = rhs.dyn_cast<FlatMCIM>()) {
      if (groups.empty() && other->groups.empty()) {
        return false;
      }

      if (groups.size() != other->groups.size()) {
        return true;
      }

      for (const auto& group: other->groups) {
        auto groupIt = llvm::find_if(groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
        });

        if (groupIt == groups.end()) {
          return true;
        }

        if (group.getKeys() != groupIt->getKeys()) {
          return true;
        }
      }

      return false;
    }

    return MCIM::Impl::operator==(rhs);
  }

  std::unique_ptr<MCIM::Impl> FlatMCIM::clone()
  {
    return std::make_unique<FlatMCIM>(*this);
  }

  MCIM::Impl& FlatMCIM::operator+=(const MCIM::Impl& rhs)
  {
    if (auto other = rhs.dyn_cast<FlatMCIM>()) {
      for (const auto& group: other->groups) {
        add(group.getKeys(), group.getDelta());
      }

      return *this;
    }

    return MCIM::Impl::operator+=(rhs);
  }

  MCIM::Impl& FlatMCIM::operator-=(const MCIM::Impl& rhs)
  {
    if (auto other = rhs.dyn_cast<FlatMCIM>()) {
      llvm::SmallVector<MCIMElement, 3> newGroups;

      for (const auto& group: groups) {
        auto groupIt = llvm::find_if(other->groups, [&](const MCIMElement& obj) {
          return obj.getDelta() == group.getDelta();
        });

        if (groupIt == other->groups.end()) {
          newGroups.push_back(std::move(group));
        } else {
          MCIS diff = group.getKeys() - groupIt->getKeys();
          newGroups.emplace_back(std::move(diff), std::move(group.getDelta()));
        }
      }

      groups = std::move(newGroups);
      return *this;
    }

    return MCIM::Impl::operator+=(rhs);
  }

  void FlatMCIM::apply(const AccessFunction& access)
  {
    for (const auto& equation: getEquationRanges()) {
      assert(access.size() == getVariableRanges().rank());
      auto variable = access.map(equation);
      set(equation, variable);
    }
  }

  bool FlatMCIM::get(const Point& equation, const Point& variable) const
  {
    auto flatEquation = getFlatEquation(equation);
    auto flatVariable = getFlatVariable(variable);

    Delta delta(flatEquation[0], flatVariable[0]);

    return llvm::any_of(groups, [&](const MCIMElement& group) -> bool {
      return group.getDelta() == delta && group.getKeys().contains(flatEquation);
    });
  }

  void FlatMCIM::set(const Point& equation, const Point& variable)
  {
    auto flatEquation = getFlatEquation(equation);
    auto flatVariable = getFlatVariable(variable);

    Delta delta(flatEquation[0], flatVariable[0]);
    MCIS keys(MultidimensionalRange(Range(flatEquation[0], flatEquation[0]
    +1)));
    add(std::move(keys), std::move(delta));
  }

  void FlatMCIM::unset(const Point& equation, const Point& variable)
  {
    auto flatEquation = getFlatEquation(equation);

    llvm::SmallVector<MCIMElement, 3> newGroups;

    for (const auto& group: groups) {
      MCIS diff = group.getKeys() - MultidimensionalRange(Range(flatEquation[0], flatEquation[0] + 1));

      if (!diff.empty()) {
        newGroups.emplace_back(std::move(diff), std::move(group.getDelta()));
      }
    }

    groups = std::move(newGroups);
  }

  bool FlatMCIM::empty() const
  {
    return groups.empty();
  }

  void FlatMCIM::clear()
  {
    groups.clear();
  }

  MCIS FlatMCIM::flattenRows() const
  {
    MCIS result;

    for (const auto& group: groups) {
      result += group.getValues();
    }

    return unflattenMCIS(result, getVariableRanges(), variableDimensions);
  }

  MCIS FlatMCIM::flattenColumns() const
  {
    MCIS result;

    for (const auto& group: groups) {
      result += group.getKeys();
    }

    return unflattenMCIS(result, getEquationRanges(), equationDimensions);
  }

  std::unique_ptr<MCIM::Impl> FlatMCIM::filterRows(const MCIS& filter) const
  {
    MCIS flattenedFilter = flattenMCIS(filter, getEquationRanges(), equationDimensions);
    auto result = std::make_unique<FlatMCIM>(getEquationRanges(), getVariableRanges());

    for (const MCIMElement& group: groups) {
      if (auto& equations = group.getKeys(); equations.overlaps(flattenedFilter)) {
        result->add(equations.intersect(flattenedFilter), group.getDelta());
      }
    }

    return result;
  }

  std::unique_ptr<MCIM::Impl> FlatMCIM::filterColumns(const MCIS& filter) const
  {
    MCIS flattenedFilter = flattenMCIS(filter, getVariableRanges(), variableDimensions);
    auto result = std::make_unique<FlatMCIM>(getEquationRanges(), getVariableRanges());

    for (const auto& group: groups) {
      auto invertedGroup = group.inverse();

      if (auto& variables = invertedGroup.getKeys(); variables.overlaps(flattenedFilter)) {
        MCIS filteredVariables = variables.intersect(flattenedFilter);
        MCIMElement filteredVariableGroup(std::move(filteredVariables), invertedGroup.getDelta());
        MCIMElement filteredEquations = filteredVariableGroup.inverse();
        result->add(std::move(filteredEquations.getKeys()), std::move(filteredEquations.getDelta()));
      }
    }

    return result;
  }

  std::vector<std::unique_ptr<MCIM::Impl>> FlatMCIM::splitGroups() const
  {
    std::vector<std::unique_ptr<MCIM::Impl>> result;

    for (const auto& group: groups) {
      auto mcim = std::make_unique<FlatMCIM>(getEquationRanges(), getVariableRanges());
      mcim->groups.push_back(group);
      result.push_back(std::move(mcim));
    }

    return result;
  }

  Point FlatMCIM::getFlatEquation(const Point& equation) const
  {
    assert(equation.rank() == getEquationRanges().rank());

    llvm::SmallVector<Point::data_type, 3> indexes(equation.begin(), equation.end());

    llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> rescaled;
    convertIndexesToZeroBased<Point::data_type>(indexes, getEquationRanges(), rescaled);

    auto flattened = flattenIndexes(rescaled, equationDimensions);
    return Point(flattened);
  }

  Point FlatMCIM::getFlatVariable(const Point& variable) const
  {
    assert(variable.rank() == getVariableRanges().rank());

    llvm::SmallVector<Point::data_type, 3> indexes(variable.begin(), variable.end());

    llvm::SmallVector<std::make_unsigned_t<Point::data_type>, 3> rescaled;
    convertIndexesToZeroBased<Point::data_type>(indexes, getVariableRanges(), rescaled);

    auto flattened = flattenIndexes(rescaled, variableDimensions);
    return Point(flattened);
  }

  void FlatMCIM::add(MCIS keys, Delta delta)
  {
    auto groupIt = llvm::find_if(groups, [&](const MCIMElement& group) {
      return group.getDelta() == delta;
    });

    if (groupIt == groups.end()) {
      groups.emplace_back(std::move(keys), std::move(delta));
    } else {
      groupIt->addKeys(std::move(keys));
    }
  }
}
