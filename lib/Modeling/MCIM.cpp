#include "llvm/Support/Casting.h"
#include "marco/Modeling/MCIM.h"
#include "marco/Modeling/MCIMImpl.h"
#include <numeric>

using namespace ::marco;
using namespace ::marco::modeling;
using namespace ::marco::modeling::internal;

//===----------------------------------------------------------------------===//
// MCIM iterator
//===----------------------------------------------------------------------===//

namespace marco::modeling::internal
{
  MCIM::IndicesIterator::IndicesIterator(
      const MultidimensionalRange& equationRange,
      const MultidimensionalRange& variableRange,
      std::function<MultidimensionalRange::const_iterator(const MultidimensionalRange&)> initFunction)
      : eqCurrentIt(initFunction(equationRange)),
        eqEndIt(equationRange.end()),
        varBeginIt(variableRange.begin()),
        varCurrentIt(initFunction(variableRange)),
        varEndIt(variableRange.end())
  {
    assert(eqCurrentIt == eqEndIt || varCurrentIt != varEndIt);
  }

  bool MCIM::IndicesIterator::operator==(const MCIM::IndicesIterator& it) const
  {
    return eqCurrentIt == it.eqCurrentIt && eqEndIt == it.eqEndIt && varBeginIt == it.varBeginIt
        && varCurrentIt == it.varCurrentIt && varEndIt == it.varEndIt;
  }

  bool MCIM::IndicesIterator::operator!=(const MCIM::IndicesIterator& it) const
  {
    return eqCurrentIt != it.eqCurrentIt || eqEndIt != it.eqEndIt || varBeginIt != it.varBeginIt
        || varCurrentIt != it.varCurrentIt || varEndIt != it.varEndIt;
  }

  MCIM::IndicesIterator& MCIM::IndicesIterator::operator++()
  {
    advance();
    return *this;
  }

  MCIM::IndicesIterator MCIM::IndicesIterator::operator++(int)
  {
    auto temp = *this;
    advance();
    return temp;
  }

  MCIM::IndicesIterator::value_type MCIM::IndicesIterator::operator*() const
  {
    return std::make_pair(*eqCurrentIt, *varCurrentIt);
  }

  void MCIM::IndicesIterator::advance()
  {
    if (eqCurrentIt == eqEndIt) {
      return;
    }

    ++varCurrentIt;

    if (varCurrentIt == varEndIt) {
      ++eqCurrentIt;

      if (eqCurrentIt == eqEndIt) {
        return;
      }

      varCurrentIt = varBeginIt;
    }
  }
}

//===----------------------------------------------------------------------===//
// MCIM implementation
//===----------------------------------------------------------------------===//

namespace marco::modeling::internal
{
  MCIM::Impl::Impl(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
      : equationRanges(std::move(equationRanges)), variableRanges(std::move(variableRanges))
  {
  }

  MCIM::Impl::~Impl() = default;

  std::unique_ptr<MCIM::Impl> MCIM::Impl::clone()
  {
    return std::make_unique<Impl>(*this);
  }

  const MultidimensionalRange& MCIM::Impl::getEquationRanges() const
  {
    return equationRanges;
  }

  const MultidimensionalRange& MCIM::Impl::getVariableRanges() const
  {
    return variableRanges;
  }

  bool MCIM::Impl::operator==(const MCIM::Impl& rhs) const
  {
    if (equationRanges != rhs.equationRanges) {
      return false;
    }

    if (variableRanges != rhs.variableRanges) {
      return false;
    }

    for (const auto& [equation, variable] : getIndices()) {
      if (get(equation, variable) != rhs.get(equation, variable)) {
        return false;
      }
    }

    return true;
  }

  bool MCIM::Impl::operator!=(const MCIM::Impl& rhs) const
  {
    if (getEquationRanges() != rhs.getEquationRanges()) {
      return true;
    }

    if (getVariableRanges() != rhs.getVariableRanges()) {
      return true;
    }

    for (const auto& [equation, variable] : getIndices()) {
      if (get(equation, variable) != rhs.get(equation, variable)) {
        return true;
      }
    }

    return false;
  }

  llvm::iterator_range<MCIM::IndicesIterator> MCIM::Impl::getIndices() const
  {
    IndicesIterator begin(equationRanges, variableRanges, [](const MultidimensionalRange& range) {
      return range.begin();
    });

    IndicesIterator end(equationRanges, variableRanges, [](const MultidimensionalRange& range) {
      return range.end();
    });

    return llvm::iterator_range<MCIM::IndicesIterator>(begin, end);
  }

  MCIM::Impl& MCIM::Impl::operator+=(const MCIM::Impl& rhs)
  {
    assert(equationRanges == rhs.equationRanges && "Different equation ranges");
    assert(variableRanges == rhs.variableRanges && "Different variable ranges");

    for (const auto& group : rhs.groups) {
      add(group.second.getKeys(), group.first);
    }

    return *this;
  }

  MCIM::Impl& MCIM::Impl::operator-=(const MCIM::Impl& rhs)
  {
    assert(equationRanges == rhs.equationRanges && "Different equation ranges");
    assert(variableRanges == rhs.variableRanges && "Different variable ranges");

    std::map<Delta, MCIMElement> newGroups;

    for (const auto& group : groups) {
      auto groupIt = rhs.groups.find(group.first);

      if (groupIt == rhs.groups.end()) {
        newGroups.try_emplace(group.first, std::move(group.second));
      } else {
        IndexSet diff = group.second.getKeys() - groupIt->second.getKeys();
        newGroups.try_emplace(group.first, MCIMElement(std::move(diff)));
      }
    }

    groups = std::move(newGroups);
    return *this;
  }

  void MCIM::Impl::apply(const AccessFunction& access)
  {
    apply(equationRanges, access);
  }

  void MCIM::Impl::apply(const MultidimensionalRange& equations, const AccessFunction& access)
  {
    assert(equationRanges.contains(equations));

    bool isIdentityLike = llvm::all_of(llvm::enumerate(access), [](const auto& dimensionAccess) {
      if (dimensionAccess.value().isConstantAccess()) {
        return false;
      }

      return dimensionAccess.index() == dimensionAccess.value().getInductionVariableIndex();
    });

    if (isIdentityLike) {
      auto mappedVariableRanges = access.map(equations);
      set(equations, mappedVariableRanges);

    } else {
      // In case of constant accesses or out-of-order induction variables
      // the delta would be wrong. Thus, we need to iterate over all the
      // equations indices.

      for (const auto& equationIndices : equations) {
        auto variableIndices = access.map(equationIndices);
        set(equationIndices, variableIndices);
      }
    }
  }

  bool MCIM::Impl::get(const Point& equation, const Point& variable) const
  {
    assert(equationRanges.contains(equation) && "Equation indices don't belong to the equation ranges");
    assert(variableRanges.contains(variable) && "Variable indices don't belong to the variable ranges");

    auto delta = getDelta(equation, variable);
    const auto& key = getKey(equation, variable);

    return llvm::any_of(groups, [&](const auto& group) -> bool {
      return group.first == delta && group.second.getKeys().contains(key);
    });
  }

  void MCIM::Impl::set(const Point& equation, const Point& variable)
  {
    assert(equationRanges.contains(equation) && "Equation indices don't belong to the equation ranges");
    assert(variableRanges.contains(variable) && "Variable indices don't belong to the variable ranges");

    auto delta = getDelta(equation, variable);
    IndexSet keys(getKey(equation, variable));
    add(std::move(keys), std::move(delta));
  }

  void MCIM::Impl::set(const MultidimensionalRange& equations, const MultidimensionalRange& variables)
  {
    assert(equations.rank() == getEquationRanges().rank());
    assert(variables.rank() == getVariableRanges().rank());

    assert(equations.rank() < variables.rank() || llvm::all_of(equations, [&](const auto& equation) {
             auto delta = getDelta(equations, variables);
             IndexSet key(equation);
             MCIMElement group(key);

             IndexSet reducedValues;

             auto groupValues = group.getValues(delta);
             for (const auto& value : groupValues.getRanges()) {
               reducedValues += value.slice(variables.rank());
             }

             return llvm::none_of(reducedValues.getRanges(), [&](const auto& range) {
               return llvm::any_of(range, [&](const auto& point) {
                 return !variables.contains(point);
               });
             });
           }));

    assert(equations.rank() >= variables.rank() || llvm::all_of(variables, [&](const auto& variable) {
             auto delta = getDelta(equations, variables);
             IndexSet key(variable);
             MCIMElement group(key);

             IndexSet reducedValues;

             auto groupValues = group.getValues(delta);
             for (const auto& value : groupValues.getRanges()) {
               reducedValues += value.slice(equations.rank());
             }

             return llvm::none_of(reducedValues.getRanges(), [&](const auto& range) {
               return llvm::any_of(range, [&](const auto& point) {
                 return !equations.contains(point);
               });
             });
           }));

    IndexSet keys(getKey(equations, variables));
    auto delta = getDelta(equations, variables);

    add(std::move(keys), std::move(delta));
  }

  void MCIM::Impl::unset(const Point& equation, const Point& variable)
  {
    assert(equationRanges.contains(equation) && "Equation indices don't belong to the equation ranges");
    assert(variableRanges.contains(variable) && "Variable indices don't belong to the variable ranges");

    const auto& key = getKey(equation, variable);
    MultidimensionalRange keyRange(key);

    std::map<Delta, MCIMElement> newGroups;

    for (const auto& group : groups) {
      IndexSet diff = group.second.getKeys() - keyRange;

      if (!diff.empty()) {
        newGroups.try_emplace(group.first, MCIMElement(std::move(diff)));
      }
    }

    groups = std::move(newGroups);
  }

  bool MCIM::Impl::empty() const
  {
    return groups.empty();
  }

  void MCIM::Impl::clear()
  {
    groups.clear();
  }

  IndexSet MCIM::Impl::flattenRows() const
  {
    IndexSet result;

    if (equationRanges.rank() >= variableRanges.rank()) {
      for (const auto& group : groups) {
        auto values = group.second.getValues(group.first);
        for (const auto& range : values.getRanges()) {
          result += range.slice(variableRanges.rank());
        }
      }
    } else {
      for (const auto& group : groups) {
        const auto& keys = group.second.getKeys();
        assert(keys.rank() == variableRanges.rank());
        result += keys;
      }
    }

    return result;
  }

  IndexSet MCIM::Impl::flattenColumns() const
  {
    IndexSet result;

    if (equationRanges.rank() >= variableRanges.rank()) {
      for (const auto& group : groups) {
        const auto& keys = group.second.getKeys();
        assert(keys.rank() == equationRanges.rank());
        result += keys;
      }
    } else {
      for (const auto& group : groups) {
        auto values = group.second.getValues(group.first);
        for (const auto& range : values.getRanges()) {
          result += range.slice(equationRanges.rank());
        }
      }
    }

    return result;
  }

  std::unique_ptr<MCIM::Impl> MCIM::Impl::filterRows(const IndexSet& filter) const
  {
    auto result = std::make_unique<MCIM::Impl>(equationRanges, variableRanges);

    if (equationRanges.rank() >= variableRanges.rank()) {
      for (const auto& group : groups) {
        if (auto& equations = group.second.getKeys(); equations.overlaps(filter)) {
          result->add(equations.intersect(filter), group.first);
        }
      }
    } else {
      auto rankDifference = variableRanges.rank() - equationRanges.rank();

      for (const auto& group : groups) {
        auto invertedGroup = group.second.inverse(group.first);
        IndexSet equations;

        auto invertedKeys = invertedGroup.getKeys();
        for (const auto& extendedEquations : invertedKeys.getRanges()) {
          equations += extendedEquations.slice(equationRanges.rank());
        }

        if (equations.overlaps(filter)) {
          IndexSet filteredEquations = equations.intersect(filter);
          IndexSet filteredExtendedEquations;

          for (const auto& filteredEquation : filteredEquations.getRanges()) {
            std::vector<Range> ranges;

            for (size_t i = 0; i < filteredEquation.rank(); ++i) {
              ranges.push_back(std::move(filteredEquation[i]));
            }

            for (size_t i = 0; i < rankDifference; ++i) {
              ranges.push_back(Range(0, 1));
            }

            filteredExtendedEquations += MultidimensionalRange(std::move(ranges));
          }

          MCIMElement filteredEquationGroup(std::move(filteredExtendedEquations));
          MCIMElement filteredVariables = filteredEquationGroup.inverse(group.first.inverse());
          result->add(std::move(filteredVariables.getKeys()), group.first);
        }
      }
    }

    return result;
  }

  std::unique_ptr<MCIM::Impl> MCIM::Impl::filterColumns(const IndexSet& filter) const
  {
    auto result = std::make_unique<MCIM::Impl>(equationRanges, variableRanges);

    if (equationRanges.rank() == variableRanges.rank()) {
      for (const auto& group : groups) {
        auto invertedGroup = group.second.inverse(group.first);
        const auto& variables = invertedGroup.getKeys();

        if (variables.overlaps(filter)) {
          IndexSet filteredVariables = variables.intersect(filter);
          MCIMElement filteredVariableGroup(std::move(filteredVariables));
          MCIMElement filteredEquations = filteredVariableGroup.inverse(group.first.inverse());
          result->add(std::move(filteredEquations.getKeys()), group.first);
        }
      }
    } else if (equationRanges.rank() > variableRanges.rank()) {
      auto rankDifference = equationRanges.rank() - variableRanges.rank();

      for (const auto& group : groups) {
        auto invertedGroup = group.second.inverse(group.first);
        IndexSet variables;

        auto invertedKeys = invertedGroup.getKeys();
        for (const auto& extendedVariables : invertedKeys.getRanges()) {
          variables += extendedVariables.slice(variableRanges.rank());
        }

        if (variables.overlaps(filter)) {
          IndexSet filteredVariables = variables.intersect(filter);
          IndexSet filteredExtendedVariables;

          for (const auto& filteredVariable : filteredVariables.getRanges()) {
            std::vector<Range> ranges;

            for (size_t i = 0; i < filteredVariable.rank(); ++i) {
              ranges.push_back(std::move(filteredVariable[i]));
            }

            for (size_t i = 0; i < rankDifference; ++i) {
              ranges.push_back(Range(0, 1));
            }

            filteredExtendedVariables += MultidimensionalRange(std::move(ranges));
          }

          MCIMElement filteredVariableGroup(std::move(filteredExtendedVariables));
          MCIMElement filteredEquations = filteredVariableGroup.inverse(group.first.inverse());
          result->add(std::move(filteredEquations.getKeys()), group.first);
        }
      }
    } else {
      for (const auto& group : groups) {
        if (auto& equations = group.second.getKeys(); equations.overlaps(filter)) {
          result->add(equations.intersect(filter), group.first);
        }
      }
    }

    return result;
  }

  std::vector<std::unique_ptr<MCIM::Impl>> MCIM::Impl::splitGroups() const
  {
    std::vector<std::unique_ptr<MCIM::Impl>> result;

    for (const auto& group : groups) {
      auto entry = std::make_unique<MCIM::Impl>(equationRanges, variableRanges);
      entry->groups.insert(group);
      result.push_back(std::move(entry));
    }

    return result;
  }

  MCIM::Impl::Delta MCIM::Impl::getDelta(const Point& equation, const Point& variable) const
  {
    if (equation.rank() >= variable.rank()) {
      return Delta(equation, variable);
    }

    return Delta(variable, equation);
  }

  MCIM::Impl::Delta MCIM::Impl::getDelta(const MultidimensionalRange& equations, const MultidimensionalRange& variables) const
  {
    if (equations.rank() >= variables.rank()) {
      return Delta(equations, variables);
    }

    return Delta(variables, equations);
  }

  const Point& MCIM::Impl::getKey(const Point& equation, const Point& variable) const
  {
    if (equation.rank() >= variable.rank()) {
      return equation;
    }

    return variable;
  }

  const MultidimensionalRange& MCIM::Impl::getKey(const MultidimensionalRange& equations, const MultidimensionalRange& variables) const
  {
    if (equations.rank() >= variables.rank()) {
      return equations;
    }

    return variables;
  }

  void MCIM::Impl::add(IndexSet equations, Delta delta)
  {
    groups[delta].addKeys(std::move(equations));
  }
}

//===----------------------------------------------------------------------===//
// Delta
//===----------------------------------------------------------------------===//

namespace marco::modeling::internal
{
  MCIM::Impl::Delta::Delta(const Point& keys, const Point& values)
  {
    assert(keys.rank() >= values.rank());
    auto rankDifference = keys.rank() - values.rank();
    auto minRank = std::min(keys.rank(), values.rank());

    for (size_t i = 0; i < minRank; ++i) {
      offsets.push_back(values[i] - keys[i]);
    }

    for (size_t i = 0; i < rankDifference; ++i) {
      offsets.push_back(-1 * keys[values.rank() + i]);
    }
  }

  MCIM::Impl::Delta::Delta(const MultidimensionalRange& keys, const MultidimensionalRange& values)
  {
    assert(keys.rank() >= values.rank());
    auto rankDifference = keys.rank() - values.rank();
    auto minRank = std::min(keys.rank(), values.rank());

    for (size_t i = 0; i < minRank; ++i) {
      offsets.push_back(values[i].getBegin() - keys[i].getBegin());
    }

    for (size_t i = 0; i < rankDifference; ++i) {
      offsets.push_back(-1 * keys[values.rank() + i].getBegin());
    }
  }

  bool MCIM::Impl::Delta::operator==(const MCIM::Impl::Delta& other) const
  {
    return offsets == other.offsets;
  }

  bool MCIM::Impl::Delta::operator<(const Delta& other) const
  {
    assert(size() == other.size());

    for (const auto& [lhs, rhs] : llvm::zip(offsets, other.offsets)) {
      if (rhs > lhs) {
        return true;
      }

      if (rhs < lhs) {
        return false;
      }
    }

    return false;
  }

  long MCIM::Impl::Delta::operator[](size_t index) const
  {
    assert(index < offsets.size());
    return offsets[index];
  }

  size_t MCIM::Impl::Delta::size() const
  {
    return offsets.size();
  }

  MCIM::Impl::Delta MCIM::Impl::Delta::inverse() const
  {
    Delta result(*this);

    for (auto& value : result.offsets) {
      value *= -1;
    }

    return result;
  }
}

//===----------------------------------------------------------------------===//
// MCIMElement
//===----------------------------------------------------------------------===//

namespace marco::modeling::internal
{
  MCIM::Impl::MCIMElement::MCIMElement() = default;

  MCIM::Impl::MCIMElement::MCIMElement(IndexSet keys)
      : keys(std::move(keys))
  {
  }

  const IndexSet& MCIM::Impl::MCIMElement::getKeys() const
  {
    return keys;
  }

  void MCIM::Impl::MCIMElement::addKeys(IndexSet newKeys)
  {
    keys += std::move(newKeys);
  }

  IndexSet MCIM::Impl::MCIMElement::getValues(const Delta& delta) const
  {
    IndexSet result;

    for (const auto& keyRange : keys.getRanges()) {
      std::vector<Range> valueRanges;

      for (size_t i = 0, e = keyRange.rank(); i < e; ++i) {
        valueRanges.emplace_back(keyRange[i].getBegin() + delta[i], keyRange[i].getEnd() + delta[i]);
      }

      result += MultidimensionalRange(valueRanges);
    }

    return result;
  }

  MCIM::Impl::MCIMElement MCIM::Impl::MCIMElement::inverse(const Delta& delta) const
  {
    return MCIM::Impl::MCIMElement(getValues(delta));
  }
}

//===----------------------------------------------------------------------===//
// MCIM
//===----------------------------------------------------------------------===//

namespace marco::modeling::internal
{
  MCIM::MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
    : impl(std::make_unique<Impl>(std::move(equationRanges), std::move(variableRanges)))
  {
  }

  MCIM::MCIM(std::unique_ptr<Impl> impl)
    : impl(std::move(impl))
  {
  }

  MCIM::MCIM(const MCIM& other)
    : impl(other.impl->clone())
  {
  }

  MCIM::MCIM(MCIM&& other) = default;

  MCIM::~MCIM() = default;

  MCIM& MCIM::operator=(const MCIM& other)
  {
    MCIM result(other);
    swap(*this, result);
    return *this;
  }

  MCIM& MCIM::operator=(MCIM&& other) = default;

  void swap(MCIM& first, MCIM& second)
  {
    using std::swap;
    swap(first.impl, second.impl);
  }

  bool MCIM::operator==(const MCIM& other) const
  {
    return *impl == *other.impl;
  }

  bool MCIM::operator!=(const MCIM& other) const
  {
    return *impl != *other.impl;
  }

  const MultidimensionalRange& MCIM::getEquationRanges() const
  {
    return impl->getEquationRanges();
  }

  const MultidimensionalRange& MCIM::getVariableRanges() const
  {
    return impl->getVariableRanges();
  }

  llvm::iterator_range<MCIM::IndicesIterator> MCIM::getIndices() const
  {
    return impl->getIndices();
  }

  MCIM& MCIM::operator+=(const MCIM& rhs)
  {
    *impl += *rhs.impl;
    return *this;
  }

  MCIM MCIM::operator+(const MCIM& rhs) const
  {
    MCIM result = *this;
    result += rhs;
    return result;
  }

  MCIM& MCIM::operator-=(const MCIM& rhs)
  {
    *impl -= *rhs.impl;
    return *this;
  }

  MCIM MCIM::operator-(const MCIM& rhs) const
  {
    MCIM result = *this;
    result -= rhs;
    return result;
  }

  void MCIM::apply(const AccessFunction& access)
  {
    impl->apply(access);
  }

  void MCIM::apply(const MultidimensionalRange& equations, const AccessFunction& access)
  {
    impl->apply(equations, access);
  }

  bool MCIM::get(const Point& equation, const Point& variable) const
  {
    return impl->get(equation, variable);
  }

  void MCIM::set(const Point& equation, const Point& variable)
  {
    impl->set(equation, variable);
  }

  void MCIM::unset(const Point& equation, const Point& variable)
  {
    impl->unset(equation, variable);
  }

  bool MCIM::empty() const
  {
    return impl->empty();
  }

  void MCIM::clear()
  {
    impl->clear();
  }

  IndexSet MCIM::flattenRows() const
  {
    return impl->flattenRows();
  }

  IndexSet MCIM::flattenColumns() const
  {
    return impl->flattenColumns();
  }

  MCIM MCIM::filterRows(const IndexSet& filter) const
  {
    return MCIM(impl->filterRows(filter));
  }

  MCIM MCIM::filterColumns(const IndexSet& filter) const
  {
    return MCIM(impl->filterColumns(filter));
  }

  std::vector<MCIM> MCIM::splitGroups() const
  {
    std::vector<MCIM> result;
    auto groups = impl->splitGroups();

    for (auto& group : groups) {
      result.push_back(MCIM(std::move(group)));
    }

    return result;
  }
}

namespace
{
  template<class T>
  static size_t numDigits(T value)
  {
    if (value > -10 && value < 10) {
      return 1;
    }

    size_t digits = 0;

    while (value != 0) {
      value /= 10;
      ++digits;
    }

    return digits;
  }
}

static size_t getRangeMaxColumns(const Range& range)
{
  size_t beginDigits = numDigits(range.getBegin());
  size_t endDigits = numDigits(range.getEnd());

  if (range.getBegin() < 0) {
    ++beginDigits;
  }

  if (range.getEnd() < 0) {
    ++endDigits;
  }

  return std::max(beginDigits, endDigits);
}

static size_t getIndicesWidth(const Point& indexes)
{
  size_t result = 0;

  for (const auto& index : indexes) {
    result += numDigits(index);

    if (index < 0) {
      ++result;
    }
  }

  return result;
}

static size_t getWrappedIndexesLength(size_t indexesLength, size_t numberOfIndexes)
{
  size_t result = indexesLength;

  result += 1; // '(' character
  result += numberOfIndexes - 1; // ',' characters
  result += 1; // ')' character

  return result;
}

namespace marco::modeling::internal
{
  std::ostream& operator<<(std::ostream& stream, const MCIM& mcim)
  {
    const auto& equationRanges = mcim.getEquationRanges();
    const auto& variableRanges = mcim.getVariableRanges();

    // Determine the max widths of the indexes of the equation, so that they
    // will be properly aligned.
    llvm::SmallVector<size_t, 3> equationIndexesCols;

    for (size_t i = 0, e = equationRanges.rank(); i < e; ++i) {
      equationIndexesCols.push_back(getRangeMaxColumns(equationRanges[i]));
    }

    size_t equationIndexesMaxWidth = std::accumulate(equationIndexesCols.begin(), equationIndexesCols.end(), static_cast<size_t>(0));
    size_t equationIndexesColumnWidth = getWrappedIndexesLength(equationIndexesMaxWidth, equationRanges.rank());

    // Determine the max column width, so that the horizontal spacing is the
    // same among all the items.
    llvm::SmallVector<size_t, 3> variableIndexesCols;

    for (size_t i = 0, e = variableRanges.rank(); i < e; ++i) {
      variableIndexesCols.push_back(getRangeMaxColumns(variableRanges[i]));
    }

    size_t variableIndexesMaxWidth = std::accumulate(variableIndexesCols.begin(), variableIndexesCols.end(), static_cast<size_t>(0));
    size_t variableIndexesColumnWidth = getWrappedIndexesLength(variableIndexesMaxWidth, variableRanges.rank());

    // Print the spacing of the first line
    for (size_t i = 0, e = equationIndexesColumnWidth; i < e; ++i) {
      stream << " ";
    }

    // Print the variable indexes
    for (const auto& variableIndexes : variableRanges) {
      stream << " ";
      size_t columnWidth = getIndicesWidth(variableIndexes);

      for (size_t i = columnWidth; i < variableIndexesMaxWidth; ++i) {
        stream << " ";
      }

      stream << variableIndexes;
    }

    // The first line containing the variable indexes is finished
    stream << "\n";

    // Print a line for each equation
    for (const auto& equation : equationRanges) {
      for (size_t i = getIndicesWidth(equation); i < equationIndexesMaxWidth; ++i) {
        stream << " ";
      }

      stream << equation;

      for (const auto& variable : variableRanges) {
        stream << " ";

        size_t columnWidth = variableIndexesColumnWidth;
        size_t spacesAfter = (columnWidth - 1) / 2;
        size_t spacesBefore = columnWidth - 1 - spacesAfter;

        for (size_t i = 0; i < spacesBefore; ++i) {
          stream << " ";
        }

        stream << (mcim.get(equation, variable) ? 1 : 0);

        for (size_t i = 0; i < spacesAfter; ++i) {
          stream << " ";
        }
      }

      stream << "\n";
    }

    return stream;
  }
}
