#include "llvm/Support/Casting.h"
#include "marco/modeling/MCIM.h"
#include "marco/modeling/MCIMFlat.h"
#include "marco/modeling/MCIMRegular.h"
#include <numeric>

namespace marco::modeling::internal
{
  MCIM::IndexesIterator::IndexesIterator(
      const MultidimensionalRange& equationRange,
      const MultidimensionalRange& variableRange,
      std::function<MultidimensionalRange::const_iterator(const MultidimensionalRange&)> initFunction)
      : eqCurrentIt(initFunction(equationRange)),
        eqEndIt(equationRange.end()),
        varBeginIt(variableRange.begin()),
        varCurrentIt(initFunction(variableRange)),
        varEndIt(variableRange.end())
  {
    if (eqCurrentIt != eqEndIt) {
      assert(varCurrentIt != varEndIt);
    }
  }

  bool MCIM::IndexesIterator::operator==(const MCIM::IndexesIterator& it) const
  {
    return eqCurrentIt == it.eqCurrentIt && eqEndIt == it.eqEndIt && varBeginIt == it.varBeginIt
        && varCurrentIt == it.varCurrentIt && varEndIt == it.varEndIt;
  }

  bool MCIM::IndexesIterator::operator!=(const MCIM::IndexesIterator& it) const
  {
    return eqCurrentIt != it.eqCurrentIt || eqEndIt != it.eqEndIt || varBeginIt != it.varBeginIt
        || varCurrentIt != it.varCurrentIt || varEndIt != it.varEndIt;
  }

  MCIM::IndexesIterator& MCIM::IndexesIterator::operator++()
  {
    advance();
    return *this;
  }

  MCIM::IndexesIterator MCIM::IndexesIterator::operator++(int)
  {
    auto temp = *this;
    advance();
    return temp;
  }

  MCIM::IndexesIterator::value_type MCIM::IndexesIterator::operator*() const
  {
    return std::make_pair(*eqCurrentIt, *varCurrentIt);
  }

  void MCIM::IndexesIterator::advance()
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

  MCIM::Impl::Impl(MCIMKind kind, MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
      : kind(kind), equationRanges(std::move(equationRanges)), variableRanges(std::move(variableRanges))
  {
  }

  MCIM::Impl::Impl(const Impl& other) = default;

  MCIM::Impl::~Impl() = default;

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
    for (const auto&[equation, variable]: getIndexes()) {
      if (get(equation, variable) != rhs.get(equation, variable)) {
        return false;
      }
    }

    return true;
  }

  bool MCIM::Impl::operator!=(const MCIM::Impl& rhs) const
  {
    for (const auto&[equation, variable]: getIndexes()) {
      if (get(equation, variable) != rhs.get(equation, variable)) {
        return true;
      }
    }

    return false;
  }

  llvm::iterator_range<MCIM::IndexesIterator> MCIM::Impl::getIndexes() const
  {
    IndexesIterator begin(getEquationRanges(), getVariableRanges(), [](const MultidimensionalRange& range) {
      return range.begin();
    });

    IndexesIterator end(getEquationRanges(), getVariableRanges(), [](const MultidimensionalRange& range) {
      return range.end();
    });

    return llvm::iterator_range<MCIM::IndexesIterator>(begin, end);
  }

  MCIM::Impl& MCIM::Impl::operator+=(const MCIM::Impl& rhs)
  {
    for (const auto&[equation, variable]: getIndexes()) {
      if (rhs.get(equation, variable)) {
        set(equation, variable);
      }
    }

    return *this;
  }

  MCIM::Impl& MCIM::Impl::operator-=(const MCIM::Impl& rhs)
  {
    for (const auto&[equation, variable]: getIndexes()) {
      set(equation, variable);
    }

    for (const auto&[equation, variable]: getIndexes()) {
      if (rhs.get(equation, variable)) {
        unset(equation, variable);
      }
    }

    return *this;
  }

  MCIM::MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
  {
    if (equationRanges.rank() == variableRanges.rank()) {
      impl = std::make_unique<RegularMCIM>(std::move(equationRanges), std::move(variableRanges));
    } else {
      impl = std::make_unique<FlatMCIM>(std::move(equationRanges), std::move(variableRanges));
    }
  }

  MCIM::MCIM(std::unique_ptr<Impl> impl) : impl(std::move(impl))
  {
  }

  MCIM::MCIM(const MCIM& other) : impl(other.impl->clone())
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

  void swap(MCIM& first, MCIM& second)
  {
    using std::swap;
    swap(first.impl, second.impl);
  }

  bool MCIM::operator==(const MCIM& other) const
  {
    if (getEquationRanges() != other.getEquationRanges()) {
      return false;
    }

    if (getVariableRanges() != other.getVariableRanges()) {
      return false;
    }

    return (*impl) == *other.impl;
  }

  bool MCIM::operator!=(const MCIM& other) const
  {
    if (getEquationRanges() != other.getEquationRanges()) {
      return true;
    }

    if (getVariableRanges() != other.getVariableRanges()) {
      return true;
    }

    return (*impl) != *other.impl;
  }

  const MultidimensionalRange& MCIM::getEquationRanges() const
  {
    return impl->getEquationRanges();
  }

  const MultidimensionalRange& MCIM::getVariableRanges() const
  {
    return impl->getVariableRanges();
  }

  llvm::iterator_range<MCIM::IndexesIterator> MCIM::getIndexes() const
  {
    return impl->getIndexes();
  }

  MCIM& MCIM::operator+=(const MCIM& rhs)
  {
    assert(getEquationRanges() == rhs.getEquationRanges() && "Different equation ranges");
    assert(getVariableRanges() == rhs.getVariableRanges() && "Different variable ranges");

    (*impl) += *rhs.impl;
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
    assert(getEquationRanges() == rhs.getEquationRanges() && "Different equation ranges");
    assert(getVariableRanges() == rhs.getVariableRanges() && "Different variable ranges");

    (*impl) -= *rhs.impl;
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

  bool MCIM::get(const Point& equation, const Point& variable) const
  {
    assert(getEquationRanges().contains(equation) && "Equation indexes don't belong to the equation ranges");
    assert(getVariableRanges().contains(variable) && "Variable indexes don't belong to the variable ranges");
    return impl->get(equation, variable);
  }

  void MCIM::set(const Point& equation, const Point& variable)
  {
    assert(getEquationRanges().contains(equation) && "Equation indexes don't belong to the equation ranges");
    assert(getVariableRanges().contains(variable) && "Variable indexes don't belong to the variable ranges");
    impl->set(equation, variable);
  }

  void MCIM::unset(const Point& equation, const Point& variable)
  {
    assert(getEquationRanges().contains(equation) && "Equation indexes don't belong to the equation ranges");
    assert(getVariableRanges().contains(variable) && "Variable indexes don't belong to the variable ranges");
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

    for (auto& group: groups) {
      result.push_back(MCIM(std::move(group)));
    }

    return result;
  }

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

  static size_t getIndexesWidth(const Point& indexes)
  {
    size_t result = 0;

    for (const auto& index: indexes) {
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

    size_t equationIndexesMaxWidth = std::accumulate(equationIndexesCols.begin(), equationIndexesCols.end(), 0);
    size_t equationIndexesColumnWidth = getWrappedIndexesLength(equationIndexesMaxWidth, equationRanges.rank());

    // Determine the max column width, so that the horizontal spacing is the
    // same among all the items.
    llvm::SmallVector<size_t, 3> variableIndexesCols;

    for (size_t i = 0, e = variableRanges.rank(); i < e; ++i) {
      variableIndexesCols.push_back(getRangeMaxColumns(variableRanges[i]));
    }

    size_t variableIndexesMaxWidth = std::accumulate(variableIndexesCols.begin(), variableIndexesCols.end(), 0);
    size_t variableIndexesColumnWidth = getWrappedIndexesLength(variableIndexesMaxWidth, variableRanges.rank());

    // Print the spacing of the first line
    for (size_t i = 0, e = equationIndexesColumnWidth; i < e; ++i) {
      stream << " ";
    }

    // Print the variable indexes
    for (const auto& variableIndexes: variableRanges) {
      stream << " ";
      size_t columnWidth = getIndexesWidth(variableIndexes);

      for (size_t i = columnWidth; i < variableIndexesMaxWidth; ++i) {
        stream << " ";
      }

      stream << variableIndexes;
    }

    // The first line containing the variable indexes is finished
    stream << "\n";

    // Print a line for each equation
    for (const auto& equation: equationRanges) {
      for (size_t i = getIndexesWidth(equation); i < equationIndexesMaxWidth; ++i) {
        stream << " ";
      }

      stream << equation;

      for (const auto& variable: variableRanges) {
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
