#include "marco/Modeling/MCIM.h"
#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/LocalMatchingSolutions.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace ::marco;
using namespace ::marco::modeling;
using namespace ::marco::modeling::internal;

//===----------------------------------------------------------------------===//
// MCIM iterator
//===----------------------------------------------------------------------===//

namespace marco::modeling::internal {
MCIM::IndicesIterator::IndicesIterator(
    const IndexSet &equationRange, const IndexSet &variableRange,
    llvm::function_ref<IndexSet::const_point_iterator(const IndexSet &)>
        initFunction)
    : eqCurrentIt(initFunction(equationRange)), eqEndIt(equationRange.end()),
      varBeginIt(variableRange.begin()),
      varCurrentIt(initFunction(variableRange)), varEndIt(variableRange.end()) {
  assert(eqCurrentIt == eqEndIt || varCurrentIt != varEndIt);
}

bool MCIM::IndicesIterator::operator==(const MCIM::IndicesIterator &it) const {
  return eqCurrentIt == it.eqCurrentIt && eqEndIt == it.eqEndIt &&
         varBeginIt == it.varBeginIt && varCurrentIt == it.varCurrentIt &&
         varEndIt == it.varEndIt;
}

bool MCIM::IndicesIterator::operator!=(const MCIM::IndicesIterator &it) const {
  return eqCurrentIt != it.eqCurrentIt || eqEndIt != it.eqEndIt ||
         varBeginIt != it.varBeginIt || varCurrentIt != it.varCurrentIt ||
         varEndIt != it.varEndIt;
}

MCIM::IndicesIterator &MCIM::IndicesIterator::operator++() {
  advance();
  return *this;
}

MCIM::IndicesIterator MCIM::IndicesIterator::operator++(int) {
  auto temp = *this;
  advance();
  return temp;
}

MCIM::IndicesIterator::value_type MCIM::IndicesIterator::operator*() const {
  return std::make_pair(*eqCurrentIt, *varCurrentIt);
}

void MCIM::IndicesIterator::advance() {
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
} // namespace marco::modeling::internal

//===----------------------------------------------------------------------===//
// MCIM
//===----------------------------------------------------------------------===//

namespace marco::modeling::internal {
MCIM::MCIM(MultidimensionalRange equationRanges,
           MultidimensionalRange variableRanges)
    : equationRanges(std::move(equationRanges)),
      variableRanges(std::move(variableRanges)) {}

MCIM::MCIM(IndexSet equationRanges, IndexSet variableRanges)
    : equationRanges(std::move(equationRanges)),
      variableRanges(std::move(variableRanges)) {}

MCIM::MCIM(const MCIM &other)
    : equationRanges(other.equationRanges),
      variableRanges(other.variableRanges), points(other.points) {
  for (const auto &group : other.groups) {
    addGroup(group->clone());
  }
}

MCIM::MCIM(MCIM &&other) = default;

MCIM::~MCIM() = default;

MCIM &MCIM::operator=(const MCIM &other) {
  MCIM result(other);
  swap(*this, result);
  return *this;
}

MCIM &MCIM::operator=(MCIM &&other) = default;

void swap(MCIM &first, MCIM &second) {
  using std::swap;
  swap(first.equationRanges, second.equationRanges);
  swap(first.variableRanges, second.variableRanges);
  swap(first.groups, second.groups);
  swap(first.points, second.points);
}

const IndexSet &MCIM::getEquationRanges() const { return equationRanges; }

const IndexSet &MCIM::getVariableRanges() const { return variableRanges; }

bool MCIM::operator==(const MCIM &rhs) const {
  if (equationRanges != rhs.equationRanges) {
    return false;
  }

  if (variableRanges != rhs.variableRanges) {
    return false;
  }

  auto indices = llvm::make_range(indicesBegin(), indicesEnd());

  for (const auto &[equation, variable] : indices) {
    if (get(equation, variable) != rhs.get(equation, variable)) {
      return false;
    }
  }

  return true;
}

bool MCIM::operator!=(const MCIM &rhs) const {
  if (getEquationRanges() != rhs.getEquationRanges()) {
    return true;
  }

  if (getVariableRanges() != rhs.getVariableRanges()) {
    return true;
  }

  auto indices = llvm::make_range(indicesBegin(), indicesEnd());

  for (const auto &[equation, variable] : indices) {
    if (get(equation, variable) != rhs.get(equation, variable)) {
      return true;
    }
  }

  return false;
}

MCIM::IndicesIterator MCIM::indicesBegin() const {
  return IndicesIterator(equationRanges, variableRanges,
                         [](const IndexSet &range) { return range.begin(); });
}

MCIM::IndicesIterator MCIM::indicesEnd() const {
  return IndicesIterator(equationRanges, variableRanges,
                         [](const IndexSet &range) { return range.end(); });
}

MCIM &MCIM::operator+=(const MCIM &rhs) {
  assert(equationRanges == rhs.equationRanges && "Different equation ranges");
  assert(variableRanges == rhs.variableRanges && "Different variable ranges");

  for (const auto &group : rhs.groups) {
    apply(group->getKeys(), group->getAccessFunction());
  }

  for (Point otherPoint : rhs.points) {
    Point equation = otherPoint.takeFront(equationRanges.rank());
    Point variable = otherPoint.takeBack(variableRanges.rank());
    set(equation, variable);
  }

  assert(std::all_of(rhs.indicesBegin(), rhs.indicesEnd(),
                     [&](std::pair<Point, Point> coordinates) {
                       if (rhs.get(coordinates.first, coordinates.second)) {
                         return get(coordinates.first, coordinates.second);
                       }

                       return true;
                     }) &&
         "Not all points were correctly added");

  return *this;
}

MCIM MCIM::operator+(const MCIM &rhs) const {
  MCIM result = *this;
  result += rhs;
  return result;
}

MCIM &MCIM::operator-=(const MCIM &rhs) {
  assert(equationRanges == rhs.equationRanges && "Different equation ranges");
  assert(variableRanges == rhs.variableRanges && "Different variable ranges");

  std::vector<std::unique_ptr<MCIMGroup>> oldGroups;
  swap(oldGroups, groups);

  for (auto &rhsGroup : rhs.groups) {
    for (auto &group : oldGroups) {
      if (group->getAccessFunction() == rhsGroup->getAccessFunction()) {
        group->removeKeys(rhsGroup->getKeys());
      }
    }
  }

  for (auto &group : oldGroups) {
    if (!group->empty()) {
      // Remove the keys that overlap with the extra points.
      for (const MultidimensionalRange &range :
           llvm::make_range(rhs.points.rangesBegin(), rhs.points.rangesEnd())) {
        MultidimensionalRange equations =
            range.takeFirstDimensions(rhs.equationRanges.rank());

        MultidimensionalRange variables =
            range.takeLastDimensions(rhs.variableRanges.rank());

        IndexSet inverseKeys = group->getAccessFunction().inverseMap(
            IndexSet(variables), group->getKeys());

        IndexSet keysIntersection = inverseKeys.intersect(equations);
        group->removeKeys(keysIntersection);
      }
    }

    if (!group->empty()) {
      addGroup(std::move(group));
    }
  }

  points -= rhs.points;

  assert(std::all_of(rhs.indicesBegin(), rhs.indicesEnd(),
                     [&](std::pair<Point, Point> coordinates) {
                       if (rhs.get(coordinates.first, coordinates.second)) {
                         return !get(coordinates.first, coordinates.second);
                       }

                       return true;
                     }) &&
         "Not all points were correctly removed");

  return *this;
}

MCIM MCIM::operator-(const MCIM &rhs) const {
  MCIM result = *this;
  result -= rhs;
  return result;
}

void MCIM::apply(const AccessFunction &access) {
  apply(equationRanges, access);
}

void MCIM::apply(const MultidimensionalRange &equations,
                 const AccessFunction &access) {
  assert(equationRanges.contains(equations));
  apply(IndexSet(equations), access);
}

void MCIM::apply(const IndexSet &equations, const AccessFunction &access) {
  assert(equationRanges.contains(equations));

  auto groupIt =
      std::find_if(groups.begin(), groups.end(),
                   [&access](const std::unique_ptr<MCIMGroup> &group) {
                     return group->getAccessFunction() == access;
                   });

  if (groupIt != groups.end()) {
    (*groupIt)->addKeys(equations);
  } else {
    auto &group = addGroup(MCIMGroup::build(access));
    group.addKeys(equations);
  }
}

bool MCIM::get(const Point &equation, const Point &variable) const {
  assert(equationRanges.contains(equation) &&
         "Equation indices don't belong to the equation ranges");
  assert(variableRanges.contains(variable) &&
         "Variable indices don't belong to the variable ranges");

  if (points.contains(equation.append(variable))) {
    return true;
  }

  return llvm::any_of(groups, [&](const std::unique_ptr<MCIMGroup> &group) {
    return group->has(equation, variable);
  });
}

void MCIM::set(const Point &equation, const Point &variable) {
  assert(equationRanges.contains(equation) &&
         "Equation indices don't belong to the equation ranges");
  assert(variableRanges.contains(variable) &&
         "Variable indices don't belong to the variable ranges");

  bool anyGroupSet = false;

  for (auto &group : groups) {
    anyGroupSet |= group->set(equation, variable);
  }

  if (!anyGroupSet) {
    points += equation.append(variable);
  }
}

void MCIM::unset(const Point &equation, const Point &variable) {
  assert(equationRanges.contains(equation) &&
         "Equation indices don't belong to the equation ranges");
  assert(variableRanges.contains(variable) &&
         "Variable indices don't belong to the variable ranges");

  std::vector<std::unique_ptr<MCIMGroup>> newGroups;

  for (auto &group : groups) {
    if (group->unset(equation, variable)) {
      if (!group->empty()) {
        newGroups.push_back(std::move(group));
      }
    }
  }

  groups = std::move(newGroups);
  points -= equation.append(variable);
}

bool MCIM::empty() const { return groups.empty() && points.empty(); }

void MCIM::clear() {
  groups.clear();
  points.clear();
}

IndexSet MCIM::flattenRows() const {
  IndexSet result;

  for (const auto &group : groups) {
    result += group->getValues();
  }

  result += points.takeLastDimensions(variableRanges.rank());
  return result;
}

IndexSet MCIM::flattenColumns() const {
  IndexSet result;

  for (const auto &group : groups) {
    result += group->getKeys();
  }

  result += points.takeFirstDimensions(equationRanges.rank());
  return result;
}

MCIM MCIM::filterRows(const IndexSet &filter) const {
  MCIM result(equationRanges, variableRanges);

  for (const auto &group : groups) {
    std::unique_ptr<MCIMGroup> filteredGroup = group->filterKeys(filter);

    if (!filteredGroup->empty()) {
      result.addGroup(std::move(filteredGroup));
    }
  }

  IndexSet rows = points.takeFirstDimensions(equationRanges.rank());
  IndexSet filteredRows = rows.intersect(filter);

  if (!filteredRows.empty()) {
    IndexSet enrichedRows = filteredRows.append(variableRanges);
    IndexSet filteredPoints = points.intersect(enrichedRows);
    result.points += filteredPoints;
  }

  return result;
}

MCIM MCIM::filterColumns(const IndexSet &filter) const {
  MCIM result(equationRanges, variableRanges);

  for (const auto &group : groups) {
    std::unique_ptr<MCIMGroup> filteredGroup = group->filterValues(filter);

    if (!filteredGroup->empty()) {
      result.addGroup(std::move(filteredGroup));
    }
  }

  IndexSet columns = points.takeLastDimensions(variableRanges.rank());
  IndexSet filteredColumns = columns.intersect(filter);

  if (!filteredColumns.empty()) {
    IndexSet enrichedColumns = filteredColumns.prepend(equationRanges);
    IndexSet filteredPoints = points.intersect(enrichedColumns);
    result.points += filteredPoints;
  }

  return result;
}

std::vector<MCIM> MCIM::splitGroups() const {
  std::vector<MCIM> result;

  for (const auto &group : groups) {
    assert(equationRanges.contains(group->getKeys()));
    assert(variableRanges.contains(group->getValues()));

    auto localSolutions =
        solveLocalMatchingProblem(group->getKeys(), group->getValues(),
                                  group->getAccessFunction().clone());

    for (auto &localSolution : localSolutions) {
      assert(localSolution.equationRanges == group->getKeys());
      assert(localSolution.variableRanges == group->getValues());

      localSolution.equationRanges = equationRanges;
      localSolution.variableRanges = variableRanges;

      result.push_back(std::move(localSolution));
    }
  }

  for (Point point : points) {
    MCIM group(equationRanges, variableRanges);
    Point equation = point.takeFront(equationRanges.rank());
    Point variable = point.takeBack(variableRanges.rank());
    group.set(equation, variable);
  }

  return result;
}

MCIMGroup &MCIM::addGroup(std::unique_ptr<MCIMGroup> group) {
  assert(equationRanges.contains(group->getKeys()));
  assert(variableRanges.contains(group->getValues()));
  return *groups.emplace_back(std::move(group));
}
} // namespace marco::modeling::internal

namespace {
template <class T>
static size_t numDigits(T value) {
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
} // namespace

static size_t getRangeMaxColumns(const Range &range) {
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

static size_t getIndicesWidth(const Point &indexes) {
  size_t result = 0;

  for (const auto &index : indexes) {
    result += numDigits(index);

    if (index < 0) {
      ++result;
    }
  }

  return result;
}

static size_t getWrappedIndexesLength(size_t indexesLength,
                                      size_t numberOfIndexes) {
  size_t result = indexesLength;

  result += 1;                   // '(' character
  result += numberOfIndexes - 1; // ',' characters
  result += 1;                   // ')' character

  return result;
}

namespace marco::modeling::internal {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MCIM &obj) {
  const auto &equationRanges = obj.getEquationRanges();
  const auto &variableRanges = obj.getVariableRanges();

  // Determine the max widths of the indexes of the equation, so that they
  // will be properly aligned.
  llvm::SmallVector<size_t, 3> equationIndexesCols;

  for (const MultidimensionalRange &range : llvm::make_range(
           equationRanges.rangesBegin(), equationRanges.rangesEnd())) {
    for (size_t i = 0, e = range.rank(); i < e; ++i) {
      equationIndexesCols.push_back(getRangeMaxColumns(range[i]));
    }
  }

  size_t equationIndexesMaxWidth =
      std::accumulate(equationIndexesCols.begin(), equationIndexesCols.end(),
                      static_cast<size_t>(0));

  size_t equationIndexesColumnWidth =
      getWrappedIndexesLength(equationIndexesMaxWidth, equationRanges.rank());

  // Determine the max column width, so that the horizontal spacing is the
  // same among all the items.
  llvm::SmallVector<size_t, 3> variableIndexesCols;

  for (const MultidimensionalRange &range : llvm::make_range(
           variableRanges.rangesBegin(), variableRanges.rangesEnd())) {
    for (size_t i = 0, e = range.rank(); i < e; ++i) {
      variableIndexesCols.push_back(getRangeMaxColumns(range[i]));
    }
  }

  size_t variableIndexesMaxWidth =
      std::accumulate(variableIndexesCols.begin(), variableIndexesCols.end(),
                      static_cast<size_t>(0));

  size_t variableIndexesColumnWidth =
      getWrappedIndexesLength(variableIndexesMaxWidth, variableRanges.rank());

  // Print the spacing of the first line
  for (size_t i = 0, e = equationIndexesColumnWidth; i < e; ++i) {
    os << " ";
  }

  // Print the variable indexes
  for (const auto &variableIndexes : variableRanges) {
    os << " ";
    size_t columnWidth = getIndicesWidth(variableIndexes);

    for (size_t i = columnWidth; i < variableIndexesMaxWidth; ++i) {
      os << " ";
    }

    os << variableIndexes;
  }

  // The first line containing the variable indexes is finished
  os << "\n";

  // Print a line for each equation
  for (const auto &equation : equationRanges) {
    for (size_t i = getIndicesWidth(equation); i < equationIndexesMaxWidth;
         ++i) {
      os << " ";
    }

    os << equation;

    for (const auto &variable : variableRanges) {
      os << " ";

      size_t columnWidth = variableIndexesColumnWidth;
      size_t spacesAfter = (columnWidth - 1) / 2;
      size_t spacesBefore = columnWidth - 1 - spacesAfter;

      for (size_t i = 0; i < spacesBefore; ++i) {
        os << " ";
      }

      os << (obj.get(equation, variable) ? 1 : 0);

      for (size_t i = 0; i < spacesAfter; ++i) {
        os << " ";
      }
    }

    os << "\n";
  }

  return os;
}
} // namespace marco::modeling::internal
