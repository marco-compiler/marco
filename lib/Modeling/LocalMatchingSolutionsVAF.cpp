#include "marco/Modeling/LocalMatchingSolutionsVAF.h"

using namespace ::marco::modeling;
using namespace ::marco::modeling::internal;

namespace
{
  class GeneratorDefault : public VAFSolutions::Generator
  {
    public:
      GeneratorDefault(
        const IndexSet& matrixEquationIndices,
        const IndexSet& matrixVariableIndices,
        const AccessFunction& accessFunction);

      bool hasValue() const override;

      MCIM getValue() const override;

      void fetchNext() override;

    private:
      void initialize();

    private:
      // The domain upon which the matrix has to be constructed.
      const IndexSet* matrixEquationIndices;
      const IndexSet* matrixVariableIndices;

      // The access function.
      const AccessFunction* accessFunction;

      // The next value to be returned.
      std::unique_ptr<MCIM> value;

      // Data used for progressive computation.
      std::unique_ptr<IndexSet::const_point_iterator> pointsIt;
      std::unique_ptr<IndexSet::const_point_iterator> pointsEndIt;
  };
}

GeneratorDefault::GeneratorDefault(
    const IndexSet& matrixEquationIndices,
    const IndexSet& matrixVariableIndices,
    const AccessFunction& accessFunction)
    : matrixEquationIndices(&matrixEquationIndices),
      matrixVariableIndices(&matrixVariableIndices),
      accessFunction(&accessFunction),
      pointsIt(nullptr),
      pointsEndIt(nullptr)
{
  initialize();
}

void GeneratorDefault::initialize()
{
  pointsIt = std::make_unique<IndexSet::const_point_iterator>(
      matrixEquationIndices->begin());

  pointsEndIt = std::make_unique<IndexSet::const_point_iterator>(
      matrixEquationIndices->end());

  fetchNext();
}

bool GeneratorDefault::hasValue() const
{
  return value != nullptr;
}

MCIM GeneratorDefault::getValue() const
{
  assert(hasValue());
  return *value;
}

void GeneratorDefault::fetchNext()
{
  if (*pointsIt == *pointsEndIt) {
    value = nullptr;
  } else {
    Point equationPoint = **pointsIt;
    Point variablePoint = accessFunction->map(equationPoint);

    MCIM matrix(*matrixEquationIndices, *matrixVariableIndices);
    matrix.set(equationPoint, variablePoint);
    value = std::make_unique<MCIM>(std::move(matrix));

    ++(*pointsIt);
  }
}

namespace
{
  class GeneratorRotoTranslation : public VAFSolutions::Generator
  {
    public:
      GeneratorRotoTranslation(
          const IndexSet& matrixEquationIndices,
          const IndexSet& matrixVariableIndices,
          const AccessFunctionRotoTranslation& accessFunction);

      bool hasValue() const override;

      MCIM getValue() const override;

      void fetchNext() override;

    private:
      void initialize();

    private:
      // The domain upon which the matrix has to be constructed.
      const IndexSet* matrixEquationIndices;
      const IndexSet* matrixVariableIndices;

      // The access function.
      const AccessFunctionRotoTranslation* accessFunction;

      // The iterator for the current multidimensional range of the equation
      // indices.
      IndexSet::const_range_iterator currentEquationsRangeIt;

      // The next value to be returned.
      std::unique_ptr<MCIM> value;

      // Data used for progressive computation.
      llvm::SmallVector<Range, 3> ranges;
      std::unique_ptr<MultidimensionalRange> unusedRange;
      llvm::SmallVector<size_t, 3> unusedRangeOriginalPosition;
      std::unique_ptr<MultidimensionalRange::const_iterator> unusedRangeIt;
      std::unique_ptr<MultidimensionalRange::const_iterator> unusedRangeEnd;
  };
}

GeneratorRotoTranslation::GeneratorRotoTranslation(
    const IndexSet& matrixEquationIndices,
    const IndexSet& matrixVariableIndices,
    const AccessFunctionRotoTranslation& accessFunction)
    : matrixEquationIndices(&matrixEquationIndices),
      matrixVariableIndices(&matrixVariableIndices),
      accessFunction(&accessFunction),
      currentEquationsRangeIt(matrixEquationIndices.rangesBegin()),
      value(nullptr)
{
  initialize();
}

void GeneratorRotoTranslation::initialize()
{
  fetchNext();
}

bool GeneratorRotoTranslation::hasValue() const
{
  return value != nullptr;
}

MCIM GeneratorRotoTranslation::getValue() const
{
  assert(hasValue());
  return *value;
}

void GeneratorRotoTranslation::fetchNext()
{
  if (currentEquationsRangeIt == matrixEquationIndices->rangesEnd()) {
    value = nullptr;
    return;
  }

  if (unusedRangeIt == nullptr || *unusedRangeIt == *unusedRangeEnd) {
    llvm::SmallVector<size_t, 3> inductionsUsage;
    accessFunction->countVariablesUsages(inductionsUsage);

    ranges.clear();
    ranges.resize(matrixEquationIndices->rank(), Range(0, 1));

    llvm::SmallVector<Range, 3> unusedRanges;

    unusedRangeOriginalPosition.clear();
    unusedRangeOriginalPosition.resize(matrixEquationIndices->rank(), 0);

    // We need to separate the unused induction variables. Those are in fact
    // the ones leading to repetitions among variable usages, and thus lead
    // to a new group for each time they change value.

    for (const auto& usage: llvm::enumerate(inductionsUsage)) {
      if (usage.value() == 0) {
        unusedRangeOriginalPosition[unusedRanges.size()] = usage.index();
        unusedRanges.push_back((*currentEquationsRangeIt)[usage.index()]);
      } else {
        ranges[usage.index()] = (*currentEquationsRangeIt)[usage.index()];
      }
    }

    if (unusedRanges.empty()) {
      // There are no unused variables, so we can just add the whole equation
      // and variable ranges to the matrix.

      MCIM matrix(*matrixEquationIndices, *matrixVariableIndices);
      MultidimensionalRange equations(ranges);
      matrix.apply(*currentEquationsRangeIt, *accessFunction);
      value = std::make_unique<MCIM>(std::move(matrix));

      unusedRange = nullptr;
      ++currentEquationsRangeIt;

    } else {
      // Theoretically, it would be sufficient to store just the iterators of
      // the reordered multidimensional range. However, their implementation
      // may rely on the range existence, and having it allocated on the
      // stack may lead to dangling pointers. Thus, we also store the range
      // inside the class.

      unusedRange =
          std::make_unique<MultidimensionalRange>(std::move(unusedRanges));

      unusedRangeIt =
          std::make_unique<MultidimensionalRange::const_iterator>(
              unusedRange->begin());

      unusedRangeEnd =
          std::make_unique<MultidimensionalRange::const_iterator>(
              unusedRange->end());
    }
  }

  if (unusedRange != nullptr) {
    assert(unusedRangeIt != nullptr);
    assert(unusedRangeEnd != nullptr);

    MCIM matrix(*matrixEquationIndices, *matrixVariableIndices);
    auto unusedIndices = **unusedRangeIt;

    for (size_t i = 0; i < unusedIndices.rank(); ++i) {
      ranges[unusedRangeOriginalPosition[i]] =
          Range(unusedIndices[i], unusedIndices[i] + 1);
    }

    MultidimensionalRange equations(ranges);
    matrix.apply(*currentEquationsRangeIt, *accessFunction);
    value = std::make_unique<MCIM>(std::move(matrix));

    ++(*unusedRangeIt);

    if (*unusedRangeIt == *unusedRangeEnd) {
      ++currentEquationsRangeIt;
    }
  }
}

namespace marco::modeling::internal
{
  VAFSolutions::VAFSolutions(
      llvm::ArrayRef<AccessFunction> accessFunctions,
      IndexSet equationIndices,
      IndexSet variableIndices)
      : accessFunctions(accessFunctions.begin(), accessFunctions.end()),
        equationIndices(std::move(equationIndices)),
        variableIndices(std::move(variableIndices))
  {
    initialize();
  }

  MCIM& VAFSolutions::operator[](size_t index)
  {
    assert(index < size());

    while (matrices.size() <= index &&
           currentAccessFunction < accessFunctions.size()) {
      fetchNext();
    }

    return matrices[index];
  }

  size_t VAFSolutions::size() const
  {
    return solutionsAmount;
  }

  void VAFSolutions::initialize()
  {
    llvm::sort(
        accessFunctions,
        [&](const AccessFunction& lhs, const AccessFunction& rhs) -> bool {
          return compareAccessFunctions(lhs, rhs);
        });

    // Determine the amount of solutions. Precomputing it allows the actual
    // solutions to be determined only when requested. This is useful when
    // the solutions set will be processed only if a certain amount of
    // solutions is found, as it may allow for skipping their computation.

    solutionsAmount = 0;

    for (const AccessFunction& accessFunction : accessFunctions) {
      solutionsAmount += getSolutionsAmount(accessFunction);
    }
  }

  void VAFSolutions::fetchNext()
  {
    if (generator == nullptr) {
      if (currentAccessFunction < accessFunctions.size()) {
        generator = getGenerator(accessFunctions[currentAccessFunction++]);
        ++currentAccessFunction;
      }
    }

    while (generator &&
           !generator->hasValue() &&
           currentAccessFunction < accessFunctions.size()) {
      // Advance to the first generator with a valid value.
      generator = getGenerator(accessFunctions[currentAccessFunction++]);
    }

    if (generator && generator->hasValue()) {
      matrices.push_back(generator->getValue());
      generator->fetchNext();
    } else {
      generator = nullptr;
    }
  }

  bool VAFSolutions::compareAccessFunctions(
      const AccessFunction& lhs, const AccessFunction& rhs) const
  {
    if (auto casted = lhs.dyn_cast<AccessFunctionRotoTranslation>()) {
      return compareAccessFunctions(*casted, rhs);
    }

    return false;
  }

  bool VAFSolutions::compareAccessFunctions(
      const AccessFunctionRotoTranslation& lhs,
      const AccessFunction& rhs) const
  {
    if (auto casted = rhs.dyn_cast<AccessFunctionRotoTranslation>()) {
      return compareAccessFunctions(lhs, *casted);
    }

    return true;
  }

  bool VAFSolutions::compareAccessFunctions(
      const AccessFunctionRotoTranslation& lhs,
      const AccessFunctionRotoTranslation& rhs) const
  {
    // Sort the access functions so that we prefer the ones referring to
    // induction variables. More in details, we prefer the ones covering the
    // most of them, as they are the ones that lead to the least amount of
    // groups.

    llvm::SmallVector<size_t, 3> lhsUsages;
    llvm::SmallVector<size_t, 3> rhsUsages;

    lhs.countVariablesUsages(lhsUsages);
    rhs.countVariablesUsages(rhsUsages);

    auto counter = [](const size_t& usage) {
      return usage == 1;
    };

    size_t lhsUniqueUsages = llvm::count_if(lhsUsages, counter);
    size_t rhsUniqueUsages = llvm::count_if(rhsUsages, counter);

    return lhsUniqueUsages >= rhsUniqueUsages;
  }

  size_t VAFSolutions::getSolutionsAmount(
      const AccessFunction& accessFunction) const
  {
    if (auto casted =
            accessFunction.dyn_cast<AccessFunctionRotoTranslation>()) {
      return getSolutionsAmount(*casted);
    }

    return equationIndices.flatSize();
  }

  size_t VAFSolutions::getSolutionsAmount(
      const AccessFunctionRotoTranslation& accessFunction) const
  {
    size_t result = 0;

    llvm::SmallVector<size_t, 3> inductionsUsage;
    accessFunction.countVariablesUsages(inductionsUsage);

    for (const MultidimensionalRange& range : llvm::make_range(
             equationIndices.rangesBegin(), equationIndices.rangesEnd())) {
      size_t count = 1;

      for (const auto& usage : llvm::enumerate(inductionsUsage)) {
        if (usage.value() == 0) {
          count *= range[usage.index()].size();
        }
      }

      result += count;
    }

    return result;
  }

  std::unique_ptr<VAFSolutions::Generator> VAFSolutions::getGenerator(
      const AccessFunction& accessFunction) const
  {
    if (auto casted =
            accessFunction.dyn_cast<AccessFunctionRotoTranslation>()) {
      return std::make_unique<GeneratorRotoTranslation>(
          equationIndices, variableIndices, *casted);
    }

    return std::make_unique<GeneratorDefault>(
        equationIndices, variableIndices, accessFunction);
  }
}
