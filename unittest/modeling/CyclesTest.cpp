#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "marco/modeling/Cycles.h"

using namespace ::marco::modeling;
using ::marco::modeling::IndexSet;
using ::testing::Return;

class Variable
{
  public:
    MOCK_CONST_METHOD0(name, std::string());
};

namespace marco::modeling::dependency
{
  template<>
  struct VariableTraits<Variable*>
  {
    using Id = std::string;

    static Id getId(Variable* const* variable)
    {
      return (*variable)->name();
    }
  };
}

class Equation
{
  public:
    using AccessProperty = std::string;
    using Access = dependency::Access<Variable*, AccessProperty>;

    MOCK_CONST_METHOD0(name, std::string());
    MOCK_CONST_METHOD0(rank, size_t());
    MOCK_CONST_METHOD1(rangeBegin, size_t(size_t index));
    MOCK_CONST_METHOD1(rangeEnd, size_t(size_t index));
    MOCK_CONST_METHOD0(write, Access());
    MOCK_CONST_METHOD0(reads, std::vector<Access>());
};

namespace marco::modeling::dependency
{
  template<>
  struct EquationTraits<Equation*>
  {
    using Id = std::string;

    static Id getId(Equation* const* equation)
    {
      return (*equation)->name();
    }

    static size_t getNumOfIterationVars(Equation* const* equation)
    {
      return (*equation)->rank();
    }

    static MultidimensionalRange getIterationRanges(Equation* const* equation)
    {
      std::vector<Range> ranges;

      for (size_t i = 0, e = getNumOfIterationVars(equation); i < e; ++i) {
        ranges.emplace_back((*equation)->rangeBegin(i), (*equation)->rangeEnd(i));
      }

      return MultidimensionalRange(std::move(ranges));
    }

    using VariableType = Variable*;
    using AccessProperty = Equation::AccessProperty;

    static Access<VariableType, AccessProperty> getWrite(Equation* const* equation)
    {
      return (*equation)->write();
    }

    static std::vector<Access<VariableType, AccessProperty>> getReads(Equation* const* equation)
    {
      return (*equation)->reads();
    }
  };
}

MATCHER_P(CycleStartingWithEquation, name, "") {
  return arg.getEquation()->name() == name;
}

template<typename Cycles>
const auto& getEquationCycles(const Cycles& cycles, std::string equation) {
  return *std::find_if(cycles.begin(), cycles.end(), [&](const auto& cycle) {
    return cycle.getEquation()->name() == equation;
  });
}

class Path
{
  public:
    struct Step
    {
        size_t totalCoveredIndexes;
        IndexSet indexes;
        std::string access;
        std::string nextEquation;
    };

  private:
    using Container = std::vector<Step>;

  public:
    using const_iterator = Container::const_iterator;

    Path(llvm::ArrayRef<Step> steps) : steps(steps.begin(), steps.end())
    {
    }

    const_iterator begin() const
    {
      return steps.begin();
    }

    const_iterator end() const
    {
      return steps.end();
    }

  private:
    Container steps;
};

MATCHER_P(HasPath, path, "") {
  const auto* equation = &arg;

  for (const Path::Step& step : path) {
    size_t totalCoveredIndexes = 0;

    for (const auto& interval : *equation) {
      totalCoveredIndexes += interval.getRange().flatSize();
    }

    if (totalCoveredIndexes != step.totalCoveredIndexes) {
      return false;
    }

    // Search the indexes
    auto interval = llvm::find_if(*equation, [&](const auto& interval) {
      return step.indexes == interval.getRange();
    });

    if (interval == equation->end()) {
      return false;
    }

    // Search the access
    auto dependencies = interval->getDestinations();

    auto next = llvm::find_if(dependencies, [&](const auto& dependency) {
      return dependency.getAccess().getProperty() == step.access;
    });

    if (next == dependencies.end()) {
      return false;
    }

    if (next->getNode().getEquation()->name() != step.nextEquation) {
      return false;
    }

    // Go on with the next equation
    equation = &next->getNode();
  }

  return true;
}

/**
 * for i in 3:8
 *   x[i + 1] = f0(x[i - 1], x[i + 3])
 */
 /*
TEST(Cycles, selfDependency) {
  Variable x;
  EXPECT_CALL(x, name()).WillRepeatedly(Return("x"));

  Equation eq1;
  EXPECT_CALL(eq1, name()).WillRepeatedly(Return("eq1"));
  EXPECT_CALL(eq1, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq1, rangeBegin(0)).WillRepeatedly(Return(3));
  EXPECT_CALL(eq1, rangeEnd(0)).WillRepeatedly(Return(9));

  Equation::Access eq1w(&x, AccessFunction(DimensionAccess::relative(0, 1)), "eq1w");
  EXPECT_CALL(eq1, write()).WillRepeatedly(Return(eq1w));

  std::vector<Equation::Access> eq1r = {
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, -1)), "eq1r1"),
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, 3)), "eq1r2")
  };

  EXPECT_CALL(eq1, reads()).WillRepeatedly(Return(eq1r));

  CyclesFinder<Variable*, Equation*> graph(&eq1);
  auto cycles = graph.getEquationsCycles();

  // TODO
  EXPECT_THAT(cycles, testing::SizeIs(1));

  EXPECT_THAT(cycles, testing::Contains(CycleStartingWithEquation("eq1")));

  EXPECT_THAT(getEquationCycles(cycles, "eq1"), HasPath(Path({
      { 2, MCIS({5, 6, 7, 8, 9}), "eq1r1", "eq1"}
  })));

  EXPECT_THAT(getEquationCycles(cycles, "eq1"), HasPath(Path({
      { 2, MCIS({10, 11, 12, 13}), "eq1r2", "eq1"}
  })));
}
  */

/**
 * for i in 3:8
 *   x[i - 1] = f0(y[i + 9])
 *
 * for i in 10:13
 *   y[i + 3] = f1(x[i - 7])
 */
TEST(Cycles, oneStepCycle) {
  Variable x;
  EXPECT_CALL(x, name()).WillRepeatedly(Return("x"));

  Variable y;
  EXPECT_CALL(y, name()).WillRepeatedly(Return("y"));

  Equation eq1;
  EXPECT_CALL(eq1, name()).WillRepeatedly(Return("eq1"));
  EXPECT_CALL(eq1, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq1, rangeBegin(0)).WillRepeatedly(Return(3));
  EXPECT_CALL(eq1, rangeEnd(0)).WillRepeatedly(Return(9));

  Equation::Access eq1w(&x, AccessFunction(DimensionAccess::relative(0, -1)), "eq1w");
  EXPECT_CALL(eq1, write()).WillRepeatedly(Return(eq1w));

  std::vector<Equation::Access> eq1r = {
      Equation::Access(&y, AccessFunction(DimensionAccess::relative(0, 9)), "eq1r1")
  };

  EXPECT_CALL(eq1, reads()).WillRepeatedly(Return(eq1r));

  Equation eq2;
  EXPECT_CALL(eq2, name()).WillRepeatedly(Return("eq2"));
  EXPECT_CALL(eq2, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq2, rangeBegin(0)).WillRepeatedly(Return(10));
  EXPECT_CALL(eq2, rangeEnd(0)).WillRepeatedly(Return(14));

  Equation::Access eq2w(&y, AccessFunction(DimensionAccess::relative(0, 3)), "eq2w");
  EXPECT_CALL(eq2, write()).WillRepeatedly(Return(eq2w));

  std::vector<Equation::Access> eq2r = {
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, -7)), "eq2r1")
  };

  EXPECT_CALL(eq2, reads()).WillRepeatedly(Return(eq2r));

  CyclesFinder<Variable*, Equation*> graph({ &eq1, &eq2 });
  auto cycles = graph.getEquationsCycles();

  EXPECT_THAT(cycles, testing::SizeIs(2));

  EXPECT_THAT(cycles, testing::Contains(CycleStartingWithEquation("eq1")));
  EXPECT_THAT(cycles, testing::Contains(CycleStartingWithEquation("eq2")));

  EXPECT_THAT(getEquationCycles(cycles, "eq1"), HasPath(Path({
    { 4, IndexSet({4, 5, 6, 7}), "eq1r1", "eq2"}
  })));

  EXPECT_THAT(getEquationCycles(cycles, "eq2"), HasPath(Path({
      { 4, IndexSet({10, 11, 12, 13}), "eq2r1", "eq1"}
  })));
}

/**
 * for i in 3:8
 *   x[i - 1] = f0(y[i + 9])
 *
 * for i in 10:14
 *   y[i + 3] = f1(z[i - 4])
 *
 * for i in 3:6
 *   z[i + 1] = f2(x[i - 2])
 */
TEST(SCC, twoStepsCycle) {
  Variable x;
  EXPECT_CALL(x, name()).WillRepeatedly(Return("x"));

  Variable y;
  EXPECT_CALL(y, name()).WillRepeatedly(Return("y"));

  Variable z;
  EXPECT_CALL(z, name()).WillRepeatedly(Return("z"));

  Equation eq1;
  EXPECT_CALL(eq1, name()).WillRepeatedly(Return("eq1"));
  EXPECT_CALL(eq1, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq1, rangeBegin(0)).WillRepeatedly(Return(3));
  EXPECT_CALL(eq1, rangeEnd(0)).WillRepeatedly(Return(9));

  Equation::Access eq1w(&x, AccessFunction(DimensionAccess::relative(0, -1)), "eq1w");
  EXPECT_CALL(eq1, write()).WillRepeatedly(Return(eq1w));

  std::vector<Equation::Access> eq1r = {
      Equation::Access(&y, AccessFunction(DimensionAccess::relative(0, 9)), "eq1r1")
  };

  EXPECT_CALL(eq1, reads()).WillRepeatedly(Return(eq1r));

  Equation eq2;
  EXPECT_CALL(eq2, name()).WillRepeatedly(Return("eq2"));
  EXPECT_CALL(eq2, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq2, rangeBegin(0)).WillRepeatedly(Return(10));
  EXPECT_CALL(eq2, rangeEnd(0)).WillRepeatedly(Return(15));

  Equation::Access eq2w(&y, AccessFunction(DimensionAccess::relative(0, 3)), "eq2w");
  EXPECT_CALL(eq2, write()).WillRepeatedly(Return(eq2w));

  std::vector<Equation::Access> eq2r = {
      Equation::Access(&z, AccessFunction(DimensionAccess::relative(0, -4)), "eq2r1")
  };

  EXPECT_CALL(eq2, reads()).WillRepeatedly(Return(eq2r));

  Equation eq3;
  EXPECT_CALL(eq3, name()).WillRepeatedly(Return("eq3"));
  EXPECT_CALL(eq3, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq3, rangeBegin(0)).WillRepeatedly(Return(3));
  EXPECT_CALL(eq3, rangeEnd(0)).WillRepeatedly(Return(7));

  Equation::Access eq3w(&z, AccessFunction(DimensionAccess::relative(0, 1)), "eq3w");
  EXPECT_CALL(eq3, write()).WillRepeatedly(Return(eq3w));

  std::vector<Equation::Access> eq3r = {
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, -2)), "eq3r1")
  };

  EXPECT_CALL(eq3, reads()).WillRepeatedly(Return(eq3r));

  CyclesFinder<Variable*, Equation*> graph({ &eq1, &eq2, &eq3 });
  auto cycles = graph.getEquationsCycles();

  EXPECT_THAT(cycles, testing::SizeIs(3));

  EXPECT_THAT(cycles, testing::Contains(CycleStartingWithEquation("eq1")));
  EXPECT_THAT(cycles, testing::Contains(CycleStartingWithEquation("eq2")));
  EXPECT_THAT(cycles, testing::Contains(CycleStartingWithEquation("eq3")));

  EXPECT_THAT(getEquationCycles(cycles, "eq1"), HasPath(Path({
      { 2, IndexSet({4, 5}), "eq1r1", "eq2"},
      { 2, IndexSet({10, 11}), "eq2r1", "eq3"}
  })));

  EXPECT_THAT(getEquationCycles(cycles, "eq2"), HasPath(Path({
      { 2, IndexSet({10, 11}), "eq2r1", "eq3"},
      { 2, IndexSet({5, 6}), "eq3r1", "eq1"}
  })));

  EXPECT_THAT(getEquationCycles(cycles, "eq3"), HasPath(Path({
      { 2, IndexSet({5, 6}), "eq3r1", "eq1"},
      { 2, IndexSet({4, 5}), "eq1r1", "eq2"}
  })));
}

/**
 * for i in 3:8
 *   x[i - 1] = f0(y[i + 9], z[i + 1])
 *
 * for i in 10:14
 *   y[i + 3] = f1(x[i - 7])
 *
 * for i in 11:12
 *   z[i - 4] = f2(x[i - 6])
 */
TEST(SCC, oneStepCycleWithMultipleReads) {
  Variable x;
  EXPECT_CALL(x, name()).WillRepeatedly(Return("x"));

  Variable y;
  EXPECT_CALL(y, name()).WillRepeatedly(Return("y"));

  Variable z;
  EXPECT_CALL(z, name()).WillRepeatedly(Return("z"));

  Equation eq1;
  EXPECT_CALL(eq1, name()).WillRepeatedly(Return("eq1"));
  EXPECT_CALL(eq1, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq1, rangeBegin(0)).WillRepeatedly(Return(3));
  EXPECT_CALL(eq1, rangeEnd(0)).WillRepeatedly(Return(9));

  Equation::Access eq1w(&x, AccessFunction(DimensionAccess::relative(0, -1)), "eq1w");
  EXPECT_CALL(eq1, write()).WillRepeatedly(Return(eq1w));

  std::vector<Equation::Access> eq1r = {
      Equation::Access(&y, AccessFunction(DimensionAccess::relative(0, 9)), "eq1r1"),
      Equation::Access(&z, AccessFunction(DimensionAccess::relative(0, 1)), "eq1r2")
  };

  EXPECT_CALL(eq1, reads()).WillRepeatedly(Return(eq1r));

  Equation eq2;
  EXPECT_CALL(eq2, name()).WillRepeatedly(Return("eq2"));
  EXPECT_CALL(eq2, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq2, rangeBegin(0)).WillRepeatedly(Return(10));
  EXPECT_CALL(eq2, rangeEnd(0)).WillRepeatedly(Return(15));

  Equation::Access eq2w(&y, AccessFunction(DimensionAccess::relative(0, 3)), "eq2w");
  EXPECT_CALL(eq2, write()).WillRepeatedly(Return(eq2w));

  std::vector<Equation::Access> eq2r = {
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, -7)), "eq2r1")
  };

  EXPECT_CALL(eq2, reads()).WillRepeatedly(Return(eq2r));

  Equation eq3;
  EXPECT_CALL(eq3, name()).WillRepeatedly(Return("eq3"));
  EXPECT_CALL(eq3, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq3, rangeBegin(0)).WillRepeatedly(Return(11));
  EXPECT_CALL(eq3, rangeEnd(0)).WillRepeatedly(Return(13));

  Equation::Access eq3w(&z, AccessFunction(DimensionAccess::relative(0, -4)), "eq3w");
  EXPECT_CALL(eq3, write()).WillRepeatedly(Return(eq3w));

  std::vector<Equation::Access> eq3r = {
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, -6)), "eq3r1")
  };

  EXPECT_CALL(eq3, reads()).WillRepeatedly(Return(eq3r));

  CyclesFinder<Variable*, Equation*> graph({ &eq1, &eq2, &eq3 });
  auto cycles = graph.getEquationsCycles();

  EXPECT_THAT(cycles, testing::SizeIs(3));

  EXPECT_THAT(cycles, testing::Contains(CycleStartingWithEquation("eq1")));
  EXPECT_THAT(cycles, testing::Contains(CycleStartingWithEquation("eq2")));
  EXPECT_THAT(cycles, testing::Contains(CycleStartingWithEquation("eq3")));

  EXPECT_THAT(getEquationCycles(cycles, "eq1"), HasPath(Path({
      { 5, IndexSet({4, 5}), "eq1r1", "eq2"}
  })));

  EXPECT_THAT(getEquationCycles(cycles, "eq1"), HasPath(Path({
      { 5, IndexSet({6, 7}), "eq1r1", "eq2"}
  })));

  EXPECT_THAT(getEquationCycles(cycles, "eq1"), HasPath(Path({
      { 5, IndexSet({6, 7}), "eq1r2", "eq3"}
  })));

  EXPECT_THAT(getEquationCycles(cycles, "eq1"), HasPath(Path({
      { 5, IndexSet({8}), "eq1r1", "eq2"}
  })));

  EXPECT_THAT(getEquationCycles(cycles, "eq2"), HasPath(Path({
      { 5, IndexSet({10, 11, 12, 13, 14}), "eq2r1", "eq1"}
  })));

  EXPECT_THAT(getEquationCycles(cycles, "eq3"), HasPath(Path({
      { 2, IndexSet({11, 12}), "eq3r1", "eq1"}
  })));
}
