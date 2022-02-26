#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "marco/Modeling/Scheduling.h"

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

/**
 * for i in 3:8
 *   x[i - 1] = f0(x[i - 2])
 */
TEST(Scheduling, forwardSchedulable) {
  Variable x;
  EXPECT_CALL(x, name()).WillRepeatedly(Return("x"));

  Equation eq1;
  EXPECT_CALL(eq1, name()).WillRepeatedly(Return("eq1"));
  EXPECT_CALL(eq1, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq1, rangeBegin(0)).WillRepeatedly(Return(3));
  EXPECT_CALL(eq1, rangeEnd(0)).WillRepeatedly(Return(9));

  Equation::Access eq1w(&x, AccessFunction(DimensionAccess::relative(0, -1)), "eq1w");
  EXPECT_CALL(eq1, write()).WillRepeatedly(Return(eq1w));

  std::vector<Equation::Access> eq1r = {
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, -2)), "eq1r1")
  };

  EXPECT_CALL(eq1, reads()).WillRepeatedly(Return(eq1r));

  Scheduler<Variable*, Equation*> scheduler;
  auto schedule = scheduler.schedule({ &eq1 });

  EXPECT_THAT(schedule, testing::SizeIs(1));
  EXPECT_EQ(schedule[0].getEquation()->name(), "eq1");

  auto scheduledIndexes = schedule[0].getIndexes();
  EXPECT_EQ(scheduledIndexes[0].getBegin(), 3);
  EXPECT_EQ(scheduledIndexes[0].getEnd(), 9);

  EXPECT_EQ(schedule[0].getIterationDirection(), marco::modeling::scheduling::Direction::Forward);
}

/**
 * for i in 3:8
 *   x[i + 1] = f0(x[i + 2])
 */
TEST(Scheduling, backwardSchedulable) {
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
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, 2)), "eq1r1")
  };

  EXPECT_CALL(eq1, reads()).WillRepeatedly(Return(eq1r));

  Scheduler<Variable*, Equation*> scheduler;
  auto schedule = scheduler.schedule({ &eq1 });

  EXPECT_THAT(schedule, testing::SizeIs(1));
  EXPECT_EQ(schedule[0].getEquation()->name(), "eq1");

  auto scheduledIndexes = schedule[0].getIndexes();
  EXPECT_EQ(scheduledIndexes[0].getBegin(), 3);
  EXPECT_EQ(scheduledIndexes[0].getEnd(), 9);

  EXPECT_EQ(schedule[0].getIterationDirection(), marco::modeling::scheduling::Direction::Backward);
}

/**
 * for i in 3:8
 *   x[i + 1] = f0(x[i + 2])
 */
TEST(Scheduling, test) {
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
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, 2)), "eq1r1")
  };

  EXPECT_CALL(eq1, reads()).WillRepeatedly(Return(eq1r));

  Scheduler<Variable*, Equation*> scheduler;
  auto schedule = scheduler.schedule({ &eq1 });

  EXPECT_THAT(schedule, testing::SizeIs(1));
  EXPECT_EQ(schedule[0].getEquation()->name(), "eq1");

  auto scheduledIndexes = schedule[0].getIndexes();
  EXPECT_EQ(scheduledIndexes[0].getBegin(), 3);
  EXPECT_EQ(scheduledIndexes[0].getEnd(), 9);

  EXPECT_EQ(schedule[0].getIterationDirection(), marco::modeling::scheduling::Direction::Backward);
}
