#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <marco/modeling/Scheduling.h>

using namespace ::marco::modeling;
using ::marco::modeling::internal::MCIS;
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

    static long getRangeBegin(Equation* const* equation, size_t inductionVarIndex)
    {
      return (*equation)->rangeBegin(inductionVarIndex);
    }

    static long getRangeEnd(Equation* const* equation, size_t inductionVarIndex)
    {
      return (*equation)->rangeEnd(inductionVarIndex);
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
 *   x[i - 1] = f0(y[i + 9])
 *
 * for i in 10:13
 *   y[i + 3] = f1(x[i - 7], z[i + 5])
 *
 * for i in 14:24
 *   z[i] = 9
 */
TEST(SCC, testSchedule) {
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
  EXPECT_CALL(eq2, rangeEnd(0)).WillRepeatedly(Return(14));

  Equation::Access eq2w(&y, AccessFunction(DimensionAccess::relative(0, 3)), "eq2w");
  EXPECT_CALL(eq2, write()).WillRepeatedly(Return(eq2w));

  std::vector<Equation::Access> eq2r = {
      Equation::Access(&x, AccessFunction(DimensionAccess::relative(0, -7)), "eq2r1"),
      Equation::Access(&z, AccessFunction(DimensionAccess::relative(0, 5)), "eq2r2")
  };

  EXPECT_CALL(eq2, reads()).WillRepeatedly(Return(eq2r));

  Equation eq3;
  EXPECT_CALL(eq3, name()).WillRepeatedly(Return("eq3"));
  EXPECT_CALL(eq3, rank()).WillRepeatedly(Return(1));
  EXPECT_CALL(eq3, rangeBegin(0)).WillRepeatedly(Return(15));
  EXPECT_CALL(eq3, rangeEnd(0)).WillRepeatedly(Return(25));

  Equation::Access eq3w(&z, AccessFunction(DimensionAccess::relative(0, 0)), "eq3w");
  EXPECT_CALL(eq3, write()).WillRepeatedly(Return(eq3w));

  std::vector<Equation::Access> eq3r;
  EXPECT_CALL(eq3, reads()).WillRepeatedly(Return(eq3r));

  Scheduler<Variable*, Equation*> scheduler;
  //auto schedule = scheduler.schedule({ &eq1, &eq2, &eq3 });
}