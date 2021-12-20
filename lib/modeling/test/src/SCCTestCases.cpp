#include <marco/modeling/AccessFunction.h>

#include "SCCTestCases.h"

using namespace ::marco::modeling::internal;

namespace marco::modeling::scc::test
{
  VVarDependencyGraph<Variable, Equation> testCase1()
  {
    using Access = Access<Variable>;

    Variable x("x", {5});
    Variable y("y", {5});

    Equation eq1(
        "eq1",
        Access(x, AccessFunction(DimensionAccess::relative(0, 0))),
        {
          Access(y, AccessFunction(DimensionAccess::relative(0, 1)))
        });

    eq1.addIterationRange(Range(2, 6));

    Equation eq2(
        "eq2",
        Access(y, AccessFunction(DimensionAccess::relative(0, 2))),
        {
          Access(x, AccessFunction(DimensionAccess::relative(0, 0)))
        });

    eq2.addIterationRange(Range(3, 5));

    return VVarDependencyGraph<Variable, Equation>({eq1, eq2});
  }

  VVarDependencyGraph<Variable, Equation> testCase2()
  {
    using Access = Access<Variable>;

    Variable x("x", {5});
    Variable y("y", {5});

    Equation eq1(
        "eq1",
        Access(x, AccessFunction(DimensionAccess::relative(0, 0))),
        {
            Access(y, AccessFunction(DimensionAccess::relative(0, -2))),
            Access(y, AccessFunction(DimensionAccess::relative(0, -1)))
        });

    eq1.addIterationRange(Range(3, 6));

    Equation eq2(
        "eq2",
        Access(y, AccessFunction(DimensionAccess::relative(0, 0))),
        {
            Access(x, AccessFunction(DimensionAccess::relative(0, 2)))
        });

    eq2.addIterationRange(Range(1, 4));

    Equation eq3(
        "eq3",
        Access(y, AccessFunction(DimensionAccess::relative(0, 0))),
        {
            Access(x, AccessFunction(DimensionAccess::relative(0, -1)))
        });

    eq3.addIterationRange(Range(4, 5));

    return VVarDependencyGraph<Variable, Equation>({eq1, eq2, eq3});
  }
}
