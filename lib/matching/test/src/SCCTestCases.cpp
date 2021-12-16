#include "SCCTestCases.h"

namespace marco::scc
{
  VVarDependencyGraph<Variable, Equation> testCase1()
  {
    using Access = Access<Variable>;

    Variable x("x");
    Variable y("y");

    Equation eq1("eq1", Access(x), {
      Access(y)
    });

    Equation eq2("eq2", Access(y), {
      Access(x)
    });

    return VVarDependencyGraph<Variable, Equation>({ eq1, eq2 });
  }
}