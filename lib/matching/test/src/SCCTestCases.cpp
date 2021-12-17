#include "SCCTestCases.h"

namespace marco::scc
{
  VVarDependencyGraph<Variable, Equation> testCase1()
  {
    using Access = Access<Variable>;

    Variable x("x");
    Variable y("y");
    Variable z("z");
    Variable t("t");

    Equation eq1("eq1", Access(x), {
      Access(y)
    });

    Equation eq2("eq2", Access(y), {
      Access(x)
    });

    Equation eq3("eq3", Access(z), {
            Access(t)
    });

    Equation eq4("eq4", Access(t), {
            Access(z)
    });

    return VVarDependencyGraph<Variable, Equation>({ eq1, eq2, eq3, eq4 });
  }
}