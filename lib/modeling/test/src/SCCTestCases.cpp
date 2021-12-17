#include "SCCTestCases.h"

namespace marco::modeling::scc::test
{
  VVarDependencyGraph<Variable, Equation> testCase1()
  {
    using Access = Access<Variable>;

    Variable x("x");
    Variable y("y");
    Variable z("z");
    Variable t("t");
    Variable k("k");

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
      Access(k)
    });

    Equation eq5("eq5", Access(k), {
      Access(z)
    });

    return VVarDependencyGraph<Variable, Equation>({ eq1, eq2, eq3, eq4, eq5 });
  }
}