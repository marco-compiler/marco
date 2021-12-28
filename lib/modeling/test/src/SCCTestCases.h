#ifndef MARCO_MODELING_TEST_SCCTESTCASES_H
#define MARCO_MODELING_TEST_SCCTESTCASES_H

#include <marco/modeling/VVarDependencyGraph.h>

#include "SCCCommon.h"

namespace marco::modeling::scc::test
{
  VVarDependencyGraph<Variable, Equation> testCase1();
  VVarDependencyGraph<Variable, Equation> testCase2();
  VVarDependencyGraph<Variable, Equation> testCase3();
}

#endif // MARCO_MODELING_TEST_SCCTESTCASES_H
