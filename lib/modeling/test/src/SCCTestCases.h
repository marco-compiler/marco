#ifndef MARCO_SCCTESTCASES_H
#define MARCO_SCCTESTCASES_H

#include <marco/modeling/VVarDependencyGraph.h>

#include "SCCCommon.h"

namespace marco::modeling::scc::test
{
  VVarDependencyGraph<Variable, Equation> testCase1();
}

#endif //MARCO_SCCTESTCASES_H
