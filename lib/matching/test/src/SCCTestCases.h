#ifndef MARCO_SCCTESTCASES_H
#define MARCO_SCCTESTCASES_H

#include <marco/matching/VVarDependencyGraph.h>

#include "SCCCommon.h"

namespace marco::scc
{
  VVarDependencyGraph<Variable, Equation> testCase1();
}

#endif //MARCO_SCCTESTCASES_H
