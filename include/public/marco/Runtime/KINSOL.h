#ifndef MARCO_RUNTIME_KINSOL_H
#define MARCO_RUNTIME_KINSOL_H

#include "marco/Runtime/Mangling.h"
#include <cstdint>

//===----------------------------------------------------------------------===//
// CLI
//===----------------------------------------------------------------------===//

#ifdef MARCO_CLI

#include "marco/Runtime/CLI.h"

namespace marco::runtime::kinsol
{
  std::unique_ptr<cli::Category> getCLIOptionsCategory();
}

#endif

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(kinsolCreate, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(kinsolInit, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolStep, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolPrintStatistics, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolFree, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolSetStartTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(kinsolSetEndTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(kinsolSetTimeStep, void, PTR(void), double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(kinsolAddEquation, int64_t, PTR(void), PTR(int64_t), int64_t)

RUNTIME_FUNC_DECL(kinsolAddResidual, void, PTR(void), int64_t, PTR(void))

RUNTIME_FUNC_DECL(kinsolAddJacobian, void, PTR(void), int64_t, int64_t, PTR(void))

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(kinsolAddAlgebraicVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))
RUNTIME_FUNC_DECL(kinsolAddStateVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

RUNTIME_FUNC_DECL(kinsolSetDerivative, void, PTR(void), int64_t, PTR(void), PTR(void), PTR(void))

RUNTIME_FUNC_DECL(kinsolAddVariableAccess, void, PTR(void), int64_t, int64_t, PTR(int64_t), int64_t)

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(kinsolGetCurrentTime, float, PTR(void))
RUNTIME_FUNC_DECL(kinsolGetCurrentTime, double, PTR(void))

#endif // MARCO_RUNTIME_KINSOL_H
