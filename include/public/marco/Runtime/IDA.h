#ifndef MARCO_RUNTIME_IDA_H
#define MARCO_RUNTIME_IDA_H

#include "marco/Runtime/Mangling.h"

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaCreate, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(idaInit, void, PTR(void))

RUNTIME_FUNC_DECL(idaStep, void, PTR(void))

RUNTIME_FUNC_DECL(printStatistics, void, PTR(void))

RUNTIME_FUNC_DECL(idaFree, void, PTR(void))

RUNTIME_FUNC_DECL(idaSetStartTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetEndTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetTimeStep, void, PTR(void), double)

RUNTIME_FUNC_DECL(idaSetRelativeTolerance, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetAbsoluteTolerance, void, PTR(void), double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaAddEquation, int64_t, PTR(void), PTR(int64_t), int64_t)

RUNTIME_FUNC_DECL(idaAddResidual, void, PTR(void), int64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddJacobian, void, PTR(void), int64_t, int64_t, PTR(void))

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaAddAlgebraicVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void))
RUNTIME_FUNC_DECL(idaAddStateVariable, int64_t, PTR(void), PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddVariableAccess, void, PTR(void), int64_t, int64_t, PTR(int64_t), int64_t)

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaGetVariable, PTR(void), PTR(void), int64_t)
RUNTIME_FUNC_DECL(idaGetDerivative, PTR(void), PTR(void), int64_t)

RUNTIME_FUNC_DECL(idaGetCurrentTime, float, PTR(void))
RUNTIME_FUNC_DECL(idaGetCurrentTime, double, PTR(void))

#endif // MARCO_RUNTIME_IDA_H
