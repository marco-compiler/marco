#ifndef MARCO_RUNTIME_IDA_H
#define MARCO_RUNTIME_IDA_H

#include "marco/Runtime/ArrayDescriptor.h"
#include "marco/Runtime/Mangling.h"

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaCreate, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(idaInit, bool, PTR(void))

RUNTIME_FUNC_DECL(idaStep, bool, PTR(void))
RUNTIME_FUNC_DECL(idaStep, bool, PTR(void), float)
RUNTIME_FUNC_DECL(idaStep, bool, PTR(void), double)

RUNTIME_FUNC_DECL(printStatistics, void, PTR(void))

RUNTIME_FUNC_DECL(idaFree, bool, PTR(void))

RUNTIME_FUNC_DECL(idaSetStartTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetEndTime, void, PTR(void), double)

RUNTIME_FUNC_DECL(idaSetRelativeTolerance, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetAbsoluteTolerance, void, PTR(void), double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaAddEquation, int64_t, PTR(void), ARRAY(int64_t))

RUNTIME_FUNC_DECL(addResidual, void, PTR(void), int32_t, RESIDUAL(float))
RUNTIME_FUNC_DECL(addResidual, void, PTR(void), int64_t, RESIDUAL(double))

RUNTIME_FUNC_DECL(addJacobian, void, PTR(void), int32_t, JACOBIAN(float))
RUNTIME_FUNC_DECL(addJacobian, void, PTR(void), int64_t, JACOBIAN(double))

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(addVariable, int64_t, PTR(void), ARRAY(double), bool)
RUNTIME_FUNC_DECL(addVariableAccess, void, PTR(void), int64_t, int64_t, ARRAY(int64_t))

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(getVariable, PTR(void), PTR(void), int64_t)
RUNTIME_FUNC_DECL(getDerivative, PTR(void), PTR(void), int64_t)

RUNTIME_FUNC_DECL(idaGetCurrentTime, float, PTR(void))
RUNTIME_FUNC_DECL(idaGetCurrentTime, double, PTR(void))

#endif // MARCO_RUNTIME_IDA_H
