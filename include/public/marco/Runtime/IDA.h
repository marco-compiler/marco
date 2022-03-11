#ifndef MARCO_RUNTIME_IDA_H
#define MARCO_RUNTIME_IDA_H

#include "marco/Runtime/ArrayDescriptor.h"
#include "marco/Runtime/Mangling.h"

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaCreate, PTR(void), int32_t)
RUNTIME_FUNC_DECL(idaCreate, PTR(void), int64_t)

RUNTIME_FUNC_DECL(idaInit, bool, PTR(void))

RUNTIME_FUNC_DECL(idaStep, bool, PTR(void))
RUNTIME_FUNC_DECL(idaStep, bool, PTR(void), float)
RUNTIME_FUNC_DECL(idaStep, bool, PTR(void), double)

RUNTIME_FUNC_DECL(printStatistics, void, PTR(void))

RUNTIME_FUNC_DECL(idaFree, bool, PTR(void))

RUNTIME_FUNC_DECL(setStartTime, void, PTR(void), float)
RUNTIME_FUNC_DECL(setStartTime, void, PTR(void), double)

RUNTIME_FUNC_DECL(setEndTime, void, PTR(void), float)
RUNTIME_FUNC_DECL(setEndTime, void, PTR(void), double)

RUNTIME_FUNC_DECL(setRelativeTolerance, void, PTR(void), float)
RUNTIME_FUNC_DECL(setRelativeTolerance, void, PTR(void), double)

RUNTIME_FUNC_DECL(setAbsoluteTolerance, void, PTR(void), float)
RUNTIME_FUNC_DECL(setAbsoluteTolerance, void, PTR(void), double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(addEquation, int32_t, PTR(void), ARRAY(int32_t))
RUNTIME_FUNC_DECL(addEquation, int64_t, PTR(void), ARRAY(int64_t))

RUNTIME_FUNC_DECL(addResidual, void, PTR(void), int32_t, RESIDUAL(float))
RUNTIME_FUNC_DECL(addResidual, void, PTR(void), int64_t, RESIDUAL(double))

RUNTIME_FUNC_DECL(addJacobian, void, PTR(void), int32_t, JACOBIAN(float))
RUNTIME_FUNC_DECL(addJacobian, void, PTR(void), int64_t, JACOBIAN(double))

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(addVariable, int32_t, PTR(void), ARRAY(float), bool)
RUNTIME_FUNC_DECL(addVariable, int64_t, PTR(void), ARRAY(double), bool)

RUNTIME_FUNC_DECL(addVariableAccess, void, PTR(void), int32_t, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DECL(addVariableAccess, void, PTR(void), int64_t, int64_t, ARRAY(int64_t))

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(getVariable, PTR(void), PTR(void), int32_t)
RUNTIME_FUNC_DECL(getVariable, PTR(void), PTR(void), int64_t)

RUNTIME_FUNC_DECL(getDerivative, PTR(void), PTR(void), int32_t)
RUNTIME_FUNC_DECL(getDerivative, PTR(void), PTR(void), int64_t)

RUNTIME_FUNC_DECL(getCurrentTime, float, PTR(void))
RUNTIME_FUNC_DECL(getCurrentTime, double, PTR(void))

#endif // MARCO_RUNTIME_IDA_H
