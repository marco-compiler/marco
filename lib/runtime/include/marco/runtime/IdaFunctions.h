#ifndef MARCO_RUNTIME_IDAFUNCTIONS_H
#define MARCO_RUNTIME_IDAFUNCTIONS_H

#include "ArrayDescriptor.h"
#include "Mangling.h"

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaAllocData, PTR(void), int32_t)
RUNTIME_FUNC_DECL(idaAllocData, PTR(void), int64_t)

RUNTIME_FUNC_DECL(idaInit, bool, PTR(void), int32_t)
RUNTIME_FUNC_DECL(idaInit, bool, PTR(void), int64_t)

RUNTIME_FUNC_DECL(idaStep, bool, PTR(void))

RUNTIME_FUNC_DECL(idaFreeData, bool, PTR(void))

RUNTIME_FUNC_DECL(addTime, void, PTR(void), float, float, float)
RUNTIME_FUNC_DECL(addTime, void, PTR(void), double, double, double)

RUNTIME_FUNC_DECL(addTolerance, void, PTR(void), float, float)
RUNTIME_FUNC_DECL(addTolerance, void, PTR(void), double, double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(addColumnIndex, void, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(addColumnIndex, void, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(addEqDimension, void, PTR(void), ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DECL(addEqDimension, void, PTR(void), ARRAY(int64_t), ARRAY(int64_t))

RUNTIME_FUNC_DECL(addResidual, void, PTR(void), RESIDUAL(float))
RUNTIME_FUNC_DECL(addResidual, void, PTR(void), RESIDUAL(double))

RUNTIME_FUNC_DECL(addJacobian, void, PTR(void), JACOBIAN(float))
RUNTIME_FUNC_DECL(addJacobian, void, PTR(void), JACOBIAN(double))

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(addVariable, void, PTR(void), int32_t, ARRAY(int32_t), bool)
RUNTIME_FUNC_DECL(addVariable, void, PTR(void), int32_t, ARRAY(float), bool)
RUNTIME_FUNC_DECL(addVariable, void, PTR(void), int64_t, ARRAY(int64_t), bool)
RUNTIME_FUNC_DECL(addVariable, void, PTR(void), int64_t, ARRAY(double), bool)

RUNTIME_FUNC_DECL(addVarAccess, int32_t, PTR(void), int32_t, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DECL(addVarAccess, int64_t, PTR(void), int64_t, ARRAY(int64_t), ARRAY(int64_t))

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(getIdaTime, float, PTR(void))
RUNTIME_FUNC_DECL(getIdaTime, double, PTR(void))

RUNTIME_FUNC_DECL(updateIdaVariable, void, PTR(void), int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DECL(updateIdaVariable, void, PTR(void), int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DECL(updateIdaVariable, void, PTR(void), int32_t, ARRAY(float))
RUNTIME_FUNC_DECL(updateIdaVariable, void, PTR(void), int64_t, ARRAY(double))

RUNTIME_FUNC_DECL(updateIdaDerivative, void, PTR(void), int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DECL(updateIdaDerivative, void, PTR(void), int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DECL(updateIdaDerivative, void, PTR(void), int32_t, ARRAY(float))
RUNTIME_FUNC_DECL(updateIdaDerivative, void, PTR(void), int64_t, ARRAY(double))

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(printStatistics, void, PTR(void))

#endif	// MARCO_RUNTIME_IDAFUNCTIONS_H
