#ifndef MARCO_RUNTIME_IDAFUNCTIONS_H
#define MARCO_RUNTIME_IDAFUNCTIONS_H

#include "ArrayDescriptor.h"
#include "Mangling.h"

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(allocIdaUserData, PTR(void), int32_t)
RUNTIME_FUNC_DECL(allocIdaUserData, PTR(void), int64_t)

RUNTIME_FUNC_DECL(idaInit, bool, PTR(void), int32_t)
RUNTIME_FUNC_DECL(idaInit, bool, PTR(void), int64_t)

RUNTIME_FUNC_DECL(idaStep, bool, PTR(void))

RUNTIME_FUNC_DECL(freeIdaUserData, bool, PTR(void))

RUNTIME_FUNC_DECL(addTime, void, PTR(void), float, float, float)
RUNTIME_FUNC_DECL(addTime, void, PTR(void), double, double, double)

RUNTIME_FUNC_DECL(addTolerance, void, PTR(void), float, float)
RUNTIME_FUNC_DECL(addTolerance, void, PTR(void), double, double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(addRowLength, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(addRowLength, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(addColumnIndex, void, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(addColumnIndex, void, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(addEqDimension, void, PTR(void), ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DECL(addEqDimension, void, PTR(void), ARRAY(int64_t), ARRAY(int64_t))

RUNTIME_FUNC_DECL(addResidual, void, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(addResidual, void, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(addJacobian, void, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(addJacobian, void, PTR(void), int64_t, int64_t)


//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(addVarOffset, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(addVarOffset, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(addVarDimension, void, PTR(void), ARRAY(int32_t))
RUNTIME_FUNC_DECL(addVarDimension, void, PTR(void), ARRAY(int64_t))

RUNTIME_FUNC_DECL(addVarAccess, int32_t, PTR(void), int32_t, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DECL(addVarAccess, int64_t, PTR(void), int64_t, ARRAY(int64_t), ARRAY(int64_t))

RUNTIME_FUNC_DECL(setInitialValue, void, PTR(void), int32_t, int32_t, int32_t, bool)
RUNTIME_FUNC_DECL(setInitialValue, void, PTR(void), int32_t, int32_t, float, bool)
RUNTIME_FUNC_DECL(setInitialValue, void, PTR(void), int64_t, int64_t, int64_t, bool)
RUNTIME_FUNC_DECL(setInitialValue, void, PTR(void), int64_t, int64_t, double, bool)

RUNTIME_FUNC_DECL(setInitialArray, void, PTR(void), int32_t, int32_t, ARRAY(int32_t), bool)
RUNTIME_FUNC_DECL(setInitialArray, void, PTR(void), int32_t, int32_t, ARRAY(float), bool)
RUNTIME_FUNC_DECL(setInitialArray, void, PTR(void), int64_t, int64_t, ARRAY(int64_t), bool)
RUNTIME_FUNC_DECL(setInitialArray, void, PTR(void), int64_t, int64_t, ARRAY(double), bool)

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(getIdaTime, float, PTR(void))
RUNTIME_FUNC_DECL(getIdaTime, double, PTR(void))

RUNTIME_FUNC_DECL(getIdaVariable, float, PTR(void), int32_t)
RUNTIME_FUNC_DECL(getIdaVariable, double, PTR(void), int64_t)

RUNTIME_FUNC_DECL(getIdaDerivative, float, PTR(void), int32_t)
RUNTIME_FUNC_DECL(getIdaDerivative, double, PTR(void), int64_t)

//===----------------------------------------------------------------------===//
// Lambda constructions
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(lambdaConstant, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaConstant, int64_t, PTR(void), int64_t)
RUNTIME_FUNC_DECL(lambdaConstant, int32_t, PTR(void), float)
RUNTIME_FUNC_DECL(lambdaConstant, int64_t, PTR(void), double)

RUNTIME_FUNC_DECL(lambdaTime, int32_t, PTR(void))
RUNTIME_FUNC_DECL(lambdaTime, int64_t, PTR(void))

RUNTIME_FUNC_DECL(lambdaInduction, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaInduction, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaVariable, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaVariable, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaDerivative, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaDerivative, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaNegate, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaNegate, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaAdd, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(lambdaAdd, int64_t, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(lambdaSub, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(lambdaSub, int64_t, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(lambdaMul, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(lambdaMul, int64_t, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(lambdaDiv, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(lambdaDiv, int64_t, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(lambdaPow, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(lambdaPow, int64_t, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(lambdaAtan2, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DECL(lambdaAtan2, int64_t, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(lambdaAbs, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaAbs, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaSign, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaSign, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaSqrt, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaSqrt, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaExp, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaExp, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaLog, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaLog, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaLog10, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaLog10, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaSin, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaSin, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaCos, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaCos, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaTan, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaTan, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaAsin, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaAsin, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaAcos, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaAcos, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaAtan, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaAtan, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaSinh, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaSinh, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaCosh, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaCosh, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaTanh, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DECL(lambdaTanh, int64_t, PTR(void), int64_t)

RUNTIME_FUNC_DECL(lambdaCall, int32_t, PTR(void), int32_t, FUNCTION(float), FUNCTION(float))
RUNTIME_FUNC_DECL(lambdaCall, int64_t, PTR(void), int64_t, FUNCTION(double), FUNCTION(double))

//===----------------------------------------------------------------------===//
// Debugging and Statistics
//===----------------------------------------------------------------------===//

extern "C"
{
	int64_t getNumberOfForEquations(void* userData);
	int64_t getNumberOfEquations(void* userData);
	int64_t getNumberOfNonZeroValues(void* userData);

	int64_t getIdaRowLength(void* userData, int64_t index);
	std::vector<std::pair<size_t, size_t>> getIdaDimension(void* userData, int64_t index);

	int64_t numSteps(void* userData);
	int64_t numResEvals(void* userData);
	int64_t numJacEvals(void* userData);
	int64_t numNonlinIters(void* userData);

	std::string getIncidenceMatrix(void* userData);
}

#endif	// MARCO_RUNTIME_IDAFUNCTIONS_H
