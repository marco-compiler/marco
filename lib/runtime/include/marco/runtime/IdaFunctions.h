#ifndef MARCO_RUNTIME_IDAFUNCTIONS_H
#define MARCO_RUNTIME_IDAFUNCTIONS_H

#include "ArrayDescriptor.h"
#include "Mangling.h"

extern "C"
{
	// Allocation, initialization, usage and deletion
	void* allocIdaUserData(sunindextype equationsNumber);
	bool idaInit(void* userData);
	bool idaStep(void* userData);
	bool freeIdaUserData(void* userData);

	void addTime(void* userData, realtype startTime, realtype endTime, realtype timeStep);
	void addTolerance(void* userData, realtype relTol, realtype absTol);

	// Equation setters
	sunindextype addRowLength(void* userData, sunindextype rowLength);
	void addColumnIndex(void* userData, sunindextype rowIndex, sunindextype accessIndex);
	void addEquationDimension(void* userData, sunindextype index, sunindextype min, sunindextype max);
	void addResidual(void* userData, sunindextype leftIndex, sunindextype rightIndex);
	void addJacobian(void* userData, sunindextype leftIndex, sunindextype rightIndex);

	// Variable setters
	sunindextype addVariableOffset(void* userData, sunindextype offset);
	void addVariableDimension(void* userData, sunindextype index, sunindextype dim);
	sunindextype addNewVariableAccess(void* userData, sunindextype var, sunindextype off, sunindextype ind);
	void addVariableAccess(void* userData, sunindextype index, sunindextype off, sunindextype ind);

	RUNTIME_FUNC_DECL(setInitialValue, void, voidptr, int32_t, int32_t, int32_t, bool)
	RUNTIME_FUNC_DECL(setInitialValue, void, voidptr, int32_t, int32_t, float, bool)
	RUNTIME_FUNC_DECL(setInitialValue, void, voidptr, int64_t, int64_t, int64_t, bool)
	RUNTIME_FUNC_DECL(setInitialValue, void, voidptr, int64_t, int64_t, double, bool)

	RUNTIME_FUNC_DECL(setInitialArray, void, voidptr, int32_t, int32_t, ARRAY(int32_t), bool)
	RUNTIME_FUNC_DECL(setInitialArray, void, voidptr, int32_t, int32_t, ARRAY(float), bool)
	RUNTIME_FUNC_DECL(setInitialArray, void, voidptr, int64_t, int64_t, ARRAY(int64_t), bool)
	RUNTIME_FUNC_DECL(setInitialArray, void, voidptr, int64_t, int64_t, ARRAY(double), bool)

	// Getters
	realtype getIdaTime(void* userData);
	realtype getIdaVariable(void* userData, sunindextype index);
	realtype getIdaDerivative(void* userData, sunindextype index);
	sunindextype getNumberOfEquations(void* userData);
	sunindextype getNumberOfNonZeroValues(void* userData);

	sunindextype getIdaRowLength(void* userData, sunindextype index);
	std::vector<std::pair<size_t, size_t>> getIdaDimension(void* userData, sunindextype index);

	// Statistics
	sunindextype numSteps(void* userData);
	sunindextype numResEvals(void* userData);
	sunindextype numJacEvals(void* userData);
	sunindextype numNonlinIters(void* userData);

	std::string getIncidenceMatrix(void* userData);

	// Lambda constructions
	sunindextype lambdaConstant(void* userData, realtype constant);
	sunindextype lambdaTime(void* userData);
	sunindextype lambdaInduction(void* userData, sunindextype induction);
	sunindextype lambdaVariable(void* userData, sunindextype accessIndex);
	sunindextype lambdaDerivative(void* userData, sunindextype accessIndex);

	sunindextype lambdaAdd(void* userData, sunindextype leftIndex, sunindextype rightIndex);
	sunindextype lambdaSub(void* userData, sunindextype leftIndex, sunindextype rightIndex);
	sunindextype lambdaMul(void* userData, sunindextype leftIndex, sunindextype rightIndex);
	sunindextype lambdaDiv(void* userData, sunindextype leftIndex, sunindextype rightIndex);
	sunindextype lambdaPow(void* userData, sunindextype leftIndex, sunindextype rightIndex);
	sunindextype lambdaAtan2(void* userData, sunindextype leftIndex, sunindextype rightIndex);

	sunindextype lambdaNegate(void* userData, sunindextype operandIndex);
	sunindextype lambdaAbs(void* userData, sunindextype operandIndex);
	sunindextype lambdaSign(void* userData, sunindextype operandIndex);
	sunindextype lambdaSqrt(void* userData, sunindextype operandIndex);
	sunindextype lambdaExp(void* userData, sunindextype operandIndex);
	sunindextype lambdaLog(void* userData, sunindextype operandIndex);
	sunindextype lambdaLog10(void* userData, sunindextype operandIndex);

	sunindextype lambdaSin(void* userData, sunindextype operandIndex);
	sunindextype lambdaCos(void* userData, sunindextype operandIndex);
	sunindextype lambdaTan(void* userData, sunindextype operandIndex);
	sunindextype lambdaAsin(void* userData, sunindextype operandIndex);
	sunindextype lambdaAcos(void* userData, sunindextype operandIndex);
	sunindextype lambdaAtan(void* userData, sunindextype operandIndex);
	sunindextype lambdaSinh(void* userData, sunindextype operandIndex);
	sunindextype lambdaCosh(void* userData, sunindextype operandIndex);
	sunindextype lambdaTanh(void* userData, sunindextype operandIndex);

	sunindextype lambdaCall(void* userData, sunindextype operandIndex, realtype (*function)(realtype), realtype (*pderFunc)(realtype));
}

#endif	// MARCO_RUNTIME_IDAFUNCTIONS_H
