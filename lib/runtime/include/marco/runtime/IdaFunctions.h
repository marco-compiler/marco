#ifndef MARCO_RUNTIME_IDAFUNCTIONS_H
#define MARCO_RUNTIME_IDAFUNCTIONS_H

#include "ArrayDescriptor.h"
#include "Mangling.h"

extern "C"
{
	// Allocation, initialization, usage and deletion
	void* allocIdaUserData(sunindextype equationsNumber);
	bool freeIdaUserData(void* userData);

	void setInitialValue(void* userData, sunindextype index, sunindextype length, realtype value, bool isState);
	void setInitialArray(void* userData, sunindextype index, sunindextype length, UnsizedArrayDescriptor<realtype> array, bool isState);
	bool idaInit(void* userData);
	bool idaStep(void* userData);

	// Equation setters
	void addTime(void* userData, realtype startTime, realtype stopTime);
	void addTolerance(void* userData, realtype relTol, realtype absTol);

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
