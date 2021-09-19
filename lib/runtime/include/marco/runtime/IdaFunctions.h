#ifndef MARCO_RUNTIME_IDAFUNCTIONS_H
#define MARCO_RUNTIME_IDAFUNCTIONS_H

#include "ArrayDescriptor.h"
#include "Mangling.h"

extern "C"
{
	// Allocation, initialization, usage and deletion
	void* allocIdaUserData(sunindextype neq, sunindextype nnz);
	bool freeIdaUserData(void* userData);

	void setInitialValue(void* userData, sunindextype index, sunindextype length, realtype value, bool isState);
	bool idaInit(void* userData);
	bool idaStep(void* userData);

	// Setters
	void addTime(void* userData, realtype startTime, realtype stopTime);
	void addTolerance(void* userData, realtype relTol, realtype absTol);

	void addRowLength(void* userData, sunindextype rowLength);
	void addDimension(void* userData, sunindextype index, sunindextype min, sunindextype max);
	void addResidual(void* userData, sunindextype leftIndex, sunindextype rightIndex);
	void addJacobian(void* userData, sunindextype leftIndex, sunindextype rightIndex);

	// Getters
	realtype getIdaTime(void* userData);
	realtype getIdaVariable(void* userData, sunindextype index);
	realtype getIdaDerivative(void* userData, sunindextype index);

	sunindextype getIdaRowLength(void* userData, sunindextype index);
	std::vector<std::pair<sunindextype, sunindextype>> getIdaDimension(void* userData, sunindextype index);

	// Statistics
	sunindextype numSteps(void* userData);
	sunindextype numResEvals(void* userData);
	sunindextype numJacEvals(void* userData);
	sunindextype numNonlinIters(void* userData);

	// Lambda helpers
	sunindextype addNewLambdaAccess(void* userData, sunindextype off, sunindextype ind);
	void addLambdaAccess(void* userData, sunindextype index, sunindextype off, sunindextype ind);
	void addLambdaDimension(void* userData, sunindextype index, sunindextype dim);

	// Lambda constructions
	sunindextype lambdaConstant(void* userData, realtype constant);
	sunindextype lambdaTime(void* userData);
	sunindextype lambdaScalarVariable(void* userData, sunindextype offset);
	sunindextype lambdaScalarDerivative(void* userData, sunindextype offset);
	sunindextype lambdaVectorVariable(void* userData, sunindextype offset, sunindextype index);
	sunindextype lambdaVectorDerivative(void* userData, sunindextype offset, sunindextype index);

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

	sunindextype lambdaCall(void* userData, sunindextype operandIndex, realtype (*function)(realtype));
}

#endif	// MARCO_RUNTIME_IDAFUNCTIONS_H
