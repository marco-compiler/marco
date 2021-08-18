#ifndef MARCO_RUNTIME_IDAFUNCTIONS_H
#define MARCO_RUNTIME_IDAFUNCTIONS_H

#include "ArrayDescriptor.h"
#include "Mangling.h"

extern "C"
{
	// Allocation, initialization and deletion.
	void *allocIdaUserData(int64_t neq, int64_t nnz);
	bool freeIdaUserData(void *userData);

	void setInitialValues(
			void *userData, size_t index, double value, bool isState);
	bool idaInit(void *userData);
	int8_t idaStep(void *userData);

	// Setters
	void addTime(void *userData, double startTime, double stopTime);
	void addTolerances(void *userData, double relTol, double absTol);

	void addRowLength(void *userData, size_t rowLength);
	void addDimension(void *userData, size_t index, size_t min, size_t max);
	void addResidual(void *userData, size_t leftIndex, size_t rightIndex);
	void addJacobian(void *userData, size_t leftIndex, size_t rightIndex);

	// Getters
	double getIdaTime(void *userData);
	double getIdaVariable(void *userData, size_t index);
	double getIdaDerivative(void *userData, size_t index);

	size_t getIdaRowLength(void *userData, size_t index);
	std::vector<std::pair<size_t, size_t>> getIdaDimension(
			void *userData, size_t index);

	// Statistics
	int64_t numSteps(void *userData);
	int64_t numResEvals(void *userData);
	int64_t numJacEvals(void *userData);
	int64_t numNonlinIters(void *userData);

	// Lambda helpers
	size_t addNewLambdaAccess(void *userData, int64_t off, int64_t ind);
	void addLambdaAccess(void *userData, size_t index, int64_t off, int64_t ind);
	void addLambdaDimension(void *userData, size_t index, size_t dim);

	// Lambda constructions
	size_t lambdaConstant(void *userData, double constant);
	size_t lambdaTime(void *userData);
	size_t lambdaScalarVariable(void *userData, size_t offset);
	size_t lambdaScalarDerivative(void *userData, size_t offset);
	size_t lambdaVectorVariable(void *userData, size_t offset, size_t index);
	size_t lambdaVectorDerivative(void *userData, size_t offset, size_t index);

	size_t lambdaNegate(void *userData, size_t operandIndex);
	size_t lambdaAdd(void *userData, size_t leftIndex, size_t rightIndex);
	size_t lambdaSub(void *userData, size_t leftIndex, size_t rightIndex);
	size_t lambdaMul(void *userData, size_t leftIndex, size_t rightIndex);
	size_t lambdaDiv(void *userData, size_t leftIndex, size_t rightIndex);

	size_t lambdaPow(void *userData, size_t baseIndex, size_t exponentIndex);
	size_t lambdaAbs(void *userData, size_t operandIndex);
	size_t lambdaSign(void *userData, size_t operandIndex);
	size_t lambdaSqrt(void *userData, size_t operandIndex);
	size_t lambdaExp(void *userData, size_t operandIndex);
	size_t lambdaLog(void *userData, size_t operandIndex);
	size_t lambdaLog10(void *userData, size_t operandIndex);

	size_t lambdaSin(void *userData, size_t operandIndex);
	size_t lambdaCos(void *userData, size_t operandIndex);
	size_t lambdaTan(void *userData, size_t operandIndex);
	size_t lambdaAsin(void *userData, size_t operandIndex);
	size_t lambdaAcos(void *userData, size_t operandIndex);
	size_t lambdaAtan(void *userData, size_t operandIndex);
	size_t lambdaSinh(void *userData, size_t operandIndex);
	size_t lambdaCosh(void *userData, size_t operandIndex);
	size_t lambdaTanh(void *userData, size_t operandIndex);
}

#endif	// MARCO_RUNTIME_IDAFUNCTIONS_H
