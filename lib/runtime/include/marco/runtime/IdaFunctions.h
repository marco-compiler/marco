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
			void *userData, int64_t index, double value, bool isState);
	bool idaInit(void *userData);
	int64_t idaStep(void *userData);

	// Setters
	void addTime(void *userData, double startTime, double stopTime);
	void addTolerances(void *userData, double relTol, double absTol);

	void addRowLength(void *userData, int64_t rowLength);
	void addDimension(void *userData, int64_t index, int64_t min, int64_t max);
	void addResidual(void *userData, int64_t leftIndex, int64_t rightIndex);
	void addJacobian(void *userData, int64_t leftIndex, int64_t rightIndex);

	// Getters
	double getIdaTime(void *userData);
	double getIdaVariable(void *userData, int64_t index);
	double getIdaDerivative(void *userData, int64_t index);

	int64_t getIdaRowLength(void *userData, int64_t index);
	std::vector<std::pair<int64_t, int64_t>> getIdaDimension(
			void *userData, int64_t index);

	// Statistics
	int64_t numSteps(void *userData);
	int64_t numResEvals(void *userData);
	int64_t numJacEvals(void *userData);
	int64_t numNonlinIters(void *userData);

	// Lambda helpers
	int64_t addNewLambdaAccess(void *userData, int64_t off, int64_t ind);
	void addLambdaAccess(void *userData, int64_t index, int64_t off, int64_t ind);
	void addLambdaDimension(void *userData, int64_t index, int64_t dim);

	// Lambda constructions
	int64_t lambdaConstant(void *userData, double constant);
	int64_t lambdaTime(void *userData);
	int64_t lambdaScalarVariable(void *userData, int64_t offset);
	int64_t lambdaScalarDerivative(void *userData, int64_t offset);
	int64_t lambdaVectorVariable(void *userData, int64_t offset, int64_t index);
	int64_t lambdaVectorDerivative(void *userData, int64_t offset, int64_t index);

	int64_t lambdaNegate(void *userData, int64_t operandIndex);
	int64_t lambdaAdd(void *userData, int64_t leftIndex, int64_t rightIndex);
	int64_t lambdaSub(void *userData, int64_t leftIndex, int64_t rightIndex);
	int64_t lambdaMul(void *userData, int64_t leftIndex, int64_t rightIndex);
	int64_t lambdaDiv(void *userData, int64_t leftIndex, int64_t rightIndex);

	int64_t lambdaPow(void *userData, int64_t baseIndex, int64_t exponentIndex);
	int64_t lambdaAbs(void *userData, int64_t operandIndex);
	int64_t lambdaSign(void *userData, int64_t operandIndex);
	int64_t lambdaSqrt(void *userData, int64_t operandIndex);
	int64_t lambdaExp(void *userData, int64_t operandIndex);
	int64_t lambdaLog(void *userData, int64_t operandIndex);
	int64_t lambdaLog10(void *userData, int64_t operandIndex);

	int64_t lambdaSin(void *userData, int64_t operandIndex);
	int64_t lambdaCos(void *userData, int64_t operandIndex);
	int64_t lambdaTan(void *userData, int64_t operandIndex);
	int64_t lambdaAsin(void *userData, int64_t operandIndex);
	int64_t lambdaAcos(void *userData, int64_t operandIndex);
	int64_t lambdaAtan(void *userData, int64_t operandIndex);
	int64_t lambdaSinh(void *userData, int64_t operandIndex);
	int64_t lambdaCosh(void *userData, int64_t operandIndex);
	int64_t lambdaTanh(void *userData, int64_t operandIndex);
}

#endif	// MARCO_RUNTIME_IDAFUNCTIONS_H
