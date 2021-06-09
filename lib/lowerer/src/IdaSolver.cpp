#include "modelica/lowerer/IdaSolver.hpp"

#include "modelica/model/ModErrors.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

IdaSolver::IdaSolver(
		LowererContext &context,
		const SmallVector<ModBltBlock, 3> &bltBlocks,
		const realtype startTime,
		const realtype stopTime,
		const realtype relativeTolerance,
		const realtype absoluteTolerance)
		: context(context),
			bltBlocks(bltBlocks),
			startTime(startTime),
			stopTime(stopTime),
			relativeTolerance(relativeTolerance),
			absoluteTolerance(absoluteTolerance),
			equationsNumber(computeNEQ()),
			nonZeroValuesNumber(computeNNZ())
{
}

IdaSolver::~IdaSolver()
{
	// Free memory
	IDAFree(&idaMemory);
	SUNLinSolFree(linearSolver);
	SUNMatDestroy(sparseMatrix);
	N_VDestroy(variablesVector);
	N_VDestroy(derivativesVector);
	N_VDestroy(idVector);
}

Error IdaSolver::init()
{
	// Create and initialize the required N-vectors for the variables
	variablesVector = N_VNew_Serial(equationsNumber);
	if (Error error = checkRetval((void *) variablesVector, "N_VNew_Serial", 0);
			error)
		return move(error);
	derivativesVector = N_VNew_Serial(equationsNumber);
	if (Error error = checkRetval((void *) derivativesVector, "N_VNew_Serial", 0);
			error)
		return move(error);
	idVector = N_VNew_Serial(equationsNumber);
	if (Error error = checkRetval((void *) idVector, "N_VNew_Serial", 0); error)
		return move(error);

	initVectors();

	// Call IDACreate and IDAInit to initialize IDA memory
	idaMemory = IDACreate();
	if (Error error = checkRetval((void *) idaMemory, "IDACreate", 0); error)
		return move(error);

	returnValue = IDASetUserData(idaMemory, (void *) this);
	if (Error error = checkRetval(&returnValue, "IDASetUserData", 1); error)
		return move(error);

	// Set which components are algebraic or differential
	returnValue = IDASetId(idaMemory, idVector);
	if (Error error = checkRetval(&returnValue, "IDASetId", 1); error)
		return move(error);

	returnValue = IDAInit(
			idaMemory,
			IdaSolver::residualFunction,
			startTime,
			variablesVector,
			derivativesVector);
	if (Error error = checkRetval(&returnValue, "IDAInit", 1); error)
		return move(error);

	// Call IDASVtolerances to set tolerances
	returnValue =
			IDASStolerances(idaMemory, relativeTolerance, absoluteTolerance);
	if (Error error = checkRetval(&returnValue, "IDASStolerances", 1); error)
		return move(error);

	// Create sparse SUNMatrix for use in linear solver
	sparseMatrix = SUNSparseMatrix(
			equationsNumber, equationsNumber, nonZeroValuesNumber, CSR_MAT);
	if (Error error = checkRetval((void *) sparseMatrix, "SUNSparseMatrix", 0);
			error)
		return move(error);

	// Create KLU SUNLinearSolver object
	linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix);
	if (Error error = checkRetval((void *) linearSolver, "SUNLinSol_KLU", 0);
			error)
		return move(error);

	// Attach the matrix and linear solver
	returnValue = IDASetLinearSolver(idaMemory, linearSolver, sparseMatrix);
	if (Error error = checkRetval(&returnValue, "IDASetLinearSolver", 1); error)
		return move(error);

	// Set the user-supplied Jacobian routine
	returnValue = IDASetJacFn(idaMemory, IdaSolver::jacobianMatrix);
	if (Error error = checkRetval(&returnValue, "IDASetJacFn", 1); error)
		return move(error);

	// Call IDACalcIC to correct the initial values
	returnValue = IDACalcIC(idaMemory, IDA_YA_YDP_INIT, stopTime);
	if (Error error = checkRetval(&returnValue, "IDACalcIC", 1); error)
		return move(error);

	return Error::success();
}

Expected<bool> IdaSolver::step()
{
	returnValue = IDASolve(
			idaMemory,
			stopTime,
			&time,
			variablesVector,
			derivativesVector,
			IDA_ONE_STEP);

	if (Error error = checkRetval(&returnValue, "IDASolve", 1); error)
		return move(error);

	return time < stopTime;
}

sunindextype IdaSolver::computeNEQ()
{
	sunindextype result = 0;
	for (ModBltBlock bltBlock : bltBlocks)
		result += bltBlock.size();
	return result;
}

sunindextype IdaSolver::computeNNZ()
{
	sunindextype result = 0, rowLength = 0;
	for (ModBltBlock bltBlock : bltBlocks)
	{
		rowLength += bltBlock.size();
		result += rowLength * bltBlock.size();
	}
	return result;
}

void IdaSolver::initVectors()
{
	variablesValues = N_VGetArrayPointer(variablesVector);
	derivativesValues = N_VGetArrayPointer(derivativesVector);
	idValues = N_VGetArrayPointer(idVector);
	// TODO: Create and load problem data block
	// TODO: Initialize variablesValues, derivativesValues, idValues
}

int IdaSolver::residualFunction(
		realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data)
{
	realtype *yval = N_VGetArrayPointer(yy);
	realtype *ypval = N_VGetArrayPointer(yp);
	realtype *rval = N_VGetArrayPointer(rr);

	IdaSolver *idaSolver = static_cast<IdaSolver *>(user_data);

	// TODO: Copmute the Residual Function.
	// For every equation in the matrix:
	//		For every induction in that equation:
	//			Assign to rval[i] the value where matched variables are substituted
	//			with yval and ypval, while how to compute the other hidden variables
	//			is done through the context.

	return 0;
}

int IdaSolver::jacobianMatrix(
		realtype tt,
		realtype cj,
		N_Vector yy,
		N_Vector yp,
		N_Vector resvec,
		SUNMatrix JJ,
		void *user_data,
		N_Vector tempv1,
		N_Vector tempv2,
		N_Vector tempv3)
{
	realtype *yval = N_VGetArrayPointer(yy);
	sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(JJ);
	sunindextype *colvals = SUNSparseMatrix_IndexValues(JJ);
	realtype *data = SUNSparseMatrix_Data(JJ);
	SUNMatZero(JJ);

	IdaSolver *idaSolver = static_cast<IdaSolver *>(user_data);

	// TODO: Compute the Jacobian Matrix.
	// For every equation in the matrix:
	//		For every induction in that equation:
	//			Assign to JJ[i][j] the partial derivative wrt the matched variables,
	//			which are substituted with yval and ypval, while how to compute the
	//			other hidden variables is done through the context.

	return 0;
}

Error IdaSolver::checkRetval(void *returnvalue, const char *funcname, int opt)
{
	// Check if SUNDIALS function returned NULL pointer (no memory allocated)
	if (opt == 0 && returnvalue == NULL)
		return make_error<SundialsError>(returnvalue, funcname, opt);

	// Check if SUNDIALS function returned a positive integer value
	if (opt == 1 && *((int *) returnvalue) < 0)
		return make_error<SundialsError>(returnvalue, funcname, opt);

	// Check if function returned NULL pointer (no memory allocated)
	if (opt == 2 && returnvalue == NULL)
		return make_error<SundialsError>(returnvalue, funcname, opt);

	return Error::success();
}
