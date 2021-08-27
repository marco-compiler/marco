#include <ida/ida.h>
#include <marco/runtime/IdaFunctions.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>

#define exitOnError(success)                                                   \
	if (!success)                                                                \
		return false;

using Dimensions = std::vector<std::pair<int64_t, int64_t>>;
using Indexes = std::vector<int64_t>;
using Function = std::function<double(
		double tt, double cj, double *yy, double *yp, Indexes &ind, int64_t var)>;

/**
 * Container for all the data and lambda functions required by IDA in order to
 * compute the residual functions and the jacobian matrix.
 */
typedef struct IdaUserData
{
	// Model data
	std::vector<int64_t> rowLengths;
	std::vector<Dimensions> dimensions;
	std::vector<Function> residuals;
	std::vector<Function> jacobians;

	// Lambdas
	std::vector<std::pair<Function, Function>> lambdas;
	std::vector<std::vector<std::pair<int64_t, int64_t>>> lambdaAccesses;
	std::vector<std::vector<int64_t>> lambdaDimensions;

	// Simulation times
	realtype startTime;
	realtype stopTime;
	realtype time;

	// Error tolerances
	realtype relativeTolerance;
	realtype absoluteTolerance;

	// Matrix size
	sunindextype equationsNumber;
	sunindextype nonZeroValuesNumber;

	// Variables vectors and values
	N_Vector variablesVector;
	N_Vector derivativesVector;
	N_Vector idVector;
	realtype *variablesValues;
	realtype *derivativesValues;
	realtype *idValues;

	// IDA classes
	void *idaMemory;
	SUNMatrix sparseMatrix;
	SUNLinearSolver linearSolver;
	SUNNonlinearSolver nonlinearSolver;
} IdaUserData;

bool updateIndexes(Indexes &indexes, Dimensions dimension)
{
	for (int dim = dimension.size() - 1; true; dim--)
	{
		indexes[dim]++;
		if (indexes[dim] == dimension[dim].second)
		{
			if (dim == 0)
				return true;
			else
				indexes[dim] = dimension[dim].first;
		}
		else
		{
			return false;
		}
	}

	assert(false && "Unreachable");
}

/**
 * IDAResFn user-defined residual function, passed to IDA through IDAInit.
 * It contains how to compute the Residual function of the system, starting
 * from the provided UserData struct, iterating through every equation.
 */
int residualFunction(
		realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void *userData)
{
	realtype *yval = N_VGetArrayPointer(yy);
	realtype *ypval = N_VGetArrayPointer(yp);
	realtype *rval = N_VGetArrayPointer(rr);

	IdaUserData *data = static_cast<IdaUserData *>(userData);

	// For every vector equation
	for (size_t eq = 0; eq < data->dimensions.size(); eq++)
	{
		bool finished = false;

		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (size_t dim = 0; dim < data->dimensions[eq].size(); dim++)
			indexes.push_back(data->dimensions[eq][dim].first);

		// For every scalar equation in the vector equation
		while (!finished)
		{
			// Compute the i-th residual function
			*rval++ = data->residuals[eq](tt, 0, yval, ypval, indexes, 0);

			// Update multidimensional interval, exit while loop if finished
			finished = updateIndexes(indexes, data->dimensions[eq]);
		}
	}

	return 0;
}

/**
 * IDALsJacFn user-defined Jacobian approximation function, passed to IDA
 * through IDASetJacFn. It contains how to compute the Jacobian Matrix of
 * the system, starting from the provided UserData struct, iterating through
 * every equation and variable. The matrix is represented in CSR format.
 */
int jacobianMatrix(
		realtype tt,
		realtype cj,
		N_Vector yy,
		N_Vector yp,
		N_Vector rr,
		SUNMatrix JJ,
		void *userData,
		N_Vector tempv1,
		N_Vector tempv2,
		N_Vector tempv3)
{
	assert(SUNSparseMatrix_SparseType(JJ) == CSR_MAT);

	realtype *yval = N_VGetArrayPointer(yy);
	realtype *ypval = N_VGetArrayPointer(yp);

	sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(JJ);
	sunindextype *colvals = SUNSparseMatrix_IndexValues(JJ);

	realtype *jacobian = SUNSparseMatrix_Data(JJ);
	// SUNMatZero(JJ);

	IdaUserData *data = static_cast<IdaUserData *>(userData);

	int64_t nnzElements = 0;
	*rowptrs++ = nnzElements;

	// For every vector equation
	for (size_t eq = 0; eq < data->dimensions.size(); eq++)
	{
		bool finished = false;

		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (size_t dim = 0; dim < data->dimensions[eq].size(); dim++)
			indexes.push_back(data->dimensions[eq][dim].first);

		// For every scalar equation in the vector equation
		while (!finished)
		{
			nnzElements += data->rowLengths[eq];
			*rowptrs++ = nnzElements;

			// For every variable with respect to which every equation must be
			// partially differentiated
			for (int64_t var = 0; var < data->rowLengths[eq]; var++)
			{
				// Compute the i-th jacobian value
				*jacobian++ = data->jacobians[eq](tt, cj, yval, ypval, indexes, var);
				*colvals++ = var;
			}

			// Update multidimensional interval, exit while loop if finished
			finished = updateIndexes(indexes, data->dimensions[eq]);
		}
	}

	return 0;
}

/**
 * Check an IDA function return value in order to find possible failures.
 */
bool checkRetval(void *retval, const char *funcname, int opt)
{
	// Check if SUNDIALS function returned NULL pointer (no memory allocated)
	if (opt == 0 && retval == NULL)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname
								 << "() failed - returned NULL pointer\n";
		return false;
	}

	// Check if SUNDIALS function returned a positive integer value
	if (opt == 1 && *((int *) retval) < 0)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname
								 << "() failed  with return value = " << *(int *) retval
								 << "\n";
		return false;
	}

	return true;
}

//===----------------------------------------------------------------------===//
// Allocation, initialization and deletion
//===----------------------------------------------------------------------===//

/**
 * Instantiate and initialize the struct of data needed by IDA, given the total
 * number of equations and the maximum number of non-zero values of the jacobian
 * matrix.
 */
void *allocIdaUserData(int64_t neq, int64_t nnz)
{
	IdaUserData *data = new IdaUserData;

	data->equationsNumber = neq;
	data->nonZeroValuesNumber = nnz;

	// Create and initialize the required N-vectors for the variables.
	data->variablesVector = N_VNew_Serial(data->equationsNumber);
	assert(checkRetval((void *) data->variablesVector, "N_VNew_Serial", 0));

	data->derivativesVector = N_VNew_Serial(data->equationsNumber);
	assert(checkRetval((void *) data->derivativesVector, "N_VNew_Serial", 0));

	data->idVector = N_VNew_Serial(data->equationsNumber);
	assert(checkRetval((void *) data->idVector, "N_VNew_Serial", 0));

	data->variablesValues = N_VGetArrayPointer(data->variablesVector);
	data->derivativesValues = N_VGetArrayPointer(data->derivativesVector);
	data->idValues = N_VGetArrayPointer(data->idVector);

	return static_cast<void *>(data);
}

/**
 * Free all the data allocated by IDA.
 */
bool freeIdaUserData(void *userData)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	// Free IDA memory
	IDAFree(&data->idaMemory);
	int retval = SUNNonlinSolFree(data->nonlinearSolver);
	exitOnError(checkRetval(&retval, "SUNNonlinSolFree", 1));
	retval = SUNLinSolFree(data->linearSolver);
	exitOnError(checkRetval(&retval, "SUNLinSolFree", 1));
	SUNMatDestroy(data->sparseMatrix);
	N_VDestroy(data->variablesVector);
	N_VDestroy(data->derivativesVector);
	N_VDestroy(data->idVector);
	delete data;

	return true;
}

/**
 * Set the initial value of the index-th variable and if it is a state variable.
 */
void setInitialValues(void *userData, int64_t index, double value, bool isState)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	data->variablesValues[index] = value;
	data->derivativesValues[index] = 0.0;
	data->idValues[index] = isState ? 1.0 : 0.0;
}

/**
 * Instantiate and initialize all the classes needed by IDA in order to solve
 * the given system of equations. It must be called before the first usage of
 * step(). It may fail in case of malformed model.
 */
bool idaInit(void *userData)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	// Initialize IDA memory.
	data->idaMemory = IDACreate();
	exitOnError(checkRetval((void *) data->idaMemory, "IDACreate", 0));

	int retval = IDASetUserData(data->idaMemory, (void *) data);
	exitOnError(checkRetval(&retval, "IDASetUserData", 1));

	retval = IDASetId(data->idaMemory, data->idVector);
	exitOnError(checkRetval(&retval, "IDASetId", 1));

	retval = IDASetStopTime(data->idaMemory, data->stopTime);
	exitOnError(checkRetval(&retval, "IDASetStopTime", 1));

	retval = IDAInit(
			data->idaMemory,
			residualFunction,
			data->startTime,
			data->variablesVector,
			data->derivativesVector);
	exitOnError(checkRetval(&retval, "IDAInit", 1));

	// Call IDASStolerances to set tolerances.
	retval = IDASStolerances(
			data->idaMemory, data->relativeTolerance, data->absoluteTolerance);
	exitOnError(checkRetval(&retval, "IDASStolerances", 1));

	// Create sparse SUNMatrix for use in linear solver.
	data->sparseMatrix = SUNSparseMatrix(
			data->equationsNumber,
			data->equationsNumber,
			data->nonZeroValuesNumber,
			CSR_MAT);
	exitOnError(checkRetval((void *) data->sparseMatrix, "SUNSparseMatrix", 0));

	// Create and attach a KLU SUNLinearSolver object.
	data->linearSolver = SUNLinSol_KLU(data->variablesVector, data->sparseMatrix);
	exitOnError(checkRetval((void *) data->linearSolver, "SUNLinSol_KLU", 0));

	retval = IDASetLinearSolver(
			data->idaMemory, data->linearSolver, data->sparseMatrix);
	exitOnError(checkRetval(&retval, "IDASetLinearSolver", 1));

	// Create and attach a Newton NonlinearSolver object.
	data->nonlinearSolver = SUNNonlinSol_Newton(data->variablesVector);
	exitOnError(
			checkRetval((void *) data->nonlinearSolver, "SUNNonlinSol_Newton", 0));

	retval = IDASetNonlinearSolver(data->idaMemory, data->nonlinearSolver);
	exitOnError(checkRetval(&retval, "IDASetNonlinearSolver", 1));

	// Set the user-supplied Jacobian routine.
	retval = IDASetJacFn(data->idaMemory, jacobianMatrix);
	exitOnError(checkRetval(&retval, "IDASetJacFn", 1));

	// Call IDACalcIC to correct the initial values.
	retval = IDACalcIC(data->idaMemory, IDA_YA_YDP_INIT, data->stopTime);
	exitOnError(checkRetval(&retval, "IDACalcIC", 1));

	return true;
}

/**
 * Invoke IDA to perform one step of the computation. Returns 1 if the
 * computation has not reached the 'stopTime' seconds limit, 0 if it has reached
 * the end of the computation, -1 if it fails.
 */
int64_t idaStep(void *userData)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	// Execute one step
	int retval = IDASolve(
			data->idaMemory,
			data->stopTime,
			&data->time,
			data->variablesVector,
			data->derivativesVector,
			IDA_ONE_STEP);

	// Check if the solver failed
	if (!checkRetval(&retval, "IDASolve", 1))
		return -1;

	// Return if the computation has not reached the stop time yet.
	return data->time < data->stopTime ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// Setters
//===----------------------------------------------------------------------===//

/**
 * Add the start time and stop time to the user data.
 */
void addTime(void *userData, double startTime, double stopTime)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	data->startTime = startTime;
	data->stopTime = stopTime;
	data->time = startTime;
}

/**
 * Add the relative tolerance and the absolute tolerance to the user data.
 */
void addTolerances(void *userData, double relTol, double absTol)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	data->relativeTolerance = relTol;
	data->absoluteTolerance = absTol;
}

/**
 * Add the length of index-th row of the jacobian matrix to the user data.
 */
void addRowLength(void *userData, int64_t rowLength)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	data->rowLengths.push_back(rowLength);
}

/**
 * Add the dimension of the index-th equation to the user data.
 */
void addDimension(void *userData, int64_t index, int64_t min, int64_t max)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	if (index == (int64_t) data->dimensions.size())
		data->dimensions.push_back({});
	data->dimensions[index].push_back({ min - 1, max - 1 });
}

/**
 * Add the lambda that computes the index-th residual function to the user data.
 */
void addResidual(void *userData, int64_t leftIndex, int64_t rightIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	// Create the lambda function that subtract from the right side of the
	// equation, the left side of the equation.
	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;

	Function residual = [left, right](
													double tt,
													double cj,
													double *yy,
													double *yp,
													Indexes &ind,
													double var) -> double {
		return right(tt, cj, yy, yp, ind, var) - left(tt, cj, yy, yp, ind, var);
	};

	// Add the residual lambda function to the user data.
	data->residuals.push_back(std::move(residual));
}

/**
 * Add the lambda that computes the index-th jacobian row to the user data.
 */
void addJacobian(void *userData, int64_t leftIndex, int64_t rightIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	// Create the lambda function that subtract from the derivative of right side
	// of the equation, the derivative of  left side of the equation.
	Function left = data->lambdas[leftIndex].second;
	Function right = data->lambdas[rightIndex].second;

	Function jacobian = [left, right](
													double tt,
													double cj,
													double *yy,
													double *yp,
													Indexes &ind,
													double var) -> double {
		return right(tt, cj, yy, yp, ind, var) - left(tt, cj, yy, yp, ind, var);
	};

	// Add the jacobian lambda function to the user data.
	data->jacobians.push_back(std::move(jacobian));
}

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

double getIdaTime(void *userData)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	return data->time;
}

double getIdaVariable(void *userData, int64_t index)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	return data->variablesValues[index];
}

double getIdaDerivative(void *userData, int64_t index)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	return data->derivativesValues[index];
}

int64_t getIdaRowLength(void *userData, int64_t index)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	return data->rowLengths[index];
}

std::vector<std::pair<int64_t, int64_t>> getIdaDimension(
		void *userData, int64_t index)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	return data->dimensions[index];
}

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

int64_t numSteps(void *userData)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	int64_t nst;
	IDAGetNumSteps(data->idaMemory, &nst);
	return nst;
}

int64_t numResEvals(void *userData)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	int64_t nre;
	IDAGetNumResEvals(data->idaMemory, &nre);
	return nre;
}

int64_t numJacEvals(void *userData)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	int64_t nje;
	IDAGetNumJacEvals(data->idaMemory, &nje);
	return nje;
}

int64_t numNonlinIters(void *userData)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	int64_t nni;
	IDAGetNumNonlinSolvIters(data->idaMemory, &nni);
	return nni;
}

//===----------------------------------------------------------------------===//
// Lambda helpers
//===----------------------------------------------------------------------===//

int64_t addNewLambdaAccess(void *userData, int64_t off, int64_t ind)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	data->lambdaAccesses.push_back({ { off, ind } });
	return data->lambdaAccesses.size() - 1;
}

void addLambdaAccess(void *userData, int64_t index, int64_t off, int64_t ind)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	data->lambdaAccesses[index].push_back({ off, ind });
}

void addLambdaDimension(void *userData, int64_t index, int64_t dim)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);
	if (index == (int64_t) data->lambdaDimensions.size())
		data->lambdaDimensions.push_back({});
	data->lambdaDimensions[index].push_back(dim);
}

//===----------------------------------------------------------------------===//
// Lambda constructions
//===----------------------------------------------------------------------===//

int64_t lambdaConstant(void *userData, double constant)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function first = [constant](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double { return constant; };

	Function second =
			[](double tt, double cj, double *yy, double *yp, Indexes &ind, double var)
			-> double { return 0.0; };

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaTime(void *userData)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function first =
			[](double tt, double cj, double *yy, double *yp, Indexes &ind, double var)
			-> double { return tt; };

	Function second =
			[](double tt, double cj, double *yy, double *yp, Indexes &ind, double var)
			-> double { return 0.0; };

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaScalarVariable(void *userData, int64_t offset)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function first = [offset](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double { return yy[offset]; };

	Function second = [offset](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		if (offset == var)
			return 1.0;
		return 0.0;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaScalarDerivative(void *userData, int64_t offset)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function first = [offset](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double { return yp[offset]; };

	Function second = [offset](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		if (offset == var)
			return cj;
		return 0.0;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaVectorVariable(void *userData, int64_t offset, int64_t index)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	std::vector<std::pair<int64_t, int64_t>> access = data->lambdaAccesses[index];
	std::vector<int64_t> dim = data->lambdaDimensions[index];

	Function first = [offset, access, dim](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		int64_t varOffset = 0;

		for (size_t i = 0; i < access.size(); i++)
		{
			auto acc = access[i];
			int64_t accOffset = acc.first + (acc.second != -1 ? ind[acc.second] : 0);
			varOffset += accOffset * dim[i];
		}

		return yy[offset + varOffset];
	};

	Function second = [offset, access, dim](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		int64_t varOffset = 0;

		for (size_t i = 0; i < access.size(); i++)
		{
			auto acc = access[i];
			int64_t accOffset = acc.first + (acc.second != -1 ? ind[acc.second] : 0);
			varOffset += accOffset * dim[i];
		}

		if (offset + varOffset == var)
			return 1.0;
		return 0.0;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaVectorDerivative(void *userData, int64_t offset, int64_t index)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	std::vector<std::pair<int64_t, int64_t>> access = data->lambdaAccesses[index];
	std::vector<int64_t> dim = data->lambdaDimensions[index];

	Function first = [offset, access, dim](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		int64_t varOffset = 0;

		for (size_t i = 0; i < access.size(); i++)
		{
			auto acc = access[i];
			int64_t accOffset = acc.first + (acc.second != -1 ? ind[acc.second] : 0);
			varOffset += accOffset * dim[i];
		}

		return yp[offset + varOffset];
	};

	Function second = [offset, access, dim](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		int64_t varOffset = 0;

		for (size_t i = 0; i < access.size(); i++)
		{
			auto acc = access[i];
			int64_t accOffset = acc.first + (acc.second != -1 ? ind[acc.second] : 0);
			varOffset += accOffset * dim[i];
		}

		if (offset + varOffset == var)
			return cj;
		return 0.0;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaNegate(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return -operand(tt, cj, yy, yp, ind, var);
	};

	Function second = [derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return -derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaAdd(void *userData, int64_t leftIndex, int64_t rightIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;
	Function derLeft = data->lambdas[leftIndex].second;
	Function derRight = data->lambdas[rightIndex].second;

	Function first = [left, right](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return left(tt, cj, yy, yp, ind, var) + right(tt, cj, yy, yp, ind, var);
	};

	Function second = [derLeft, derRight](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return derLeft(tt, cj, yy, yp, ind, var) +
					 derRight(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaSub(void *userData, int64_t leftIndex, int64_t rightIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;
	Function derLeft = data->lambdas[leftIndex].second;
	Function derRight = data->lambdas[rightIndex].second;

	Function first = [left, right](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return left(tt, cj, yy, yp, ind, var) - right(tt, cj, yy, yp, ind, var);
	};

	Function second = [derLeft, derRight](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return derLeft(tt, cj, yy, yp, ind, var) -
					 derRight(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaMul(void *userData, int64_t leftIndex, int64_t rightIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;
	Function derLeft = data->lambdas[leftIndex].second;
	Function derRight = data->lambdas[rightIndex].second;

	Function first = [left, right](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return left(tt, cj, yy, yp, ind, var) * right(tt, cj, yy, yp, ind, var);
	};

	Function second = [left, right, derLeft, derRight](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return left(tt, cj, yy, yp, ind, var) * derRight(tt, cj, yy, yp, ind, var) +
					 right(tt, cj, yy, yp, ind, var) * derLeft(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaDiv(void *userData, int64_t leftIndex, int64_t rightIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;
	Function derLeft = data->lambdas[leftIndex].second;
	Function derRight = data->lambdas[rightIndex].second;

	Function first = [left, right](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return left(tt, cj, yy, yp, ind, var) / right(tt, cj, yy, yp, ind, var);
	};

	Function second = [left, right, derLeft, derRight](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		double rightValue = right(tt, cj, yy, yp, ind, var);
		double dividend =
				rightValue * derLeft(tt, cj, yy, yp, ind, var) -
				left(tt, cj, yy, yp, ind, var) * derRight(tt, cj, yy, yp, ind, var);
		return dividend / (rightValue * rightValue);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaPow(void *userData, int64_t baseIndex, int64_t exponentIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function base = data->lambdas[baseIndex].first;
	Function exponent = data->lambdas[exponentIndex].first;
	Function derBase = data->lambdas[baseIndex].second;

	Function first = [base, exponent](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::pow(
				base(tt, cj, yy, yp, ind, var), exponent(tt, cj, yy, yp, ind, var));
	};

	Function second = [base, exponent, derBase](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		double exponentValue = exponent(tt, cj, yy, yp, ind, var);
		return exponentValue *
					 std::pow(base(tt, cj, yy, yp, ind, var), exponentValue) *
					 derBase(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaAbs(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::abs(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		double x = operand(tt, cj, yy, yp, ind, var);

		return (x > 0.0) - (x < 0.0);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaSign(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		double x = operand(tt, cj, yy, yp, ind, var);

		return (x > 0.0) - (x < 0.0);
	};

	Function second =
			[](double tt, double cj, double *yy, double *yp, Indexes &ind, double var)
			-> double { return 0.0; };

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaSqrt(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::sqrt(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return derOperand(tt, cj, yy, yp, ind, var) /
					 std::sqrt(operand(tt, cj, yy, yp, ind, var)) / 2;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaExp(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::exp(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return std::exp(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaLog(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::log(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return derOperand(tt, cj, yy, yp, ind, var) /
					 operand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaLog10(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::log10(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return derOperand(tt, cj, yy, yp, ind, var) /
					 operand(tt, cj, yy, yp, ind, var) / std::log(10);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaSin(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::sin(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return std::cos(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaCos(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::cos(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return -std::sin(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaTan(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::tan(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		double tanOperandValue = std::tan(operand(tt, cj, yy, yp, ind, var));
		return (1 + tanOperandValue * tanOperandValue) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaAsin(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::asin(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		double operandValue = operand(tt, cj, yy, yp, ind, var);
		return derOperand(tt, cj, yy, yp, ind, var) /
					 (std::sqrt(1 - operandValue * operandValue));
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaAcos(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::acos(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		double operandValue = operand(tt, cj, yy, yp, ind, var);
		return -derOperand(tt, cj, yy, yp, ind, var) /
					 (std::sqrt(1 - operandValue * operandValue));
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaAtan(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::atan(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		double operandValue = operand(tt, cj, yy, yp, ind, var);
		return derOperand(tt, cj, yy, yp, ind, var) /
					 (1 + operandValue * operandValue);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaSinh(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::sinh(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return std::cosh(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaCosh(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::cosh(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		return std::sinh(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

int64_t lambdaTanh(void *userData, int64_t operandIndex)
{
	IdaUserData *data = static_cast<IdaUserData *>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 double tt,
											 double cj,
											 double *yy,
											 double *yp,
											 Indexes &ind,
											 double var) -> double {
		return std::tanh(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												double tt,
												double cj,
												double *yy,
												double *yp,
												Indexes &ind,
												double var) -> double {
		double tanhOperandValue = std::tanh(operand(tt, cj, yy, yp, ind, var));
		return (1 - tanhOperandValue * tanhOperandValue) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}
