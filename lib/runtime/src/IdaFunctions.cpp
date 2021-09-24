#include <ida/ida.h>
#include <marco/runtime/IdaFunctions.h>
#include <nvector/nvector_serial.h>
#include <set>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>

#define exitOnError(success)                                                   \
	if (!success)                                                                \
		return false;

using Dimension = std::vector<std::pair<sunindextype, sunindextype>>;
using Access = std::vector<std::pair<sunindextype, sunindextype>>;
using Indexes = std::vector<sunindextype>;
using Function = std::function<realtype(
		realtype tt,
		realtype cj,
		realtype* yy,
		realtype* yp,
		Indexes& ind,
		sunindextype var)>;

/**
 * Container for all the data and lambda functions required by IDA in order to
 * compute the residual functions and the jacobian matrix.
 */
typedef struct IdaUserData
{
	// Equations data
	std::vector<sunindextype> rowLengths;
	std::vector<std::vector<sunindextype>> columnIndexes;
	std::vector<Dimension> equationDimensions;
	std::vector<Function> residuals;
	std::vector<Function> jacobians;
	std::vector<std::pair<Function, Function>> lambdas;

	// Variables data
	std::vector<sunindextype> variableOffsets;
	std::vector<std::pair<sunindextype, Access>> variableAccesses;
	std::vector<std::vector<sunindextype>> variableDimensions;

	// Matrix size
	sunindextype equationsNumber;
	sunindextype nonZeroValuesNumber;

	// Simulation times
	realtype startTime;
	realtype stopTime;
	realtype time;

	// Error tolerances
	realtype relativeTolerance;
	realtype absoluteTolerance;

	// Variables vectors and values
	N_Vector variablesVector;
	N_Vector derivativesVector;
	N_Vector idVector;
	realtype* variablesValues;
	realtype* derivativesValues;
	realtype* idValues;

	// IDA classes
	void* idaMemory;
	SUNMatrix sparseMatrix;
	SUNLinearSolver linearSolver;
	SUNNonlinearSolver nonlinearSolver;
} IdaUserData;

bool updateIndexes(Indexes& indexes, Dimension dimension)
{
	for (sunindextype dim = dimension.size() - 1; true; dim--)
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
		realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void* userData)
{
	realtype* yval = N_VGetArrayPointer(yy);
	realtype* ypval = N_VGetArrayPointer(yp);
	realtype* rval = N_VGetArrayPointer(rr);

	IdaUserData* data = static_cast<IdaUserData*>(userData);

	// For every vector equation
	for (size_t eq = 0; eq < data->equationDimensions.size(); eq++)
	{
		bool finished = false;

		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (size_t dim = 0; dim < data->equationDimensions[eq].size(); dim++)
			indexes.push_back(data->equationDimensions[eq][dim].first);

		// For every scalar equation in the vector equation
		while (!finished)
		{
			// Compute the i-th residual function
			*rval++ = data->residuals[eq](tt, 0, yval, ypval, indexes, 0);

			// Update multidimensional interval, exit while loop if finished
			finished = updateIndexes(indexes, data->equationDimensions[eq]);
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
		void* userData,
		N_Vector tempv1,
		N_Vector tempv2,
		N_Vector tempv3)
{
	assert(SUNSparseMatrix_SparseType(JJ) == CSR_MAT);

	realtype* yval = N_VGetArrayPointer(yy);
	realtype* ypval = N_VGetArrayPointer(yp);

	sunindextype* rowptrs = SUNSparseMatrix_IndexPointers(JJ);
	sunindextype* colvals = SUNSparseMatrix_IndexValues(JJ);

	realtype* jacobian = SUNSparseMatrix_Data(JJ);
	// SUNMatZero(JJ);

	IdaUserData* data = static_cast<IdaUserData*>(userData);

	sunindextype nnzElements = 0;
	*rowptrs++ = nnzElements;

	// For every vector equation
	for (size_t eq = 0; eq < data->equationDimensions.size(); eq++)
	{
		bool finished = false;

		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (size_t dim = 0; dim < data->equationDimensions[eq].size(); dim++)
			indexes.push_back(data->equationDimensions[eq][dim].first);

		// For every scalar equation in the vector equation
		while (!finished)
		{
			nnzElements += data->rowLengths[eq];
			*rowptrs++ = nnzElements;

			// Compute the column indexes that may be non-zeros.
			std::set<sunindextype> columnIndexesSet;
			for (sunindextype accessIndex : data->columnIndexes[eq])
			{
				sunindextype varOffset = 0;
				sunindextype varIndex = data->variableAccesses[accessIndex].first;
				auto dimensions = data->variableDimensions[varIndex];

				for (size_t i = 0;
						 i < data->variableAccesses[accessIndex].second.size();
						 i++)
				{
					auto acc = data->variableAccesses[accessIndex].second[i];
					sunindextype accOffset =
							acc.first + (acc.second != -1 ? indexes[acc.second] : 0);
					varOffset += accOffset * dimensions[i];
				}

				columnIndexesSet.insert(data->variableOffsets[varIndex] + varOffset);
			}

			assert((size_t) data->rowLengths[eq] == columnIndexesSet.size());

			// For every variable with respect to which every equation must be
			// partially differentiated
			for (sunindextype var : columnIndexesSet)
			{
				// Compute the i-th jacobian value
				*jacobian++ = data->jacobians[eq](tt, cj, yval, ypval, indexes, var);
				*colvals++ = var;
			}

			// Update multidimensional interval, exit while loop if finished
			finished = updateIndexes(indexes, data->equationDimensions[eq]);
		}
	}

	return 0;
}

/**
 * Check an IDA function return value in order to find possible failures.
 */
bool checkRetval(void* retval, const char* funcname, int opt)
{
	// Check if SUNDIALS function returned NULL pointer (no memory allocated)
	if (opt == 0 && retval == NULL)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname
								 << "() failed - returned NULL pointer\n";
		return false;
	}

	// Check if SUNDIALS function returned a positive integer value
	if (opt == 1 && *((int*) retval) < 0)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname
								 << "() failed  with return value = " << *(int*) retval << "\n";
		return false;
	}

	return true;
}

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

/**
 * Instantiate and initialize the struct of data needed by IDA, given the total
 * number of equations and the maximum number of non-zero values of the jacobian
 * matrix.
 */
void* allocIdaUserData(sunindextype neq, sunindextype nnz)
{
	IdaUserData* data = new IdaUserData;

	data->equationsNumber = neq;
	data->nonZeroValuesNumber = nnz;

	// Create and initialize the required N-vectors for the variables.
	data->variablesVector = N_VNew_Serial(data->equationsNumber);
	assert(checkRetval((void*) data->variablesVector, "N_VNew_Serial", 0));

	data->derivativesVector = N_VNew_Serial(data->equationsNumber);
	assert(checkRetval((void*) data->derivativesVector, "N_VNew_Serial", 0));

	data->idVector = N_VNew_Serial(data->equationsNumber);
	assert(checkRetval((void*) data->idVector, "N_VNew_Serial", 0));

	data->variablesValues = N_VGetArrayPointer(data->variablesVector);
	data->derivativesValues = N_VGetArrayPointer(data->derivativesVector);
	data->idValues = N_VGetArrayPointer(data->idVector);

	return static_cast<void*>(data);
}

/**
 * Free all the data allocated by IDA.
 */
bool freeIdaUserData(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

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
void setInitialValue(
		void* userData,
		sunindextype index,
		sunindextype length,
		realtype value,
		bool isState)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	sunindextype offset = data->variableOffsets[index];
	realtype idValue = isState ? 1.0 : 0.0;

	for (sunindextype i = 0; i < length; i++)
	{
		data->variablesValues[offset + i] = value;
		data->derivativesValues[offset + i] = 0.0;
		data->idValues[offset + i] = idValue;
	}
}

/**
 * Instantiate and initialize all the classes needed by IDA in order to solve
 * the given system of equations. It must be called before the first usage of
 * step(). It may fail in case of malformed model.
 */
bool idaInit(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->equationsNumber == 0)
		return true;

	// Initialize IDA memory.
	data->idaMemory = IDACreate();
	exitOnError(checkRetval((void*) data->idaMemory, "IDACreate", 0));

	int retval = IDASetUserData(data->idaMemory, (void*) data);
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
	exitOnError(checkRetval((void*) data->sparseMatrix, "SUNSparseMatrix", 0));

	// Create and attach a KLU SUNLinearSolver object.
	data->linearSolver = SUNLinSol_KLU(data->variablesVector, data->sparseMatrix);
	exitOnError(checkRetval((void*) data->linearSolver, "SUNLinSol_KLU", 0));

	retval = IDASetLinearSolver(
			data->idaMemory, data->linearSolver, data->sparseMatrix);
	exitOnError(checkRetval(&retval, "IDASetLinearSolver", 1));

	// Create and attach a Newton NonlinearSolver object.
	data->nonlinearSolver = SUNNonlinSol_Newton(data->variablesVector);
	exitOnError(
			checkRetval((void*) data->nonlinearSolver, "SUNNonlinSol_Newton", 0));

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
 * Invoke IDA to perform one step of the computation. Returns false if the
 * computation failed, true otherwise.
 */
bool idaStep(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->equationsNumber == 0)
		return true;

	// Execute one step
	int retval = IDASolve(
			data->idaMemory,
			data->stopTime,
			&data->time,
			data->variablesVector,
			data->derivativesVector,
			IDA_ONE_STEP);

	// Check if the solver failed
	exitOnError(checkRetval(&retval, "IDASolve", 1));

	return true;
}

//===----------------------------------------------------------------------===//
// Setters
//===----------------------------------------------------------------------===//

/**
 * Add the start time and stop time to the user data.
 */
void addTime(void* userData, realtype startTime, realtype stopTime)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	data->startTime = startTime;
	data->stopTime = stopTime;
	data->time = startTime;
}

/**
 * Add the relative tolerance and the absolute tolerance to the user data.
 */
void addTolerance(void* userData, realtype relTol, realtype absTol)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	data->relativeTolerance = relTol;
	data->absoluteTolerance = absTol;
}

/**
 * Add the length of index-th row of the jacobian matrix to the user data.
 */
sunindextype addRowLength(void* userData, sunindextype rowLength)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	data->rowLengths.push_back(rowLength);
	return data->rowLengths.size() - 1;
}

/**
 * Add the access index of a non-zero value contained in the rowIndex-th row of
 * the jacobian matrix to the user data.
 */
void addColumnIndex(
		void* userData, sunindextype rowIndex, sunindextype accessIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if ((size_t) rowIndex == data->columnIndexes.size())
		data->columnIndexes.push_back({ accessIndex });
	else
		data->columnIndexes[rowIndex].push_back(accessIndex);
}

/**
 * Add the dimension of the index-th equation to the user data.
 */
void addEquationDimension(
		void* userData, sunindextype index, sunindextype min, sunindextype max)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if ((size_t) index == data->equationDimensions.size())
		data->equationDimensions.push_back({ { min, max } });
	else
		data->equationDimensions[index].push_back({ min, max });
}

/**
 * Add the lambda that computes the index-th residual function to the user data.
 * Must be used before the add jacobian function.
 */
void addResidual(
		void* userData, sunindextype leftIndex, sunindextype rightIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	// Create the lambda function that subtract from the right side of the
	// equation, the left side of the equation.
	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;

	Function residual = [left, right](
													realtype tt,
													realtype cj,
													realtype* yy,
													realtype* yp,
													Indexes& ind,
													realtype var) -> realtype {
		return right(tt, cj, yy, yp, ind, var) - left(tt, cj, yy, yp, ind, var);
	};

	// Add the residual lambda function to the user data.
	data->residuals.push_back(std::move(residual));
}

/**
 * Add the lambda that computes the index-th jacobian row to the user data.
 * Must be used after the add residual function.
 */
void addJacobian(
		void* userData, sunindextype leftIndex, sunindextype rightIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	// Create the lambda function that subtract from the derivative of right side
	// of the equation, the derivative of  left side of the equation.
	Function left = data->lambdas[leftIndex].second;
	Function right = data->lambdas[rightIndex].second;

	Function jacobian = [left, right](
													realtype tt,
													realtype cj,
													realtype* yy,
													realtype* yp,
													Indexes& ind,
													realtype var) -> realtype {
		return right(tt, cj, yy, yp, ind, var) - left(tt, cj, yy, yp, ind, var);
	};

	// Add the jacobian lambda function to the user data.
	data->jacobians.push_back(std::move(jacobian));

	data->lambdas.clear();
}

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

sunindextype addVariableOffset(void* userData, sunindextype offset)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	data->variableOffsets.push_back(offset);
	return data->variableOffsets.size() - 1;
}

void addVariableDimension(void* userData, sunindextype index, sunindextype dim)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	if ((size_t) index == data->variableDimensions.size())
		data->variableDimensions.push_back({ dim });
	else
		data->variableDimensions[index].push_back(dim);
}

sunindextype addNewVariableAccess(
		void* userData, sunindextype var, sunindextype off, sunindextype ind)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	data->variableAccesses.push_back({ var, { { off, ind } } });
	return data->variableAccesses.size() - 1;
}

void addVariableAccess(
		void* userData, sunindextype index, sunindextype off, sunindextype ind)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	assert((size_t) index == data->variableAccesses.size() - 1);
	data->variableAccesses[index].second.push_back({ off, ind });
}

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

realtype getIdaTime(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	// Return the stop time if the whole system is trivial.
	if (data->equationsNumber == 0)
		return data->stopTime;

	return data->time;
}

realtype getIdaVariable(void* userData, sunindextype index)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	assert(index < data->equationsNumber);
	return data->variablesValues[index];
}

realtype getIdaDerivative(void* userData, sunindextype index)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	assert(index < data->equationsNumber);
	return data->derivativesValues[index];
}

sunindextype getIdaRowLength(void* userData, sunindextype index)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	assert(index < data->equationsNumber);
	return data->rowLengths[index];
}

Dimension getIdaDimension(void* userData, sunindextype index)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	assert(index < data->equationsNumber);
	return data->equationDimensions[index];
}

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

sunindextype numSteps(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	sunindextype nst;
	IDAGetNumSteps(data->idaMemory, &nst);
	return nst;
}

sunindextype numResEvals(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	sunindextype nre;
	IDAGetNumResEvals(data->idaMemory, &nre);
	return nre;
}

sunindextype numJacEvals(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	sunindextype nje;
	IDAGetNumJacEvals(data->idaMemory, &nje);
	return nje;
}

sunindextype numNonlinIters(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	sunindextype nni;
	IDAGetNumNonlinSolvIters(data->idaMemory, &nni);
	return nni;
}

//===----------------------------------------------------------------------===//
// Lambda constructions
//===----------------------------------------------------------------------===//

sunindextype lambdaConstant(void* userData, realtype constant)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function first = [constant](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype { return constant; };

	Function second = [](realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype { return 0.0; };

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaTime(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function first = [](realtype tt,
											realtype cj,
											realtype* yy,
											realtype* yp,
											Indexes& ind,
											realtype var) -> realtype { return tt; };

	Function second = [](realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype { return 0.0; };

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaInduction(void* userData, sunindextype induction)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function first = [induction](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype { return ind[induction]; };

	Function second = [](realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype { return 0.0; };

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaVariable(void* userData, sunindextype accessIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	sunindextype variableIndex = data->variableAccesses[accessIndex].first;
	sunindextype offset = data->variableOffsets[variableIndex];
	std::vector<sunindextype> dim = data->variableDimensions[variableIndex];
	Access access = data->variableAccesses[accessIndex].second;

	Function first = [offset, dim, access](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		sunindextype varOffset = 0;

		for (size_t i = 0; i < access.size(); i++)
		{
			auto acc = access[i];
			sunindextype accOffset =
					acc.first + (acc.second != -1 ? ind[acc.second] : 0);
			varOffset += accOffset * dim[i];
		}

		return yy[offset + varOffset];
	};

	Function second = [offset, dim, access](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		sunindextype varOffset = 0;

		for (size_t i = 0; i < access.size(); i++)
		{
			auto acc = access[i];
			sunindextype accOffset =
					acc.first + (acc.second != -1 ? ind[acc.second] : 0);
			varOffset += accOffset * dim[i];
		}

		if (offset + varOffset == var)
			return 1.0;
		return 0.0;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaDerivative(void* userData, sunindextype accessIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	sunindextype variableIndex = data->variableAccesses[accessIndex].first;
	sunindextype offset = data->variableOffsets[variableIndex];
	std::vector<sunindextype> dim = data->variableDimensions[variableIndex];
	Access access = data->variableAccesses[accessIndex].second;

	Function first = [offset, dim, access](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		sunindextype varOffset = 0;

		for (size_t i = 0; i < access.size(); i++)
		{
			auto acc = access[i];
			sunindextype accOffset =
					acc.first + (acc.second != -1 ? ind[acc.second] : 0);
			varOffset += accOffset * dim[i];
		}

		return yp[offset + varOffset];
	};

	Function second = [offset, dim, access](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		sunindextype varOffset = 0;

		for (size_t i = 0; i < access.size(); i++)
		{
			auto acc = access[i];
			sunindextype accOffset =
					acc.first + (acc.second != -1 ? ind[acc.second] : 0);
			varOffset += accOffset * dim[i];
		}

		if (offset + varOffset == var)
			return cj;
		return 0.0;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaNegate(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return -operand(tt, cj, yy, yp, ind, var);
	};

	Function second = [derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return -derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaAdd(
		void* userData, sunindextype leftIndex, sunindextype rightIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;
	Function derLeft = data->lambdas[leftIndex].second;
	Function derRight = data->lambdas[rightIndex].second;

	Function first = [left, right](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return left(tt, cj, yy, yp, ind, var) + right(tt, cj, yy, yp, ind, var);
	};

	Function second = [derLeft, derRight](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return derLeft(tt, cj, yy, yp, ind, var) +
					 derRight(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaSub(
		void* userData, sunindextype leftIndex, sunindextype rightIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;
	Function derLeft = data->lambdas[leftIndex].second;
	Function derRight = data->lambdas[rightIndex].second;

	Function first = [left, right](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return left(tt, cj, yy, yp, ind, var) - right(tt, cj, yy, yp, ind, var);
	};

	Function second = [derLeft, derRight](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return derLeft(tt, cj, yy, yp, ind, var) -
					 derRight(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaMul(
		void* userData, sunindextype leftIndex, sunindextype rightIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;
	Function derLeft = data->lambdas[leftIndex].second;
	Function derRight = data->lambdas[rightIndex].second;

	Function first = [left, right](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return left(tt, cj, yy, yp, ind, var) * right(tt, cj, yy, yp, ind, var);
	};

	Function second = [left, right, derLeft, derRight](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return left(tt, cj, yy, yp, ind, var) * derRight(tt, cj, yy, yp, ind, var) +
					 right(tt, cj, yy, yp, ind, var) * derLeft(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaDiv(
		void* userData, sunindextype leftIndex, sunindextype rightIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function left = data->lambdas[leftIndex].first;
	Function right = data->lambdas[rightIndex].first;
	Function derLeft = data->lambdas[leftIndex].second;
	Function derRight = data->lambdas[rightIndex].second;

	Function first = [left, right](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return left(tt, cj, yy, yp, ind, var) / right(tt, cj, yy, yp, ind, var);
	};

	Function second = [left, right, derLeft, derRight](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		realtype rightValue = right(tt, cj, yy, yp, ind, var);
		realtype dividend =
				rightValue * derLeft(tt, cj, yy, yp, ind, var) -
				left(tt, cj, yy, yp, ind, var) * derRight(tt, cj, yy, yp, ind, var);
		return dividend / (rightValue * rightValue);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaPow(
		void* userData, sunindextype baseIndex, sunindextype exponentIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function base = data->lambdas[baseIndex].first;
	Function exponent = data->lambdas[exponentIndex].first;
	Function derBase = data->lambdas[baseIndex].second;
	Function derExponent = data->lambdas[exponentIndex].second;

	Function first = [base, exponent](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::pow(
				base(tt, cj, yy, yp, ind, var), exponent(tt, cj, yy, yp, ind, var));
	};

	Function second = [base, exponent, derBase, derExponent](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		realtype baseValue = base(tt, cj, yy, yp, ind, var);
		realtype exponentValue = exponent(tt, cj, yy, yp, ind, var);
		return std::pow(baseValue, exponentValue) *
					 (derExponent(tt, cj, yy, yp, ind, var) * std::log(baseValue) +
						exponentValue * derBase(tt, cj, yy, yp, ind, var) / baseValue);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaAtan2(
		void* userData, sunindextype yIndex, sunindextype xIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function y = data->lambdas[yIndex].first;
	Function x = data->lambdas[xIndex].first;
	Function derY = data->lambdas[yIndex].second;
	Function derX = data->lambdas[xIndex].second;

	Function first = [y, x](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::atan2(y(tt, cj, yy, yp, ind, var), x(tt, cj, yy, yp, ind, var));
	};

	Function second = [y, x, derY, derX](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		realtype yValue = y(tt, cj, yy, yp, ind, var);
		realtype xValue = x(tt, cj, yy, yp, ind, var);
		return (derY(tt, cj, yy, yp, ind, var) * xValue -
						yValue * derX(tt, cj, yy, yp, ind, var)) /
					 (yValue * yValue + xValue * xValue);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaAbs(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::abs(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		realtype x = operand(tt, cj, yy, yp, ind, var);

		return (x > 0.0) - (x < 0.0);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaSign(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		realtype x = operand(tt, cj, yy, yp, ind, var);

		return (x > 0.0) - (x < 0.0);
	};

	Function second = [](realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype { return 0.0; };

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaSqrt(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::sqrt(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return derOperand(tt, cj, yy, yp, ind, var) /
					 std::sqrt(operand(tt, cj, yy, yp, ind, var)) / 2;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaExp(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::exp(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return std::exp(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaLog(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::log(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return derOperand(tt, cj, yy, yp, ind, var) /
					 operand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaLog10(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::log10(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return derOperand(tt, cj, yy, yp, ind, var) /
					 (operand(tt, cj, yy, yp, ind, var) * std::log(10));
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaSin(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::sin(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return std::cos(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaCos(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::cos(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return -std::sin(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaTan(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::tan(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		realtype cosOperandValue = std::cos(operand(tt, cj, yy, yp, ind, var));
		return derOperand(tt, cj, yy, yp, ind, var) /
					 (cosOperandValue * cosOperandValue);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaAsin(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::asin(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		realtype operandValue = operand(tt, cj, yy, yp, ind, var);
		return derOperand(tt, cj, yy, yp, ind, var) /
					 (std::sqrt(1 - operandValue * operandValue));
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaAcos(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::acos(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		realtype operandValue = operand(tt, cj, yy, yp, ind, var);
		return -derOperand(tt, cj, yy, yp, ind, var) /
					 (std::sqrt(1 - operandValue * operandValue));
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaAtan(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::atan(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		realtype operandValue = operand(tt, cj, yy, yp, ind, var);
		return derOperand(tt, cj, yy, yp, ind, var) /
					 (1 + operandValue * operandValue);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaSinh(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::sinh(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return std::cosh(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaCosh(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::cosh(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return std::sinh(operand(tt, cj, yy, yp, ind, var)) *
					 derOperand(tt, cj, yy, yp, ind, var);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaTanh(void* userData, sunindextype operandIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return std::tanh(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		realtype coshOperandValue = std::cosh(operand(tt, cj, yy, yp, ind, var));
		return derOperand(tt, cj, yy, yp, ind, var) /
					 (coshOperandValue * coshOperandValue);
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

sunindextype lambdaCall(
		void* userData, sunindextype operandIndex, realtype (*function)(realtype))
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;

	Function first = [function, operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return ((realtype(*)(realtype)) function)(
				operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [](realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype { return 0.0; };

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}
