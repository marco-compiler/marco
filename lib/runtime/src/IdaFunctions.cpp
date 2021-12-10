#include <algorithm>
#include <ida/ida.h>
#include <marco/runtime/IdaFunctions.h>
#include <nvector/nvector_serial.h>
#include <set>
#include <stdlib.h>
#include <sundials/sundials_config.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

#define PRINT_JACOBIAN false

#define INIT_TIME_STEP 1e-6
#define ALGEBRAIC_TOLERANCE 1e-12

#define MAX_NUM_STEPS 1000
#define MAX_ERR_TEST_FAIL 10
#define MAX_NONLIN_ITERS 4
#define MAX_CONV_FAILS 10
#define NONLIN_CONV_COEF 0.33

#define MAX_NUM_STEPS_IC 5
#define MAX_NUM_JACS_IC 4
#define MAX_NUM_ITERS_IC 10
#define NONLIN_CONV_COEF_IC 0.0033

#define SUPPRESS_ALG SUNFALSE
#define LINE_SEARCH_OFF SUNFALSE

#define exitOnError(success) if (!success) return false;

using Indexes = std::vector<size_t>;
using Access = std::vector<std::pair<sunindextype, sunindextype>>;

using VarDimension = std::vector<size_t>;
using EqDimension = std::vector<std::pair<size_t, size_t>>;

using ResidualFunction = std::function<realtype(
		realtype tt,
		realtype* yy,
		realtype* yp,
		sunindextype* ind)>;

using JacobianFunction = std::function<realtype(
		realtype tt,
		realtype* yy,
		realtype* yp,
		sunindextype* ind,
		realtype cj,
		sunindextype var)>;

/**
 * Container for all the data required by IDA in order to compute the residual
 * functions and the jacobian matrix.
 */
typedef struct IdaUserData
{
	// Equations data
	std::vector<std::vector<size_t>> accessIndexes;
	std::vector<EqDimension> equationDimensions;
	std::vector<ResidualFunction> residuals;
	std::vector<JacobianFunction> jacobians;

	// Variables data
	std::vector<sunindextype> variableOffsets;
	std::vector<VarDimension> variableDimensions;
	std::vector<std::pair<sunindextype, Access>> variableAccesses;

	// Matrix size
	size_t vectorVariablesNumber;
	size_t vectorEquationsNumber;
	sunindextype scalarEquationsNumber;
	sunindextype nonZeroValuesNumber;

	// Jacobian indexes
	std::vector<size_t> equationIndexes;
	std::vector<std::vector<size_t>> nnzElements;
	std::vector<std::vector<std::vector<size_t>>> columnIndexes;

	// Simulation times
	realtype startTime;
	realtype endTime;
	realtype timeStep;
	realtype time;
	realtype nextStop;

	// Simulation options
	bool equidistantTimeGrid;
	sunindextype threads;

	// Error tolerances
	realtype relativeTolerance;
	realtype absoluteTolerance;

	// Variables vectors and values
	N_Vector variablesVector;
	N_Vector derivativesVector;
	N_Vector idVector;
	N_Vector tolerancesVector;
	realtype* variableValues;
	realtype* derivativeValues;
	realtype* idValues;
	realtype* toleranceValues;

	// IDA classes
	void* idaMemory;
	SUNMatrix sparseMatrix;
	SUNLinearSolver linearSolver;
} IdaUserData;

/**
 * Given an array of indexes and the dimension of an equation, increase the
 * indexes within the induction bounds of the equation. Return false if the
 * indexes exceed the equation bounds, which means the computation has finished,
 * true otherwise.
 */
static bool updateIndexes(sunindextype* indexes, const EqDimension& dimension)
{
	for (sunindextype dim = dimension.size() - 1; dim >= 0; dim--)
	{
		indexes[dim]++;

		if ((size_t) indexes[dim] == dimension[dim].second)
			indexes[dim] = dimension[dim].first;
		else
			return true;
	}

	return false;
}

/**
 * Given an array of indexes, the dimension of a variable and the type of
 * access, return the index needed to access the flattened multidimensional
 * variable.
 */
static sunindextype computeOffset(
		const sunindextype* indexes,
		const VarDimension& dimensions,
		const Access& accesses)
{
	assert(accesses.size() == dimensions.size());

	sunindextype offset = 0;

	for (size_t i = 0; i < accesses.size(); ++i)
	{
		sunindextype accessOffset =
				accesses[i].first +
				(accesses[i].second != -1 ? indexes[accesses[i].second] : 0);
		offset = offset * dimensions[i] + accessOffset;
	}

	return offset;
}

/**
 * Check if SUNDIALS function returned NULL pointer (no memory allocated).
 */
static bool checkAllocation(void* retval, const char* funcname)
{
	if (retval == NULL)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname;
		llvm::errs() << "() failed - returned NULL pointer\n";
		return false;
	}

	return true;
}

/**
 * Check if SUNDIALS function returned a success value (positive integer).
 */
static bool checkRetval(int retval, const char* funcname)
{
	if (retval < 0)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname;
		llvm::errs() << "() failed  with return value = " << retval << "\n";
		return false;
	}

	return true;
}

/**
 * IDAResFn user-defined residual function, passed to IDA through IDAInit.
 * It contains how to compute the Residual Function of the system, starting
 * from the provided UserData struct, iterating through every equation.
 */
int residualFunction(
		realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void* userData)
{
	realtype* yval = N_VGetArrayPointer(yy);
	realtype* ypval = N_VGetArrayPointer(yp);
	realtype* rval = N_VGetArrayPointer(rr);

	IdaUserData* data = static_cast<IdaUserData*>(userData);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(data->threads)
#endif
	for (size_t eq = 0; eq < data->vectorEquationsNumber; eq++)
	{
		// For every vector equation
		size_t residualIndex = data->equationIndexes[eq];

		// Initialize the multidimensional interval of the vector equation
		sunindextype* indexes = new sunindextype[data->equationDimensions[eq].size()];

		for (size_t i = 0; i < data->equationDimensions[eq].size(); i++)
			indexes[i] = data->equationDimensions[eq][i].first;

		// For every scalar equation in the vector equation
		do
		{
			// Compute the i-th residual function
			rval[residualIndex++] =
					data->residuals[eq](tt, yval, ypval, indexes);
		} while (updateIndexes(indexes, data->equationDimensions[eq]));

		assert(residualIndex == data->equationIndexes[eq+1]);
		delete[] indexes;
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
	realtype* yval = N_VGetArrayPointer(yy);
	realtype* ypval = N_VGetArrayPointer(yp);

	sunindextype* rowptrs = SUNSparseMatrix_IndexPointers(JJ);
	sunindextype* colvals = SUNSparseMatrix_IndexValues(JJ);

	realtype* jacobian = SUNSparseMatrix_Data(JJ);

	IdaUserData* data = static_cast<IdaUserData*>(userData);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(data->threads)
#endif
	for (size_t eq = 0; eq < data->vectorEquationsNumber; eq++)
	{
		// For every vector equation
		size_t rowIndex = data->equationIndexes[eq];
		size_t columnIndex = 0;

		// Initialize the multidimensional interval of the vector equation
		sunindextype* indexes = new sunindextype[data->equationDimensions[eq].size()];

		for (size_t i = 0; i < data->equationDimensions[eq].size(); i++)
			indexes[i] = data->equationDimensions[eq][i].first;

		// For every scalar equation in the vector equation
		do
		{
			size_t jacobianIndex = data->nnzElements[eq][columnIndex];
			rowptrs[rowIndex++] = data->nnzElements[eq][columnIndex];

			// For every variable with respect to which every equation must be
			// partially differentiated
			for (sunindextype var : data->columnIndexes[eq][columnIndex])
			{
				// Compute the i-th jacobian value
				jacobian[jacobianIndex] =
						data->jacobians[eq](tt, yval, ypval, indexes, cj, var);
				colvals[jacobianIndex++] = var;
			}

			// Update multidimensional interval, exit while loop if finished
			columnIndex++;
		} while (updateIndexes(indexes, data->equationDimensions[eq]));

		assert(rowIndex == data->equationIndexes[eq+1]);
		delete[] indexes;
	}

	rowptrs[data->scalarEquationsNumber] = data->nonZeroValuesNumber;

	return 0;
}

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

/**
 * Instantiate and initialize the struct of data needed by IDA, given the total
 * number of scalar equations.
 */
template<typename T>
inline void* idaAllocData(T scalarEquationsNumber)
{
	IdaUserData* data = new IdaUserData;

	data->scalarEquationsNumber = scalarEquationsNumber;
	data->vectorEquationsNumber = 0;
	data->vectorVariablesNumber = 0;

	// Create and initialize the required N-vectors for the variables.
	data->variablesVector = N_VNew_Serial(data->scalarEquationsNumber);
	assert(checkAllocation((void*) data->variablesVector, "N_VNew_Serial"));

	data->derivativesVector = N_VNew_Serial(data->scalarEquationsNumber);
	assert(checkAllocation((void*) data->derivativesVector, "N_VNew_Serial"));

	data->idVector = N_VNew_Serial(data->scalarEquationsNumber);
	assert(checkAllocation((void*) data->idVector, "N_VNew_Serial"));

	data->tolerancesVector = N_VNew_Serial(data->scalarEquationsNumber);
	assert(checkAllocation((void*) data->tolerancesVector, "N_VNew_Serial"));

	data->variableValues = N_VGetArrayPointer(data->variablesVector);
	data->derivativeValues = N_VGetArrayPointer(data->derivativesVector);
	data->idValues = N_VGetArrayPointer(data->idVector);
	data->toleranceValues = N_VGetArrayPointer(data->tolerancesVector);

	return static_cast<void*>(data);
}

RUNTIME_FUNC_DEF(idaAllocData, PTR(void), int32_t)
RUNTIME_FUNC_DEF(idaAllocData, PTR(void), int64_t)

/**
 * Precompute the row and column indexes of all non-zero values in the Jacobian
 * Matrix. This avoids the recomputation of such indexes and allows
 * parallelization of the Jacobian computation. Returns the number of non-zero
 * values in the Jacobian Matrix.
 */
sunindextype precomputeJacobianIndexes(IdaUserData* data)
{
	sunindextype nnzElements = 0;

	data->equationIndexes.resize(data->vectorEquationsNumber + 1);
	data->columnIndexes.resize(data->vectorEquationsNumber);
	data->nnzElements.resize(data->vectorEquationsNumber);

	data->equationIndexes[0] = 0;

	for (size_t eq = 0; eq < data->vectorEquationsNumber; eq++)
	{
		sunindextype equationSize = 1;

		// Initialize the multidimensional interval of the vector equation
		sunindextype* indexes = new sunindextype[data->equationDimensions[eq].size()];

		for (size_t i = 0; i < data->equationDimensions[eq].size(); i++)
		{
			auto dim = data->equationDimensions[eq][i];
			indexes[i] = dim.first;
			equationSize *= (dim.second - dim.first);
		}

		// Compute the number of scalar equations in every vector equation
		data->equationIndexes[eq + 1] = equationSize + data->equationIndexes[eq];

		// For every scalar equation in the vector equation
		do
		{
			// Compute the column indexes that may be non-zeros
			std::set<size_t> columnIndexesSet;
			for (sunindextype accessIndex : data->accessIndexes[eq])
			{
				sunindextype varIndex = data->variableAccesses[accessIndex].first;
				VarDimension dimensions = data->variableDimensions[varIndex];
				Access access = data->variableAccesses[accessIndex].second;

				sunindextype varOffset = computeOffset(indexes, dimensions, access);
				columnIndexesSet.insert(data->variableOffsets[varIndex] + varOffset);
			}

			// Compute the number of non-zero values in each scalar equation
			data->columnIndexes[eq].push_back(
					std::vector(columnIndexesSet.begin(), columnIndexesSet.end()));
			data->nnzElements[eq].push_back(nnzElements);
			nnzElements += data->columnIndexes[eq].back().size();

			// Update multidimensional interval, exit while loop if finished
		} while (updateIndexes(indexes, data->equationDimensions[eq]));

		delete[] indexes;
	}

	return nnzElements;
}

/**
 * Instantiate and initialize all the classes needed by IDA in order to solve
 * the given system of equations. It also sets optional simulation parameters
 * for IDA. It must be called before the first usage of idaStep() and after a
 * call to idaAllocData(). It may fail in case of malformed model.
 */
template<typename T>
inline bool idaInit(void* userData, T threads)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->scalarEquationsNumber == 0)
		return true;

	// Compute the total amount of non-zero values in the Jacobian Matrix.
	data->nonZeroValuesNumber = precomputeJacobianIndexes(data);
	data->threads = threads == 0 ? omp_get_max_threads() : threads;

	// Check that the data was correctly initialized.	
	assert(data->vectorEquationsNumber == data->equationDimensions.size());
	assert(data->vectorEquationsNumber == data->residuals.size());
	assert(data->vectorEquationsNumber == data->jacobians.size());
	assert(data->vectorEquationsNumber == data->nnzElements.size());
	assert(data->vectorEquationsNumber == data->columnIndexes.size());
	assert(data->vectorVariablesNumber == data->variableOffsets.size());
	assert(data->vectorVariablesNumber == data->variableDimensions.size());

	// Create and initialize IDA memory.
	data->idaMemory = IDACreate();
	exitOnError(checkAllocation((void*) data->idaMemory, "IDACreate"));

	int retval = IDAInit(
			data->idaMemory,
			residualFunction,
			data->startTime,
			data->variablesVector,
			data->derivativesVector);
	exitOnError(checkRetval(retval, "IDAInit"));

	// Call IDASVtolerances to set tolerances.
	retval = IDASVtolerances(data->idaMemory, data->relativeTolerance, data->tolerancesVector);
	exitOnError(checkRetval(retval, "IDASVtolerances"));

	// Create sparse SUNMatrix for use in linear solver.
	data->sparseMatrix = SUNSparseMatrix(
			data->scalarEquationsNumber,
			data->scalarEquationsNumber,
			data->nonZeroValuesNumber,
			CSR_MAT);
	exitOnError(checkAllocation((void*) data->sparseMatrix, "SUNSparseMatrix"));

	// Create and attach a KLU SUNLinearSolver object.
	data->linearSolver = SUNLinSol_KLU(data->variablesVector, data->sparseMatrix);
	exitOnError(checkAllocation((void*) data->linearSolver, "SUNLinSol_KLU"));

	retval = IDASetLinearSolver(data->idaMemory, data->linearSolver, data->sparseMatrix);
	exitOnError(checkRetval(retval, "IDASetLinearSolver"));

	// Set the user-supplied Jacobian routine.
	retval = IDASetJacFn(data->idaMemory, jacobianMatrix);
	exitOnError(checkRetval(retval, "IDASetJacFn"));

	// Add the remaining mandatory paramters.
	retval = IDASetUserData(data->idaMemory, (void*) data);
	exitOnError(checkRetval(retval, "IDASetUserData"));

	retval = IDASetId(data->idaMemory, data->idVector);
	exitOnError(checkRetval(retval, "IDASetId"));

	retval = IDASetStopTime(data->idaMemory, data->endTime);
	exitOnError(checkRetval(retval, "IDASetStopTime"));

	// Add the remaining optional paramters.
	retval = IDASetInitStep(data->idaMemory, 0.0);
	exitOnError(checkRetval(retval, "IDASetInitStep"));

	retval = IDASetMaxStep(data->idaMemory, data->endTime);
	exitOnError(checkRetval(retval, "IDASetMaxStep"));

	retval = IDASetSuppressAlg(data->idaMemory, SUPPRESS_ALG);
	exitOnError(checkRetval(retval, "IDASetSuppressAlg"));

	// Increase the maximum number of iterations taken by IDA before failing.
	retval = IDASetMaxNumSteps(data->idaMemory, MAX_NUM_STEPS);
	exitOnError(checkRetval(retval, "IDASetMaxNumSteps"));

	retval = IDASetMaxErrTestFails(data->idaMemory, MAX_ERR_TEST_FAIL);
	exitOnError(checkRetval(retval, "IDASetMaxErrTestFails"));

	retval = IDASetMaxNonlinIters(data->idaMemory, MAX_NONLIN_ITERS);
	exitOnError(checkRetval(retval, "IDASetMaxNonlinIters"));

	retval = IDASetMaxConvFails(data->idaMemory, MAX_CONV_FAILS);
	exitOnError(checkRetval(retval, "IDASetMaxConvFails"));

	retval = IDASetNonlinConvCoef(data->idaMemory, NONLIN_CONV_COEF);
	exitOnError(checkRetval(retval, "IDASetNonlinConvCoef"));

	// Increase the maximum number of iterations taken by IDA IC before failing.
	retval = IDASetMaxNumStepsIC(data->idaMemory, MAX_NUM_STEPS_IC);
	exitOnError(checkRetval(retval, "IDASetMaxNumStepsIC"));

	retval = IDASetMaxNumJacsIC(data->idaMemory, MAX_NUM_JACS_IC);
	exitOnError(checkRetval(retval, "IDASetMaxNumJacsIC"));

	retval = IDASetMaxNumItersIC(data->idaMemory, MAX_NUM_ITERS_IC);
	exitOnError(checkRetval(retval, "IDASetMaxNumItersIC"));

	retval = IDASetNonlinConvCoefIC(data->idaMemory, NONLIN_CONV_COEF_IC);
	exitOnError(checkRetval(retval, "IDASetNonlinConvCoefIC"));

	retval = IDASetLineSearchOffIC(data->idaMemory, LINE_SEARCH_OFF);
	exitOnError(checkRetval(retval, "IDASetLineSearchOffIC"));

	// Call IDACalcIC to correct the initial values.
	retval = IDACalcIC(data->idaMemory, IDA_YA_YDP_INIT, INIT_TIME_STEP);
	exitOnError(checkRetval(retval, "IDACalcIC"));

	return true;
}

RUNTIME_FUNC_DEF(idaInit, bool, PTR(void), int32_t)
RUNTIME_FUNC_DEF(idaInit, bool, PTR(void), int64_t)

/**
 * Invoke IDA to perform one step of the computation. Returns false if the
 * computation failed, true otherwise.
 */
inline bool idaStep(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->scalarEquationsNumber == 0)
		return true;

	// Execute one step
	int retval = IDASolve(
			data->idaMemory,
			data->nextStop,
			&data->time,
			data->variablesVector,
			data->derivativesVector,
			data->equidistantTimeGrid ? IDA_NORMAL : IDA_ONE_STEP);

	if (data->equidistantTimeGrid)
		data->nextStop += data->timeStep;

	// Check if the solver failed
	exitOnError(checkRetval(retval, "IDASolve"));

	return true;
}

RUNTIME_FUNC_DEF(idaStep, bool, PTR(void))

/**
 * Free all the data allocated by IDA.
 */
inline bool idaFreeData(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->scalarEquationsNumber == 0)
		return true;

	// Free IDA memory
	IDAFree(&data->idaMemory);

	int retval = SUNLinSolFree(data->linearSolver);
	exitOnError(checkRetval(retval, "SUNLinSolFree"));

	SUNMatDestroy(data->sparseMatrix);
	N_VDestroy(data->variablesVector);
	N_VDestroy(data->derivativesVector);
	N_VDestroy(data->idVector);
	N_VDestroy(data->tolerancesVector);

	delete data;

	return true;
}

RUNTIME_FUNC_DEF(idaFreeData, bool, PTR(void))

/**
 * Add the start time, the end time and the step time to the user data. If a
 * positive time step is given, the output will show the variables in an
 * equidistant time grid based on the step time paramter. Otherwise, the output
 * will show the variables at every step of the computation.
 */
template<typename T>
inline void addTime(void* userData, T startTime, T endTime, T timeStep)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	data->startTime = startTime;
	data->endTime = endTime;
	data->timeStep = timeStep;
	data->time = startTime;
	data->equidistantTimeGrid = timeStep != -1;
	data->nextStop = data->equidistantTimeGrid ? timeStep : endTime;
}

RUNTIME_FUNC_DEF(addTime, void, PTR(void), float, float, float)
RUNTIME_FUNC_DEF(addTime, void, PTR(void), double, double, double)

/**
 * Add the relative tolerance and the absolute tolerance to the user data.
 */
template<typename T>
inline void addTolerance(void* userData, T relTol, T absTol)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	data->relativeTolerance = relTol;
	data->absoluteTolerance = absTol;
}

RUNTIME_FUNC_DEF(addTolerance, void, PTR(void), float, float)
RUNTIME_FUNC_DEF(addTolerance, void, PTR(void), double, double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

/**
 * Add the access index of a non-zero value contained in the rowIndex-th row of
 * the jacobian matrix to the user data.
 */
template<typename T>
inline void addColumnIndex(void* userData, T rowIndex, T accessIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	assert(rowIndex >= 0);
	assert((size_t) rowIndex < data->vectorEquationsNumber);

	if ((size_t) rowIndex == data->accessIndexes.size())
		data->accessIndexes.push_back({ (size_t) accessIndex });
	else
		data->accessIndexes[rowIndex].push_back(accessIndex);
}

RUNTIME_FUNC_DEF(addColumnIndex, void, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DEF(addColumnIndex, void, PTR(void), int64_t, int64_t)

/**
 * Add the dimension of the index-th equation to the user data.
 */
template<typename T>
inline void addEqDimension(
		void* userData,
		UnsizedArrayDescriptor<T> start,
		UnsizedArrayDescriptor<T> end)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	assert(start.getRank() == 1 && end.getRank() == 1);
	assert(start.getDimensionSize(0) == end.getDimensionSize(0));
	assert(start.getNumElements() == end.getNumElements());

	data->vectorEquationsNumber++;
	data->equationDimensions.push_back({});
	for (size_t i = 0; i < start.getNumElements(); i++)
		data->equationDimensions.back().push_back({ start[i], end[i] });
}

RUNTIME_FUNC_DEF(addEqDimension, void, PTR(void), ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(addEqDimension, void, PTR(void), ARRAY(int64_t), ARRAY(int64_t))

/**
 * Add the function pointer that computes the index-th residual function to the
 * user data.
 */
template<typename T>
inline void addResidual(void* userData, T residualFunction)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	data->residuals.push_back(residualFunction);
}

#if defined(SUNDIALS_SINGLE_PRECISION)
RUNTIME_FUNC_DEF(addResidual, void, PTR(void), RESIDUAL(float))
#elif defined(SUNDIALS_DOUBLE_PRECISION)
RUNTIME_FUNC_DEF(addResidual, void, PTR(void), RESIDUAL(double))
#endif

/**
 * Add the function pointer that computes the index-th jacobian row to the user
 * data.
 */
template<typename T>
inline void addJacobian(void* userData, T jacobianFunction)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	data->jacobians.push_back(jacobianFunction);
}

#if defined(SUNDIALS_SINGLE_PRECISION)
RUNTIME_FUNC_DEF(addJacobian, void, PTR(void), JACOBIAN(float))
#elif defined(SUNDIALS_DOUBLE_PRECISION)
RUNTIME_FUNC_DEF(addJacobian, void, PTR(void), JACOBIAN(double))
#endif

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

/**
 * Add and initialize a new variable given its array.
 * 
 * @param userData opaque pointer to the IDA user data.
 * @param offset offset of the current variable from the beginning of the array.
 * @param array allocation operation containing the rank and dimensions.
 * @param isState indicates if the variable is differential or algebraic.
 */
template<typename T, typename U>
inline void addVariable(
		void* userData,
		T offset,
		UnsizedArrayDescriptor<U> array,
		bool isState)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	assert(offset >= 0);
	assert(offset + array.getNumElements() <= (size_t) data->scalarEquationsNumber);

	// Add variable offset.
	data->vectorVariablesNumber++;
	data->variableOffsets.push_back(offset);

	// Add variable dimensions.
	VarDimension dimensions(array.getDimensions());
	data->variableDimensions.push_back(dimensions);

	// Compute idValue and absoluteTolerance.
	realtype idValue = isState ? 1.0 : 0.0;
	realtype absTol = isState 
			? data->absoluteTolerance
			: std::min(ALGEBRAIC_TOLERANCE, data->absoluteTolerance);

	// Initialize derivativeValues, idValues and absoluteTolerances.
	std::fill_n(&data->derivativeValues[offset], array.getNumElements(), 0.0);
	std::fill_n(&data->idValues[offset], array.getNumElements(), idValue);
	std::fill_n(&data->toleranceValues[offset], array.getNumElements(), absTol);
}

RUNTIME_FUNC_DEF(addVariable, void, PTR(void), int32_t, ARRAY(float), bool)
RUNTIME_FUNC_DEF(addVariable, void, PTR(void), int64_t, ARRAY(double), bool)

/**
 * Return the pointer to the start of the memory of the requested variable given
 * its offset and if it is a derivative or not.
 */
template<typename T>
inline void* getVariableAlloc(void* userData, T offset, bool isDerivative)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	assert(offset >= 0);
	assert(offset < data->scalarEquationsNumber);

	if (isDerivative)
		return (void*) &data->derivativeValues[offset];
	return (void*) &data->variableValues[offset];
}

RUNTIME_FUNC_DEF(getVariableAlloc, PTR(void), PTR(void), int32_t, bool)
RUNTIME_FUNC_DEF(getVariableAlloc, PTR(void), PTR(void), int64_t, bool)

/**
 * Add a variable access to the var-th variable, where ind is the induction
 * variable and off is the access offset.
 */
template<typename T>
inline int64_t addVarAccess(
		void* userData,
		T var,
		UnsizedArrayDescriptor<T> off,
		UnsizedArrayDescriptor<T> ind)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	assert(var >= 0);
	assert(off.getRank() == 1 && ind.getRank() == 1);
	assert(off.getDimensionSize(0) == ind.getDimensionSize(0));
	assert(off.getNumElements() == ind.getNumElements());

	data->variableAccesses.push_back({ var, {} });
	for (size_t i = 0; i < off.getNumElements(); i++)
		data->variableAccesses.back().second.push_back({ off[i], ind[i] });

	return data->variableAccesses.size() - 1;
}

RUNTIME_FUNC_DEF(addVarAccess, int32_t, PTR(void), int32_t, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(addVarAccess, int64_t, PTR(void), int64_t, ARRAY(int64_t), ARRAY(int64_t))

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

/**
 * Returns the time reached by the solver after the last step. 
 */
inline double getIdaTime(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	// Return the stop time if the whole system is trivial.
	if (data->scalarEquationsNumber == 0)
		return data->endTime;

	return data->time;
}

RUNTIME_FUNC_DEF(getIdaTime, float, PTR(void))
RUNTIME_FUNC_DEF(getIdaTime, double, PTR(void))

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

/**
 * Prints the Jacobian incidence matrix of the system.
 */
static void printIncidenceMatrix(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	llvm::errs() << "\n";

	// For every vector equation
	for (size_t eq = 0; eq < data->vectorEquationsNumber; eq++)
	{
		// Initialize the multidimensional interval of the vector equation
		sunindextype* indexes = new sunindextype[data->equationDimensions[eq].size()];

		for (size_t i = 0; i < data->equationDimensions[eq].size(); i++)
			indexes[i] = data->equationDimensions[eq][i].first;

		// For every scalar equation in the vector equation
		do
		{
			llvm::errs() << "│";

			// Compute the column indexes that may be non-zeros.
			std::set<sunindextype> columnIndexesSet;
			for (size_t accessIndex : data->accessIndexes[eq])
			{
				sunindextype varIndex = data->variableAccesses[accessIndex].first;
				VarDimension dimensions = data->variableDimensions[varIndex];
				Access access = data->variableAccesses[accessIndex].second;
				sunindextype varOffset = computeOffset(indexes, dimensions, access);
				columnIndexesSet.insert(data->variableOffsets[varIndex] + varOffset);
			}

			for (sunindextype i = 0; i < data->scalarEquationsNumber; i++)
			{
				if (columnIndexesSet.find(i) != columnIndexesSet.end())
					llvm::errs() << "*";
				else
					llvm::errs() << " ";

				if (i < data->scalarEquationsNumber - 1)
					llvm::errs() << " ";
			}

			llvm::errs() << "│\n";
		} while (updateIndexes(indexes, data->equationDimensions[eq]));

		delete[] indexes;
	}
}

/**
 * Prints statistics about the computation of the system.
 */
inline void printStatistics(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->scalarEquationsNumber == 0)
		return;

	if (PRINT_JACOBIAN)
		printIncidenceMatrix(userData);

	int64_t nst, nre, nje, nni, nli, netf, nncf;
	realtype is, ls;

	IDAGetNumSteps(data->idaMemory, &nst);
	IDAGetNumResEvals(data->idaMemory, &nre);
	IDAGetNumJacEvals(data->idaMemory, &nje);
	IDAGetNumNonlinSolvIters(data->idaMemory, &nni);
	IDAGetNumLinIters(data->idaMemory, &nli);
	IDAGetNumErrTestFails(data->idaMemory, &netf);
	IDAGetNumNonlinSolvConvFails(data->idaMemory, &nncf);
	IDAGetActualInitStep(data->idaMemory, &is);
	IDAGetLastStep(data->idaMemory, &ls);

	llvm::errs() << "\nFinal Run Statistics:\n";

	llvm::errs() << "Number of vector equations       = ";
	llvm::errs() << data->vectorEquationsNumber << "\n";
	llvm::errs() << "Number of scalar equations       = ";
	llvm::errs() << data->scalarEquationsNumber << "\n";
	llvm::errs() << "Number of non-zero values        = ";
	llvm::errs() << data->nonZeroValuesNumber << "\n";

	llvm::errs() << "Number of steps                  = " << nst << "\n";
	llvm::errs() << "Number of residual evaluations   = " << nre << "\n";
	llvm::errs() << "Number of Jacobian evaluations   = " << nje << "\n";

	llvm::errs() << "Number of nonlinear iterations   = " << nni << "\n";
	llvm::errs() << "Number of linear iterations      = " << nli << "\n";
	llvm::errs() << "Number of error test failures    = " << netf << "\n";
	llvm::errs() << "Number of nonlin. conv. failures = " << nncf << "\n";

	llvm::errs() << "Actual initial step size used    = " << is << "\n";
	llvm::errs() << "Step size used for the last step = " << ls << "\n";
}

RUNTIME_FUNC_DEF(printStatistics, void, PTR(void))
