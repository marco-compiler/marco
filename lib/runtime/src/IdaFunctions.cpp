#include <cmath>
#include <ida/ida.h>
#include <marco/runtime/IdaFunctions.h>
#include <nvector/nvector_serial.h>
#include <set>
#include <sstream>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

#define exitOnError(success)                                                   \
	if (!success)                                                                \
		return false;

using EqDimension = std::vector<std::pair<size_t, size_t>>;

using Access = std::vector<std::pair<sunindextype, sunindextype>>;
using Indexes = std::vector<size_t>;
using VarDimension = std::vector<size_t>;

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
	std::vector<std::vector<size_t>> accessIndexes;
	std::vector<EqDimension> equationDimensions;
	std::vector<Function> residuals;
	std::vector<Function> jacobians;
	std::vector<std::pair<Function, Function>> lambdas;

	// Variables data
	std::vector<sunindextype> variableOffsets;
	std::vector<std::pair<sunindextype, Access>> variableAccesses;
	std::vector<VarDimension> variableDimensions;

	// Matrix size
	size_t forEquationsNumber;
	sunindextype equationsNumber;
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
	realtype* variablesValues;
	realtype* derivativesValues;
	realtype* idValues;

	// IDA classes
	void* idaMemory;
	SUNMatrix sparseMatrix;
	SUNLinearSolver linearSolver;
} IdaUserData;

/**
 * Given a list of indexes and the dimension of an equation, increase the
 * indexes within the induction bounds of the equation. Return false if the
 * indexes exceed the equation bounds, which means the computation has finished,
 * true otherwise.
 */
static bool updateIndexes(Indexes& indexes, const EqDimension& dimension)
{
	for (sunindextype dim = dimension.size() - 1; dim >= 0; dim--)
	{
		indexes[dim]++;

		if (indexes[dim] == dimension[dim].second)
			indexes[dim] = dimension[dim].first;
		else
			return true;
	}

	return false;
}

static bool updateIndexes(Indexes& indexes, const VarDimension& dimension)
{
	EqDimension eqDimension;

	for (size_t dim : dimension)
		eqDimension.push_back({ 0, dim });

	return updateIndexes(indexes, eqDimension);
}

/**
 * Given a set of indexes, the dimension of a variable and the type of access,
 * return the index needed to access the flattened multidimensional variable.
 */
static sunindextype computeOffset(
		const Indexes& indexes,
		const VarDimension& dimensions,
		const Access& accesses)
{
	assert(accesses.size() == dimensions.size());

	sunindextype offset =
			accesses[0].first +
			(accesses[0].second != -1 ? indexes[accesses[0].second] : 0);

	for (size_t i = 1; i < accesses.size(); ++i)
	{
		sunindextype accessOffset =
				accesses[i].first +
				(accesses[i].second != -1 ? indexes[accesses[i].second] : 0);
		offset = offset * dimensions[i] + accessOffset;
	}

	return offset;
}

/**
 * Check an IDA function return value in order to find possible failures.
 */
static bool checkRetval(void* retval, const char* funcname, int opt)
{
	// Check if SUNDIALS function returned NULL pointer (no memory allocated)
	if (opt == 0 && retval == NULL)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname;
		llvm::errs() << "() failed - returned NULL pointer\n";
		return false;
	}

	// Check if SUNDIALS function returned a positive integer value
	if (opt == 1 && *((int*) retval) < 0)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname;
		llvm::errs() << "() failed  with return value = " << *(int*) retval << "\n";
		return false;
	}

	return true;
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

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(data->threads)
#endif
	for (size_t eq = 0; eq < data->forEquationsNumber; eq++)
	{
		// For every vector equation
		size_t residualIndex = data->equationIndexes[eq];

		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (const auto& dim : data->equationDimensions[eq])
			indexes.push_back(dim.first);

		// For every scalar equation in the vector equation
		do
		{
			// Compute the i-th residual function
			rval[residualIndex++] =
					data->residuals[eq](tt, 0, yval, ypval, indexes, 0);
		} while (updateIndexes(indexes, data->equationDimensions[eq]));
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
	for (size_t eq = 0; eq < data->forEquationsNumber; eq++)
	{
		// For every vector equation
		size_t rowIndex = data->equationIndexes[eq];
		size_t columnIndex = 0;

		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (const auto& dim : data->equationDimensions[eq])
			indexes.push_back(dim.first);

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
						data->jacobians[eq](tt, cj, yval, ypval, indexes, var);
				colvals[jacobianIndex++] = var;
			}

			// Update multidimensional interval, exit while loop if finished
			columnIndex++;
		} while (updateIndexes(indexes, data->equationDimensions[eq]));
	}

	rowptrs[data->equationsNumber] = data->nonZeroValuesNumber;

	return 0;
}

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

/**
 * Instantiate and initialize the struct of data needed by IDA, given the total
 * number of equations and the maximum number of non-zero values of the jacobian
 * matrix.
 */
template<typename T>
inline void* allocIdaUserData(T equationsNumber)
{
	IdaUserData* data = new IdaUserData;

	data->equationsNumber = equationsNumber;

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

RUNTIME_FUNC_DEF(allocIdaUserData, PTR(void), int32_t)
RUNTIME_FUNC_DEF(allocIdaUserData, PTR(void), int64_t)

/**
 * Precompute the row and column indexes of all non-zero values in the Jacobian
 * Matrix. This avoids the recomputation of such indexes and allows
 * parallelization of the Jacobian computation. Returns the number of non-zero
 * values in the Jacobian Matrix.
 */
sunindextype precomputeJacobianIndexes(IdaUserData* data)
{
	sunindextype nnzElements = 0;
	data->forEquationsNumber = data->equationDimensions.size();

	data->equationIndexes.resize(data->forEquationsNumber + 1);
	data->columnIndexes.resize(data->forEquationsNumber);
	data->nnzElements.resize(data->forEquationsNumber);

	data->equationIndexes[0] = 0;

	for (size_t eq = 0; eq < data->forEquationsNumber; eq++)
	{
		sunindextype equationSize = 1;

		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (const auto& dim : data->equationDimensions[eq])
		{
			indexes.push_back(dim.first);
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
	}

	return nnzElements;
}

/**
 * Instantiate and initialize all the classes needed by IDA in order to solve
 * the given system of equations. It must be called before the first usage of
 * step(). It may fail in case of malformed model.
 */
template<typename T>
inline bool idaInit(void* userData, T threads)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->equationsNumber == 0)
		return true;

	// Compute the total amount of non-zero values in the Jacobian Matrix.
	data->nonZeroValuesNumber = precomputeJacobianIndexes(data);
	data->threads = threads == 0 ? omp_get_max_threads() : threads;

	// Create and initialize IDA memory.
	data->idaMemory = IDACreate();
	exitOnError(checkRetval((void*) data->idaMemory, "IDACreate", 0));

	int retval = IDAInit(
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

	// Set the user-supplied Jacobian routine.
	retval = IDASetJacFn(data->idaMemory, jacobianMatrix);
	exitOnError(checkRetval(&retval, "IDASetJacFn", 1));

	// Add the remaining optional paramters.
	retval = IDASetUserData(data->idaMemory, (void*) data);
	exitOnError(checkRetval(&retval, "IDASetUserData", 1));

	retval = IDASetId(data->idaMemory, data->idVector);
	exitOnError(checkRetval(&retval, "IDASetId", 1));

	retval = IDASetStopTime(data->idaMemory, data->endTime);
	exitOnError(checkRetval(&retval, "IDASetStopTime", 1));

	// Increase the maximum number of steps taken by IDA before failing.
	retval = IDASetMaxNumSteps(data->idaMemory, 1000);
	exitOnError(checkRetval(&retval, "IDASetMaxNumSteps", 1));

	retval = IDASetMaxNumStepsIC(data->idaMemory, 50);
	exitOnError(checkRetval(&retval, "IDASetMaxNumStepsIC", 1));

	retval = IDASetMaxNumJacsIC(data->idaMemory, 40);
	exitOnError(checkRetval(&retval, "IDASetMaxNumJacsIC", 1));

	retval = IDASetMaxNumItersIC(data->idaMemory, 100);
	exitOnError(checkRetval(&retval, "IDASetMaxNumItersIC", 1));

	// Call IDACalcIC to correct the initial values.
	retval = IDACalcIC(data->idaMemory, IDA_YA_YDP_INIT, data->timeStep);
	exitOnError(checkRetval(&retval, "IDACalcIC", 1));

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

	if (data->equationsNumber == 0)
		return true;

	// Execute one step
	int retval = IDASolve(
			data->idaMemory,
			data->nextStop,
			&data->time,
			data->variablesVector,
			data->derivativesVector,
			data->equidistantTimeGrid ? IDA_NORMAL : IDA_ONE_STEP);

	if (data->nextStop < data->endTime)
		data->nextStop += data->timeStep;

	// Check if the solver failed
	exitOnError(checkRetval(&retval, "IDASolve", 1));

	return true;
}

RUNTIME_FUNC_DEF(idaStep, bool, PTR(void))

/**
 * Free all the data allocated by IDA.
 */
inline bool freeIdaUserData(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->equationsNumber == 0)
		return true;

	// Free IDA memory
	IDAFree(&data->idaMemory);

	int retval = SUNLinSolFree(data->linearSolver);
	exitOnError(checkRetval(&retval, "SUNLinSolFree", 1));

	SUNMatDestroy(data->sparseMatrix);
	N_VDestroy(data->variablesVector);
	N_VDestroy(data->derivativesVector);
	N_VDestroy(data->idVector);

	delete data;

	return true;
}

RUNTIME_FUNC_DEF(freeIdaUserData, bool, PTR(void))

/**
 * Add the start time, the end time and the step time to the user data. If the
 * step is equal to the end time, the output will show the variables at every
 * step of the computation. Otherwise the output will show the variables in an
 * equidistant time grid based on the step time paramter.
 */
template<typename T>
inline void addTime(void* userData, T startTime, T endTime, T timeStep)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	data->startTime = startTime;
	data->endTime = endTime;
	data->timeStep = timeStep;
	data->time = startTime;
	data->nextStop = timeStep;
	data->equidistantTimeGrid = endTime != timeStep;
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

	data->equationDimensions.push_back({});
	for (size_t i = 0; i < start.getNumElements(); i++)
		data->equationDimensions.back().push_back({ start[i], end[i] });
}

RUNTIME_FUNC_DEF(addEqDimension, void, PTR(void), ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(addEqDimension, void, PTR(void), ARRAY(int64_t), ARRAY(int64_t))

/**
 * Add the lambda that computes the index-th residual function to the user data.
 * Must be used before the add jacobian function.
 */
template<typename T>
inline void addResidual(void* userData, T leftIndex, T rightIndex)
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

RUNTIME_FUNC_DEF(addResidual, void, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DEF(addResidual, void, PTR(void), int64_t, int64_t)

/**
 * Add the lambda that computes the index-th jacobian row to the user data.
 * Must be used after the add residual function.
 */
template<typename T>
inline void addJacobian(void* userData, T leftIndex, T rightIndex)
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

RUNTIME_FUNC_DEF(addJacobian, void, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DEF(addJacobian, void, PTR(void), int64_t, int64_t)

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

/**
 * Add a new variable given its monodimensional length.
 * Return the variable index.
 */
template<typename T>
inline int64_t addVarOffset(void* userData, T offset)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	data->variableOffsets.push_back(offset);
	return data->variableOffsets.size() - 1;
}

RUNTIME_FUNC_DEF(addVarOffset, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(addVarOffset, int64_t, PTR(void), int64_t)

/**
 * Add a new dimension to the index-th variable of size dim.
 */
template<typename T>
inline void addVarDimension(void* userData, UnsizedArrayDescriptor<T> dimensions)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	assert(dimensions.getRank() == 1);

	data->variableDimensions.push_back({});
	for (T& dim : dimensions)
		data->variableDimensions.back().push_back(dim);
}

RUNTIME_FUNC_DEF(addVarDimension, void, PTR(void), ARRAY(int32_t))
RUNTIME_FUNC_DEF(addVarDimension, void, PTR(void), ARRAY(int64_t))

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
	assert(off.getRank() == 1 && ind.getRank() == 1);
	assert(off.getDimensionSize(0) == ind.getDimensionSize(0));

	data->variableAccesses.push_back({ var, {} });
	for (size_t i = 0; i < off.getNumElements(); i++)
		data->variableAccesses.back().second.push_back({ off[i], ind[i] });

	return data->variableAccesses.size() - 1;
}

RUNTIME_FUNC_DEF(addVarAccess, int32_t, PTR(void), int32_t, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(addVarAccess, int64_t, PTR(void), int64_t, ARRAY(int64_t), ARRAY(int64_t))

/**
 * Set the initial values of the index-th variable given its array, which is
 * represented as a modelica alloc operation. Initialize every other value not
 * included in the array to zero.
 */
template<typename T, typename U>
inline void setInitialValue(
		void* userData,
		T index,
		UnsizedArrayDescriptor<U> array,
		bool isState)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	sunindextype offset = data->variableOffsets[index];
	VarDimension dimensions = data->variableDimensions[index];

	U* arrayData = array.getData();
	realtype idValue = isState ? 1.0 : 0.0;

	Indexes indexes;
	for (size_t dim = 0; dim < dimensions.size(); dim++)
		indexes.push_back(0);

	do
	{
		if (array.hasData(indexes))
			data->variablesValues[offset] = *arrayData++;
		else
			data->variablesValues[offset] = 0;

		data->derivativesValues[offset] = 0.0;
		data->idValues[offset++] = idValue;
	} while (updateIndexes(indexes, dimensions));
}

RUNTIME_FUNC_DEF(setInitialValue, void, PTR(void), int32_t, ARRAY(int32_t), bool)
RUNTIME_FUNC_DEF(setInitialValue, void, PTR(void), int32_t, ARRAY(float), bool)
RUNTIME_FUNC_DEF(setInitialValue, void, PTR(void), int64_t, ARRAY(int64_t), bool)
RUNTIME_FUNC_DEF(setInitialValue, void, PTR(void), int64_t, ARRAY(double), bool)

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

inline double getIdaTime(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	// Return the stop time if the whole system is trivial.
	if (data->equationsNumber == 0)
		return data->endTime;

	return data->time;
}

RUNTIME_FUNC_DEF(getIdaTime, float, PTR(void))
RUNTIME_FUNC_DEF(getIdaTime, double, PTR(void))

template<typename T>
inline double getIdaVariable(void* userData, T index)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	assert(index < data->equationsNumber);
	return data->variablesValues[index];
}

RUNTIME_FUNC_DEF(getIdaVariable, float, PTR(void), int32_t)
RUNTIME_FUNC_DEF(getIdaVariable, double, PTR(void), int64_t)

template<typename T>
inline double getIdaDerivative(void* userData, T index)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	assert(index < data->equationsNumber);
	return data->derivativesValues[index];
}

RUNTIME_FUNC_DEF(getIdaDerivative, float, PTR(void), int32_t)
RUNTIME_FUNC_DEF(getIdaDerivative, double, PTR(void), int64_t)

//===----------------------------------------------------------------------===//
// Lambda constructions
//===----------------------------------------------------------------------===//

template<typename T>
inline int64_t lambdaConstant(void* userData, T constant)
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

RUNTIME_FUNC_DEF(lambdaConstant, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaConstant, int64_t, PTR(void), int64_t)
RUNTIME_FUNC_DEF(lambdaConstant, int32_t, PTR(void), float)
RUNTIME_FUNC_DEF(lambdaConstant, int64_t, PTR(void), double)

inline int64_t lambdaTime(void* userData)
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

RUNTIME_FUNC_DEF(lambdaTime, int32_t, PTR(void))
RUNTIME_FUNC_DEF(lambdaTime, int64_t, PTR(void))

template<typename T>
inline int64_t lambdaInduction(void* userData, T induction)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function first = [induction](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype { return ind[induction] + 1; };

	Function second = [](realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype { return 0.0; };

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

RUNTIME_FUNC_DEF(lambdaInduction, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaInduction, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaVariable(void* userData, T accessIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	sunindextype variableIndex = data->variableAccesses[accessIndex].first;
	sunindextype offset = data->variableOffsets[variableIndex];
	VarDimension dim = data->variableDimensions[variableIndex];
	Access acc = data->variableAccesses[accessIndex].second;

	Function first = [offset, dim, acc](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		sunindextype accessOffset = computeOffset(ind, dim, acc);

		return yy[offset + accessOffset];
	};

	Function second = [offset, dim, acc](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		sunindextype accessOffset = computeOffset(ind, dim, acc);

		if (offset + accessOffset == var)
			return 1.0;
		return 0.0;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

RUNTIME_FUNC_DEF(lambdaVariable, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaVariable, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaDerivative(void* userData, T accessIndex)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	sunindextype variableIndex = data->variableAccesses[accessIndex].first;
	sunindextype offset = data->variableOffsets[variableIndex];
	VarDimension dim = data->variableDimensions[variableIndex];
	Access acc = data->variableAccesses[accessIndex].second;

	Function first = [offset, dim, acc](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		sunindextype accessOffset = computeOffset(ind, dim, acc);

		return yp[offset + accessOffset];
	};

	Function second = [offset, dim, acc](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		sunindextype accessOffset = computeOffset(ind, dim, acc);

		if (offset + accessOffset == var)
			return cj;
		return 0.0;
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

RUNTIME_FUNC_DEF(lambdaDerivative, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaDerivative, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaNegate(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaNegate, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaNegate, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaAdd(void* userData, T leftIndex, T rightIndex)
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

RUNTIME_FUNC_DEF(lambdaAdd, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DEF(lambdaAdd, int64_t, PTR(void), int64_t, int64_t)

template<typename T>
inline int64_t lambdaSub(void* userData, T leftIndex, T rightIndex)
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

RUNTIME_FUNC_DEF(lambdaSub, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DEF(lambdaSub, int64_t, PTR(void), int64_t, int64_t)

template<typename T>
inline int64_t lambdaMul(void* userData, T leftIndex, T rightIndex)
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

RUNTIME_FUNC_DEF(lambdaMul, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DEF(lambdaMul, int64_t, PTR(void), int64_t, int64_t)

template<typename T>
inline int64_t lambdaDiv(void* userData, T leftIndex, T rightIndex)
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

RUNTIME_FUNC_DEF(lambdaDiv, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DEF(lambdaDiv, int64_t, PTR(void), int64_t, int64_t)

template<typename T>
inline int64_t lambdaPow(void* userData, T baseIndex, T exponentIndex)
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

RUNTIME_FUNC_DEF(lambdaPow, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DEF(lambdaPow, int64_t, PTR(void), int64_t, int64_t)

template<typename T>
inline int64_t lambdaAtan2(void* userData, T yIndex, T xIndex)
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

RUNTIME_FUNC_DEF(lambdaAtan2, int32_t, PTR(void), int32_t, int32_t)
RUNTIME_FUNC_DEF(lambdaAtan2, int64_t, PTR(void), int64_t, int64_t)

template<typename T>
inline int64_t lambdaAbs(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaAbs, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaAbs, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaSign(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaSign, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaSign, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaSqrt(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaSqrt, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaSqrt, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaExp(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaExp, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaExp, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaLog(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaLog, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaLog, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaLog10(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaLog10, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaLog10, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaSin(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaSin, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaSin, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaCos(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaCos, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaCos, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaTan(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaTan, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaTan, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaAsin(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaAsin, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaAsin, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaAcos(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaAcos, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaAcos, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaAtan(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaAtan, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaAtan, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaSinh(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaSinh, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaSinh, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaCosh(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaCosh, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaCosh, int64_t, PTR(void), int64_t)

template<typename T>
inline int64_t lambdaTanh(void* userData, T operandIndex)
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

RUNTIME_FUNC_DEF(lambdaTanh, int32_t, PTR(void), int32_t)
RUNTIME_FUNC_DEF(lambdaTanh, int64_t, PTR(void), int64_t)

template<typename T, typename U>
inline int64_t lambdaCall(void* userData, T operandIndex, U function, U pderFunc)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	Function operand = data->lambdas[operandIndex].first;
	Function derOperand = data->lambdas[operandIndex].second;

	Function first = [function, operand](
											 realtype tt,
											 realtype cj,
											 realtype* yy,
											 realtype* yp,
											 Indexes& ind,
											 realtype var) -> realtype {
		return function(operand(tt, cj, yy, yp, ind, var));
	};

	Function second = [pderFunc, operand, derOperand](
												realtype tt,
												realtype cj,
												realtype* yy,
												realtype* yp,
												Indexes& ind,
												realtype var) -> realtype {
		return derOperand(tt, cj, yy, yp, ind, var) *
					 pderFunc(operand(tt, cj, yy, yp, ind, var));
	};

	data->lambdas.push_back({ std::move(first), std::move(second) });
	return data->lambdas.size() - 1;
}

RUNTIME_FUNC_DEF(lambdaCall, int32_t, PTR(void), int32_t, FUNCTION(float), FUNCTION(float))
RUNTIME_FUNC_DEF(lambdaCall, int64_t, PTR(void), int64_t, FUNCTION(double), FUNCTION(double))

//===----------------------------------------------------------------------===//
// Debugging and Statistics
//===----------------------------------------------------------------------===//

void setInitialValue(void* userData, int64_t index, int64_t length, double value, bool isState)
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

int64_t getNumberOfForEquations(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	return data->forEquationsNumber;
}

int64_t getNumberOfEquations(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	return data->equationsNumber;
}

int64_t getNumberOfNonZeroValues(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	return data->nonZeroValuesNumber;
}

EqDimension getIdaDimension(void* userData, int64_t index)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	assert(index < data->equationsNumber);
	return data->equationDimensions[index];
}

int64_t numSteps(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	int64_t nst;
	IDAGetNumSteps(data->idaMemory, &nst);
	return nst;
}

int64_t numResEvals(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	int64_t nre;
	IDAGetNumResEvals(data->idaMemory, &nre);
	return nre;
}

int64_t numJacEvals(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	int64_t nje;
	IDAGetNumJacEvals(data->idaMemory, &nje);
	return nje;
}

int64_t numNonlinIters(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	int64_t nni;
	IDAGetNumNonlinSolvIters(data->idaMemory, &nni);
	return nni;
}

int64_t numLinIters(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	int64_t nli;
	IDAGetNumLinIters(data->idaMemory, &nli);
	return nli;
}

/**
 * Returns the Jacobian incidence matrix of the system inside a string.
 */
std::string getIncidenceMatrix(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	std::stringstream result;

	result << "\n";

	// For every vector equation
	for (size_t eq = 0; eq < data->forEquationsNumber; eq++)
	{
		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (const auto& dim : data->equationDimensions[eq])
			indexes.push_back(dim.first);

		// For every scalar equation in the vector equation
		do
		{
			result << "│";

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

			for (sunindextype i = 0; i < data->equationsNumber; i++)
			{
				if (columnIndexesSet.find(i) != columnIndexesSet.end())
					result << "*";
				else
					result << " ";

				if (i < data->equationsNumber - 1)
					result << " ";
			}

			result << "│\n";
		} while (updateIndexes(indexes, data->equationDimensions[eq]));
	}

	return result.str();
}
