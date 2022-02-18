#include "ida/ida.h"
#include "marco/runtime/IDA.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <functional>
#include <iostream>
#include <set>

#define exitOnError(success) if (!success) return false;

using Access = std::vector<std::pair<sunindextype, sunindextype>>;
using VarAccessList = std::vector<std::pair<sunindextype, Access>>;

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

// Debugging options
const bool printJacobian = false;

// Arbitrary initial guesses on 20/12/2021 Modelica Call
const realtype algebraicTolerance = 1e-12;
const realtype timeScalingFactorInit = 1e5;

// Default IDA values
const realtype initTimeStep = 0.0;

const long maxNumSteps = 1e4;
const int maxErrTestFail = 10;
const int maxNonlinIters = 4;
const int maxConvFails = 10;
const realtype nonlinConvCoef = 0.33;

const int maxNumStepsIC = 5;
const int maxNumJacsIC = 4;
const int maxNumItersIC = 10;
const realtype nonlinConvCoefIC = 0.0033;

const int suppressAlg = SUNFALSE;
const int lineSearchOff = SUNFALSE;

/// Container for all the data required by IDA in order to compute the residual
/// functions and the Jacobian matrix.
typedef struct IdaUserData
{
	// Model size
	size_t vectorVariablesNumber;
	size_t vectorEquationsNumber;
	sunindextype scalarEquationsNumber;
	sunindextype nonZeroValuesNumber;

	// Equations data
	std::vector<EqDimension> equationDimensions;
	std::vector<ResidualFunction> residuals;
	std::vector<JacobianFunction> jacobians;
	std::vector<VarAccessList> variableAccesses;

	// Variables data
	std::vector<sunindextype> variableOffsets;
	std::vector<VarDimension> variableDimensions;

	// Simulation times
	realtype startTime;
	realtype endTime;
	realtype timeStep;
	realtype time;
	realtype nextStop;

	// Simulation options
	bool equidistantTimeGrid;

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

/// Given an array of indexes and the dimension of an equation, increase the
/// indexes within the induction bounds of the equation. Return false if the
/// indexes exceed the equation bounds, which means the computation has finished,
/// true otherwise.
static bool updateIndexes(sunindextype* indexes, const EqDimension& dimension)
{
  for (sunindextype i = 0, e = dimension.size(); i < e; ++i) {
    auto pos = e - i - 1;
    ++indexes[pos];

    if ((size_t) indexes[pos] == dimension[pos].second) {
      indexes[pos] = dimension[pos].first;
    } else {
      return true;
    }
  }

	return false;
}

/// Given an array of indexes, the dimension of a variable and the type of
/// access, return the index needed to access the flattened multidimensional
/// variable.
static sunindextype computeOffset(
    const sunindextype* indexes, const VarDimension& dimensions, const Access& accesses)
{
	assert(accesses.size() == dimensions.size());
	sunindextype offset = 0;

	for (size_t i = 0; i < accesses.size(); ++i) {
		sunindextype accessOffset =
				accesses[i].first +
				(accesses[i].second != -1 ? indexes[accesses[i].second] : 0);

		offset = offset * dimensions[i] + accessOffset;
	}

	return offset;
}

/// Compute the column indexes of the current row of the Jacobian Matrix given
/// the current vector equation and an array of indexes.
static std::set<size_t> computeIndexSet(IdaUserData* data, size_t eq, sunindextype* indexes)
{
	std::set<size_t> columnIndexesSet;

	for (auto& access : data->variableAccesses[eq]) {
		VarDimension& dimensions = data->variableDimensions[access.first];
		sunindextype varOffset = computeOffset(indexes, dimensions, access.second);
		columnIndexesSet.insert(data->variableOffsets[access.first] + varOffset);
	}

	return columnIndexesSet;
}

/// Check if SUNDIALS function returned NULL pointer (no memory allocated).
static bool checkAllocation(void* retval, const char* funcname)
{
	if (retval == nullptr) {
		std::cerr << "SUNDIALS_ERROR: " << funcname;
		std::cerr << "() failed - returned NULL pointer" << std::endl;
		return false;
	}

	return true;
}

/// Check if SUNDIALS function returned a success value (positive integer).
static bool checkRetval(int retval, const char* funcname)
{
	if (retval < 0) {
		std::cerr << "SUNDIALS_ERROR: " << funcname;
		std::cerr << "() failed with return value = " << retval << std::endl;
		return false;
	}

	return true;
}

/// IDAResFn user-defined residual function, passed to IDA through IDAInit.
/// It contains how to compute the Residual Function of the system, starting
/// from the provided UserData struct, iterating through every equation.
int residualFunction(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void* userData)
{
	realtype* yval = N_VGetArrayPointer(yy);
	realtype* ypval = N_VGetArrayPointer(yp);
	realtype* rval = N_VGetArrayPointer(rr);

	IdaUserData* data = static_cast<IdaUserData*>(userData);

	// For every vector equation
	for (size_t eq = 0; eq < data->vectorEquationsNumber; ++eq) {
		// Initialize the multidimensional interval of the vector equation
		sunindextype indexes[data->equationDimensions[eq].size()];

		for (size_t i = 0; i < data->equationDimensions[eq].size(); i++) {
			indexes[i] = data->equationDimensions[eq][i].first;
    }

		// For every scalar equation in the vector equation
		do {
			// Compute the i-th residual function
			*rval++ = data->residuals[eq](tt, yval, ypval, indexes);
		} while (updateIndexes(indexes, data->equationDimensions[eq]));
	}

	assert(rval == N_VGetArrayPointer(rr) + data->scalarEquationsNumber);

	return IDA_SUCCESS;
}

/// IDALsJacFn user-defined Jacobian approximation function, passed to IDA
/// through IDASetJacFn. It contains how to compute the Jacobian Matrix of
/// the system, starting from the provided UserData struct, iterating through
/// every equation and variable. The matrix is represented in CSR format.
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

	sunindextype nnzElements = 0;
	*rowptrs++ = nnzElements;

	// For every vector equation
	for (size_t eq = 0; eq < data->vectorEquationsNumber; ++eq) {
		// Initialize the multidimensional interval of the vector equation
		sunindextype indexes[data->equationDimensions[eq].size()];

		for (size_t i = 0; i < data->equationDimensions[eq].size(); ++i) {
      indexes[i] = data->equationDimensions[eq][i].first;
    }

		// For every scalar equation in the vector equation
		do {
			// Compute the column indexes that may be non-zeros
			std::set<size_t> columnIndexesSet = computeIndexSet(data, eq, indexes);

			nnzElements += columnIndexesSet.size();
			*rowptrs++ = nnzElements;

			// For every variable with respect to which every equation must be
			// partially differentiated
			for (sunindextype var : columnIndexesSet) {
				// Compute the i-th Jacobian value
				*jacobian++ = data->jacobians[eq](tt, yval, ypval, indexes, cj, var);
				*colvals++ = var;
			}

		} while (updateIndexes(indexes, data->equationDimensions[eq]));
	}

	assert(rowptrs == SUNSparseMatrix_IndexPointers(JJ) + data->scalarEquationsNumber + 1);
	assert(colvals == SUNSparseMatrix_IndexValues(JJ) + data->nonZeroValuesNumber);
	assert(jacobian == SUNSparseMatrix_Data(JJ) + data->nonZeroValuesNumber);

	return IDA_SUCCESS;
}

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

/// Instantiate and initialize the struct of data needed by IDA, given the total
/// number of scalar equations.
template<typename T>
inline void* idaAllocData_pvoid(T scalarEquationsNumber, T vectorEquationsNumber, T vectorVariablesNumber)
{
	IdaUserData* data = new IdaUserData;

	data->scalarEquationsNumber = scalarEquationsNumber;
	data->vectorEquationsNumber = 0;
	data->vectorVariablesNumber = 0;
	data->nonZeroValuesNumber = 0;

	// Create and initialize the required N-vectors for the variables.
	data->variablesVector = N_VNew_Serial(data->scalarEquationsNumber);
	assert(checkAllocation(static_cast<void*>(data->variablesVector), "N_VNew_Serial"));

	data->derivativesVector = N_VNew_Serial(data->scalarEquationsNumber);
	assert(checkAllocation(static_cast<void*>(data->derivativesVector), "N_VNew_Serial"));

	data->idVector = N_VNew_Serial(data->scalarEquationsNumber);
	assert(checkAllocation(static_cast<void*>(data->idVector), "N_VNew_Serial"));

	data->tolerancesVector = N_VNew_Serial(data->scalarEquationsNumber);
	assert(checkAllocation(static_cast<void*>(data->tolerancesVector), "N_VNew_Serial"));

	data->variableValues = N_VGetArrayPointer(data->variablesVector);
	data->derivativeValues = N_VGetArrayPointer(data->derivativesVector);
	data->idValues = N_VGetArrayPointer(data->idVector);
	data->toleranceValues = N_VGetArrayPointer(data->tolerancesVector);

	data->equationDimensions.resize(vectorEquationsNumber);
	data->residuals.resize(vectorEquationsNumber);
	data->jacobians.resize(vectorEquationsNumber);
	data->variableAccesses.resize(vectorEquationsNumber);

	data->variableOffsets.resize(vectorVariablesNumber);
	data->variableDimensions.resize(vectorVariablesNumber);

	return static_cast<void*>(data);
}

RUNTIME_FUNC_DEF(idaAllocData, PTR(void), int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(idaAllocData, PTR(void), int64_t, int64_t, int64_t)

/// Compute the number of non-zero values in the Jacobian Matrix. Also compute
/// the column indexes of all non-zero values in the Jacobian Matrix. This avoids
/// the recomputation of such indexes during the Jacobian evaluation.
void computeNNZ(IdaUserData* data)
{
	for (size_t eq = 0; eq < data->vectorEquationsNumber; ++eq) {
		// Initialize the multidimensional interval of the vector equation
		sunindextype indexes[data->equationDimensions[eq].size()];

		for (size_t i = 0; i < data->equationDimensions[eq].size(); ++i) {
      indexes[i] = data->equationDimensions[eq][i].first;
    }

		// For every scalar equation in the vector equation
		do {
			// Compute the column indexes that may be non-zeros
			data->nonZeroValuesNumber += computeIndexSet(data, eq, indexes).size();
		} while (updateIndexes(indexes, data->equationDimensions[eq]));
	}
}

/// Instantiate and initialize all the classes needed by IDA in order to solve
/// the given system of equations. It also sets optional simulation parameters
/// for IDA. It must be called before the first usage of idaStep() and after a
/// call to idaAllocData(). It may fail in case of malformed model.
inline bool idaInit_i1(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->scalarEquationsNumber == 0) {
    return true;
  }

	// Compute the total amount of non-zero values in the Jacobian Matrix.
	computeNNZ(data);

	// Create and initialize IDA memory.
	data->idaMemory = IDACreate();
	exitOnError(checkAllocation(static_cast<void*>(data->idaMemory), "IDACreate"))

	int retval = IDAInit(
			data->idaMemory,
			residualFunction,
			data->startTime,
			data->variablesVector,
			data->derivativesVector);

	exitOnError(checkRetval(retval, "IDAInit"))

	// Set tolerance and id of every scalar value.
	retval = IDASVtolerances(data->idaMemory, data->relativeTolerance, data->tolerancesVector);
	exitOnError(checkRetval(retval, "IDASVtolerances"))
	N_VDestroy(data->tolerancesVector);

	retval = IDASetId(data->idaMemory, data->idVector);
	exitOnError(checkRetval(retval, "IDASetId"))
	N_VDestroy(data->idVector);

	// Create sparse SUNMatrix for use in linear solver.
	data->sparseMatrix = SUNSparseMatrix(
			data->scalarEquationsNumber,
			data->scalarEquationsNumber,
			data->nonZeroValuesNumber,
			CSR_MAT);

	exitOnError(checkAllocation(static_cast<void*>(data->sparseMatrix), "SUNSparseMatrix"))

	// Create and attach a KLU SUNLinearSolver object.
	data->linearSolver = SUNLinSol_KLU(data->variablesVector, data->sparseMatrix);
	exitOnError(checkAllocation(static_cast<void*>(data->linearSolver), "SUNLinSol_KLU"))

	retval = IDASetLinearSolver(data->idaMemory, data->linearSolver, data->sparseMatrix);
	exitOnError(checkRetval(retval, "IDASetLinearSolver"))

	// Set the user-supplied Jacobian routine.
	retval = IDASetJacFn(data->idaMemory, jacobianMatrix);
	exitOnError(checkRetval(retval, "IDASetJacFn"))

	// Add the remaining mandatory parameters.
	retval = IDASetUserData(data->idaMemory, static_cast<void*>(data));
	exitOnError(checkRetval(retval, "IDASetUserData"))

	retval = IDASetStopTime(data->idaMemory, data->endTime);
	exitOnError(checkRetval(retval, "IDASetStopTime"))

	// Add the remaining optional parameters.
	retval = IDASetInitStep(data->idaMemory, initTimeStep);
	exitOnError(checkRetval(retval, "IDASetInitStep"))

	retval = IDASetMaxStep(data->idaMemory, data->endTime);
	exitOnError(checkRetval(retval, "IDASetMaxStep"))

	retval = IDASetSuppressAlg(data->idaMemory, suppressAlg);
	exitOnError(checkRetval(retval, "IDASetSuppressAlg"))

	// Increase the maximum number of iterations taken by IDA before failing.
	retval = IDASetMaxNumSteps(data->idaMemory, maxNumSteps);
	exitOnError(checkRetval(retval, "IDASetMaxNumSteps"))

	retval = IDASetMaxErrTestFails(data->idaMemory, maxErrTestFail);
	exitOnError(checkRetval(retval, "IDASetMaxErrTestFails"))

	retval = IDASetMaxNonlinIters(data->idaMemory, maxNonlinIters);
	exitOnError(checkRetval(retval, "IDASetMaxNonlinIters"))

	retval = IDASetMaxConvFails(data->idaMemory, maxConvFails);
	exitOnError(checkRetval(retval, "IDASetMaxConvFails"))

	retval = IDASetNonlinConvCoef(data->idaMemory, nonlinConvCoef);
	exitOnError(checkRetval(retval, "IDASetNonlinConvCoef"))

	// Increase the maximum number of iterations taken by IDA IC before failing.
	retval = IDASetMaxNumStepsIC(data->idaMemory, maxNumStepsIC);
	exitOnError(checkRetval(retval, "IDASetMaxNumStepsIC"))

	retval = IDASetMaxNumJacsIC(data->idaMemory, maxNumJacsIC);
	exitOnError(checkRetval(retval, "IDASetMaxNumJacsIC"))

	retval = IDASetMaxNumItersIC(data->idaMemory, maxNumItersIC);
	exitOnError(checkRetval(retval, "IDASetMaxNumItersIC"))

	retval = IDASetNonlinConvCoefIC(data->idaMemory, nonlinConvCoefIC);
	exitOnError(checkRetval(retval, "IDASetNonlinConvCoefIC"))

	retval = IDASetLineSearchOffIC(data->idaMemory, lineSearchOff);
	exitOnError(checkRetval(retval, "IDASetLineSearchOffIC"))

	// Call IDACalcIC to correct the initial values.
	realtype firstOutTime = (data->endTime - data->startTime) / timeScalingFactorInit;
	retval = IDACalcIC(data->idaMemory, IDA_YA_YDP_INIT, firstOutTime);
	exitOnError(checkRetval(retval, "IDACalcIC"))

	return true;
}

RUNTIME_FUNC_DEF(idaInit, bool, PTR(void))

/// Invoke IDA to perform one step of the computation. Returns false if the
/// computation failed, true otherwise.
inline bool idaStep_i1(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->scalarEquationsNumber == 0) {
    return true;
  }

	// Execute one step
	int retval = IDASolve(
			data->idaMemory,
			data->nextStop,
			&data->time,
			data->variablesVector,
			data->derivativesVector,
			data->equidistantTimeGrid ? IDA_NORMAL : IDA_ONE_STEP);

	if (data->equidistantTimeGrid) {
    data->nextStop += data->timeStep;
  }

	// Check if the solver failed
	exitOnError(checkRetval(retval, "IDASolve"))

	return true;
}

RUNTIME_FUNC_DEF(idaStep, bool, PTR(void))

/// Free all the data allocated by IDA.
inline bool idaFreeData_i1(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->scalarEquationsNumber == 0) {
    return true;
  }

	// Deallocate the IDA memory
	IDAFree(&data->idaMemory);

	int retval = SUNLinSolFree(data->linearSolver);
	exitOnError(checkRetval(retval, "SUNLinSolFree"))

	SUNMatDestroy(data->sparseMatrix);
	N_VDestroy(data->variablesVector);
	N_VDestroy(data->derivativesVector);

	delete data;

	return true;
}

RUNTIME_FUNC_DEF(idaFreeData, bool, PTR(void))

/// Add the start time, the end time and the step time to the IDA user data. If a
/// positive time step is given, the output will show the variables in an
/// equidistant time grid based on the step time parameter. Otherwise, the output
/// will show the variables at every step of the computation.
template<typename T>
inline void addTime_void(void* userData, T startTime, T endTime, T timeStep)
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

/// Add the relative tolerance and the absolute tolerance to the IDA user data.
template<typename T>
inline void addTolerance_void(void* userData, T relTol, T absTol)
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

/// Add the dimension of an equation to the IDA user data.
template<typename T>
inline void addEquation_void(void* userData, UnsizedArrayDescriptor<T> dimension)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	assert(dimension.getRank() == 2);
	assert(dimension.getDimension(0) == 2);

	// Add the start and end dimensions of the current equation.
	EqDimension& eqDimension = data->equationDimensions[data->vectorEquationsNumber];

  using dimension_t = typename UnsizedArrayDescriptor<T>::dimension_t;
  dimension_t size = dimension.getDimension(1);

	for (dimension_t i = 0; i < size; ++i) {
    eqDimension.push_back({ dimension[i], dimension[i + size] });
  }

	data->vectorEquationsNumber++;
}

RUNTIME_FUNC_DEF(addEquation, void, PTR(void), ARRAY(int32_t))
RUNTIME_FUNC_DEF(addEquation, void, PTR(void), ARRAY(int64_t))

/// Add the function pointer that computes the index-th residual function to the
/// IDA user data.
template<typename T>
inline void addResidual_void(void* userData, T residualFunction)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	data->residuals[data->vectorEquationsNumber - 1] = residualFunction;
}

#if defined(SUNDIALS_SINGLE_PRECISION)
RUNTIME_FUNC_DEF(addResidual, void, PTR(void), RESIDUAL(float))
#elif defined(SUNDIALS_DOUBLE_PRECISION)
RUNTIME_FUNC_DEF(addResidual, void, PTR(void), RESIDUAL(double))
#endif

/// Add the function pointer that computes the index-th jacobian row to the user
/// data.
template<typename T>
inline void addJacobian_void(void* userData, T jacobianFunction)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);
	data->jacobians[data->vectorEquationsNumber - 1] = jacobianFunction;
}

#if defined(SUNDIALS_SINGLE_PRECISION)
RUNTIME_FUNC_DEF(addJacobian, void, PTR(void), JACOBIAN(float))
#elif defined(SUNDIALS_DOUBLE_PRECISION)
RUNTIME_FUNC_DEF(addJacobian, void, PTR(void), JACOBIAN(double))
#endif

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

/// Add and initialize a new variable given its array.
///
/// @param userData  opaque pointer to the IDA user data.
/// @param offset    offset of the current variable from the beginning of the array.
/// @param array     allocation operation containing the rank and dimensions.
/// @param isState   indicates if the variable is differential or algebraic.
template<typename T, typename U>
inline void addVariable_void(
		void* userData,
		T offset,
		UnsizedArrayDescriptor<U> array,
		bool isState)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	assert(offset >= 0);
	assert(offset + array.getNumElements() <= (size_t) data->scalarEquationsNumber);

	// Add variable offset and dimension.
	data->variableOffsets[data->vectorVariablesNumber] = offset;
	data->variableDimensions[data->vectorVariablesNumber] = array.getDimensions();
	data->vectorVariablesNumber++;

	// Compute idValue and absoluteTolerance.
	realtype idValue = isState ? 1.0 : 0.0;
	realtype absTol = isState 
			? data->absoluteTolerance
			: std::min(algebraicTolerance, data->absoluteTolerance);

	// Initialize derivativeValues, idValues and absoluteTolerances.
  using dimension_t = typename UnsizedArrayDescriptor<U>::dimension_t;

	for (dimension_t i = 0, e = array.getNumElements(); i < e; ++i) {
		data->derivativeValues[offset + i] = 0.0;
		data->idValues[offset + i] = idValue;
		data->toleranceValues[offset + i] = absTol;
	}
}

RUNTIME_FUNC_DEF(addVariable, void, PTR(void), int32_t, ARRAY(float), bool)
RUNTIME_FUNC_DEF(addVariable, void, PTR(void), int64_t, ARRAY(double), bool)

/// Add a variable access to the var-th variable, where ind is the induction
/// variable and off is the access offset.
template<typename T>
inline void addVarAccess_void(
		void* userData,
		T variableIndex,
		UnsizedArrayDescriptor<T> access)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	assert(variableIndex >= 0);
	assert((size_t) variableIndex < data->vectorVariablesNumber);
	assert(access.getRank() == 2);
	assert(access.getDimension(0) == 2);
	assert(access.getDimension(1) == data->variableDimensions[variableIndex].size());

	VarAccessList& varAccessList = data->variableAccesses[data->vectorEquationsNumber - 1];
	varAccessList.push_back({ variableIndex, {} });

  using dimension_t = typename UnsizedArrayDescriptor<T>::dimension_t;
  dimension_t size = access.getDimension(1);

	for (dimension_t i = 0; i < size; ++i) {
    varAccessList.back().second.push_back({ access[i], access[i + size] });
  }
}

RUNTIME_FUNC_DEF(addVarAccess, void, PTR(void), int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(addVarAccess, void, PTR(void), int64_t, ARRAY(int64_t))

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

/// Returns the pointer to the start of the memory of the requested variable
/// given its offset and if it is a derivative or not.
template<typename T>
inline void* getVariableAlloc_pvoid(void* userData, T offset, bool isDerivative)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	assert(offset >= 0);
	assert(offset < data->scalarEquationsNumber);

	if (isDerivative) {
    return static_cast<void*>(&data->derivativeValues[offset]);
  }

	return static_cast<void*>(&data->variableValues[offset]);
}

RUNTIME_FUNC_DEF(getVariableAlloc, PTR(void), PTR(void), int32_t, bool)
RUNTIME_FUNC_DEF(getVariableAlloc, PTR(void), PTR(void), int64_t, bool)

/// Returns the time reached by the solver after the last step.
template<typename T>
inline T getIdaTime(void* userData)
{
  IdaUserData* data = static_cast<IdaUserData*>(userData);

  // Return the stop time if the whole system is trivial.
  if (data->scalarEquationsNumber == 0) {
    return data->endTime;
  }

  return data->time;
}

inline float getIdaTime_f32(void* userData)
{
	return getIdaTime<float>(userData);
}

inline double getIdaTime_f64(void* userData)
{
  return getIdaTime<double>(userData);
}

RUNTIME_FUNC_DEF(getIdaTime, float, PTR(void))
RUNTIME_FUNC_DEF(getIdaTime, double, PTR(void))

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

/// Prints the Jacobian incidence matrix of the system.
static void printIncidenceMatrix(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	std::cerr << std::endl;

	// For every vector equation
	for (size_t eq = 0; eq < data->vectorEquationsNumber; ++eq) {
		// Initialize the multidimensional interval of the vector equation
		sunindextype indexes[data->equationDimensions[eq].size()];

		for (size_t i = 0; i < data->equationDimensions[eq].size(); ++i) {
      indexes[i] = data->equationDimensions[eq][i].first;
    }

		// For every scalar equation in the vector equation
		do {
			std::cerr << "│";

			// Get the column indexes that may be non-zeros.
			std::set<size_t> columnIndexesSet = computeIndexSet(data, eq, indexes);

			for (sunindextype i = 0; i < data->scalarEquationsNumber; ++i) {
				if (columnIndexesSet.find(i) != columnIndexesSet.end()) {
          std::cerr << "*";
        } else {
          std::cerr << " ";
        }

				if (i < data->scalarEquationsNumber - 1) {
          std::cerr << " ";
        }
			}

			std::cerr << "│" << std::endl;
		} while (updateIndexes(indexes, data->equationDimensions[eq]));
	}
}

/// Prints statistics about the computation of the system.
inline void printStatistics_void(void* userData)
{
	IdaUserData* data = static_cast<IdaUserData*>(userData);

	if (data->scalarEquationsNumber == 0) {
    return;
  }

	if (printJacobian) {
    printIncidenceMatrix(userData);
  }

	long nst, nre, nje, nni, nli, netf, nncf;
	realtype ais, ls;

	IDAGetNumSteps(data->idaMemory, &nst);
	IDAGetNumResEvals(data->idaMemory, &nre);
	IDAGetNumJacEvals(data->idaMemory, &nje);
	IDAGetNumNonlinSolvIters(data->idaMemory, &nni);
	IDAGetNumLinIters(data->idaMemory, &nli);
	IDAGetNumErrTestFails(data->idaMemory, &netf);
	IDAGetNumNonlinSolvConvFails(data->idaMemory, &nncf);
	IDAGetActualInitStep(data->idaMemory, &ais);
	IDAGetLastStep(data->idaMemory, &ls);

	std::cerr << std::endl << "Final Run Statistics:" << std::endl;

	std::cerr << "Number of vector equations       = ";
	std::cerr << data->vectorEquationsNumber << std::endl;
	std::cerr << "Number of scalar equations       = ";
	std::cerr << data->scalarEquationsNumber << std::endl;
	std::cerr << "Number of non-zero values        = ";
	std::cerr << data->nonZeroValuesNumber << std::endl;

	std::cerr << "Number of steps                  = " << nst << std::endl;
	std::cerr << "Number of residual evaluations   = " << nre << std::endl;
	std::cerr << "Number of Jacobian evaluations   = " << nje << std::endl;

	std::cerr << "Number of nonlinear iterations   = " << nni << std::endl;
	std::cerr << "Number of linear iterations      = " << nli << std::endl;
	std::cerr << "Number of error test failures    = " << netf << std::endl;
	std::cerr << "Number of nonlin. conv. failures = " << nncf << std::endl;

	std::cerr << "Actual initial step size used    = " << ais << std::endl;
	std::cerr << "Step size used for the last step = " << ls << std::endl;
}

RUNTIME_FUNC_DEF(printStatistics, void, PTR(void))
