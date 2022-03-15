#include "ida/ida.h"
#include "marco/Runtime/IDA.h"
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
using DerivativeVariable = std::pair<sunindextype, std::vector<sunindextype>>;

using VarDimension = std::vector<size_t>;
using EqDimension = std::vector<std::pair<size_t, size_t>>;

using ResidualFunction = std::function<
    realtype(
        void* userData,
        realtype time,
        sunindextype* ind)>;

using JacobianFunction = std::function<
    realtype(
        void* userData,
        realtype time,
        sunindextype* eqIndices,
        sunindextype var,
        sunindextype* varIndices,
        realtype alpha)>;

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

namespace
{
  /// Container for all the data required by IDA in order to compute the residual
  /// functions and the Jacobian matrix.
  struct IDAUserData
  {
    // The whole simulation data
    void* simulationData;

    // Model size
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
    realtype time;

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
  };
}

#ifdef MARCO_PROFILING

#include "marco/Runtime/Profiling.h"

namespace
{
  class IDAProfiler : public Profiler
  {
    public:
      IDAProfiler() : Profiler("IDA")
      {
        registerProfiler(*this);
      }

      void reset() override
      {
        initialConditionsTimer.reset();
        stepsTimer.reset();
      }

      void print() const override
      {
        std::cerr << "Time spent in computing the initial conditions: "
                  << initialConditionsTimer.totalElapsedTime() << " ms\n";

        std::cerr << "Time spent in IDA steps: "
                  << stepsTimer.totalElapsedTime() << " ms\n";
      }

      Timer initialConditionsTimer;
      Timer stepsTimer;
  };

  IDAProfiler& profiler()
  {
    static IDAProfiler obj;
    return obj;
  }
}

#endif

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

/*
/// Compute the column indexes of the current row of the Jacobian Matrix given
/// the current vector equation and an array of indexes.
static std::set<size_t> computeIndexSet(IDAUserData* data, size_t eq, sunindextype* indexes)
{
  std::set<size_t> columnIndexesSet;

  for (auto& access: data->variableAccesses[eq]) {
    VarDimension& dimensions = data->variableDimensions[access.first];
    sunindextype varOffset = computeOffset(indexes, dimensions, access.second);
    columnIndexesSet.insert(data->variableOffsets[access.first] + varOffset);
  }

  return columnIndexesSet;
}
*/

/// Compute the column indexes of the current row of the Jacobian Matrix given
/// the current vector equation and an array of indexes.
static std::set<DerivativeVariable> computeIndexSet(IDAUserData* data, size_t eq, sunindextype* eqIndexes)
{
  std::set<DerivativeVariable> columnIndexesSet;

  for (auto& access : data->variableAccesses[eq]) {
    sunindextype variableIndex = access.first;
    Access variableAccess = access.second;
    assert(variableAccess.size() == data->variableDimensions[variableIndex].size());

    DerivativeVariable newEntry = {variableIndex, {}};

    for (size_t i = 0; i < variableAccess.size(); ++i) {
      sunindextype induction = variableAccess[i].second;
      sunindextype index = induction != -1 ? eqIndexes[induction] : 0;
      newEntry.second.push_back(index);
    }

    columnIndexesSet.insert(newEntry);
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
static int residualFunction(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void* userData)
{
  realtype* yval = N_VGetArrayPointer(yy);
  realtype* ypval = N_VGetArrayPointer(yp);
  realtype* rval = N_VGetArrayPointer(rr);

  IDAUserData* data = static_cast<IDAUserData*>(userData);

  // For every vector equation
  for (size_t eq = 0; eq < data->equationDimensions.size(); ++eq) {
    // Initialize the multidimensional interval of the vector equation
    sunindextype indexes[data->equationDimensions[eq].size()];

    for (size_t i = 0; i < data->equationDimensions[eq].size(); i++) {
      indexes[i] = data->equationDimensions[eq][i].first;
    }

    // For every scalar equation in the vector equation
    do {
      // Compute the i-th residual function
      *rval++ = data->residuals[eq](userData, tt, indexes);
    } while (updateIndexes(indexes, data->equationDimensions[eq]));
  }

  assert(rval == N_VGetArrayPointer(rr) + data->scalarEquationsNumber);

  return IDA_SUCCESS;
}

/// IDALsJacFn user-defined Jacobian approximation function, passed to IDA
/// through IDASetJacFn. It contains how to compute the Jacobian Matrix of
/// the system, starting from the provided UserData struct, iterating through
/// every equation and variable. The matrix is represented in CSR format.
static int jacobianMatrix(
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

  IDAUserData* data = static_cast<IDAUserData*>(userData);

  sunindextype nnzElements = 0;
  *rowptrs++ = nnzElements;

  // For every vector equation
  for (size_t eq = 0; eq < data->equationDimensions.size(); ++eq) {
    // Initialize the multidimensional interval of the vector equation
    sunindextype eqIndexes[data->equationDimensions[eq].size()];

    for (size_t i = 0; i < data->equationDimensions[eq].size(); ++i) {
      eqIndexes[i] = data->equationDimensions[eq][i].first;
    }

    // For every scalar equation in the vector equation
    do {
      // Compute the column indexes that may be non-zeros
      std::set<DerivativeVariable> columnIndexesSet = computeIndexSet(data, eq, eqIndexes);

      nnzElements += columnIndexesSet.size();
      *rowptrs++ = nnzElements;

      // For every variable with respect to which every equation must be
      // partially differentiated
      for (DerivativeVariable var: columnIndexesSet) {
        // Compute the i-th Jacobian value
        sunindextype* varIndexes = &var.second[0];
        *jacobian++ = data->jacobians[eq](userData, tt, eqIndexes, var.first, varIndexes, cj);
        *colvals++ = var.first;
      }

    } while (updateIndexes(eqIndexes, data->equationDimensions[eq]));
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
// CREATE INSTANCE
static void* idaCreate_pvoid(T scalarEquationsNumber)
{
  IDAUserData* data = new IDAUserData;

  data->scalarEquationsNumber = scalarEquationsNumber;
  data->variableOffsets.push_back(0);

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

  return static_cast<void*>(data);
}

RUNTIME_FUNC_DEF(idaCreate, PTR(void), int32_t)

RUNTIME_FUNC_DEF(idaCreate, PTR(void), int64_t)

/// Compute the number of non-zero values in the Jacobian Matrix. Also compute
/// the column indexes of all non-zero values in the Jacobian Matrix. This avoids
/// the recomputation of such indexes during the Jacobian evaluation.
static void computeNNZ(IDAUserData* data)
{
  data->nonZeroValuesNumber = 0;

  for (size_t eq = 0; eq < data->equationDimensions.size(); ++eq) {
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
static bool idaInit_i1(void* userData)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

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

  #ifdef MARCO_PROFILING
  profiler().initialConditionsTimer.start();
  #endif

  retval = IDACalcIC(data->idaMemory, IDA_YA_YDP_INIT, firstOutTime);

  #ifdef MARCO_PROFILING
  profiler().initialConditionsTimer.stop();
  #endif

  exitOnError(checkRetval(retval, "IDACalcIC"))
  return true;
}

RUNTIME_FUNC_DEF(idaInit, bool, PTR(void))

/// Invoke IDA to perform one step of the computation. If a time step is given,
/// the output will show the variables in an equidistant time grid based on the
/// step time parameter. Otherwise, the output will show the variables at every
/// step of the computation. Returns true if the computation was successful,
/// false otherwise.
template<typename T = double>
static bool idaStep_i1(void* userData, T timeStep = -1)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);
  bool equidistantTimeGrid = timeStep > 0;

  if (data->scalarEquationsNumber == 0) {
    return true;
  }

  // Execute one step
  #ifdef MARCO_PROFILING
  profiler().stepsTimer.start();
  #endif

  int retval = IDASolve(
      data->idaMemory,
      equidistantTimeGrid ? (data->time + timeStep) : data->endTime,
      &data->time,
      data->variablesVector,
      data->derivativesVector,
      equidistantTimeGrid ? IDA_NORMAL : IDA_ONE_STEP);

  #ifdef MARCO_PROFILING
  profiler().stepsTimer.stop();
  #endif

  // Check if the solver failed
  exitOnError(checkRetval(retval, "IDASolve"))

  return true;
}

RUNTIME_FUNC_DEF(idaStep, bool, PTR(void))

RUNTIME_FUNC_DEF(idaStep, bool, PTR(void), float)

RUNTIME_FUNC_DEF(idaStep, bool, PTR(void), double)

/// Free all the data allocated by IDA.
static bool idaFree_i1(void* userData)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

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

RUNTIME_FUNC_DEF(idaFree, bool, PTR(void))

/// Add the start time to the IDA user data.
template<typename T>
static void setStartTime_void(void* userData, T startTime)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  data->startTime = startTime;
  data->time = startTime;
}

RUNTIME_FUNC_DEF(setStartTime, void, PTR(void), float)

RUNTIME_FUNC_DEF(setStartTime, void, PTR(void), double)

/// Add the end time to the IDA user data.
template<typename T>
static void setEndTime_void(void* userData, T endTime)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  data->endTime = endTime;
}

RUNTIME_FUNC_DEF(setEndTime, void, PTR(void), float)

RUNTIME_FUNC_DEF(setEndTime, void, PTR(void), double)

/// Add the relative tolerance to the IDA user data.
template<typename T>
static void setRelativeTolerance_void(void* userData, T relativeTolerance)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  data->relativeTolerance = relativeTolerance;
}

RUNTIME_FUNC_DEF(setRelativeTolerance, void, PTR(void), float)

RUNTIME_FUNC_DEF(setRelativeTolerance, void, PTR(void), double)

/// Add the absolute tolerance to the IDA user data.
template<typename T>
static void setAbsoluteTolerance_void(void* userData, T absoluteTolerance)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  data->absoluteTolerance = absoluteTolerance;
}

RUNTIME_FUNC_DEF(setAbsoluteTolerance, void, PTR(void), float)

RUNTIME_FUNC_DEF(setAbsoluteTolerance, void, PTR(void), double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

/// Add the dimension of an equation to the IDA user data.
template<typename T>
static T addEquation(void* userData, UnsizedArrayDescriptor<T> dimension)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  assert(dimension.getRank() == 2);
  assert(dimension.getDimension(0) == 2);

  // Add the start and end dimensions of the current equation.
  EqDimension eqDimension = {};

  using dimension_t = typename UnsizedArrayDescriptor<T>::dimension_t;
  dimension_t size = dimension.getDimension(1);

  for (dimension_t i = 0; i < size; ++i) {
    eqDimension.push_back({dimension[i], dimension[i + size]});
  }

  data->equationDimensions.push_back(eqDimension);

  // Return the index of the equation.
  return data->equationDimensions.size() - 1;
}

static int32_t addEquation_i32(void* userData, UnsizedArrayDescriptor<int32_t> dimension)
{
  return addEquation<int32_t>(userData, dimension);
}

static int64_t addEquation_i64(void* userData, UnsizedArrayDescriptor<int64_t> dimension)
{
  return addEquation<int64_t>(userData, dimension);
}

RUNTIME_FUNC_DEF(addEquation, int32_t, PTR(void), ARRAY(int32_t))

RUNTIME_FUNC_DEF(addEquation, int64_t, PTR(void), ARRAY(int64_t))

/// Add the function pointer that computes the index-th residual function to the
/// IDA user data.
template<typename T, typename U>
static void addResidual_void(void* userData, T equationIndex, U residualFunction)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  assert(equationIndex >= 0);

  if (data->residuals.size() <= (size_t) equationIndex)
    data->residuals.resize(equationIndex + 1);

  data->residuals[equationIndex] = residualFunction;
}

#if defined(SUNDIALS_SINGLE_PRECISION)
RUNTIME_FUNC_DEF(addResidual, void, PTR(void), int32_t, RESIDUAL(float))
#elif defined(SUNDIALS_DOUBLE_PRECISION)

RUNTIME_FUNC_DEF(addResidual, void, PTR(void), int64_t, RESIDUAL(double))

#endif

/// Add the function pointer that computes the index-th jacobian row to the user
/// data.
template<typename T, typename U>
static void addJacobian_void(void* userData, T equationIndex, U jacobianFunction)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  assert(equationIndex >= 0);

  if (data->jacobians.size() <= (size_t) equationIndex)
    data->jacobians.resize(equationIndex + 1);

  data->jacobians[equationIndex] = jacobianFunction;
}

#if defined(SUNDIALS_SINGLE_PRECISION)
RUNTIME_FUNC_DEF(addJacobian, void, PTR(void), int32_t, JACOBIAN(float))
#elif defined(SUNDIALS_DOUBLE_PRECISION)

RUNTIME_FUNC_DEF(addJacobian, void, PTR(void), int64_t, JACOBIAN(double))

#endif

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

/// Add and initialize a new variable given its array.
///
/// @param userData  opaque pointer to the IDA user data.
/// @param array     allocation operation containing the rank and dimensions.
/// @param isState   indicates if the variable is differential or algebraic.
template<typename T, typename U>
static T addVariable(
    void* userData,
    UnsizedArrayDescriptor<U> array,
    bool isState)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  assert(data->variableOffsets.size() == data->variableDimensions.size() + 1);

  // Add variable offset and dimension.
  size_t offset = data->variableOffsets.back();
  data->variableOffsets.push_back(array.getNumElements());
  data->variableDimensions.push_back(array.getDimensions());

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

  // Return the index of the variable.
  return data->variableDimensions.size() - 1;
}

static int32_t addVariable_i32(void* userData, UnsizedArrayDescriptor<float> array, bool isState)
{
  return addVariable<int32_t, float>(userData, array, isState);
}

static int64_t addVariable_i64(void* userData, UnsizedArrayDescriptor<double> array, bool isState)
{
  return addVariable<int64_t, double>(userData, array, isState);
}

RUNTIME_FUNC_DEF(addVariable, int32_t, PTR(void), ARRAY(float), bool)

RUNTIME_FUNC_DEF(addVariable, int64_t, PTR(void), ARRAY(double), bool)

/// Add a variable access to the var-th variable, where ind is the induction
/// variable and off is the access offset.
template<typename T>
static void addVarAccess_void(
    void* userData,
    T equationIndex,
    T variableIndex,
    UnsizedArrayDescriptor<T> access)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  assert(equationIndex >= 0);
  assert((size_t) equationIndex < data->equationDimensions.size());
  assert(variableIndex >= 0);
  assert((size_t) variableIndex < data->variableDimensions.size());
  assert(access.getRank() == 2);
  assert(access.getDimension(0) == 2);
  assert(access.getDimension(1) == data->variableDimensions[variableIndex].size());

  if (data->variableAccesses.size() <= (size_t) equationIndex)
    data->variableAccesses.resize(equationIndex + 1);

  VarAccessList& varAccessList = data->variableAccesses[equationIndex];
  varAccessList.push_back({variableIndex, {}});

  using dimension_t = typename UnsizedArrayDescriptor<T>::dimension_t;
  dimension_t size = access.getDimension(1);

  for (dimension_t i = 0; i < size; ++i) {
    varAccessList.back().second.push_back({access[i], access[i + size]});
  }
}

RUNTIME_FUNC_DEF(addVarAccess, void, PTR(void), int32_t, int32_t, ARRAY(int32_t))

RUNTIME_FUNC_DEF(addVarAccess, void, PTR(void), int64_t, int64_t, ARRAY(int64_t))

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

/// Returns the pointer to the start of the memory of the requested variable.
template<typename T>
static void* getVariable_pvoid(void* userData, T variableIndex)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  assert(variableIndex >= 0);
  assert((size_t) variableIndex < data->variableDimensions.size());

  return static_cast<void*>(&data->variableValues[data->variableOffsets[variableIndex]]);
}

RUNTIME_FUNC_DEF(getVariable, PTR(void), PTR(void), int32_t)

RUNTIME_FUNC_DEF(getVariable, PTR(void), PTR(void), int64_t)

/// Returns the pointer to the start of the memory of the requested derivative.
template<typename T>
static void* getDerivative_pvoid(void* userData, T derivativeIndex)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  assert(derivativeIndex >= 0);
  assert((size_t) derivativeIndex < data->variableDimensions.size());

  return static_cast<void*>(&data->derivativeValues[data->variableOffsets[derivativeIndex]]);
}

RUNTIME_FUNC_DEF(getDerivative, PTR(void), PTR(void), int32_t)

RUNTIME_FUNC_DEF(getDerivative, PTR(void), PTR(void), int64_t)

/// Returns the time reached by the solver after the last step.
template<typename T>
static T getCurrentTime(void* userData)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  // Return the stop time if the whole system is trivial.
  if (data->scalarEquationsNumber == 0) {
    return data->endTime;
  }

  return data->time;
}

static float getCurrentTime_f32(void* userData)
{
  return getCurrentTime<float>(userData);
}

static double getCurrentTime_f64(void* userData)
{
  return getCurrentTime<double>(userData);
}

RUNTIME_FUNC_DEF(getCurrentTime, float, PTR(void))

RUNTIME_FUNC_DEF(getCurrentTime, double, PTR(void))

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

/// Prints the Jacobian incidence matrix of the system.
static void printIncidenceMatrix(void* userData)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  std::cerr << std::endl;

  // For every vector equation
  for (size_t eq = 0; eq < data->equationDimensions.size(); ++eq) {
    // Initialize the multidimensional interval of the vector equation
    sunindextype indexes[data->equationDimensions[eq].size()];

    for (size_t i = 0; i < data->equationDimensions[eq].size(); ++i) {
      indexes[i] = data->equationDimensions[eq][i].first;
    }

    // For every scalar equation in the vector equation
    do {
      std::cerr << "│";

      // Get the column indexes that may be non-zeros.
      std::set<size_t> columnIndexesSet;

      for (auto& access : data->variableAccesses[eq]) {
        VarDimension& dimensions = data->variableDimensions[access.first];
        sunindextype varOffset = computeOffset(indexes, dimensions, access.second);
        columnIndexesSet.insert(data->variableOffsets[access.first] + varOffset);
      }

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
static void printStatistics_void(void* userData)
{
  IDAUserData* data = static_cast<IDAUserData*>(userData);

  if (data->scalarEquationsNumber == 0) {
    return;
  }

  if (printJacobian) {
    printIncidenceMatrix(data);
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
  std::cerr << data->equationDimensions.size() << std::endl;
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
