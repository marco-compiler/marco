#include "marco/Runtime/IDA.h"
#include "marco/Runtime/ArrayDescriptor.h"
#include "marco/Runtime/MemoryManagement.h"
#include "ida/ida.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <cassert>
#include <climits>
#include <functional>
#include <iostream>
#include <set>

using Access = std::vector<std::pair<sunindextype, sunindextype>>;
using VarAccessList = std::vector<std::pair<sunindextype, Access>>;

using DerivativeVariable = std::pair<size_t, std::vector<size_t>>;

namespace
{
  class VariableIndicesIterator;

  class VariableDimensions
  {
    private:
      using Container = std::vector<size_t>;

    public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      VariableDimensions(size_t rank);

      size_t& operator[](size_t index);
      const size_t& operator[](size_t index) const;

      size_t rank() const;

      const_iterator begin() const;
      const_iterator end() const;

      VariableIndicesIterator indicesBegin() const;
      VariableIndicesIterator indicesEnd() const;

    private:
      Container dimensions;
  };

  class VariableIndicesIterator
  {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = size_t*;
      using difference_type = std::ptrdiff_t;
      using pointer = size_t**;
      using reference = size_t*&;

      ~VariableIndicesIterator()
      {
        delete indices;
      }

      static VariableIndicesIterator begin(const VariableDimensions& dimensions)
      {
        VariableIndicesIterator result(dimensions);

        for (size_t i = 0; i < dimensions.rank(); ++i) {
          result.indices[i] = 0;
        }

        return result;
      }

      static VariableIndicesIterator end(const VariableDimensions& dimensions)
      {
        VariableIndicesIterator result(dimensions);

        for (size_t i = 0; i < dimensions.rank(); ++i) {
          result.indices[i] = dimensions[i];
        }

        return result;
      }

      bool operator==(const VariableIndicesIterator& it) const
      {
        if (dimensions != it.dimensions) {
          return false;
        }

        for (size_t i = 0; i < dimensions->rank(); ++i) {
          if (indices[i] != it.indices[i]) {
            return false;
          }
        }

        return true;
      }

      bool operator!=(const VariableIndicesIterator& it) const
      {
        if (dimensions != it.dimensions) {
          return true;
        }

        for (size_t i = 0; i < dimensions->rank(); ++i) {
          if (indices[i] != it.indices[i]) {
            return true;
          }
        }

        return false;
      }

      VariableIndicesIterator& operator++()
      {
        fetchNext();
        return *this;
      }

      VariableIndicesIterator operator++(int)
      {
        auto temp = *this;
        fetchNext();
        return temp;
      }

      size_t* operator*() const
      {
        return indices;
      }

    private:
      VariableIndicesIterator(const VariableDimensions& dimensions) : dimensions(&dimensions)
      {
        indices = new size_t[dimensions.rank()];
      }

      void fetchNext()
      {
        size_t rank = dimensions->rank();
        size_t posFromLast = 0;

        assert(std::none_of(dimensions->begin(), dimensions->end(), [](const auto& dimension) {
          return dimension == 0;
        }));

        while (posFromLast < rank && ++indices[rank - posFromLast - 1] == (*dimensions)[rank - posFromLast - 1]) {
          ++posFromLast;
        }

        if (posFromLast != rank) {
          for (size_t i = 0; i < posFromLast; ++i) {
            indices[rank - i - 1] = 0;
          }
        }
      }

    private:
      size_t* indices;
      const VariableDimensions* dimensions;
  };

  VariableDimensions::VariableDimensions(size_t rank)
  {
    dimensions.resize(rank, 0);
  }

  size_t& VariableDimensions::operator[](size_t index)
  {
    return dimensions[index];
  }

  const size_t& VariableDimensions::operator[](size_t index) const
  {
    return dimensions[index];
  }

  size_t VariableDimensions::rank() const
  {
    return dimensions.size();
  }

  VariableDimensions::const_iterator VariableDimensions::begin() const
  {
    return dimensions.begin();
  }

  VariableDimensions::const_iterator VariableDimensions::end() const
  {
    return dimensions.end();
  }

  VariableIndicesIterator VariableDimensions::indicesBegin() const
  {
    return VariableIndicesIterator::begin(*this);
  }

  VariableIndicesIterator VariableDimensions::indicesEnd() const
  {
    return VariableIndicesIterator::end(*this);
  }
}

using EqDimension = std::vector<std::pair<size_t, size_t>>;

template<typename FloatType>
using VariableGetterFunction = FloatType(*)(void*, size_t*);

template<typename FloatType>
using VariableSetterFunction = void(*)(void*, FloatType, size_t*);

template<typename FloatType>
using ResidualFunction = FloatType(*)(FloatType, void*, size_t*);

template<typename FloatType>
using JacobianFunction = FloatType(*)(FloatType, void*, size_t*, size_t*, FloatType);

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
  class IDAInstance
  {
    public:
      static constexpr realtype kUndefinedTimeStep = -1;

      IDAInstance(int64_t marcoBitWidth, int64_t scalarEquationsNumber);

      ~IDAInstance();

      void setStartTime(double time);
      void setEndTime(double time);
      void setTimeStep(double time);

      void setRelativeTolerance(double tolerance);
      void setAbsoluteTolerance(double tolerance);

      /// Add and initialize a new variable given its array.
      int64_t addAlgebraicVariable(void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter);

      int64_t addStateVariable(void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter);

      void setDerivative(int64_t stateVariable, void* derivative, void* getter, void* setter);

      /// Add the dimension of an equation to the IDA user data.
      int64_t addEquation(int64_t* ranges, int64_t rank);

      void addVariableAccess(int64_t equationIndex, int64_t variableIndex, int64_t* access, int64_t rank);

      /// Add the function pointer that computes the index-th residual function to the
      /// IDA user data.
      void addResidualFunction(int64_t equationIndex, void* residualFunction);

      /// Add the function pointer that computes the index-th jacobian row to the user
      /// data.
      void addJacobianFunction(int64_t equationIndex, int64_t variableIndex, void* jacobianFunction);

      /// Instantiate and initialize all the classes needed by IDA in order to solve
      /// the given system of equations. It also sets optional simulation parameters
      /// for IDA. It must be called before the first usage of idaStep() and after a
      /// call to idaAllocData(). It may fail in case of malformed model.
      bool initialize();

      /// Invoke IDA to perform one step of the computation. If a time step is given,
      /// the output will show the variables in an equidistant time grid based on the
      /// step time parameter. Otherwise, the output will show the variables at every
      /// step of the computation. Returns true if the computation was successful,
      /// false otherwise.
      bool step();

      /// Returns the time reached by the solver after the last step.
      realtype getCurrentTime() const;

      void* getVariable(int64_t variableIndex) const;

      void* getDerivative(int64_t derivativeIndex) const;

      /// Prints statistics regarding the computation of the system.
      void printStatistics() const;

      /// IDAResFn user-defined residual function, passed to IDA through IDAInit.
      /// It contains how to compute the Residual Function of the system, starting
      /// from the provided UserData struct, iterating through every equation.
      static int residualFunction(
          realtype time,
          N_Vector variables, N_Vector derivatives, N_Vector residuals,
          void* userData);

      /// IDALsJacFn user-defined Jacobian approximation function, passed to IDA
      /// through IDASetJacFn. It contains how to compute the Jacobian Matrix of
      /// the system, starting from the provided UserData struct, iterating through
      /// every equation and variable. The matrix is represented in CSR format.
      static int jacobianMatrix(
          realtype time, realtype alpha,
          N_Vector variables, N_Vector derivatives, N_Vector residuals,
          SUNMatrix jacobianMatrix,
          void* userData,
          N_Vector tempv1, N_Vector tempv2, N_Vector tempv3);

    private:
      std::set<DerivativeVariable> computeIndexSet(size_t eq, size_t* eqIndexes) const;

      void computeNNZ();

      void copyVariablesFromMARCO(N_Vector values);

      void copyDerivativesFromMARCO(N_Vector values);

      void copyVariablesIntoMARCO(N_Vector values);

      void copyDerivativesIntoMARCO(N_Vector values);

      /// Prints the Jacobian incidence matrix of the system.
      void printIncidenceMatrix() const;

    private:
      bool initialized;

      int64_t marcoBitWidth;
      static constexpr int64_t idaBitWidth = sizeof(realtype) * CHAR_BIT;

      // Model size
      int64_t scalarEquationsNumber;
      int64_t nonZeroValuesNumber;

      // Equations data
      std::vector<EqDimension> equationDimensions;
      std::vector<void*> residuals;
      std::vector<std::vector<void*>> jacobians;
      std::vector<VarAccessList> variableAccesses;

      // The offset of each array variable inside the flattened variables vector
      std::vector<sunindextype> variableOffsets;

      // The dimensions list of each array variable
      std::vector<VariableDimensions> variablesDimensions;
      std::vector<VariableDimensions> derivativesDimensions;

      // Simulation times
      realtype startTime;
      realtype endTime;
      realtype timeStep;
      realtype currentTime;

      // Error tolerances
      realtype relativeTolerance;
      realtype absoluteTolerance;

      // Variables vectors and values
      N_Vector variablesVector;
      N_Vector derivativesVector;

      // The vector stores whether each scalar variable is an algebraic or a state one.
      // 0 = algebraic
      // 1 = state
      N_Vector idVector;

      // The tolerance for each scalar variable.
      N_Vector tolerancesVector;

      // The variables upon which MARCO operates (in other words, the ones retrieved
      // through idaGetVariable and idaGetDerivative).
      void* marcoVariableValues;
      void* marcoDerivativeValues;

      // IDA classes
      void* idaMemory;
      SUNMatrix sparseMatrix;
      SUNLinearSolver linearSolver;

      std::vector<void*> variables;
      std::vector<void*> derivatives;

      std::vector<void*> variablesGetters;
      std::vector<void*> derivativesGetters;

      std::vector<void*> variablesSetters;
      std::vector<void*> derivativesSetters;

      void** simulationData;
  };
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

/// Given an array of indexes and the dimension of an equation, increase the
/// indexes within the induction bounds of the equation. Return false if the
/// indexes exceed the equation bounds, which means the computation has finished,
/// true otherwise.
static bool updateIndexes(size_t* indexes, const EqDimension& dimension)
{
  for (size_t i = 0, e = dimension.size(); i < e; ++i) {
    size_t pos = e - i - 1;
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
static size_t computeOffset(
    const size_t* indexes, const VariableDimensions& dimensions, const Access& accesses)
{
  assert(accesses.size() == dimensions.rank());
  size_t offset = 0;

  for (size_t i = 0; i < accesses.size(); ++i) {
    int64_t induction = accesses[i].first;
    size_t accessOffset = induction != -1 ? indexes[induction] : 0;
    accessOffset += accesses[i].second;

    offset = offset * dimensions[i] + accessOffset;
  }

  return offset;
}


/// Given the dimension of a variable and the already flattened accesses, return
/// the index of the flattened multidimensional variable.
static size_t computeOffset(const VariableDimensions& dimensions, const std::vector<size_t>& accesses)
{
	assert(accesses.size() == dimensions.rank());

	size_t offset = 0;

	for (size_t i = 0; i < accesses.size(); ++i) {
		offset = offset * dimensions[i] + accesses[i];
	}

	return offset;
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
      std::cerr << "Time spent on computing the initial conditions: " << initialConditionsTimer.totalElapsedTime() << " ms\n";
      std::cerr << "Time spent on IDA steps: " << stepsTimer.totalElapsedTime() << " ms\n";
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

namespace
{
  IDAInstance::IDAInstance(int64_t marcoBitWidth, int64_t scalarEquationsNumber)
      : initialized(false),
        marcoBitWidth(marcoBitWidth),
        scalarEquationsNumber(scalarEquationsNumber),
        timeStep(kUndefinedTimeStep)
  {
    variableOffsets.push_back(0);

    if (scalarEquationsNumber != 0) {
      // Create and initialize the required N-vectors for the variables.
      variablesVector = N_VNew_Serial(scalarEquationsNumber);
      assert(checkAllocation(static_cast<void*>(variablesVector), "N_VNew_Serial"));

      derivativesVector = N_VNew_Serial(scalarEquationsNumber);
      assert(checkAllocation(static_cast<void*>(derivativesVector), "N_VNew_Serial"));

      idVector = N_VNew_Serial(scalarEquationsNumber);
      assert(checkAllocation(static_cast<void*>(idVector), "N_VNew_Serial"));

      tolerancesVector = N_VNew_Serial(scalarEquationsNumber);
      assert(checkAllocation(static_cast<void*>(tolerancesVector), "N_VNew_Serial"));

      if (marcoBitWidth == idaBitWidth) {
        marcoVariableValues = static_cast<void*>(N_VGetArrayPointer(variablesVector));
        marcoDerivativeValues = static_cast<void*>(N_VGetArrayPointer(derivativesVector));
      } else {
        marcoVariableValues = std::malloc(scalarEquationsNumber * (marcoBitWidth / CHAR_BIT));
        marcoDerivativeValues = std::malloc(scalarEquationsNumber * (marcoBitWidth / CHAR_BIT));
      }
    }
  }

  IDAInstance::~IDAInstance()
  {
    assert(initialized && "The IDA instance has not been initialized yet");

    if (scalarEquationsNumber != 0) {
      for (auto* variable : variables) {
        heapFree(variable);
      }

      for (auto* derivative : derivatives) {
        heapFree(derivative);
      }

      delete simulationData;

      IDAFree(&idaMemory);
      SUNLinSolFree(linearSolver);
      SUNMatDestroy(sparseMatrix);
      N_VDestroy(variablesVector);
      N_VDestroy(derivativesVector);
    }
  }

  void IDAInstance::setStartTime(double time)
  {
    startTime = time;
  }

  void IDAInstance::setEndTime(double time)
  {
    endTime = time;
  }

  void IDAInstance::setTimeStep(double step)
  {
    assert(step > 0);
    timeStep = step;
  }

  void IDAInstance::setRelativeTolerance(double tolerance)
  {
    relativeTolerance = tolerance;
  }

  void IDAInstance::setAbsoluteTolerance(double tolerance)
  {
    absoluteTolerance = tolerance;
  }

  int64_t IDAInstance::addAlgebraicVariable(void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter)
  {
    assert(!initialized && "The IDA instance has already been initialized");
    assert(variableOffsets.size() == variablesDimensions.size() + 1);

    // Add variable offset and dimensions.
    VariableDimensions varDimension(rank);
    int64_t flatSize = 1;

    for (int64_t i = 0; i < rank; ++i) {
      flatSize *= dimensions[i];
      varDimension[i] = dimensions[i];
    }

    variablesDimensions.push_back(std::move(varDimension));

    size_t offset = variableOffsets.back();
    variableOffsets.push_back(offset + flatSize);

    // Initialize derivativeValues, idValues and absoluteTolerances
    auto* idValues = N_VGetArrayPointer(idVector);
    auto* toleranceValues = N_VGetArrayPointer(tolerancesVector);

    realtype absTol = std::min(algebraicTolerance, absoluteTolerance);

    for (int64_t i = 0; i < flatSize; ++i) {
      idValues[offset + i] = 0;
      toleranceValues[offset + i] = absTol;
    }

    variables.push_back(variable);
    variablesGetters.push_back(getter);
    variablesSetters.push_back(setter);

    // Return the index of the variable.
    return variablesDimensions.size() - 1;
  }

  int64_t IDAInstance::addStateVariable(void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter)
  {
    assert(!initialized && "The IDA instance has already been initialized");
    assert(variableOffsets.size() == variablesDimensions.size() + 1);

    // Add variable offset and dimensions
    VariableDimensions variableDimensions(rank);
    int64_t flatSize = 1;

    for (int64_t i = 0; i < rank; ++i) {
      flatSize *= dimensions[i];
      variableDimensions[i] = dimensions[i];
    }

    variablesDimensions.push_back(variableDimensions);

    // Each scalar state variable has a scalar derivative
    derivativesDimensions.push_back(variableDimensions);

    // Store the position to the start of the flattened array
    size_t offset = variableOffsets.back();
    variableOffsets.push_back(offset + flatSize);

    // Initialize the derivatives, the id values and the absolute tolerances
    auto* derivativeValues = N_VGetArrayPointer(derivativesVector);
    auto* idValues = N_VGetArrayPointer(idVector);
    auto* toleranceValues = N_VGetArrayPointer(tolerancesVector);

    for (int64_t i = 0; i < flatSize; ++i) {
      derivativeValues[offset + i] = 0;
      idValues[offset + i] = 1;
      toleranceValues[offset + i] = absoluteTolerance;
    }

    variables.push_back(variable);
    variablesGetters.push_back(getter);
    variablesSetters.push_back(setter);

    derivatives.push_back(nullptr);
    derivativesGetters.push_back(nullptr);
    derivativesSetters.push_back(nullptr);

    // Return the index of the variable
    return variablesDimensions.size() - 1;
  }

  void IDAInstance::setDerivative(int64_t stateVariable, void* derivative, void* getter, void* setter)
  {
    assert(!initialized && "The IDA instance has already been initialized");
    assert(variableOffsets.size() == variablesDimensions.size() + 1);

    assert((size_t) stateVariable < derivatives.size());
    assert((size_t) stateVariable < derivativesGetters.size());
    assert((size_t) stateVariable < derivativesSetters.size());

    derivatives[stateVariable] = derivative;
    derivativesGetters[stateVariable] = getter;
    derivativesSetters[stateVariable] = setter;
  }

  int64_t IDAInstance::addEquation(int64_t* ranges, int64_t rank)
  {
    assert(!initialized && "The IDA instance has already been initialized");

    // Add the start and end dimensions of the current equation.
    EqDimension eqDimension = {};

    size_t numElements = rank * 2;

    for (size_t i = 0; i < numElements; i += 2) {
      eqDimension.push_back({ranges[i], ranges[i + 1]});
    }

    equationDimensions.push_back(eqDimension);

    // Return the index of the equation.
    return equationDimensions.size() - 1;
  }

  void IDAInstance::addVariableAccess(int64_t equationIndex, int64_t variableIndex, int64_t* access, int64_t rank)
  {
    assert(!initialized && "The IDA instance has already been initialized");

    assert(equationIndex >= 0);
    assert((size_t) equationIndex < equationDimensions.size());
    assert(variableIndex >= 0);
    assert((size_t) variableIndex < variablesDimensions.size());

    if (variableAccesses.size() <= (size_t) equationIndex) {
      variableAccesses.resize(equationIndex + 1);
    }

    auto& varAccessList = variableAccesses[equationIndex];
    varAccessList.push_back({variableIndex, {}});

    size_t numElements = rank * 2;

    for (size_t i = 0; i < numElements; i += 2) {
      varAccessList.back().second.push_back({access[i], access[i + 1]});
    }
  }

  void IDAInstance::addResidualFunction(int64_t equationIndex, void* residualFunction)
  {
    assert(!initialized && "The IDA instance has already been initialized");

    assert(equationIndex >= 0);

    if (residuals.size() <= (size_t) equationIndex) {
      residuals.resize(equationIndex + 1);
    }

    residuals[equationIndex] = residualFunction;
  }

  void IDAInstance::addJacobianFunction(int64_t equationIndex, int64_t variableIndex, void* jacobianFunction)
  {
    assert(!initialized && "The IDA instance has already been initialized");

    assert(equationIndex >= 0);
    assert(variableIndex >= 0);

    if (jacobians.size() <= (size_t) equationIndex) {
      jacobians.resize(equationIndex + 1);
    }

    if (jacobians[equationIndex].size() <= (size_t) variableIndex) {
      jacobians[equationIndex].resize(variableIndex + 1);
    }

    jacobians[equationIndex][variableIndex] = jacobianFunction;
  }

  bool IDAInstance::initialize()
  {
    assert(!initialized && "The IDA instance has already been initialized");
    initialized = true;

    currentTime = startTime;

    if (scalarEquationsNumber == 0) {
      // IDA has nothing to solve
      return true;
    }

    simulationData = new void*[variables.size() + derivatives.size()];

    for (size_t i = 0; i < variables.size(); ++i) {
      simulationData[i] = variables[i];
    }

    for (size_t i = 0; i < derivatives.size(); ++i) {
      simulationData[i + variables.size()] = derivatives[i];
    }

    copyVariablesFromMARCO(variablesVector);
    copyDerivativesFromMARCO(derivativesVector);

    // Compute the total amount of non-zero values in the Jacobian Matrix.
    computeNNZ();

    // Create and initialize IDA memory.
    idaMemory = IDACreate();

    if (!checkAllocation(idaMemory, "IDACreate")) {
      return false;
    }

    int retval = IDAInit(idaMemory, residualFunction, startTime, variablesVector, derivativesVector);

    if (!checkRetval(retval, "IDAInit")) {
      return false;
    }

    // Set tolerance and id of every scalar value.
    // The vectors are then deallocated as no longer needed inside the runtime library.
    // The IDA library, in fact, creates an internal copy of them.
    retval = IDASVtolerances(idaMemory, relativeTolerance, tolerancesVector);

    if (!checkRetval(retval, "IDASVtolerances")) {
      return false;
    }

    N_VDestroy(tolerancesVector);

    retval = IDASetId(idaMemory, idVector);

    if (!checkRetval(retval, "IDASetId")) {
      return false;
    }

    N_VDestroy(idVector);

    // Create sparse SUNMatrix for use in linear solver.
    sparseMatrix = SUNSparseMatrix(
        scalarEquationsNumber,
        scalarEquationsNumber,
        nonZeroValuesNumber,
        CSR_MAT);

    if (!checkAllocation(static_cast<void*>(sparseMatrix), "SUNSparseMatrix")) {
      return false;
    }

    // Create and attach a KLU SUNLinearSolver object.
    linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix);

    if (!checkAllocation(static_cast<void*>(linearSolver), "SUNLinSol_KLU")) {
      return false;
    }

    retval = IDASetLinearSolver(idaMemory, linearSolver, sparseMatrix);

    if (!checkRetval(retval, "IDASetLinearSolver")) {
      return false;
    }

    // Set the user-supplied Jacobian routine.
    retval = IDASetJacFn(idaMemory, jacobianMatrix);

    if (!checkRetval(retval, "IDASetJacFn")) {
      return false;
    }

    // Add the remaining mandatory parameters.
    retval = IDASetUserData(idaMemory, static_cast<void*>(this));

    if (!checkRetval(retval, "IDASetUserData")) {
      return false;
    }

    retval = IDASetStopTime(idaMemory, endTime);

    if (!checkRetval(retval, "IDASetStopTime")) {
      return false;
    }

    // Add the remaining optional parameters.
    retval = IDASetInitStep(idaMemory, initTimeStep);

    if (!checkRetval(retval, "IDASetInitStep")) {
      return false;
    }

    retval = IDASetMaxStep(idaMemory, endTime);

    if (!checkRetval(retval, "IDASetMaxStep")) {
      return false;
    }

    retval = IDASetSuppressAlg(idaMemory, suppressAlg);

    if (!checkRetval(retval, "IDASetSuppressAlg")) {
      return false;
    }

    // Increase the maximum number of iterations taken by IDA before failing.
    retval = IDASetMaxNumSteps(idaMemory, maxNumSteps);

    if (!checkRetval(retval, "IDASetMaxNumSteps")) {
      return false;
    }

    retval = IDASetMaxErrTestFails(idaMemory, maxErrTestFail);

    if (!checkRetval(retval, "IDASetMaxErrTestFails")) {
      return false;
    }

    retval = IDASetMaxNonlinIters(idaMemory, maxNonlinIters);

    if (!checkRetval(retval, "IDASetMaxNonlinIters")) {
      return false;
    }

    retval = IDASetMaxConvFails(idaMemory, maxConvFails);

    if (!checkRetval(retval, "IDASetMaxConvFails")) {
      return false;
    }

    retval = IDASetNonlinConvCoef(idaMemory, nonlinConvCoef);

    if (!checkRetval(retval, "IDASetNonlinConvCoef")) {
      return false;
    }

    // Increase the maximum number of iterations taken by IDA IC before failing.
    retval = IDASetMaxNumStepsIC(idaMemory, maxNumStepsIC);

    if (!checkRetval(retval, "IDASetMaxNumStepsIC")) {
      return false;
    }

    retval = IDASetMaxNumJacsIC(idaMemory, maxNumJacsIC);

    if (!checkRetval(retval, "IDASetMaxNumJacsIC")) {
      return false;
    }

    retval = IDASetMaxNumItersIC(idaMemory, maxNumItersIC);

    if (!checkRetval(retval, "IDASetMaxNumItersIC")) {
      return false;
    }

    retval = IDASetNonlinConvCoefIC(idaMemory, nonlinConvCoefIC);

    if (!checkRetval(retval, "IDASetNonlinConvCoefIC")) {
      return false;
    }

    retval = IDASetLineSearchOffIC(idaMemory, lineSearchOff);

    if (!checkRetval(retval, "IDASetLineSearchOffIC")) {
      return false;
    }

    // Call IDACalcIC to correct the initial values.
    realtype firstOutTime = (endTime - startTime) / timeScalingFactorInit;

#ifdef MARCO_PROFILING
    profiler().initialConditionsTimer.start();
#endif

    std::cerr << "Starting IDACalcIC\n";
    retval = IDACalcIC(idaMemory, IDA_YA_YDP_INIT, firstOutTime);
    std::cerr << "Finished IDACalcIC\n";

#ifdef MARCO_PROFILING
    profiler().initialConditionsTimer.stop();
#endif

    if (!checkRetval(retval, "IDACalcIC")) {
      return false;
    }

    return true;
  }

  bool IDAInstance::step()
  {
    std::cerr << "IDA step\n";

    assert(initialized && "The IDA instance has not been initialized yet");
    bool equidistantTimeGrid = timeStep != kUndefinedTimeStep;

    if (scalarEquationsNumber == 0) {
      // IDA has nothing to solve. Just increment the time.

      if (timeStep == kUndefinedTimeStep) {
        currentTime = endTime;
      } else {
        currentTime += timeStep;
      }

      return true;
    }

#ifdef MARCO_PROFILING
    profiler().stepsTimer.start();
#endif

    // Execute one step
    int retval = IDASolve(
        idaMemory,
        equidistantTimeGrid ? (currentTime + timeStep) : endTime,
        &currentTime,
        variablesVector,
        derivativesVector,
        equidistantTimeGrid ? IDA_NORMAL : IDA_ONE_STEP);

#ifdef MARCO_PROFILING
    profiler().stepsTimer.stop();
#endif

    if (marcoBitWidth != idaBitWidth) {
      auto* variableValues = N_VGetArrayPointer(variablesVector);
      auto* derivativeValues = N_VGetArrayPointer(derivativesVector);

      for (int64_t i = 0; i < scalarEquationsNumber; ++i) {
        if (marcoBitWidth == 32) {
          static_cast<float*>(marcoVariableValues)[i] = static_cast<float>(variableValues[i]);
          static_cast<float*>(marcoDerivativeValues)[i] = static_cast<float>(derivativeValues[i]);
        } else {
          static_cast<double*>(marcoVariableValues)[i] = static_cast<double>(variableValues[i]);
          static_cast<double*>(marcoDerivativeValues)[i] = static_cast<double>(derivativeValues[i]);
        }
      }
    }

    // Check if the solver failed
    if (!checkRetval(retval, "IDASolve")) {
      return false;
    }

    return true;
  }

  realtype IDAInstance::getCurrentTime() const
  {
    return currentTime;
  }

  void* IDAInstance::getVariable(int64_t variableIndex) const
  {
    assert(variableIndex >= 0);
    assert(static_cast<size_t>(variableIndex) < variablesDimensions.size());

    size_t offset = variableOffsets[variableIndex];

    if (marcoBitWidth == 32) {
      return static_cast<void*>(&static_cast<float*>(marcoVariableValues)[offset]);
    }

    return static_cast<void*>(&static_cast<double*>(marcoVariableValues)[offset]);
  }

  void* IDAInstance::getDerivative(int64_t derivativeIndex) const
  {
    assert(derivativeIndex >= 0);
    assert(static_cast<size_t>(derivativeIndex) < variablesDimensions.size());

    size_t offset = variableOffsets[derivativeIndex];

    if (marcoBitWidth == 32) {
      return static_cast<void*>(&static_cast<float*>(marcoDerivativeValues)[offset]);
    }

    return static_cast<void*>(&static_cast<double*>(marcoDerivativeValues)[offset]);
  }

  int IDAInstance::residualFunction(realtype time, N_Vector variables, N_Vector derivatives, N_Vector residuals, void* userData)
  {
    realtype* rval = N_VGetArrayPointer(residuals);
    auto* instance = static_cast<IDAInstance*>(userData);

    instance->copyVariablesIntoMARCO(variables);
    instance->copyDerivativesIntoMARCO(derivatives);

    // For every vector equation
    for (size_t eq = 0; eq < instance->equationDimensions.size(); ++eq) {
      // Initialize the multidimensional interval of the vector equation
      size_t equationIndices[instance->equationDimensions[eq].size()];

      for (size_t i = 0; i < instance->equationDimensions[eq].size(); i++) {
        equationIndices[i] = instance->equationDimensions[eq][i].first;
      }

      // For every scalar equation in the vector equation
      do {
        // Compute the i-th residual function
        if (instance->marcoBitWidth == 32) {
          auto residualFunction = reinterpret_cast<ResidualFunction<float>>(instance->residuals[eq]);
          auto residualFunctionResult = residualFunction(time, instance->simulationData, equationIndices);
          std::cerr << "Residual function (with time " << time << "): " << residualFunctionResult << "\n";
          *rval++ = residualFunctionResult;
        } else {
          auto residualFunction = reinterpret_cast<ResidualFunction<double>>(instance->residuals[eq]);
          auto residualFunctionResult = residualFunction(time, instance->simulationData, equationIndices);
          std::cerr << "Residual function (with time " << time << "): " << residualFunctionResult << "\n";
          *rval++ = residualFunctionResult;
        }
      } while (updateIndexes(equationIndices, instance->equationDimensions[eq]));
    }

    assert(rval == N_VGetArrayPointer(residuals) + instance->scalarEquationsNumber);

    return IDA_SUCCESS;
  }

  int IDAInstance::jacobianMatrix(
      realtype time, realtype alpha,
      N_Vector variables, N_Vector derivatives, N_Vector residuals,
      SUNMatrix jacobianMatrix,
      void* userData,
      N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
  {
    sunindextype* rowptrs = SUNSparseMatrix_IndexPointers(jacobianMatrix);
    sunindextype* colvals = SUNSparseMatrix_IndexValues(jacobianMatrix);
    realtype* jacobian = SUNSparseMatrix_Data(jacobianMatrix);

    auto* instance = static_cast<IDAInstance*>(userData);

    instance->copyVariablesIntoMARCO(variables);
    instance->copyDerivativesIntoMARCO(derivatives);

    sunindextype nnzElements = 0;
    *rowptrs++ = nnzElements;

    // For every vector equation
    for (size_t eq = 0; eq < instance->equationDimensions.size(); ++eq) {
      // Initialize the multidimensional interval of the vector equation
      size_t equationIndices[instance->equationDimensions[eq].size()];

      for (size_t i = 0; i < instance->equationDimensions[eq].size(); ++i) {
        equationIndices[i] = instance->equationDimensions[eq][i].first;
      }

      // For every scalar equation in the vector equation
      do {
        // Compute the column indexes that may be non-zeros
        std::set<DerivativeVariable> columnIndexesSet = instance->computeIndexSet(eq, equationIndices);

        nnzElements += columnIndexesSet.size();
        *rowptrs++ = nnzElements;

        // For every variable with respect to which every equation must be
        // partially differentiated
        for (DerivativeVariable var: columnIndexesSet) {
          // Compute the i-th Jacobian value
          size_t* variableIndices = &var.second[0];

          if (instance->marcoBitWidth == 32) {
            auto jacobianFunction = reinterpret_cast<JacobianFunction<float>>(instance->jacobians[eq][var.first]);
            auto jacobianFunctionResult = jacobianFunction(time, instance->simulationData, equationIndices, variableIndices, alpha);
            std::cerr << "Jacobian function (with time " << time << ", alpha " << alpha << "): " << jacobianFunctionResult << "\n";
            *jacobian++ = jacobianFunctionResult;
          } else {
            auto jacobianFunction = reinterpret_cast<JacobianFunction<double>>(instance->jacobians[eq][var.first]);
            auto jacobianFunctionResult = jacobianFunction(time, instance->simulationData, equationIndices, variableIndices, alpha);
            std::cerr << "Jacobian function (with time " << time << ", alpha " << alpha << "): " << jacobianFunctionResult << "\n";
            *jacobian++ = jacobianFunctionResult;
          }

          *colvals++ = instance->variableOffsets[var.first] + computeOffset(instance->variablesDimensions[var.first], var.second);
        }
      } while (updateIndexes(equationIndices, instance->equationDimensions[eq]));
    }

    assert(rowptrs == SUNSparseMatrix_IndexPointers(jacobianMatrix) + instance->scalarEquationsNumber + 1);
    assert(colvals == SUNSparseMatrix_IndexValues(jacobianMatrix) + instance->nonZeroValuesNumber);
    assert(jacobian == SUNSparseMatrix_Data(jacobianMatrix) + instance->nonZeroValuesNumber);

    return IDA_SUCCESS;
  }

  /// Compute the column indexes of the current row of the Jacobian Matrix given
  /// the current vector equation and an array of indexes.
  std::set<DerivativeVariable> IDAInstance::computeIndexSet(size_t eq, size_t* eqIndexes) const
  {
    std::set<DerivativeVariable> columnIndexesSet;

    for (auto& access : variableAccesses[eq]) {
      size_t variableIndex = access.first;
      Access variableAccess = access.second;
      assert(variableAccess.size() == variablesDimensions[variableIndex].rank());

      DerivativeVariable newEntry = {variableIndex, {}};

      for (size_t i = 0; i < variableAccess.size(); ++i) {
        int64_t induction = variableAccess[i].first;
        size_t index = induction != -1 ? eqIndexes[induction] : 0;
        index += variableAccess[i].second;
        newEntry.second.push_back(index);
      }

      columnIndexesSet.insert(newEntry);
    }

    return columnIndexesSet;
  }

  /// Compute the number of non-zero values in the Jacobian Matrix. Also compute
  /// the column indexes of all non-zero values in the Jacobian Matrix. This avoids
  /// the recomputation of such indexes during the Jacobian evaluation.
  void IDAInstance::computeNNZ()
  {
    nonZeroValuesNumber = 0;

    for (size_t eq = 0; eq < equationDimensions.size(); ++eq) {
      // Initialize the multidimensional interval of the vector equation
      size_t indexes[equationDimensions[eq].size()];

      for (size_t i = 0; i < equationDimensions[eq].size(); ++i) {
        indexes[i] = equationDimensions[eq][i].first;
      }

      // For every scalar equation in the vector equation
      do {
        // Compute the column indexes that may be non-zeros
        nonZeroValuesNumber += computeIndexSet(eq, indexes).size();
      } while (updateIndexes(indexes, equationDimensions[eq]));
    }
  }

  void IDAInstance::copyVariablesFromMARCO(N_Vector values)
  {
    assert(variables.size() == variablesDimensions.size());
    assert(variables.size() == variablesGetters.size());

    auto* valuesPtr = N_VGetArrayPointer(values);

    for (size_t i = 0; i < variables.size(); ++i) {
      auto* descriptor = variables[i];
      const auto& dimensions = variablesDimensions[i];
      assert(variablesGetters[i] != nullptr);

      for (auto indices = dimensions.indicesBegin(), end = dimensions.indicesEnd(); indices != end; ++indices) {
        if (marcoBitWidth == 32) {
          auto getterFn = reinterpret_cast<VariableGetterFunction<float>>(variablesGetters[i]);
          auto value = static_cast<realtype>(getterFn(descriptor, *indices));
          *valuesPtr = value;
        } else {
          auto getterFn = reinterpret_cast<VariableGetterFunction<double>>(variablesGetters[i]);
          auto value = static_cast<realtype>(getterFn(descriptor, *indices));
          *valuesPtr = value;
        }

        ++valuesPtr;
      }
    }
  }

  void IDAInstance::copyDerivativesFromMARCO(N_Vector values)
  {
    assert(derivatives.size() == derivativesDimensions.size());
    assert(derivatives.size() == derivativesGetters.size());

    auto* valuesPtr = N_VGetArrayPointer(values);

    for (size_t i = 0; i < derivatives.size(); ++i) {
      auto* descriptor = derivatives[i];
      const auto& dimensions = derivativesDimensions[i];
      assert(derivativesGetters[i] != nullptr);

      for (auto indices = dimensions.indicesBegin(), end = dimensions.indicesEnd(); indices != end; ++indices) {
        if (marcoBitWidth == 32) {
          auto getterFn = reinterpret_cast<VariableGetterFunction<float>>(derivativesGetters[i]);
          auto value = static_cast<realtype>(getterFn(descriptor, *indices));
          *valuesPtr = value;
        } else {
          auto getterFn = reinterpret_cast<VariableGetterFunction<double>>(derivativesGetters[i]);
          auto value = static_cast<realtype>(getterFn(descriptor, *indices));
          *valuesPtr = value;
        }

        ++valuesPtr;
      }
    }
  }

  void IDAInstance::copyVariablesIntoMARCO(N_Vector values)
  {
    assert(variables.size() == variablesDimensions.size());
    assert(variables.size() == variablesSetters.size());

    auto* valuesPtr = N_VGetArrayPointer(values);

    for (size_t i = 0; i < variables.size(); ++i) {
      auto* descriptor = variables[i];
      const auto& dimensions = variablesDimensions[i];
      assert(variablesSetters[i] != nullptr);

      for (auto indices = dimensions.indicesBegin(), end = dimensions.indicesEnd(); indices != end; ++indices) {
        if (marcoBitWidth == 32) {
          auto setterFn = reinterpret_cast<VariableSetterFunction<float>>(variablesSetters[i]);
          auto value = static_cast<float>(*valuesPtr);
          std::cerr << "Calling variable setter with value " << value << "\n";
          setterFn(descriptor, value, *indices);
        } else {
          auto setterFn = reinterpret_cast<VariableSetterFunction<double>>(variablesSetters[i]);
          auto value = static_cast<double>(*valuesPtr);
          std::cerr << "Calling variable setter with value " << value << "\n";
          setterFn(descriptor, value, *indices);
        }

        ++valuesPtr;
      }
    }
  }

  void IDAInstance::copyDerivativesIntoMARCO(N_Vector values)
  {
    assert(derivatives.size() == derivativesDimensions.size());
    assert(derivatives.size() == derivativesSetters.size());

    auto* valuesPtr = N_VGetArrayPointer(values);

    for (size_t i = 0; i < derivatives.size(); ++i) {
      auto* descriptor = derivatives[i];
      const auto& dimensions = derivativesDimensions[i];
      assert(variablesSetters[i] != nullptr);

      for (auto indices = dimensions.indicesBegin(), end = dimensions.indicesEnd(); indices != end; ++indices) {
        if (marcoBitWidth == 32) {
          auto setterFn = reinterpret_cast<VariableSetterFunction<float>>(derivativesSetters[i]);
          auto value = static_cast<float>(*valuesPtr);
          std::cerr << "Calling derivative setter with value " << value << "\n";
          setterFn(descriptor, value, *indices);
        } else {
          auto setterFn = reinterpret_cast<VariableSetterFunction<double>>(derivativesSetters[i]);
          auto value = static_cast<double>(*valuesPtr);
          std::cerr << "Calling derivative setter with value " << value << "\n";
          setterFn(descriptor, value, *indices);
        }

        ++valuesPtr;
      }
    }
  }

  void IDAInstance::printStatistics() const
  {
    if (scalarEquationsNumber == 0) {
      return;
    }

    if (printJacobian) {
      printIncidenceMatrix();
    }

    long nst, nre, nje, nni, nli, netf, nncf;
    realtype ais, ls;

    IDAGetNumSteps(idaMemory, &nst);
    IDAGetNumResEvals(idaMemory, &nre);
    IDAGetNumJacEvals(idaMemory, &nje);
    IDAGetNumNonlinSolvIters(idaMemory, &nni);
    IDAGetNumLinIters(idaMemory, &nli);
    IDAGetNumErrTestFails(idaMemory, &netf);
    IDAGetNumNonlinSolvConvFails(idaMemory, &nncf);
    IDAGetActualInitStep(idaMemory, &ais);
    IDAGetLastStep(idaMemory, &ls);

    std::cerr << std::endl << "Final Run Statistics:" << std::endl;

    std::cerr << "Number of vector equations       = ";
    std::cerr << equationDimensions.size() << std::endl;
    std::cerr << "Number of scalar equations       = ";
    std::cerr << scalarEquationsNumber << std::endl;
    std::cerr << "Number of non-zero values        = ";
    std::cerr << nonZeroValuesNumber << std::endl;

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

  void IDAInstance::printIncidenceMatrix() const
  {
    std::cerr << std::endl;

    // For every vector equation
    for (size_t eq = 0; eq < equationDimensions.size(); ++eq) {
      // Initialize the multidimensional interval of the vector equation
      size_t indexes[equationDimensions[eq].size()];

      for (size_t i = 0; i < equationDimensions[eq].size(); ++i) {
        indexes[i] = equationDimensions[eq][i].first;
      }

      // For every scalar equation in the vector equation
      do {
        std::cerr << "│";

        // Get the column indexes that may be non-zeros.
        std::set<size_t> columnIndexesSet;

        for (auto& access : variableAccesses[eq]) {
          const VariableDimensions& dimensions = variablesDimensions[access.first];
          sunindextype varOffset = computeOffset(indexes, dimensions, access.second);
          columnIndexesSet.insert(variableOffsets[access.first] + varOffset);
        }

        for (int64_t i = 0; i < scalarEquationsNumber; ++i) {
          if (columnIndexesSet.find(i) != columnIndexesSet.end()) {
            std::cerr << "*";
          } else {
            std::cerr << " ";
          }

          if (i < scalarEquationsNumber - 1) {
            std::cerr << " ";
          }
        }

        std::cerr << "│" << std::endl;
      } while (updateIndexes(indexes, equationDimensions[eq]));
    }
  }
}

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

/// Instantiate and initialize the struct of data needed by IDA, given the total
/// number of scalar equations.

static void* idaCreate_pvoid(int64_t scalarEquationsNumber, int64_t bitWidth)
{
  auto* instance = new IDAInstance(bitWidth, scalarEquationsNumber);
  return static_cast<void*>(instance);
}

RUNTIME_FUNC_DEF(idaCreate, PTR(void), int64_t, int64_t)

static void idaInit_void(void* userData)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  [[maybe_unused]] bool result = instance->initialize();
  assert(result && "Can't initialize the IDA instance");
}

RUNTIME_FUNC_DEF(idaInit, void, PTR(void))

static void idaStep_void(void* userData)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  [[maybe_unused]] bool result = instance->step();
  assert(result && "IDA step failed");
}

RUNTIME_FUNC_DEF(idaStep, void, PTR(void))

static void idaFree_void(void* userData)
{
  auto* data = static_cast<IDAInstance*>(userData);
  delete data;
}

RUNTIME_FUNC_DEF(idaFree, void, PTR(void))

static void idaSetStartTime_void(void* userData, double startTime)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->setStartTime(startTime);
}

RUNTIME_FUNC_DEF(idaSetStartTime, void, PTR(void), double)

static void idaSetEndTime_void(void* userData, double endTime)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->setEndTime(endTime);
}

RUNTIME_FUNC_DEF(idaSetEndTime, void, PTR(void), double)

static void idaSetTimeStep_void(void* userData, double timeStep)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->setTimeStep(timeStep);
}

RUNTIME_FUNC_DEF(idaSetTimeStep, void, PTR(void), double)

static void idaSetRelativeTolerance_void(void* userData, double relativeTolerance)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->setRelativeTolerance(relativeTolerance);
}

RUNTIME_FUNC_DEF(idaSetRelativeTolerance, void, PTR(void), double)

static void idaSetAbsoluteTolerance_void(void* userData, double absoluteTolerance)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->setAbsoluteTolerance(absoluteTolerance);
}

RUNTIME_FUNC_DEF(idaSetAbsoluteTolerance, void, PTR(void), double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

static int64_t idaAddEquation_i64(void* userData, int64_t* ranges, int64_t rank)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  return instance->addEquation(ranges, rank);
}

RUNTIME_FUNC_DEF(idaAddEquation, int64_t, PTR(void), PTR(int64_t), int64_t)

static void idaAddResidual_void(void* userData, int64_t equationIndex, void* residualFunction)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->addResidualFunction(equationIndex, residualFunction);
}

RUNTIME_FUNC_DEF(idaAddResidual, void, PTR(void), int64_t, PTR(void))

static void idaAddJacobian_void(void* userData, int64_t equationIndex, int64_t variableIndex, void* jacobianFunction)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->addJacobianFunction(equationIndex, variableIndex, jacobianFunction);
}

RUNTIME_FUNC_DEF(idaAddJacobian, void, PTR(void), int64_t, int64_t, PTR(void))

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

static int64_t idaAddAlgebraicVariable_i64(void* userData, void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  return instance->addAlgebraicVariable(variable, dimensions, rank, getter, setter);
}

RUNTIME_FUNC_DEF(idaAddAlgebraicVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

static int64_t idaAddStateVariable_i64(void* userData, void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  return instance->addStateVariable(variable, dimensions, rank, getter, setter);
}

RUNTIME_FUNC_DEF(idaAddStateVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

static void idaSetDerivative_void(void* userData, int64_t stateVariable, void* derivative, void* getter, void* setter)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->setDerivative(stateVariable, derivative, getter, setter);
}

RUNTIME_FUNC_DEF(idaSetDerivative, void, PTR(void), int64_t, PTR(void), PTR(void), PTR(void))

/// Add a variable access to the var-th variable, where ind is the induction
/// variable and off is the access offset.
static void idaAddVariableAccess_void(
    void* userData, int64_t equationIndex, int64_t variableIndex, int64_t* access, int64_t rank)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->addVariableAccess(equationIndex, variableIndex, access, rank);
}

RUNTIME_FUNC_DEF(idaAddVariableAccess, void, PTR(void), int64_t, int64_t, PTR(int64_t), int64_t)

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

/// Returns the pointer to the start of the memory of the requested variable.
static void* idaGetVariable_pvoid(void* userData, int64_t variableIndex)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  return instance->getVariable(variableIndex);
}

RUNTIME_FUNC_DEF(idaGetVariable, PTR(void), PTR(void), int64_t)

/// Returns the pointer to the start of the memory of the requested derivative.
static void* idaGetDerivative_pvoid(void* userData, int64_t derivativeIndex)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  return instance->getDerivative(derivativeIndex);
}

RUNTIME_FUNC_DEF(idaGetDerivative, PTR(void), PTR(void), int64_t)

static float idaGetCurrentTime_f32(void* userData)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  return static_cast<float>(instance->getCurrentTime());
}

static double idaGetCurrentTime_f64(void* userData)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  return static_cast<double>(instance->getCurrentTime());
}

RUNTIME_FUNC_DEF(idaGetCurrentTime, float, PTR(void))

RUNTIME_FUNC_DEF(idaGetCurrentTime, double, PTR(void))

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static void printStatistics_void(void* userData)
{
  auto* instance = static_cast<IDAInstance*>(userData);
  instance->printStatistics();
}

RUNTIME_FUNC_DEF(printStatistics, void, PTR(void))
