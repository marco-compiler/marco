#include "marco/Runtime/Solvers/KINSOL/Instance.h"
#include "marco/Runtime/Solvers/KINSOL/Options.h"
#include "marco/Runtime/Solvers/KINSOL/Profiler.h"
#include "marco/Runtime/Support/MemoryManagement.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <set>

using namespace marco::runtime::kinsol;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace marco::runtime::kinsol
{
  VariableDimensions::VariableDimensions(size_t rank)
  {
    dimensions.resize(rank, 0);
  }

  size_t VariableDimensions::rank() const
  {
    return dimensions.size();
  }

  size_t& VariableDimensions::operator[](size_t index)
  {
    return dimensions[index];
  }

  const size_t& VariableDimensions::operator[](size_t index) const
  {
    return dimensions[index];
  }

  VariableDimensions::const_iterator VariableDimensions::begin() const
  {
    assert(isValid() && "Variable dimensions have not been initialized");
    return dimensions.begin();
  }

  VariableDimensions::const_iterator VariableDimensions::end() const
  {
    assert(isValid() && "Variable dimensions have not been initialized");
    return dimensions.end();
  }

  VariableIndicesIterator VariableDimensions::indicesBegin() const
  {
    assert(isValid() && "Variable dimensions have not been initialized");
    return VariableIndicesIterator::begin(*this);
  }

  VariableIndicesIterator VariableDimensions::indicesEnd() const
  {
    assert(isValid() && "Variable dimensions have not been initialized");
    return VariableIndicesIterator::end(*this);
  }

  bool VariableDimensions::isValid() const
  {
    return std::none_of(dimensions.begin(), dimensions.end(), [](const auto& dimension) {
      return dimension == 0;
    });
  }

  VariableIndicesIterator::~VariableIndicesIterator()
  {
    delete indices;
  }

  VariableIndicesIterator VariableIndicesIterator::begin(const VariableDimensions& dimensions)
  {
    VariableIndicesIterator result(dimensions);

    for (size_t i = 0; i < dimensions.rank(); ++i) {
      result.indices[i] = 0;
    }

    return result;
  }

  VariableIndicesIterator VariableIndicesIterator::end(const VariableDimensions& dimensions)
  {
    VariableIndicesIterator result(dimensions);

    for (size_t i = 0; i < dimensions.rank(); ++i) {
      result.indices[i] = dimensions[i];
    }

    return result;
  }

  bool VariableIndicesIterator::operator==(const VariableIndicesIterator& it) const
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

  bool VariableIndicesIterator::operator!=(const VariableIndicesIterator& it) const
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

  VariableIndicesIterator& VariableIndicesIterator::operator++()
  {
    fetchNext();
    return *this;
  }

  VariableIndicesIterator VariableIndicesIterator::operator++(int)
  {
    auto temp = *this;
    fetchNext();
    return temp;
  }

  size_t* VariableIndicesIterator::operator*() const
  {
    return indices;
  }

  VariableIndicesIterator::VariableIndicesIterator(const VariableDimensions& dimensions) : dimensions(&dimensions)
  {
    indices = new size_t[dimensions.rank()];
  }

  void VariableIndicesIterator::fetchNext()
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

//===---------------------------------------------------------------------===//
// Solver
//===---------------------------------------------------------------------===//

namespace marco::runtime::kinsol
{
  KINSOLInstance::KINSOLInstance(int64_t marcoBitWidth, int64_t scalarEquationsNumber)
      : initialized(false),
        marcoBitWidth(marcoBitWidth),
        scalarEquationsNumber(scalarEquationsNumber)
  {
    SUNContext_Create(nullptr, &ctx);

    variableOffsets.push_back(0);

    // Create and initialize the required N-vectors for the variables.
    variablesVector = N_VNew_Serial(scalarEquationsNumber, ctx);
    assert(checkAllocation(static_cast<void*>(variablesVector), "N_VNew_Serial"));

    derivativesVector = N_VNew_Serial(scalarEquationsNumber, ctx);
    assert(checkAllocation(static_cast<void*>(derivativesVector), "N_VNew_Serial"));

    idVector = N_VNew_Serial(scalarEquationsNumber, ctx);
    assert(checkAllocation(static_cast<void*>(idVector), "N_VNew_Serial"));

    tolerancesVector = N_VNew_Serial(scalarEquationsNumber, ctx);
    assert(checkAllocation(static_cast<void*>(tolerancesVector), "N_VNew_Serial"));

    variableScaleVector = N_VNew_Serial(scalarEquationsNumber, ctx);
    assert(checkAllocation(static_cast<void*>(variableScaleVector), "N_VNew_Serial"));

    residualScaleVector = N_VNew_Serial(scalarEquationsNumber, ctx);
    assert(checkAllocation(static_cast<void*>(residualScaleVector), "N_VNew_Serial"));
  }

  KINSOLInstance::~KINSOLInstance()
  {
    assert(initialized && "The KINSOL instance has not been initialized yet");

    for (auto* variable : parameters) {
      heapFree(variable);
    }

    for (auto* variable : variables) {
      heapFree(variable);
    }

    for (auto* derivative : derivatives) {
      heapFree(derivative);
    }

    delete simulationData;

    KINFree(&kinsolMemory);
    SUNLinSolFree(linearSolver);
    SUNMatDestroy(sparseMatrix);
    N_VDestroy(variablesVector);
    N_VDestroy(derivativesVector);
    N_VDestroy(idVector);
    N_VDestroy(tolerancesVector);
  }

  int64_t KINSOLInstance::addAlgebraicVariable(void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter)
  {
    assert(!initialized && "The KINSOL instance has already been initialized");
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
    auto* derivativeValues = N_VGetArrayPointer(derivativesVector);
    auto* idValues = N_VGetArrayPointer(idVector);
    auto* toleranceValues = N_VGetArrayPointer(tolerancesVector);

    realtype absTol = std::min(getOptions().algebraicTolerance, getOptions().absoluteTolerance);

    for (int64_t i = 0; i < flatSize; ++i) {
      derivativeValues[offset + i] = 0;
      idValues[offset + i] = 0;
      toleranceValues[offset + i] = absTol;
    }

    variables.push_back(variable);
    variablesGetters.push_back(getter);
    variablesSetters.push_back(setter);

    // Return the index of the variable.
    return variablesDimensions.size() - 1;
  }

  int64_t KINSOLInstance::addStateVariable(void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter)
  {
    assert(!initialized && "The KINSOL instance has already been initialized");
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
      toleranceValues[offset + i] = getOptions().absoluteTolerance;
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

  void KINSOLInstance::setDerivative(int64_t stateVariable, void* derivative, void* getter, void* setter)
  {
    assert(!initialized && "The KINSOL instance has already been initialized");
    assert(variableOffsets.size() == variablesDimensions.size() + 1);

    assert((size_t) stateVariable < derivatives.size());
    assert((size_t) stateVariable < derivativesGetters.size());
    assert((size_t) stateVariable < derivativesSetters.size());

    derivatives[stateVariable] = derivative;
    derivativesGetters[stateVariable] = getter;
    derivativesSetters[stateVariable] = setter;
  }

  int64_t KINSOLInstance::addEquation(int64_t* ranges, int64_t rank)
  {
    assert(!initialized && "The KINSOL instance has already been initialized");

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

  void KINSOLInstance::addVariableAccess(int64_t equationIndex, int64_t variableIndex, int64_t* access, int64_t rank)
  {
    assert(!initialized && "The KINSOL instance has already been initialized");

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

  void KINSOLInstance::addResidualFunction(int64_t equationIndex, void* residualFunction)
  {
    assert(!initialized && "The KINSOL instance has already been initialized");

    assert(equationIndex >= 0);

    if (residuals.size() <= (size_t) equationIndex) {
      residuals.resize(equationIndex + 1);
    }

    residuals[equationIndex] = residualFunction;
  }

  void KINSOLInstance::addJacobianFunction(int64_t equationIndex, int64_t variableIndex, void* jacobianFunction)
  {
    assert(!initialized && "The KINSOL instance has already been initialized");

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

  bool KINSOLInstance::initialize()
  {
    assert(!initialized && "The KINSOL instance has already been initialized");
    initialized = true;

    if (scalarEquationsNumber == 0) {
      // KINSOL has nothing to solve
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

    auto* variableScaleValues = N_VGetArrayPointer(variableScaleVector);
    for(int i = 0; i < scalarEquationsNumber; ++i) {
      variableScaleValues[i] = 1.0;
    }

    auto* residualScaleValues = N_VGetArrayPointer(residualScaleVector);
    for(int i = 0; i < scalarEquationsNumber; ++i) {
      residualScaleValues[i] = 1.0;
    }

    // Compute the total amount of non-zero values in the Jacobian Matrix.
    computeNNZ();

    // Create and initialize KINSOL memory.
    kinsolMemory = KINCreate(ctx);

    if (!checkAllocation(kinsolMemory, "KINCreate")) {
      return false;
    }

    if (!kinsolInit()) {
      return false;
    }

    // Create sparse SUNMatrix for use in linear solver.
    sparseMatrix = SUNSparseMatrix(
        scalarEquationsNumber,
        scalarEquationsNumber,
        nonZeroValuesNumber,
        CSR_MAT,
        ctx);

    if (!checkAllocation(static_cast<void*>(sparseMatrix), "SUNSparseMatrix")) {
      return false;
    }

    // Create and attach a KLU SUNLinearSolver object.
    linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix, ctx);

    if (!checkAllocation(static_cast<void*>(linearSolver), "SUNLinSol_KLU")) {
      return false;
    }

    if (!kinsolSetLinearSolver()) {
      return false;
    }

    if (!kinsolSetUserData() ||
        !kinsolSetJacobianFunction()) {
      return false;
    }

    if (!kinsolFNTolerance()) {
      return false;
    }

    return true;
  }

  bool KINSOLInstance::step()
  {
    assert(initialized && "The KINSOL instance has not been initialized yet");

    if (scalarEquationsNumber == 0) {
      return true;
    }

    KINSOL_PROFILER_STEP_START;

    // Execute one step
    auto solveRetVal = KINSol(
        kinsolMemory,
        variablesVector,
        KIN_LINESEARCH,
        variableScaleVector,
        residualScaleVector);

    KINSOL_PROFILER_STEP_STOP;

    if (solveRetVal != KIN_SUCCESS) {
      if (solveRetVal == KIN_INITIAL_GUESS_OK) {
        std::cerr << "KINSol - The guess u satisfied the system F(u) = 0" << std::endl;
      } else if (solveRetVal == KIN_STEP_LT_STPTOL) {
        std::cerr << "KINSol - KINSOL stopped based on scaled step length" << std::endl;
      } else if (solveRetVal == KIN_MEM_NULL) {
        std::cerr << "KINSol - The kinsol_mem pointer is NULL" << std::endl;
      } else if (solveRetVal == KIN_ILL_INPUT) {
        std::cerr << "KINSol - One of the inputs to KINSol was illegal, or some other input to the solver was either illegal or missing" << std::endl;
      } else if (solveRetVal == KIN_NO_MALLOC) {
        std::cerr << "KINSol - The KINSOL memory was not allocated by a call to KINCreate()" << std::endl;
      } else if (solveRetVal == KIN_MEM_FAIL) {
        std::cerr << "KINSol - A memory allocation failed" << std::endl;
      } else if (solveRetVal == KIN_LINESEARCH_NONCONV) {
        std::cerr << "KINSol - The line search algorithm was unable to find an iterate sufficiently distinct from the current iterate" << std::endl;
      } else if (solveRetVal == KIN_MAXITER_REACHED) {
        std::cerr << "KINSol - The maximum number of nonlinear iterations has been reached" << std::endl;
      } else if (solveRetVal == KIN_MXNEWT_5X_EXCEEDED) {
        std::cerr << "KINSol - Five consecutive steps have been taken that satisfy the inequality" << std::endl;
      } else if (solveRetVal == KIN_LINESEARCH_BCFAIL) {
        std::cerr << "KINSol - The line search algorithm was unable to satisfy the beta-condition for MXNBCF+1 iterations" << std::endl;
      } else if (solveRetVal == KIN_LINSOLV_NO_RECOVERY) {
        std::cerr << "KINSol - The user-supplied routine psolve encountered a recoverable error, but the preconditioner is already current" << std::endl;
      } else if (solveRetVal == KIN_LSETUP_FAIL) {
        std::cerr << "KINSol - The linear solverâ€™s setup function failed in an unrecoverable manner" << std::endl;
      } else if (solveRetVal == KIN_LSOLVE_FAIL) {
        std::cerr << "KINSol - The linear solverâ€™s solve function failed in an unrecoverable manner" << std::endl;
      } else if (solveRetVal == KIN_SYSFUNC_FAIL) {
        std::cerr << "KINSol - The system function failed in an unrecoverable manner" << std::endl;
      } else if (solveRetVal == KIN_FIRST_SYSFUNC_ERR) {
        std::cerr << "KINSol - The system function failed recoverably at the first call" << std::endl;
      } else if (solveRetVal == KIN_REPTD_SYSFUNC_ERR) {
        std::cerr << "KINSol - The system function had repeated recoverable errors. No recovery is possible." << std::endl;
      }

      return false;
    }

    copyVariablesIntoMARCO(variablesVector);
    copyDerivativesIntoMARCO(derivativesVector);

    return true;
  }

  int KINSOLInstance::residualFunction(N_Vector variables, N_Vector residuals, void* userData)
  {
    realtype* rval = N_VGetArrayPointer(residuals);
    auto* instance = static_cast<KINSOLInstance*>(userData);

    instance->copyVariablesIntoMARCO(variables);

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
          auto residualFunctionResult = residualFunction(0.0, instance->simulationData, equationIndices);
          *rval++ = residualFunctionResult;
        } else {
          auto residualFunction = reinterpret_cast<ResidualFunction<double>>(instance->residuals[eq]);
          auto residualFunctionResult = residualFunction(0.0, instance->simulationData, equationIndices);
          *rval++ = residualFunctionResult;
        }
      } while (updateIndexes(equationIndices, instance->equationDimensions[eq]));
    }

    assert(rval == N_VGetArrayPointer(residuals) + instance->scalarEquationsNumber);

    return KIN_SUCCESS;
  }

  int KINSOLInstance::jacobianMatrix(
      N_Vector variables, N_Vector residuals,
      SUNMatrix jacobianMatrix,
      void* userData,
      N_Vector tempv1, N_Vector tempv2)
  {
    sunindextype* rowptrs = SUNSparseMatrix_IndexPointers(jacobianMatrix);
    sunindextype* colvals = SUNSparseMatrix_IndexValues(jacobianMatrix);
    realtype* jacobian = SUNSparseMatrix_Data(jacobianMatrix);

    auto* instance = static_cast<KINSOLInstance*>(userData);

    instance->copyVariablesIntoMARCO(variables);

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
            auto jacobianFunctionResult = jacobianFunction(0.0, instance->simulationData, equationIndices, variableIndices, 0.0);
            *jacobian++ = jacobianFunctionResult;
          } else {
            auto jacobianFunction = reinterpret_cast<JacobianFunction<double>>(instance->jacobians[eq][var.first]);
            auto jacobianFunctionResult = jacobianFunction(0.0, instance->simulationData, equationIndices, variableIndices, 0.0);
            *jacobian++ = jacobianFunctionResult;
          }

          *colvals++ = instance->variableOffsets[var.first] + computeOffset(instance->variablesDimensions[var.first], var.second);
        }
      } while (updateIndexes(equationIndices, instance->equationDimensions[eq]));
    }

    assert(rowptrs == SUNSparseMatrix_IndexPointers(jacobianMatrix) + instance->scalarEquationsNumber + 1);
    assert(colvals == SUNSparseMatrix_IndexValues(jacobianMatrix) + instance->nonZeroValuesNumber);
    assert(jacobian == SUNSparseMatrix_Data(jacobianMatrix) + instance->nonZeroValuesNumber);

    return KIN_SUCCESS;
  }

  /// Compute the column indexes of the current row of the Jacobian Matrix given
  /// the current vector equation and an array of indexes.
  std::set<DerivativeVariable> KINSOLInstance::computeIndexSet(size_t eq, size_t* eqIndexes) const
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
  void KINSOLInstance::computeNNZ()
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

  void KINSOLInstance::copyVariablesFromMARCO(N_Vector values)
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

  void KINSOLInstance::copyDerivativesFromMARCO(N_Vector values)
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

  void KINSOLInstance::copyVariablesIntoMARCO(N_Vector values)
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
          setterFn(descriptor, value, *indices);
        } else {
          auto setterFn = reinterpret_cast<VariableSetterFunction<double>>(variablesSetters[i]);
          auto value = static_cast<double>(*valuesPtr);
          setterFn(descriptor, value, *indices);
        }

        ++valuesPtr;
      }
    }
  }

  void KINSOLInstance::copyDerivativesIntoMARCO(N_Vector values)
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
          setterFn(descriptor, value, *indices);
        } else {
          auto setterFn = reinterpret_cast<VariableSetterFunction<double>>(derivativesSetters[i]);
          auto value = static_cast<double>(*valuesPtr);
          setterFn(descriptor, value, *indices);
        }

        ++valuesPtr;
      }
    }
  }

  void KINSOLInstance::printStatistics() const
  {
    if (scalarEquationsNumber == 0) {
      return;
    }

    if (getOptions().printJacobian) {
      printIncidenceMatrix();
    }

    long nje, nni, nli;

    KINGetNumJacEvals(kinsolMemory, &nje);
    KINGetNumNonlinSolvIters(kinsolMemory, &nni);
    KINGetNumLinIters(kinsolMemory, &nli);

    std::cerr << std::endl << "Final Run Statistics:" << std::endl;

    std::cerr << "Number of vector equations       = ";
    std::cerr << equationDimensions.size() << std::endl;
    std::cerr << "Number of scalar equations       = ";
    std::cerr << scalarEquationsNumber << std::endl;
    std::cerr << "Number of non-zero values        = ";
    std::cerr << nonZeroValuesNumber << std::endl;

    std::cerr << "Number of Jacobian evaluations   = " << nje << std::endl;

    std::cerr << "Number of nonlinear iterations   = " << nni << std::endl;
    std::cerr << "Number of linear iterations      = " << nli << std::endl;
  }

  void KINSOLInstance::printIncidenceMatrix() const
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

  bool KINSOLInstance::kinsolInit()
  {
    auto retVal = KINInit(kinsolMemory, residualFunction, variablesVector);

    if (retVal == KIN_MEM_NULL) {
      std::cerr << "KINInit - The kinsol_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == KIN_MEM_FAIL) {
      std::cerr << "KINInit - A memory allocation request has failed" << std::endl;
      return false;
    }

    if (retVal == KIN_ILL_INPUT) {
      std::cerr << "KINInit - An input argument to KINInit has an illegal value" << std::endl;
      return false;
    }

    return retVal == KIN_SUCCESS;
  }

  bool KINSOLInstance::kinsolFNTolerance()
  {
    auto retVal = KINSetFuncNormTol(kinsolMemory, getOptions().fnormtol);

    if (retVal == KIN_MEM_NULL) {
      std::cerr << "KINSVtolerances - The kinsol_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == KIN_ILL_INPUT) {
      std::cerr << "KINSVtolerances - The relative error tolerance was negative or the absolute tolerance vector had a negative component" << std::endl;
      return false;
    }

    return retVal == KIN_SUCCESS;
  }

  bool KINSOLInstance::kinsolSSTolerance()
  {
    auto retVal = KINSetScaledStepTol(kinsolMemory, getOptions().scsteptol);

    if (retVal == KIN_MEM_NULL) {
      std::cerr << "KINSVtolerances - The kinsol_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == KIN_ILL_INPUT) {
      std::cerr << "KINSVtolerances - The relative error tolerance was negative or the absolute tolerance vector had a negative component" << std::endl;
      return false;
    }

    return retVal == KIN_SUCCESS;
  }

  bool KINSOLInstance::kinsolSetLinearSolver()
  {
    auto retVal = KINSetLinearSolver(kinsolMemory, linearSolver, sparseMatrix);

    if (retVal == KINLS_MEM_NULL) {
      std::cerr << "KINSetLinearSolver - The kinsol_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == KINLS_ILL_INPUT) {
      std::cerr << "KINSetLinearSolver - The KINLS interface is not compatible with the LS or J input objects or is incompatible with the N_Vector object passed to KINInit" << std::endl;
      return false;
    }

    if (retVal == KINLS_SUNLS_FAIL) {
      std::cerr << "KINSetLinearSolver - A call to the LS object failed" << std::endl;
      return false;
    }

    if (retVal == KINLS_MEM_FAIL) {
      std::cerr << "KINSetLinearSolver - A memory allocation request failed" << std::endl;
      return false;
    }

    return retVal == KINLS_SUCCESS;
  }

  bool KINSOLInstance::kinsolSetUserData()
  {
    auto retVal = KINSetUserData(kinsolMemory, this);

    if (retVal == KIN_MEM_NULL) {
      std::cerr << "KINSetUserData - The kinsol_mem pointer is NULL" << std::endl;
      return false;
    }

    return retVal == KIN_SUCCESS;
  }

  bool KINSOLInstance::kinsolSetJacobianFunction()
  {
    auto retVal = KINSetJacFn(kinsolMemory, jacobianMatrix);

    if (retVal == KIN_MEM_NULL) {
      std::cerr << "KINSetJacFn - The kinsol_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == KINLS_LMEM_NULL) {
      std::cerr << "KINSetJacFn - The KINLS linear solver interface has not been initialized" << std::endl;
      return false;
    }

    return retVal == KIN_SUCCESS;
  }
}

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

/// Instantiate and initialize the struct of data needed by KINSOL, given the total
/// number of scalar equations.

static void* kinsolCreate_pvoid(int64_t scalarEquationsNumber, int64_t bitWidth)
{
  auto* instance = new KINSOLInstance(bitWidth, scalarEquationsNumber);
  return static_cast<void*>(instance);
}

RUNTIME_FUNC_DEF(kinsolCreate, PTR(void), int64_t, int64_t)

static void kinsolInit_void(void* userData)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  [[maybe_unused]] bool result = instance->initialize();
  assert(result && "Can't initialize the KINSOL instance");
}

RUNTIME_FUNC_DEF(kinsolInit, void, PTR(void))

static void kinsolStep_void(void* userData)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  [[maybe_unused]] bool result = instance->step();
  assert(result && "KINSOL step failed");
}

RUNTIME_FUNC_DEF(kinsolStep, void, PTR(void))

static void kinsolFree_void(void* userData)
{
  auto* data = static_cast<KINSOLInstance*>(userData);
  delete data;
}

RUNTIME_FUNC_DEF(kinsolFree, void, PTR(void))

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

static int64_t kinsolAddEquation_i64(void* userData, int64_t* ranges, int64_t rank)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  return instance->addEquation(ranges, rank);
}

RUNTIME_FUNC_DEF(kinsolAddEquation, int64_t, PTR(void), PTR(int64_t), int64_t)

static void kinsolAddResidual_void(void* userData, int64_t equationIndex, void* residualFunction)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  instance->addResidualFunction(equationIndex, residualFunction);
}

RUNTIME_FUNC_DEF(kinsolAddResidual, void, PTR(void), int64_t, PTR(void))

static void kinsolAddJacobian_void(void* userData, int64_t equationIndex, int64_t variableIndex, void* jacobianFunction)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  instance->addJacobianFunction(equationIndex, variableIndex, jacobianFunction);
}

RUNTIME_FUNC_DEF(kinsolAddJacobian, void, PTR(void), int64_t, int64_t, PTR(void))

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

static int64_t kinsolAddAlgebraicVariable_i64(void* userData, void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  return instance->addAlgebraicVariable(variable, dimensions, rank, getter, setter);
}

RUNTIME_FUNC_DEF(kinsolAddAlgebraicVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

static int64_t kinsolAddStateVariable_i64(void* userData, void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  return instance->addStateVariable(variable, dimensions, rank, getter, setter);
}

RUNTIME_FUNC_DEF(kinsolAddStateVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

static void kinsolSetDerivative_void(void* userData, int64_t stateVariable, void* derivative, void* getter, void* setter)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  instance->setDerivative(stateVariable, derivative, getter, setter);
}

RUNTIME_FUNC_DEF(kinsolSetDerivative, void, PTR(void), int64_t, PTR(void), PTR(void), PTR(void))

/// Add a variable access to the var-th variable, where ind is the induction
/// variable and off is the access offset.
static void kinsolAddVariableAccess_void(
    void* userData, int64_t equationIndex, int64_t variableIndex, int64_t* access, int64_t rank)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  instance->addVariableAccess(equationIndex, variableIndex, access, rank);
}

RUNTIME_FUNC_DEF(kinsolAddVariableAccess, void, PTR(void), int64_t, int64_t, PTR(int64_t), int64_t)

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static void kinsolPrintStatistics_void(void* userData)
{
  auto* instance = static_cast<KINSOLInstance*>(userData);
  instance->printStatistics();
}

RUNTIME_FUNC_DEF(kinsolPrintStatistics, void, PTR(void))
