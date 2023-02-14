#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "marco/Runtime/Solvers/IDA/Profiler.h"
#include "marco/Runtime/MemoryManagement.h"
#include <atomic>
#include <cassert>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>

using namespace ::marco::runtime;
using namespace ::marco::runtime::ida;

//===---------------------------------------------------------------------===//
// Utilities
//===---------------------------------------------------------------------===//

namespace marco::runtime::ida
{
  VariableDimensions::VariableDimensions(size_t rank)
  {
    dimensions.resize(rank, 0);
  }

  size_t VariableDimensions::rank() const
  {
    return dimensions.size();
  }

  int64_t& VariableDimensions::operator[](size_t index)
  {
    return dimensions[index];
  }

  const int64_t& VariableDimensions::operator[](size_t index) const
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
    delete[] indices;
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

  const int64_t* VariableIndicesIterator::operator*() const
  {
    return indices;
  }

  VariableIndicesIterator::VariableIndicesIterator(
      const VariableDimensions& dimensions)
      : dimensions(&dimensions)
  {
    indices = new int64_t[dimensions.rank()];
  }

  void VariableIndicesIterator::fetchNext()
  {
    size_t rank = dimensions->rank();
    size_t posFromLast = 0;

    assert(std::none_of(
        dimensions->begin(), dimensions->end(),
        [](const auto& dimension) {
          return dimension == 0;
        }));

    while (posFromLast < rank && ++indices[rank - posFromLast - 1] ==
               (*dimensions)[rank - posFromLast - 1]) {
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
static bool checkAllocation(void* retval, const char* functionName)
{
  if (retval == nullptr) {
    std::cerr << "SUNDIALS_ERROR: " << functionName;
    std::cerr << "() failed - returned NULL pointer" << std::endl;
    return false;
  }

  return true;
}

/// Apply an access function to the indices of an equation in order to get the
/// indices of the accessed scalar variable.
static std::vector<int64_t> applyAccessFunction(
    const int64_t* equationIndices, const Access& access)
{
  std::vector<int64_t> variableIndices;

  for (const auto& dimensionAccess : access) {
    int64_t inductionVariableIndex = dimensionAccess.first;

    if (dimensionAccess.first == -1) {
      variableIndices.push_back(dimensionAccess.second);
    } else {
      variableIndices.push_back(
          equationIndices[inductionVariableIndex] + dimensionAccess.second);
    }
  }

  return variableIndices;
}

static std::vector<int64_t> applyAccessFunction(
    const std::vector<int64_t>& equationIndices, const Access& access)
{
  return applyAccessFunction(equationIndices.data(), access);
}

/// Apply an access function to the ranges of an equation in order to get the
/// range of indices of the accessed variable.
static MultidimensionalRange applyAccessFunction(
    const MultidimensionalRange& equationIndices, const Access& access)
{
  MultidimensionalRange ranges;

  for (const auto& dimensionAccess : access) {
    int64_t inductionVariableIndex = dimensionAccess.first;

    if (inductionVariableIndex == -1) {
      ranges.push_back({
          dimensionAccess.second,
          dimensionAccess.second + 1
      });
    } else {
      ranges.push_back({
          equationIndices[inductionVariableIndex].begin + dimensionAccess.second,
          equationIndices[inductionVariableIndex].end + dimensionAccess.second,
      });
    }
  }

  return ranges;
}

/// Given an array of indices and the dimensions of an equation, increase the
/// indices within the induction bounds of the equation. Return false if the
/// indices exceed the equation bounds, which means the computation has finished,
/// true otherwise.
static bool advanceIndices(int64_t* indices, const MultidimensionalRange& ranges)
{
  for (size_t i = 0, e = ranges.size(); i < e; ++i) {
    size_t pos = e - i - 1;
    ++indices[pos];

    if (indices[pos] == ranges[pos].end) {
      indices[pos] = ranges[pos].begin;
    } else {
      return true;
    }
  }

  return false;
}

static bool advanceIndices(
    std::vector<int64_t>& indices, const MultidimensionalRange& ranges)
{
  return advanceIndices(indices.data(), ranges);
}

/// Given an array of indexes, the dimension of a variable and the type of
/// access, return the index needed to access the flattened multidimensional
/// variable.
static size_t computeOffset(
    const int64_t* indices,
    const VariableDimensions& dimensions,
    const Access& access)
{
  assert(access.size() == dimensions.rank());
  size_t offset = 0;

  for (size_t i = 0; i < access.size(); ++i) {
    int64_t induction = access[i].first;
    size_t accessOffset = induction != -1 ? indices[induction] : 0;
    accessOffset += access[i].second;

    offset = offset * dimensions[i] + accessOffset;
  }

  return offset;
}

/// Get the flat index corresponding to a multidimensional access.
/// Example:
///   x[d1][d2][d3]
///   x[i][j][k] -> x[i * d2 * d3 + j * d3 + k]
static int64_t getFlatIndex(
    const VariableDimensions& dimensions, const int64_t* indices)
{
  int64_t offset = indices[0];

  for (size_t i = 1, e = dimensions.rank(); i < e; ++i) {
    offset = offset * dimensions[i] + indices[i];
  }

  return offset;
}

static int64_t getFlatIndex(
    const VariableDimensions& dimensions, const std::vector<int64_t>& indices)
{
  assert(indices.size() == dimensions.rank());
  return getFlatIndex(dimensions, indices.data());
}

//===---------------------------------------------------------------------===//
// Solver
//===---------------------------------------------------------------------===//

namespace marco::runtime::ida
{
  IDAInstance::IDAInstance(int64_t scalarEquationsNumber)
      : initialized(false),
        scalarEquationsNumber(scalarEquationsNumber),
        precomputedAccesses(false),
        startTime(getOptions().startTime),
        endTime(getOptions().endTime),
        timeStep(getOptions().timeStep)
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
  }

  IDAInstance::~IDAInstance()
  {
    assert(initialized && "The IDA instance has not been initialized yet");

    for (auto* variable : parametricVariables) {
      heapFree(variable);
    }

    for (auto* variable : algebraicVariables) {
      heapFree(variable);
    }

    for (auto* variable : stateVariables) {
      heapFree(variable);
    }

    for (auto* derivative : derivativeVariables) {
      heapFree(derivative);
    }

    N_VDestroy(variablesVector);
    N_VDestroy(derivativesVector);
    N_VDestroy(idVector);
    N_VDestroy(tolerancesVector);

    if (scalarEquationsNumber != 0) {
      delete simulationData;
      IDAFree(&idaMemory);
      SUNLinSolFree(linearSolver);
      SUNMatDestroy(sparseMatrix);
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

  void IDAInstance::addParametricVariable(void* variable)
  {
    parametricVariables.push_back(variable);
  }

  int64_t IDAInstance::addAlgebraicVariable(
      void* variable,
      int64_t* dimensions,
      int64_t rank,
      VariableGetter getterFunction,
      VariableSetter setterFunction)
  {
    assert(!initialized && "The IDA instance has already been initialized");
    assert(variableOffsets.size() == getNumOfArrayVariables() + 1);

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

    // Initialize derivativeValues, idValues and absoluteTolerances.
    auto* derivativeValues = N_VGetArrayPointer(derivativesVector);
    auto* idValues = N_VGetArrayPointer(idVector);
    auto* toleranceValues = N_VGetArrayPointer(tolerancesVector);

    realtype absTol = std::min(
        getOptions().maxAlgebraicAbsoluteTolerance,
        getOptions().absoluteTolerance);

    for (int64_t i = 0; i < flatSize; ++i) {
      derivativeValues[offset + i] = 0;
      idValues[offset + i] = 0;
      toleranceValues[offset + i] = absTol;
    }

    algebraicVariables.push_back(variable);

    algebraicAndStateVariables.push_back(variable);
    algebraicAndStateVariablesGetters.push_back(getterFunction);
    algebraicAndStateVariablesSetters.push_back(setterFunction);

    // Return the index of the variable.
    size_t idaVariable = getNumOfArrayVariables() - 1;
    algebraicVariablesMapping[idaVariable] = algebraicVariables.size() - 1;
    return idaVariable;
  }

  int64_t IDAInstance::addStateVariable(
      void* variable,
      int64_t* dimensions,
      int64_t rank,
      VariableGetter getterFunction,
      VariableSetter setterFunction)
  {
    assert(!initialized && "The IDA instance has already been initialized");
    assert(variableOffsets.size() == getNumOfArrayVariables() + 1);

    // Add variable offset and dimensions
    VariableDimensions variableDimensions(rank);
    int64_t flatSize = 1;

    for (int64_t i = 0; i < rank; ++i) {
      flatSize *= dimensions[i];
      variableDimensions[i] = dimensions[i];
    }

    variablesDimensions.push_back(variableDimensions);

    // Store the position of the start of the flattened array.
    int64_t offset = variableOffsets.back();
    variableOffsets.push_back(offset + flatSize);

    // Initialize the derivatives, the id values and the absolute tolerances.
    auto* derivativeValues = N_VGetArrayPointer(derivativesVector);
    auto* idValues = N_VGetArrayPointer(idVector);
    auto* toleranceValues = N_VGetArrayPointer(tolerancesVector);

    for (int64_t i = 0; i < flatSize; ++i) {
      derivativeValues[offset + i] = 0;
      idValues[offset + i] = 1;
      toleranceValues[offset + i] = getOptions().absoluteTolerance;
    }

    stateVariables.push_back(variable);

    algebraicAndStateVariables.push_back(variable);
    algebraicAndStateVariablesGetters.push_back(getterFunction);
    algebraicAndStateVariablesSetters.push_back(setterFunction);

    derivativeVariables.push_back(nullptr);
    derivativeVariablesGetters.push_back(nullptr);
    derivativeVariablesSetters.push_back(nullptr);

    // Return the position of the IDA variable.
    size_t idaVariable = getNumOfArrayVariables() - 1;
    stateVariablesMapping[idaVariable] = stateVariables.size() - 1;
    return static_cast<int64_t>(idaVariable);
  }

  void IDAInstance::setDerivative(
      int64_t idaStateVariable,
      void* derivative,
      VariableGetter getterFunction,
      VariableSetter setterFunction)
  {
    assert(!initialized && "The IDA instance has already been initialized");
    assert(variableOffsets.size() == getNumOfArrayVariables() + 1);

    assert(stateVariablesMapping.find(idaStateVariable) !=
           stateVariablesMapping.end());

    size_t stateVariablePosition = stateVariablesMapping[idaStateVariable];

    assert(stateVariablePosition < derivativeVariables.size());
    assert(stateVariablePosition < derivativeVariablesGetters.size());
    assert(stateVariablePosition < derivativeVariablesSetters.size());

    derivativeVariables[stateVariablePosition] = derivative;
    derivativeVariablesGetters[stateVariablePosition] = getterFunction;
    derivativeVariablesSetters[stateVariablePosition] = setterFunction;
  }

  int64_t IDAInstance::addEquation(
      int64_t* ranges,
      int64_t equationRank,
      int64_t writtenVariable,
      int64_t* writeAccess)
  {
    assert(!initialized && "The IDA instance has already been initialized");

    // Add the start and end dimensions of the current equation.
    MultidimensionalRange eqRanges = {};

    for (size_t i = 0, e = equationRank * 2; i < e; i += 2) {
      int64_t begin = ranges[i];
      int64_t end = ranges[i + 1];
      eqRanges.push_back({ begin, end });
    }

    equationRanges.push_back(eqRanges);

    // Store the information about the written scalar variable.
    Access access;

    const VariableDimensions& variableDimensions =
        variablesDimensions[writtenVariable];

    auto variableRank = static_cast<int64_t>(variableDimensions.rank());

    for (int64_t i = 0, e = variableRank * 2; i < e; i += 2) {
      access.push_back({ writeAccess[i], writeAccess[i + 1] });
    }

    writeAccesses.emplace_back(writtenVariable, access);

    // Return the index of the equation.
    return static_cast<int64_t>(getNumOfVectorizedEquations() - 1);
  }

  void IDAInstance::addVariableAccess(
      int64_t equationIndex,
      int64_t variableIndex,
      int64_t* access,
      int64_t rank)
  {
    assert(!initialized && "The IDA instance has already been initialized");

    assert(equationIndex >= 0);
    assert((size_t) equationIndex < getNumOfVectorizedEquations());
    assert(variableIndex >= 0);
    assert((size_t) variableIndex < getNumOfArrayVariables());

    precomputedAccesses = true;

    if (variableAccesses.size() <= (size_t) equationIndex) {
      variableAccesses.resize(equationIndex + 1);
    }

    auto& varAccessList = variableAccesses[equationIndex];
    varAccessList.push_back({variableIndex, {}});

    size_t numElements = rank * 2;

    for (size_t i = 0; i < numElements; i += 2) {
      varAccessList.back().second.emplace_back(access[i], access[i + 1]);
    }
  }

  void IDAInstance::setResidualFunction(
      int64_t equationIndex,
      ResidualFunction residualFunction)
  {
    assert(!initialized && "The IDA instance has already been initialized");

    assert(equationIndex >= 0);

    if (residualFunctions.size() <= (size_t) equationIndex) {
      residualFunctions.resize(equationIndex + 1, nullptr);
    }

    residualFunctions[equationIndex] = residualFunction;
  }

  void IDAInstance::addJacobianFunction(
      int64_t equationIndex,
      int64_t variableIndex,
      JacobianFunction jacobianFunction)
  {
    assert(!initialized && "The IDA instance has already been initialized");

    assert(equationIndex >= 0);
    assert(variableIndex >= 0);

    if (jacobianFunctions.size() <= (size_t) equationIndex) {
      jacobianFunctions.resize(equationIndex + 1, {});
    }

    if (jacobianFunctions[equationIndex].size() <= (size_t) variableIndex) {
      jacobianFunctions[equationIndex].resize(variableIndex + 1, nullptr);
    }

    jacobianFunctions[equationIndex][variableIndex] = jacobianFunction;
  }

  bool IDAInstance::initialize()
  {
    assert(!initialized && "The IDA instance has already been initialized");
    initialized = true;

    currentTime = startTime;

    if (scalarEquationsNumber == 0) {
      // IDA has nothing to solve.
      return true;
    }

    size_t numOfParametricVariables = parametricVariables.size();
    size_t numOfAlgebraicVariables = algebraicVariables.size();
    size_t numOfStateVariables = stateVariables.size();
    size_t numOfDerivativeVariables = derivativeVariables.size();

    assert(numOfStateVariables == numOfDerivativeVariables &&
           "The number of derivative variables doesn't match the number of "
           "state variables");

    size_t numOfVariables = numOfParametricVariables +
        numOfAlgebraicVariables +
        numOfStateVariables +
        numOfDerivativeVariables;

    // Determine the order in which the equations must be processed when
    // computing residuals and jacobians.
    assert(getNumOfVectorizedEquations() == writeAccesses.size());
    equationsProcessingOrder.resize(getNumOfVectorizedEquations());

    for (size_t i = 0, e = getNumOfVectorizedEquations(); i < e; ++i) {
      equationsProcessingOrder[i] = i;
    }

    std::sort(equationsProcessingOrder.begin(), equationsProcessingOrder.end(),
              [&](size_t firstEquation, size_t secondEquation) {
                int64_t firstWrittenVariable =
                    writeAccesses[firstEquation].first;

                int64_t secondWrittenVariable =
                    writeAccesses[secondEquation].first;

                if (firstWrittenVariable != secondWrittenVariable) {
                  return firstWrittenVariable < secondWrittenVariable;
                }

                MultidimensionalRange firstWrittenIndices =
                    applyAccessFunction(
                        equationRanges[firstEquation],
                        writeAccesses[firstEquation].second);

                MultidimensionalRange secondWrittenIndices =
                    applyAccessFunction(
                        equationRanges[secondEquation],
                        writeAccesses[secondEquation].second);

                return firstWrittenIndices < secondWrittenIndices;
              });

    // Check that all the residual functions have been set.
    assert(residualFunctions.size() == getNumOfVectorizedEquations());

    assert(std::all_of(
        residualFunctions.begin(), residualFunctions.end(),
        [](const ResidualFunction& function) {
          return function != nullptr;
        }));

    // If the IDA instance is not informed about the acceCheck that all the jacobian functions have been set.
    assert(precomputedAccesses ||
           jacobianFunctions.size() == getNumOfVectorizedEquations());

    assert(precomputedAccesses ||
           std::all_of(
               jacobianFunctions.begin(), jacobianFunctions.end(),
               [&](std::vector<JacobianFunction> functions) {
                 if (functions.size() != algebraicAndStateVariables.size()) {
                   return false;
                 }

                 return std::all_of(
                     functions.begin(), functions.end(),
                     [](const JacobianFunction& function) {
                       return function != nullptr;
                     });
               }));

    // Construct the variables list to be passed to the residual and Jacobian
    // functions.
    simulationData = new void*[numOfVariables];

    for (size_t i = 0; i < numOfVariables; ++i) {
      simulationData[i] = nullptr;
    }

    for (size_t i = 0; i < numOfParametricVariables; ++i) {
      size_t pos = i;
      simulationData[pos] = parametricVariables[i];
    }

    for (size_t i = 0; i < numOfAlgebraicVariables; ++i) {
      size_t pos = i + numOfParametricVariables;
      simulationData[pos] = algebraicVariables[i];
    }

    for (size_t i = 0; i < numOfStateVariables; ++i) {
      size_t pos = i + numOfParametricVariables + numOfAlgebraicVariables;
      simulationData[pos] = stateVariables[i];
    }

    // Check that all the derivatives have been set.
    assert(std::none_of(
               derivativeVariables.begin(), derivativeVariables.end(),
               [](void* variable) {
                 return variable == nullptr;
               }) && "Not all the derivative variables have been set");

    // Check that all the derivatives are unique.
    assert(std::all_of(
               derivativeVariables.begin(), derivativeVariables.end(),
               [&](void* variable) {
                 size_t count = std::count(
                     derivativeVariables.begin(),
                     derivativeVariables.end(),
                     variable);

                 return count == 1;
               }) && "The same derivative has been set for multiple state variables");

    for (size_t i = 0; i < numOfDerivativeVariables; ++i) {
      size_t pos = i + numOfParametricVariables +
          numOfAlgebraicVariables +
          numOfStateVariables;

      simulationData[pos] = derivativeVariables[i];
    }

    assert(std::none_of(
        simulationData, simulationData + numOfVariables,
        [](void* variable) {
          return variable == nullptr;
        }));

    // Reserve the space for data of the jacobian matrix.
    jacobianMatrixData.resize(scalarEquationsNumber);

    for (size_t eq : equationsProcessingOrder) {
      std::vector<int64_t> equationIndices;

      size_t equationRank = getEquationRank(eq);
      equationIndices.resize(equationRank);

      for (size_t i = 0; i < equationRank; ++i) {
        equationIndices[i] = equationRanges[eq][i].begin;
      }

      int64_t equationVariable = writeAccesses[eq].first;
      size_t equationArrayVariableOffset = variableOffsets[equationVariable];

      do {
        std::vector<int64_t> equationVariableIndices = applyAccessFunction(
            equationIndices, writeAccesses[eq].second);

        size_t equationScalarVariableOffset = getFlatIndex(
            variablesDimensions[equationVariable],
            equationVariableIndices);

        size_t scalarEquationIndex =
            equationArrayVariableOffset + equationScalarVariableOffset;

        // Compute the column indexes that may be non-zeros.
        std::vector<JacobianColumn> jacobianColumns =
            computeJacobianColumns(eq, equationIndices.data());

        jacobianMatrixData[scalarEquationIndex].resize(jacobianColumns.size());
      } while (advanceIndices(equationIndices, equationRanges[eq]));
    }

    // Initialize the values of the variables living inside IDA.
    copyVariablesFromMARCO(variablesVector, derivativesVector);

    // Compute the total amount of non-zero values in the Jacobian Matrix.
    computeNNZ();

    // Create and initialize the memory for IDA.
    idaMemory = IDACreate(ctx);

    if (!checkAllocation(idaMemory, "IDACreate")) {
      return false;
    }

    if (!idaInit()) {
      return false;
    }

    if (!idaSVTolerances()) {
      return false;
    }

    // Create sparse SUNMatrix for use in linear solver.
    sparseMatrix = SUNSparseMatrix(
        scalarEquationsNumber,
        scalarEquationsNumber,
        nonZeroValuesNumber,
        CSR_MAT,
        ctx);

    if (!checkAllocation(
            static_cast<void*>(sparseMatrix), "SUNSparseMatrix")) {
      return false;
    }

    // Create and attach a KLU SUNLinearSolver object.
    linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix, ctx);

    if (!checkAllocation(static_cast<void*>(linearSolver), "SUNLinSol_KLU")) {
      return false;
    }

    if (!idaSetLinearSolver()) {
      return false;
    }

    if (!idaSetUserData() ||
        !idaSetMaxNumSteps() ||
        !idaSetInitialStepSize() ||
        !idaSetMinStepSize() ||
        !idaSetMaxStepSize() ||
        !idaSetStopTime() ||
        !idaSetMaxErrTestFails() ||
        !idaSetSuppressAlg() ||
        !idaSetId() ||
        !idaSetJacobianFunction() ||
        !idaSetMaxNonlinIters() ||
        !idaSetMaxConvFails() ||
        !idaSetNonlinConvCoef() ||
        !idaSetNonlinConvCoefIC() ||
        !idaSetMaxNumStepsIC() ||
        !idaSetMaxNumJacsIC() ||
        !idaSetMaxNumItersIC() ||
        !idaSetLineSearchOffIC()) {
      return false;
    }

    return true;
  }

  bool IDAInstance::calcIC()
  {
    if (scalarEquationsNumber == 0) {
      // IDA has nothing to solve
      return true;
    }

    realtype firstOutTime = (endTime - startTime) / getOptions().timeScalingFactorInit;

    IDA_PROFILER_IC_START;
    auto calcICRetVal = IDACalcIC(idaMemory, IDA_YA_YDP_INIT, firstOutTime);
    IDA_PROFILER_IC_STOP;

    if (calcICRetVal != IDA_SUCCESS) {
      if (calcICRetVal == IDALS_MEM_NULL) {
        std::cerr << "IDACalcIC - The ida_mem pointer is NULL" << std::endl;
      } else if (calcICRetVal == IDA_NO_MALLOC) {
        std::cerr << "IDACalcIC - The allocation function IDAInit has not been called" << std::endl;
      } else if (calcICRetVal == IDA_ILL_INPUT) {
        std::cerr << "IDACalcIC - One of the input arguments was illegal" << std::endl;
      } else if (calcICRetVal == IDA_LSETUP_FAIL) {
        std::cerr << "IDACalcIC - The linear solver’s setup function failed in an unrecoverable manner" << std::endl;
      } else if (calcICRetVal == IDA_LINIT_FAIL) {
        std::cerr << "IDACalcIC - The linear solver’s initialization function failed" << std::endl;
      } else if (calcICRetVal == IDA_LSOLVE_FAIL) {
        std::cerr << "IDACalcIC - The linear solver’s solve function failed in an unrecoverable manner" << std::endl;
      } else if (calcICRetVal == IDA_BAD_EWT) {
        std::cerr << "IDACalcIC - Some component of the error weight vector is zero (illegal), either for the input value of y0 or a corrected value" << std::endl;
      } else if (calcICRetVal == IDA_FIRST_RES_FAIL) {
        std::cerr << "IDACalcIC - The user’s residual function returned a recoverable error flag on the first call, but IDACalcIC was unable to recover" << std::endl;
      } else if (calcICRetVal == IDA_RES_FAIL) {
        std::cerr << "IDACalcIC - The user’s residual function returned a nonrecoverable error flag" << std::endl;
      } else if (calcICRetVal == IDA_NO_RECOVERY) {
        std::cerr << "IDACalcIC - The user’s residual function, or the linear solver’s setup or solve function had a recoverable error, but IDACalcIC was unable to recover" << std::endl;
      } else if (calcICRetVal == IDA_CONSTR_FAIL) {
        std::cerr << "IDACalcIC - IDACalcIC was unable to find a solution satisfying the inequality constraints" << std::endl;
      } else if (calcICRetVal == IDA_LINESEARCH_FAIL) {
        std::cerr << "IDACalcIC - The linesearch algorithm failed to find a solution with a step larger than steptol in weighted RMS norm, and within the allowed number of backtracks" << std::endl;
      } else if (calcICRetVal == IDA_CONV_FAIL) {
        std::cerr << "IDACalcIC - IDACalcIC failed to get convergence of the Newton iterations" << std::endl;
      }

      return false;
    }

    return true;
  }

  bool IDAInstance::step()
  {
    assert(initialized && "The IDA instance has not been initialized yet");

    if (scalarEquationsNumber == 0) {
      // IDA has nothing to solve. Just increment the time.

      if (getOptions().equidistantTimeGrid) {
        currentTime += timeStep;
      } else {
        currentTime = endTime;
      }

      return true;
    }

    // Execute one step
    IDA_PROFILER_STEP_START;

    realtype tout = getOptions().equidistantTimeGrid ? (currentTime + timeStep) : endTime;

    auto solveRetVal = IDASolve(
        idaMemory,
        tout,
        &currentTime,
        variablesVector,
        derivativesVector,
        getOptions().equidistantTimeGrid ? IDA_NORMAL : IDA_ONE_STEP);

    IDA_PROFILER_STEP_STOP;

    if (solveRetVal != IDA_SUCCESS) {
      if (solveRetVal == IDA_TSTOP_RETURN) {
        return true;
      }

      if (solveRetVal == IDA_ROOT_RETURN) {
        return true;
      }

      if (solveRetVal == IDA_MEM_NULL) {
        std::cerr << "IDASolve - The ida_mem pointer is NULL" << std::endl;
      } else if (solveRetVal == IDA_ILL_INPUT) {
        std::cerr << "IDASolve - One of the inputs to IDASolve was illegal, or some other input to the solver was either illegal or missing" << std::endl;
      } else if (solveRetVal == IDA_TOO_MUCH_WORK) {
        std::cerr << "IDASolve - The solver took mxstep internal steps but could not reach tout" << std::endl;
      } else if (solveRetVal == IDA_TOO_MUCH_ACC) {
        std::cerr << "IDASolve - The solver could not satisfy the accuracy demanded by the user for some internal step" << std::endl;
      } else if (solveRetVal == IDA_ERR_FAIL) {
        std::cerr << "IDASolve - Error test failures occurred too many times during one internal time step or occurred with |h| = hmin" << std::endl;
      } else if (solveRetVal == IDA_CONV_FAIL) {
        std::cerr << "IDASolve - Convergence test failures occurred too many times during one internal time step or occurred with |h| = hmin" << std::endl;
      } else if (solveRetVal == IDA_LINIT_FAIL) {
        std::cerr << "IDASolve - The linear solver’s initialization function failed" << std::endl;
      } else if (solveRetVal == IDA_LSETUP_FAIL) {
        std::cerr << "IDASolve - The linear solver’s setup function failed in an unrecoverable manner" << std::endl;
      } else if (solveRetVal == IDA_LSOLVE_FAIL) {
        std::cerr << "IDASolve - The linear solver’s solve function failed in an unrecoverable manner" << std::endl;
      } else if (solveRetVal == IDA_CONSTR_FAIL) {
        std::cerr << "IDASolve - The inequality constraints were violated and the solver was unable to recover" << std::endl;
      } else if (solveRetVal == IDA_REP_RES_ERR) {
        std::cerr << "IDASolve - The user’s residual function repeatedly returned a recoverable error flag, but the solver was unable to recover" << std::endl;
      } else if (solveRetVal == IDA_RES_FAIL) {
        std::cerr << "IDASolve - The user’s residual function returned a nonrecoverable error flag" << std::endl;
      } else if (solveRetVal == IDA_RTFUNC_FAIL) {
        std::cerr << "IDASolve - The rootfinding function failed" << std::endl;
      }

      return false;
    }

    copyVariablesIntoMARCO(variablesVector, derivativesVector);

    return true;
  }

  realtype IDAInstance::getCurrentTime() const
  {
    return currentTime;
  }

  int IDAInstance::residualFunction(
      realtype time,
      N_Vector variables,
      N_Vector derivatives,
      N_Vector residuals,
      void* userData)
  {
    realtype* rval = N_VGetArrayPointer(residuals);
    auto* instance = static_cast<IDAInstance*>(userData);

    // Copy the values of the variables and derivatives provided by IDA into
    // the variables owned by MARCO, so that the residual functions operate on
    // the current iteration values.
    instance->copyVariablesIntoMARCO(variables, derivatives);

    // For every vectorized equation, set the residual values of the variables
    // it writes into.

    instance->scalarEquationsParallelIteration(
        [&](size_t eq, const std::vector<int64_t>& equationIndices) {
          assert(equationIndices.size() == instance->getEquationRank(eq));

          int64_t writtenVariable = instance->writeAccesses[eq].first;
          size_t arrayVariableOffset = instance->variableOffsets[writtenVariable];

          std::vector<int64_t> writtenVariableIndices = applyAccessFunction(
              equationIndices, instance->writeAccesses[eq].second);

          size_t scalarVariableOffset = getFlatIndex(
              instance->variablesDimensions[writtenVariable],
              writtenVariableIndices);

          auto residualFunctionResult = instance->residualFunctions[eq](
              time, instance->simulationData, equationIndices.data());

          *(rval + arrayVariableOffset + scalarVariableOffset) = residualFunctionResult;
        });

    return IDA_SUCCESS;
  }

  int IDAInstance::jacobianMatrix(
      realtype time, realtype alpha,
      N_Vector variables, N_Vector derivatives, N_Vector residuals,
      SUNMatrix jacobianMatrix,
      void* userData,
      N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
  {
    realtype* jacobian = SUNSparseMatrix_Data(jacobianMatrix);

    auto* instance = static_cast<IDAInstance*>(userData);

    // Copy the values of the variables and derivatives provided by IDA into
    // the variables owned by MARCO, so that the jacobian functions operate on
    // the current iteration values.
    instance->copyVariablesIntoMARCO(variables, derivatives);

    // For every vectorized equation, compute its row within the Jacobian
    // matrix.

    instance->scalarEquationsParallelIteration(
        [&](size_t eq, const std::vector<int64_t>& equationIndices) {
          int64_t equationVariable = instance->writeAccesses[eq].first;

          size_t equationArrayVariableOffset =
              instance->variableOffsets[equationVariable];

            std::vector<int64_t> equationVariableIndices = applyAccessFunction(
                equationIndices, instance->writeAccesses[eq].second);

            size_t equationScalarVariableOffset = getFlatIndex(
                instance->variablesDimensions[equationVariable],
                equationVariableIndices);

            size_t scalarEquationIndex =
                equationArrayVariableOffset + equationScalarVariableOffset;

            // Compute the column indexes that may be non-zeros
            std::vector<JacobianColumn> jacobianColumns =
                instance->computeJacobianColumns(eq, equationIndices.data());

            // For every scalar variable with respect to which the equation must be
            // partially differentiated.
            for (size_t i = 0, e = jacobianColumns.size(); i < e; ++i) {
              const JacobianColumn& column = jacobianColumns[i];
              const int64_t* variableIndices = column.second.data();

              size_t arrayVariableOffset = instance->variableOffsets[column.first];

              size_t scalarVariableOffset = getFlatIndex(
                  instance->variablesDimensions[column.first],
                  column.second);

              assert(instance->jacobianFunctions[eq][column.first] != nullptr);

              auto jacobianFunctionResult =
                  instance->jacobianFunctions[eq][column.first](
                      time,
                      instance->simulationData,
                      equationIndices.data(),
                      variableIndices,
                      alpha);

              instance->jacobianMatrixData[scalarEquationIndex][i].second =
                  jacobianFunctionResult;

              instance->jacobianMatrixData[scalarEquationIndex][i].first =
                  arrayVariableOffset + scalarVariableOffset;
            }
        });

    sunindextype* rowPtrs = SUNSparseMatrix_IndexPointers(jacobianMatrix);
    sunindextype* columnIndices = SUNSparseMatrix_IndexValues(jacobianMatrix);

    sunindextype offset = 0;
    *rowPtrs++ = offset;

    for (const auto& row : instance->jacobianMatrixData) {
      offset += row.size();
      *rowPtrs++ = offset;

      for (const auto& column : row) {
        *columnIndices++ = column.first;
        *jacobian++ = column.second;
      }
    }

    assert(rowPtrs == SUNSparseMatrix_IndexPointers(jacobianMatrix) + instance->scalarEquationsNumber + 1);
    assert(columnIndices == SUNSparseMatrix_IndexValues(jacobianMatrix) + instance->nonZeroValuesNumber);
    assert(jacobian == SUNSparseMatrix_Data(jacobianMatrix) + instance->nonZeroValuesNumber);

    return IDA_SUCCESS;
  }

  size_t IDAInstance::getNumOfArrayVariables() const
  {
    return variablesDimensions.size();
  }

  size_t IDAInstance::getNumOfVectorizedEquations() const
  {
    return equationRanges.size();
  }

  size_t IDAInstance::getEquationRank(size_t equation) const
  {
    return equationRanges[equation].size();
  }

  /// Determine which of the columns of the current Jacobian row has to be
  /// populated, and with respect to which variable the partial derivative has
  /// to be performed. The row is determined by the indices of the equation.
  std::vector<JacobianColumn> IDAInstance::computeJacobianColumns(
      size_t eq, const int64_t* equationIndices) const
  {
    assert(initialized && "The IDA instance has not been initialized yet");
    std::set<JacobianColumn> uniqueColumns;

    if (precomputedAccesses) {
      for (const auto& access : variableAccesses[eq]) {
        sunindextype variableIndex = access.first;
        Access variableAccess = access.second;

        assert(variableAccess.size() ==
               variablesDimensions[variableIndex].rank());

        uniqueColumns.insert({
            variableIndex,
            applyAccessFunction(equationIndices, variableAccess)
        });
      }
    } else {
      for (size_t variableIndex = 0, e = getNumOfArrayVariables();
           variableIndex < e; ++variableIndex) {
        const auto& dimensions = variablesDimensions[variableIndex];

        for (auto indices = dimensions.indicesBegin(),
                  end = dimensions.indicesEnd();
             indices != end; ++indices) {
          JacobianColumn column(variableIndex, {});

          for (size_t dim = 0; dim < dimensions.rank(); ++dim) {
            column.second.push_back((*indices)[dim]);
          }

          uniqueColumns.insert(std::move(column));
        }
      }
    }

    std::vector<JacobianColumn> orderedColumns;

    for (const JacobianColumn& column : uniqueColumns) {
      orderedColumns.push_back(column);
    }

    std::sort(orderedColumns.begin(), orderedColumns.end(),
              [](const JacobianColumn& first, const JacobianColumn& second) {
                if (first.first != second.first) {
                  return first.first < second.first;
                }

                assert(first.second.size() == second.second.size());

                for (int i = 0, e = first.second.size(); i < e; ++i) {
                  if (first.second[i] < second.second[i]) {
                    return true;
                  }
                }

                return false;
              });

    return orderedColumns;
  }

  /// Compute the number of non-zero values in the Jacobian Matrix. Also
  /// compute the column indexes of all non-zero values in the Jacobian Matrix.
  /// This allows to avoid the recomputation of such indexes during the
  /// Jacobian evaluation.
  void IDAInstance::computeNNZ()
  {
    assert(initialized && "The IDA instance has not been initialized yet");
    nonZeroValuesNumber = 0;

    std::vector<int64_t> equationIndices;

    for (size_t eq = 0; eq < getNumOfVectorizedEquations(); ++eq) {
      // Initialize the multidimensional interval of the vector equation.
      size_t equationRank = equationRanges[eq].size();
      equationIndices.resize(equationRank);

      for (size_t i = 0; i < equationRank; ++i) {
        const auto& iterationRange = equationRanges[eq][i];
        size_t beginIndex = iterationRange.begin;
        equationIndices[i] = beginIndex;
      }

      // For every scalar equation in the vector equation.
      do {
        // Compute the column indexes that may be non-zeros
        nonZeroValuesNumber += computeJacobianColumns(eq, equationIndices.data()).size();
      } while (advanceIndices(equationIndices, equationRanges[eq]));
    }
  }

  void IDAInstance::copyVariablesFromMARCO(
      N_Vector algebraicAndStateVariablesVector,
      N_Vector derivativeVariablesVector)
  {
    auto* algebraicAndStateVariablesPtr =
        N_VGetArrayPointer(algebraicAndStateVariablesVector);

    auto* derivativeVariablesPtr =
        N_VGetArrayPointer(derivativeVariablesVector);

    for (size_t i = 0; i < getNumOfArrayVariables(); ++i) {
      size_t arrayVariableOffset = variableOffsets[i];
      const auto& dimensions = variablesDimensions[i];

      for (auto indices = dimensions.indicesBegin(), end = dimensions.indicesEnd(); indices != end; ++indices) {
        size_t scalarVariableOffset = getFlatIndex(dimensions, *indices);
        size_t index = arrayVariableOffset + scalarVariableOffset;
        auto getterFn = algebraicAndStateVariablesGetters[i];

        auto value = static_cast<realtype>(
            getterFn(algebraicAndStateVariables[i], *indices));

        algebraicAndStateVariablesPtr[index] = value;

        auto derivativeVariablePositionIt = stateVariablesMapping.find(i);

        if (derivativeVariablePositionIt != stateVariablesMapping.end()) {
          size_t derivativeVariablePos = derivativeVariablePositionIt->second;
          auto getterFn = derivativeVariablesGetters[derivativeVariablePos];

          auto value = static_cast<realtype>(
              getterFn(derivativeVariables[derivativeVariablePos], *indices));

          derivativeVariablesPtr[index] = value;
        }
      }
    }
  }

  void IDAInstance::copyVariablesIntoMARCO(
      N_Vector algebraicAndStateVariablesVector,
      N_Vector derivativeVariablesVector)
  {
    auto* algebraicAndStateVariablesPtr =
        N_VGetArrayPointer(algebraicAndStateVariablesVector);

    auto* derivativeVariablesPtr =
        N_VGetArrayPointer(derivativeVariablesVector);

    for (size_t i = 0; i < getNumOfArrayVariables(); ++i) {
      size_t arrayVariableOffset = variableOffsets[i];
      const auto& dimensions = variablesDimensions[i];

      for (auto indices = dimensions.indicesBegin(), end = dimensions.indicesEnd(); indices != end; ++indices) {
        size_t scalarVariableOffset = getFlatIndex(dimensions, *indices);
        size_t index = arrayVariableOffset + scalarVariableOffset;

        auto setterFn = algebraicAndStateVariablesSetters[i];
        auto value = static_cast<double>(algebraicAndStateVariablesPtr[index]);
        setterFn(algebraicAndStateVariables[i], value, *indices);

        auto derivativeVariablePositionIt = stateVariablesMapping.find(i);

        if (derivativeVariablePositionIt != stateVariablesMapping.end()) {
          size_t derivativeVariablePos = derivativeVariablePositionIt->second;
          auto setterFn = derivativeVariablesSetters[derivativeVariablePos];
          auto value = static_cast<double>(derivativeVariablesPtr[index]);
          setterFn(derivativeVariables[derivativeVariablePos], value, *indices);
        }
      }
    }
  }

  void IDAInstance::scalarEquationsParallelIteration(
      std::function<void(
          size_t equation,
          const std::vector<int64_t>& equationIndices)> processFn)
  {
    size_t processedEquations = 0;
    std::vector<int64_t> equationIndices;
    std::mutex mutex;

    auto setBeginIndices = [&](size_t eq) {
      size_t equationRank = getEquationRank(eq);
      equationIndices.resize(equationRank);

      for (size_t i = 0; i < equationRank; ++i) {
        equationIndices[i] = equationRanges[eq][i].begin;
      }
    };

    // Function to advance the indices by one, or move to the next equation if
    // the current one has been fully visited.
    auto getEquationAndAdvance =
        [&](size_t& eq, std::vector<int64_t>& indices) {
          std::lock_guard<std::mutex> lockGuard(mutex);

          if (processedEquations >= getNumOfVectorizedEquations()) {
            return false;
          }

          eq = equationsProcessingOrder[processedEquations];
          indices = equationIndices;

          if (!advanceIndices(equationIndices, equationRanges[eq])) {
            if (++processedEquations < getNumOfVectorizedEquations()) {
              setBeginIndices(equationsProcessingOrder[processedEquations]);
            }
          }

          return true;
        };

    setBeginIndices(equationsProcessingOrder[processedEquations]);

    for (unsigned int i = 0, e = threadPool.getNumOfThreads(); i < e; ++i) {
      threadPool.async([&]() {
        size_t equation;
        std::vector<int64_t> indices;

        while (getEquationAndAdvance(equation, indices)) {
          processFn(equation, indices);
        }
      });
    }

    threadPool.wait();
  }

  void IDAInstance::printStatistics() const
  {
    if (scalarEquationsNumber == 0) {
      return;
    }

    if (getOptions().printJacobian) {
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
    std::cerr << getNumOfVectorizedEquations() << std::endl;
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
    for (size_t eq = 0; eq < getNumOfVectorizedEquations(); ++eq) {
      // Initialize the multidimensional interval of the vector equation
      int64_t indexes[equationRanges[eq].size()];

      for (size_t i = 0; i < equationRanges[eq].size(); ++i) {
        indexes[i] = equationRanges[eq][i].begin;
      }

      // For every scalar equation in the vector equation
      do {
        std::cerr << "│";

        // Get the column indexes that may be non-zeros.
        std::set<size_t> columnIndexesSet;

        for (auto& access : variableAccesses[eq]) {
          const VariableDimensions& dimensions = variablesDimensions[access.first];
          size_t varOffset = computeOffset(indexes, dimensions, access.second);
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
      } while (advanceIndices(indexes, equationRanges[eq]));
    }
  }

  bool IDAInstance::idaInit()
  {
    auto retVal = IDAInit(idaMemory, residualFunction, startTime, variablesVector, derivativesVector);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDAInit - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_MEM_FAIL) {
      std::cerr << "IDAInit - A memory allocation request has failed" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDAInit - An input argument to IDAInit has an illegal value" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSVTolerances()
  {
    auto retVal = IDASVtolerances(idaMemory, getOptions().relativeTolerance, tolerancesVector);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASVtolerances - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_NO_MALLOC) {
      std::cerr << "IDASVtolerances - The allocation function IDAInit(has not been called" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASVtolerances - The relative error tolerance was negative or the absolute tolerance vector had a negative component" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetLinearSolver()
  {
    auto retVal = IDASetLinearSolver(idaMemory, linearSolver, sparseMatrix);

    if (retVal == IDALS_MEM_NULL) {
      std::cerr << "IDASetLinearSolver - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDALS_ILL_INPUT) {
      std::cerr << "IDASetLinearSolver - The IDALS interface is not compatible with the LS or J input objects or is incompatible with the N_Vector object passed to IDAInit" << std::endl;
      return false;
    }

    if (retVal == IDALS_SUNLS_FAIL) {
      std::cerr << "IDASetLinearSolver - A call to the LS object failed" << std::endl;
      return false;
    }

    if (retVal == IDALS_MEM_FAIL) {
      std::cerr << "IDASetLinearSolver - A memory allocation request failed" << std::endl;
      return false;
    }

    return retVal == IDALS_SUCCESS;
  }

  bool IDAInstance::idaSetUserData()
  {
    auto retVal = IDASetUserData(idaMemory, this);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetUserData - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetMaxNumSteps()
  {
    auto retVal = IDASetMaxNumSteps(idaMemory, getOptions().maxSteps);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxNumSteps - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASetMaxNumSteps - Either hmax is not positive or it is smaller than the minimum allowable step" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetInitialStepSize()
  {
    auto retVal = IDASetInitStep(idaMemory, getOptions().initialStepSize);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetInitStep - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetMinStepSize()
  {
    auto retVal = IDASetMinStep(idaMemory, getOptions().minStepSize);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMinStep - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASetMinStep - hmin is negative" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetMaxStepSize()
  {
    auto retVal = IDASetMaxStep(idaMemory, getOptions().maxStepSize);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxStep - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASetMaxStep - Either hmax is not positive or it is smaller than the minimum allowable step" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetStopTime()
  {
    auto retVal = IDASetStopTime(idaMemory, endTime);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxStep - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASetMaxStep - The value of tstop is not beyond the current t value" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetMaxErrTestFails()
  {
    auto retVal = IDASetMaxErrTestFails(idaMemory, getOptions().maxErrTestFails);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxErrTestFails - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetSuppressAlg()
  {
    auto retVal = IDASetSuppressAlg(idaMemory, getOptions().suppressAlg);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetSuppressAlg - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetId()
  {
    auto retVal = IDASetId(idaMemory, idVector);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetId - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetJacobianFunction()
  {
    auto retVal = IDASetJacFn(idaMemory, jacobianMatrix);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetJacFn - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDALS_LMEM_NULL) {
      std::cerr << "IDASetJacFn - The IDALS linear solver interface has not been initialized" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetMaxNonlinIters()
  {
    auto retVal = IDASetMaxNonlinIters(idaMemory, getOptions().maxNonlinIters);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxNonlinIters - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_MEM_FAIL) {
      std::cerr << "IDASetMaxNonlinIters - The SUNNonlinearSolver object is NULL" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetMaxConvFails()
  {
    auto retVal = IDASetMaxConvFails(idaMemory, getOptions().maxConvFails);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxConvFails - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetNonlinConvCoef()
  {
    auto retVal = IDASetNonlinConvCoef(idaMemory, getOptions().nonlinConvCoef);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetNonlinConvCoef - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASetNonlinConvCoef - The value of nlscoef is <= 0" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetNonlinConvCoefIC()
  {
    auto retVal = IDASetNonlinConvCoefIC(idaMemory, getOptions().nonlinConvCoefIC);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetNonlinConvCoefIC - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASetNonlinConvCoefIC - The epiccon factor is <= 0" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetMaxNumStepsIC()
  {
    auto retVal = IDASetMaxNumStepsIC(idaMemory, getOptions().maxStepsIC);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxNumStepsIC - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASetMaxNumStepsIC - maxnh is non-positive" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetMaxNumJacsIC()
  {
    auto retVal = IDASetMaxNumJacsIC(idaMemory, getOptions().maxNumJacsIC);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxNumJacsIC - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASetMaxNumJacsIC - maxnj is non-positive" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetMaxNumItersIC()
  {
    auto retVal = IDASetMaxNumItersIC(idaMemory, getOptions().maxNumItersIC);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxNumItersIC - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    if (retVal == IDA_ILL_INPUT) {
      std::cerr << "IDASetMaxNumItersIC - maxnit is non-positive" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  bool IDAInstance::idaSetLineSearchOffIC()
  {
    auto retVal = IDASetLineSearchOffIC(idaMemory, getOptions().lineSearchOff);

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetLineSearchOffIC - The ida_mem pointer is NULL" << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }
}

//===---------------------------------------------------------------------===//
// Exported functions
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// idaCreate

static void* idaCreate_pvoid(int64_t scalarEquationsNumber)
{
  auto* instance = new IDAInstance(scalarEquationsNumber);
  return static_cast<void*>(instance);
}

RUNTIME_FUNC_DEF(idaCreate, PTR(void), int64_t)

//===---------------------------------------------------------------------===//
// idaInit

static void idaInit_void(void* instance)
{
  [[maybe_unused]] bool result =
      static_cast<IDAInstance*>(instance)->initialize();

  assert(result && "Can't initialize the IDA instance");
}

RUNTIME_FUNC_DEF(idaInit, void, PTR(void))

//===---------------------------------------------------------------------===//
// idaCalcIC

static void idaCalcIC_void(void* instance)
{
  [[maybe_unused]] bool result = static_cast<IDAInstance*>(instance)->calcIC();
  assert(result && "Can't compute the initial values of the variables");
}

RUNTIME_FUNC_DEF(idaCalcIC, void, PTR(void))

//===---------------------------------------------------------------------===//
// idaStep

static void idaStep_void(void* instance)
{
  [[maybe_unused]] bool result = static_cast<IDAInstance*>(instance)->step();
  assert(result && "IDA step failed");
}

RUNTIME_FUNC_DEF(idaStep, void, PTR(void))

//===---------------------------------------------------------------------===//
// idaFree

static void idaFree_void(void* instance)
{
  delete static_cast<IDAInstance*>(instance);
}

RUNTIME_FUNC_DEF(idaFree, void, PTR(void))

//===---------------------------------------------------------------------===//
// idaSetStartTime

static void idaSetStartTime_void(void* instance, double startTime)
{
  static_cast<IDAInstance*>(instance)->setStartTime(startTime);
}

RUNTIME_FUNC_DEF(idaSetStartTime, void, PTR(void), double)

//===---------------------------------------------------------------------===//
// idaSetEndTime

static void idaSetEndTime_void(void* instance, double endTime)
{
  static_cast<IDAInstance*>(instance)->setEndTime(endTime);
}

RUNTIME_FUNC_DEF(idaSetEndTime, void, PTR(void), double)

//===---------------------------------------------------------------------===//
// idaSetTimeStep

static void idaSetTimeStep_void(void* instance, double timeStep)
{
  static_cast<IDAInstance*>(instance)->setTimeStep(timeStep);
}

RUNTIME_FUNC_DEF(idaSetTimeStep, void, PTR(void), double)

//===---------------------------------------------------------------------===//
// idaGetCurrentTime

static double idaGetCurrentTime_f64(void* instance)
{
  return static_cast<double>(
      static_cast<IDAInstance*>(instance)->getCurrentTime());
}

RUNTIME_FUNC_DEF(idaGetCurrentTime, double, PTR(void))

//===---------------------------------------------------------------------===//
// idaAddParametricVariable

static void idaAddParametricVariable_void(void* instance, void* variable)
{
  static_cast<IDAInstance*>(instance)->addParametricVariable(variable);
}

RUNTIME_FUNC_DEF(idaAddParametricVariable, void, PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaAddAlgebraicVariable

static int64_t idaAddAlgebraicVariable_i64(
    void* instance,
    void* variable,
    int64_t* dimensions,
    int64_t rank,
    void* getter,
    void* setter)
{
  return static_cast<IDAInstance*>(instance)->addAlgebraicVariable(
      variable, dimensions, rank,
      reinterpret_cast<VariableGetter>(getter),
      reinterpret_cast<VariableSetter>(setter));
}

RUNTIME_FUNC_DEF(idaAddAlgebraicVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaAddStateVariable

static int64_t idaAddStateVariable_i64(
    void* instance,
    void* variable,
    int64_t* dimensions,
    int64_t rank,
    void* getter,
    void* setter)
{
  return static_cast<IDAInstance*>(instance)->addStateVariable(
      variable, dimensions, rank,
      reinterpret_cast<VariableGetter>(getter),
      reinterpret_cast<VariableSetter>(setter));
}

RUNTIME_FUNC_DEF(idaAddStateVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaSetDerivative

static void idaSetDerivative_void(
    void* instance,
    int64_t stateVariable,
    void* derivative,
    void* getter,
    void* setter)
{
  static_cast<IDAInstance*>(instance)->setDerivative(
      stateVariable, derivative,
      reinterpret_cast<VariableGetter>(getter),
      reinterpret_cast<VariableSetter>(setter));
}

RUNTIME_FUNC_DEF(idaSetDerivative, void, PTR(void), int64_t, PTR(void), PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaAddVariableAccess

static void idaAddVariableAccess_void(
    void* instance,
    int64_t equationIndex,
    int64_t variableIndex,
    int64_t* access,
    int64_t rank)
{
  static_cast<IDAInstance*>(instance)->addVariableAccess(
      equationIndex, variableIndex, access, rank);
}

RUNTIME_FUNC_DEF(idaAddVariableAccess, void, PTR(void), int64_t, int64_t, PTR(int64_t), int64_t)

//===---------------------------------------------------------------------===//
// idaAddEquation

static int64_t idaAddEquation_i64(
    void* instance,
    int64_t* ranges,
    int64_t rank,
    int64_t writtenVariable,
    int64_t* writeAccess)
{
  return static_cast<IDAInstance*>(instance)->addEquation(
      ranges, rank, writtenVariable, writeAccess);
}

RUNTIME_FUNC_DEF(idaAddEquation, int64_t, PTR(void), PTR(int64_t), int64_t, int64_t, PTR(int64_t))

//===---------------------------------------------------------------------===//
// idaSetResidual

static void idaSetResidual_void(
    void* instance,
    int64_t equationIndex,
    void* residualFunction)
{
  static_cast<IDAInstance*>(instance)->setResidualFunction(
      equationIndex,
      reinterpret_cast<ResidualFunction>(residualFunction));
}

RUNTIME_FUNC_DEF(idaSetResidual, void, PTR(void), int64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaAddJacobian

static void idaAddJacobian_void(
    void* instance,
    int64_t equationIndex,
    int64_t variableIndex,
    void* jacobianFunction)
{
  static_cast<IDAInstance*>(instance)->addJacobianFunction(
      equationIndex, variableIndex,
      reinterpret_cast<JacobianFunction>(jacobianFunction));
}

RUNTIME_FUNC_DEF(idaAddJacobian, void, PTR(void), int64_t, int64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaPrintStatistics

static void printStatistics_void(void* instance)
{
  static_cast<IDAInstance*>(instance)->printStatistics();
}

RUNTIME_FUNC_DEF(printStatistics, void, PTR(void))
