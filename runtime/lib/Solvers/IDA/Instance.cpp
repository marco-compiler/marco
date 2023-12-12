#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "marco/Runtime/Solvers/IDA/Profiler.h"
#include "marco/Runtime/Support/MemoryManagement.h"
#include <atomic>
#include <cassert>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>

using namespace ::marco::runtime;
using namespace ::marco::runtime::ida;

//===---------------------------------------------------------------------===//
// Utilities
//===---------------------------------------------------------------------===//

static void printIndices(const std::vector<int64_t>& indices)
{
  std::cerr << "[";

  for (size_t i = 0, e = indices.size(); i < e; ++i) {
    if (i != 0) {
      std::cerr << ", ";
    }

    std::cerr << indices[i];
  }

  std::cerr << "]";
}

static void printIndices(const std::vector<uint64_t>& indices)
{
  std::cerr << "[";

  for (size_t i = 0, e = indices.size(); i < e; ++i) {
    if (i != 0) {
      std::cerr << ", ";
    }

    std::cerr << indices[i];
  }

  std::cerr << "]";
}

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

  uint64_t& VariableDimensions::operator[](size_t index)
  {
    return dimensions[index];
  }

  const uint64_t& VariableDimensions::operator[](size_t index) const
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
    return std::none_of(
        dimensions.begin(), dimensions.end(), [](const auto& dimension) {
          return dimension == 0;
        });
  }

  VariableIndicesIterator::~VariableIndicesIterator()
  {
    delete[] indices;
  }

  VariableIndicesIterator VariableIndicesIterator::begin(
      const VariableDimensions& dimensions)
  {
    VariableIndicesIterator result(dimensions);

    for (size_t i = 0; i < dimensions.rank(); ++i) {
      result.indices[i] = 0;
    }

    return result;
  }

  VariableIndicesIterator VariableIndicesIterator::end(
      const VariableDimensions& dimensions)
  {
    VariableIndicesIterator result(dimensions);

    for (size_t i = 0; i < dimensions.rank(); ++i) {
      result.indices[i] = dimensions[i];
    }

    return result;
  }

  bool VariableIndicesIterator::operator==(
      const VariableIndicesIterator& it) const
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

  bool VariableIndicesIterator::operator!=(
      const VariableIndicesIterator& it) const
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

  const uint64_t* VariableIndicesIterator::operator*() const
  {
    return indices;
  }

  VariableIndicesIterator::VariableIndicesIterator(
      const VariableDimensions& dimensions)
      : dimensions(&dimensions)
  {
    indices = new uint64_t[dimensions.rank()];
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

static bool advanceVariableIndices(
    uint64_t* indices, const VariableDimensions& dimensions)
{
  for (size_t i = 0, e = dimensions.rank(); i < e; ++i) {
    size_t pos = e - i - 1;
    ++indices[pos];

    if (indices[pos] == dimensions[pos]) {
      indices[pos] = 0;
    } else {
      return true;
    }
  }

  for (size_t i = 0, e = dimensions.rank(); i < e; ++i) {
    indices[i] = dimensions[i];
  }

  return false;
}

static bool advanceVariableIndices(
    std::vector<uint64_t>& indices, const VariableDimensions& dimensions)
{
  return advanceVariableIndices(indices.data(), dimensions);
}

static bool advanceVariableIndicesUntil(
    std::vector<uint64_t>& indices,
    const VariableDimensions& dimensions,
    const std::vector<uint64_t>& end)
{
  assert(indices.size() == end.size());

  if (!advanceVariableIndices(indices, dimensions)) {
    return false;
  }

  return indices != end;
}

/// Given an array of indices and the dimensions of an equation, increase the
/// indices within the induction bounds of the equation. Return false if the
/// indices exceed the equation bounds, which means the computation has
/// finished, true otherwise.
static bool advanceEquationIndices(
    int64_t* indices, const MultidimensionalRange& ranges)
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

  for (size_t i = 0, e = ranges.size(); i < e; ++i) {
    indices[i] = ranges[i].end;
  }

  return false;
}

static bool advanceEquationIndices(
    std::vector<int64_t>& indices, const MultidimensionalRange& ranges)
{
  return advanceEquationIndices(indices.data(), ranges);
}

static bool advanceEquationIndicesUntil(
    std::vector<int64_t>& indices,
    const MultidimensionalRange& ranges,
    const std::vector<int64_t>& end)
{
  assert(indices.size() == end.size());

  if (!advanceEquationIndices(indices, ranges)) {
    return false;
  }

  return indices != end;
}

/// Get the flat index corresponding to a multidimensional access.
/// Example:
///   x[d1][d2][d3]
///   x[i][j][k] -> x[i * d2 * d3 + j * d3 + k]
static uint64_t getVariableFlatIndex(
    const VariableDimensions& dimensions,
    const uint64_t* indices)
{
  uint64_t offset = indices[0];

  for (size_t i = 1, e = dimensions.rank(); i < e; ++i) {
    offset = offset * dimensions[i] + indices[i];
  }

  return offset;
}

static uint64_t getVariableFlatIndex(
    const VariableDimensions& dimensions,
    const std::vector<uint64_t>& indices)
{
  assert(indices.size() == dimensions.rank());
  return getVariableFlatIndex(dimensions, indices.data());
}

static uint64_t getEquationFlatIndex(
    const std::vector<int64_t>& indices,
    const MultidimensionalRange& ranges)
{
  assert(indices[0] >= ranges[0].begin);
  uint64_t offset = indices[0] - ranges[0].begin;

  for (size_t i = 1, e = ranges.size(); i < e; ++i) {
    assert(ranges[i].end > ranges[i].begin);
    offset = offset * (ranges[i].end - ranges[i].begin) +
        (indices[i] - ranges[i].begin);
  }

  assert(offset >= 0);
  return offset;
}

static void getEquationIndicesFromFlatIndex(
    uint64_t flatIndex,
    std::vector<int64_t>& result,
    const MultidimensionalRange& ranges)
{
  result.resize(ranges.size());
  uint64_t size = 1;

  for (size_t i = 1, e = ranges.size(); i < e; ++i) {
    assert(ranges[i].end > ranges[i].begin);
    size *= ranges[i].end - ranges[i].begin;
  }

  for (size_t i = 1, e = ranges.size(); i < e; ++i) {
    result[i - 1] =
        static_cast<int64_t>(flatIndex / size) + ranges[i - 1].begin;

    flatIndex %= size;
    assert(ranges[i].end > ranges[i].begin);
    size /= ranges[i].end - ranges[i].begin;
  }

  result[ranges.size() - 1] =
      static_cast<int64_t>(flatIndex) + ranges.back().begin;

  assert(size == 1);

  assert(([&]() -> bool {
           for (size_t i = 0, e = result.size(); i < e; ++i) {
             if (result[i] < ranges[i].begin ||
                 result[i] >= ranges[i].end) {
               return false;
             }
           }

           return true;
         }()) && "Wrong index unflattening result");
}

//===---------------------------------------------------------------------===//
// Solver
//===---------------------------------------------------------------------===//

namespace marco::runtime::ida
{
  IDAInstance::IDAInstance()
      : startTime(getOptions().startTime),
        endTime(getOptions().endTime),
        timeStep(getOptions().timeStep)
  {
    // Initially there is no variable in the instance.
    variableOffsets.push_back(0);

    if (getOptions().debug) {
      std::cerr << "Instance created" << std::endl;
    }
  }

  IDAInstance::~IDAInstance()
  {
    if (getNumOfScalarEquations() != 0) {
      N_VDestroy(variablesVector);
      N_VDestroy(derivativesVector);
      N_VDestroy(idVector);
      N_VDestroy(tolerancesVector);

      IDAFree(&idaMemory);
      SUNLinSolFree(linearSolver);
      SUNMatDestroy(sparseMatrix);
    }

    if (getOptions().debug) {
      std::cerr << "Instance destroyed" << std::endl;
    }
  }

  void IDAInstance::setStartTime(double time)
  {
    startTime = time;

    if (getOptions().debug) {
      std::cerr << "Start time set to " << startTime << std::endl;
    }
  }

  void IDAInstance::setEndTime(double time)
  {
    endTime = time;

    if (getOptions().debug) {
      std::cerr << "End time set to " << endTime << std::endl;
    }
  }

  void IDAInstance::setTimeStep(double step)
  {
    assert(step > 0);
    timeStep = step;

    if (getOptions().debug) {
      std::cerr << "Time step set to " << timeStep << std::endl;
    }
  }

  Variable IDAInstance::addAlgebraicVariable(
      uint64_t rank,
      const uint64_t* dimensions,
      VariableGetter getterFunction,
      VariableSetter setterFunction,
      const char* name)
  {
    if (getOptions().debug) {
      std::cerr << "Adding algebraic variable";

      if (name != nullptr) {
        std::cerr << " \"" << name << "\"";
      }

      std::cerr << std::endl;
    }

    // Add variable offset and dimensions.
    assert(variableOffsets.size() == variablesDimensions.size() + 1);

    VariableDimensions varDimension(rank);
    uint64_t flatSize = 1;

    for (uint64_t i = 0; i < rank; ++i) {
      flatSize *= dimensions[i];
      varDimension[i] = dimensions[i];
    }

    variablesDimensions.push_back(std::move(varDimension));

    size_t offset = variableOffsets.back();
    variableOffsets.push_back(offset + flatSize);

    // Store the getter and setter functions.
    algebraicAndStateVariablesGetters.push_back(getterFunction);
    algebraicAndStateVariablesSetters.push_back(setterFunction);

    // Return the index of the variable.
    Variable id = getNumOfArrayVariables() - 1;

    if (getOptions().debug) {
      std::cerr << "  - ID: " << id << std::endl;
      std::cerr << "  - Rank: " << rank << std::endl;
      std::cerr << "  - Dimensions: [";

      for (uint64_t i = 0; i < rank; ++i) {
        if (i != 0) {
          std::cerr << ",";
        }

        std::cerr << dimensions[i];
      }

      std::cerr << "]" << std::endl;
      std::cerr << "  - Getter function address: "
                << reinterpret_cast<void*>(getterFunction) << std::endl;
      std::cerr << "  - Setter function address: "
                << reinterpret_cast<void*>(setterFunction) << std::endl;
    }

    return id;
  }

  Variable IDAInstance::addStateVariable(
      uint64_t rank,
      const uint64_t* dimensions,
      VariableGetter stateGetterFunction,
      VariableSetter stateSetterFunction,
      VariableGetter derivativeGetterFunction,
      VariableSetter derivativeSetterFunction,
      const char* name)
  {
    if (getOptions().debug) {
      std::cerr << "Adding state variable";

      if (name != nullptr) {
        std::cerr << " \"" << name << "\"";
      }

      std::cerr << std::endl;
    }

    assert(variableOffsets.size() == getNumOfArrayVariables() + 1);

    // Add variable offset and dimensions.
    VariableDimensions variableDimensions(rank);
    uint64_t flatSize = 1;

    for (uint64_t i = 0; i < rank; ++i) {
      flatSize *= dimensions[i];
      variableDimensions[i] = dimensions[i];
    }

    variablesDimensions.push_back(variableDimensions);

    // Store the position of the start of the flattened array.
    uint64_t offset = variableOffsets.back();
    variableOffsets.push_back(offset + flatSize);

    // Store the getter and setter functions for the state variable.
    algebraicAndStateVariablesGetters.push_back(stateGetterFunction);
    algebraicAndStateVariablesSetters.push_back(stateSetterFunction);

    // Store the getter and setter functions for the derivative variable.
    derivativeVariablesGetters.push_back(derivativeGetterFunction);
    derivativeVariablesSetters.push_back(derivativeSetterFunction);

    // Return the index of the variable.
    Variable id = getNumOfArrayVariables() - 1;
    stateVariablesMapping[id] = derivativeVariablesGetters.size() - 1;

    if (getOptions().debug) {
      std::cerr << "  - ID: " << id << std::endl;
      std::cerr << "  - Rank: " << rank << std::endl;
      std::cerr << "  - Dimensions: [";

      for (uint64_t i = 0; i < rank; ++i) {
        if (i != 0) {
          std::cerr << ",";
        }

        std::cerr << dimensions[i];
      }

      std::cerr << "]" << std::endl;
      std::cerr << "  - State variable getter function address: "
                << reinterpret_cast<void*>(stateGetterFunction) << std::endl;
      std::cerr << "  - State variable setter function address: "
                << reinterpret_cast<void*>(stateSetterFunction) << std::endl;
      std::cerr << "  - Derivative variable getter function address: "
                << reinterpret_cast<void*>(derivativeGetterFunction)
                << std::endl;
      std::cerr << "  - Derivative variable setter function address: "
                << reinterpret_cast<void*>(derivativeSetterFunction)
                << std::endl;
    }

    return id;
  }

  Equation IDAInstance::addEquation(
      const int64_t* ranges,
      uint64_t equationRank,
      Variable writtenVariable,
      AccessFunction writeAccess,
      const char* stringRepresentation)
  {
    if (getOptions().debug) {
      std::cerr << "Adding equation";

      if (stringRepresentation != nullptr) {
        std::cerr << " \"" << stringRepresentation << "\"";
      }

      std::cerr << std::endl;
    }

    // Add the start and end dimensions of the current equation.
    MultidimensionalRange eqRanges = {};

    for (size_t i = 0, e = equationRank * 2; i < e; i += 2) {
      int64_t begin = ranges[i];
      int64_t end = ranges[i + 1];
      eqRanges.push_back({ begin, end });
    }

    equationRanges.push_back(eqRanges);

    // Add the write access.
    writeAccesses.emplace_back(writtenVariable, writeAccess);

    // Return the index of the equation.
    Equation id = getNumOfVectorizedEquations() - 1;

    if (getOptions().debug) {
      std::cerr << "  - ID: " << id << std::endl;
      std::cerr << "  - Rank: " << equationRank << std::endl;
      std::cerr << "  - Ranges: [";

      for (uint64_t i = 0; i < equationRank; ++i) {
        if (i != 0) {
          std::cerr << ",";
        }

        std::cerr << "[" << ranges[i * 2] << "," << (ranges[i * 2 + 1] - 1) << "]";
      }

      std::cerr << "]" << std::endl;
      std::cerr << "  - Written variable ID: " << writtenVariable << std::endl;
      std::cerr << "  - Write access function address: "
                << reinterpret_cast<void*>(writeAccess) << std::endl;
    }

    return id;
  }

  void IDAInstance::addVariableAccess(
      Equation equation,
      Variable variable,
      AccessFunction accessFunction)
  {
    if (getOptions().debug) {
      std::cerr << "Adding access information" << std::endl;
      std::cerr << "  - Equation: " << equation << std::endl;
      std::cerr << "  - Variable: " << variable << std::endl;
      std::cerr << "  - Access function address: "
                << reinterpret_cast<void*>(accessFunction) << std::endl;
    }

    assert(equation < getNumOfVectorizedEquations());
    assert(variable < getNumOfArrayVariables());

    precomputedAccesses = true;

    if (variableAccesses.size() <= (size_t) equation) {
      variableAccesses.resize(equation + 1);
    }

    auto& varAccessList = variableAccesses[equation];
    varAccessList.emplace_back(variable, accessFunction);
  }

  void IDAInstance::setResidualFunction(
      Equation equation,
      ResidualFunction residualFunction)
  {
    if (getOptions().debug) {
      std::cerr << "Setting residual function for equation " << equation
                << ". Address: " << reinterpret_cast<void*>(residualFunction)
                << std::endl;
    }

    if (residualFunctions.size() <= equation) {
      residualFunctions.resize(equation + 1, nullptr);
    }

    residualFunctions[equation] = residualFunction;
  }

  void IDAInstance::addJacobianFunction(
      Equation equation,
      Variable variable,
      JacobianFunction jacobianFunction)
  {
    if (getOptions().debug) {
      std::cerr << "Setting jacobian function for equation " << equation
                << " and variable " << variable << ". Address: "
                << reinterpret_cast<void*>(jacobianFunction) << std::endl;
    }

    if (jacobianFunctions.size() <= equation) {
      jacobianFunctions.resize(equation + 1, {});
    }

    if (jacobianFunctions[equation].size() <= variable) {
      jacobianFunctions[equation].resize(variable + 1, nullptr);
    }

    jacobianFunctions[equation][variable] = jacobianFunction;
  }

  bool IDAInstance::initialize()
  {
    assert(!initialized && "The IDA instance has already been initialized");

    if (getOptions().debug) {
      std::cerr << "Performing initialization" << std::endl;
    }

    currentTime = startTime;

    // Compute the number of scalar variables.
    scalarVariablesNumber = 0;

    for (Variable var = 0, e = getNumOfArrayVariables(); var < e; ++var) {
      scalarVariablesNumber += getVariableFlatSize(var);
    }

    // Compute the number of scalar equations.
    scalarEquationsNumber = 0;

    for (Equation eq = 0, e = getNumOfVectorizedEquations(); eq < e; ++eq) {
      scalarEquationsNumber += getEquationFlatSize(eq);
    }

    assert(getNumOfScalarVariables() == getNumOfScalarEquations() &&
           "Unbalanced system");

    if (scalarEquationsNumber == 0) {
      // IDA has nothing to solve.
      initialized = true;
      return true;
    }

    // Create the SUNDIALS context.
    if (SUNContext_Create(nullptr, &ctx) != 0) {
      return false;
    }

    // Create and initialize the variables vector.
    variablesVector = N_VNew_Serial(
            static_cast<sunindextype>(scalarVariablesNumber), ctx);

    assert(checkAllocation(
            static_cast<void*>(variablesVector), "N_VNew_Serial"));

    for (uint64_t i = 0; i < scalarVariablesNumber; ++i) {
      N_VGetArrayPointer(variablesVector)[i] = 0;
    }

    // Create and initialize the derivatives vector.
    derivativesVector = N_VNew_Serial(
            static_cast<sunindextype>(scalarVariablesNumber), ctx);

    assert(checkAllocation(
            static_cast<void*>(derivativesVector), "N_VNew_Serial"));

    for (uint64_t i = 0; i < scalarVariablesNumber; ++i) {
      N_VGetArrayPointer(derivativesVector)[i] = 0;
    }

    // Create and initialize the IDs vector.
    idVector = N_VNew_Serial(
            static_cast<sunindextype>(scalarVariablesNumber), ctx);

    assert(checkAllocation(static_cast<void*>(idVector), "N_VNew_Serial"));

    for (Variable var = 0; var < getNumOfArrayVariables(); ++var) {
      VariableKind variableKind = getVariableKind(var);
      uint64_t arrayOffset = variableOffsets[var];
      uint64_t flatSize = getVariableFlatSize(var);

      for (uint64_t scalarOffset = 0; scalarOffset < flatSize; ++scalarOffset) {
        uint64_t offset = arrayOffset + scalarOffset;

        if (variableKind == VariableKind::ALGEBRAIC) {
          N_VGetArrayPointer(idVector)[offset] = 0;
        } else if (variableKind == VariableKind::STATE) {
          N_VGetArrayPointer(idVector)[offset] = 1;
        }
      }
    }

    // Create and initialize the tolerances vector.
    tolerancesVector = N_VNew_Serial(
            static_cast<sunindextype>(scalarVariablesNumber), ctx);

    assert(checkAllocation(
            static_cast<void*>(tolerancesVector), "N_VNew_Serial"));

    for (Variable var = 0; var < getNumOfArrayVariables(); ++var) {
      VariableKind variableKind = getVariableKind(var);
      uint64_t arrayOffset = variableOffsets[var];
      uint64_t flatSize = getVariableFlatSize(var);

      for (uint64_t scalarOffset = 0; scalarOffset < flatSize; ++scalarOffset) {
          uint64_t offset = arrayOffset + scalarOffset;

          if (variableKind == VariableKind::ALGEBRAIC) {
            N_VGetArrayPointer(tolerancesVector)[offset] = std::min(
                    getOptions().maxAlgebraicAbsoluteTolerance,
                    getOptions().absoluteTolerance);
          } else if (variableKind == VariableKind::STATE) {
              N_VGetArrayPointer(tolerancesVector)[offset] =
                    getOptions().absoluteTolerance;
          }
      }
    }

    // Determine the order in which the equations must be processed when
    // computing residuals and jacobians.
    assert(getNumOfVectorizedEquations() == writeAccesses.size());
    equationsProcessingOrder.resize(getNumOfVectorizedEquations());

    for (size_t i = 0, e = getNumOfVectorizedEquations(); i < e; ++i) {
      equationsProcessingOrder[i] = i;
    }

    if (getOptions().debug) {
      std::cerr << "Equations processing order: [";

      for (size_t i = 0, e = equationsProcessingOrder.size(); i < e; ++i) {
        if (i != 0) {
          std::cerr << ", ";
        }

        std::cerr << equationsProcessingOrder[i];
      }

      std::cerr << "]" << std::endl;
    }

    // Check that all the residual functions have been set.
    assert(residualFunctions.size() == getNumOfVectorizedEquations());

    assert(std::all_of(
        residualFunctions.begin(), residualFunctions.end(),
        [](const ResidualFunction& function) {
          return function != nullptr;
        }));

    // Check if the IDA instance is not informed about the accesses that all
    // the jacobian functions have been set.
    assert(precomputedAccesses ||
           jacobianFunctions.size() == getNumOfVectorizedEquations());

    assert(precomputedAccesses ||
           std::all_of(
               jacobianFunctions.begin(), jacobianFunctions.end(),
               [&](std::vector<JacobianFunction> functions) {
                 if (functions.size() !=
                        algebraicAndStateVariablesGetters.size()) {
                   return false;
                 }

                 return std::all_of(
                     functions.begin(), functions.end(),
                     [](const JacobianFunction& function) {
                       return function != nullptr;
                     });
               }));

    // Check that all the getters and setters have been set.
    assert(std::none_of(
               algebraicAndStateVariablesGetters.begin(),
               algebraicAndStateVariablesGetters.end(),
               [](VariableGetter getter) {
                 return getter == nullptr;
               }) && "Not all the variable getters have been set");

    assert(std::none_of(
              algebraicAndStateVariablesSetters.begin(),
              algebraicAndStateVariablesSetters.end(),
              [](VariableSetter setter) {
                  return setter == nullptr;
              }) && "Not all the variable setters have been set");

    assert(std::none_of(
               derivativeVariablesGetters.begin(),
               derivativeVariablesGetters.end(),
               [](VariableGetter getter) {
                 return getter == nullptr;
               }) && "Not all the derivative getters have been set");

    assert(std::none_of(
              derivativeVariablesSetters.begin(),
              derivativeVariablesSetters.end(),
              [](VariableSetter setter) {
                  return setter == nullptr;
              }) && "Not all the derivative setters have been set");

    // Reserve the space for data of the jacobian matrix.
    if (getOptions().debug) {
      std::cerr << "Reserving space for the data of the Jacobian matrix"
                << std::endl;
    }

    jacobianMatrixData.resize(scalarEquationsNumber);

    for (Equation eq : equationsProcessingOrder) {
      uint64_t equationRank = getEquationRank(eq);

      std::vector<int64_t> equationIndices;
      getEquationBeginIndices(eq, equationIndices);

      Variable writtenVariable = getWrittenVariable(eq);

      uint64_t writtenVariableRank = getVariableRank(writtenVariable);
      uint64_t writtenVariableArrayOffset = variableOffsets[writtenVariable];

      do {
        std::vector<uint64_t> writtenVariableIndices;
        writtenVariableIndices.resize(writtenVariableRank, 0);

        AccessFunction writeAccessFunction = getWriteAccessFunction(eq);

        writeAccessFunction(
            equationIndices.data(),
            writtenVariableIndices.data());

        if (getOptions().debug) {
          std::cerr << "    Variable indices: ";
          printIndices(writtenVariableIndices);
          std::cerr << std::endl;
        }

        uint64_t equationScalarVariableOffset = getVariableFlatIndex(
            variablesDimensions[writtenVariable],
            writtenVariableIndices);

        uint64_t scalarEquationIndex =
            writtenVariableArrayOffset + equationScalarVariableOffset;

        // Compute the column indexes that may be non-zeros.
        std::vector<JacobianColumn> jacobianColumns =
            computeJacobianColumns(eq, equationIndices.data());

        jacobianMatrixData[scalarEquationIndex].resize(jacobianColumns.size());

        if (getOptions().debug) {
          std::cerr << "  - Equation " << eq << std::endl;
          std::cerr << "    Equation indices: ";
          printIndices(equationIndices);
          std::cerr << std::endl;

          std::cerr << "    Variable indices: ";
          printIndices(writtenVariableIndices);
          std::cerr << std::endl;

          std::cerr << "    Scalar equation index: " << scalarEquationIndex
                    << std::endl;

          std::cerr << "    Number of possibly non-zero columns: "
                    << jacobianColumns.size() << std::endl;
        }
      } while (advanceEquationIndices(equationIndices, equationRanges[eq]));
    }

    // Compute the total amount of non-zero values in the Jacobian Matrix.
    computeNNZ();

    // Compute the equation chunks for each thread.
    computeThreadChunks();

    // Initialize the values of the variables living inside IDA.
    copyVariablesFromMARCO(variablesVector, derivativesVector);

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
        static_cast<sunindextype>(scalarEquationsNumber),
        static_cast<sunindextype>(scalarEquationsNumber),
        static_cast<sunindextype>(nonZeroValuesNumber),
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

    initialized = true;

    if (getOptions().debug) {
      std::cerr << "Initialization completed" << std::endl;
    }

    return true;
  }

  bool IDAInstance::calcIC()
  {
    if (!initialized) {
      if (!initialize()) {
        return false;
      }
    }

    if (getNumOfScalarEquations() == 0) {
      // IDA has nothing to solve
      return true;
    }

    realtype firstOutTime = (endTime - startTime) /
        getOptions().timeScalingFactorInit;

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

    auto getConsistentIcRetVal =
        IDAGetConsistentIC(idaMemory, variablesVector, derivativesVector);

    if (getConsistentIcRetVal != IDA_SUCCESS) {
      if (getConsistentIcRetVal == IDA_ILL_INPUT) {
        std::cerr << "IDAGetConsistentIC - Called before the first IDASolve" << std::endl;
      } else if (getConsistentIcRetVal == IDA_MEM_NULL) {
        std::cerr << "IDAGetConsistentIC - The ida_mem pointer is NULL" << std::endl;
      }

      return false;
    }

    copyVariablesIntoMARCO(variablesVector, derivativesVector);
    return true;
  }

  bool IDAInstance::step()
  {
    if (!initialized) {
      if (!initialize()) {
        return false;
      }
    }

    if (getNumOfScalarEquations() == 0) {
      // IDA has nothing to solve. Just increment the time.

      if (getOptions().equidistantTimeGrid) {
        currentTime += timeStep;
      } else {
        currentTime = endTime;
      }

      return true;
    }

    // Execute one step.
    IDA_PROFILER_STEPS_COUNTER_INCREMENT;
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
    IDA_PROFILER_RESIDUALS_CALL_COUNTER_INCREMENT;

    realtype* rval = N_VGetArrayPointer(residuals);
    auto* instance = static_cast<IDAInstance*>(userData);

    // Copy the values of the variables and derivatives provided by IDA into
    // the variables owned by MARCO, so that the residual functions operate on
    // the current iteration values.
    instance->copyVariablesIntoMARCO(variables, derivatives);

    // For every vectorized equation, set the residual values of the variables
    // it writes into.
    IDA_PROFILER_RESIDUALS_START;

    instance->equationsParallelIteration(
        [&](Equation eq, const std::vector<int64_t>& equationIndices) {
          uint64_t equationRank = instance->getEquationRank(eq);
          assert(equationIndices.size() == equationRank);

          Variable writtenVariable = instance->getWrittenVariable(eq);

          uint64_t writtenVariableArrayOffset =
                  instance->variableOffsets[writtenVariable];

          uint64_t writtenVariableRank =
              instance->getVariableRank(writtenVariable);

          std::vector<uint64_t> writtenVariableIndices(writtenVariableRank, 0);

          AccessFunction writeAccessFunction =
              instance->getWriteAccessFunction(eq);

          writeAccessFunction(
              equationIndices.data(),
              writtenVariableIndices.data());

          uint64_t writtenVariableScalarOffset = getVariableFlatIndex(
              instance->variablesDimensions[writtenVariable],
              writtenVariableIndices);

          uint64_t offset =
              writtenVariableArrayOffset + writtenVariableScalarOffset;

          auto residualFn = instance->residualFunctions[eq];
          auto* eqIndicesPtr = equationIndices.data();

          auto residualFunctionResult = residualFn(time, eqIndicesPtr);
          *(rval + offset) = residualFunctionResult;
        });

    IDA_PROFILER_RESIDUALS_STOP;

    if (getOptions().debug) {
      std::cerr << "Residuals function called" << std::endl;
      std::cerr << "Variables:" << std::endl;
      instance->printVariablesVector(variables);
      std::cerr << "Derivatives:" << std::endl;
      instance->printDerivativesVector(derivatives);
      std::cerr << "Residuals vector:" << std::endl;
      instance->printResidualsVector(residuals);
    }

    return IDA_SUCCESS;
  }

  int IDAInstance::jacobianMatrix(
      realtype time, realtype alpha,
      N_Vector variables, N_Vector derivatives, N_Vector residuals,
      SUNMatrix jacobianMatrix,
      void* userData,
      N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
  {
    IDA_PROFILER_PARTIAL_DERIVATIVES_CALL_COUNTER_INCREMENT;

    realtype* jacobian = SUNSparseMatrix_Data(jacobianMatrix);
    auto* instance = static_cast<IDAInstance*>(userData);

    // Copy the values of the variables and derivatives provided by IDA into
    // the variables owned by MARCO, so that the jacobian functions operate on
    // the current iteration values.
    instance->copyVariablesIntoMARCO(variables, derivatives);

    // For every vectorized equation, compute its row within the Jacobian
    // matrix.
    IDA_PROFILER_PARTIAL_DERIVATIVES_START;

    instance->equationsParallelIteration(
        [&](Equation eq, const std::vector<int64_t>& equationIndices) {
            uint64_t equationRank = instance->getEquationRank(eq);
            Variable writtenVariable = instance->getWrittenVariable(eq);

            uint64_t writtenVariableArrayOffset =
                instance->variableOffsets[writtenVariable];

            uint64_t writtenVariableRank =
                instance->getVariableRank(writtenVariable);

            std::vector<uint64_t> writtenVariableIndices;
            writtenVariableIndices.resize(writtenVariableRank, 0);

            AccessFunction writeAccessFunction =
                instance->getWriteAccessFunction(eq);

            writeAccessFunction(
                equationIndices.data(),
                writtenVariableIndices.data());

            uint64_t writtenVariableScalarOffset = getVariableFlatIndex(
                instance->variablesDimensions[writtenVariable],
                writtenVariableIndices);

            uint64_t scalarEquationIndex =
                writtenVariableArrayOffset + writtenVariableScalarOffset;

            assert(scalarEquationIndex < instance->getNumOfScalarEquations());

            // Compute the column indexes that may be non-zeros.
            std::vector<JacobianColumn> jacobianColumns =
                instance->computeJacobianColumns(eq, equationIndices.data());

            // For every scalar variable with respect to which the equation must be
            // partially differentiated.
            for (size_t i = 0, e = jacobianColumns.size(); i < e; ++i) {
              const JacobianColumn& column = jacobianColumns[i];
              Variable variable = column.first;
              const auto& variableIndices = column.second;

              uint64_t variableArrayOffset = instance->variableOffsets[variable];

              uint64_t variableScalarOffset = getVariableFlatIndex(
                  instance->variablesDimensions[variable],
                  column.second);

              assert(instance->jacobianFunctions[eq][variable] != nullptr);

              auto jacobianFunctionResult =
                  instance->jacobianFunctions[eq][variable](
                      time,
                      equationIndices.data(),
                      variableIndices.data(),
                      alpha);

              instance->jacobianMatrixData[scalarEquationIndex][i].second =
                  jacobianFunctionResult;

              auto index = static_cast<sunindextype>(
                  variableArrayOffset + variableScalarOffset);

              instance->jacobianMatrixData[scalarEquationIndex][i].first =
                  index;
            }
        });

    sunindextype* rowPtrs = SUNSparseMatrix_IndexPointers(jacobianMatrix);
    sunindextype* columnIndices = SUNSparseMatrix_IndexValues(jacobianMatrix);

    sunindextype offset = 0;
    *rowPtrs++ = offset;

    for (const auto& row : instance->jacobianMatrixData) {
      offset += static_cast<sunindextype>(row.size());
      *rowPtrs++ = offset;

      for (const auto& column : row) {
        *columnIndices++ = column.first;
        *jacobian++ = column.second;
      }
    }

    assert(rowPtrs == SUNSparseMatrix_IndexPointers(jacobianMatrix) + instance->getNumOfScalarEquations() + 1);
    assert(columnIndices == SUNSparseMatrix_IndexValues(jacobianMatrix) + instance->nonZeroValuesNumber);
    assert(jacobian == SUNSparseMatrix_Data(jacobianMatrix) + instance->nonZeroValuesNumber);

    IDA_PROFILER_PARTIAL_DERIVATIVES_STOP;

    if (getOptions().debug) {
      std::cerr << "Jacobian matrix function called" << std::endl;
      std::cerr << "Time: " << time << std::endl;
      std::cerr << "Alpha: " << alpha << std::endl;
      std::cerr << "Variables:" << std::endl;
      instance->printVariablesVector(variables);
      std::cerr << "Derivatives:" << std::endl;
      instance->printDerivativesVector(derivatives);
      std::cerr << "Residuals vector:" << std::endl;
      instance->printResidualsVector(residuals);
      std::cerr << "Jacobian matrix:" << std::endl;
      instance->printJacobianMatrix(jacobianMatrix);
    }

    return IDA_SUCCESS;
  }

  uint64_t IDAInstance::getNumOfArrayVariables() const
  {
    return variablesDimensions.size();
  }

  uint64_t IDAInstance::getNumOfScalarVariables() const
  {
      return scalarVariablesNumber;
  }

  VariableKind IDAInstance::getVariableKind(Variable variable) const
  {
    auto it = stateVariablesMapping.find(variable);
    auto endIt = stateVariablesMapping.end();
    return it == endIt ? VariableKind::ALGEBRAIC : VariableKind::STATE;
  }

  uint64_t IDAInstance::getVariableFlatSize(Variable variable) const
  {
    uint64_t result = 1;

    for (uint64_t dimension : variablesDimensions[variable]) {
      result *= dimension;
    }

    return result;
  }

  uint64_t IDAInstance::getNumOfVectorizedEquations() const
  {
    return equationRanges.size();
  }

  uint64_t IDAInstance::getNumOfScalarEquations() const
  {
    return scalarEquationsNumber;
  }

  uint64_t IDAInstance::getEquationRank(Equation equation) const
  {
    return equationRanges[equation].size();
  }

  uint64_t IDAInstance::getEquationFlatSize(Equation equation) const
  {
    assert(equation < getNumOfVectorizedEquations());
    uint64_t result = 1;

    for (const Range& range : equationRanges[equation]) {
      result *= range.end - range.begin;
    }

    return result;
  }

  Variable IDAInstance::getWrittenVariable(Equation equation) const
  {
    return writeAccesses[equation].first;
  }

  AccessFunction IDAInstance::getWriteAccessFunction(Equation equation) const
  {
    return writeAccesses[equation].second;
  }

  uint64_t IDAInstance::getVariableRank(Variable variable) const
  {
    return variablesDimensions[variable].rank();
  }

  /// Determine which of the columns of the current Jacobian row has to be
  /// populated, and with respect to which variable the partial derivative has
  /// to be performed. The row is determined by the indices of the equation.
  std::vector<JacobianColumn> IDAInstance::computeJacobianColumns(
      Equation eq, const int64_t* equationIndices) const
  {
    std::set<JacobianColumn> uniqueColumns;

    if (precomputedAccesses) {
      for (const auto& access : variableAccesses[eq]) {
        Variable variable = access.first;
        AccessFunction accessFunction = access.second;

        uint64_t equationRank = getEquationRank(eq);
        uint64_t variableRank = getVariableRank(variable);

        std::vector<uint64_t> variableIndices;
        variableIndices.resize(variableRank, 0);
        accessFunction(equationIndices, variableIndices.data());

        assert([&]() -> bool {
          for (uint64_t i = 0; i < variableRank; ++i) {
            if (variableIndices[i] >= variablesDimensions[variable][i]) {
              return false;
            }
          }

          return true;
        }() && "Access out of bounds");

        uniqueColumns.insert({variable, variableIndices});
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

                for (size_t i = 0, e = first.second.size(); i < e; ++i) {
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
    nonZeroValuesNumber = 0;
    std::vector<int64_t> equationIndices;

    for (size_t eq = 0; eq < getNumOfVectorizedEquations(); ++eq) {
      // Initialize the multidimensional interval of the vector equation.
      uint64_t equationRank = equationRanges[eq].size();
      equationIndices.resize(equationRank);

      for (size_t i = 0; i < equationRank; ++i) {
        const auto& iterationRange = equationRanges[eq][i];
        int64_t beginIndex = iterationRange.begin;
        equationIndices[i] = beginIndex;
      }

      // For every scalar equation in the vector equation.
      do {
        // Compute the column indexes that may be non-zeros
        nonZeroValuesNumber +=
            computeJacobianColumns(eq, equationIndices.data()).size();

      } while (advanceEquationIndices(equationIndices, equationRanges[eq]));
    }
  }

  void IDAInstance::computeThreadChunks()
  {
    unsigned int numOfThreads = threadPool.getNumOfThreads();

    int64_t chunksFactor = getOptions().equationsChunksFactor;
    int64_t numOfChunks = numOfThreads * chunksFactor;

    uint64_t numOfVectorizedEquations = getNumOfVectorizedEquations();
    uint64_t numOfScalarEquations = getNumOfScalarEquations();

    size_t chunkSize =
        (numOfScalarEquations + numOfChunks - 1) / numOfChunks;

    // The number of vectorized equations whose indices have been completely
    // assigned.
    uint64_t processedEquations = 0;

    while (processedEquations < numOfVectorizedEquations) {
      Equation equation = equationsProcessingOrder[processedEquations];
      uint64_t equationFlatSize = getEquationFlatSize(equation);
      uint64_t equationFlatIndex = 0;

      // Divide the ranges into chunks.
      while (equationFlatIndex < equationFlatSize) {
        uint64_t beginFlatIndex = equationFlatIndex;

        uint64_t endFlatIndex = std::min(
            beginFlatIndex + static_cast<uint64_t>(chunkSize),
            equationFlatSize);

        std::vector<int64_t> beginIndices;
        std::vector<int64_t> endIndices;

        getEquationIndicesFromFlatIndex(
            beginFlatIndex, beginIndices, equationRanges[equation]);

        if (endFlatIndex == equationFlatSize) {
          getEquationEndIndices(equation, endIndices);
        } else {
          getEquationIndicesFromFlatIndex(
              endFlatIndex, endIndices, equationRanges[equation]);
        }

        threadEquationsChunks.emplace_back(
            equation, std::move(beginIndices), std::move(endIndices));

        // Move to the next chunk.
        equationFlatIndex = endFlatIndex;
      }

      // Move to the next vectorized equation.
      ++processedEquations;
    }
  }

  void IDAInstance::copyVariablesFromMARCO(
      N_Vector algebraicAndStateVariablesVector,
      N_Vector derivativeVariablesVector)
  {
    if (getOptions().debug) {
      std::cerr << "Copying variables from MARCO" << std::endl;
    }

    IDA_PROFILER_COPY_VARS_FROM_MARCO_START;

    realtype* varsPtr = N_VGetArrayPointer(algebraicAndStateVariablesVector);
    realtype* dersPtr = N_VGetArrayPointer(derivativeVariablesVector);
    uint64_t numOfArrayVariables = getNumOfArrayVariables();

    for (Variable var = 0; var < numOfArrayVariables; ++var) {
      uint64_t variableArrayOffset = variableOffsets[var];
      const auto& dimensions = variablesDimensions[var];

      std::vector<uint64_t> varIndices;
      getVariableBeginIndices(var, varIndices);

      do {
        uint64_t variableScalarOffset =
            getVariableFlatIndex(dimensions, varIndices.data());

        uint64_t offset = variableArrayOffset + variableScalarOffset;

        // Get the state / algebraic variable.
        auto getterFn = algebraicAndStateVariablesGetters[var];
        auto value = static_cast<realtype>(getterFn(varIndices.data()));
        varsPtr[offset] = value;

        if (getOptions().debug) {
          std::cerr << "Got var " << var << " ";
          printIndices(varIndices);
          std::cerr << " with value " << std::fixed << std::setprecision(9)
                    << value << std::endl;
        }

        // Get the derivative variable, if the variable was a state.
        auto derivativeVariablePositionIt = stateVariablesMapping.find(var);

        if (derivativeVariablePositionIt != stateVariablesMapping.end()) {
          auto derGetterFn =
              derivativeVariablesGetters[derivativeVariablePositionIt->second];

          auto derValue =
              static_cast<realtype>(derGetterFn(varIndices.data()));

          dersPtr[offset] = derValue;

          if (getOptions().debug) {
            std::cerr << "Got der(var " << var << ") ";
            printIndices(varIndices);
            std::cerr << " with value " << std::fixed << std::setprecision(9)
                      << derValue << std::endl;
          }
        }
      } while (advanceVariableIndices(varIndices, variablesDimensions[var]));
    }

    IDA_PROFILER_COPY_VARS_FROM_MARCO_STOP;
  }

  void IDAInstance::copyVariablesIntoMARCO(
      N_Vector algebraicAndStateVariablesVector,
      N_Vector derivativeVariablesVector)
  {
    if (getOptions().debug) {
      std::cerr << "Copying variables into MARCO" << std::endl;
    }

    IDA_PROFILER_COPY_VARS_INTO_MARCO_START;

    realtype* varsPtr = N_VGetArrayPointer(algebraicAndStateVariablesVector);
    realtype* dersPtr = N_VGetArrayPointer(derivativeVariablesVector);
    uint64_t numOfArrayVariables = getNumOfArrayVariables();

    for (Variable var = 0; var < numOfArrayVariables; ++var) {
      uint64_t variableArrayOffset = variableOffsets[var];
      const auto& dimensions = variablesDimensions[var];

      std::vector<uint64_t> varIndices;
      getVariableBeginIndices(var, varIndices);

      do {
        uint64_t variableScalarOffset =
            getVariableFlatIndex(dimensions, varIndices.data());

        uint64_t offset = variableArrayOffset + variableScalarOffset;

        // Set the state / algebraic variable.
        auto setterFn = algebraicAndStateVariablesSetters[var];
        auto value = static_cast<double>(varsPtr[offset]);

        if (getOptions().debug) {
          std::cerr << "Setting var " << var << " ";
          printIndices(varIndices);
          std::cerr << " to " << value << std::endl;
        }

        setterFn(value, varIndices.data());

        assert([&]() -> bool {
          auto getterFn = algebraicAndStateVariablesGetters[var];
          return getterFn(varIndices.data()) == value;
        }() && "Variable value not set correctly");

        // Set the derivative variable, if the variable was a state.
        auto derivativeVariablePositionIt = stateVariablesMapping.find(var);

        if (derivativeVariablePositionIt != stateVariablesMapping.end()) {
          auto derSetterFn =
              derivativeVariablesSetters[derivativeVariablePositionIt->second];

          auto derValue = static_cast<double>(dersPtr[offset]);

          if (getOptions().debug) {
            std::cerr << "Setting der(var " << var << ") ";
            printIndices(varIndices);
            std::cerr << " to " << derValue << std::endl;
          }

          derSetterFn(derValue, varIndices.data());

          assert([&]() -> bool {
            auto derGetterFn = derivativeVariablesGetters[
                derivativeVariablePositionIt->second];

            return derGetterFn(varIndices.data()) == derValue;
          }() && "Derivative value not set correctly");
        }
      } while (advanceVariableIndices(varIndices, variablesDimensions[var]));
    }

    IDA_PROFILER_COPY_VARS_INTO_MARCO_STOP;
  }

  void IDAInstance::vectorEquationsParallelIteration(
      std::function<void(Equation equation)> processFn)
  {
    std::mutex mutex;
    size_t processedEquations = 0;

    // Function to move to the next equation.
    auto getEquationAndAdvance =
        [&](Equation& eq) {
          std::lock_guard<std::mutex> lockGuard(mutex);

          if (processedEquations >= getNumOfVectorizedEquations()) {
            return false;
          }

          eq = equationsProcessingOrder[processedEquations++];
          return true;
        };

    for (unsigned int i = 0, e = threadPool.getNumOfThreads(); i < e; ++i) {
      threadPool.async([&]() {
        Equation equation;

        while (getEquationAndAdvance(equation)) {
          processFn(equation);
        }
      });
    }

    threadPool.wait();
  }

  void IDAInstance::scalarEquationsParallelIteration(
      std::function<void(
          Equation equation,
          const std::vector<int64_t>& equationIndices)> processFn)
  {
    size_t processedEquations = 0;
    std::vector<int64_t> equationIndices;
    std::mutex mutex;

    // Function to advance the indices by one, or move to the next equation if
    // the current one has been fully visited.
    auto getEquationAndAdvance =
        [&](Equation& eq, std::vector<int64_t>& indices) {
          std::lock_guard<std::mutex> lockGuard(mutex);

          if (processedEquations >= getNumOfVectorizedEquations()) {
            return false;
          }

          eq = equationsProcessingOrder[processedEquations];
          indices = equationIndices;

          if (!advanceEquationIndices(equationIndices, equationRanges[eq])) {
            if (++processedEquations < getNumOfVectorizedEquations()) {
              getEquationBeginIndices(
                  equationsProcessingOrder[processedEquations],
                  equationIndices);
            }
          }

          return true;
        };

    getEquationBeginIndices(
        equationsProcessingOrder[processedEquations], equationIndices);

    for (unsigned int i = 0, e = threadPool.getNumOfThreads(); i < e; ++i) {
      threadPool.async([&]() {
        Equation equation;
        std::vector<int64_t> indices;

        while (getEquationAndAdvance(equation, indices)) {
          processFn(equation, indices);
        }
      });
    }

    threadPool.wait();
  }

  void IDAInstance::equationsParallelIteration(
      std::function<void(
          Equation equation,
          const std::vector<int64_t>& equationIndices)> processFn)
  {
    // Shard the work among multiple threads.
    unsigned int numOfThreads = threadPool.getNumOfThreads();
    std::atomic_size_t chunkIndex = 0;

    for (unsigned int thread = 0; thread < numOfThreads; ++thread) {
      threadPool.async([&]() {
        size_t assignedChunk;

        while ((assignedChunk = chunkIndex++) < threadEquationsChunks.size()) {
          const ThreadEquationsChunk& chunk =
              threadEquationsChunks[assignedChunk];

          Equation equation = std::get<0>(chunk);
          std::vector<int64_t> equationIndices = std::get<1>(chunk);

          do {
              processFn(equation, equationIndices);
          } while (advanceEquationIndicesUntil(
              equationIndices, equationRanges[equation], std::get<2>(chunk)));
        }
      });
    }

    threadPool.wait();
  }

  void IDAInstance::getVariableBeginIndices(
      Variable variable, std::vector<uint64_t>& indices) const
  {
    uint64_t variableRank = getVariableRank(variable);
    indices.resize(variableRank);

    for (uint64_t i = 0; i < variableRank; ++i) {
      indices[i] = 0;
    }
  }

  void IDAInstance::getVariableEndIndices(
      Variable variable, std::vector<uint64_t>& indices) const
  {
    uint64_t variableRank = getVariableRank(variable);
    indices.resize(variableRank);

    for (uint64_t i = 0; i < variableRank; ++i) {
      indices[i] = variablesDimensions[variable][i];
    }
  }

  void IDAInstance::getEquationBeginIndices(
      Equation equation, std::vector<int64_t>& indices) const
  {
    uint64_t equationRank = getEquationRank(equation);
    indices.resize(equationRank);

    for (uint64_t i = 0; i < equationRank; ++i) {
      indices[i] = equationRanges[equation][i].begin;
    }
  }

  void IDAInstance::getEquationEndIndices(
      Equation equation, std::vector<int64_t>& indices) const
  {
    uint64_t equationRank = getEquationRank(equation);
    indices.resize(equationRank);

    for (uint64_t i = 0; i < equationRank; ++i) {
      indices[i] = equationRanges[equation][i].end;
    }
  }

  void IDAInstance::printStatistics() const
  {
    if (getNumOfScalarEquations() == 0) {
      return;
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
    std::cerr << getNumOfScalarEquations() << std::endl;
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
    auto retVal = IDASetMaxNumStepsIC(
            idaMemory, static_cast<int>(getOptions().maxStepsIC));

    if (retVal == IDA_MEM_NULL) {
      std::cerr << "IDASetMaxNumStepsIC - The ida_mem pointer is NULL"
                << std::endl;
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
      std::cerr << "IDASetMaxNumItersIC - The ida_mem pointer is NULL"
                << std::endl;
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
      std::cerr << "IDASetLineSearchOffIC - The ida_mem pointer is NULL"
                << std::endl;
      return false;
    }

    return retVal == IDA_SUCCESS;
  }

  void IDAInstance::getWritingEquation(
      Variable variable,
      const std::vector<uint64_t>& variableIndices,
      Equation& equation,
      std::vector<int64_t>& equationIndices) const
  {
    bool found = false;
    uint64_t numOfVectorizedEquations = getNumOfVectorizedEquations();

    for (Equation eq = 0; eq < numOfVectorizedEquations; ++eq) {
      Variable writtenVariable = getWrittenVariable(eq);

      if (writtenVariable == variable) {
        std::vector<int64_t> writingEquationIndices;
        getEquationBeginIndices(eq, writingEquationIndices);

        std::vector<uint64_t> writtenVariableIndices(
            getVariableRank(writtenVariable));

        AccessFunction writeAccessFunction = getWriteAccessFunction(eq);

        do {
          writeAccessFunction(writingEquationIndices.data(),
                              writtenVariableIndices.data());

          if (writtenVariableIndices == variableIndices) {
            assert(!found &&
                   "Multiple equations writing to the same variable");
            found = true;
            equation = eq;
            equationIndices = writingEquationIndices;
          }
        } while (advanceEquationIndices(
            writingEquationIndices, equationRanges[eq]));
      }
    }

    assert(found && "Writing equation not found");
  }

  void IDAInstance::printVariablesVector(N_Vector variables) const
  {
    realtype* data = N_VGetArrayPointer(variables);
    uint64_t numOfArrayVariables = getNumOfArrayVariables();

    for (Variable var = 0; var < numOfArrayVariables; ++var) {
      std::vector<uint64_t> indices;
      getVariableBeginIndices(var, indices);

      do {
        std::cerr << "var " << var << " ";
        printIndices(indices);
        std::cerr << "\t" << std::fixed << std::setprecision(9)
                  << *data << std::endl;
        ++data;
      } while (advanceVariableIndices(indices, variablesDimensions[var]));
    }
  }

  void IDAInstance::printDerivativesVector(N_Vector derivatives) const
  {
    realtype* data = N_VGetArrayPointer(derivatives);
    uint64_t numOfArrayVariables = getNumOfArrayVariables();

    for (Variable var = 0; var < numOfArrayVariables; ++var) {
      auto it = stateVariablesMapping.find(var);

      if (it != stateVariablesMapping.end()) {
        std::vector<uint64_t> indices;
        getVariableBeginIndices(var, indices);

        do {
          std::cerr << "der(var " << var << ") ";
          printIndices(indices);
          std::cerr << "\t" << std::fixed << std::setprecision(9)
                    << *data << std::endl;
          ++data;
        } while (advanceVariableIndices(indices, variablesDimensions[var]));
      }
    }
  }

  void IDAInstance::printResidualsVector(N_Vector residuals) const
  {
    realtype* data = N_VGetArrayPointer(residuals);
    uint64_t numOfArrayVariables = getNumOfArrayVariables();

    for (Variable var = 0; var < numOfArrayVariables; ++var) {
      std::vector<uint64_t> variableIndices;
      getVariableBeginIndices(var, variableIndices);

      do {
        Equation eq;
        std::vector<int64_t> equationIndices;
        getWritingEquation(var, variableIndices, eq, equationIndices);

        std::cerr << "eq " << eq << " ";
        printIndices(equationIndices);
        std::cerr << " (writing to var " << var;
        printIndices(variableIndices);
        std::cerr << ")" << "\t" << std::fixed << std::setprecision(9)
                  << *data << "\n";
        ++data;
      } while (advanceVariableIndices(
          variableIndices, variablesDimensions[var]));
    }
  }

  // Highly inefficient, use only for debug purposes.
  static sunrealtype getCellFromSparseMatrix(
      SUNMatrix matrix,
      uint64_t rowIndex,
      uint64_t columnIndex)
  {
    realtype* data = SUNSparseMatrix_Data(matrix);
    sunindextype rows = SUNSparseMatrix_Rows(matrix);
    sunindextype nnz = SUNSparseMatrix_NNZ(matrix);

    sunindextype* rowPtrs = SUNSparseMatrix_IndexPointers(matrix);
    sunindextype* columnIndices = SUNSparseMatrix_IndexValues(matrix);

    sunindextype beginIndex = rowPtrs[rowIndex];
    sunindextype endIndex = rowPtrs[rowIndex + 1];

    for (sunindextype i = beginIndex; i < endIndex; ++i) {
      if (columnIndices[i] == static_cast<sunindextype>(columnIndex)) {
        return data[i];
      }
    }

    return 0;
  }

  void IDAInstance::printJacobianMatrix(SUNMatrix jacobianMatrix) const
  {
    uint64_t numOfArrayVariables = getNumOfArrayVariables();

    // Print the heading row.
    for (Variable var = 0; var < numOfArrayVariables; ++var) {
      std::vector<uint64_t> variableIndices;
      getVariableBeginIndices(var, variableIndices);

      do {
        std::cerr << "\tvar " << var << " ";
        printIndices(variableIndices);
      } while (advanceVariableIndices(
          variableIndices, variablesDimensions[var]));
    }

    std::cerr << std::endl;

    // Print the rows containing the values.
    uint64_t rowFlatIndex = 0;

    for (Variable eqVar = 0; eqVar < numOfArrayVariables; ++eqVar) {
      std::vector<uint64_t> eqVarIndices;
      getVariableBeginIndices(eqVar, eqVarIndices);

      do {
        Equation eq;
        std::vector<int64_t> equationIndices;
        getWritingEquation(eqVar, eqVarIndices, eq, equationIndices);

        std::cerr << "eq " << eq << " ";
        printIndices(equationIndices);
        std::cerr << " (writing to var " << eqVar << " ";
        printIndices(eqVarIndices);
        std::cerr << ")";

        uint64_t columnFlatIndex = 0;

        for (Variable indVar = 0; indVar < numOfArrayVariables; ++indVar) {
          std::vector<uint64_t> indVarIndices;
          getVariableBeginIndices(indVar, indVarIndices);

          do {
            auto value = getCellFromSparseMatrix(
                jacobianMatrix, rowFlatIndex, columnFlatIndex);

            std::cerr << "\t" << std::fixed << std::setprecision(9) << value;
            columnFlatIndex++;
          } while (advanceVariableIndices(
              indVarIndices, variablesDimensions[indVar]));
        }

        std::cerr << std::endl;
        rowFlatIndex++;
      } while (advanceVariableIndices(
          eqVarIndices, variablesDimensions[eqVar]));
    }
  }
}

//===---------------------------------------------------------------------===//
// Exported functions
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// idaCreate

static void* idaCreate_pvoid()
{
  auto* instance = new IDAInstance();
  return static_cast<void*>(instance);
}

RUNTIME_FUNC_DEF(idaCreate, PTR(void))

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
// idaAddAlgebraicVariable

static uint64_t idaAddAlgebraicVariable_i64(
    void* instance,
    uint64_t rank,
    uint64_t* dimensions,
    void* getter,
    void* setter,
    void* name)
{
  return static_cast<IDAInstance*>(instance)->addAlgebraicVariable(
      rank, dimensions,
      reinterpret_cast<VariableGetter>(getter),
      reinterpret_cast<VariableSetter>(setter),
      static_cast<const char*>(name));
}

RUNTIME_FUNC_DEF(idaAddAlgebraicVariable, uint64_t, PTR(void), uint64_t, PTR(uint64_t), PTR(void), PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaAddStateVariable

static uint64_t idaAddStateVariable_i64(
    void* instance,
    uint64_t rank,
    uint64_t* dimensions,
    void* stateGetter,
    void* stateSetter,
    void* derivativeGetter,
    void* derivativeSetter,
    void* name)
{
  return static_cast<IDAInstance*>(instance)->addStateVariable(
      rank, dimensions,
      reinterpret_cast<VariableGetter>(stateGetter),
      reinterpret_cast<VariableSetter>(stateSetter),
      reinterpret_cast<VariableGetter>(derivativeGetter),
      reinterpret_cast<VariableSetter>(derivativeSetter),
      static_cast<const char*>(name));
}

RUNTIME_FUNC_DEF(idaAddStateVariable, uint64_t, PTR(void), uint64_t, PTR(uint64_t), PTR(void), PTR(void), PTR(void), PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaAddVariableAccess

static void idaAddVariableAccess_void(
    void* instance,
    uint64_t equationIndex,
    uint64_t variableIndex,
    void* accessFunction)
{
  static_cast<IDAInstance*>(instance)->addVariableAccess(
      equationIndex, variableIndex,
      reinterpret_cast<AccessFunction>(accessFunction));
}

RUNTIME_FUNC_DEF(idaAddVariableAccess, void, PTR(void), uint64_t, uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaAddEquation

static uint64_t idaAddEquation_i64(
    void* instance,
    int64_t* ranges,
    uint64_t rank,
    uint64_t writtenVariable,
    void* writeAccessFunction,
    void* stringRepresentation)
{
  return static_cast<IDAInstance*>(instance)->addEquation(
      ranges, rank, writtenVariable,
      reinterpret_cast<AccessFunction>(writeAccessFunction),
      static_cast<const char*>(stringRepresentation));
}

RUNTIME_FUNC_DEF(idaAddEquation, uint64_t, PTR(void), PTR(int64_t), uint64_t, uint64_t, PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaSetResidual

static void idaSetResidual_void(
    void* instance,
    uint64_t equationIndex,
    void* residualFunction)
{
  static_cast<IDAInstance*>(instance)->setResidualFunction(
      equationIndex,
      reinterpret_cast<ResidualFunction>(residualFunction));
}

RUNTIME_FUNC_DEF(idaSetResidual, void, PTR(void), uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaAddJacobian

static void idaAddJacobian_void(
    void* instance,
    uint64_t equationIndex,
    uint64_t variableIndex,
    void* jacobianFunction)
{
  static_cast<IDAInstance*>(instance)->addJacobianFunction(
      equationIndex, variableIndex,
      reinterpret_cast<JacobianFunction>(jacobianFunction));
}

RUNTIME_FUNC_DEF(idaAddJacobian, void, PTR(void), uint64_t, uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaPrintStatistics

static void printStatistics_void(void* instance)
{
  static_cast<IDAInstance*>(instance)->printStatistics();
}

RUNTIME_FUNC_DEF(printStatistics, void, PTR(void))

#endif // SUNDIALS_ENABLE
