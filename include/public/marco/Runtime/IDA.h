#ifndef MARCO_RUNTIME_IDA_H
#define MARCO_RUNTIME_IDA_H

#include "marco/Runtime/Mangling.h"
#include "ida/ida.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <set>
#include <vector>

namespace marco::runtime::ida
{
  struct Options
  {
    // Whether to print the Jacobian matrices while debugging
    bool printJacobian = false;

    // Arbitrary initial guesses on 20/12/2021 Modelica Call
    realtype algebraicTolerance = 1e-12;
    realtype timeScalingFactorInit = 1e5;

    // Default IDA values
    realtype initTimeStep = 0.0;

    long maxNumSteps = 1e4;
    int maxErrTestFail = 10;
    int maxNonlinIters = 4;
    int maxConvFails = 10;
    realtype nonlinConvCoef = 0.33;

    int maxNumStepsIC = 5;
    int maxNumJacsIC = 4;
    int maxNumItersIC = 10;
    realtype nonlinConvCoefIC = 0.0033;

    int suppressAlg = SUNFALSE;
    int lineSearchOff = SUNFALSE;
  };

  Options& getOptions();

  using Access = std::vector<std::pair<sunindextype, sunindextype>>;
  using VarAccessList = std::vector<std::pair<sunindextype, Access>>;

  using DerivativeVariable = std::pair<size_t, std::vector<size_t>>;

  class VariableIndicesIterator;

  /// The list of dimensions of an array variable.
  class VariableDimensions
  {
    private:
      using Container = std::vector<size_t>;

    public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      VariableDimensions(size_t rank);

      size_t rank() const;

      size_t& operator[](size_t index);
      const size_t& operator[](size_t index) const;

      /// @name Dimensions iterators
      /// {

      const_iterator begin() const;
      const_iterator end() const;

      /// }
      /// @name Indices iterators
      /// {

      VariableIndicesIterator indicesBegin() const;
      VariableIndicesIterator indicesEnd() const;

      /// }

    private:
      /// Check that all the dimensions have been correctly initialized.
      [[maybe_unused]] bool isValid() const;

    private:
      Container dimensions;
  };

  /// This class is used to iterate on all the possible combination of indices of a variable.
  class VariableIndicesIterator
  {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = size_t*;
      using difference_type = std::ptrdiff_t;
      using pointer = size_t**;
      using reference = size_t*&;

      ~VariableIndicesIterator();

      static VariableIndicesIterator begin(const VariableDimensions& dimensions);

      static VariableIndicesIterator end(const VariableDimensions& dimensions);

      bool operator==(const VariableIndicesIterator& it) const;

      bool operator!=(const VariableIndicesIterator& it) const;

      VariableIndicesIterator& operator++();
      VariableIndicesIterator operator++(int);

      size_t* operator*() const;

    private:
      VariableIndicesIterator(const VariableDimensions& dimensions);

      void fetchNext();

    private:
      size_t* indices;
      const VariableDimensions* dimensions;
  };

  using EqDimension = std::vector<std::pair<size_t, size_t>>;

  /// Signature of variable getter functions.
  /// The 1st argument is an opaque pointer to the variable descriptor.
  /// The 2nd argument is a pointer to the indices list.
  /// The result is the scalar value.
  template<typename FloatType>
  using VariableGetterFunction = FloatType(*)(void*, size_t*);

  /// Signature of variable setter functions.
  /// The 1st argument is an opaque pointer to the variable descriptor.
  /// The 2nd argument is the value to be set.
  /// The 3rd argument is a pointer to the indices list.
  template<typename FloatType>
  using VariableSetterFunction = void(*)(void*, FloatType, size_t*);

  /// Signature of residual functions.
  /// The 1st argument is the current time.
  /// The 2nd argument is an opaque pointer to the simulation data.
  /// The 3rd argument is a pointer to the list of equation indices.
  /// The result is the residual value.
  template<typename FloatType>
  using ResidualFunction = FloatType(*)(FloatType, void*, size_t*);

  /// Signature of Jacobian functions.
  /// The 1st argument is the current time.
  /// The 2nd argument is an opaque pointer to the simulation data.
  /// The 3rd argument is a pointer to the list of equation indices.
  /// The 4th argument is a pointer to the list of variable indices.
  /// The 5th argument is the 'alpha' value.
  /// The result is the Jacobian value.
  template<typename FloatType>
  using JacobianFunction = FloatType(*)(FloatType, void*, size_t*, size_t*, FloatType);

  class IDAInstance
  {
    public:
      /// Constant used to indicate that no fixed time step has been set.
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
      SUNContext ctx;
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
//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaCreate, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(idaInit, void, PTR(void))

RUNTIME_FUNC_DECL(idaStep, void, PTR(void))

RUNTIME_FUNC_DECL(printStatistics, void, PTR(void))

RUNTIME_FUNC_DECL(idaFree, void, PTR(void))

RUNTIME_FUNC_DECL(idaSetStartTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetEndTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetTimeStep, void, PTR(void), double)

RUNTIME_FUNC_DECL(idaSetRelativeTolerance, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetAbsoluteTolerance, void, PTR(void), double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaAddEquation, int64_t, PTR(void), PTR(int64_t), int64_t)

RUNTIME_FUNC_DECL(idaAddResidual, void, PTR(void), int64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddJacobian, void, PTR(void), int64_t, int64_t, PTR(void))

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaAddAlgebraicVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))
RUNTIME_FUNC_DECL(idaAddStateVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaSetDerivative, void, PTR(void), int64_t, PTR(void), PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaAddVariableAccess, void, PTR(void), int64_t, int64_t, PTR(int64_t), int64_t)

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaGetCurrentTime, float, PTR(void))
RUNTIME_FUNC_DECL(idaGetCurrentTime, double, PTR(void))

#endif // MARCO_RUNTIME_IDA_H
