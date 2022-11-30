#ifndef MARCO_RUNTIME_SOLVERS_IDA_INSTANCE_H
#define MARCO_RUNTIME_SOLVERS_IDA_INSTANCE_H

#include "marco/Runtime/Mangling.h"
#include "marco/Runtime/Modeling/MultidimensionalRange.h"
#include "ida/ida.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <map>
#include <set>
#include <vector>

namespace marco::runtime::ida
{
  using Access = std::vector<std::pair<int64_t, int64_t>>;
  using VarAccessList = std::vector<std::pair<sunindextype, Access>>;

  /// A column of the Jacobian matrix.
  /// The first element represents the array variable with respect to which the
  /// partial derivative has to be computed. The second element represents the
  /// indices of the scalar variable.
  using JacobianColumn = std::pair<size_t, std::vector<int64_t>>;

  class VariableIndicesIterator;

  /// The list of dimensions of an array variable.
  class VariableDimensions
  {
    private:
      using Container = std::vector<int64_t>;

    public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      VariableDimensions(size_t rank);

      size_t rank() const;

      int64_t& operator[](size_t index);
      const int64_t& operator[](size_t index) const;

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

  /// This class is used to iterate on all the possible combination of indices
  /// of a variable.
  class VariableIndicesIterator
  {
    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = const int64_t*;
      using difference_type = std::ptrdiff_t;
      using pointer = const int64_t**;
      using reference = const int64_t*&;

      ~VariableIndicesIterator();

      static VariableIndicesIterator begin(
          const VariableDimensions& dimensions);

      static VariableIndicesIterator end(
          const VariableDimensions& dimensions);

      bool operator==(const VariableIndicesIterator& it) const;

      bool operator!=(const VariableIndicesIterator& it) const;

      VariableIndicesIterator& operator++();
      VariableIndicesIterator operator++(int);

      const int64_t* operator*() const;

    private:
      VariableIndicesIterator(const VariableDimensions& dimensions);

      void fetchNext();

    private:
      int64_t* indices;
      const VariableDimensions* dimensions;
  };

  /// Signature of variable getter functions.
  /// The 1st argument is an opaque pointer to the variable descriptor.
  /// The 2nd argument is a pointer to the indices list.
  /// The result is the scalar value.
  template<typename FloatType>
  using VariableGetterFunction = FloatType(*)(void*, const int64_t*);

  /// Signature of variable setter functions.
  /// The 1st argument is an opaque pointer to the variable descriptor.
  /// The 2nd argument is the value to be set.
  /// The 3rd argument is a pointer to the indices list.
  template<typename FloatType>
  using VariableSetterFunction = void(*)(void*, FloatType, const int64_t*);

  /// Signature of residual functions.
  /// The 1st argument is the current time.
  /// The 2nd argument is an opaque pointer to the simulation data.
  /// The 3rd argument is a pointer to the list of equation indices.
  /// The result is the residual value.
  template<typename FloatType>
  using ResidualFunction = FloatType(*)(FloatType, void*, const int64_t*);

  /// Signature of Jacobian functions.
  /// The 1st argument is the current time.
  /// The 2nd argument is an opaque pointer to the simulation data.
  /// The 3rd argument is a pointer to the list of equation indices.
  /// The 4th argument is a pointer to the list of variable indices.
  /// The 5th argument is the 'alpha' value.
  /// The result is the Jacobian value.
  template<typename FloatType>
  using JacobianFunction = FloatType(*)(FloatType, void*, const int64_t*, const int64_t*, FloatType);

  class IDAInstance
  {
    public:
      IDAInstance(int64_t marcoBitWidth, int64_t scalarEquationsNumber);

      ~IDAInstance();

      void setStartTime(double time);
      void setEndTime(double time);
      void setTimeStep(double time);

      /// Add a parametric variable.
      void addParametricVariable(void* variable);

      /// Add and initialize a new variable given its array.
      int64_t addAlgebraicVariable(
          void* variable,
          int64_t* dimensions,
          int64_t rank,
          void* getterFunction,
          void* setterFunction);

      int64_t addStateVariable(
          void* variable,
          int64_t* dimensions,
          int64_t rank,
          void* getterFunction,
          void* setterFunction);

      void setDerivative(
          int64_t idaStateVariable,
          void* derivative,
          void* getterFunction,
          void* setterFunction);

      /// Add the information about an equation the is handled by IDA.
      int64_t addEquation(
          int64_t* ranges,
          int64_t rank,
          int64_t writtenVariable,
          int64_t* writtenVariableIndices);

      void addVariableAccess(
          int64_t equationIndex,
          int64_t variableIndex,
          int64_t* access,
          int64_t rank);

      /// Add the function pointer that computes the index-th residual function
      /// to the IDA user data.
      void setResidualFunction(
          int64_t equationIndex, void* residualFunction);

      /// Add the function pointer that computes the index-th Jacobian row to
      /// the user data.
      void addJacobianFunction(
          int64_t equationIndex,
          int64_t variableIndex,
          void* jacobianFunction);

      /// Instantiate and initialize all the classes needed by IDA in order to
      /// solve the given system of equations. It also sets optional simulation
      /// parameters for IDA. It must be called before the first usage of
      /// idaStep() and after a call to idaAllocData(). It may fail in case of
      /// malformed model.
      bool initialize();

      /// Invoke IDA to perform the computation of the initial values of the
      /// variables. Returns true if the computation was successful, false
      /// otherwise.
      bool calcIC();

      /// Invoke IDA to perform one step of the computation. If a time step is
      /// given, the output will show the variables in an equidistant time grid
      /// based on the step time parameter. Otherwise, the output will show the
      /// variables at every step of the computation. Returns true if the
      /// computation was successful, false otherwise.
      bool step();

      /// Returns the time reached by the solver after the last step.
      realtype getCurrentTime() const;

      /// Prints statistics regarding the computation of the system.
      void printStatistics() const;

      /// IDAResFn user-defined residual function, passed to IDA through
      /// IDAInit. It contains how to compute the Residual Function of the
      /// system, starting from the provided UserData struct, iterating through
      /// every equation.
      static int residualFunction(
          realtype time,
          N_Vector variables, N_Vector derivatives, N_Vector residuals,
          void* userData);

      /// IDALsJacFn user-defined Jacobian approximation function, passed to
      /// IDA through IDASetJacFn. It contains how to compute the Jacobian
      /// Matrix of the system, starting from the provided UserData struct,
      /// iterating through every equation and variable. The matrix is
      /// represented in CSR format.
      static int jacobianMatrix(
          realtype time,
          realtype alpha,
          N_Vector variables,
          N_Vector derivatives,
          N_Vector residuals,
          SUNMatrix jacobianMatrix,
          void* userData,
          N_Vector tempv1,
          N_Vector tempv2,
          N_Vector tempv3);

    private:
      size_t getNumOfVectorizedEquations() const;

      size_t getEquationRank(size_t equation) const;

      std::vector<JacobianColumn> computeIndexSet(
          size_t eq, int64_t* eqIndexes) const;

      void computeNNZ();

      void copyVariablesFromMARCO(
          N_Vector algebraicAndStateVariablesVector,
          N_Vector derivativeVariablesVector);

      void copyVariablesIntoMARCO(
          N_Vector algebraicAndStateVariablesVector,
          N_Vector derivativeVariablesVector);

      /// Prints the Jacobian incidence matrix of the system.
      void printIncidenceMatrix() const;

    private:
      /// @name Forwarded methods
      /// {

      bool idaInit();
      bool idaSVTolerances();
      bool idaSetLinearSolver();
      bool idaSetUserData();
      bool idaSetMaxNumSteps();
      bool idaSetInitialStepSize();
      bool idaSetMinStepSize();
      bool idaSetMaxStepSize();
      bool idaSetStopTime();
      bool idaSetMaxErrTestFails();
      bool idaSetSuppressAlg();
      bool idaSetId();
      bool idaSetJacobianFunction();
      bool idaSetMaxNonlinIters();
      bool idaSetMaxConvFails();
      bool idaSetNonlinConvCoef();
      bool idaSetNonlinConvCoefIC();
      bool idaSetMaxNumStepsIC();
      bool idaSetMaxNumJacsIC();
      bool idaSetMaxNumItersIC();
      bool idaSetLineSearchOffIC();

      /// }

    private:
      // Sundials context.
      SUNContext ctx;

      // Whether the instance has been inizialized or not.
      bool initialized;

      int64_t marcoBitWidth;

      // Model size.
      int64_t scalarEquationsNumber;
      int64_t nonZeroValuesNumber;

      // The iteration ranges of the vectorized equations.
      std::vector<MultidimensionalRange> equationRanges;

      // The array variables written by the equations.
      // The i-th position contains the information about the variable written
      // by the i-th equation: the first element is the index of the IDA
      // variable, while the second represents the ranges of the scalar
      // variable.
      std::vector<std::pair<int64_t, Access>> writeAccesses;

      // The order in which the equations must be processed when computing
      // residuals and jacobians in order to match the order of the variables.
      // For example, if equation 0 writes to variable 1 and equation 1 writes
      // to variable 0, then the processing order is equation 1 -> equation 0.
      std::vector<size_t> equationsProcessingOrder;

      // The residual functions associated with the equations.
      // The i-th position contains the pointer to the residual function of the
      // i-th equation.
      std::vector<void*> residualFunctions;

      // The jacobian functions associated with the equations.
      // The i-th position contains the list of partial derivative functions of
      // the i-th equation. The j-th function represents the function to
      // compute the derivative with respect to the j-th variable.
      std::vector<std::vector<void*>> jacobianFunctions;

      std::vector<VarAccessList> variableAccesses;

      // The offset of each array variable inside the flattened variables
      // vector.
      std::vector<sunindextype> variableOffsets;

      // The dimensions list of each array variable.
      std::vector<VariableDimensions> variablesDimensions;

      // Simulation times.
      realtype startTime;
      realtype endTime;
      realtype timeStep;
      realtype currentTime;

      // Variables vectors and values.
      N_Vector variablesVector;
      N_Vector derivativesVector;

      // The vector stores whether each scalar variable is an algebraic or a
      // state one.
      // 0 = algebraic
      // 1 = state
      N_Vector idVector;

      // The tolerance for each scalar variable.
      N_Vector tolerancesVector;

      // IDA classes.
      void* idaMemory;
      SUNMatrix sparseMatrix;
      SUNLinearSolver linearSolver;

      std::vector<void*> parametricVariables;
      std::vector<void*> algebraicVariables;
      std::vector<void*> stateVariables;

      std::vector<void*> algebraicAndStateVariables;
      std::vector<void*> algebraicAndStateVariablesGetters;
      std::vector<void*> algebraicAndStateVariablesSetters;

      std::vector<void*> derivativeVariables;
      std::vector<void*> derivativeVariablesGetters;
      std::vector<void*> derivativeVariablesSetters;

      // Mapping from the IDA variable position to algebraic variables
      // position.
      std::map<size_t, size_t> algebraicVariablesMapping;

      // Mapping from the IDA variable position to state variables position.
      std::map<size_t, size_t> stateVariablesMapping;

      void** simulationData;
  };
}

//===---------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===---------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaCreate, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(idaInit, void, PTR(void))

RUNTIME_FUNC_DECL(idaCalcIC, void, PTR(void))

RUNTIME_FUNC_DECL(idaStep, void, PTR(void))

RUNTIME_FUNC_DECL(printStatistics, void, PTR(void))

RUNTIME_FUNC_DECL(idaFree, void, PTR(void))

RUNTIME_FUNC_DECL(idaSetStartTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetEndTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(idaSetTimeStep, void, PTR(void), double)

//===---------------------------------------------------------------------===//
// Equation setters
//===---------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaAddEquation, int64_t, PTR(void), PTR(int64_t), int64_t, int64_t, PTR(int64_t))

RUNTIME_FUNC_DECL(idaSetResidual, void, PTR(void), int64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddJacobian, void, PTR(void), int64_t, int64_t, PTR(void))

//===---------------------------------------------------------------------===//
// Variable setters
//===---------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaAddAlgebraicVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))
RUNTIME_FUNC_DECL(idaAddStateVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaAddParametricVariable, void, PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaSetDerivative, void, PTR(void), int64_t, PTR(void), PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaAddVariableAccess, void, PTR(void), int64_t, int64_t, PTR(int64_t), int64_t)

//===---------------------------------------------------------------------===//
// Getters
//===---------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaGetCurrentTime, float, PTR(void))
RUNTIME_FUNC_DECL(idaGetCurrentTime, double, PTR(void))

#endif // MARCO_RUNTIME_SOLVERS_IDA_INSTANCE_H
