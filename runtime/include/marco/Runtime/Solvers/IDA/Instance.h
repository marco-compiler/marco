#ifndef MARCO_RUNTIME_SOLVERS_IDA_INSTANCE_H
#define MARCO_RUNTIME_SOLVERS_IDA_INSTANCE_H

#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/SUNDIALS/Instance.h"
#include "ida/ida.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <map>
#include <set>
#include <vector>

namespace marco::runtime::sundials::ida
{
  enum class VariableKind
  {
    ALGEBRAIC,
    STATE
  };

  /// Signature of residual functions.
  /// The 1st argument is the current time.
  /// The 2nd argument is a pointer to the list of equation indices.
  /// The result is the residual value.
  using ResidualFunction = double(*)(double, const int64_t*);

  /// Signature of Jacobian functions.
  /// The 1st argument is the current time.
  /// The 2nd argument is a pointer to the list of equation indices.
  /// The 3rd argument is a pointer to the list of variable indices.
  /// The 4th argument is the 'alpha' value.
  /// The result is the Jacobian value.
  using JacobianFunction = double(*)(
      double, const int64_t*, const uint64_t*, double);

  class IDAInstance
  {
    public:
      IDAInstance();

      ~IDAInstance();

      void setStartTime(double time);
      void setEndTime(double time);
      void setTimeStep(double time);

      Variable addAlgebraicVariable(
          uint64_t rank,
          const uint64_t* dimensions,
          VariableGetter getterFunction,
          VariableSetter setterFunction,
          const char* name);

      Variable addStateVariable(
          uint64_t rank,
          const uint64_t* dimensions,
          VariableGetter stateGetterFunction,
          VariableSetter stateSetterFunction,
          VariableGetter derivativeGetterFunction,
          VariableSetter derivativeSetterFunction,
          const char* name);

      /// Add the information about an equation that is handled by IDA.
      Equation addEquation(
          const int64_t* ranges,
          uint64_t rank,
          Variable writtenVariable,
          AccessFunction writeAccessFunction,
          const char* stringRepresentation);

      void addVariableAccess(
          Equation equation,
          Variable variableIndex,
          AccessFunction accessFunction);

      /// Add the function pointer that computes the residual value of an
      /// equation.
      void setResidualFunction(
          Equation equationIndex,
          ResidualFunction residualFunction);

      /// Add the function pointer that computes a partial derivative of an
      /// equation.
      void addJacobianFunction(
          Equation equationIndex,
          Variable variableIndex,
          JacobianFunction jacobianFunction);

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
      [[nodiscard]] uint64_t getNumOfArrayVariables() const;

      [[nodiscard]] uint64_t getNumOfScalarVariables() const;

      [[nodiscard]] VariableKind getVariableKind(Variable variable) const;

      [[nodiscard]] uint64_t getVariableFlatSize(Variable variable) const;

      [[nodiscard]] uint64_t getNumOfVectorizedEquations() const;

      [[nodiscard]] uint64_t getNumOfScalarEquations() const;

      [[nodiscard]] uint64_t getEquationRank(Equation equation) const;

      [[nodiscard]] uint64_t getEquationFlatSize(Equation equation) const;

      [[nodiscard]] Variable getWrittenVariable(Equation equation) const;

      [[nodiscard]] AccessFunction getWriteAccessFunction(Equation equation) const;

      [[nodiscard]] uint64_t getVariableRank(Variable variable) const;

      std::vector<JacobianColumn> computeJacobianColumns(
          Equation eq, const int64_t* equationIndices) const;

      void computeNNZ();

      void computeThreadChunks();

      void copyVariablesFromMARCO(
          N_Vector algebraicAndStateVariablesVector,
          N_Vector derivativeVariablesVector);

      void copyVariablesIntoMARCO(
          N_Vector algebraicAndStateVariablesVector,
          N_Vector derivativeVariablesVector);

      void vectorEquationsParallelIteration(
          std::function<void(Equation equation)> processFn);

      void scalarEquationsParallelIteration(
          std::function<void(
              Equation equation,
              const std::vector<int64_t>& equationIndices)> processFn);

      void equationsParallelIteration(
          std::function<void(
              Equation equation,
              const std::vector<int64_t>& equationIndices)> processFn);

      void getVariableBeginIndices(
          Variable variable, std::vector<uint64_t>& indices) const;

      void getVariableEndIndices(
          Variable variable, std::vector<uint64_t>& indices) const;

      void getEquationBeginIndices(
          Equation equation, std::vector<int64_t>& indices) const;

      void getEquationEndIndices(
          Equation equation, std::vector<int64_t>& indices) const;

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
      /// @name Utility functions
      /// {

      /// Get the scalar equation writing to a certain scalar variable.
      /// Warning: extremely slow, to be used only for debug purposes.
      void getWritingEquation(
          Variable variable,
          const std::vector<uint64_t>& variableIndices,
          Equation& equation,
          std::vector<int64_t>& equationIndices) const;

      /// }
      /// @name Debug functions
      /// {
      void printVariablesVector(N_Vector variables) const;

      void printDerivativesVector(N_Vector derivatives) const;

      void printResidualsVector(N_Vector residuals) const;

      void printJacobianMatrix(SUNMatrix jacobianMatrix) const;

      /// }

    private:
      // SUNDIALS context.
      SUNContext ctx{nullptr};

      // Whether the instance has been inizialized or not.
      bool initialized{false};

      // Model size.
      uint64_t scalarVariablesNumber{0};
      uint64_t scalarEquationsNumber{0};
      uint64_t nonZeroValuesNumber{0};

      // The iteration ranges of the vectorized equations.
      std::vector<MultidimensionalRange> equationRanges;

      // The array variables written by the equations.
      // The i-th position contains the information about the variable written
      // by the i-th equation: the first element is the index of the IDA
      // variable, while the second represents the ranges of the scalar
      // variable.
      std::vector<std::pair<Variable, AccessFunction>> writeAccesses;

      // The order in which the equations must be processed when computing
      // residuals and partial derivatives.
      std::vector<Equation> equationsProcessingOrder;

      // The residual functions associated with the equations.
      // The i-th position contains the pointer to the residual function of the
      // i-th equation.
      std::vector<ResidualFunction> residualFunctions;

      // The jacobian functions associated with the equations.
      // The i-th position contains the list of partial derivative functions of
      // the i-th equation. The j-th function represents the function to
      // compute the derivative with respect to the j-th variable.
      std::vector<std::vector<JacobianFunction>> jacobianFunctions;

      // Whether the IDA instance is informed about the accesses to the
      // variables.
      bool precomputedAccesses{false};

      std::vector<VarAccessList> variableAccesses;

      // The offset of each array variable inside the flattened variables
      // vector.
      std::vector<uint64_t> variableOffsets;

      // The dimensions list of each array variable.
      std::vector<VariableDimensions> variablesDimensions;

      // Simulation times.
      realtype startTime;
      realtype endTime;
      realtype timeStep;
      realtype currentTime = 0;

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

      // Support structure for the computation of the jacobian matrix.
      // The outer vector has a number of elements equal to the scalar number
      // of equations. Each of them represents a row of the matrix and consists
      // in a vector of paired elements. The first element of each pair
      // represents the index of the column (that is, the independent scalar
      // variable for the partial derivative) while the second one is the
      // value of the partial derivative.
      std::vector<std::vector<std::pair<sunindextype, double>>> jacobianMatrixData;

      SUNLinearSolver linearSolver;

      std::vector<VariableGetter> algebraicAndStateVariablesGetters;
      std::vector<VariableSetter> algebraicAndStateVariablesSetters;

      std::vector<VariableGetter> derivativeVariablesGetters;
      std::vector<VariableSetter> derivativeVariablesSetters;

      // Mapping from the IDA variable position to state variables position.
      std::map<Variable, size_t> stateVariablesMapping;

      // Thread pool.
      ThreadPool threadPool;

      // A chunk of equations to be processed by a thread.
      // A chunk is composed of:
      //   - the identifier (position) of the equation.
      //   - the begin indices (included)
      //   - the end indices (excluded)
      using ThreadEquationsChunk = std::tuple<
          Equation, std::vector<int64_t>, std::vector<int64_t>>;

      // The list of chunks the threads will process. Each thread elaborates
      // one chunk at a time.
      // The information is computed only once during the initialization to
      // save time during the actual simulation.
      std::vector<ThreadEquationsChunk> threadEquationsChunks;
  };
}

//===---------------------------------------------------------------------===//
// Exported functions
//===---------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaCreate, PTR(void))

RUNTIME_FUNC_DECL(idaCalcIC, void, PTR(void))

RUNTIME_FUNC_DECL(idaStep, void, PTR(void))

RUNTIME_FUNC_DECL(idaFree, void, PTR(void))

RUNTIME_FUNC_DECL(idaSetStartTime, void, PTR(void), double)

RUNTIME_FUNC_DECL(idaSetEndTime, void, PTR(void), double)

RUNTIME_FUNC_DECL(idaSetTimeStep, void, PTR(void), double)

RUNTIME_FUNC_DECL(idaGetCurrentTime, double, PTR(void))

RUNTIME_FUNC_DECL(idaAddAlgebraicVariable, uint64_t, PTR(void), uint64_t, PTR(uint64_t), PTR(void), PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaAddStateVariable, uint64_t, PTR(void), uint64_t, PTR(uint64_t), PTR(void), PTR(void), PTR(void), PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaAddVariableAccess, void, PTR(void), uint64_t, uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddEquation, uint64_t, PTR(void), PTR(int64_t), uint64_t, uint64_t, PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaSetResidual, void, PTR(void), uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddJacobian, void, PTR(void), uint64_t, uint64_t, PTR(void))

RUNTIME_FUNC_DECL(printStatistics, void, PTR(void))

#endif // SUNDIALS_ENABLE

#endif // MARCO_RUNTIME_SOLVERS_IDA_INSTANCE_H
