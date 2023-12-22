#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H

#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/SUNDIALS/Instance.h"
#include "kinsol/kinsol.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <set>
#include <vector>

namespace marco::runtime::sundials::kinsol
{
  /// Signature of residual functions.
  /// The 1st argument is a pointer to the list of equation indices.
  /// The result is the residual value.
  using ResidualFunction = double(*)(const int64_t*);

  /// Signature of Jacobian functions.
  /// The 1st argument is a pointer to the list of equation indices.
  /// The 2nd argument is a pointer to the list of variable indices.
  /// The result is the Jacobian value.
  using JacobianFunction = double(*)(const int64_t*, const uint64_t*);

  class KINSOLInstance
  {
    public:
      KINSOLInstance();

      ~KINSOLInstance();

      Variable addVariable(
          uint64_t rank,
          const uint64_t* dimensions,
          VariableGetter getterFunction,
          VariableSetter setterFunction,
          const char* name);

      /// Add the information about an equation that is handled by KINSOL.
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

      /// Instantiate and initialize all the classes needed by KINSOL in order to
      /// solve the given system of equations. It also sets optional simulation
      /// parameters for KINSOL.
      bool initialize();

      bool solve();

      /// KINSOLResFn user-defined residual function, passed to KINSOL through KINSOLInit.
      /// It contains how to compute the Residual Function of the system, starting
      /// from the provided UserData struct, iterating through every equation.
      static int residualFunction(
          N_Vector variables, N_Vector residuals, void* userData);

      static int residualFunction(
          realtype time,
          N_Vector variables,
          N_Vector residuals,
          void* userData);

      static int jacobianMatrix(
          N_Vector variables,
          N_Vector residuals,
          SUNMatrix jacobianMatrix,
          void* userData,
          N_Vector tempv1,
          N_Vector tempv2);

    private:
      [[nodiscard]] uint64_t getNumOfArrayVariables() const;

      [[nodiscard]] uint64_t getNumOfScalarVariables() const;

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

      void copyVariablesFromMARCO(N_Vector variables);

      void copyVariablesIntoMARCO(N_Vector variables);

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

      bool kinsolInit();
      bool kinsolFNTolerance();
      bool kinsolSSTolerance();
      bool kinsolSetLinearSolver();
      bool kinsolSetUserData();
      bool kinsolSetJacobianFunction();

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

      // Variables vectors and values.
      N_Vector variablesVector;

      // The tolerance for each scalar variable.
      N_Vector tolerancesVector;

      N_Vector variableScaleVector;
      N_Vector residualScaleVector;

      // KINSOL classes.
      void* kinsolMemory;

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

      std::vector<VariableGetter> variableGetters;
      std::vector<VariableSetter> variableSetters;

      // Thread pool.
      ThreadPool threadPool;

      // A chunk of equations to be processed by a thread.
      // A chunk is composed of:
      //   - the identifier (position) of the equation.
      //   - the begin indices (included)
      //   - the end indices (exluded)
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

RUNTIME_FUNC_DECL(kinsolCreate, PTR(void))

RUNTIME_FUNC_DECL(kinsolStep, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolFree, void, PTR(void))

RUNTIME_FUNC_DECL(idaAddVariable, uint64_t, PTR(void), uint64_t, PTR(uint64_t), PTR(void), PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaAddVariableAccess, void, PTR(void), uint64_t, uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddEquation, uint64_t, PTR(void), PTR(int64_t), uint64_t, uint64_t, PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaSetResidual, void, PTR(void), uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddJacobian, void, PTR(void), uint64_t, uint64_t, PTR(void))

#endif // SUNDIALS_ENABLE

#endif // MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H
