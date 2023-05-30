#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H

#include "marco/Runtime/Support/Mangling.h"
#include "kinsol/kinsol.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <set>
#include <vector>

namespace marco::runtime::kinsol
{
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

  class KINSOLInstance
  {
    public:
      KINSOLInstance(int64_t marcoBitWidth, int64_t scalarEquationsNumber);

      ~KINSOLInstance();

      /// Add and initialize a new variable given its array.
      int64_t addAlgebraicVariable(void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter);

      int64_t addStateVariable(void* variable, int64_t* dimensions, int64_t rank, void* getter, void* setter);

      void setDerivative(int64_t stateVariable, void* derivative, void* getter, void* setter);

      /// Add the dimension of an equation to the KINSOL user data.
      int64_t addEquation(int64_t* ranges, int64_t rank);

      void addVariableAccess(int64_t equationIndex, int64_t variableIndex, int64_t* access, int64_t rank);

      /// Add the function pointer that computes the index-th residual function to the
      /// KINSOL user data.
      void addResidualFunction(int64_t equationIndex, void* residualFunction);

      /// Add the function pointer that computes the index-th jacobian row to the user
      /// data.
      void addJacobianFunction(int64_t equationIndex, int64_t variableIndex, void* jacobianFunction);

      /// Instantiate and initialize all the classes needed by KINSOL in order to solve
      /// the given system of equations. It also sets optional simulation parameters
      /// for KINSOL. It must be called before the first usage of kinsolStep() and after a
      /// call to kinsolAllocData(). It may fail in case of malformed model.
      bool initialize();

      /// Invoke KINSOL to perform one step of the computation. If a time step is given,
      /// the output will show the variables in an equidistant time grid based on the
      /// step time parameter. Otherwise, the output will show the variables at every
      /// step of the computation. Returns true if the computation was successful,
      /// false otherwise.
      bool step();

      /// Prints statistics regarding the computation of the system.
      void printStatistics() const;

      /// KINSOLResFn user-defined residual function, passed to KINSOL through KINSOLInit.
      /// It contains how to compute the Residual Function of the system, starting
      /// from the provided UserData struct, iterating through every equation.
      static int residualFunction(
          N_Vector variables, N_Vector residuals,
          void* userData);

      /// KINSOLLsJacFn user-defined Jacobian approximation function, passed to KINSOL
      /// through KINSOLSetJacFn. It contains how to compute the Jacobian Matrix of
      /// the system, starting from the provided UserData struct, iterating through
      /// every equation and variable. The matrix is represented in CSR format.
      static int jacobianMatrix(
          N_Vector variables, N_Vector residuals,
          SUNMatrix jacobianMatrix,
          void* userData,
          N_Vector tempv1, N_Vector tempv2);

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
      bool kinsolInit();
      bool kinsolFNTolerance();
      bool kinsolSSTolerance();
      bool kinsolSetLinearSolver();
      bool kinsolSetUserData();
      bool kinsolSetJacobianFunction();

    private:
      SUNContext ctx;
      bool initialized;

      int64_t marcoBitWidth;

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

      // Variables vectors and values
      N_Vector variablesVector;
      N_Vector derivativesVector;

      // The vector stores whether each scalar variable is an algebraic or a state one.
      // 0 = algebraic
      // 1 = state
      N_Vector idVector;

      // The tolerance for each scalar variable.
      N_Vector tolerancesVector;

      // The scale for each scalar variable
      N_Vector variableScaleVector;

      // The scale for each residual
      N_Vector residualScaleVector;

      // KINSOL classes
      void* kinsolMemory;
      SUNMatrix sparseMatrix;
      SUNLinearSolver linearSolver;

      std::vector<void*> parameters;
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

RUNTIME_FUNC_DECL(kinsolCreate, PTR(void), int64_t, int64_t)

RUNTIME_FUNC_DECL(kinsolInit, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolStep, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolPrintStatistics, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolFree, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolSetStartTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(kinsolSetEndTime, void, PTR(void), double)
RUNTIME_FUNC_DECL(kinsolSetTimeStep, void, PTR(void), double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(kinsolAddEquation, int64_t, PTR(void), PTR(int64_t), int64_t)

RUNTIME_FUNC_DECL(kinsolAddResidual, void, PTR(void), int64_t, PTR(void))

RUNTIME_FUNC_DECL(kinsolAddJacobian, void, PTR(void), int64_t, int64_t, PTR(void))

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(kinsolAddAlgebraicVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))
RUNTIME_FUNC_DECL(kinsolAddStateVariable, int64_t, PTR(void), PTR(void), PTR(int64_t), int64_t, PTR(void), PTR(void))

RUNTIME_FUNC_DECL(kinsolSetDerivative, void, PTR(void), int64_t, PTR(void), PTR(void), PTR(void))

RUNTIME_FUNC_DECL(kinsolAddVariableAccess, void, PTR(void), int64_t, int64_t, PTR(int64_t), int64_t)

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(kinsolGetCurrentTime, float, PTR(void))
RUNTIME_FUNC_DECL(kinsolGetCurrentTime, double, PTR(void))

#endif // MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H
