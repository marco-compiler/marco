#ifndef MARCO_UNITTEST_CODEGEN_UTILS_H
#define MARCO_UNITTEST_CODEGEN_UTILS_H

#include "gmock/gmock.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include <algorithm>

// Match access by variable.
MATCHER_P(AccessMatcher, variable, "") {
  return arg.getVariable()->getValue() != variable;
}

// Match access by variable and access function.
MATCHER_P2(AccessMatcher, variable, accessFunction, "") {
  if (arg.getVariable()->getValue() != variable) {
    return false;
  }

  if (arg.getAccessFunction() != accessFunction) {
    return false;
  }

  return true;
}

// Match access by variable, access function and equation path.
MATCHER_P3(AccessMatcher, variable, accessFunction, path, "") {
  if (arg.getVariable()->getValue() != variable) {
    return false;
  }

  if (arg.getAccessFunction() != accessFunction) {
    return false;
  }

  if (arg.getPath() != path) {
    return false;
  }

  return true;
}

namespace marco::codegen::test
{
  /// Create a model with variables of given types.
  mlir::modelica::ModelOp createModel(mlir::OpBuilder& builder, mlir::TypeRange varTypes);

  /// Create an equation with with a certain body and optional iteration ranges.
  /// The callback function is used to create the body of the equation. When called,
  /// the insertion point of the nested builder is already set to the beginning of
  /// the equation body.
  ///
  /// @param builder          operation builder
  /// @param model            model into which the equation has to be created
  /// @param iterationRanges  optional iteration ranges
  /// @param bodyFn           callback function used to populate the equation body
  /// @return the created equation operation
  mlir::modelica::EquationOp createEquation(
      mlir::OpBuilder& builder,
      mlir::modelica::ModelOp model,
      llvm::ArrayRef<std::pair<long, long>> iterationRanges,
      std::function<void(mlir::OpBuilder&, mlir::ValueRange)> bodyFn);

  void createEquationSides(mlir::OpBuilder& builder, mlir::ValueRange lhs, mlir::ValueRange rhs);

  template<typename Cycle>
  class CyclesPermutation
  {
    public:
      CyclesPermutation(std::vector<Cycle>& cycles) : original(cycles), cycles(cycles)
      {
        for (size_t i = 0; i < cycles.size(); ++i) {
          indices.push_back(i);
        }
      }

      bool nextPermutation()
      {
        bool result = std::next_permutation(indices.begin(), indices.end());

        if (result) {
          for (size_t i = 0; i < indices.size(); ++i) {
            cycles[i] = original[indices[i]];
          }
        }

        return result;
      }

    private:
      std::vector<Cycle> original;
      std::vector<size_t> indices;
      std::vector<Cycle>& cycles;
  };
}

#endif // MARCO_UNITTEST_CODEGEN_UTILS_H
