#ifndef MARCO_UNITTEST_CODEGEN_UTILS_H
#define MARCO_UNITTEST_CODEGEN_UTILS_H

#include "gmock/gmock.h"
#include "marco/Codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/Model/Model.h"

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
  modelica::ModelOp createModel(mlir::OpBuilder& builder, mlir::TypeRange varTypes);

  /// Map the variables of a model.
  Variables mapVariables(modelica::ModelOp model);

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
  modelica::EquationOp createEquation(
      mlir::OpBuilder& builder,
      modelica::ModelOp model,
      llvm::ArrayRef<std::pair<long, long>> iterationRanges,
      std::function<void(mlir::OpBuilder&)> bodyFn);
}

#endif // MARCO_UNITTEST_CODEGEN_UTILS_H
