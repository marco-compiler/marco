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
}

#endif // MARCO_UNITTEST_CODEGEN_UTILS_H
