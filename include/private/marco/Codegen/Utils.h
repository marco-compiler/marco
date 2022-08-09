#ifndef MARCO_CODEGEN_UTILS_H
#define MARCO_CODEGEN_UTILS_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <functional>

namespace marco::codegen
{
  mlir::Type getMostGenericType(mlir::Value x, mlir::Value y);
  mlir::Type getMostGenericType(mlir::Type x, mlir::Type y);

  std::string getUniqueSymbolName(mlir::ModuleOp module, std::function<std::string()> tryFn);

  void copyArray(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value source, mlir::Value destination);

  //modeling::IndexSet getIndexSet(mlir::modelica::IterationSpace iterationSpace);
}

#endif // MARCO_CODEGEN_UTILS_H
