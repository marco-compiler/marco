#ifndef MARCO_CODEGEN_UTILS_H
#define MARCO_CODEGEN_UTILS_H

#include "mlir/IR/Builders.h"

namespace marco::codegen
{
  mlir::Type getMostGenericType(mlir::Value x, mlir::Value y);
  mlir::Type getMostGenericType(mlir::Type x, mlir::Type y);

  void copyArray(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value source, mlir::Value destination);
}

#endif // MARCO_CODEGEN_UTILS_H
