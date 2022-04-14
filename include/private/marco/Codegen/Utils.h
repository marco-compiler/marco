#ifndef MARCO_CODEGEN_UTILS_H
#define MARCO_CODEGEN_UTILS_H

#include "mlir/IR/Builders.h"

namespace marco::codegen
{
  void copyArray(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value source, mlir::Value destination);
}

#endif // MARCO_CODEGEN_UTILS_H
