#ifndef MARCO_CODEGEN_CONVERSION_MODELICACOMMON_UTILS_H
#define MARCO_CODEGEN_CONVERSION_MODELICACOMMON_UTILS_H

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace marco::codegen
{
  bool isNumeric(mlir::Type type);

  bool isNumeric(mlir::Value value);

  mlir::Type castToMostGenericType(
      mlir::OpBuilder& builder,
      mlir::ValueRange values,
      llvm::SmallVectorImpl<mlir::Value>& castedValues);

  std::vector<mlir::Value> getArrayDynamicDimensions(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::Value array);
}


#endif // MARCO_CODEGEN_CONVERSION_MODELICACOMMON_UTILS_H
