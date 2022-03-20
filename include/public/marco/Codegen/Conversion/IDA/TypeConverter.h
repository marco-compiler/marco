#ifndef MARCO_CODEGEN_CONVERSION_IDA_TYPECONVERTER_H
#define MARCO_CODEGEN_CONVERSION_IDA_TYPECONVERTER_H

#include "marco/Dialect/IDA/IDADialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ida
{
  // We inherit from the LLVMTypeConverter in order to retrieve the converted MLIR index type.
  class TypeConverter : public mlir::LLVMTypeConverter
  {
    public:
      TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options);

      mlir::Type convertInstanceType(InstanceType type);
      mlir::Type convertEquationType(EquationType type);
      mlir::Type convertVariableType(VariableType type);

      llvm::Optional<mlir::Value> opaquePointerTypeTargetMaterialization(
          mlir::OpBuilder& builder, mlir::LLVM::LLVMPointerType resultType, mlir::ValueRange inputs, mlir::Location loc) const;

      llvm::Optional<mlir::Value> instanceTypeSourceMaterialization(
          mlir::OpBuilder& builder, InstanceType resultType, mlir::ValueRange inputs, mlir::Location loc) const;
  };
}

#endif // MARCO_CODEGEN_CONVERSION_IDA_TYPECONVERTER_H
